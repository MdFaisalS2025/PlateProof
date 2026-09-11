"""Auditable, deterministic entity resolution between official government
restaurant identities (NYC DOHMH / Florida DBPR) and Michelin restaurant
identities.

Design notes
------------
* Official identifiers (``restaurant_id``, i.e. ``nyc:<camis>`` /
  ``florida:<license_number>``) are authoritative and are never modified,
  replaced, or derived from a Michelin match anywhere in this module.
* Matching is performed once against a stable :class:`MichelinRestaurant`
  identity, never once per :class:`~plateproof.ingestion.michelin.MichelinDistinctionEvent`.
  Multiple guide years, multiple simultaneous distinctions (e.g. a star and a
  Green Star in one edition), or a restaurant absent from a later edition are
  not matching conflicts -- they are simply facts about one already-matched
  restaurant identity, attached afterward (see ``export_review_queue``).
* Scoring weights, the missing-evidence cap, and the address-conflict floor
  below are versioned, provisional heuristics documented and tested against a
  small labeled validation matrix (see ``tests/matching/test_validation_matrix.py``)
  -- not proven statistical probabilities. ``overall_score`` is a
  matching-confidence heuristic, never a health or safety score.
* No embeddings, LLM, or learned model is used. Name similarity uses RapidFuzz
  (MIT-licensed, local, free -- see ``pyproject.toml``); every component score
  stays visible on :class:`MatchEvidence`.
* Conflict policy is strict: any official or Michelin identity involved in
  more than one accept/review-eligible pair has ALL of its pairs forced to
  ``review``. There is no score-margin exception for keeping a "winner".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

import polars as pl
from pydantic import BaseModel, ConfigDict
from rapidfuzz import fuzz

from plateproof.matching.normalize import distinctive_name_tokens, normalize_text, street_number

Jurisdiction = Literal["nyc", "florida"]

MATCHER_VERSION = "michelin-entity-match-v1"

#: Provisional, versioned scoring weights -- see the module docstring.
_WEIGHTS: dict[str, float] = {
    "name": 0.40,
    "address": 0.25,
    "postal": 0.15,
    "city": 0.10,
    "distance": 0.10,
}
ADDRESS_CONFLICT_FLOOR = 0.5
DISTANCE_SCORE_THRESHOLD_KM = 1.5
MISSING_EVIDENCE_CAP = 0.90
ACCEPT_THRESHOLD = 0.92
REVIEW_THRESHOLD = 0.75
_GEO_BLOCK_RADIUS_KM = 2.0

_REQUIRED_COMPONENTS = ("address_similarity", "postal_code_match", "city_region_match")
_OPTIONAL_COMPONENTS = ("distance_score",)


@dataclass(frozen=True)
class OfficialRestaurant:
    """The official-side identity used for matching. Never mutated by matching."""

    restaurant_id: str
    jurisdiction: Jurisdiction
    source_id: str
    name: str | None
    normalized_name: str | None
    address_as_published: str | None
    normalized_address: str | None
    city: str | None
    postal_code: str | None
    latitude: float | None
    longitude: float | None


class MatchEvidence(BaseModel):
    """Independently inspectable evidence for one (official, Michelin) pair."""

    model_config = ConfigDict(frozen=True)

    official_restaurant_id: str
    michelin_restaurant_id: str
    jurisdiction: Jurisdiction
    name_similarity: float
    name_similarity_components: dict[str, float]
    address_similarity: float | None
    postal_code_match: bool | None
    city_region_match: bool | None
    distance_km: float | None
    distance_score: float | None
    blocking_rules: list[str]
    required_evidence_missing: list[str]
    optional_evidence_missing: list[str]
    conflicting_evidence: list[str]
    overall_score: float
    decision: Literal["accept", "review", "reject"]
    reasons: list[str]
    conflicting_candidates: list[str] = []
    matcher_version: str
    generated_at: datetime


# --------------------------------------------------------------------------- #
# Official-restaurant projections                                             #
# --------------------------------------------------------------------------- #


def official_restaurants_from_nyc_events(
    inspection_events: pl.DataFrame,
) -> list[OfficialRestaurant]:
    """Project one row per distinct NYC ``restaurant_id`` from Task 2's
    normalized inspection-event table (which already carries address fields
    for NYC)."""
    from plateproof.matching.normalize import normalize_address, normalize_name

    if inspection_events.height == 0:
        return []
    columns = ["restaurant_id", "source_id", "dba", "building", "street", "boro_raw", "zipcode"]
    available = [c for c in columns if c in inspection_events.columns]
    distinct = inspection_events.select(available).unique(subset=["restaurant_id"], keep="first")
    results: list[OfficialRestaurant] = []
    for row in distinct.to_dicts():
        name = row.get("dba")
        address_parts = [row.get("building"), row.get("street")]
        address_as_published = " ".join(p for p in address_parts if p) or None
        normalized_addr = normalize_address(address_as_published)
        results.append(
            OfficialRestaurant(
                restaurant_id=row["restaurant_id"],
                jurisdiction="nyc",
                source_id=row["source_id"],
                name=name,
                normalized_name=normalize_name(name) if name else None,
                address_as_published=address_as_published,
                normalized_address=normalized_addr.normalized if normalized_addr else None,
                city=row.get("boro_raw"),
                postal_code=row.get("zipcode"),
                latitude=None,
                longitude=None,
            )
        )
    return results


_FLORIDA_STAGING_REQUIRED = (
    "license_number",
    "dba",
    "location_address",
    "location_city",
    "location_zip",
)


def official_restaurants_from_florida(
    staging: pl.DataFrame, inspection_events: pl.DataFrame
) -> tuple[list[OfficialRestaurant], list[str]]:
    """Project one row per distinct Florida ``restaurant_id``.

    Florida's normalized ``inspection_events`` table (Task 3) deliberately
    leaves address fields null -- they exist only in the staging frame
    (:func:`plateproof.ingestion.florida.load_florida_extracts` output). This
    function therefore *requires* the staging frame and joins it to the event
    table's distinct ``restaurant_id`` values by license number.

    Raises ``ValueError`` naming any missing required staging column. Returns
    ``(officials, missing_restaurant_ids)`` where ``missing_restaurant_ids``
    lists any ``restaurant_id`` present in ``inspection_events`` but not
    resolvable in ``staging`` -- an inconsistency between the two frames,
    reported rather than silently dropped without a trace.
    """
    from plateproof.matching.normalize import normalize_address, normalize_name

    missing_cols = [c for c in _FLORIDA_STAGING_REQUIRED if c not in staging.columns]
    if missing_cols:
        raise ValueError(f"Florida staging frame is missing required column(s): {missing_cols}")

    if inspection_events.height == 0:
        return [], []

    staging_by_license = (
        staging.select(list(_FLORIDA_STAGING_REQUIRED))
        .unique(subset=["license_number"], keep="first")
        .to_dicts()
    )
    staging_index = {f"florida:{row['license_number']}": row for row in staging_by_license}

    restaurant_ids = inspection_events.get_column("restaurant_id").unique().to_list()
    officials: list[OfficialRestaurant] = []
    missing: list[str] = []
    for restaurant_id in sorted(restaurant_ids):
        staging_row = staging_index.get(restaurant_id)
        if staging_row is None:
            missing.append(restaurant_id)
            continue
        name = staging_row.get("dba")
        address_as_published = staging_row.get("location_address")
        normalized_addr = normalize_address(address_as_published)
        officials.append(
            OfficialRestaurant(
                restaurant_id=restaurant_id,
                jurisdiction="florida",
                source_id=staging_row["license_number"],
                name=name,
                normalized_name=normalize_name(name) if name else None,
                address_as_published=address_as_published,
                normalized_address=normalized_addr.normalized if normalized_addr else None,
                city=staging_row.get("location_city"),
                postal_code=staging_row.get("location_zip"),
                latitude=None,
                longitude=None,
            )
        )
    return officials, missing


# --------------------------------------------------------------------------- #
# Candidate generation (blocking)                                             #
# --------------------------------------------------------------------------- #


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * radius_km * math.asin(math.sqrt(a))


def generate_candidate_pairs(
    officials: list[OfficialRestaurant], restaurants: list[Any]
) -> list[tuple[OfficialRestaurant, Any, list[str]]]:
    """Generate (official, michelin, blocking_rules) candidates.

    Hard jurisdiction filter first (a candidate is never even considered
    across jurisdictions). Within one jurisdiction, a pair becomes a candidate
    if ANY of these fire -- deliberately not dependent on name-token order or
    a "first word" match, so "The", chef names, hotel names, and punctuation
    differences never hide a legitimate candidate:

    * equal postal code,
    * equal street number and matching normalized city,
    * a shared *distinctive* name token (stopwords excluded) within the same
      normalized city,
    * geographic proximity, when both sides have coordinates.
    """
    by_jurisdiction: dict[str, list[Any]] = {}
    for restaurant in restaurants:
        by_jurisdiction.setdefault(restaurant.jurisdiction_candidate, []).append(restaurant)

    pairs: list[tuple[OfficialRestaurant, Any, list[str]]] = []
    for official in officials:
        for restaurant in by_jurisdiction.get(official.jurisdiction, []):
            rules: list[str] = []
            if (
                official.postal_code
                and restaurant.postal_code
                and official.postal_code == restaurant.postal_code
            ):
                rules.append("postal_code_exact")

            official_city = normalize_text(official.city) if official.city else None
            restaurant_city = normalize_text(restaurant.city) if restaurant.city else None
            same_city = official_city is not None and official_city == restaurant_city

            official_num = street_number(official.normalized_address)
            restaurant_num = street_number(restaurant.normalized_address)
            if same_city and official_num and restaurant_num and official_num == restaurant_num:
                rules.append("street_number_and_city")

            if same_city and official.normalized_name and restaurant.normalized_name:
                shared = distinctive_name_tokens(
                    official.normalized_name
                ) & distinctive_name_tokens(restaurant.normalized_name)
                if shared:
                    rules.append("shared_name_token_and_city")

            if (
                official.latitude is not None
                and official.longitude is not None
                and restaurant.latitude is not None
                and restaurant.longitude is not None
            ):
                distance = _haversine_km(
                    official.latitude, official.longitude, restaurant.latitude, restaurant.longitude
                )
                if distance <= _GEO_BLOCK_RADIUS_KM:
                    rules.append("geo_proximity")

            if rules:
                pairs.append((official, restaurant, rules))
    return pairs


# --------------------------------------------------------------------------- #
# Scoring                                                                      #
# --------------------------------------------------------------------------- #


def compute_name_similarity(a: str, b: str) -> tuple[float, dict[str, float]]:
    """Token-aware and character-aware similarity via RapidFuzz. Every
    component stays visible; the overall value is their simple average --
    a documented, provisional combination, not a fitted model."""
    wratio = fuzz.WRatio(a, b) / 100.0
    token_sort = fuzz.token_sort_ratio(a, b) / 100.0
    token_set = fuzz.token_set_ratio(a, b) / 100.0
    components = {"wratio": wratio, "token_sort_ratio": token_sort, "token_set_ratio": token_set}
    overall = (wratio + token_sort + token_set) / 3.0
    return overall, components


def _decide(score: float) -> Literal["accept", "review", "reject"]:
    if score >= ACCEPT_THRESHOLD:
        return "accept"
    if score >= REVIEW_THRESHOLD:
        return "review"
    return "reject"


def match_candidate(
    official: OfficialRestaurant,
    restaurant: Any,
    blocking_rules: list[str],
    *,
    generated_at: datetime | None = None,
) -> MatchEvidence:
    """Score one (official, Michelin) candidate pair.

    Acceptance requires strong corroborating identity evidence, never name
    alone: ``address_similarity``, ``postal_code_match``, and
    ``city_region_match`` are the *required* components -- any of them being
    unavailable caps the score below ``ACCEPT_THRESHOLD`` (see
    ``MISSING_EVIDENCE_CAP``). ``distance_score`` is *optional*: its absence
    alone never triggers the cap, so an otherwise-complete Florida match (no
    coordinates anywhere in Task 3's data) can still reach ``accept``.
    Conflicting (present-but-different) postal code or address evidence also
    forces the cap, independent of the weighted average.
    """
    name_similarity, name_components = compute_name_similarity(
        official.normalized_name or "", restaurant.normalized_name
    )

    address_similarity: float | None = None
    if official.normalized_address and restaurant.normalized_address:
        address_similarity = (
            fuzz.ratio(official.normalized_address, restaurant.normalized_address) / 100.0
        )

    postal_code_match: bool | None = None
    if official.postal_code and restaurant.postal_code:
        postal_code_match = official.postal_code == restaurant.postal_code

    city_region_match: bool | None = None
    if official.city and restaurant.city:
        city_region_match = normalize_text(official.city) == normalize_text(restaurant.city)

    distance_km: float | None = None
    distance_score: float | None = None
    if (
        official.latitude is not None
        and official.longitude is not None
        and restaurant.latitude is not None
        and restaurant.longitude is not None
    ):
        distance_km = _haversine_km(
            official.latitude, official.longitude, restaurant.latitude, restaurant.longitude
        )
        distance_score = max(0.0, min(1.0, 1.0 - distance_km / DISTANCE_SCORE_THRESHOLD_KM))

    values: dict[str, float | None] = {
        "name": name_similarity,
        "address": address_similarity,
        "postal": (1.0 if postal_code_match else 0.0) if postal_code_match is not None else None,
        "city": (1.0 if city_region_match else 0.0) if city_region_match is not None else None,
        "distance": distance_score,
    }
    available = {key: value for key, value in values.items() if value is not None}
    weight_sum = sum(_WEIGHTS[key] for key in available)
    raw_score = (
        sum(_WEIGHTS[key] * value for key, value in available.items()) / weight_sum
        if weight_sum
        else 0.0
    )

    required_missing = []
    if address_similarity is None:
        required_missing.append("address_similarity")
    if postal_code_match is None:
        required_missing.append("postal_code_match")
    if city_region_match is None:
        required_missing.append("city_region_match")
    optional_missing = list(_OPTIONAL_COMPONENTS) if distance_score is None else []

    conflicting: list[str] = []
    if postal_code_match is False:
        conflicting.append("postal_code_conflict")
    if address_similarity is not None and address_similarity < ADDRESS_CONFLICT_FLOOR:
        conflicting.append("address_conflict")

    overall_score = raw_score
    if required_missing or conflicting:
        overall_score = min(overall_score, MISSING_EVIDENCE_CAP)

    decision = _decide(overall_score)

    reasons: list[str] = []
    if required_missing:
        reasons.append(f"required evidence missing: {', '.join(required_missing)}")
    if conflicting:
        reasons.append(f"conflicting evidence: {', '.join(conflicting)}")
    if not required_missing and not conflicting:
        reasons.append("all required identity evidence present and non-conflicting")

    return MatchEvidence(
        official_restaurant_id=official.restaurant_id,
        michelin_restaurant_id=restaurant.michelin_restaurant_id,
        jurisdiction=official.jurisdiction,
        name_similarity=name_similarity,
        name_similarity_components=name_components,
        address_similarity=address_similarity,
        postal_code_match=postal_code_match,
        city_region_match=city_region_match,
        distance_km=distance_km,
        distance_score=distance_score,
        blocking_rules=blocking_rules,
        required_evidence_missing=required_missing,
        optional_evidence_missing=optional_missing,
        conflicting_evidence=conflicting,
        overall_score=overall_score,
        decision=decision,
        reasons=reasons,
        conflicting_candidates=[],
        matcher_version=MATCHER_VERSION,
        generated_at=generated_at if generated_at is not None else datetime.now(UTC),
    )


# --------------------------------------------------------------------------- #
# Conflict policy (strict) and one-to-one enforcement                         #
# --------------------------------------------------------------------------- #


def resolve_matches(evidence_list: list[MatchEvidence]) -> list[MatchEvidence]:
    """Apply the strict conflict policy.

    Any ``official_restaurant_id`` or ``michelin_restaurant_id`` appearing in
    more than one accept/review-eligible pair has EVERY one of its pairs
    forced to ``review`` (an existing ``accept`` is downgraded), with
    ``conflicting_candidates`` populated. There is no score-margin exception:
    conflict resolution is left to an explicit future human-review decision.
    """
    eligible = [e for e in evidence_list if e.decision in ("accept", "review")]
    by_official: dict[str, list[MatchEvidence]] = {}
    by_michelin: dict[str, list[MatchEvidence]] = {}
    for evidence in eligible:
        by_official.setdefault(evidence.official_restaurant_id, []).append(evidence)
        by_michelin.setdefault(evidence.michelin_restaurant_id, []).append(evidence)

    conflicted_officials = {oid for oid, group in by_official.items() if len(group) > 1}
    conflicted_michelin = {mid for mid, group in by_michelin.items() if len(group) > 1}

    resolved: list[MatchEvidence] = []
    for evidence in evidence_list:
        in_conflict = (
            evidence.official_restaurant_id in conflicted_officials
            or evidence.michelin_restaurant_id in conflicted_michelin
        )
        if not in_conflict:
            resolved.append(evidence)
            continue
        others = sorted(
            {
                e.michelin_restaurant_id
                for e in by_official.get(evidence.official_restaurant_id, [])
                if e.michelin_restaurant_id != evidence.michelin_restaurant_id
            }
            | {
                e.official_restaurant_id
                for e in by_michelin.get(evidence.michelin_restaurant_id, [])
                if e.official_restaurant_id != evidence.official_restaurant_id
            }
        )
        new_decision = "review" if evidence.decision == "accept" else evidence.decision
        new_reasons = [
            *evidence.reasons,
            "candidate conflict: forced to review, no margin exception applied",
        ]
        resolved.append(
            evidence.model_copy(
                update={
                    "decision": new_decision,
                    "conflicting_candidates": others,
                    "reasons": new_reasons,
                }
            )
        )
    return resolved


def assert_one_to_one(evidence_list: list[MatchEvidence]) -> None:
    """Validate that the FINAL ACCEPTED subset forms a one-to-one mapping
    between official restaurants and Michelin restaurants. Intended as a
    structural safety net after :func:`resolve_matches`, not the primary
    conflict-detection mechanism."""
    accepted = [e for e in evidence_list if e.decision == "accept"]
    officials = [e.official_restaurant_id for e in accepted]
    michelins = [e.michelin_restaurant_id for e in accepted]
    if len(officials) != len(set(officials)) or len(michelins) != len(set(michelins)):
        raise ValueError("accepted matches violate the one-to-one official-to-Michelin invariant")


# --------------------------------------------------------------------------- #
# Review queue export                                                         #
# --------------------------------------------------------------------------- #

REVIEW_QUEUE_SCHEMA: dict[str, pl.DataType] = {
    "official_restaurant_id": pl.String(),
    "official_name": pl.String(),
    "official_normalized_name": pl.String(),
    "official_address_as_published": pl.String(),
    "official_normalized_address": pl.String(),
    "official_city": pl.String(),
    "official_postal_code": pl.String(),
    "michelin_restaurant_id": pl.String(),
    "michelin_name_as_published": pl.String(),
    "michelin_normalized_name": pl.String(),
    "michelin_address_as_published": pl.String(),
    "michelin_normalized_address": pl.String(),
    "michelin_city": pl.String(),
    "michelin_postal_code": pl.String(),
    "name_similarity": pl.Float64(),
    "address_similarity": pl.Float64(),
    "postal_code_match": pl.Boolean(),
    "city_region_match": pl.Boolean(),
    "distance_km": pl.Float64(),
    "distance_score": pl.Float64(),
    "required_evidence_missing": pl.List(pl.String()),
    "optional_evidence_missing": pl.List(pl.String()),
    "conflicting_evidence": pl.List(pl.String()),
    "overall_score": pl.Float64(),
    "decision": pl.String(),
    "reasons": pl.List(pl.String()),
    "blocking_rules": pl.List(pl.String()),
    "conflicting_candidates": pl.List(pl.String()),
    "distinction_events": pl.List(pl.String()),
    "identity_source_note": pl.String(),
    "matcher_version": pl.String(),
    "generated_at": pl.Datetime("us", "UTC"),
    "reviewer_decision": pl.String(),
    "reviewer_notes": pl.String(),
}


def export_review_queue(
    evidence_list: list[MatchEvidence],
    officials_by_id: dict[str, OfficialRestaurant],
    restaurants_by_id: dict[str, Any],
    distinction_events: list[Any],
    *,
    include_rejected: bool = False,
) -> pl.DataFrame:
    """One row per (official, Michelin) ENTITY pair -- never one row per
    distinction event. Each row's ``distinction_events`` column summarizes
    every guide-year/distinction fact attached to that Michelin restaurant, so
    a reviewer sees the full edition history alongside the candidate match.
    ``reviewer_decision``/``reviewer_notes`` are always blank for a human to
    fill in later.
    """
    events_by_restaurant: dict[str, list[Any]] = {}
    for event in distinction_events:
        events_by_restaurant.setdefault(event.michelin_restaurant_id, []).append(event)

    rows: list[dict[str, Any]] = []
    for evidence in evidence_list:
        if not include_rejected and evidence.decision == "reject":
            continue
        official = officials_by_id[evidence.official_restaurant_id]
        restaurant = restaurants_by_id[evidence.michelin_restaurant_id]
        events = sorted(
            events_by_restaurant.get(evidence.michelin_restaurant_id, []),
            key=lambda e: (e.guide_year, e.distinction),
        )
        rows.append(
            {
                "official_restaurant_id": official.restaurant_id,
                "official_name": official.name,
                "official_normalized_name": official.normalized_name,
                "official_address_as_published": official.address_as_published,
                "official_normalized_address": official.normalized_address,
                "official_city": official.city,
                "official_postal_code": official.postal_code,
                "michelin_restaurant_id": restaurant.michelin_restaurant_id,
                "michelin_name_as_published": restaurant.name_as_published,
                "michelin_normalized_name": restaurant.normalized_name,
                "michelin_address_as_published": restaurant.address_as_published,
                "michelin_normalized_address": restaurant.normalized_address,
                "michelin_city": restaurant.city,
                "michelin_postal_code": restaurant.postal_code,
                "name_similarity": evidence.name_similarity,
                "address_similarity": evidence.address_similarity,
                "postal_code_match": evidence.postal_code_match,
                "city_region_match": evidence.city_region_match,
                "distance_km": evidence.distance_km,
                "distance_score": evidence.distance_score,
                "required_evidence_missing": evidence.required_evidence_missing,
                "optional_evidence_missing": evidence.optional_evidence_missing,
                "conflicting_evidence": evidence.conflicting_evidence,
                "overall_score": evidence.overall_score,
                "decision": evidence.decision,
                "reasons": evidence.reasons,
                "blocking_rules": evidence.blocking_rules,
                "conflicting_candidates": evidence.conflicting_candidates,
                "distinction_events": [f"{e.guide_year}:{e.distinction}" for e in events],
                "identity_source_note": restaurant.identity_source_note,
                "matcher_version": evidence.matcher_version,
                "generated_at": evidence.generated_at,
                "reviewer_decision": None,
                "reviewer_notes": None,
            }
        )
    return pl.DataFrame(rows, schema=REVIEW_QUEUE_SCHEMA)
