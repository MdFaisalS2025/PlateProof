"""Michelin ingestion: an optional, manually maintained, provenance-rich seed CSV.

Source policy (read before touching this module)
--------------------------------------------------
PlateProof does not scrape Michelin Guide pages, hidden endpoints, search
systems, or APIs; does not use Michelin's paid API tiers; and does not use
third-party "Michelin restaurant" datasets whose own documented origin is
scraping the Guide site. The only supported source is a small, optional,
hand-curated CSV that a maintainer fills in by transcribing *facts* (name,
address, distinction, edition year, a citation URL) from a source they have
verified is appropriate to use -- this module records that citation, it does
not judge it. Concretely:

* Every row must carry its own ``source_url``, ``source_title``,
  ``source_publisher``, ``source_access_date``, and ``source_license_note``.
* ``source_license_note`` is documented curator input describing the *cited
  source's* stated reuse terms (e.g. "Wikipedia article table, CC BY-SA 4.0").
  This module records that note; it cannot and does not verify that the
  curator's use of the source is lawful. **Dataset maintainers are responsible
  for confirming their own reuse rights before adding a row.**
* Only factual metadata is modeled here (name, location, distinction, dates,
  a citation). There is no field for review text, editorial descriptions,
  photographs, or logos, and none should ever be added.
* Nothing in this module implies endorsement by Michelin or by any cited
  source (e.g. Wikimedia); PlateProof is independent (see CLAUDE.md).
* A Michelin distinction is contextual metadata. It is never treated as, or
  merged with, health-inspection evidence, and it never implies food safety.

No committed fixture or template in this repository contains a real Michelin
restaurant -- only fictional examples (see ``tests/ingestion/fixtures/`` and
``data/reference/michelin_seed_template.csv``).

Entity model
------------
A restaurant's *identity* (name/address/location) is modeled separately from
its edition-specific *distinctions* (a restaurant can hold a star tier in one
year, a different tier the next, a Green Star alongside a star in the same
year, or nothing in a later year -- none of that is a conflict; see
:func:`distinction_events_as_of`). Matching (in
``plateproof.matching.entity_resolution``) is performed once against the
restaurant identity, never once per distinction event.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from plateproof.matching.normalize import normalize_address, normalize_name

Jurisdiction = Literal["nyc", "florida"]
MichelinDistinction = Literal[
    "michelin_selected", "bib_gourmand", "one_star", "two_stars", "three_stars", "green_star"
]
ProvenanceConfidence = Literal["verified_primary", "verified_secondary"]

MICHELIN_DISTINCTIONS: frozenset[str] = frozenset(
    {"michelin_selected", "bib_gourmand", "one_star", "two_stars", "three_stars", "green_star"}
)
_PROVENANCE_CONFIDENCE_VALUES: frozenset[str] = frozenset(
    {"verified_primary", "verified_secondary"}
)

MICHELIN_SEED_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "name_as_published",
        "city",
        "region",
        "jurisdiction_candidate",
        "distinction",
        "guide_name",
        "guide_year",
        "source_url",
        "source_title",
        "source_publisher",
        "source_access_date",
        "source_license_note",
        "provenance_confidence",
    }
)

MICHELIN_SEED_OPTIONAL_COLUMNS: frozenset[str] = frozenset(
    {
        "address_as_published",
        "postal_code",
        "latitude",
        "longitude",
        "announced_date",
        "source_row_ref",
        "curator_restaurant_ref",
    }
)

_RESTAURANT_ID_DIGEST_HEX = 32
_EVENT_ID_DIGEST_HEX = 32


class MichelinRestaurant(BaseModel):
    """A stable restaurant identity, independent of any single guide edition."""

    model_config = ConfigDict(frozen=True)

    michelin_restaurant_id: str
    name_as_published: str
    normalized_name: str
    address_as_published: str | None
    normalized_address: str | None
    city: str | None
    region: str | None
    postal_code: str | None
    latitude: float | None
    longitude: float | None
    jurisdiction_candidate: Jurisdiction | None
    identity_source_note: str
    curator_restaurant_ref: str | None = None


class MichelinDistinctionEvent(BaseModel):
    """One edition-specific factual distinction, attached to a restaurant identity."""

    model_config = ConfigDict(frozen=True)

    michelin_distinction_event_id: str
    michelin_restaurant_id: str
    distinction: MichelinDistinction
    guide_name: str
    guide_year: int
    announced_date: date | None
    source_url: str
    source_title: str
    source_publisher: str
    source_access_date: date
    source_license_note: str
    provenance_confidence: ProvenanceConfidence
    source_row_ref: str | None = None


class MichelinLoadReport(BaseModel):
    """Deterministic, auditable summary of one seed-file load."""

    model_config = ConfigDict(frozen=True)

    input_row_count: int
    restaurant_count: int
    distinction_event_count: int
    rejected_row_count: int
    rejected_reasons: dict[str, int]
    conflicting_restaurant_ids: list[str]
    conflicting_distinction_event_ids: list[str]
    possible_identity_drift_curator_refs: list[str]


@dataclass(frozen=True)
class MichelinLoadResult:
    restaurants: list[MichelinRestaurant]
    distinction_events: list[MichelinDistinctionEvent]
    report: MichelinLoadReport


def compute_michelin_restaurant_id(
    jurisdiction_candidate: str | None,
    normalized_name: str,
    normalized_address: str | None,
    postal_code: str | None,
) -> str:
    """Deterministic identity id.

    Identity-change policy: the id is a function of jurisdiction, normalized
    name, normalized address, and postal code only. If any of those change
    between rows describing what a curator believes is "the same" restaurant
    (a relocation, or an inconsistently transcribed address), the computed id
    changes too and two separate :class:`MichelinRestaurant` records result.
    This is deliberate: identity is derived from auditable, computed fields,
    never from curator say-so alone. :func:`load_michelin_seed` reports such
    cases (see ``possible_identity_drift_curator_refs``) for human review
    rather than silently merging or silently splitting them further.
    """
    key = "\x1f".join(
        [
            jurisdiction_candidate or "",
            normalized_name,
            normalized_address or "",
            postal_code or "",
        ]
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:_RESTAURANT_ID_DIGEST_HEX]
    return f"michelin:r:{digest}"


def compute_distinction_event_id(
    michelin_restaurant_id: str,
    guide_year: int,
    distinction: str,
    source_ref: str,
) -> str:
    """Deterministic event id from restaurant id + edition + distinction + a
    stable source reference (``source_row_ref`` when given, else
    ``source_url``)."""
    key = "\x1f".join([michelin_restaurant_id, str(guide_year), distinction, source_ref])
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:_EVENT_ID_DIGEST_HEX]
    return f"michelin:e:{digest}"


def distinction_events_as_of(
    events: list[MichelinDistinctionEvent],
    michelin_restaurant_id: str,
    as_of_date: date,
) -> list[MichelinDistinctionEvent]:
    """Distinction events for one restaurant known to have been announced on
    or before ``as_of_date``.

    Conservative by design: an event whose ``announced_date`` is unknown is
    **excluded**, never assumed to have been in effect. This module never
    infers that a distinction persists into a later edition, and never infers
    that its absence from a later edition means it was lost -- only recorded
    facts (an announcement date) are used.
    """
    return [
        event
        for event in events
        if event.michelin_restaurant_id == michelin_restaurant_id
        and event.announced_date is not None
        and event.announced_date <= as_of_date
    ]


def assert_events_reference_known_restaurants(
    restaurants: list[MichelinRestaurant],
    events: list[MichelinDistinctionEvent],
) -> None:
    """Raise ``ValueError`` naming any event whose ``michelin_restaurant_id``
    does not correspond to a restaurant in ``restaurants``."""
    known = {r.michelin_restaurant_id for r in restaurants}
    dangling = sorted(
        {e.michelin_restaurant_id for e in events if e.michelin_restaurant_id not in known}
    )
    if dangling:
        raise ValueError(
            f"distinction events reference unknown Michelin restaurant id(s): {dangling}"
        )


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _parse_date(value: str | None) -> tuple[date | None, bool]:
    """Returns (value, malformed). Blank -> (None, False). Unparseable -> (None, True)."""
    text = _clean(value)
    if text is None:
        return None, False
    try:
        return date.fromisoformat(text), False
    except ValueError:
        return None, True


def _parse_float(value: str | None) -> float | None:
    text = _clean(value)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_michelin_seed(path: str | Path | None) -> MichelinLoadResult:
    """Load and validate a hand-curated Michelin seed CSV.

    ``path=None`` means Michelin is not configured -- returns an empty result,
    no error (the application must run fully with Michelin data absent). A
    ``path`` that does not exist is an error (``FileNotFoundError``): the
    caller asked for a specific file. An existing, header-only file is valid
    and returns an empty result.

    Rows failing structural or provenance validation are rejected (counted by
    reason in the report), never silently coerced. Rows sharing computed
    identity are collapsed into one :class:`MichelinRestaurant`; disagreement
    on fields not covered by the identity hash (e.g. inconsistent ``city``
    text for the same name/address/postal) is reported as a conflict, not
    silently resolved by picking one value.
    """
    if path is None:
        empty_report = MichelinLoadReport(
            input_row_count=0,
            restaurant_count=0,
            distinction_event_count=0,
            rejected_row_count=0,
            rejected_reasons={},
            conflicting_restaurant_ids=[],
            conflicting_distinction_event_ids=[],
            possible_identity_drift_curator_refs=[],
        )
        return MichelinLoadResult([], [], empty_report)

    resolved = Path(path)
    with resolved.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = frozenset(reader.fieldnames or [])
        missing = MICHELIN_SEED_REQUIRED_COLUMNS - fieldnames
        if missing:
            raise ValueError(f"Michelin seed CSV is missing required column(s): {sorted(missing)}")
        rows = list(reader)

    rejected_reasons: dict[str, int] = {}
    restaurants_by_id: dict[str, MichelinRestaurant] = {}
    identity_fields_by_id: dict[str, tuple[str | None, str | None]] = {}  # (city, region)
    conflicting_restaurant_ids: set[str] = set()
    events_by_id: dict[str, MichelinDistinctionEvent] = {}
    conflicting_event_ids: set[str] = set()
    curator_ref_ids: dict[str, set[str]] = {}

    def _reject(reason: str) -> None:
        rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1

    for row in rows:
        name = _clean(row.get("name_as_published"))
        city = _clean(row.get("city"))
        region = _clean(row.get("region"))
        jurisdiction = _clean(row.get("jurisdiction_candidate"))
        distinction = _clean(row.get("distinction"))
        guide_name = _clean(row.get("guide_name"))
        source_url = _clean(row.get("source_url"))
        source_title = _clean(row.get("source_title"))
        source_publisher = _clean(row.get("source_publisher"))
        source_license_note = _clean(row.get("source_license_note"))
        provenance_confidence = _clean(row.get("provenance_confidence"))

        if not name or not city or not region or not guide_name:
            _reject("missing_required_field")
            continue
        if jurisdiction not in ("nyc", "florida"):
            _reject("missing_jurisdiction_candidate")
            continue
        if distinction not in MICHELIN_DISTINCTIONS:
            _reject("unsupported_distinction")
            continue
        if provenance_confidence not in _PROVENANCE_CONFIDENCE_VALUES:
            _reject("invalid_provenance_confidence")
            continue
        if not (source_url and source_title and source_publisher and source_license_note):
            _reject("missing_provenance")
            continue

        guide_year_raw = _clean(row.get("guide_year"))
        try:
            guide_year = int(guide_year_raw) if guide_year_raw is not None else None
        except ValueError:
            guide_year = None
        if guide_year is None:
            _reject("malformed_guide_year")
            continue

        source_access_date, access_date_malformed = _parse_date(row.get("source_access_date"))
        if source_access_date is None or access_date_malformed:
            _reject("missing_provenance")
            continue

        announced_date, announced_malformed = _parse_date(row.get("announced_date"))
        if announced_malformed:
            _reject("malformed_announced_date")
            continue

        address_as_published = _clean(row.get("address_as_published"))
        normalized_addr = normalize_address(address_as_published)
        postal_code = _clean(row.get("postal_code"))
        latitude = _parse_float(row.get("latitude"))
        longitude = _parse_float(row.get("longitude"))
        curator_ref = _clean(row.get("curator_restaurant_ref"))
        source_row_ref = _clean(row.get("source_row_ref"))

        normalized_nm = normalize_name(name)
        restaurant_id = compute_michelin_restaurant_id(
            jurisdiction,
            normalized_nm,
            normalized_addr.normalized if normalized_addr else None,
            postal_code,
        )

        if restaurant_id in restaurants_by_id:
            existing_city, existing_region = identity_fields_by_id[restaurant_id]
            if existing_city != city or existing_region != region:
                conflicting_restaurant_ids.add(restaurant_id)
        else:
            restaurants_by_id[restaurant_id] = MichelinRestaurant(
                michelin_restaurant_id=restaurant_id,
                name_as_published=name,
                normalized_name=normalized_nm,
                address_as_published=address_as_published,
                normalized_address=normalized_addr.normalized if normalized_addr else None,
                city=city,
                region=region,
                postal_code=postal_code,
                latitude=latitude,
                longitude=longitude,
                jurisdiction_candidate=jurisdiction,
                identity_source_note=f"seed row: {source_row_ref or source_url}",
                curator_restaurant_ref=curator_ref,
            )
            identity_fields_by_id[restaurant_id] = (city, region)

        if curator_ref:
            curator_ref_ids.setdefault(curator_ref, set()).add(restaurant_id)

        event_ref = source_row_ref or source_url
        event_id = compute_distinction_event_id(restaurant_id, guide_year, distinction, event_ref)
        event = MichelinDistinctionEvent(
            michelin_distinction_event_id=event_id,
            michelin_restaurant_id=restaurant_id,
            distinction=distinction,
            guide_name=guide_name,
            guide_year=guide_year,
            announced_date=announced_date,
            source_url=source_url,
            source_title=source_title,
            source_publisher=source_publisher,
            source_access_date=source_access_date,
            source_license_note=source_license_note,
            provenance_confidence=provenance_confidence,
            source_row_ref=source_row_ref,
        )
        if event_id in events_by_id:
            if events_by_id[event_id] != event:
                conflicting_event_ids.add(event_id)
        else:
            events_by_id[event_id] = event

    possible_drift = sorted(ref for ref, ids in curator_ref_ids.items() if len(ids) > 1)

    report = MichelinLoadReport(
        input_row_count=len(rows),
        restaurant_count=len(restaurants_by_id),
        distinction_event_count=len(events_by_id),
        rejected_row_count=sum(rejected_reasons.values()),
        rejected_reasons=rejected_reasons,
        conflicting_restaurant_ids=sorted(conflicting_restaurant_ids),
        conflicting_distinction_event_ids=sorted(conflicting_event_ids),
        possible_identity_drift_curator_refs=possible_drift,
    )
    return MichelinLoadResult(
        restaurants=list(restaurants_by_id.values()),
        distinction_events=list(events_by_id.values()),
        report=report,
    )
