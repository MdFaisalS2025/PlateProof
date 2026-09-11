"""Tests for plateproof.matching.entity_resolution.

All restaurants used here are fictional synthetic fixtures, not real
establishments.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

FIXED_TS = datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC)


def _official(**overrides: Any) -> Any:
    from plateproof.matching.entity_resolution import OfficialRestaurant

    defaults: dict[str, Any] = dict(
        restaurant_id="nyc:1000001",
        jurisdiction="nyc",
        source_id="1000001",
        name="Example Bistro",
        normalized_name="example bistro",
        address_as_published="100 Fictional Ave",
        normalized_address="100 fictional ave",
        city="New York",
        postal_code="10001",
        latitude=None,
        longitude=None,
    )
    defaults.update(overrides)
    return OfficialRestaurant(**defaults)


def _michelin(**overrides: Any) -> Any:
    from plateproof.ingestion.michelin import MichelinRestaurant

    defaults: dict[str, Any] = dict(
        michelin_restaurant_id="michelin:r:aaaa",
        name_as_published="Example Bistro",
        normalized_name="example bistro",
        address_as_published="100 Fictional Ave",
        normalized_address="100 fictional ave",
        city="New York",
        region="NY",
        postal_code="10001",
        latitude=None,
        longitude=None,
        jurisdiction_candidate="nyc",
        identity_source_note="fixture",
        curator_restaurant_ref=None,
    )
    defaults.update(overrides)
    return MichelinRestaurant(**defaults)


# --- OfficialRestaurant projections -----------------------------------------


def test_official_restaurants_from_nyc_events_uses_task2_fields() -> None:
    from plateproof.ingestion.nyc import build_nyc_inspection_events, load_nyc_raw
    from plateproof.matching.entity_resolution import official_restaurants_from_nyc_events

    fixture = Path(__file__).parent.parent / "ingestion" / "fixtures" / "nyc_sample_soda.csv"
    result = build_nyc_inspection_events(load_nyc_raw(fixture), ingested_at=FIXED_TS)
    officials = official_restaurants_from_nyc_events(result.inspection_events)
    assert officials
    assert all(o.jurisdiction == "nyc" for o in officials)
    assert all(o.restaurant_id.startswith("nyc:") for o in officials)
    # restaurant_id/source_id are never modified by projection
    ids = {o.restaurant_id for o in officials}
    assert ids == set(result.inspection_events.get_column("restaurant_id").to_list())


def test_official_restaurants_from_florida_requires_staging_join() -> None:
    from plateproof.ingestion.florida import (
        FloridaExtractSource,
        build_florida_inspection_events,
        load_florida_extracts,
    )
    from plateproof.matching.entity_resolution import official_restaurants_from_florida

    fixture = Path(__file__).parent.parent / "ingestion" / "fixtures" / "fl_sample_current.csv"
    staging = load_florida_extracts([FloridaExtractSource(path=fixture)])
    result = build_florida_inspection_events(staging, ingested_at=FIXED_TS)
    officials, missing = official_restaurants_from_florida(staging, result.inspection_events)
    assert officials
    assert all(o.jurisdiction == "florida" for o in officials)
    # Florida's normalized inspection_events never carries address -- must come from staging
    assert any(o.address_as_published for o in officials)
    assert missing == []


def test_official_restaurants_from_florida_missing_staging_columns_raises() -> None:
    import polars as pl
    import pytest

    from plateproof.matching.entity_resolution import official_restaurants_from_florida

    bad_staging = pl.DataFrame({"license_number": ["1"]})
    empty_events = pl.DataFrame(
        schema={"restaurant_id": pl.String, "jurisdiction": pl.String, "source_id": pl.String}
    )
    with pytest.raises(ValueError, match="location_address"):
        official_restaurants_from_florida(bad_staging, empty_events)


def test_official_restaurants_from_florida_reports_inconsistent_identity() -> None:
    import polars as pl

    from plateproof.matching.entity_resolution import official_restaurants_from_florida

    staging = pl.DataFrame(
        {
            "license_number": ["1000001"],
            "dba": ["Anna's Diner"],
            "location_address": ["1 Main St"],
            "location_city": ["Miami"],
            "location_zip": ["33101"],
        }
    )
    events = pl.DataFrame(
        {
            "restaurant_id": ["florida:1000001", "florida:9999999"],
            "jurisdiction": ["florida", "florida"],
            "source_id": ["1000001", "9999999"],
        }
    )
    officials, missing = official_restaurants_from_florida(staging, events)
    assert [o.restaurant_id for o in officials] == ["florida:1000001"]
    assert missing == ["florida:9999999"]


def test_official_identifiers_never_modified_by_matching() -> None:
    from plateproof.matching import entity_resolution as module

    official = _official()
    restaurant = _michelin()
    before = official.restaurant_id
    evidence = module.match_candidate(
        official, restaurant, blocking_rules=["postal_code_exact"], generated_at=FIXED_TS
    )
    assert official.restaurant_id == before
    assert evidence.official_restaurant_id == before


# --- blocking / candidate generation ----------------------------------------


def test_jurisdiction_hard_filter_produces_no_cross_jurisdiction_pairs() -> None:
    from plateproof.matching.entity_resolution import generate_candidate_pairs

    official = _official(jurisdiction="nyc")
    restaurant = _michelin(jurisdiction_candidate="florida")
    pairs = generate_candidate_pairs([official], [restaurant])
    assert pairs == []


def test_blocking_does_not_depend_on_first_name_token() -> None:
    from plateproof.matching.entity_resolution import generate_candidate_pairs

    # Different first tokens, different postal/address (so postal/street blocking
    # cannot fire) -- only a shared distinctive token plus city should connect them.
    official = _official(
        name="Chef Daniel's Grill Room",
        normalized_name="chef daniels grill room",
        postal_code="10099",
        address_as_published="999 Other Ave",
        normalized_address="999 other ave",
    )
    restaurant = _michelin(
        name_as_published="The Grill Room by Chef Daniel",
        normalized_name="grill room by chef daniel",
        postal_code="10001",
    )
    pairs = generate_candidate_pairs([official], [restaurant])
    assert len(pairs) == 1
    assert "shared_name_token_and_city" in pairs[0][2]


def test_blocking_rules_are_recorded() -> None:
    from plateproof.matching.entity_resolution import generate_candidate_pairs

    official = _official()
    restaurant = _michelin()
    pairs = generate_candidate_pairs([official], [restaurant])
    assert len(pairs) == 1
    _, _, rules = pairs[0]
    assert "postal_code_exact" in rules


# --- scoring / required vs optional evidence --------------------------------


def test_name_alone_never_accepts() -> None:
    from plateproof.matching.entity_resolution import match_candidate

    official = _official(
        normalized_address=None, address_as_published=None, postal_code=None, city=None
    )
    restaurant = _michelin()
    evidence = match_candidate(official, restaurant, blocking_rules=[], generated_at=FIXED_TS)
    assert evidence.decision != "accept"
    assert "address_similarity" in evidence.required_evidence_missing


def test_missing_distance_alone_does_not_cap_florida_match() -> None:
    from plateproof.matching.entity_resolution import match_candidate

    official = _official(
        jurisdiction="florida", restaurant_id="florida:1", latitude=None, longitude=None
    )
    restaurant = _michelin(jurisdiction_candidate="florida", latitude=None, longitude=None)
    evidence = match_candidate(
        official, restaurant, blocking_rules=["postal_code_exact"], generated_at=FIXED_TS
    )
    assert evidence.required_evidence_missing == []
    assert "distance_score" in evidence.optional_evidence_missing
    assert evidence.decision == "accept"
    assert evidence.overall_score >= 0.92


def test_missing_address_prevents_automatic_acceptance_for_chain() -> None:
    from plateproof.matching.entity_resolution import match_candidate

    official = _official(normalized_address=None, address_as_published=None)
    restaurant = _michelin()  # exact name + exact postal, but official has no address
    evidence = match_candidate(
        official, restaurant, blocking_rules=["postal_code_exact"], generated_at=FIXED_TS
    )
    assert evidence.decision != "accept"
    assert "address_similarity" in evidence.required_evidence_missing


def test_conflicting_postal_code_prevents_automatic_acceptance() -> None:
    from plateproof.matching.entity_resolution import match_candidate

    official = _official(postal_code="10001")
    restaurant = _michelin(postal_code="10099")
    evidence = match_candidate(
        official, restaurant, blocking_rules=["shared_name_token_and_city"], generated_at=FIXED_TS
    )
    assert evidence.decision != "accept"
    assert "postal_code_conflict" in evidence.conflicting_evidence


def test_conflicting_address_prevents_automatic_acceptance() -> None:
    from plateproof.matching.entity_resolution import match_candidate

    official = _official(
        normalized_address="1 far away road", address_as_published="1 Far Away Road"
    )
    restaurant = _michelin()
    evidence = match_candidate(
        official, restaurant, blocking_rules=["postal_code_exact"], generated_at=FIXED_TS
    )
    assert evidence.decision != "accept"
    assert "address_conflict" in evidence.conflicting_evidence


def test_exact_match_accepts_with_full_evidence() -> None:
    from plateproof.matching.entity_resolution import match_candidate

    official = _official()
    restaurant = _michelin()
    evidence = match_candidate(
        official, restaurant, blocking_rules=["postal_code_exact"], generated_at=FIXED_TS
    )
    assert evidence.decision == "accept"
    assert evidence.overall_score >= 0.92


def test_accept_threshold_boundary() -> None:
    from plateproof.matching.entity_resolution import _decide

    assert _decide(0.92) == "accept"
    assert _decide(0.9199) == "review"


def test_review_boundary() -> None:
    from plateproof.matching.entity_resolution import _decide

    assert _decide(0.75) == "review"
    assert _decide(0.7499) == "reject"


def test_reject_below_threshold() -> None:
    from plateproof.matching.entity_resolution import _decide

    assert _decide(0.5) == "reject"


def test_similar_name_nearby_address_lands_in_review_or_accept() -> None:
    from plateproof.matching.entity_resolution import match_candidate

    official = _official(name="Example Bistros", normalized_name="example bistros")
    restaurant = _michelin()
    evidence = match_candidate(
        official, restaurant, blocking_rules=["postal_code_exact"], generated_at=FIXED_TS
    )
    assert evidence.decision in ("review", "accept")


def test_same_name_distant_address_rejects() -> None:
    from plateproof.matching.entity_resolution import match_candidate

    official = _official(
        normalized_address="9999 distant blvd",
        address_as_published="9999 Distant Blvd",
        postal_code="99999",
        city="Nowhere",
    )
    restaurant = _michelin()
    evidence = match_candidate(
        official, restaurant, blocking_rules=["shared_name_token_and_city"], generated_at=FIXED_TS
    )
    assert evidence.decision == "reject"


def test_different_names_at_same_address_reject_despite_full_address_match() -> None:
    from plateproof.matching.entity_resolution import match_candidate

    official = _official(name="Rose Club", normalized_name="rose club")
    restaurant = _michelin(name_as_published="Example Bistro", normalized_name="example bistro")
    evidence = match_candidate(
        official, restaurant, blocking_rules=["postal_code_exact"], generated_at=FIXED_TS
    )
    assert evidence.decision != "accept"


def test_deterministic_score_for_same_inputs() -> None:
    from plateproof.matching.entity_resolution import match_candidate

    official = _official()
    restaurant = _michelin()
    a = match_candidate(
        official, restaurant, blocking_rules=["postal_code_exact"], generated_at=FIXED_TS
    )
    b = match_candidate(
        official, restaurant, blocking_rules=["postal_code_exact"], generated_at=FIXED_TS
    )
    assert a.overall_score == b.overall_score
    assert a.decision == b.decision


# --- conflict policy (strict) -----------------------------------------------


def test_one_michelin_vs_multiple_officials_forces_review() -> None:
    from plateproof.matching.entity_resolution import match_candidate, resolve_matches

    restaurant = _michelin()
    official_a = _official(restaurant_id="nyc:1")
    official_b = _official(restaurant_id="nyc:2")
    ev_a = match_candidate(
        official_a, restaurant, blocking_rules=["postal_code_exact"], generated_at=FIXED_TS
    )
    ev_b = match_candidate(
        official_b, restaurant, blocking_rules=["postal_code_exact"], generated_at=FIXED_TS
    )
    resolved = resolve_matches([ev_a, ev_b])
    assert all(e.decision != "accept" for e in resolved)
    assert all(e.conflicting_candidates for e in resolved)


def test_multiple_michelin_vs_one_official_forces_review() -> None:
    from plateproof.matching.entity_resolution import match_candidate, resolve_matches

    official = _official()
    restaurant_a = _michelin(michelin_restaurant_id="michelin:r:aaaa")
    restaurant_b = _michelin(michelin_restaurant_id="michelin:r:bbbb")
    ev_a = match_candidate(
        official, restaurant_a, blocking_rules=["postal_code_exact"], generated_at=FIXED_TS
    )
    ev_b = match_candidate(
        official, restaurant_b, blocking_rules=["postal_code_exact"], generated_at=FIXED_TS
    )
    resolved = resolve_matches([ev_a, ev_b])
    assert all(e.decision != "accept" for e in resolved)


def test_no_margin_exception_even_with_large_score_gap() -> None:
    """Two candidates for one official, both accept/review-eligible, with a large
    score gap (0.98 vs 0.80). The strict policy forces BOTH to review -- no
    margin ever preserves the higher-scoring "winner" as an accept."""
    from plateproof.matching.entity_resolution import MatchEvidence, resolve_matches

    def _evidence(michelin_id: str, score: float, decision: str) -> MatchEvidence:
        return MatchEvidence(
            official_restaurant_id="nyc:1",
            michelin_restaurant_id=michelin_id,
            jurisdiction="nyc",
            name_similarity=score,
            name_similarity_components={},
            address_similarity=score,
            postal_code_match=True,
            city_region_match=True,
            distance_km=None,
            distance_score=None,
            blocking_rules=["postal_code_exact"],
            required_evidence_missing=[],
            optional_evidence_missing=["distance_score"],
            conflicting_evidence=[],
            overall_score=score,
            decision=decision,  # type: ignore[arg-type]
            reasons=[],
            conflicting_candidates=[],
            matcher_version="test",
            generated_at=FIXED_TS,
        )

    strong = _evidence("michelin:r:strong", 0.98, "accept")
    weak = _evidence("michelin:r:weak", 0.80, "review")
    resolved = resolve_matches([strong, weak])
    assert all(e.decision != "accept" for e in resolved)
    assert all(e.conflicting_candidates for e in resolved)


def test_assert_one_to_one_passes_on_clean_accepted_set() -> None:
    from plateproof.matching.entity_resolution import assert_one_to_one, match_candidate

    official = _official()
    restaurant = _michelin()
    evidence = match_candidate(
        official, restaurant, blocking_rules=["postal_code_exact"], generated_at=FIXED_TS
    )
    assert assert_one_to_one([evidence]) is None


def test_assert_one_to_one_raises_on_violation() -> None:
    import pytest

    from plateproof.matching.entity_resolution import MatchEvidence, assert_one_to_one

    def _accepted(official_id: str, michelin_id: str) -> MatchEvidence:
        return MatchEvidence(
            official_restaurant_id=official_id,
            michelin_restaurant_id=michelin_id,
            jurisdiction="nyc",
            name_similarity=1.0,
            name_similarity_components={},
            address_similarity=1.0,
            postal_code_match=True,
            city_region_match=True,
            distance_km=None,
            distance_score=None,
            blocking_rules=[],
            required_evidence_missing=[],
            optional_evidence_missing=["distance_score"],
            conflicting_evidence=[],
            overall_score=1.0,
            decision="accept",
            reasons=[],
            conflicting_candidates=[],
            matcher_version="test",
            generated_at=FIXED_TS,
        )

    with pytest.raises(ValueError):
        assert_one_to_one([_accepted("nyc:1", "michelin:r:a"), _accepted("nyc:2", "michelin:r:a")])


def test_shuffled_input_produces_stable_output() -> None:
    from plateproof.matching.entity_resolution import (
        generate_candidate_pairs,
        match_candidate,
        resolve_matches,
    )

    officials = [_official(restaurant_id=f"nyc:{i}", postal_code="10001") for i in range(3)]
    restaurants = [
        _michelin(michelin_restaurant_id=f"michelin:r:{i}", postal_code="10001") for i in range(3)
    ]

    def _run(off: list[Any], mich: list[Any]) -> set[tuple[str, str, str]]:
        pairs = generate_candidate_pairs(off, mich)
        evidence = [match_candidate(o, m, rules, generated_at=FIXED_TS) for o, m, rules in pairs]
        resolved = resolve_matches(evidence)
        return {(e.official_restaurant_id, e.michelin_restaurant_id, e.decision) for e in resolved}

    forward = _run(officials, restaurants)
    backward = _run(list(reversed(officials)), list(reversed(restaurants)))
    assert forward == backward


# --- review queue -------------------------------------------------------


def test_export_review_queue_is_entity_level_not_per_distinction_event() -> None:
    from plateproof.ingestion.michelin import MichelinDistinctionEvent
    from plateproof.matching.entity_resolution import export_review_queue, match_candidate

    official = _official()
    restaurant = _michelin()
    evidence = match_candidate(
        official, restaurant, blocking_rules=["postal_code_exact"], generated_at=FIXED_TS
    )
    events = [
        MichelinDistinctionEvent(
            michelin_distinction_event_id=f"michelin:e:{i}",
            michelin_restaurant_id=restaurant.michelin_restaurant_id,
            distinction="one_star",
            guide_name="Fixture Guide 2024",
            guide_year=2024 + i,
            announced_date=None,
            source_url="https://example.invalid/fixture",
            source_title="Fixture",
            source_publisher="Fixture publisher",
            source_access_date=FIXED_TS.date(),
            source_license_note="fixture only",
            provenance_confidence="verified_secondary",
            source_row_ref=None,
        )
        for i in range(2)
    ]
    queue = export_review_queue(
        [evidence],
        {official.restaurant_id: official},
        {restaurant.michelin_restaurant_id: restaurant},
        events,
        include_rejected=True,
    )
    assert queue.height == 1  # one row per entity pair, not one per distinction event
    row = queue.to_dicts()[0]
    assert len(row["distinction_events"]) == 2
    assert row["reviewer_decision"] is None
    assert row["reviewer_notes"] is None


def test_review_queue_excludes_rejected_by_default() -> None:
    from plateproof.matching.entity_resolution import export_review_queue, match_candidate

    official = _official(
        normalized_address="9999 distant blvd",
        address_as_published="9999 Distant Blvd",
        postal_code="99999",
    )
    restaurant = _michelin()
    evidence = match_candidate(official, restaurant, blocking_rules=[], generated_at=FIXED_TS)
    assert evidence.decision == "reject"
    queue = export_review_queue(
        [evidence],
        {official.restaurant_id: official},
        {restaurant.michelin_restaurant_id: restaurant},
        [],
    )
    assert queue.height == 0


# --- empty / disabled Michelin data ------------------------------------


def test_empty_michelin_dataset_produces_no_matches_no_errors() -> None:
    from plateproof.matching.entity_resolution import generate_candidate_pairs

    officials = [_official()]
    assert generate_candidate_pairs(officials, []) == []


def test_no_health_or_safety_inference_fields_on_match_evidence() -> None:
    from plateproof.matching.entity_resolution import MatchEvidence

    forbidden = {"health_score", "safety_score", "risk_band", "grade"}
    assert forbidden.isdisjoint(MatchEvidence.model_fields)
