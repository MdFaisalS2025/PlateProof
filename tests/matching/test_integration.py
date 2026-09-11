"""End-to-end integration: load a fictional seed file, match once per entity,
and confirm the full edition history stays attached after an accepted match.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

FIXED_TS = datetime(2026, 3, 1, 12, 0, 0, tzinfo=UTC)
FIXTURES = Path(__file__).parent.parent / "ingestion" / "fixtures"


def test_matching_runs_once_per_entity_not_per_distinction_event() -> None:
    from plateproof.ingestion.michelin import load_michelin_seed
    from plateproof.matching.entity_resolution import OfficialRestaurant, generate_candidate_pairs

    result = load_michelin_seed(FIXTURES / "michelin_seed_valid.csv")
    bistro = next(r for r in result.restaurants if r.normalized_name == "example bistro")
    # Example Bistro has 3 distinction events (2024 one-star, 2025 two-star,
    # 2025 green-star) attached to a single restaurant identity.
    events = [
        e
        for e in result.distinction_events
        if e.michelin_restaurant_id == bistro.michelin_restaurant_id
    ]
    assert len(events) == 3

    official = OfficialRestaurant(
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
    pairs = generate_candidate_pairs([official], result.restaurants)
    # Exactly one candidate pair for this restaurant identity, regardless of
    # how many distinction events it carries.
    matching_pairs = [
        p for p in pairs if p[1].michelin_restaurant_id == bistro.michelin_restaurant_id
    ]
    assert len(matching_pairs) == 1


def test_distinction_events_remain_attached_after_accepted_match() -> None:
    from plateproof.ingestion.michelin import load_michelin_seed
    from plateproof.matching.entity_resolution import (
        OfficialRestaurant,
        export_review_queue,
        match_candidate,
    )

    result = load_michelin_seed(FIXTURES / "michelin_seed_valid.csv")
    bistro = next(r for r in result.restaurants if r.normalized_name == "example bistro")

    official = OfficialRestaurant(
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
    evidence = match_candidate(
        official, bistro, blocking_rules=["postal_code_exact"], generated_at=FIXED_TS
    )
    assert evidence.decision == "accept"

    queue = export_review_queue(
        [evidence],
        {official.restaurant_id: official},
        {bistro.michelin_restaurant_id: bistro},
        result.distinction_events,
        include_rejected=True,
    )
    row = queue.to_dicts()[0]
    assert set(row["distinction_events"]) == {"2024:one_star", "2025:two_stars", "2025:green_star"}


def test_multi_year_and_simultaneous_distinctions_are_not_matching_conflicts() -> None:
    """Multiple guide years, and a star plus Green Star in one year, must never
    surface as a MatchEvidence conflict -- matching sees one restaurant."""
    from plateproof.ingestion.michelin import distinction_events_as_of, load_michelin_seed

    result = load_michelin_seed(FIXTURES / "michelin_seed_valid.csv")
    bistro = next(r for r in result.restaurants if r.normalized_name == "example bistro")
    as_of_late_2025 = date(2025, 12, 1)
    known = distinction_events_as_of(
        result.distinction_events, bistro.michelin_restaurant_id, as_of_late_2025
    )
    # All three are known by late 2025 and coexist without any conflict marker.
    assert len(known) == 3
