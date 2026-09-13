"""Tests for plateproof.graph.queries: latest-inspection tie-breaking,
same-day handling, recurring-violation frequency semantics, dated
description labels, and Michelin as-of exclusion of unknown announced
dates."""

from __future__ import annotations

from datetime import date
from typing import Any


def test_latest_inspection_picks_newest_date_then_id(
    make_build_input: Any, restaurant_row: Any, inspection_row: Any
) -> None:
    from plateproof.graph.builder import build_graph
    from plateproof.graph.queries import latest_inspection

    build_input = make_build_input(
        restaurants=[restaurant_row(restaurant_id="nyc:1")],
        inspections=[
            inspection_row(
                inspection_id="nyc:1:1", restaurant_id="nyc:1", inspection_date=date(2025, 1, 1)
            ),
            inspection_row(
                inspection_id="nyc:1:2", restaurant_id="nyc:1", inspection_date=date(2025, 6, 1)
            ),
        ],
    )
    result = build_graph(build_input)
    latest = latest_inspection(result.graph, "nyc:1")
    assert latest is not None
    assert latest.inspection_id == "nyc:1:2"
    assert latest.inspection_date == date(2025, 6, 1)
    assert latest.same_day_count == 1


def test_latest_inspection_reports_same_day_count_for_conflicting_records(
    make_build_input: Any, restaurant_row: Any, inspection_row: Any
) -> None:
    from plateproof.graph.builder import build_graph
    from plateproof.graph.queries import latest_inspection

    build_input = make_build_input(
        restaurants=[restaurant_row(restaurant_id="nyc:1")],
        inspections=[
            inspection_row(
                inspection_id="nyc:1:1", restaurant_id="nyc:1", inspection_date=date(2025, 6, 1)
            ),
            inspection_row(
                inspection_id="nyc:1:2", restaurant_id="nyc:1", inspection_date=date(2025, 6, 1)
            ),
        ],
    )
    result = build_graph(build_input)
    latest = latest_inspection(result.graph, "nyc:1")
    assert latest is not None
    assert latest.same_day_count == 2
    # deterministic tie-break: highest inspection_id wins, same as
    # Repository.list_inspections' ORDER BY ... inspection_id DESC.
    assert latest.inspection_id == "nyc:1:2"


def test_latest_inspection_returns_none_with_no_history(
    make_build_input: Any, restaurant_row: Any
) -> None:
    from plateproof.graph.builder import build_graph
    from plateproof.graph.queries import latest_inspection

    result = build_graph(make_build_input(restaurants=[restaurant_row(restaurant_id="nyc:1")]))
    assert latest_inspection(result.graph, "nyc:1") is None


def test_recurring_violation_counts_distinguish_occurrences_from_total_cited(
    make_build_input: Any, restaurant_row: Any, inspection_row: Any, violation_row: Any
) -> None:
    from plateproof.graph.builder import build_graph
    from plateproof.graph.queries import recurring_violation_codes

    build_input = make_build_input(
        restaurants=[restaurant_row(restaurant_id="nyc:1")],
        inspections=[
            inspection_row(
                inspection_id="nyc:1:1", restaurant_id="nyc:1", inspection_date=date(2025, 1, 1)
            ),
            inspection_row(
                inspection_id="nyc:1:2", restaurant_id="nyc:1", inspection_date=date(2025, 6, 1)
            ),
        ],
        violations=[
            violation_row(
                violation_event_id="nyc:v:1",
                inspection_id="nyc:1:1",
                restaurant_id="nyc:1",
                inspection_date=date(2025, 1, 1),
                violation_code_norm="04L",
                count=3,
            ),
            violation_row(
                violation_event_id="nyc:v:2",
                inspection_id="nyc:1:2",
                restaurant_id="nyc:1",
                inspection_date=date(2025, 6, 1),
                violation_code_norm="04L",
                count=1,
                violation_description="Updated description",
            ),
        ],
    )
    result = build_graph(build_input)
    recurring = recurring_violation_codes(result.graph, "nyc:1")
    assert len(recurring) == 1
    item = recurring[0]
    assert item.violation_code_norm == "04L"
    assert item.occurrence_count == 2
    assert item.total_cited_count == 4  # 3 + 1, never conflated with occurrence_count
    assert item.most_recent_description == "Updated description"
    assert item.most_recent_description_date == date(2025, 6, 1)


def test_recurring_violation_codes_excludes_codes_below_minimum(
    make_build_input: Any, restaurant_row: Any, inspection_row: Any, violation_row: Any
) -> None:
    from plateproof.graph.builder import build_graph
    from plateproof.graph.queries import recurring_violation_codes

    build_input = make_build_input(
        restaurants=[restaurant_row(restaurant_id="nyc:1")],
        inspections=[inspection_row(inspection_id="nyc:1:1", restaurant_id="nyc:1")],
        violations=[
            violation_row(
                violation_event_id="nyc:v:1",
                inspection_id="nyc:1:1",
                restaurant_id="nyc:1",
                violation_code_norm="04L",
            )
        ],
    )
    result = build_graph(build_input)
    assert recurring_violation_codes(result.graph, "nyc:1") == []


def test_most_recent_description_for_code_never_overwrites_history(
    make_build_input: Any, restaurant_row: Any, inspection_row: Any, violation_row: Any
) -> None:
    from plateproof.graph.builder import build_graph
    from plateproof.graph.queries import (
        most_recent_description_for_code,
        restaurant_violation_occurrences,
    )

    build_input = make_build_input(
        restaurants=[restaurant_row(restaurant_id="nyc:1")],
        inspections=[
            inspection_row(
                inspection_id="nyc:1:1", restaurant_id="nyc:1", inspection_date=date(2024, 1, 1)
            ),
            inspection_row(
                inspection_id="nyc:1:2", restaurant_id="nyc:1", inspection_date=date(2025, 1, 1)
            ),
        ],
        violations=[
            violation_row(
                violation_event_id="nyc:v:1",
                inspection_id="nyc:1:1",
                restaurant_id="nyc:1",
                inspection_date=date(2024, 1, 1),
                violation_code_norm="04L",
                violation_description="Old wording",
            ),
            violation_row(
                violation_event_id="nyc:v:2",
                inspection_id="nyc:1:2",
                restaurant_id="nyc:1",
                inspection_date=date(2025, 1, 1),
                violation_code_norm="04L",
                violation_description="New wording",
            ),
        ],
    )
    result = build_graph(build_input)
    newest = most_recent_description_for_code(result.graph, "nyc:1", "04L")
    assert newest == ("New wording", date(2025, 1, 1))
    # The older description is still queryable on its own occurrence node.
    descriptions = {
        o["violation_description"] for o in restaurant_violation_occurrences(result.graph, "nyc:1")
    }
    assert descriptions == {"Old wording", "New wording"}


def test_distinctions_as_of_excludes_unknown_announced_date(
    make_build_input: Any, restaurant_row: Any
) -> None:
    from plateproof.graph.builder import build_graph
    from plateproof.graph.queries import distinctions_as_of

    build_input = make_build_input(
        restaurants=[restaurant_row(restaurant_id="nyc:1")],
        michelin_restaurants=[
            {
                "michelin_restaurant_id": "michelin:r:aaa",
                "name_as_published": "Fictional Bistro",
                "jurisdiction_candidate": "nyc",
            }
        ],
        michelin_distinction_events=[
            {
                "michelin_distinction_event_id": "michelin:e:known",
                "michelin_restaurant_id": "michelin:r:aaa",
                "distinction": "one_star",
                "guide_name": "Fictional Guide",
                "guide_year": 2024,
                "announced_date": date(2024, 1, 1),
                "source_url": "https://example.invalid/fictional",
            },
            {
                "michelin_distinction_event_id": "michelin:e:unknown",
                "michelin_restaurant_id": "michelin:r:aaa",
                "distinction": "two_stars",
                "guide_name": "Fictional Guide",
                "guide_year": 2025,
                "announced_date": None,
                "source_url": "https://example.invalid/fictional",
            },
        ],
        restaurant_michelin_matches=[
            {
                "official_restaurant_id": "nyc:1",
                "michelin_restaurant_id": "michelin:r:aaa",
                "decision": "accept",
                "generated_at": "2026-01-01T00:00:00Z",
            }
        ],
    )
    result = build_graph(build_input)
    events = distinctions_as_of(result.graph, "nyc:1", date(2026, 1, 1))
    ids = {e["michelin_distinction_event_id"] for e in events}
    assert ids == {"michelin:e:known"}


def test_distinctions_as_of_respects_the_as_of_date_boundary(
    make_build_input: Any, restaurant_row: Any
) -> None:
    from plateproof.graph.builder import build_graph
    from plateproof.graph.queries import distinctions_as_of

    build_input = make_build_input(
        restaurants=[restaurant_row(restaurant_id="nyc:1")],
        michelin_restaurants=[
            {
                "michelin_restaurant_id": "michelin:r:aaa",
                "name_as_published": "Fictional Bistro",
                "jurisdiction_candidate": "nyc",
            }
        ],
        michelin_distinction_events=[
            {
                "michelin_distinction_event_id": "michelin:e:future",
                "michelin_restaurant_id": "michelin:r:aaa",
                "distinction": "one_star",
                "guide_name": "Fictional Guide",
                "guide_year": 2027,
                "announced_date": date(2027, 1, 1),
                "source_url": "https://example.invalid/fictional",
            }
        ],
        restaurant_michelin_matches=[
            {
                "official_restaurant_id": "nyc:1",
                "michelin_restaurant_id": "michelin:r:aaa",
                "decision": "accept",
                "generated_at": "2026-01-01T00:00:00Z",
            }
        ],
    )
    result = build_graph(build_input)
    assert distinctions_as_of(result.graph, "nyc:1", date(2026, 1, 1)) == []


def test_current_documented_location_and_cuisine(
    make_build_input: Any, restaurant_row: Any
) -> None:
    from plateproof.graph.builder import build_graph
    from plateproof.graph.queries import current_documented_cuisine, current_documented_location

    build_input = make_build_input(
        restaurants=[
            restaurant_row(
                restaurant_id="nyc:1", city="Manhattan", postal_code="10001", cuisine="American"
            )
        ],
    )
    result = build_graph(build_input)
    location = current_documented_location(result.graph, "nyc:1")
    assert location is not None
    assert location["city"] == "Manhattan"
    assert location["postal_code"] == "10001"
    assert current_documented_cuisine(result.graph, "nyc:1") == "american"
