"""Tests for plateproof.ingestion.michelin: loading the curated seed CSV.

All fixtures here are fictional (see module docstrings) -- no real Michelin
restaurant is committed to this repository.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def test_absent_path_returns_empty_result() -> None:
    from plateproof.ingestion.michelin import load_michelin_seed

    result = load_michelin_seed(None)
    assert result.restaurants == []
    assert result.distinction_events == []
    assert result.report.input_row_count == 0


def test_missing_file_path_raises_filenotfound(tmp_path: Path) -> None:
    from plateproof.ingestion.michelin import load_michelin_seed

    with pytest.raises(FileNotFoundError):
        load_michelin_seed(tmp_path / "absent.csv")


def test_empty_header_only_file_returns_empty_result(tmp_path: Path) -> None:
    from plateproof.ingestion.michelin import MICHELIN_SEED_REQUIRED_COLUMNS, load_michelin_seed

    header = ",".join(sorted(MICHELIN_SEED_REQUIRED_COLUMNS))
    path = tmp_path / "empty.csv"
    path.write_text(header + "\n", encoding="utf-8")
    result = load_michelin_seed(path)
    assert result.restaurants == []
    assert result.distinction_events == []
    assert result.report.input_row_count == 0


def test_valid_seed_builds_restaurants_and_events() -> None:
    from plateproof.ingestion.michelin import load_michelin_seed

    result = load_michelin_seed(FIXTURES / "michelin_seed_valid.csv")
    assert result.report.input_row_count == 5
    # Example Bistro (3 rows) collapses to one restaurant identity.
    assert result.report.restaurant_count == 3
    assert result.report.distinction_event_count == 5
    assert result.report.rejected_row_count == 0


def test_multi_year_distinction_history_preserved() -> None:
    from plateproof.ingestion.michelin import load_michelin_seed

    result = load_michelin_seed(FIXTURES / "michelin_seed_valid.csv")
    bistro = next(r for r in result.restaurants if r.normalized_name == "example bistro")
    events = [
        e
        for e in result.distinction_events
        if e.michelin_restaurant_id == bistro.michelin_restaurant_id
    ]
    years = sorted(e.guide_year for e in events)
    assert years == [2024, 2025, 2025]


def test_star_and_green_star_same_year_are_two_events_one_restaurant() -> None:
    from plateproof.ingestion.michelin import load_michelin_seed

    result = load_michelin_seed(FIXTURES / "michelin_seed_valid.csv")
    bistro = next(r for r in result.restaurants if r.normalized_name == "example bistro")
    events_2025 = [
        e
        for e in result.distinction_events
        if e.michelin_restaurant_id == bistro.michelin_restaurant_id and e.guide_year == 2025
    ]
    assert {e.distinction for e in events_2025} == {"two_stars", "green_star"}


def test_no_mutable_current_distinction_field_exists() -> None:
    from plateproof.ingestion.michelin import MichelinRestaurant

    assert "distinction" not in MichelinRestaurant.model_fields
    assert "current_distinction" not in MichelinRestaurant.model_fields


def test_jurisdiction_candidate_separates_records() -> None:
    from plateproof.ingestion.michelin import load_michelin_seed

    result = load_michelin_seed(FIXTURES / "michelin_seed_valid.csv")
    by_jurisdiction = {r.jurisdiction_candidate for r in result.restaurants}
    assert by_jurisdiction == {"nyc", "florida"}


def test_unsupported_distinction_rejected() -> None:
    from plateproof.ingestion.michelin import load_michelin_seed

    result = load_michelin_seed(FIXTURES / "michelin_seed_invalid_rows.csv")
    assert result.report.rejected_reasons.get("unsupported_distinction", 0) == 1


def test_invalid_provenance_confidence_rejected() -> None:
    from plateproof.ingestion.michelin import load_michelin_seed

    result = load_michelin_seed(FIXTURES / "michelin_seed_invalid_rows.csv")
    assert result.report.rejected_reasons.get("invalid_provenance_confidence", 0) == 1


def test_missing_source_url_rejected() -> None:
    from plateproof.ingestion.michelin import load_michelin_seed

    result = load_michelin_seed(FIXTURES / "michelin_seed_invalid_rows.csv")
    assert result.report.rejected_reasons.get("missing_provenance", 0) == 1


def test_malformed_guide_year_rejected() -> None:
    from plateproof.ingestion.michelin import load_michelin_seed

    result = load_michelin_seed(FIXTURES / "michelin_seed_invalid_rows.csv")
    assert result.report.rejected_reasons.get("malformed_guide_year", 0) == 1


def test_missing_jurisdiction_candidate_rejected() -> None:
    from plateproof.ingestion.michelin import load_michelin_seed

    result = load_michelin_seed(FIXTURES / "michelin_seed_invalid_rows.csv")
    assert result.report.rejected_reasons.get("missing_jurisdiction_candidate", 0) == 1
    assert result.report.rejected_row_count == 5


def test_conflicting_restaurant_identity_detected() -> None:
    from plateproof.ingestion.michelin import load_michelin_seed

    result = load_michelin_seed(FIXTURES / "michelin_seed_conflicts.csv")
    assert result.report.conflicting_restaurant_ids, "expected a conflicting restaurant id"


def test_conflicting_distinction_event_detected() -> None:
    from plateproof.ingestion.michelin import load_michelin_seed

    result = load_michelin_seed(FIXTURES / "michelin_seed_conflicts.csv")
    assert result.report.conflicting_distinction_event_ids, "expected a conflicting event id"


def test_identity_drift_reported_not_merged() -> None:
    from plateproof.ingestion.michelin import load_michelin_seed

    result = load_michelin_seed(FIXTURES / "michelin_seed_identity_drift.csv")
    assert "drift-diner" in result.report.possible_identity_drift_curator_refs
    # Not silently merged: two distinct computed identities remain.
    drift_restaurants = [r for r in result.restaurants if r.curator_restaurant_ref == "drift-diner"]
    assert len({r.michelin_restaurant_id for r in drift_restaurants}) == 2


def test_deterministic_ids_are_stable_across_runs() -> None:
    from plateproof.ingestion.michelin import load_michelin_seed

    a = load_michelin_seed(FIXTURES / "michelin_seed_valid.csv")
    b = load_michelin_seed(FIXTURES / "michelin_seed_valid.csv")
    assert sorted(r.michelin_restaurant_id for r in a.restaurants) == sorted(
        r.michelin_restaurant_id for r in b.restaurants
    )
    assert sorted(e.michelin_distinction_event_id for e in a.distinction_events) == sorted(
        e.michelin_distinction_event_id for e in b.distinction_events
    )


def test_events_reference_known_restaurant_ids() -> None:
    from plateproof.ingestion.michelin import (
        assert_events_reference_known_restaurants,
        load_michelin_seed,
    )

    result = load_michelin_seed(FIXTURES / "michelin_seed_valid.csv")
    assert (
        assert_events_reference_known_restaurants(result.restaurants, result.distinction_events)
        is None
    )


def test_referential_integrity_detects_dangling_event() -> None:
    from plateproof.ingestion.michelin import (
        MichelinDistinctionEvent,
        assert_events_reference_known_restaurants,
    )

    dangling = MichelinDistinctionEvent(
        michelin_distinction_event_id="michelin:e:doesnotmatter",
        michelin_restaurant_id="michelin:r:doesnotexist",
        distinction="one_star",
        guide_name="MICHELIN Guide New York City 2025",
        guide_year=2025,
        announced_date=None,
        source_url="https://example.invalid/fixture",
        source_title="Fixture",
        source_publisher="Fixture publisher",
        source_access_date=date(2026, 1, 1),
        source_license_note="fixture only",
        provenance_confidence="verified_secondary",
        source_row_ref=None,
    )
    with pytest.raises(ValueError, match="doesnotexist"):
        assert_events_reference_known_restaurants([], [dangling])


def test_distinction_events_as_of_excludes_future_announcement() -> None:
    from plateproof.ingestion.michelin import distinction_events_as_of, load_michelin_seed

    result = load_michelin_seed(FIXTURES / "michelin_seed_valid.csv")
    bistro = next(r for r in result.restaurants if r.normalized_name == "example bistro")
    as_of_2025_inspection = date(2025, 6, 1)
    known = distinction_events_as_of(
        result.distinction_events, bistro.michelin_restaurant_id, as_of_2025_inspection
    )
    years = {e.guide_year for e in known}
    # The 2024 one-star (announced 2023-11-01) is known by mid-2025.
    assert 2024 in years
    # The 2025 two-star/green-star were announced 2024-11-01 -- also known.
    assert 2025 in years


def test_distinction_events_as_of_excludes_unannounced_future_events() -> None:
    from plateproof.ingestion.michelin import distinction_events_as_of, load_michelin_seed

    result = load_michelin_seed(FIXTURES / "michelin_seed_valid.csv")
    bistro = next(r for r in result.restaurants if r.normalized_name == "example bistro")
    early_2024 = date(2024, 1, 1)
    known = distinction_events_as_of(
        result.distinction_events, bistro.michelin_restaurant_id, early_2024
    )
    # Only the 2024 one-star (announced 2023-11-01) predates this date.
    assert {e.guide_year for e in known} == {2024}


def test_attribution_and_license_note_preserved() -> None:
    from plateproof.ingestion.michelin import load_michelin_seed

    result = load_michelin_seed(FIXTURES / "michelin_seed_valid.csv")
    event = result.distinction_events[0]
    assert event.source_publisher == "Wikipedia contributors"
    assert "CC BY-SA" in event.source_license_note


def test_committed_template_matches_loader_contract_and_loads_empty() -> None:
    from plateproof.ingestion.michelin import (
        MICHELIN_SEED_OPTIONAL_COLUMNS,
        MICHELIN_SEED_REQUIRED_COLUMNS,
        load_michelin_seed,
    )

    template = (
        Path(__file__).parent.parent.parent / "data" / "reference" / "michelin_seed_template.csv"
    )
    header = template.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert set(header) == MICHELIN_SEED_REQUIRED_COLUMNS | MICHELIN_SEED_OPTIONAL_COLUMNS
    result = load_michelin_seed(template)
    assert result.restaurants == []
    assert result.distinction_events == []


def test_no_expressive_content_columns_exist() -> None:
    from plateproof.ingestion.michelin import MichelinDistinctionEvent, MichelinRestaurant

    forbidden = {"review_text", "description", "editorial_summary", "photo_url", "logo_url"}
    assert forbidden.isdisjoint(MichelinRestaurant.model_fields)
    assert forbidden.isdisjoint(MichelinDistinctionEvent.model_fields)
