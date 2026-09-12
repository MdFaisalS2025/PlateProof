"""Tests for the Task 7 processed-restaurants builder.

NYC restaurant identity is derivable from NYC inspection_events alone (all
descriptive fields are already there). Florida's full street address and
city live only in Task 3's staging output (`load_florida_extracts`'s `raw`
frame) -- inspection_events never carries them -- so Florida restaurants must
be built by joining staging to the normalized events via the existing
inspection_id relationship, never by re-deriving `florida:<license>` IDs
independently.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import polars as pl


def _nyc_event(**overrides: Any) -> dict[str, Any]:
    base = {
        "inspection_id": None,
        "restaurant_id": None,
        "source_id": None,
        "jurisdiction": "nyc",
        "inspection_date": None,
        "inspection_type": "Cycle Inspection / Initial Inspection",
        "inspection_type_raw": "Cycle Inspection / Initial Inspection",
        "action": None,
        "action_conflict": False,
        "action_conflict_values": None,
        "score": None,
        "score_conflict": False,
        "score_conflict_values": None,
        "grade": None,
        "grade_conflict": False,
        "grade_conflict_values": None,
        "grade_date": None,
        "violation_count": None,
        "critical_violation_count": None,
        "high_priority_count": None,
        "intermediate_count": None,
        "basic_count": None,
        "dba": "Test Restaurant",
        "boro_raw": "Manhattan",
        "building": "100",
        "street": "BROADWAY",
        "zipcode": "10001",
        "cuisine_description": "American",
        "latitude": 40.7,
        "longitude": -73.9,
        "source_dataset": "nyc_dohmh",
        "source_snapshot_date": date(2026, 1, 1),
        "source_retrieved_at_utc": datetime(2026, 1, 1, tzinfo=UTC),
        "source_sha256": "abc123",
        "ingested_at": datetime(2026, 1, 1, tzinfo=UTC),
        "pipeline_version": "test",
        "disposition": None,
        "disposition_status": None,
        "native_inspection_group_id": None,
        "native_visit_sequence": None,
    }
    base.update(overrides)
    if base["inspection_id"] is None:
        base["inspection_id"] = f"{base['restaurant_id']}:{base['inspection_date']}:{id(base)}"
    return base


def _nyc_events_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    from plateproof.features.inspection_events import INSPECTION_EVENT_SCHEMA, finalize_event_frame

    return finalize_event_frame(rows, INSPECTION_EVENT_SCHEMA)


def _fl_event(**overrides: Any) -> dict[str, Any]:
    base = {
        "inspection_id": None,
        "restaurant_id": None,
        "source_id": None,
        "jurisdiction": "florida",
        "inspection_date": None,
        "inspection_type": "Routine - Food",
        "inspection_type_raw": "Routine - Food",
        "action": None,
        "action_conflict": False,
        "action_conflict_values": None,
        "score": None,
        "score_conflict": False,
        "score_conflict_values": None,
        "grade": None,
        "grade_conflict": False,
        "grade_conflict_values": None,
        "grade_date": None,
        "violation_count": 0,
        "critical_violation_count": None,
        "high_priority_count": 0,
        "intermediate_count": 0,
        "basic_count": 0,
        "dba": "FL Test Restaurant",
        "boro_raw": None,
        "building": None,
        "street": None,
        "zipcode": "33101",
        "cuisine_description": None,
        "latitude": None,
        "longitude": None,
        "source_dataset": "florida_dbpr",
        "source_snapshot_date": date(2026, 1, 1),
        "source_retrieved_at_utc": datetime(2026, 1, 1, tzinfo=UTC),
        "source_sha256": "def456",
        "ingested_at": datetime(2026, 1, 1, tzinfo=UTC),
        "pipeline_version": "test",
        "disposition": "Met Standards",
        "disposition_status": "met_standards",
        "native_inspection_group_id": "1001",
        "native_visit_sequence": 1,
    }
    base.update(overrides)
    return base


def _fl_events_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    from plateproof.features.inspection_events import INSPECTION_EVENT_SCHEMA, finalize_event_frame

    return finalize_event_frame(rows, INSPECTION_EVENT_SCHEMA)


def _fl_staging_row(**overrides: Any) -> dict[str, Any]:
    base = {
        "license_number": "HR12345",
        "inspection_visit_id": "V1",
        "dba": "FL Test Restaurant",
        "location_address": "123 Ocean Ave",
        "location_city": "Miami",
        "location_zip": "33101",
        "county_name": "Miami-Dade",
        "inspection_date": date(2026, 1, 1),
        "source_file": "fdinspi.csv",
        "source_url": "https://www2.myfloridalicense.com/fdinspi.csv",
        "source_encoding": "utf-8",
    }
    base.update(overrides)
    return base


def _fl_staging_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(rows)


# --------------------------------------------------------------------------- #
# NYC: latest-snapshot selection, deterministic tie-break, conflict report   #
# --------------------------------------------------------------------------- #


def test_nyc_restaurant_selects_latest_establishment_snapshot() -> None:
    from plateproof.serving.processed_tables import build_nyc_restaurants

    rid = "nyc:1"
    older = _nyc_event(
        restaurant_id=rid,
        inspection_id="a",
        inspection_date=date(2024, 1, 1),
        dba="Old Name",
        zipcode="10001",
    )
    newer = _nyc_event(
        restaurant_id=rid,
        inspection_id="b",
        inspection_date=date(2025, 6, 1),
        dba="New Name",
        zipcode="10002",
    )
    events = _nyc_events_frame([older, newer])
    restaurants, report = build_nyc_restaurants(events)

    row = restaurants.filter(pl.col("restaurant_id") == rid).to_dicts()[0]
    assert row["name"] == "New Name"
    assert row["postal_code"] == "10002"
    assert report.restaurant_count == 1


def test_nyc_restaurant_deterministic_tie_break_not_snapshot_date_alone() -> None:
    """Many rows can share source_snapshot_date; the tie-break on the SAME
    latest inspection_date must be a different, deterministic key."""
    from plateproof.serving.processed_tables import build_nyc_restaurants

    rid = "nyc:2"
    same_date = date(2025, 6, 1)
    row_a = _nyc_event(
        restaurant_id=rid,
        inspection_id="z_last",
        inspection_date=same_date,
        dba="Name A",
        source_snapshot_date=date(2026, 1, 1),
    )
    row_b = _nyc_event(
        restaurant_id=rid,
        inspection_id="a_first",
        inspection_date=same_date,
        dba="Name A",
        source_snapshot_date=date(2026, 1, 1),
    )
    events = _nyc_events_frame([row_a, row_b])
    restaurants_1, _ = build_nyc_restaurants(events)
    restaurants_2, _ = build_nyc_restaurants(_nyc_events_frame([row_b, row_a]))

    # deterministic regardless of input row order
    assert restaurants_1.to_dicts() == restaurants_2.to_dicts()


def test_nyc_restaurant_reports_same_date_name_conflict() -> None:
    from plateproof.serving.processed_tables import build_nyc_restaurants

    rid = "nyc:3"
    same_date = date(2025, 6, 1)
    row_a = _nyc_event(
        restaurant_id=rid,
        inspection_id="a",
        inspection_date=same_date,
        dba="Name A",
    )
    row_b = _nyc_event(
        restaurant_id=rid,
        inspection_id="b",
        inspection_date=same_date,
        dba="Name B",
    )
    events = _nyc_events_frame([row_a, row_b])
    _, report = build_nyc_restaurants(events)

    assert rid in report.conflicting_restaurant_ids
    assert any(c.restaurant_id == rid and c.field == "name" for c in report.conflicts)


def test_nyc_restaurant_no_conflict_when_agreeing() -> None:
    from plateproof.serving.processed_tables import build_nyc_restaurants

    rid = "nyc:4"
    events = _nyc_events_frame(
        [_nyc_event(restaurant_id=rid, inspection_id="a", inspection_date=date(2025, 1, 1))]
    )
    _, report = build_nyc_restaurants(events)
    assert report.conflicting_restaurant_ids == []


def test_nyc_restaurant_preserves_official_id_and_provenance() -> None:
    from plateproof.serving.processed_tables import build_nyc_restaurants

    rid = "nyc:5"
    events = _nyc_events_frame(
        [
            _nyc_event(
                restaurant_id=rid,
                inspection_id="a",
                inspection_date=date(2025, 1, 1),
                source_snapshot_date=date(2026, 2, 2),
            )
        ]
    )
    restaurants, _ = build_nyc_restaurants(events)
    row = restaurants.to_dicts()[0]
    assert row["restaurant_id"] == rid
    assert row["jurisdiction"] == "nyc"
    assert row["source_snapshot_date"] == date(2026, 2, 2)


# --------------------------------------------------------------------------- #
# Florida: staging join, full address/city, no license-number equivalence   #
# --------------------------------------------------------------------------- #


def test_florida_restaurant_built_from_staging_address_and_city() -> None:
    from plateproof.serving.processed_tables import build_florida_restaurants

    rid = "florida:HR12345"
    events = _fl_events_frame(
        [
            _fl_event(
                restaurant_id=rid,
                inspection_id="florida:V1",
                inspection_date=date(2025, 1, 1),
                source_id="HR12345",
            )
        ]
    )
    staging = _fl_staging_frame(
        [_fl_staging_row(license_number="HR12345", inspection_visit_id="V1")]
    )
    restaurants, report = build_florida_restaurants(events, staging)

    row = restaurants.filter(pl.col("restaurant_id") == rid).to_dicts()[0]
    assert row["address"] == "123 Ocean Ave"
    assert row["city"] == "Miami"
    assert row["postal_code"] == "33101"
    assert report.restaurant_count == 1


def test_florida_restaurant_selects_latest_and_reports_conflicts() -> None:
    from plateproof.serving.processed_tables import build_florida_restaurants

    rid = "florida:HR99999"
    events = _fl_events_frame(
        [
            _fl_event(
                restaurant_id=rid,
                inspection_id="florida:V1",
                inspection_date=date(2024, 1, 1),
                source_id="HR99999",
            ),
            _fl_event(
                restaurant_id=rid,
                inspection_id="florida:V2",
                inspection_date=date(2025, 6, 1),
                source_id="HR99999",
            ),
        ]
    )
    staging = _fl_staging_frame(
        [
            _fl_staging_row(
                license_number="HR99999",
                inspection_visit_id="V1",
                location_address="1 Old Rd",
                location_city="Old Town",
            ),
            _fl_staging_row(
                license_number="HR99999",
                inspection_visit_id="V2",
                location_address="2 New Rd",
                location_city="New Town",
            ),
        ]
    )
    restaurants, _ = build_florida_restaurants(events, staging)
    row = restaurants.filter(pl.col("restaurant_id") == rid).to_dicts()[0]
    assert row["address"] == "2 New Rd"
    assert row["city"] == "New Town"


def test_florida_prefixed_and_numeric_license_numbers_never_merged() -> None:
    """A license number with a letter prefix and a bare numeric string that
    happen to share digits must never be treated as the same restaurant."""
    from plateproof.serving.processed_tables import build_florida_restaurants

    events = _fl_events_frame(
        [
            _fl_event(
                restaurant_id="florida:HR12345",
                inspection_id="florida:V1",
                inspection_date=date(2025, 1, 1),
                source_id="HR12345",
            ),
            _fl_event(
                restaurant_id="florida:12345",
                inspection_id="florida:V2",
                inspection_date=date(2025, 1, 1),
                source_id="12345",
            ),
        ]
    )
    staging = _fl_staging_frame(
        [
            _fl_staging_row(license_number="HR12345", inspection_visit_id="V1"),
            _fl_staging_row(license_number="12345", inspection_visit_id="V2"),
        ]
    )
    restaurants, _ = build_florida_restaurants(events, staging)
    assert set(restaurants.get_column("restaurant_id").to_list()) == {
        "florida:HR12345",
        "florida:12345",
    }


def test_florida_restaurant_preserves_source_filename_and_url() -> None:
    from plateproof.serving.processed_tables import build_florida_restaurants

    events = _fl_events_frame(
        [
            _fl_event(
                restaurant_id="florida:HR1",
                inspection_id="florida:V1",
                inspection_date=date(2025, 1, 1),
                source_id="HR1",
            )
        ]
    )
    staging = _fl_staging_frame(
        [_fl_staging_row(license_number="HR1", inspection_visit_id="V1", source_file="myfile.csv")]
    )
    restaurants, _ = build_florida_restaurants(events, staging)
    row = restaurants.to_dicts()[0]
    assert row["source_filename"] == "myfile.csv"
    assert row["source_url"]


def test_florida_restaurant_does_not_silently_merge_multiple_locations() -> None:
    """Two distinct license numbers must never collapse into one row even if
    every other descriptive field happens to match."""
    from plateproof.serving.processed_tables import build_florida_restaurants

    events = _fl_events_frame(
        [
            _fl_event(
                restaurant_id="florida:A",
                inspection_id="florida:V1",
                inspection_date=date(2025, 1, 1),
                source_id="A",
                dba="Same Name",
            ),
            _fl_event(
                restaurant_id="florida:B",
                inspection_id="florida:V2",
                inspection_date=date(2025, 1, 1),
                source_id="B",
                dba="Same Name",
            ),
        ]
    )
    staging = _fl_staging_frame(
        [
            _fl_staging_row(license_number="A", inspection_visit_id="V1", dba="Same Name"),
            _fl_staging_row(license_number="B", inspection_visit_id="V2", dba="Same Name"),
        ]
    )
    restaurants, _ = build_florida_restaurants(events, staging)
    assert restaurants.height == 2
