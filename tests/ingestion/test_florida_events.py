"""Tests for build_florida_inspection_events."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl
import pytest

FIXED_TS = datetime(2026, 2, 1, 9, 0, 0, tzinfo=UTC)


def _build(*paths: Path, **kwargs: Any) -> Any:
    from plateproof.ingestion.florida import (
        FloridaExtractSource,
        build_florida_inspection_events,
        load_florida_extracts,
    )

    raw = load_florida_extracts([FloridaExtractSource(path=p) for p in paths])
    return build_florida_inspection_events(raw, ingested_at=FIXED_TS, **kwargs)


def _event(result: Any, license_number: str, visit_id: str) -> dict[str, Any]:
    frame = result.inspection_events.filter(
        (pl.col("source_id") == license_number) & (pl.col("inspection_id") == f"florida:{visit_id}")
    )
    assert frame.height == 1, f"expected exactly one event for {license_number}/{visit_id}"
    return frame.to_dicts()[0]


# --- jurisdiction scoping --------------------------------------------------


def test_lodging_rows_are_excluded(fl_current_csv: Path) -> None:
    result = _build(fl_current_csv)
    assert "florida:2000001" not in result.inspection_events.get_column("restaurant_id").to_list()
    assert result.report.non_food_service_rows_removed >= 1


def test_only_dbpr_food_service_license_types_survive(fl_current_csv: Path) -> None:
    result = _build(fl_current_csv)
    assert set(result.inspection_events.get_column("jurisdiction").to_list()) == {"florida"}


# --- identifiers -----------------------------------------------------------


def test_license_number_preserved_exactly(fl_current_csv: Path) -> None:
    result = _build(fl_current_csv)
    event = _event(result, "1000001", "9000001")
    assert event["source_id"] == "1000001"
    assert event["restaurant_id"] == "florida:1000001"


def test_inspection_id_uses_native_visit_id(fl_current_csv: Path) -> None:
    result = _build(fl_current_csv)
    event = _event(result, "1000001", "9000001")
    assert event["inspection_id"] == "florida:9000001"
    assert event["native_inspection_group_id"] == "9100001"
    assert event["native_visit_sequence"] == 1


def test_missing_license_number_row_rejected(fl_current_csv: Path) -> None:
    result = _build(fl_current_csv)
    assert result.report.missing_license_number_rows_removed == 1
    assert "9000005" not in result.inspection_events.get_column("inspection_id").to_list()


def test_missing_inspection_visit_id_row_rejected(fl_current_csv: Path) -> None:
    result = _build(fl_current_csv)
    assert result.report.missing_inspection_visit_id_rows_removed == 1


def test_unparseable_date_row_rejected(fl_current_csv: Path) -> None:
    result = _build(fl_current_csv)
    assert result.report.unparseable_date_rows_removed == 1
    assert "9000006" not in result.inspection_events.get_column("inspection_id").to_list()


def test_assert_unique_inspection_id_holds(fl_current_csv: Path) -> None:
    from plateproof.features.inspection_events import assert_unique_inspection_id

    result = _build(fl_current_csv)
    assert assert_unique_inspection_id(result.inspection_events) is None


# --- authoritative violation counts (correction #1) ------------------------


def test_violation_count_is_authoritative_total_not_category_count(fl_current_csv: Path) -> None:
    result = _build(fl_current_csv)
    event = _event(result, "1000001", "9000001")
    # 3 nonzero categories cited (04, 08, 42) -> total_violations column says "3" too,
    # so this alone wouldn't distinguish the bug; the mismatch fixtures below do.
    assert event["violation_count"] == 3
    assert event["high_priority_count"] == 1
    assert event["intermediate_count"] == 1
    assert event["basic_count"] == 1


def test_violation_count_uses_source_total_even_when_it_disagrees_with_categories(
    fl_current_csv: Path,
) -> None:
    result = _build(fl_current_csv)
    # Row H: total=9, but only 3 categories are cited and H+I+B sums to 3.
    event = _event(result, "1000004", "9000007")
    assert event["violation_count"] == 9
    assert event["high_priority_count"] == 1
    assert event["intermediate_count"] == 1
    assert event["basic_count"] == 1


def test_critical_violation_count_is_null_not_zero_for_florida(fl_current_csv: Path) -> None:
    result = _build(fl_current_csv)
    events = result.inspection_events
    assert events.get_column("critical_violation_count").null_count() == events.height


def test_florida_never_populates_nyc_only_fields(fl_current_csv: Path) -> None:
    result = _build(fl_current_csv)
    events = result.inspection_events
    for column in ("score", "grade", "grade_date", "action", "boro_raw", "building", "street"):
        assert events.get_column(column).null_count() == events.height


# --- discrepancy reporting (correction #2) ---------------------------------


def test_total_vs_hib_discrepancy_is_reported_not_corrected(fl_current_csv: Path) -> None:
    result = _build(fl_current_csv)
    assert result.report.total_vs_hib_mismatch_count == 1
    event = _event(result, "1000004", "9000007")
    assert event["violation_count"] == 9  # unmodified, not forced to equal H+I+B


def test_total_vs_category_sum_discrepancy_is_reported(fl_current_csv: Path) -> None:
    result = _build(fl_current_csv)
    assert result.report.total_vs_category_sum_mismatch_count == 1
    event = _event(result, "1000005", "9000008")
    assert event["violation_count"] == 5  # unmodified


def test_malformed_total_violations_becomes_null_not_zero(fl_current_csv: Path) -> None:
    result = _build(fl_current_csv)
    event = _event(result, "1000006", "9000009")
    assert event["violation_count"] is None
    assert result.report.malformed_count_field_rows >= 1


def test_negative_count_becomes_null_not_zero(fl_current_csv: Path) -> None:
    result = _build(fl_current_csv)
    event = _event(result, "1000007", "9000010")
    assert event["high_priority_count"] is None


# --- disposition (D6) -------------------------------------------------------


def test_disposition_preserved_verbatim_and_status_met_standards(fl_current_csv: Path) -> None:
    result = _build(fl_current_csv)
    event = _event(result, "1000001", "9000003")
    assert event["disposition"] == "Emergency Order Callback Complied"
    assert event["disposition_status"] == "met_standards"


def test_disposition_status_follow_up_required(fl_current_csv: Path) -> None:
    result = _build(fl_current_csv)
    event = _event(result, "1000001", "9000001")
    assert event["disposition"] == "Warning Issued"
    assert event["disposition_status"] == "follow_up_required"


def test_disposition_status_temporary_closure(fl_current_csv: Path) -> None:
    result = _build(fl_current_csv)
    event = _event(result, "1000001", "9000002")
    assert event["disposition"] == "Emergency order recommended"
    assert event["disposition_status"] == "temporary_closure"


def test_unknown_disposition_maps_to_unknown_not_met_standards(fl_current_csv: Path) -> None:
    result = _build(fl_current_csv)
    event = _event(result, "1000008", "9000011")
    assert event["disposition"] == "Something New DBPR Added"
    assert event["disposition_status"] == "unknown"
    assert result.report.unknown_disposition_count == 1


# --- violation events / classification (correction #8) --------------------


def test_violation_rows_use_real_dbpr_descriptions(fl_current_csv: Path) -> None:
    result = _build(fl_current_csv)
    event = _event(result, "1000001", "9000001")
    rows = result.violation_events.filter(pl.col("inspection_id") == event["inspection_id"])
    codes = {r["violation_code"]: r["violation_description"] for r in rows.to_dicts()}
    assert codes["04"] == "Facilities to maintain product temperature"
    assert codes["08"] == "Food protection, cross-contamination"


def test_violation_severity_is_other_with_null_raw_classification(fl_current_csv: Path) -> None:
    result = _build(fl_current_csv)
    violations = result.violation_events
    assert set(violations.get_column("severity").to_list()) == {"other"}
    assert violations.get_column("critical_flag_raw").null_count() == violations.height


def test_map_florida_violation_classification_all_cases() -> None:
    from plateproof.ingestion.florida import map_florida_violation_classification

    assert map_florida_violation_classification("High Priority") == "high_priority"
    assert map_florida_violation_classification("Intermediate") == "intermediate"
    assert map_florida_violation_classification("Basic") == "basic"
    assert map_florida_violation_classification("Critical") == "other"
    assert map_florida_violation_classification(None) == "other"
    assert map_florida_violation_classification("") == "other"


def test_corrected_on_site_is_null_for_florida(fl_current_csv: Path) -> None:
    result = _build(fl_current_csv)
    violations = result.violation_events
    assert violations.get_column("corrected_on_site").null_count() == violations.height


def test_violation_event_id_stable_when_another_category_added(
    fl_current_csv: Path, tmp_path: Path
) -> None:
    """Florida is one row per visit, so "adding a violation category" means
    flipping an existing row's V_NN cell from 0 to nonzero -- not appending a
    row (which would instead create a conflicting duplicate visit id). The ids
    for categories already cited must not change when another category on the
    *same* visit becomes newly nonzero."""
    from plateproof.ingestion.florida import (
        FloridaExtractSource,
        build_florida_inspection_events,
        load_florida_extracts,
    )

    base = _build(fl_current_csv)
    inspection_id = "florida:9000001"
    base_ids = {
        r["violation_code"]: r["violation_event_id"]
        for r in base.violation_events.filter(pl.col("inspection_id") == inspection_id).to_dicts()
    }
    assert base_ids  # sanity: the fixture row does cite some categories

    import csv
    import io

    lines = fl_current_csv.read_text(encoding="utf-8").splitlines()
    rows = list(csv.reader(io.StringIO("\n".join(lines))))
    header = rows[0]
    v43_col = header.index("Violation 43")
    visit_col = header.index("Inspection Visit ID")
    for row in rows[1:]:
        if row[visit_col] == "9000001":
            row[v43_col] = "1"  # was "0" in the fixture

    augmented = tmp_path / "augmented.csv"
    with augmented.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)

    raw = load_florida_extracts([FloridaExtractSource(path=augmented)])
    after = build_florida_inspection_events(raw, ingested_at=FIXED_TS)
    after_ids = {
        r["violation_code"]: r["violation_event_id"]
        for r in after.violation_events.filter(pl.col("inspection_id") == inspection_id).to_dicts()
    }
    for code, vid in base_ids.items():
        assert after_ids[code] == vid
    assert "43" in after_ids and "43" not in base_ids


def test_violation_event_id_independent_of_input_row_order(
    fl_current_csv: Path, tmp_path: Path
) -> None:
    from plateproof.ingestion.florida import (
        FloridaExtractSource,
        build_florida_inspection_events,
        load_florida_extracts,
    )

    lines = fl_current_csv.read_text(encoding="utf-8").splitlines()
    header, body = lines[0], lines[1:]
    shuffled = tmp_path / "shuffled.csv"
    shuffled.write_text("\n".join([header, *reversed(body)]) + "\n", encoding="utf-8")

    a = build_florida_inspection_events(
        load_florida_extracts([FloridaExtractSource(path=fl_current_csv)]), ingested_at=FIXED_TS
    )
    b = build_florida_inspection_events(
        load_florida_extracts([FloridaExtractSource(path=shuffled)]), ingested_at=FIXED_TS
    )
    assert sorted(a.violation_events.get_column("violation_event_id").to_list()) == sorted(
        b.violation_events.get_column("violation_event_id").to_list()
    )


# --- duplicate / conflicting Inspection Visit ID (correction #3) ----------


def test_identical_duplicate_rows_collapse_deterministically(
    fl_current_csv: Path, fl_overlap_csv: Path
) -> None:
    result = _build(fl_current_csv, fl_overlap_csv)
    matches = result.inspection_events.filter(pl.col("inspection_id") == "florida:9000012")
    assert matches.height == 1
    assert result.report.identical_duplicate_visit_rows_removed == 2  # 2 in current + 1 overlap


def test_conflicting_duplicate_rows_are_dropped_and_reported_by_default(
    fl_current_csv: Path,
) -> None:
    result = _build(fl_current_csv)
    assert result.inspection_events.filter(pl.col("inspection_id") == "florida:9000013").height == 0
    assert result.report.conflicting_visit_id_rows_removed == 2
    assert "9000013" in result.report.conflicting_visit_ids


def test_conflicting_duplicate_rows_can_raise_when_requested(fl_current_csv: Path) -> None:
    with pytest.raises(ValueError, match="9000013"):
        _build(fl_current_csv, on_conflicting_duplicate="raise")


# --- determinism & provenance ------------------------------------------


def test_ingested_at_injected_and_reused(fl_current_csv: Path) -> None:
    result = _build(fl_current_csv)
    events = result.inspection_events
    assert set(events.get_column("ingested_at").to_list()) == {FIXED_TS}
    assert result.report.ingestion_timestamp == FIXED_TS


def test_build_deterministic_for_same_inputs_and_timestamp(fl_current_csv: Path) -> None:
    a = _build(fl_current_csv)
    b = _build(fl_current_csv)
    assert a.inspection_events.equals(b.inspection_events)
    assert a.violation_events.equals(b.violation_events)
    assert a.report == b.report


def test_snapshot_date_from_source_metadata_is_documented_as_retrieval_date(
    fl_current_csv: Path,
) -> None:
    from plateproof.ingestion.florida import (
        FloridaExtractSource,
        FloridaFileMetadata,
        FloridaSnapshotMetadata,
        build_florida_inspection_events,
        load_florida_extracts,
    )

    meta = FloridaSnapshotMetadata(
        retrieved_at_utc=datetime(2026, 2, 1, 5, 0, tzinfo=UTC),
        requested_fiscal_years=["current"],
        requested_districts=[1],
        files=[
            FloridaFileMetadata(
                url="https://example/fl.csv",
                fiscal_year="current",
                district=1,
                format="csv",
                local_filename="fl_sample_current.csv",
                retrieved_at_utc=datetime(2026, 2, 1, 5, 0, tzinfo=UTC),
                sha256="0" * 64,
                byte_size=123,
                row_count=16,
                encoding="utf-8",
            )
        ],
        downloader_version="0.1.0",
    )
    raw = load_florida_extracts([FloridaExtractSource(path=fl_current_csv)])
    result = build_florida_inspection_events(raw, ingested_at=FIXED_TS, source_metadata=meta)
    events = result.inspection_events
    assert set(events.get_column("source_snapshot_date").to_list()) == {date(2026, 2, 1)}
    assert set(events.get_column("source_retrieved_at_utc").to_list()) == {meta.retrieved_at_utc}
    assert set(events.get_column("source_sha256").to_list()) == {"0" * 64}


def test_build_without_source_metadata_nulls_download_provenance(fl_current_csv: Path) -> None:
    result = _build(fl_current_csv)
    events = result.inspection_events
    assert events.get_column("source_snapshot_date").null_count() == events.height
    assert events.get_column("source_retrieved_at_utc").null_count() == events.height
    assert events.get_column("source_sha256").null_count() == events.height


# --- boundary & cross-jurisdiction compatibility (#19, #20) ----------------


def test_empty_input_produces_empty_output_and_zero_counts() -> None:
    from plateproof.features.inspection_events import (
        INSPECTION_EVENT_SCHEMA,
        VIOLATION_EVENT_SCHEMA,
    )
    from plateproof.ingestion.florida import build_florida_inspection_events

    empty = pl.DataFrame(schema={col: pl.String() for col in _florida_staging_columns()})
    result = build_florida_inspection_events(empty, ingested_at=FIXED_TS)
    assert result.inspection_events.height == 0
    assert result.violation_events.height == 0
    assert result.inspection_events.columns == list(INSPECTION_EVENT_SCHEMA)
    assert result.violation_events.columns == list(VIOLATION_EVENT_SCHEMA)
    assert result.report.input_row_count == 0
    assert result.report.output_inspection_count == 0


def test_all_invalid_input_produces_empty_output_with_removal_counts(tmp_path: Path) -> None:
    header = [
        " License Type Code",
        " License Number",
        "Inspection Number",
        "Visit Number",
        "Inspection Class",
        "Inspection Type",
        "Inspection Disposition",
        "Inspection Date",
        " Number of Total Violations",
        "Number of High Priority Violations",
        "Number of Intermediate Violations",
        "Number of Basic Violations",
        *[f"Violation {i:02d}" for i in range(1, 59)],
        "Inspection Visit ID",
    ]
    bad_row = [
        "2010",
        "",
        "9999999",
        "1",
        "Food",
        "Routine - Food",
        "Warning Issued",
        "not-a-date",
        "0",
        "0",
        "0",
        "0",
        *["0"] * 58,
        "",
    ]
    path = tmp_path / "all_bad.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        import csv

        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerow(bad_row)
    result = _build(path)
    assert result.inspection_events.height == 0
    assert result.report.missing_license_number_rows_removed == 1


def test_nyc_and_florida_event_tables_concat_without_schema_error(fl_current_csv: Path) -> None:
    from plateproof.ingestion.nyc import build_nyc_inspection_events, load_nyc_raw

    florida_result = _build(fl_current_csv)
    nyc_raw = load_nyc_raw(Path(__file__).parent / "fixtures" / "nyc_sample_soda.csv")
    nyc_result = build_nyc_inspection_events(nyc_raw, ingested_at=FIXED_TS)
    combined_events = pl.concat([nyc_result.inspection_events, florida_result.inspection_events])
    combined_violations = pl.concat([nyc_result.violation_events, florida_result.violation_events])
    assert combined_events.height == (
        nyc_result.inspection_events.height + florida_result.inspection_events.height
    )
    assert combined_violations.height == (
        nyc_result.violation_events.height + florida_result.violation_events.height
    )


def test_outputs_are_descriptive_only(fl_current_csv: Path) -> None:
    result = _build(fl_current_csv)
    leak_markers = ("prev_", "rolling_", "days_since", "_shift", "target", "label", "next_")
    for column in (*result.inspection_events.columns, *result.violation_events.columns):
        assert not any(marker in column for marker in leak_markers)


def _florida_staging_columns() -> list[str]:
    from plateproof.ingestion.florida import FL_OPTIONAL_COLUMNS, FL_REQUIRED_COLUMNS

    return sorted(
        {*FL_REQUIRED_COLUMNS, *FL_OPTIONAL_COLUMNS, "source_file", "source_url", "source_encoding"}
    )
