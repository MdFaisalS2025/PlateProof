"""Tests for the typed FloridaIngestionReport."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

FIXED_TS = datetime(2026, 2, 1, 9, 0, 0, tzinfo=UTC)


def _report(*paths: Path) -> Any:
    from plateproof.ingestion.florida import (
        FloridaExtractSource,
        build_florida_inspection_events,
        load_florida_extracts,
    )

    raw = load_florida_extracts([FloridaExtractSource(path=p) for p in paths])
    return build_florida_inspection_events(raw, ingested_at=FIXED_TS).report


def test_report_counts_match_fixture(fl_current_csv: Path) -> None:
    report = _report(fl_current_csv)
    assert report.input_row_count == 16
    assert report.output_inspection_count == 9
    assert report.output_violation_count == 14
    assert report.files_loaded_count == 1
    assert report.non_food_service_rows_removed == 1
    assert report.missing_license_number_rows_removed == 1
    assert report.missing_inspection_visit_id_rows_removed == 1
    assert report.unparseable_date_rows_removed == 1
    assert report.identical_duplicate_visit_rows_removed == 1
    assert report.conflicting_visit_id_rows_removed == 2
    assert report.conflicting_visit_ids == ["9000013"]
    assert report.total_vs_hib_mismatch_count == 1
    assert report.total_vs_category_sum_mismatch_count == 1
    assert report.malformed_count_field_rows == 2
    assert report.unknown_disposition_count == 1


def test_report_is_frozen(fl_current_csv: Path) -> None:
    import pytest

    report = _report(fl_current_csv)
    with pytest.raises((TypeError, ValueError, AttributeError)):
        report.input_row_count = 999  # type: ignore[misc]


def test_report_ingestion_timestamp_is_injected_value(fl_current_csv: Path) -> None:
    assert _report(fl_current_csv).ingestion_timestamp == FIXED_TS


def test_report_records_source_encodings(fl_current_csv: Path) -> None:
    report = _report(fl_current_csv)
    assert report.source_encodings == {"fl_sample_current.csv": "utf-8"}
