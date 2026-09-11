"""Tests for the typed NYCIngestionReport."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

FIXED_TS = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)


def _report(csv_path: Path) -> Any:
    from plateproof.ingestion.nyc import build_nyc_inspection_events, load_nyc_raw

    return build_nyc_inspection_events(load_nyc_raw(csv_path), ingested_at=FIXED_TS).report


def test_report_counts_match_golden(soda_csv: Path, golden: dict[str, Any]) -> None:
    report = _report(soda_csv)
    expected = golden["report"]
    assert report.input_row_count == expected["input_row_count"]
    assert report.output_inspection_count == expected["output_inspection_count"]
    assert report.output_violation_count == expected["output_violation_count"]
    assert report.placeholder_date_rows_removed == expected["placeholder_date_rows_removed"]
    assert report.invalid_camis_rows_removed == expected["invalid_camis_rows_removed"]
    assert report.unparseable_date_rows_removed == expected["unparseable_date_rows_removed"]
    assert report.duplicate_violation_rows_removed == expected["duplicate_violation_rows_removed"]
    assert report.score_conflict_group_count == expected["score_conflict_group_count"]
    assert report.action_conflict_group_count == expected["action_conflict_group_count"]
    assert report.grade_conflict_group_count == expected["grade_conflict_group_count"]
    assert report.snapshot_date == expected["snapshot_date"]


def test_report_ingestion_timestamp_is_injected_value(soda_csv: Path) -> None:
    assert _report(soda_csv).ingestion_timestamp == FIXED_TS


def test_report_is_frozen(soda_csv: Path) -> None:
    import pytest

    report = _report(soda_csv)
    with pytest.raises((TypeError, ValueError, AttributeError)):
        report.input_row_count = 999  # type: ignore[misc]


def test_report_snapshot_date_independent_of_row_validity(soda_csv: Path) -> None:
    assert _report(soda_csv).snapshot_date == date(2024, 9, 15)
