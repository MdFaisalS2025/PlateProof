"""Tests for build_nyc_inspection_events: removal, aggregation, conflicts, provenance."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl

FIXED_TS = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)


def _build(csv_path: Path, **kwargs: Any) -> Any:
    from plateproof.ingestion.nyc import build_nyc_inspection_events, load_nyc_raw

    raw = load_nyc_raw(csv_path)
    return build_nyc_inspection_events(raw, ingested_at=FIXED_TS, **kwargs)


def _event(result: Any, restaurant_id: str, inspection_date: str) -> dict[str, Any]:
    frame = result.inspection_events.filter(
        (pl.col("restaurant_id") == restaurant_id)
        & (pl.col("inspection_date") == date.fromisoformat(inspection_date))
    )
    assert frame.height == 1
    return frame.to_dicts()[0]


# --- removal & counting -------------------------------------------------------


def test_placeholder_date_rows_removed_and_counted(soda_csv: Path) -> None:
    result = _build(soda_csv)
    assert result.report.placeholder_date_rows_removed == 1
    dates = result.inspection_events.get_column("inspection_date").to_list()
    assert date(1900, 1, 1) not in dates


def test_invalid_camis_rows_removed_and_counted(soda_csv: Path) -> None:
    result = _build(soda_csv)
    assert result.report.invalid_camis_rows_removed == 2
    ids = result.inspection_events.get_column("restaurant_id").to_list()
    assert "nyc:" not in ids
    assert "nyc:50X99" not in ids


def test_unparseable_date_rows_removed_and_counted(soda_csv: Path) -> None:
    result = _build(soda_csv)
    assert result.report.unparseable_date_rows_removed == 1


def test_snapshot_date_uses_all_parseable_record_dates(soda_csv: Path) -> None:
    # The 2024-09-15 record_date belongs to a blank-CAMIS row that is dropped;
    # provenance must still reflect it.
    result = _build(soda_csv)
    assert result.report.snapshot_date == date(2024, 9, 15)
    assert set(result.inspection_events.get_column("source_snapshot_date").to_list()) == {
        date(2024, 9, 15)
    }


# --- aggregation ------------------------------------------------------------


def test_multiple_violation_rows_collapse_to_one_event(soda_csv: Path) -> None:
    result = _build(soda_csv)
    event = _event(result, "nyc:50190663", "2024-05-10")
    assert event["violation_count"] == 2  # 3 rows, one exact duplicate
    assert event["critical_violation_count"] == 1


def test_published_score_preserved_not_summed(soda_csv: Path) -> None:
    result = _build(soda_csv)
    event = _event(result, "nyc:50190663", "2024-05-10")
    assert event["score"] == 13.0
    assert event["score_conflict"] is False


def test_violationless_inspection_emits_zero_violation_rows(soda_csv: Path) -> None:
    result = _build(soda_csv)
    event = _event(result, "nyc:50190663", "2024-06-15")
    assert event["violation_count"] == 0
    assert event["score"] == 0.0
    joined = result.violation_events.filter(pl.col("inspection_id") == event["inspection_id"])
    assert joined.height == 0


def test_florida_only_counts_are_null(soda_csv: Path) -> None:
    result = _build(soda_csv)
    for column in ("high_priority_count", "intermediate_count", "basic_count"):
        assert (
            result.inspection_events.get_column(column).null_count()
            == result.inspection_events.height
        )


def test_inspection_event_key_is_unique(soda_csv: Path) -> None:
    from plateproof.features.inspection_events import assert_unique_inspection_key

    result = _build(soda_csv)
    assert assert_unique_inspection_key(result.inspection_events) is None


def test_inspection_type_is_whitespace_normalized(tmp_path: Path, write_csv: Any) -> None:
    header = [
        "camis",
        "inspection_date",
        "inspection_type",
        "action",
        "violation_code",
        "violation_description",
        "critical_flag",
        "score",
        "grade",
        "grade_date",
        "record_date",
    ]
    row = [
        "50190663",
        "2024-05-10T00:00:00.000",
        "Cycle Inspection /  Initial   Inspection",
        "Violations were cited in the following area(s).",
        "04L",
        "Evidence of mice",
        "Critical",
        "13",
        "A",
        "2024-05-10T00:00:00.000",
        "2024-07-01T00:00:00.000",
    ]
    result = _build(write_csv(tmp_path / "ws.csv", header, [row]))
    assert result.inspection_events.get_column("inspection_type").to_list() == [
        "Cycle Inspection / Initial Inspection"
    ]


# --- violation identity & severity ----------------------------------------


def test_violations_deduplicated_within_inspection(soda_csv: Path) -> None:
    result = _build(soda_csv)
    # R1-A rows 2/3 are identical; R2-B rows 7/8 share code + description.
    assert result.report.duplicate_violation_rows_removed == 2
    assert result.report.output_violation_count == 7


def test_violation_event_id_is_stable_when_extra_violation_added(
    soda_csv: Path, tmp_path: Path
) -> None:
    from plateproof.ingestion.nyc import build_nyc_inspection_events, load_nyc_raw

    base = build_nyc_inspection_events(load_nyc_raw(soda_csv), ingested_at=FIXED_TS)
    target = base.violation_events.filter(pl.col("violation_code") == "04L")
    original_id = target.get_column("violation_event_id").to_list()[0]

    lines = soda_csv.read_text(encoding="utf-8").splitlines()
    extra = lines[1].replace(
        ",04L,Evidence of mice or live mice present,", ",99Z,Newly inserted violation,"
    )
    augmented = tmp_path / "augmented.csv"
    augmented.write_text("\n".join([*lines, extra]) + "\n", encoding="utf-8")

    after = build_nyc_inspection_events(load_nyc_raw(augmented), ingested_at=FIXED_TS)
    after_id = (
        after.violation_events.filter(pl.col("violation_code") == "04L")
        .get_column("violation_event_id")
        .to_list()[0]
    )
    assert after_id == original_id


def test_violation_event_id_is_row_order_independent(soda_csv: Path, tmp_path: Path) -> None:
    from plateproof.ingestion.nyc import build_nyc_inspection_events, load_nyc_raw

    lines = soda_csv.read_text(encoding="utf-8").splitlines()
    header, body = lines[0], lines[1:]
    shuffled = tmp_path / "shuffled.csv"
    shuffled.write_text("\n".join([header, *reversed(body)]) + "\n", encoding="utf-8")

    a = build_nyc_inspection_events(load_nyc_raw(soda_csv), ingested_at=FIXED_TS)
    b = build_nyc_inspection_events(load_nyc_raw(shuffled), ingested_at=FIXED_TS)
    assert sorted(a.violation_events.get_column("violation_event_id").to_list()) == sorted(
        b.violation_events.get_column("violation_event_id").to_list()
    )


def test_severity_mapping_all_four_cases(soda_csv: Path) -> None:
    result = _build(soda_csv)
    rows = {
        (r["violation_code"]): (r["severity"], r["critical_flag_raw"])
        for r in result.violation_events.to_dicts()
    }
    assert rows["04L"] == ("critical", "Critical")
    assert rows["10F"] == ("noncritical", "Not Critical")
    assert rows["16A"] == ("not_applicable", "Not Applicable")
    assert rows["16B"] == ("unknown", None)


def test_corrected_on_site_is_null_for_nyc(soda_csv: Path) -> None:
    result = _build(soda_csv)
    assert result.violation_events.get_column("corrected_on_site").null_count() == (
        result.violation_events.height
    )


# --- conflict handling ---------------------------------------------------


def test_score_conflict_tied_latest_record_date_yields_null(soda_csv: Path) -> None:
    result = _build(soda_csv)
    event = _event(result, "nyc:50180187", "2024-03-02")
    assert event["score"] is None
    assert event["score_conflict"] is True
    assert event["score_conflict_values"] == [12.0, 27.0]


def test_score_conflict_unique_latest_record_date_wins(soda_csv: Path) -> None:
    result = _build(soda_csv)
    event = _event(result, "nyc:50180187", "2024-04-20")
    assert event["score"] == 9.0
    assert event["score_conflict"] is True
    assert event["score_conflict_values"] == [9.0, 20.0]


def test_action_conflict_follows_same_rule(soda_csv: Path) -> None:
    result = _build(soda_csv)
    event = _event(result, "nyc:50180187", "2024-04-20")
    assert event["action"] == "Violations were cited in the following area(s)."
    assert event["action_conflict"] is True
    assert event["action_conflict_values"] == [
        "Establishment Closed by DOHMH",
        "Violations were cited in the following area(s).",
    ]


def test_grade_conflict_and_grade_date_follow_selected_grade(soda_csv: Path) -> None:
    result = _build(soda_csv)
    event = _event(result, "nyc:50180187", "2024-04-20")
    assert event["grade"] == "A"
    assert event["grade_conflict"] is True
    assert event["grade_conflict_values"] == ["A", "B"]
    assert event["grade_date"] == date(2024, 4, 21)


def test_grade_never_coerced_and_missing_preserved_as_null(soda_csv: Path) -> None:
    result = _build(soda_csv)
    event = _event(result, "nyc:50180187", "2024-03-02")
    assert event["grade"] is None
    assert event["grade_date"] is None
    grades = set(result.inspection_events.get_column("grade").drop_nulls().to_list())
    assert grades <= {"A", "B", "C", "N", "Z", "P"}


# --- provenance & determinism ------------------------------------------


def test_injected_ingested_at_is_used_verbatim(soda_csv: Path) -> None:
    result = _build(soda_csv)
    assert set(result.inspection_events.get_column("ingested_at").to_list()) == {FIXED_TS}
    assert set(result.violation_events.get_column("ingested_at").to_list()) == {FIXED_TS}
    assert result.report.ingestion_timestamp == FIXED_TS


def test_build_is_deterministic_for_same_inputs(soda_csv: Path, metadata_json: Path) -> None:
    from plateproof.ingestion.nyc import (
        build_nyc_inspection_events,
        load_nyc_raw,
        load_source_metadata,
    )

    meta = load_source_metadata(metadata_json)
    raw = load_nyc_raw(soda_csv)
    a = build_nyc_inspection_events(raw, ingested_at=FIXED_TS, source_metadata=meta)
    b = build_nyc_inspection_events(raw, ingested_at=FIXED_TS, source_metadata=meta)
    assert a.inspection_events.equals(b.inspection_events)
    assert a.violation_events.equals(b.violation_events)
    assert a.report == b.report


def test_provenance_fields_are_distinct(soda_csv: Path, metadata_json: Path) -> None:
    from plateproof.ingestion.nyc import load_source_metadata

    meta = load_source_metadata(metadata_json)
    result = _build(soda_csv, source_metadata=meta)
    events = result.inspection_events
    assert set(events.get_column("source_snapshot_date").to_list()) == {date(2024, 9, 15)}
    assert set(events.get_column("source_retrieved_at_utc").to_list()) == {meta.retrieved_at_utc}
    assert set(events.get_column("source_sha256").to_list()) == {meta.source_sha256}
    assert set(events.get_column("ingested_at").to_list()) == {FIXED_TS}


def test_build_without_source_metadata_nulls_download_provenance(soda_csv: Path) -> None:
    result = _build(soda_csv)
    events = result.inspection_events
    assert events.get_column("source_retrieved_at_utc").null_count() == events.height
    assert events.get_column("source_sha256").null_count() == events.height
    assert set(events.get_column("source_snapshot_date").to_list()) == {date(2024, 9, 15)}


def test_outputs_are_descriptive_only(soda_csv: Path) -> None:
    from plateproof.features.inspection_events import (
        INSPECTION_EVENT_SCHEMA,
        VIOLATION_EVENT_SCHEMA,
    )

    result = _build(soda_csv)
    assert result.inspection_events.columns == list(INSPECTION_EVENT_SCHEMA)
    assert result.violation_events.columns == list(VIOLATION_EVENT_SCHEMA)
    leak_markers = ("prev_", "rolling_", "days_since", "_shift", "target", "label", "next_")
    for column in (*result.inspection_events.columns, *result.violation_events.columns):
        assert not any(marker in column for marker in leak_markers)


def test_required_only_extract_produces_events_and_violations(
    tmp_path: Path, write_csv: Any
) -> None:
    header = [
        "camis",
        "inspection_date",
        "inspection_type",
        "action",
        "violation_code",
        "violation_description",
        "critical_flag",
        "score",
        "grade",
        "grade_date",
        "record_date",
    ]
    rows = [
        [
            "50190663",
            "2024-05-10T00:00:00.000",
            "Cycle Inspection / Initial Inspection",
            "Violations were cited in the following area(s).",
            "04L",
            "Evidence of mice",
            "Critical",
            "13",
            "A",
            "2024-05-10T00:00:00.000",
            "2024-07-01T00:00:00.000",
        ],
        [
            "50190663",
            "2024-05-10T00:00:00.000",
            "Cycle Inspection / Initial Inspection",
            "Violations were cited in the following area(s).",
            "08A",
            "Facility not vermin proof",
            "Not Critical",
            "13",
            "A",
            "2024-05-10T00:00:00.000",
            "2024-07-01T00:00:00.000",
        ],
    ]
    result = _build(write_csv(tmp_path / "req_only.csv", header, rows))
    assert result.inspection_events.height == 1
    assert result.violation_events.height == 2
    assert result.inspection_events.get_column("dba").null_count() == 1
    assert result.inspection_events.get_column("score").to_list() == [13.0]


def test_outputs_match_golden_file(soda_csv: Path, golden: dict[str, Any]) -> None:
    stable_events = [
        "restaurant_id",
        "source_id",
        "inspection_date",
        "inspection_type",
        "action",
        "action_conflict",
        "score",
        "score_conflict",
        "score_conflict_values",
        "grade",
        "grade_conflict",
        "grade_conflict_values",
        "grade_date",
        "violation_count",
        "critical_violation_count",
        "high_priority_count",
        "intermediate_count",
        "basic_count",
    ]
    stable_violations = [
        "restaurant_id",
        "inspection_date",
        "violation_code",
        "violation_description",
        "critical_flag_raw",
        "severity",
        "corrected_on_site",
    ]
    result = _build(soda_csv)
    events = (
        result.inspection_events.select(stable_events)
        .sort(["restaurant_id", "inspection_date", "inspection_type"])
        .to_dicts()
    )
    violations = (
        result.violation_events.select(stable_violations)
        .sort(["restaurant_id", "inspection_date", "violation_code"])
        .to_dicts()
    )
    assert events == golden["events"]
    assert violations == golden["violations"]
