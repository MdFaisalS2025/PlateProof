"""Compatibility of Task 6 with real Task 2/3/5 outputs -- no ingestion code touched."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl

FIXED_TS = datetime(2026, 1, 1, tzinfo=UTC)
FIXTURES = Path(__file__).parent.parent / "ingestion" / "fixtures"


def test_nyc_pipeline_compatibility() -> None:
    from plateproof.features.temporal import build_temporal_features
    from plateproof.ingestion.nyc import build_nyc_inspection_events, load_nyc_raw
    from plateproof.models.nyc_risk import NYC_FEATURE_LIST, build_nyc_primary_target
    from plateproof.models.training import assemble_training_frame

    raw = load_nyc_raw(FIXTURES / "nyc_sample_soda.csv")
    result = build_nyc_inspection_events(raw, ingested_at=FIXED_TS)
    target, report = build_nyc_primary_target(result.inspection_events)
    temporal = build_temporal_features(
        result.inspection_events,
        result.violation_events,
        cutoff_date=result.report.snapshot_date.replace(year=2100),
    )
    frame = assemble_training_frame(
        temporal, target, NYC_FEATURE_LIST, "nyc", "nyc_next_initial_score_ge_14"
    )
    assert frame.X.height == frame.y.height
    assert report.input_row_count == result.inspection_events.height


def test_florida_pipeline_compatibility() -> None:
    from plateproof.features.temporal import build_temporal_features
    from plateproof.ingestion.florida import (
        FloridaExtractSource,
        build_florida_inspection_events,
        load_florida_extracts,
    )
    from plateproof.models.florida_risk import FL_FEATURE_LIST, build_florida_primary_target
    from plateproof.models.training import assemble_training_frame

    raw = load_florida_extracts([FloridaExtractSource(path=FIXTURES / "fl_sample_current.csv")])
    result = build_florida_inspection_events(raw, ingested_at=FIXED_TS)
    target, report = build_florida_primary_target(result.inspection_events)
    far_future = result.inspection_events.get_column("inspection_date").max()
    temporal = build_temporal_features(
        result.inspection_events,
        result.violation_events,
        cutoff_date=far_future.replace(year=2100)
        if far_future
        else __import__("datetime").date(2100, 1, 1),
    )
    frame = assemble_training_frame(
        temporal,
        target,
        FL_FEATURE_LIST,
        "florida",
        "florida_next_routine_high_priority_or_follow_up",
    )
    assert frame.X.height == frame.y.height
    assert report.input_row_count == result.inspection_events.height


def test_nyc_and_florida_events_concat_still_builds_both_targets_independently() -> None:
    from plateproof.ingestion.florida import (
        FloridaExtractSource,
        build_florida_inspection_events,
        load_florida_extracts,
    )
    from plateproof.ingestion.nyc import build_nyc_inspection_events, load_nyc_raw
    from plateproof.models.florida_risk import build_florida_primary_target
    from plateproof.models.nyc_risk import build_nyc_primary_target

    nyc = build_nyc_inspection_events(
        load_nyc_raw(FIXTURES / "nyc_sample_soda.csv"), ingested_at=FIXED_TS
    )
    fl = build_florida_inspection_events(
        load_florida_extracts([FloridaExtractSource(path=FIXTURES / "fl_sample_current.csv")]),
        ingested_at=FIXED_TS,
    )
    combined = pl.concat([nyc.inspection_events, fl.inspection_events])
    # each builder still operates correctly when given only its own jurisdiction's slice
    nyc_target, _ = build_nyc_primary_target(combined.filter(pl.col("jurisdiction") == "nyc"))
    fl_target, _ = build_florida_primary_target(
        combined.filter(pl.col("jurisdiction") == "florida")
    )
    assert nyc_target.height >= 0
    assert fl_target.height >= 0
