"""Contract tests for the jurisdiction-neutral shared schema module."""

from __future__ import annotations

import polars as pl
import pytest


def test_schemas_expose_expected_columns() -> None:
    from plateproof.features.inspection_events import (
        INSPECTION_EVENT_SCHEMA,
        VIOLATION_EVENT_SCHEMA,
    )

    for name in (
        "inspection_id",
        "restaurant_id",
        "source_id",
        "jurisdiction",
        "inspection_date",
        "inspection_type",
        "score",
        "score_conflict",
        "score_conflict_values",
        "grade",
        "grade_date",
        "high_priority_count",
        "intermediate_count",
        "basic_count",
        "source_snapshot_date",
        "source_retrieved_at_utc",
        "source_sha256",
        "ingested_at",
        "pipeline_version",
    ):
        assert name in INSPECTION_EVENT_SCHEMA

    for name in (
        "violation_event_id",
        "inspection_id",
        "violation_code",
        "violation_code_norm",
        "violation_description",
        "critical_flag_raw",
        "severity",
        "corrected_on_site",
    ):
        assert name in VIOLATION_EVENT_SCHEMA


def test_provenance_columns_use_explicit_polars_types() -> None:
    from plateproof.features.inspection_events import INSPECTION_EVENT_SCHEMA

    assert INSPECTION_EVENT_SCHEMA["inspection_date"] == pl.Date
    assert INSPECTION_EVENT_SCHEMA["ingested_at"] == pl.Datetime("us", "UTC")
    assert INSPECTION_EVENT_SCHEMA["source_retrieved_at_utc"] == pl.Datetime("us", "UTC")
    assert INSPECTION_EVENT_SCHEMA["violation_count"] == pl.Int32
    assert INSPECTION_EVENT_SCHEMA["score"] == pl.Float64
    assert INSPECTION_EVENT_SCHEMA["score_conflict"] == pl.Boolean
    assert INSPECTION_EVENT_SCHEMA["inspection_id"] == pl.String


def test_assert_unique_inspection_key_accepts_unique_frame() -> None:
    from plateproof.features.inspection_events import assert_unique_inspection_key

    frame = pl.DataFrame(
        {
            "restaurant_id": ["nyc:1", "nyc:1", "nyc:2"],
            "inspection_date": ["2024-01-01", "2024-02-01", "2024-01-01"],
            "inspection_type": ["A", "A", "A"],
        }
    )
    assert assert_unique_inspection_key(frame) is None


def test_assert_unique_inspection_key_raises_and_lists_duplicate() -> None:
    from plateproof.features.inspection_events import assert_unique_inspection_key

    frame = pl.DataFrame(
        {
            "restaurant_id": ["nyc:1", "nyc:1"],
            "inspection_date": ["2024-01-01", "2024-01-01"],
            "inspection_type": ["A", "A"],
        }
    )
    with pytest.raises(ValueError, match="nyc:1"):
        assert_unique_inspection_key(frame)


def test_shared_module_has_no_feature_engineering_api() -> None:
    from plateproof.features import inspection_events as module

    public = {name for name in vars(module) if not name.startswith("_")}
    allowed = {
        "INSPECTION_KEY",
        "INSPECTION_EVENT_SCHEMA",
        "VIOLATION_EVENT_SCHEMA",
        "assert_unique_inspection_key",
        "pl",
        "annotations",
    }
    assert public <= allowed, f"unexpected public names: {sorted(public - allowed)}"
    forbidden = {"shift", "rolling", "lag", "build_temporal_features", "target", "label"}
    assert not (public & forbidden)
