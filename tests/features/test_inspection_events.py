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
        "disposition",
        "disposition_status",
        "native_inspection_group_id",
        "native_visit_sequence",
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
        "count",
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
        "assert_unique_inspection_id",
        "finalize_event_frame",
        "pl",
        "Any",
        "annotations",
    }
    assert public <= allowed, f"unexpected public names: {sorted(public - allowed)}"
    forbidden = {"shift", "rolling", "lag", "build_temporal_features", "target", "label"}
    assert not (public & forbidden)


def test_assert_unique_inspection_id_accepts_unique_ids() -> None:
    from plateproof.features.inspection_events import assert_unique_inspection_id

    frame = pl.DataFrame({"inspection_id": ["a", "b", "c"]})
    assert assert_unique_inspection_id(frame) is None


def test_assert_unique_inspection_id_raises_on_duplicate() -> None:
    from plateproof.features.inspection_events import assert_unique_inspection_id

    frame = pl.DataFrame({"inspection_id": ["a", "a", "b"]})
    with pytest.raises(ValueError, match="'a'"):
        assert_unique_inspection_id(frame)


def test_assert_unique_inspection_id_permits_repeated_key_with_distinct_ids() -> None:
    # The exact Florida scenario that motivated this function: two genuinely
    # distinct visits can legitimately share (restaurant_id, inspection_date,
    # inspection_type) when a native id already tells them apart.
    from plateproof.features.inspection_events import (
        assert_unique_inspection_id,
        assert_unique_inspection_key,
    )

    frame = pl.DataFrame(
        {
            "inspection_id": ["florida:v1", "florida:v2"],
            "restaurant_id": ["florida:1", "florida:1"],
            "inspection_date": ["2024-01-01", "2024-01-01"],
            "inspection_type": ["Routine - Food", "Routine - Food"],
        }
    )
    assert assert_unique_inspection_id(frame) is None
    with pytest.raises(ValueError):
        assert_unique_inspection_key(frame)


def test_finalize_event_frame_fills_missing_schema_keys_with_null() -> None:
    from plateproof.features.inspection_events import finalize_event_frame

    schema = {"a": pl.String(), "b": pl.Int32()}
    frame = finalize_event_frame([{"a": "x"}, {"a": "y", "b": 2}], schema)
    assert frame.columns == ["a", "b"]
    assert frame.get_column("b").to_list() == [None, 2]


def test_finalize_event_frame_empty_records_produces_empty_typed_frame() -> None:
    from plateproof.features.inspection_events import finalize_event_frame

    schema = {"a": pl.String(), "b": pl.Int32()}
    frame = finalize_event_frame([], schema)
    assert frame.height == 0
    assert frame.schema["a"] == pl.String
    assert frame.schema["b"] == pl.Int32
