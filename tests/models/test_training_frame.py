"""Tests for assemble_training_frame and feature-matrix extraction."""

from __future__ import annotations

from datetime import date
from typing import Any

import polars as pl
import pytest


def _events_and_temporal(
    make_event: Any, events_frame: Any, violations_frame: Any, temporal_result: Any
) -> Any:
    rows = [
        make_event(
            restaurant_id="nyc:1",
            jurisdiction="nyc",
            inspection_id="a",
            inspection_date=date(2024, 1, 1),
            score=10.0,
            inspection_type="Cycle Inspection / Initial Inspection",
        ),
        make_event(
            restaurant_id="nyc:1",
            jurisdiction="nyc",
            inspection_id="b",
            inspection_date=date(2024, 2, 1),
            score=20.0,
            inspection_type="Cycle Inspection / Initial Inspection",
        ),
    ]
    events = events_frame(rows)
    result = temporal_result(events, violations_frame([]), date(2024, 6, 1))
    return events, result


def test_assemble_joins_by_inspection_id_only(
    make_event: Any, events_frame: Any, violations_frame: Any, temporal_result: Any
) -> None:
    from plateproof.models.nyc_risk import NYC_FEATURE_LIST, build_nyc_primary_target
    from plateproof.models.training import assemble_training_frame

    events, result = _events_and_temporal(
        make_event, events_frame, violations_frame, temporal_result
    )
    target, _ = build_nyc_primary_target(events)
    frame = assemble_training_frame(
        result, target, NYC_FEATURE_LIST, "nyc", "nyc_next_initial_score_ge_14"
    )
    assert set(frame.y.get_column("inspection_id").to_list()) <= set(
        frame.X.get_column("inspection_id").to_list()
    )
    assert frame.X.height == frame.y.height


def test_duplicate_inspection_id_in_target_fails(
    make_event: Any, events_frame: Any, violations_frame: Any, temporal_result: Any
) -> None:
    from plateproof.models.nyc_risk import NYC_FEATURE_LIST
    from plateproof.models.training import assemble_training_frame

    events, result = _events_and_temporal(
        make_event, events_frame, violations_frame, temporal_result
    )
    dup_target = pl.DataFrame({"inspection_id": ["a", "a"], "label": [1, 0]})
    with pytest.raises(ValueError, match="duplicate"):
        assemble_training_frame(result, dup_target, NYC_FEATURE_LIST, "nyc", "x")


def test_current_outcome_columns_never_enter_x(
    make_event: Any, events_frame: Any, violations_frame: Any, temporal_result: Any
) -> None:
    from plateproof.models.nyc_risk import NYC_FEATURE_LIST, build_nyc_primary_target
    from plateproof.models.training import assemble_training_frame

    events, result = _events_and_temporal(
        make_event, events_frame, violations_frame, temporal_result
    )
    target, _ = build_nyc_primary_target(events)
    frame = assemble_training_frame(result, target, NYC_FEATURE_LIST, "nyc", "t")
    for forbidden in (
        "score",
        "grade",
        "action",
        "disposition",
        "inspection_date",
        "inspection_type",
    ):
        assert forbidden not in frame.X.columns


def test_inspection_id_removed_before_estimator_fitting(
    make_event: Any, events_frame: Any, violations_frame: Any, temporal_result: Any
) -> None:
    from plateproof.models.nyc_risk import NYC_FEATURE_LIST, build_nyc_primary_target
    from plateproof.models.training import assemble_training_frame, feature_matrix

    events, result = _events_and_temporal(
        make_event, events_frame, violations_frame, temporal_result
    )
    target, _ = build_nyc_primary_target(events)
    frame = assemble_training_frame(result, target, NYC_FEATURE_LIST, "nyc", "t")
    matrix = feature_matrix(frame.X, frame.feature_order)
    assert matrix.shape == (frame.X.height, len(NYC_FEATURE_LIST))
    # inspection_id (a string column) cannot appear among numeric feature columns
    import numpy as np

    assert matrix.dtype != np.dtype("O") or "inspection_id" not in frame.feature_order


def test_rejects_florida_feature_in_nyc_call(
    make_event: Any, events_frame: Any, violations_frame: Any, temporal_result: Any
) -> None:
    from plateproof.models.nyc_risk import build_nyc_primary_target
    from plateproof.models.training import assemble_training_frame

    events, result = _events_and_temporal(
        make_event, events_frame, violations_frame, temporal_result
    )
    target, _ = build_nyc_primary_target(events)
    with pytest.raises(ValueError):
        assemble_training_frame(result, target, ["fl_previous_day_high_priority_count"], "nyc", "t")


def test_rejects_unknown_feature_name(
    make_event: Any, events_frame: Any, violations_frame: Any, temporal_result: Any
) -> None:
    from plateproof.models.nyc_risk import build_nyc_primary_target
    from plateproof.models.training import assemble_training_frame

    events, result = _events_and_temporal(
        make_event, events_frame, violations_frame, temporal_result
    )
    target, _ = build_nyc_primary_target(events)
    with pytest.raises(ValueError):
        assemble_training_frame(result, target, ["michelin_star_rank"], "nyc", "t")
