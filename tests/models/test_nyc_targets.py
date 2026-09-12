"""Tests for NYC primary/secondary target builders."""

from __future__ import annotations

from datetime import date
from typing import Any

import polars as pl
import pytest


def _label_for(target: pl.DataFrame, inspection_id: str) -> int | None:
    row = target.filter(pl.col("inspection_id") == inspection_id)
    if row.height == 0:
        return None
    value = row.get_column("label").to_list()[0]
    return None if value is None else int(value)


def test_qualifying_initial_types_use_exact_equality_not_substring(
    make_event: Any, events_frame: Any
) -> None:
    from plateproof.models.nyc_risk import build_nyc_primary_target

    events = events_frame(
        [
            make_event(
                restaurant_id="nyc:1",
                jurisdiction="nyc",
                inspection_id="e1",
                inspection_date=date(2024, 1, 1),
                inspection_type="Cycle Inspection / Initial Inspection",
                score=20.0,
            ),
            # substring match only -- must NOT qualify under exact equality
            make_event(
                restaurant_id="nyc:1",
                jurisdiction="nyc",
                inspection_id="e2",
                inspection_date=date(2024, 2, 1),
                inspection_type="Cycle Inspection / Initial Inspection Follow-up",
                score=20.0,
            ),
        ]
    )
    target, report = build_nyc_primary_target(events)
    assert _label_for(target, "e1") == 1
    assert _label_for(target, "e2") is None
    assert report.excluded_non_qualifying_type == 1


def test_reinspection_excluded(make_event: Any, events_frame: Any) -> None:
    from plateproof.models.nyc_risk import build_nyc_primary_target

    events = events_frame(
        [
            make_event(
                restaurant_id="nyc:1",
                jurisdiction="nyc",
                inspection_id="r1",
                inspection_date=date(2024, 1, 1),
                inspection_type="Cycle Inspection / Re-inspection",
                score=20.0,
            ),
        ]
    )
    target, report = build_nyc_primary_target(events)
    assert _label_for(target, "r1") is None
    assert report.excluded_non_qualifying_type == 1


def test_threshold_at_14(make_event: Any, events_frame: Any) -> None:
    from plateproof.models.nyc_risk import build_nyc_primary_target

    events = events_frame(
        [
            make_event(
                restaurant_id="nyc:1",
                jurisdiction="nyc",
                inspection_id="hi",
                inspection_date=date(2024, 1, 1),
                inspection_type="Cycle Inspection / Initial Inspection",
                score=14.0,
            ),
            make_event(
                restaurant_id="nyc:1",
                jurisdiction="nyc",
                inspection_id="lo",
                inspection_date=date(2024, 2, 1),
                inspection_type="Cycle Inspection / Initial Inspection",
                score=13.0,
            ),
        ]
    )
    target, _ = build_nyc_primary_target(events)
    assert _label_for(target, "hi") == 1
    assert _label_for(target, "lo") == 0


def test_conflicted_score_excluded(make_event: Any, events_frame: Any) -> None:
    from plateproof.models.nyc_risk import build_nyc_primary_target

    events = events_frame(
        [
            make_event(
                restaurant_id="nyc:1",
                jurisdiction="nyc",
                inspection_id="c1",
                inspection_date=date(2024, 1, 1),
                inspection_type="Cycle Inspection / Initial Inspection",
                score=20.0,
                score_conflict=True,
            ),
        ]
    )
    target, report = build_nyc_primary_target(events)
    assert _label_for(target, "c1") is None
    assert report.excluded_missing_or_conflicted_score == 1


def test_null_score_excluded_grade_never_substituted(make_event: Any, events_frame: Any) -> None:
    from plateproof.models.nyc_risk import build_nyc_primary_target

    events = events_frame(
        [
            make_event(
                restaurant_id="nyc:1",
                jurisdiction="nyc",
                inspection_id="g1",
                inspection_date=date(2024, 1, 1),
                inspection_type="Cycle Inspection / Initial Inspection",
                score=None,
                grade="C",
            ),
        ]
    )
    target, report = build_nyc_primary_target(events)
    assert _label_for(target, "g1") is None
    assert report.excluded_missing_or_conflicted_score == 1


def test_counts_by_inspection_type_reported(make_event: Any, events_frame: Any) -> None:
    from plateproof.models.nyc_risk import build_nyc_primary_target

    events = events_frame(
        [
            make_event(
                restaurant_id="nyc:1",
                jurisdiction="nyc",
                inspection_id="a",
                inspection_date=date(2024, 1, 1),
                inspection_type="Cycle Inspection / Initial Inspection",
                score=20.0,
            ),
            make_event(
                restaurant_id="nyc:1",
                jurisdiction="nyc",
                inspection_id="b",
                inspection_date=date(2024, 2, 1),
                inspection_type="Administrative Miscellaneous / Initial Inspection",
                score=None,
            ),
        ]
    )
    _, report = build_nyc_primary_target(events)
    assert report.counts_by_inspection_type["Cycle Inspection / Initial Inspection"] == 1
    assert (
        report.counts_by_inspection_type["Administrative Miscellaneous / Initial Inspection"] == 1
    )


def test_rejects_non_nyc_rows(make_event: Any, events_frame: Any) -> None:
    from plateproof.models.nyc_risk import build_nyc_primary_target

    events = events_frame(
        [
            make_event(
                restaurant_id="florida:1",
                jurisdiction="florida",
                inspection_id="f1",
                inspection_date=date(2024, 1, 1),
                inspection_type="Routine - Food",
            ),
        ]
    )
    with pytest.raises(ValueError, match="nyc"):
        build_nyc_primary_target(events)


def test_secondary_score_ge_28(make_event: Any, events_frame: Any) -> None:
    from plateproof.models.nyc_risk import build_nyc_score_ge_28_target

    events = events_frame(
        [
            make_event(
                restaurant_id="nyc:1",
                jurisdiction="nyc",
                inspection_id="hi",
                inspection_date=date(2024, 1, 1),
                inspection_type="Cycle Inspection / Initial Inspection",
                score=28.0,
            ),
            make_event(
                restaurant_id="nyc:1",
                jurisdiction="nyc",
                inspection_id="lo",
                inspection_date=date(2024, 2, 1),
                inspection_type="Cycle Inspection / Initial Inspection",
                score=27.0,
            ),
        ]
    )
    target = build_nyc_score_ge_28_target(events)
    assert _label_for(target, "hi") == 1
    assert _label_for(target, "lo") == 0


def test_secondary_any_critical_violation(make_event: Any, events_frame: Any) -> None:
    from plateproof.models.nyc_risk import build_nyc_any_critical_target

    events = events_frame(
        [
            make_event(
                restaurant_id="nyc:1",
                jurisdiction="nyc",
                inspection_id="crit",
                inspection_date=date(2024, 1, 1),
                inspection_type="Cycle Inspection / Initial Inspection",
                critical_violation_count=2,
            ),
            make_event(
                restaurant_id="nyc:1",
                jurisdiction="nyc",
                inspection_id="noncrit",
                inspection_date=date(2024, 2, 1),
                inspection_type="Cycle Inspection / Initial Inspection",
                critical_violation_count=0,
            ),
        ]
    )
    target = build_nyc_any_critical_target(events)
    assert _label_for(target, "crit") == 1
    assert _label_for(target, "noncrit") == 0


def test_feature_list_has_no_florida_columns() -> None:
    from plateproof.models.nyc_risk import NYC_FEATURE_LIST

    assert not any(name.startswith("fl_") for name in NYC_FEATURE_LIST)
    assert any(name.startswith("nyc_") for name in NYC_FEATURE_LIST)


def test_feature_list_is_ordered_subset_of_allowlist() -> None:
    from plateproof.features.temporal import MODEL_FEATURE_ALLOWLIST
    from plateproof.models.nyc_risk import NYC_FEATURE_LIST

    assert set(NYC_FEATURE_LIST) <= MODEL_FEATURE_ALLOWLIST
    assert list(NYC_FEATURE_LIST) == sorted(NYC_FEATURE_LIST)


def test_feature_list_is_a_fixed_explicit_tuple_not_derived_from_the_allowlist() -> None:
    """A future Task 5 feature added to MODEL_FEATURE_ALLOWLIST must never
    silently enter this already-frozen model's feature schema."""
    import importlib

    from plateproof.models import nyc_risk

    before = nyc_risk.NYC_FEATURE_LIST
    import plateproof.features.temporal as temporal_module

    patched_allowlist = frozenset(
        {*temporal_module.MODEL_FEATURE_ALLOWLIST, "nyc_brand_new_feature"}
    )
    original_allowlist = temporal_module.MODEL_FEATURE_ALLOWLIST
    try:
        temporal_module.MODEL_FEATURE_ALLOWLIST = patched_allowlist
        importlib.reload(nyc_risk)
        assert nyc_risk.NYC_FEATURE_LIST == before
        assert "nyc_brand_new_feature" not in nyc_risk.NYC_FEATURE_LIST
    finally:
        temporal_module.MODEL_FEATURE_ALLOWLIST = original_allowlist
        importlib.reload(nyc_risk)
