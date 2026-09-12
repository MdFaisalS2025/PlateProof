"""Tests for Florida primary/secondary target builders."""

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


def test_exact_routine_food_match(make_event: Any, events_frame: Any) -> None:
    from plateproof.models.florida_risk import build_florida_primary_target

    events = events_frame(
        [
            make_event(
                restaurant_id="florida:1",
                jurisdiction="florida",
                inspection_id="r1",
                inspection_date=date(2024, 1, 1),
                inspection_type="Routine - Food",
                native_visit_sequence=1,
                native_inspection_group_id="g1",
                high_priority_count=0,
                disposition_status="met_standards",
            ),
        ]
    )
    target, _ = build_florida_primary_target(events)
    assert _label_for(target, "r1") == 0


def test_rejects_strings_merely_containing_routine(make_event: Any, events_frame: Any) -> None:
    from plateproof.models.florida_risk import build_florida_primary_target

    events = events_frame(
        [
            make_event(
                restaurant_id="florida:1",
                jurisdiction="florida",
                inspection_id="not_r",
                inspection_date=date(2024, 1, 1),
                inspection_type="Routine - Food Complaint",
                native_visit_sequence=1,
                native_inspection_group_id="g1",
                high_priority_count=0,
                disposition_status="met_standards",
            ),
        ]
    )
    target, report = build_florida_primary_target(events)
    assert _label_for(target, "not_r") is None
    assert report.excluded_non_qualifying_type == 1


def test_callback_visit_excluded(make_event: Any, events_frame: Any) -> None:
    from plateproof.models.florida_risk import build_florida_primary_target

    events = events_frame(
        [
            make_event(
                restaurant_id="florida:1",
                jurisdiction="florida",
                inspection_id="initial",
                inspection_date=date(2024, 1, 1),
                inspection_type="Routine - Food",
                native_visit_sequence=1,
                native_inspection_group_id="g1",
                high_priority_count=1,
                disposition_status="follow_up_required",
            ),
            make_event(
                restaurant_id="florida:1",
                jurisdiction="florida",
                inspection_id="callback",
                inspection_date=date(2024, 1, 10),
                inspection_type="Routine - Food",
                native_visit_sequence=2,
                native_inspection_group_id="g1",
                high_priority_count=0,
                disposition_status="met_standards",
            ),
        ]
    )
    target, report = build_florida_primary_target(events)
    assert _label_for(target, "initial") == 1
    assert _label_for(target, "callback") is None
    assert report.excluded_not_initial_visit == 1


def test_high_priority_makes_positive(make_event: Any, events_frame: Any) -> None:
    from plateproof.models.florida_risk import build_florida_primary_target

    events = events_frame(
        [
            make_event(
                restaurant_id="florida:1",
                jurisdiction="florida",
                inspection_id="hp",
                inspection_date=date(2024, 1, 1),
                inspection_type="Routine - Food",
                native_visit_sequence=1,
                native_inspection_group_id="g1",
                high_priority_count=1,
                disposition_status="met_standards",
            ),
        ]
    )
    target, _ = build_florida_primary_target(events)
    assert _label_for(target, "hp") == 1


def test_follow_up_required_is_positive(make_event: Any, events_frame: Any) -> None:
    from plateproof.models.florida_risk import build_florida_primary_target

    events = events_frame(
        [
            make_event(
                restaurant_id="florida:1",
                jurisdiction="florida",
                inspection_id="fu",
                inspection_date=date(2024, 1, 1),
                inspection_type="Routine - Food",
                native_visit_sequence=1,
                native_inspection_group_id="g1",
                high_priority_count=0,
                disposition_status="follow_up_required",
            ),
        ]
    )
    target, _ = build_florida_primary_target(events)
    assert _label_for(target, "fu") == 1


def test_temporary_closure_is_positive_never_negative(make_event: Any, events_frame: Any) -> None:
    from plateproof.models.florida_risk import build_florida_primary_target

    events = events_frame(
        [
            make_event(
                restaurant_id="florida:1",
                jurisdiction="florida",
                inspection_id="tc",
                inspection_date=date(2024, 1, 1),
                inspection_type="Routine - Food",
                native_visit_sequence=1,
                native_inspection_group_id="g1",
                high_priority_count=0,
                disposition_status="temporary_closure",
            ),
        ]
    )
    target, _ = build_florida_primary_target(events)
    assert _label_for(target, "tc") == 1


def test_unknown_disposition_excluded(make_event: Any, events_frame: Any) -> None:
    from plateproof.models.florida_risk import build_florida_primary_target

    events = events_frame(
        [
            make_event(
                restaurant_id="florida:1",
                jurisdiction="florida",
                inspection_id="unk",
                inspection_date=date(2024, 1, 1),
                inspection_type="Routine - Food",
                native_visit_sequence=1,
                native_inspection_group_id="g1",
                high_priority_count=0,
                disposition_status="unknown",
            ),
        ]
    )
    target, report = build_florida_primary_target(events)
    assert _label_for(target, "unk") is None
    assert report.excluded_unrecognized_disposition == 1


def test_missing_high_priority_count_excluded(make_event: Any, events_frame: Any) -> None:
    from plateproof.models.florida_risk import build_florida_primary_target

    events = events_frame(
        [
            make_event(
                restaurant_id="florida:1",
                jurisdiction="florida",
                inspection_id="missing_hp",
                inspection_date=date(2024, 1, 1),
                inspection_type="Routine - Food",
                native_visit_sequence=1,
                native_inspection_group_id="g1",
                high_priority_count=None,
                disposition_status="met_standards",
            ),
        ]
    )
    target, report = build_florida_primary_target(events)
    assert _label_for(target, "missing_hp") is None
    assert report.excluded_missing_high_priority == 1


def test_ambiguous_duplicate_inspection_group_excluded_and_reported(
    make_event: Any, events_frame: Any
) -> None:
    from plateproof.models.florida_risk import build_florida_primary_target

    # Two rows both claim visit_sequence == 1 within the same group -- data
    # integrity problem; never silently pick one.
    events = events_frame(
        [
            make_event(
                restaurant_id="florida:1",
                jurisdiction="florida",
                inspection_id="dup1",
                inspection_date=date(2024, 1, 1),
                inspection_type="Routine - Food",
                native_visit_sequence=1,
                native_inspection_group_id="g1",
                high_priority_count=0,
                disposition_status="met_standards",
            ),
            make_event(
                restaurant_id="florida:1",
                jurisdiction="florida",
                inspection_id="dup2",
                inspection_date=date(2024, 1, 2),
                inspection_type="Routine - Food",
                native_visit_sequence=1,
                native_inspection_group_id="g1",
                high_priority_count=1,
                disposition_status="follow_up_required",
            ),
        ]
    )
    target, report = build_florida_primary_target(events)
    assert _label_for(target, "dup1") is None
    assert _label_for(target, "dup2") is None
    assert report.ambiguous_group_count == 1


def test_rejects_non_florida_rows(make_event: Any, events_frame: Any) -> None:
    from plateproof.models.florida_risk import build_florida_primary_target

    events = events_frame(
        [
            make_event(
                restaurant_id="nyc:1",
                jurisdiction="nyc",
                inspection_id="n1",
                inspection_date=date(2024, 1, 1),
                inspection_type="Cycle Inspection / Initial Inspection",
            ),
        ]
    )
    with pytest.raises(ValueError, match="florida"):
        build_florida_primary_target(events)


def test_secondary_temporary_closure_target(make_event: Any, events_frame: Any) -> None:
    from plateproof.models.florida_risk import build_florida_temporary_closure_target

    events = events_frame(
        [
            make_event(
                restaurant_id="florida:1",
                jurisdiction="florida",
                inspection_id="closed",
                inspection_date=date(2024, 1, 1),
                inspection_type="Routine - Food",
                native_visit_sequence=1,
                native_inspection_group_id="g1",
                disposition_status="temporary_closure",
            ),
            make_event(
                restaurant_id="florida:1",
                jurisdiction="florida",
                inspection_id="ok",
                inspection_date=date(2024, 2, 1),
                inspection_type="Routine - Food",
                native_visit_sequence=1,
                native_inspection_group_id="g2",
                disposition_status="met_standards",
            ),
        ]
    )
    target = build_florida_temporary_closure_target(events)
    assert _label_for(target, "closed") == 1
    assert _label_for(target, "ok") == 0


def test_feature_list_has_no_nyc_columns() -> None:
    from plateproof.models.florida_risk import FL_FEATURE_LIST

    assert not any(name.startswith("nyc_") for name in FL_FEATURE_LIST)
    assert any(name.startswith("fl_") for name in FL_FEATURE_LIST)


def test_feature_list_is_ordered_subset_of_allowlist() -> None:
    from plateproof.features.temporal import MODEL_FEATURE_ALLOWLIST
    from plateproof.models.florida_risk import FL_FEATURE_LIST

    assert set(FL_FEATURE_LIST) <= MODEL_FEATURE_ALLOWLIST
    assert list(FL_FEATURE_LIST) == sorted(FL_FEATURE_LIST)


def test_no_repeated_violation_category_helper_exists() -> None:
    import plateproof.models.florida_risk as module

    assert not hasattr(module, "build_florida_repeated_violation_category_target")
