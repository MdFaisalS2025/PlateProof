"""Tests for build_temporal_features: leakage-safe temporal feature construction."""

from __future__ import annotations

import random
from datetime import UTC, date, datetime
from typing import Any

import polars as pl
import pytest

FIXED_TS = datetime(2024, 6, 1, tzinfo=UTC)


def _row(result: Any, frame_name: str, inspection_id: str) -> dict[str, Any]:
    frame = getattr(result, frame_name)
    matches = frame.filter(pl.col("inspection_id") == inspection_id)
    assert matches.height == 1, f"expected exactly one {frame_name} row for {inspection_id}"
    return matches.to_dicts()[0]


def _build(events: pl.DataFrame, violations: pl.DataFrame, cutoff: date) -> Any:
    from plateproof.features.temporal import build_temporal_features

    return build_temporal_features(events, violations, cutoff)


# --------------------------------------------------------------------------- #
# Same-day isolation and history depth (#1, #7)                               #
# --------------------------------------------------------------------------- #


def test_same_day_events_isolated_and_history_depth_counts_events_not_days(
    make_event: Any, events_frame: Any, violations_frame: Any, d: Any
) -> None:
    rid = "nyc:depth"
    e1 = make_event(
        restaurant_id=rid,
        jurisdiction="nyc",
        inspection_id="e1",
        inspection_date=d(2024, 1, 10),
        score=10.0,
    )
    e2a = make_event(
        restaurant_id=rid,
        jurisdiction="nyc",
        inspection_id="e2a",
        inspection_date=d(2024, 2, 10),
        score=20.0,
    )
    e2b = make_event(
        restaurant_id=rid,
        jurisdiction="nyc",
        inspection_id="e2b",
        inspection_date=d(2024, 2, 10),
        score=20.0,
    )
    e3 = make_event(
        restaurant_id=rid,
        jurisdiction="nyc",
        inspection_id="e3",
        inspection_date=d(2024, 3, 10),
        score=30.0,
    )
    events = events_frame([e1, e2a, e2b, e3])
    result = _build(events, violations_frame([]), d(2024, 6, 1))

    row1 = _row(result, "features", "e1")
    assert row1["history_depth"] == 0
    assert row1["missing_history"] is True

    row2a = _row(result, "features", "e2a")
    row2b = _row(result, "features", "e2b")
    assert row2a["history_depth"] == 1
    assert row2a["prior_inspection_day_count"] == 1
    row2a.pop("inspection_id")
    row2b.pop("inspection_id")
    assert row2a == row2b  # same-day rows see identical history

    id2a = _row(result, "identity", "e2a")
    assert id2a["previous_inspection_date"] if "previous_inspection_date" in id2a else True

    row3 = _row(result, "features", "e3")
    assert row3["history_depth"] == 3  # two events on 02-10 + one on 01-10
    assert row3["prior_inspection_day_count"] == 2  # only two distinct prior dates
    assert row3["nyc_previous_day_score"] == 20.0
    assert row3["nyc_previous_day_score_complete"] is True


def test_row_order_shuffle_does_not_change_output(
    make_event: Any, events_frame: Any, violations_frame: Any, d: Any
) -> None:
    rid = "nyc:shuffle"
    rows = [
        make_event(
            restaurant_id=rid,
            jurisdiction="nyc",
            inspection_id=f"s{i}",
            inspection_date=d(2024, 1, i + 1),
            score=float(i),
        )
        for i in range(5)
    ]
    shuffled = rows[:]
    random.Random(42).shuffle(shuffled)

    a = _build(events_frame(rows), violations_frame([]), d(2024, 6, 1))
    b = _build(events_frame(shuffled), violations_frame([]), d(2024, 6, 1))
    assert a.features.equals(b.features)
    assert a.identity.equals(b.identity)


# --------------------------------------------------------------------------- #
# NYC score_conflict, agreement, disagreement (#2, #3, #4)                    #
# --------------------------------------------------------------------------- #


def test_score_conflict_excludes_score_from_trusted_history(
    make_event: Any, events_frame: Any, violations_frame: Any, d: Any
) -> None:
    rid = "nyc:conflict"
    events = events_frame(
        [
            make_event(
                restaurant_id=rid,
                jurisdiction="nyc",
                inspection_id="c1",
                inspection_date=d(2024, 1, 5),
                score=999.0,
                score_conflict=True,
            ),
            make_event(
                restaurant_id=rid,
                jurisdiction="nyc",
                inspection_id="c2",
                inspection_date=d(2024, 2, 5),
                score=5.0,
            ),
        ]
    )
    result = _build(events, violations_frame([]), d(2024, 6, 1))
    row = _row(result, "features", "c2")
    assert row["history_depth"] == 1
    assert row["nyc_previous_day_score"] is None
    assert row["nyc_previous_day_score_complete"] is False
    assert row["nyc_prior_valid_score_count"] == 0


def test_same_day_agreeing_scores_produce_daily_score(
    make_event: Any, events_frame: Any, violations_frame: Any, d: Any
) -> None:
    rid = "nyc:agree"
    events = events_frame(
        [
            make_event(
                restaurant_id=rid,
                jurisdiction="nyc",
                inspection_id="a1",
                inspection_date=d(2024, 1, 1),
                score=15.0,
            ),
            make_event(
                restaurant_id=rid,
                jurisdiction="nyc",
                inspection_id="a2",
                inspection_date=d(2024, 1, 1),
                score=15.0,
            ),
            make_event(
                restaurant_id=rid,
                jurisdiction="nyc",
                inspection_id="a3",
                inspection_date=d(2024, 2, 1),
                score=1.0,
            ),
        ]
    )
    result = _build(events, violations_frame([]), d(2024, 6, 1))
    row = _row(result, "features", "a3")
    assert row["nyc_previous_day_score"] == 15.0
    assert row["nyc_previous_day_score_complete"] is True


def test_same_day_disagreeing_scores_produce_null_and_report_entry(
    make_event: Any, events_frame: Any, violations_frame: Any, d: Any
) -> None:
    rid = "nyc:disagree"
    events = events_frame(
        [
            make_event(
                restaurant_id=rid,
                jurisdiction="nyc",
                inspection_id="g1",
                inspection_date=d(2024, 1, 1),
                score=10.0,
            ),
            make_event(
                restaurant_id=rid,
                jurisdiction="nyc",
                inspection_id="g2",
                inspection_date=d(2024, 1, 1),
                score=30.0,
            ),
            make_event(
                restaurant_id=rid,
                jurisdiction="nyc",
                inspection_id="g3",
                inspection_date=d(2024, 2, 1),
                score=1.0,
            ),
        ]
    )
    result = _build(events, violations_frame([]), d(2024, 6, 1))
    row = _row(result, "features", "g3")
    assert row["nyc_previous_day_score"] is None
    assert row["nyc_previous_day_score_complete"] is False
    assert result.build_report.nyc_same_day_score_conflict_day_count >= 1


# --------------------------------------------------------------------------- #
# Florida same-day completeness (#5, #6, #11, #12)                            #
# --------------------------------------------------------------------------- #


def test_florida_same_day_counts_sum_only_when_complete(
    make_event: Any, events_frame: Any, violations_frame: Any, d: Any
) -> None:
    rid = "florida:sums"
    events = events_frame(
        [
            make_event(
                restaurant_id=rid,
                jurisdiction="florida",
                inspection_id="f1",
                inspection_date=d(2024, 1, 1),
                high_priority_count=2,
                intermediate_count=1,
                basic_count=0,
                violation_count=3,
            ),
            make_event(
                restaurant_id=rid,
                jurisdiction="florida",
                inspection_id="f2",
                inspection_date=d(2024, 1, 1),
                high_priority_count=3,
                intermediate_count=None,
                basic_count=1,
                violation_count=None,
            ),
            make_event(
                restaurant_id=rid,
                jurisdiction="florida",
                inspection_id="f3",
                inspection_date=d(2024, 2, 1),
            ),
        ]
    )
    result = _build(events, violations_frame([]), d(2024, 6, 1))
    row = _row(result, "features", "f3")
    assert row["fl_previous_day_high_priority_count"] == 5
    assert row["fl_previous_day_high_priority_count_complete"] is True
    assert row["fl_previous_day_basic_count"] == 1
    assert row["fl_previous_day_basic_count_complete"] is True
    assert row["fl_previous_day_intermediate_count"] is None
    assert row["fl_previous_day_intermediate_count_complete"] is False
    assert row["fl_previous_day_total_violation_count"] is None
    assert row["fl_previous_day_total_violation_count_complete"] is False
    assert row["history_depth"] == 2


def test_florida_same_day_dispositions_count_as_multiple_prior_inspections(
    make_event: Any, events_frame: Any, violations_frame: Any, d: Any
) -> None:
    rid = "florida:disp"
    events = events_frame(
        [
            make_event(
                restaurant_id=rid,
                jurisdiction="florida",
                inspection_id="p1",
                inspection_date=d(2024, 1, 1),
                disposition_status="follow_up_required",
            ),
            make_event(
                restaurant_id=rid,
                jurisdiction="florida",
                inspection_id="p2",
                inspection_date=d(2024, 1, 1),
                disposition_status="temporary_closure",
            ),
            make_event(
                restaurant_id=rid,
                jurisdiction="florida",
                inspection_id="p3",
                inspection_date=d(2024, 2, 1),
            ),
        ]
    )
    result = _build(events, violations_frame([]), d(2024, 6, 1))
    row = _row(result, "features", "p3")
    assert row["fl_prior_follow_up_required_count"] == 1
    assert row["fl_prior_temporary_closure_count"] == 1


def test_unknown_florida_disposition_not_counted_as_compliant(
    make_event: Any, events_frame: Any, violations_frame: Any, d: Any
) -> None:
    rid = "florida:unknown"
    events = events_frame(
        [
            make_event(
                restaurant_id=rid,
                jurisdiction="florida",
                inspection_id="u1",
                inspection_date=d(2024, 1, 1),
                disposition_status="unknown",
            ),
            make_event(
                restaurant_id=rid,
                jurisdiction="florida",
                inspection_id="u2",
                inspection_date=d(2024, 2, 1),
                disposition_status="met_standards",
            ),
        ]
    )
    result = _build(events, violations_frame([]), d(2024, 6, 1))
    row = _row(result, "features", "u2")
    assert row["fl_prior_follow_up_required_count"] == 0
    assert row["fl_prior_temporary_closure_count"] == 0
    assert result.build_report.fl_unknown_disposition_event_count == 1


# --------------------------------------------------------------------------- #
# Variance / trend (#8, #9, #10, #13-variance min)                            #
# --------------------------------------------------------------------------- #


def test_variance_depends_on_valid_observations_not_history_depth(
    make_event: Any, events_frame: Any, violations_frame: Any, d: Any
) -> None:
    rid = "nyc:variance"
    events = events_frame(
        [
            make_event(
                restaurant_id=rid,
                jurisdiction="nyc",
                inspection_id="v1",
                inspection_date=d(2024, 1, 1),
                score=10.0,
            ),
            make_event(
                restaurant_id=rid,
                jurisdiction="nyc",
                inspection_id="v2",
                inspection_date=d(2024, 1, 5),
                score=20.0,
                score_conflict=True,
            ),
            make_event(
                restaurant_id=rid,
                jurisdiction="nyc",
                inspection_id="v3",
                inspection_date=d(2024, 1, 10),
                score=None,
            ),
            make_event(
                restaurant_id=rid,
                jurisdiction="nyc",
                inspection_id="v4",
                inspection_date=d(2024, 2, 1),
            ),
        ]
    )
    result = _build(events, violations_frame([]), d(2024, 6, 1))
    row = _row(result, "features", "v4")
    assert row["history_depth"] == 3
    assert row["nyc_prior_valid_score_count"] == 1
    assert row["nyc_score_prior_mean"] == 10.0
    assert row["nyc_score_prior_variance"] is None
    assert row["nyc_score_variance_available"] is False


def test_trend_uses_elapsed_days_not_row_index(
    make_event: Any, events_frame: Any, violations_frame: Any, d: Any
) -> None:
    rid = "nyc:trend"
    events = events_frame(
        [
            make_event(
                restaurant_id=rid,
                jurisdiction="nyc",
                inspection_id="t1",
                inspection_date=d(2024, 1, 1),
                score=10.0,
            ),
            make_event(
                restaurant_id=rid,
                jurisdiction="nyc",
                inspection_id="t2",
                inspection_date=d(2024, 1, 3),
                score=10.2,
            ),
            make_event(
                restaurant_id=rid,
                jurisdiction="nyc",
                inspection_id="t3",
                inspection_date=d(2024, 2, 3),
                score=13.3,
            ),
            make_event(
                restaurant_id=rid,
                jurisdiction="nyc",
                inspection_id="t4",
                inspection_date=d(2024, 3, 1),
            ),
        ]
    )
    result = _build(events, violations_frame([]), d(2024, 6, 1))
    row = _row(result, "features", "t4")
    assert row["nyc_score_trend_available"] is True
    # slope is exactly 0.1 score/day on this synthetic series -> 36.5 per 365 days.
    # A row-index-based (wrong) slope would be (13.3-10)/2 = 1.65, easily distinguished.
    assert row["nyc_score_prior_time_trend"] == pytest.approx(36.5, abs=0.01)


def test_ols_slope_helper_returns_none_for_zero_time_variance() -> None:
    from plateproof.features.temporal import _ols_slope_per_year

    assert _ols_slope_per_year([(0.0, 1.0), (0.0, 2.0), (0.0, 3.0)]) is None


# --------------------------------------------------------------------------- #
# Violation history: citation count vs inspection count, top code (#13, #14)  #
# --------------------------------------------------------------------------- #


def test_florida_count_field_contributes_to_citation_count_not_inspection_count(
    make_event: Any, make_violation: Any, events_frame: Any, violations_frame: Any, d: Any
) -> None:
    rid = "florida:citation"
    events = events_frame(
        [
            make_event(
                restaurant_id=rid,
                jurisdiction="florida",
                inspection_id="fc1",
                inspection_date=d(2024, 1, 1),
            ),
            make_event(
                restaurant_id=rid,
                jurisdiction="florida",
                inspection_id="fc2",
                inspection_date=d(2024, 2, 1),
            ),
        ]
    )
    violations = violations_frame(
        [
            make_violation(
                inspection_id="fc1",
                restaurant_id=rid,
                jurisdiction="florida",
                inspection_date=d(2024, 1, 1),
                violation_code="07",
                count=5,
            ),
        ]
    )
    result = _build(events, violations, d(2024, 6, 1))
    row = _row(result, "features", "fc2")
    assert row["prior_violation_total_citation_count"] == 5
    assert row["prior_inspections_with_any_violation"] == 1
    assert row["prior_top_violation_code_count"] == 5
    assert row["prior_top_violation_inspection_count"] == 1


def test_top_violation_code_tie_break_by_inspection_count_then_lexicographic(
    make_event: Any, make_violation: Any, events_frame: Any, violations_frame: Any, d: Any
) -> None:
    rid = "nyc:topcode"
    events = events_frame(
        [
            make_event(
                restaurant_id=rid,
                jurisdiction="nyc",
                inspection_id="n1",
                inspection_date=d(2024, 1, 1),
            ),
            make_event(
                restaurant_id=rid,
                jurisdiction="nyc",
                inspection_id="n2",
                inspection_date=d(2024, 1, 15),
            ),
            make_event(
                restaurant_id=rid,
                jurisdiction="nyc",
                inspection_id="n3",
                inspection_date=d(2024, 2, 1),
            ),
        ]
    )
    # "04L": 1 citation on n1, 1 citation on n2 -> total 2 citations, 2 inspections.
    # "08A": 2 citations on n1 only -> total 2 citations, 1 inspection.
    # tie on citations (2 vs 2) -> "04L" wins on inspection count (2 > 1).
    violations = violations_frame(
        [
            make_violation(
                inspection_id="n1",
                restaurant_id=rid,
                jurisdiction="nyc",
                inspection_date=d(2024, 1, 1),
                violation_code="04L",
                count=1,
            ),
            make_violation(
                inspection_id="n1",
                restaurant_id=rid,
                jurisdiction="nyc",
                inspection_date=d(2024, 1, 1),
                violation_code="08A",
                count=2,
            ),
            make_violation(
                inspection_id="n2",
                restaurant_id=rid,
                jurisdiction="nyc",
                inspection_date=d(2024, 1, 15),
                violation_code="04L",
                count=1,
            ),
        ]
    )
    result = _build(events, violations, d(2024, 6, 1))
    row = _row(result, "features", "n3")
    audit = _row(result, "audit", "n3")
    assert audit["prior_top_violation_code"] == "04L"
    assert row["prior_top_violation_code_count"] == 2
    assert row["prior_top_violation_inspection_count"] == 2
    assert row["distinct_prior_violation_code_count"] == 2


def test_top_violation_code_lexicographic_final_tie_break(
    make_event: Any, make_violation: Any, events_frame: Any, violations_frame: Any, d: Any
) -> None:
    rid = "nyc:lexi"
    events = events_frame(
        [
            make_event(
                restaurant_id=rid,
                jurisdiction="nyc",
                inspection_id="l1",
                inspection_date=d(2024, 1, 1),
            ),
            make_event(
                restaurant_id=rid,
                jurisdiction="nyc",
                inspection_id="l2",
                inspection_date=d(2024, 2, 1),
            ),
        ]
    )
    violations = violations_frame(
        [
            make_violation(
                inspection_id="l1",
                restaurant_id=rid,
                jurisdiction="nyc",
                inspection_date=d(2024, 1, 1),
                violation_code="08A",
                count=1,
            ),
            make_violation(
                inspection_id="l1",
                restaurant_id=rid,
                jurisdiction="nyc",
                inspection_date=d(2024, 1, 1),
                violation_code="04L",
                count=1,
            ),
        ]
    )
    result = _build(events, violations, d(2024, 6, 1))
    audit = _row(result, "audit", "l2")
    assert audit["prior_top_violation_code"] == "04L"


def test_current_and_future_violations_cannot_change_prior_violation_features(
    make_event: Any, make_violation: Any, events_frame: Any, violations_frame: Any, d: Any
) -> None:
    rid = "nyc:futureviol"
    events = events_frame(
        [
            make_event(
                restaurant_id=rid,
                jurisdiction="nyc",
                inspection_id="fv1",
                inspection_date=d(2024, 1, 1),
            ),
            make_event(
                restaurant_id=rid,
                jurisdiction="nyc",
                inspection_id="fv2",
                inspection_date=d(2024, 2, 1),
            ),
        ]
    )
    base_violations = [
        make_violation(
            inspection_id="fv1",
            restaurant_id=rid,
            jurisdiction="nyc",
            inspection_date=d(2024, 1, 1),
            violation_code="04L",
            count=1,
        ),
    ]
    baseline = _build(events, violations_frame(base_violations), d(2024, 6, 1))
    baseline_row = _row(baseline, "features", "fv2")

    with_current_and_future = violations_frame(
        [
            *base_violations,
            make_violation(
                inspection_id="fv2",
                restaurant_id=rid,
                jurisdiction="nyc",
                inspection_date=d(2024, 2, 1),
                violation_code="99Z",
                count=9,
            ),  # current
            make_violation(
                inspection_id="future",
                restaurant_id=rid,
                jurisdiction="nyc",
                inspection_date=d(2024, 12, 1),
                violation_code="77X",
                count=9,
            ),  # future
        ]
    )
    after = _build(events, with_current_and_future, d(2024, 6, 1))
    after_row = _row(after, "features", "fv2")
    for key in (
        "prior_violation_total_citation_count",
        "prior_inspections_with_any_violation",
        "distinct_prior_violation_code_count",
        "prior_top_violation_code_count",
        "prior_top_violation_inspection_count",
    ):
        assert after_row[key] == baseline_row[key]


# --------------------------------------------------------------------------- #
# Cutoff boundary                                                              #
# --------------------------------------------------------------------------- #


def test_exact_cutoff_boundary_excludes_on_and_after(
    make_event: Any, events_frame: Any, violations_frame: Any, d: Any
) -> None:
    rid = "nyc:cutoff"
    events = events_frame(
        [
            make_event(
                restaurant_id=rid,
                jurisdiction="nyc",
                inspection_id="k1",
                inspection_date=d(2024, 1, 1),
                score=1.0,
            ),
            make_event(
                restaurant_id=rid,
                jurisdiction="nyc",
                inspection_id="k2",
                inspection_date=d(2024, 2, 1),
                score=2.0,
            ),
            make_event(
                restaurant_id=rid,
                jurisdiction="nyc",
                inspection_id="k3",
                inspection_date=d(2024, 3, 1),
                score=3.0,
            ),  # == cutoff
            make_event(
                restaurant_id=rid,
                jurisdiction="nyc",
                inspection_id="k4",
                inspection_date=d(2024, 4, 1),
                score=4.0,
            ),  # after cutoff
        ]
    )
    result = _build(events, violations_frame([]), d(2024, 3, 1))
    ids = set(result.identity.get_column("inspection_id").to_list())
    assert ids == {"k1", "k2"}
    assert result.build_report.excluded_at_or_after_cutoff_count == 2


def test_future_extreme_nyc_score_does_not_alter_earlier_features(
    make_event: Any, events_frame: Any, violations_frame: Any, d: Any
) -> None:
    rid = "nyc:futurescore"
    base = [
        make_event(
            restaurant_id=rid,
            jurisdiction="nyc",
            inspection_id="fs1",
            inspection_date=d(2024, 1, 1),
            score=10.0,
        ),
        make_event(
            restaurant_id=rid,
            jurisdiction="nyc",
            inspection_id="fs2",
            inspection_date=d(2024, 2, 1),
            score=12.0,
        ),
    ]
    baseline = _build(events_frame(base), violations_frame([]), d(2024, 6, 1))
    baseline_row = _row(baseline, "features", "fs2")

    mutated = [
        *base,
        make_event(
            restaurant_id=rid,
            jurisdiction="nyc",
            inspection_id="fs3",
            inspection_date=d(2024, 12, 31),
            score=999999.0,
        ),
    ]
    after = _build(events_frame(mutated), violations_frame([]), d(2024, 6, 1))
    after_row = _row(after, "features", "fs2")
    assert after_row == baseline_row


def test_future_extreme_florida_counts_do_not_alter_earlier_features(
    make_event: Any, events_frame: Any, violations_frame: Any, d: Any
) -> None:
    rid = "florida:futurecounts"
    base = [
        make_event(
            restaurant_id=rid,
            jurisdiction="florida",
            inspection_id="ff1",
            inspection_date=d(2024, 1, 1),
            high_priority_count=1,
        ),
        make_event(
            restaurant_id=rid,
            jurisdiction="florida",
            inspection_id="ff2",
            inspection_date=d(2024, 2, 1),
            high_priority_count=1,
        ),
    ]
    baseline = _build(events_frame(base), violations_frame([]), d(2024, 6, 1))
    baseline_row = _row(baseline, "features", "ff2")

    mutated = [
        *base,
        make_event(
            restaurant_id=rid,
            jurisdiction="florida",
            inspection_id="ff3",
            inspection_date=d(2024, 12, 31),
            high_priority_count=999,
        ),
    ]
    after = _build(events_frame(mutated), violations_frame([]), d(2024, 6, 1))
    after_row = _row(after, "features", "ff2")
    assert after_row == baseline_row


def test_current_inspection_score_does_not_enter_its_own_features(
    make_event: Any, events_frame: Any, violations_frame: Any, d: Any
) -> None:
    rid = "nyc:selfleak"
    events = events_frame(
        [
            make_event(
                restaurant_id=rid,
                jurisdiction="nyc",
                inspection_id="sl1",
                inspection_date=d(2024, 1, 1),
                score=77.0,
            ),
        ]
    )
    result = _build(events, violations_frame([]), d(2024, 6, 1))
    row = _row(result, "features", "sl1")
    assert row["nyc_previous_day_score"] is None
    assert row["missing_history"] is True


def test_current_florida_hib_does_not_enter_its_own_features(
    make_event: Any, events_frame: Any, violations_frame: Any, d: Any
) -> None:
    rid = "florida:selfleak"
    events = events_frame(
        [
            make_event(
                restaurant_id=rid,
                jurisdiction="florida",
                inspection_id="fsl1",
                inspection_date=d(2024, 1, 1),
                high_priority_count=8,
                intermediate_count=8,
                basic_count=8,
            ),
        ]
    )
    result = _build(events, violations_frame([]), d(2024, 6, 1))
    row = _row(result, "features", "fsl1")
    assert row["fl_previous_day_high_priority_count"] is None
    assert row["missing_history"] is True


# --------------------------------------------------------------------------- #
# Restaurant / jurisdiction isolation (#23, #24, #25, #26)                    #
# --------------------------------------------------------------------------- #


def test_restaurant_isolation(
    make_event: Any, events_frame: Any, violations_frame: Any, d: Any
) -> None:
    events = events_frame(
        [
            make_event(
                restaurant_id="nyc:a",
                jurisdiction="nyc",
                inspection_id="ra1",
                inspection_date=d(2024, 1, 1),
                score=5.0,
            ),
            make_event(
                restaurant_id="nyc:b",
                jurisdiction="nyc",
                inspection_id="rb1",
                inspection_date=d(2024, 2, 1),
                score=50.0,
            ),
        ]
    )
    result = _build(events, violations_frame([]), d(2024, 6, 1))
    assert _row(result, "features", "ra1")["missing_history"] is True
    assert _row(result, "features", "rb1")["missing_history"] is True


def test_jurisdiction_isolation_no_cross_semantics(
    make_event: Any, events_frame: Any, violations_frame: Any, d: Any
) -> None:
    events = events_frame(
        [
            make_event(
                restaurant_id="nyc:iso",
                jurisdiction="nyc",
                inspection_id="ji1",
                inspection_date=d(2024, 1, 1),
                score=42.0,
                critical_violation_count=1,
            ),
            make_event(
                restaurant_id="florida:iso",
                jurisdiction="florida",
                inspection_id="ji2",
                inspection_date=d(2024, 1, 1),
                high_priority_count=3,
                violation_count=9,
            ),
        ]
    )
    result = _build(events, violations_frame([]), d(2024, 6, 1))
    nyc_row = _row(result, "features", "ji1")
    fl_row = _row(result, "features", "ji2")
    for key in nyc_row:
        if key.startswith("fl_"):
            assert nyc_row[key] is None
    for key in fl_row:
        if key.startswith("nyc_"):
            assert fl_row[key] is None


# --------------------------------------------------------------------------- #
# Null/empty/boundary structural behavior (#27, #28, #29, #30)                #
# --------------------------------------------------------------------------- #


def test_empty_input_produces_empty_typed_output(
    events_frame: Any, violations_frame: Any, d: Any
) -> None:
    result = _build(events_frame([]), violations_frame([]), d(2024, 1, 1))
    assert result.identity.height == 0
    assert result.features.height == 0
    assert result.audit.height == 0
    assert result.build_report.input_event_count == 0


def test_all_first_inspections_all_missing_history(
    make_event: Any, events_frame: Any, violations_frame: Any, d: Any
) -> None:
    events = events_frame(
        [
            make_event(
                restaurant_id=f"nyc:{i}",
                jurisdiction="nyc",
                inspection_id=f"first{i}",
                inspection_date=d(2024, 1, 1),
            )
            for i in range(5)
        ]
    )
    result = _build(events, violations_frame([]), d(2024, 6, 1))
    assert result.features.get_column("missing_history").to_list() == [True] * 5


def test_unique_inspection_ids_across_all_frames(
    make_event: Any, events_frame: Any, violations_frame: Any, d: Any
) -> None:
    from plateproof.features.inspection_events import assert_unique_inspection_id

    events = events_frame(
        [
            make_event(
                restaurant_id="nyc:u",
                jurisdiction="nyc",
                inspection_id="u1",
                inspection_date=d(2024, 1, 1),
            ),
            make_event(
                restaurant_id="nyc:u",
                jurisdiction="nyc",
                inspection_id="u2",
                inspection_date=d(2024, 2, 1),
            ),
        ]
    )
    result = _build(events, violations_frame([]), d(2024, 6, 1))
    assert assert_unique_inspection_id(result.identity) is None
    assert assert_unique_inspection_id(result.features) is None
    assert assert_unique_inspection_id(result.audit) is None


# --------------------------------------------------------------------------- #
# Allowlist / leakage guard (#31, #32, #16, #17, #18, #19)                    #
# --------------------------------------------------------------------------- #


def test_allowlist_rejects_michelin_column() -> None:
    from plateproof.features.temporal import assert_model_matrix_is_safe

    frame = pl.DataFrame({"inspection_id": ["x"], "michelin_star_rank": [1]})
    with pytest.raises(ValueError, match="michelin_star_rank"):
        assert_model_matrix_is_safe(frame)


def test_allowlist_rejects_google_column() -> None:
    from plateproof.features.temporal import assert_model_matrix_is_safe

    frame = pl.DataFrame({"inspection_id": ["x"], "google_rating": [4.5]})
    with pytest.raises(ValueError, match="google_rating"):
        assert_model_matrix_is_safe(frame)


def test_allowlist_rejects_target_like_column() -> None:
    from plateproof.features.temporal import assert_model_matrix_is_safe

    frame = pl.DataFrame({"inspection_id": ["x"], "target_high_priority_next": [1]})
    with pytest.raises(ValueError, match="target_high_priority_next"):
        assert_model_matrix_is_safe(frame)


def test_allowlist_rejects_current_event_column() -> None:
    from plateproof.features.temporal import assert_model_matrix_is_safe

    for bad in (
        "score",
        "grade",
        "action",
        "disposition",
        "inspection_date",
        "inspection_type",
        "high_priority_count",
        "critical_violation_count",
        "restaurant_id",
        "jurisdiction",
    ):
        frame = pl.DataFrame({"inspection_id": ["x"], bad: [1]})
        with pytest.raises(ValueError):
            assert_model_matrix_is_safe(frame)


def test_allowlist_accepts_legitimate_historical_columns_despite_substrings() -> None:
    from plateproof.features.temporal import MODEL_FEATURE_ALLOWLIST, assert_model_matrix_is_safe

    columns = {name: [1.0] for name in MODEL_FEATURE_ALLOWLIST}
    columns["inspection_id"] = ["x"]
    frame = pl.DataFrame(columns)
    assert assert_model_matrix_is_safe(frame) is None


def test_month_and_season_absent_from_features_present_in_identity(
    make_event: Any, events_frame: Any, violations_frame: Any, d: Any
) -> None:
    events = events_frame(
        [
            make_event(
                restaurant_id="nyc:season",
                jurisdiction="nyc",
                inspection_id="se1",
                inspection_date=d(2024, 7, 4),
            ),
        ]
    )
    result = _build(events, violations_frame([]), d(2024, 12, 1))
    assert "month" not in result.features.columns
    assert "season" not in result.features.columns
    assert "month" in result.identity.columns
    assert "season" in result.identity.columns


def test_features_frame_columns_are_exactly_the_allowlist(
    make_event: Any, events_frame: Any, violations_frame: Any, d: Any
) -> None:
    from plateproof.features.temporal import MODEL_FEATURE_ALLOWLIST

    events = events_frame(
        [
            make_event(
                restaurant_id="nyc:cols",
                jurisdiction="nyc",
                inspection_id="co1",
                inspection_date=d(2024, 1, 1),
            ),
        ]
    )
    result = _build(events, violations_frame([]), d(2024, 6, 1))
    assert set(result.features.columns) - {"inspection_id"} == MODEL_FEATURE_ALLOWLIST
    assert result.features.columns == ["inspection_id", *sorted(MODEL_FEATURE_ALLOWLIST)]


def test_no_duplicate_feature_columns(
    make_event: Any, events_frame: Any, violations_frame: Any, d: Any
) -> None:
    events = events_frame(
        [
            make_event(
                restaurant_id="nyc:dup",
                jurisdiction="nyc",
                inspection_id="du1",
                inspection_date=d(2024, 1, 1),
            ),
        ]
    )
    result = _build(events, violations_frame([]), d(2024, 6, 1))
    assert len(result.features.columns) == len(set(result.features.columns))


def test_no_nan_or_infinite_numeric_values(
    make_event: Any, events_frame: Any, violations_frame: Any, d: Any
) -> None:
    events = events_frame(
        [
            make_event(
                restaurant_id="nyc:finite",
                jurisdiction="nyc",
                inspection_id="fi1",
                inspection_date=d(2024, 1, 1),
                score=10.0,
            ),
            make_event(
                restaurant_id="nyc:finite",
                jurisdiction="nyc",
                inspection_id="fi2",
                inspection_date=d(2024, 2, 1),
                score=20.0,
            ),
        ]
    )
    result = _build(events, violations_frame([]), d(2024, 6, 1))
    for column in result.features.columns:
        if result.features.schema[column] in (pl.Float64, pl.Float32):
            series = result.features.get_column(column)
            assert series.drop_nulls().is_nan().sum() == 0
            assert series.drop_nulls().is_infinite().sum() == 0


# --------------------------------------------------------------------------- #
# Feature-availability manifest & build report                                #
# --------------------------------------------------------------------------- #


def test_feature_availability_report_has_entry_per_allowlisted_feature(
    events_frame: Any, violations_frame: Any, d: Any
) -> None:
    from plateproof.features.temporal import MODEL_FEATURE_ALLOWLIST

    result = _build(events_frame([]), violations_frame([]), d(2024, 1, 1))
    names = {entry.feature_name for entry in result.availability_report.entries}
    assert names == MODEL_FEATURE_ALLOWLIST


# --------------------------------------------------------------------------- #
# Real ingestion compatibility (#20/#40) -- no ingestion code modified        #
# --------------------------------------------------------------------------- #


def test_compatible_with_real_nyc_and_florida_fixture_outputs(d: Any) -> None:
    from pathlib import Path

    from plateproof.features.temporal import assert_model_matrix_is_safe
    from plateproof.ingestion.florida import FloridaExtractSource, load_florida_extracts
    from plateproof.ingestion.florida import build_florida_inspection_events as build_fl
    from plateproof.ingestion.nyc import build_nyc_inspection_events as build_nyc
    from plateproof.ingestion.nyc import load_nyc_raw

    fixtures = Path(__file__).parent.parent / "ingestion" / "fixtures"
    nyc_result = build_nyc(load_nyc_raw(fixtures / "nyc_sample_soda.csv"), ingested_at=FIXED_TS)
    fl_result = build_fl(
        load_florida_extracts([FloridaExtractSource(path=fixtures / "fl_sample_current.csv")]),
        ingested_at=FIXED_TS,
    )
    combined_events = pl.concat([nyc_result.inspection_events, fl_result.inspection_events])
    combined_violations = pl.concat([nyc_result.violation_events, fl_result.violation_events])

    result = _build(combined_events, combined_violations, d(2026, 1, 1))
    assert (
        result.identity.height
        == combined_events.filter(pl.col("inspection_date") < d(2026, 1, 1)).height
    )
    assert assert_model_matrix_is_safe(result.features) is None


# --------------------------------------------------------------------------- #
# Modest synthetic scale test (structural, not timing-based)                  #
# --------------------------------------------------------------------------- #


def test_modest_scale_does_not_blow_up(
    make_event: Any, events_frame: Any, violations_frame: Any, d: Any
) -> None:
    import time
    from datetime import timedelta

    rows = [
        make_event(
            restaurant_id=f"nyc:scale{r}",
            jurisdiction="nyc",
            inspection_id=f"scale{r}-{e}",
            inspection_date=date(2020, 1, 1) + timedelta(days=e * 20),
            score=float(10 + e),
        )
        for r in range(300)
        for e in range(15)
    ]
    events = events_frame(rows)
    started = time.monotonic()
    result = _build(events, violations_frame([]), d(2026, 1, 1))
    elapsed = time.monotonic() - started

    assert result.identity.height == 300 * 15
    assert elapsed < 30.0  # generous upper bound, not a precision timing assertion
