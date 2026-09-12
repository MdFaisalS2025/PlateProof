"""Parity tests for build_asof_temporal_features against Task 5's retrospective
build_temporal_features. The as-of ("current standing") snapshot must be
computed by the exact same accumulator arithmetic as training features --
these tests are the guardrail against two implementations silently drifting
apart. No production data; everything here is a tiny synthetic fixture.
"""

from __future__ import annotations

import random
from datetime import date, timedelta
from typing import Any

import polars as pl

# Every model feature column except the identity key -- used to diff
# retrospective vs. as-of rows column-for-column.
from plateproof.features.temporal import MODEL_FEATURE_ALLOWLIST

FEATURE_COLUMNS = sorted(MODEL_FEATURE_ALLOWLIST)


def _asof(events: pl.DataFrame, violations: pl.DataFrame, as_of: date) -> Any:
    from plateproof.features.temporal import build_asof_temporal_features

    return build_asof_temporal_features(events, violations, as_of)


def _retro(events: pl.DataFrame, violations: pl.DataFrame, cutoff: date) -> Any:
    from plateproof.features.temporal import build_temporal_features

    return build_temporal_features(events, violations, cutoff)


def _feature_row(frame: pl.DataFrame, key_col: str, key_value: str) -> dict[str, Any]:
    matches = frame.filter(pl.col(key_col) == key_value)
    assert matches.height == 1, f"expected exactly one row for {key_col}={key_value!r}"
    return matches.to_dicts()[0]


def _assert_feature_parity(retro_row: dict[str, Any], asof_row: dict[str, Any]) -> None:
    for col in FEATURE_COLUMNS:
        assert retro_row[col] == asof_row[col], (
            f"parity mismatch on {col!r}: retrospective={retro_row[col]!r} "
            f"vs as-of={asof_row[col]!r}"
        )


def _remove_inspection(
    events: pl.DataFrame, violations: pl.DataFrame, inspection_id: str
) -> tuple[pl.DataFrame, pl.DataFrame]:
    remaining_events = events.filter(pl.col("inspection_id") != inspection_id)
    remaining_violations = violations.filter(pl.col("inspection_id") != inspection_id)
    return remaining_events, remaining_violations


# --------------------------------------------------------------------------- #
# Requirement: as-of date is explicitly required, never defaulted            #
# --------------------------------------------------------------------------- #


def test_as_of_date_has_no_default() -> None:
    import inspect

    from plateproof.features.temporal import build_asof_temporal_features

    params = inspect.signature(build_asof_temporal_features).parameters
    assert params["as_of_date"].default is inspect.Parameter.empty


# --------------------------------------------------------------------------- #
# NYC parity                                                                  #
# --------------------------------------------------------------------------- #


def test_nyc_parity_with_prior_history(
    make_event: Any, events_frame: Any, violations_frame: Any, d: Any
) -> None:
    rid = "nyc:parity1"
    e1 = make_event(
        restaurant_id=rid,
        jurisdiction="nyc",
        inspection_id="e1",
        inspection_date=d(2024, 1, 10),
        score=10.0,
    )
    e2 = make_event(
        restaurant_id=rid,
        jurisdiction="nyc",
        inspection_id="e2",
        inspection_date=d(2024, 3, 10),
        score=20.0,
        critical_violation_count=2,
    )
    target = make_event(
        restaurant_id=rid,
        jurisdiction="nyc",
        inspection_id="target",
        inspection_date=d(2024, 5, 10),
        score=30.0,
        critical_violation_count=1,
    )
    events = events_frame([e1, e2, target])
    violations = violations_frame([])

    retro = _retro(events, violations, d(2024, 5, 11))
    retro_row = _feature_row(retro.features, "inspection_id", "target")

    reduced_events, reduced_violations = _remove_inspection(events, violations, "target")
    asof = _asof(reduced_events, reduced_violations, d(2024, 5, 10))
    asof_row = _feature_row(asof.features, "restaurant_id", rid)

    _assert_feature_parity(retro_row, asof_row)
    assert asof.identity.filter(pl.col("restaurant_id") == rid)["as_of_date"][0] == d(2024, 5, 10)


def test_nyc_parity_with_conflicted_score(
    make_event: Any, events_frame: Any, violations_frame: Any, d: Any
) -> None:
    rid = "nyc:conflict1"
    # two same-day rows disagree on score -> score_conflict history is None/incomplete
    e1a = make_event(
        restaurant_id=rid,
        jurisdiction="nyc",
        inspection_id="e1a",
        inspection_date=d(2024, 1, 10),
        score=10.0,
        score_conflict=True,
    )
    e1b = make_event(
        restaurant_id=rid,
        jurisdiction="nyc",
        inspection_id="e1b",
        inspection_date=d(2024, 1, 10),
        score=15.0,
        score_conflict=True,
    )
    target = make_event(
        restaurant_id=rid,
        jurisdiction="nyc",
        inspection_id="target",
        inspection_date=d(2024, 3, 10),
        score=30.0,
    )
    events = events_frame([e1a, e1b, target])
    violations = violations_frame([])

    retro = _retro(events, violations, d(2024, 3, 11))
    retro_row = _feature_row(retro.features, "inspection_id", "target")

    reduced_events, reduced_violations = _remove_inspection(events, violations, "target")
    asof = _asof(reduced_events, reduced_violations, d(2024, 3, 10))
    asof_row = _feature_row(asof.features, "restaurant_id", rid)

    _assert_feature_parity(retro_row, asof_row)
    assert asof_row["nyc_previous_day_score"] is None
    assert asof_row["nyc_previous_day_score_complete"] is False


def test_nyc_parity_zero_history(
    make_event: Any, events_frame: Any, violations_frame: Any, d: Any
) -> None:
    rid = "nyc:zero1"
    target = make_event(
        restaurant_id=rid,
        jurisdiction="nyc",
        inspection_id="target",
        inspection_date=d(2024, 5, 10),
        score=30.0,
    )
    events = events_frame([target])
    violations = violations_frame([])

    retro = _retro(events, violations, d(2024, 5, 11))
    retro_row = _feature_row(retro.features, "inspection_id", "target")

    # The restaurant's only event IS the target, dated on the as-of date --
    # the eligibility filter (inspection_date < as_of_date) excludes it from
    # history on its own; the restaurant is still known via `events`.
    asof = _asof(events, violations, d(2024, 5, 10))
    asof_row = _feature_row(asof.features, "restaurant_id", rid)

    _assert_feature_parity(retro_row, asof_row)
    assert asof_row["missing_history"] is True
    assert asof_row["history_depth"] == 0


def test_nyc_top_code_tie_behavior_parity(
    make_event: Any, make_violation: Any, events_frame: Any, violations_frame: Any, d: Any
) -> None:
    rid = "nyc:tie1"
    e1 = make_event(
        restaurant_id=rid,
        jurisdiction="nyc",
        inspection_id="e1",
        inspection_date=d(2024, 1, 10),
        score=10.0,
    )
    target = make_event(
        restaurant_id=rid,
        jurisdiction="nyc",
        inspection_id="target",
        inspection_date=d(2024, 3, 10),
        score=30.0,
    )
    v1 = make_violation(
        inspection_id="e1",
        restaurant_id=rid,
        jurisdiction="nyc",
        inspection_date=d(2024, 1, 10),
        violation_code="04L",
        count=1,
    )
    v2 = make_violation(
        inspection_id="e1",
        restaurant_id=rid,
        jurisdiction="nyc",
        inspection_date=d(2024, 1, 10),
        violation_code="02B",
        count=1,
    )
    events = events_frame([e1, target])
    violations = violations_frame([v1, v2])

    retro = _retro(events, violations, d(2024, 3, 11))
    retro_row = _feature_row(retro.features, "inspection_id", "target")

    reduced_events, reduced_violations = _remove_inspection(events, violations, "target")
    asof = _asof(reduced_events, reduced_violations, d(2024, 3, 10))
    asof_row = _feature_row(asof.features, "restaurant_id", rid)

    _assert_feature_parity(retro_row, asof_row)
    # deterministic tie policy: lexicographically smallest code wins (02B < 04L)
    assert asof.audit.filter(pl.col("restaurant_id") == rid)["prior_top_violation_code"][0] == "02B"


def test_nyc_input_order_determinism(
    make_event: Any, events_frame: Any, violations_frame: Any, d: Any
) -> None:
    rid = "nyc:order1"
    rows = [
        make_event(
            restaurant_id=rid,
            jurisdiction="nyc",
            inspection_id=f"e{i}",
            inspection_date=d(2024, 1, 1) + timedelta(days=10 * i),
            score=float(10 + i),
        )
        for i in range(5)
    ]
    events = events_frame(rows)
    violations = violations_frame([])
    as_of = d(2024, 1, 1) + timedelta(days=10 * 5)

    baseline = _asof(events, violations, as_of)
    baseline_row = _feature_row(baseline.features, "restaurant_id", rid)

    shuffled = list(rows)
    random.Random(42).shuffle(shuffled)
    shuffled_events = events_frame(shuffled)
    shuffled_result = _asof(shuffled_events, violations, as_of)
    shuffled_row = _feature_row(shuffled_result.features, "restaurant_id", rid)

    _assert_feature_parity(baseline_row, shuffled_row)


# --------------------------------------------------------------------------- #
# Florida parity                                                              #
# --------------------------------------------------------------------------- #


def test_florida_parity_with_prior_history(
    make_event: Any, events_frame: Any, violations_frame: Any, d: Any
) -> None:
    rid = "florida:parity1"
    e1 = make_event(
        restaurant_id=rid,
        jurisdiction="florida",
        inspection_id="e1",
        inspection_date=d(2024, 1, 10),
        high_priority_count=1,
        intermediate_count=2,
        basic_count=1,
        violation_count=4,
        disposition_status="met_standards",
    )
    target = make_event(
        restaurant_id=rid,
        jurisdiction="florida",
        inspection_id="target",
        inspection_date=d(2024, 3, 10),
        high_priority_count=0,
        intermediate_count=1,
        basic_count=0,
        violation_count=1,
        disposition_status="met_standards",
    )
    events = events_frame([e1, target])
    violations = violations_frame([])

    retro = _retro(events, violations, d(2024, 3, 11))
    retro_row = _feature_row(retro.features, "inspection_id", "target")

    reduced_events, reduced_violations = _remove_inspection(events, violations, "target")
    asof = _asof(reduced_events, reduced_violations, d(2024, 3, 10))
    asof_row = _feature_row(asof.features, "restaurant_id", rid)

    _assert_feature_parity(retro_row, asof_row)


def test_florida_parity_with_incomplete_counts(
    make_event: Any, events_frame: Any, violations_frame: Any, d: Any
) -> None:
    rid = "florida:incomplete1"
    e1 = make_event(
        restaurant_id=rid,
        jurisdiction="florida",
        inspection_id="e1",
        inspection_date=d(2024, 1, 10),
        high_priority_count=None,
        intermediate_count=2,
        basic_count=1,
        violation_count=None,
        disposition_status="met_standards",
    )
    target = make_event(
        restaurant_id=rid,
        jurisdiction="florida",
        inspection_id="target",
        inspection_date=d(2024, 3, 10),
        high_priority_count=0,
        intermediate_count=0,
        basic_count=0,
        violation_count=0,
        disposition_status="met_standards",
    )
    events = events_frame([e1, target])
    violations = violations_frame([])

    retro = _retro(events, violations, d(2024, 3, 11))
    retro_row = _feature_row(retro.features, "inspection_id", "target")

    reduced_events, reduced_violations = _remove_inspection(events, violations, "target")
    asof = _asof(reduced_events, reduced_violations, d(2024, 3, 10))
    asof_row = _feature_row(asof.features, "restaurant_id", rid)

    _assert_feature_parity(retro_row, asof_row)
    assert asof_row["fl_previous_day_high_priority_count"] is None
    assert asof_row["fl_previous_day_high_priority_count_complete"] is False


def test_florida_parity_with_count_valued_violations(
    make_event: Any, make_violation: Any, events_frame: Any, violations_frame: Any, d: Any
) -> None:
    rid = "florida:countval1"
    e1 = make_event(
        restaurant_id=rid,
        jurisdiction="florida",
        inspection_id="e1",
        inspection_date=d(2024, 1, 10),
        high_priority_count=1,
        intermediate_count=0,
        basic_count=0,
        violation_count=1,
        disposition_status="met_standards",
    )
    target = make_event(
        restaurant_id=rid,
        jurisdiction="florida",
        inspection_id="target",
        inspection_date=d(2024, 3, 10),
        high_priority_count=0,
        intermediate_count=0,
        basic_count=0,
        violation_count=0,
        disposition_status="met_standards",
    )
    v1 = make_violation(
        inspection_id="e1",
        restaurant_id=rid,
        jurisdiction="florida",
        inspection_date=d(2024, 1, 10),
        violation_code="08A",
        count=3,
    )
    events = events_frame([e1, target])
    violations = violations_frame([v1])

    retro = _retro(events, violations, d(2024, 3, 11))
    retro_row = _feature_row(retro.features, "inspection_id", "target")

    reduced_events, reduced_violations = _remove_inspection(events, violations, "target")
    asof = _asof(reduced_events, reduced_violations, d(2024, 3, 10))
    asof_row = _feature_row(asof.features, "restaurant_id", rid)

    _assert_feature_parity(retro_row, asof_row)
    assert asof_row["prior_violation_total_citation_count"] == 3


def test_florida_same_day_prior_inspections_parity(
    make_event: Any, events_frame: Any, violations_frame: Any, d: Any
) -> None:
    rid = "florida:sameday1"
    e1a = make_event(
        restaurant_id=rid,
        jurisdiction="florida",
        inspection_id="e1a",
        inspection_date=d(2024, 1, 10),
        high_priority_count=1,
        intermediate_count=0,
        basic_count=0,
        violation_count=1,
        disposition_status="follow_up_required",
    )
    e1b = make_event(
        restaurant_id=rid,
        jurisdiction="florida",
        inspection_id="e1b",
        inspection_date=d(2024, 1, 10),
        high_priority_count=2,
        intermediate_count=1,
        basic_count=0,
        violation_count=3,
        disposition_status="follow_up_required",
    )
    target = make_event(
        restaurant_id=rid,
        jurisdiction="florida",
        inspection_id="target",
        inspection_date=d(2024, 3, 10),
        high_priority_count=0,
        intermediate_count=0,
        basic_count=0,
        violation_count=0,
        disposition_status="met_standards",
    )
    events = events_frame([e1a, e1b, target])
    violations = violations_frame([])

    retro = _retro(events, violations, d(2024, 3, 11))
    retro_row = _feature_row(retro.features, "inspection_id", "target")

    reduced_events, reduced_violations = _remove_inspection(events, violations, "target")
    asof = _asof(reduced_events, reduced_violations, d(2024, 3, 10))
    asof_row = _feature_row(asof.features, "restaurant_id", rid)

    _assert_feature_parity(retro_row, asof_row)
    # both same-day rows fold additively: hp 1+2=3
    assert asof_row["fl_previous_day_high_priority_count"] == 3


# --------------------------------------------------------------------------- #
# Cross-cutting: exclusion boundary, schema, dtype, jurisdiction isolation    #
# --------------------------------------------------------------------------- #


def test_event_exactly_on_as_of_date_is_excluded(
    make_event: Any, events_frame: Any, violations_frame: Any, d: Any
) -> None:
    rid = "nyc:boundary1"
    on_boundary = make_event(
        restaurant_id=rid,
        jurisdiction="nyc",
        inspection_id="on_boundary",
        inspection_date=d(2024, 5, 10),
        score=999.0,
    )
    events = events_frame([on_boundary])
    result = _asof(events, violations_frame([]), d(2024, 5, 10))
    row = _feature_row(result.features, "restaurant_id", rid)
    assert row["missing_history"] is True
    assert row["history_depth"] == 0
    assert result.build_report.excluded_at_or_after_as_of_count == 1


def test_asof_feature_schema_matches_saved_model_feature_order_and_dtypes(
    make_event: Any, events_frame: Any, violations_frame: Any, d: Any
) -> None:
    from plateproof.features.temporal import FEATURE_SCHEMA
    from plateproof.models.nyc_risk import NYC_FEATURE_LIST

    rid = "nyc:schema1"
    e1 = make_event(
        restaurant_id=rid,
        jurisdiction="nyc",
        inspection_id="e1",
        inspection_date=d(2024, 1, 10),
        score=10.0,
    )
    events = events_frame([e1])
    result = _asof(events, violations_frame([]), d(2024, 3, 1))

    matrix = result.features.select(list(NYC_FEATURE_LIST))
    assert matrix.columns == list(NYC_FEATURE_LIST)
    for name in NYC_FEATURE_LIST:
        assert matrix.schema[name] == FEATURE_SCHEMA[name]


def test_asof_never_produces_florida_columns_for_nyc_restaurant(
    make_event: Any, events_frame: Any, violations_frame: Any, d: Any
) -> None:
    rid = "nyc:isolation1"
    e1 = make_event(
        restaurant_id=rid,
        jurisdiction="nyc",
        inspection_id="e1",
        inspection_date=d(2024, 1, 10),
        score=10.0,
    )
    result = _asof(events_frame([e1]), violations_frame([]), d(2024, 3, 1))
    row = _feature_row(result.features, "restaurant_id", rid)
    for name in row:
        if name.startswith("fl_"):
            assert row[name] is None


def test_asof_one_row_per_restaurant(
    make_event: Any, events_frame: Any, violations_frame: Any, d: Any
) -> None:
    r1 = make_event(
        restaurant_id="nyc:multi1",
        jurisdiction="nyc",
        inspection_id="a",
        inspection_date=d(2024, 1, 1),
        score=10.0,
    )
    r2a = make_event(
        restaurant_id="nyc:multi2",
        jurisdiction="nyc",
        inspection_id="b1",
        inspection_date=d(2024, 1, 1),
        score=10.0,
    )
    r2b = make_event(
        restaurant_id="nyc:multi2",
        jurisdiction="nyc",
        inspection_id="b2",
        inspection_date=d(2024, 2, 1),
        score=12.0,
    )
    result = _asof(events_frame([r1, r2a, r2b]), violations_frame([]), d(2024, 3, 1))
    assert result.features.height == 2
    assert set(result.identity.get_column("restaurant_id").to_list()) == {
        "nyc:multi1",
        "nyc:multi2",
    }


def test_asof_no_michelin_or_google_columns_in_matrix(
    make_event: Any, events_frame: Any, violations_frame: Any, d: Any
) -> None:
    from plateproof.features.temporal import assert_model_matrix_is_safe

    e1 = make_event(
        restaurant_id="nyc:safe1",
        jurisdiction="nyc",
        inspection_id="e1",
        inspection_date=d(2024, 1, 10),
        score=10.0,
    )
    result = _asof(events_frame([e1]), violations_frame([]), d(2024, 3, 1))
    assert_model_matrix_is_safe(result.features.drop("restaurant_id"))


def test_build_temporal_features_unchanged_after_refactor(
    make_event: Any, events_frame: Any, violations_frame: Any, d: Any
) -> None:
    """Sanity check that the extraction didn't alter retrospective output for
    a case exercising every accumulator (NYC + Florida separately, since a
    frame is single-jurisdiction per restaurant but this checks both paths
    still run through the shared helper identically)."""
    rid = "nyc:sanity1"
    e1 = make_event(
        restaurant_id=rid,
        jurisdiction="nyc",
        inspection_id="e1",
        inspection_date=d(2024, 1, 10),
        score=10.0,
        critical_violation_count=1,
    )
    e2 = make_event(
        restaurant_id=rid,
        jurisdiction="nyc",
        inspection_id="e2",
        inspection_date=d(2024, 2, 10),
        score=20.0,
        critical_violation_count=0,
    )
    events = events_frame([e1, e2])
    result = _retro(events, violations_frame([]), d(2024, 6, 1))
    row2 = _feature_row(result.features, "inspection_id", "e2")
    assert row2["history_depth"] == 1
    assert row2["nyc_previous_day_score"] == 10.0
    assert row2["nyc_prior_valid_score_count"] == 1
