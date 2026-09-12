"""Florida-specific target builders and feature list.

Florida outcomes are never converted into NYC-style points or grades, and
temporary closure is always treated as a positive (more severe than a
follow-up requirement) -- it must never become a negative example.
"""

from __future__ import annotations

import polars as pl
from pydantic import BaseModel, ConfigDict

from plateproof.features.temporal import MODEL_FEATURE_ALLOWLIST

FL_QUALIFYING_TYPE = "Routine - Food"
FL_INITIAL_VISIT_SEQUENCE = 1
FL_RECOGNIZED_DISPOSITIONS = frozenset({"met_standards", "follow_up_required", "temporary_closure"})
FL_POSITIVE_DISPOSITIONS = frozenset({"follow_up_required", "temporary_closure"})

FL_FEATURE_LIST: tuple[str, ...] = tuple(
    sorted(name for name in MODEL_FEATURE_ALLOWLIST if not name.startswith("nyc_"))
)


class FloridaTargetBuildReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_row_count: int
    positive_count: int
    negative_count: int
    excluded_non_qualifying_type: int
    excluded_not_initial_visit: int
    excluded_unrecognized_disposition: int
    excluded_missing_high_priority: int
    ambiguous_group_count: int


def _assert_florida(events: pl.DataFrame) -> None:
    jurisdictions = set(events.get_column("jurisdiction").unique().to_list())
    if jurisdictions - {"florida"}:
        raise ValueError(
            f"build_florida_primary_target requires only florida rows, found: {jurisdictions}"
        )


def _qualifying_initial_visits(events: pl.DataFrame) -> tuple[pl.DataFrame, int, int, int]:
    qualifying = events.filter(pl.col("inspection_type") == FL_QUALIFYING_TYPE)
    excluded_non_qualifying_type = events.height - qualifying.height

    initial = qualifying.filter(pl.col("native_visit_sequence") == FL_INITIAL_VISIT_SEQUENCE)
    excluded_not_initial_visit = qualifying.height - initial.height

    group_sizes = initial.group_by("native_inspection_group_id").agg(pl.len().alias("n"))
    ambiguous_groups = (
        group_sizes.filter(pl.col("n") > 1).get_column("native_inspection_group_id").to_list()
    )
    ambiguous_group_count = len(ambiguous_groups)
    unambiguous = initial.filter(~pl.col("native_inspection_group_id").is_in(ambiguous_groups))

    return (
        unambiguous,
        excluded_non_qualifying_type,
        excluded_not_initial_visit,
        ambiguous_group_count,
    )


def build_florida_primary_target(
    events: pl.DataFrame,
) -> tuple[pl.DataFrame, FloridaTargetBuildReport]:
    """label = 1 iff the initial routine-food visit has any high-priority
    violation, requires follow-up, or resulted in temporary closure; label = 0
    iff it has zero high-priority violations and a recognized non-positive
    disposition. Ambiguous native_inspection_group_id duplicates (more than
    one row claiming visit_sequence == 1) are excluded and reported, never
    silently resolved."""
    _assert_florida(events)
    unambiguous, excluded_non_qualifying_type, excluded_not_initial_visit, ambiguous_group_count = (
        _qualifying_initial_visits(events)
    )

    has_high_priority = unambiguous.filter(pl.col("high_priority_count").is_not_null())
    excluded_missing_high_priority = unambiguous.height - has_high_priority.height

    recognized = has_high_priority.filter(
        pl.col("disposition_status").is_in(FL_RECOGNIZED_DISPOSITIONS)
    )
    excluded_unrecognized_disposition = has_high_priority.height - recognized.height

    labeled = recognized.with_columns(
        (
            (pl.col("high_priority_count") >= 1)
            | pl.col("disposition_status").is_in(FL_POSITIVE_DISPOSITIONS)
        )
        .cast(pl.Int8)
        .alias("label")
    ).select(["inspection_id", "label"])

    report = FloridaTargetBuildReport(
        input_row_count=events.height,
        positive_count=int(labeled.filter(pl.col("label") == 1).height),
        negative_count=int(labeled.filter(pl.col("label") == 0).height),
        excluded_non_qualifying_type=excluded_non_qualifying_type,
        excluded_not_initial_visit=excluded_not_initial_visit,
        excluded_unrecognized_disposition=excluded_unrecognized_disposition,
        excluded_missing_high_priority=excluded_missing_high_priority,
        ambiguous_group_count=ambiguous_group_count,
    )
    return labeled, report


def build_florida_temporary_closure_target(events: pl.DataFrame) -> pl.DataFrame:
    """Secondary lightweight target: initial routine-food visit resulted in
    temporary closure, independent of high-priority-count validity."""
    _assert_florida(events)
    unambiguous, _, _, _ = _qualifying_initial_visits(events)
    recognized = unambiguous.filter(pl.col("disposition_status").is_in(FL_RECOGNIZED_DISPOSITIONS))
    return recognized.with_columns(
        (pl.col("disposition_status") == "temporary_closure").cast(pl.Int8).alias("label")
    ).select(["inspection_id", "label"])
