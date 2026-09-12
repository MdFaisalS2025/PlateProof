"""NYC-specific target builders and feature list.

No target here ever substitutes a letter grade for a missing/conflicted
numeric score, and none reuses Task 5's temporal violation-history logic --
these builders only look at each row's OWN observed outcome.
"""

from __future__ import annotations

import polars as pl
from pydantic import BaseModel, ConfigDict

from plateproof.features.temporal import MODEL_FEATURE_ALLOWLIST

NYC_PRIMARY_QUALIFYING_TYPES = frozenset(
    {
        "Cycle Inspection / Initial Inspection",
        "Pre-permit (Operational) / Initial Inspection",
        "Pre-permit (Non-operational) / Initial Inspection",
    }
)
NYC_PRIMARY_SCORE_THRESHOLD = 14.0
NYC_SCORE_GE_28_THRESHOLD = 28.0

NYC_FEATURE_LIST: tuple[str, ...] = tuple(
    sorted(name for name in MODEL_FEATURE_ALLOWLIST if not name.startswith("fl_"))
)


class NYCTargetBuildReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_row_count: int
    positive_count: int
    negative_count: int
    excluded_non_qualifying_type: int
    excluded_missing_or_conflicted_score: int
    counts_by_inspection_type: dict[str, int]


def _assert_nyc(events: pl.DataFrame) -> None:
    jurisdictions = set(events.get_column("jurisdiction").unique().to_list())
    if jurisdictions - {"nyc"}:
        raise ValueError(f"build_nyc_primary_target requires only nyc rows, found: {jurisdictions}")


def build_nyc_primary_target(events: pl.DataFrame) -> tuple[pl.DataFrame, NYCTargetBuildReport]:
    """label = 1 iff an exactly-qualifying initial inspection's own score is
    >= 14 and unconflicted; row excluded (label omitted) otherwise."""
    _assert_nyc(events)

    counts_by_type = {
        str(k): int(v)
        for k, v in events.group_by("inspection_type").agg(pl.len().alias("n")).iter_rows()
    }

    qualifying = events.filter(pl.col("inspection_type").is_in(NYC_PRIMARY_QUALIFYING_TYPES))
    excluded_non_qualifying_type = events.height - qualifying.height

    valid_score = qualifying.filter((~pl.col("score_conflict")) & pl.col("score").is_not_null())
    excluded_missing_or_conflicted_score = qualifying.height - valid_score.height

    labeled = valid_score.with_columns(
        (pl.col("score") >= NYC_PRIMARY_SCORE_THRESHOLD).cast(pl.Int8).alias("label")
    ).select(["inspection_id", "label"])

    report = NYCTargetBuildReport(
        input_row_count=events.height,
        positive_count=int(labeled.filter(pl.col("label") == 1).height),
        negative_count=int(labeled.filter(pl.col("label") == 0).height),
        excluded_non_qualifying_type=excluded_non_qualifying_type,
        excluded_missing_or_conflicted_score=excluded_missing_or_conflicted_score,
        counts_by_inspection_type=counts_by_type,
    )
    return labeled, report


def build_nyc_score_ge_28_target(events: pl.DataFrame) -> pl.DataFrame:
    """Secondary lightweight target: own score >= 28, same qualifying/validity
    rules as the primary target."""
    _assert_nyc(events)
    qualifying = events.filter(pl.col("inspection_type").is_in(NYC_PRIMARY_QUALIFYING_TYPES))
    valid_score = qualifying.filter((~pl.col("score_conflict")) & pl.col("score").is_not_null())
    return valid_score.with_columns(
        (pl.col("score") >= NYC_SCORE_GE_28_THRESHOLD).cast(pl.Int8).alias("label")
    ).select(["inspection_id", "label"])


def build_nyc_any_critical_target(events: pl.DataFrame) -> pl.DataFrame:
    """Secondary lightweight target: own critical_violation_count >= 1, same
    qualifying-type rule as the primary target (score validity irrelevant)."""
    _assert_nyc(events)
    qualifying = events.filter(pl.col("inspection_type").is_in(NYC_PRIMARY_QUALIFYING_TYPES))
    valid = qualifying.filter(pl.col("critical_violation_count").is_not_null())
    return valid.with_columns(
        (pl.col("critical_violation_count") >= 1).cast(pl.Int8).alias("label")
    ).select(["inspection_id", "label"])
