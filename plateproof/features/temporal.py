"""Leakage-safe temporal features for the NYC and Florida inspection pipelines.

``build_temporal_features`` turns the shared, jurisdiction-neutral inspection
and violation event tables (see ``plateproof.features.inspection_events``)
into three separate, ``inspection_id``-aligned outputs:

* ``identity`` -- who/when/where this prediction event is (never a model input).
* ``features`` -- the model-eligible historical feature matrix, guarded by an
  explicit allowlist (``MODEL_FEATURE_ALLOWLIST``).
* ``audit`` -- descriptive historical fields (e.g. a previous grade) kept for
  human inspection, never fed to a model.

Every feature is computed from information available *strictly before* the
prediction event: same-day events are collapsed into one date-level snapshot
before any shift/window operation, so multiple same-day inspections can never
become history for one another regardless of input order, visit sequence, or
inspection id. ``cutoff_date`` is an exclusive data-availability boundary:
prediction events must have ``inspection_date < cutoff_date``, and no source
event/violation at or after ``cutoff_date`` may ever contribute to history.

No prediction target, no Michelin data, no Google data, and no current-event
outcome field is ever produced here. NYC and Florida native measures are kept
in separate, distinctly named columns -- never blended into a common score.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any, Literal

import polars as pl
from pydantic import BaseModel, ConfigDict

from plateproof.features.inspection_events import (
    finalize_event_frame,
)

MIN_VALID_FOR_VARIANCE = 2
MIN_VALID_FOR_TREND = 3

_KNOWN_FL_DISPOSITION_STATUSES = ("met_standards", "follow_up_required", "temporary_closure")


# --------------------------------------------------------------------------- #
# Public schemas / typed outputs                                              #
# --------------------------------------------------------------------------- #

MODEL_FEATURE_ALLOWLIST: frozenset[str] = frozenset(
    {
        # shared
        "history_depth",
        "prior_inspection_day_count",
        "days_since_previous_inspection_date",
        "missing_history",
        # shared violation history
        "prior_violation_total_citation_count",
        "prior_inspections_with_any_violation",
        "distinct_prior_violation_code_count",
        "prior_top_violation_code_count",
        "prior_top_violation_inspection_count",
        # NYC
        "nyc_previous_day_score",
        "nyc_previous_day_score_complete",
        "nyc_prior_valid_score_count",
        "nyc_score_prior_mean",
        "nyc_score_prior_max",
        "nyc_score_prior_variance",
        "nyc_score_variance_available",
        "nyc_score_prior_time_trend",
        "nyc_score_trend_available",
        "nyc_previous_day_critical_violation_count",
        "nyc_previous_day_critical_violation_count_complete",
        "nyc_prior_critical_violation_total",
        # Florida
        "fl_previous_day_high_priority_count",
        "fl_previous_day_high_priority_count_complete",
        "fl_previous_day_intermediate_count",
        "fl_previous_day_intermediate_count_complete",
        "fl_previous_day_basic_count",
        "fl_previous_day_basic_count_complete",
        "fl_previous_day_total_violation_count",
        "fl_previous_day_total_violation_count_complete",
        "fl_total_violation_prior_mean",
        "fl_total_violation_prior_max",
        "fl_total_violation_prior_variance",
        "fl_total_violation_variance_available",
        "fl_total_violation_prior_time_trend",
        "fl_total_violation_trend_available",
        "fl_high_priority_prior_mean",
        "fl_high_priority_prior_max",
        "fl_prior_inspections_with_high_priority",
        "fl_prior_high_priority_total",
        "fl_prior_valid_high_priority_count",
        "fl_prior_valid_intermediate_count",
        "fl_prior_valid_basic_count",
        "fl_prior_valid_total_count",
        "fl_prior_follow_up_required_count",
        "fl_prior_temporary_closure_count",
    }
)

_FEATURE_ORDER: list[str] = ["inspection_id", *sorted(MODEL_FEATURE_ALLOWLIST)]

_INT_FEATURES = {
    "history_depth",
    "prior_inspection_day_count",
    "days_since_previous_inspection_date",
    "prior_violation_total_citation_count",
    "prior_inspections_with_any_violation",
    "distinct_prior_violation_code_count",
    "prior_top_violation_code_count",
    "prior_top_violation_inspection_count",
    "nyc_prior_valid_score_count",
    "nyc_previous_day_critical_violation_count",
    "nyc_prior_critical_violation_total",
    "fl_previous_day_high_priority_count",
    "fl_previous_day_intermediate_count",
    "fl_previous_day_basic_count",
    "fl_previous_day_total_violation_count",
    "fl_prior_inspections_with_high_priority",
    "fl_prior_high_priority_total",
    "fl_prior_valid_high_priority_count",
    "fl_prior_valid_intermediate_count",
    "fl_prior_valid_basic_count",
    "fl_prior_valid_total_count",
    "fl_prior_follow_up_required_count",
    "fl_prior_temporary_closure_count",
}
_BOOL_FEATURES = {
    "missing_history",
    "nyc_previous_day_score_complete",
    "nyc_score_variance_available",
    "nyc_score_trend_available",
    "nyc_previous_day_critical_violation_count_complete",
    "fl_previous_day_high_priority_count_complete",
    "fl_previous_day_intermediate_count_complete",
    "fl_previous_day_basic_count_complete",
    "fl_previous_day_total_violation_count_complete",
    "fl_total_violation_variance_available",
    "fl_total_violation_trend_available",
}


def _feature_dtype(name: str) -> pl.DataType:
    if name == "inspection_id":
        return pl.String()
    if name in _INT_FEATURES:
        return pl.Int32()
    if name in _BOOL_FEATURES:
        return pl.Boolean()
    return pl.Float64()


FEATURE_SCHEMA: dict[str, pl.DataType] = {name: _feature_dtype(name) for name in _FEATURE_ORDER}

IDENTITY_SCHEMA: dict[str, pl.DataType] = {
    "inspection_id": pl.String(),
    "restaurant_id": pl.String(),
    "jurisdiction": pl.String(),
    "inspection_date": pl.Date(),
    "previous_inspection_date": pl.Date(),
    "inspection_type": pl.String(),
    "dba": pl.String(),
    "cuisine_description": pl.String(),
    "boro_raw": pl.String(),
    "zipcode": pl.String(),
    "month": pl.Int8(),
    "season": pl.String(),
}

AUDIT_SCHEMA: dict[str, pl.DataType] = {
    "inspection_id": pl.String(),
    "nyc_previous_day_grade": pl.String(),
    "fl_previous_day_disposition_status": pl.String(),
    "prior_top_violation_code": pl.String(),
}


class FeatureManifestEntry(BaseModel):
    """One row of the feature-availability manifest -- see module docstring."""

    model_config = ConfigDict(frozen=True)

    feature_name: str
    jurisdiction: Literal["nyc", "florida", "shared"]
    source_columns: list[str]
    historical_window: str
    min_valid_observations: int
    null_behavior: str
    temporal_safety_explanation: str
    production_availability_explanation: str
    data_type: str
    model_eligible: bool = True


class FeatureAvailabilityReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    entries: list[FeatureManifestEntry]


class TemporalFeatureBuildReport(BaseModel):
    """Deterministic, auditable summary of one ``build_temporal_features`` run."""

    model_config = ConfigDict(frozen=True)

    input_event_count: int
    input_violation_count: int
    eligible_prediction_event_count: int
    excluded_at_or_after_cutoff_count: int
    nyc_same_day_score_incomplete_day_count: int
    nyc_same_day_score_conflict_day_count: int
    fl_same_day_incomplete_day_counts: dict[str, int]
    fl_unknown_disposition_event_count: int
    cutoff_date: date
    generated_at: datetime


@dataclass(frozen=True)
class TemporalFeatureResult:
    identity: pl.DataFrame
    features: pl.DataFrame
    audit: pl.DataFrame
    availability_report: FeatureAvailabilityReport
    build_report: TemporalFeatureBuildReport


# --------------------------------------------------------------------------- #
# Leakage guard                                                               #
# --------------------------------------------------------------------------- #

_SUSPICIOUS_TOKENS = (
    "target",
    "label",
    "outcome",
    "adjudication",
    "enforcement",
    "michelin",
    "google",
    "next_inspection",
    "future_",
)


def assert_model_matrix_is_safe(features: pl.DataFrame) -> None:
    """Primary control: every non-key column must be in ``MODEL_FEATURE_ALLOWLIST``.

    Secondary, defense-in-depth control: reject any column whose name contains
    a token strongly associated with a target/current-event/third-party field,
    even if it were somehow allowlisted. This never rejects a legitimate
    historical field merely for containing a word like "score" or
    "high_priority" -- only the closed allowlist decides that.
    """
    columns = [c for c in features.columns if c != "inspection_id"]
    disallowed = sorted(set(columns) - MODEL_FEATURE_ALLOWLIST)
    if disallowed:
        raise ValueError(f"model feature frame contains non-allowlisted column(s): {disallowed}")
    for column in columns:
        lowered = column.lower()
        if any(token in lowered for token in _SUSPICIOUS_TOKENS):
            raise ValueError(
                f"model feature frame column '{column}' matches a forbidden token "
                "even though allowlisted -- refusing"
            )


# --------------------------------------------------------------------------- #
# Small pure helpers                                                          #
# --------------------------------------------------------------------------- #


def _season_for_month(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "fall"


def _ols_slope_per_year(points: list[tuple[float, float]]) -> float | None:
    """OLS slope of y against elapsed-time x, scaled to change per 365 days.

    Requires at least MIN_VALID_FOR_TREND points and nonzero x-variance.
    Deterministic; never returns NaN/infinity (returns None instead).
    """
    n = len(points)
    if n < MIN_VALID_FOR_TREND:
        return None
    sum_x = sum(x for x, _ in points)
    sum_y = sum(y for _, y in points)
    sum_xy = sum(x * y for x, y in points)
    sum_xx = sum(x * x for x, _ in points)
    denom = n * sum_xx - sum_x * sum_x
    if denom == 0:
        return None
    slope_per_day = (n * sum_xy - sum_x * sum_y) / denom
    result = slope_per_day * 365.0
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _select_top_code(code_stats: dict[str, dict[str, Any]]) -> str | None:
    """Deterministic tie policy: greatest citations, then greatest prior
    inspection count, then lexicographically smallest normalized code."""
    if not code_stats:
        return None

    def sort_key(code: str) -> tuple[int, int, str]:
        stats = code_stats[code]
        return (-stats["citations"], -len(stats["inspection_ids"]), code)

    return min(code_stats, key=sort_key)


@dataclass
class _MeasureAccumulator:
    """Expanding, leakage-safe accumulator for one additive or non-additive
    historical measure (NYC score, NYC critical count, or one Florida count).

    ``fold_day`` is called once per (restaurant, date) group, in date order,
    strictly after that date's features have already been emitted using the
    accumulator's state as of *before* the call -- this is what guarantees
    same-day isolation.
    """

    additive: bool
    valid_count: int = 0
    total_sum: float = 0.0
    total_sumsq: float = 0.0
    max_value: float | None = None
    trend_points: list[tuple[float, float]] = field(default_factory=list)
    prev_day_value: float | None = None
    prev_day_complete: bool = False
    has_any_history: bool = False

    def snapshot(self) -> dict[str, Any]:
        mean = self.total_sum / self.valid_count if self.valid_count else None
        variance: float | None = None
        if self.valid_count >= MIN_VALID_FOR_VARIANCE:
            raw_variance = (
                self.total_sumsq - self.total_sum * self.total_sum / self.valid_count
            ) / (self.valid_count - 1)
            variance = max(raw_variance, 0.0)
        slope = _ols_slope_per_year(self.trend_points)
        return {
            "previous_day_value": self.prev_day_value if self.has_any_history else None,
            "previous_day_complete": self.prev_day_complete if self.has_any_history else False,
            "valid_count": self.valid_count,
            "sum": self.total_sum if self.valid_count else None,
            "mean": mean,
            "max": self.max_value,
            "variance": variance,
            "variance_available": variance is not None,
            "trend": slope,
            "trend_available": slope is not None,
        }

    def fold_day(self, elapsed_days: float, values: list[float | None]) -> bool:
        """Fold one day's contributing values. Returns True iff the day's
        aggregate was complete (every contributing event had a usable value,
        and -- for non-additive measures -- they all agreed)."""
        self.has_any_history = True
        if not values or any(v is None for v in values):
            self.prev_day_value = None
            self.prev_day_complete = False
            return False
        non_null: list[float] = [v for v in values if v is not None]
        if self.additive:
            day_value: float | None = float(sum(non_null))
        else:
            distinct = {round(v, 9) for v in non_null}
            day_value = non_null[0] if len(distinct) == 1 else None
        if day_value is None:
            self.prev_day_value = None
            self.prev_day_complete = False
            return False
        self.prev_day_value = day_value
        self.prev_day_complete = True
        self.valid_count += 1
        self.total_sum += day_value
        self.total_sumsq += day_value * day_value
        self.max_value = day_value if self.max_value is None else max(self.max_value, day_value)
        self.trend_points.append((elapsed_days, day_value))
        return True


# --------------------------------------------------------------------------- #
# Main entry point                                                            #
# --------------------------------------------------------------------------- #


def build_temporal_features(
    events: pl.DataFrame,
    violations: pl.DataFrame,
    cutoff_date: date,
) -> TemporalFeatureResult:
    """Build leakage-safe historical features for every eligible inspection event.

    ``cutoff_date`` is an exclusive data-availability boundary: only events with
    ``inspection_date < cutoff_date`` become prediction events, and no event or
    violation with ``inspection_date >= cutoff_date`` ever contributes to any
    row's history. History is computed on a per-(restaurant, date) snapshot
    before being joined back, so same-day events never see one another.
    """
    now = datetime.now(UTC)
    input_event_count = events.height
    input_violation_count = violations.height

    eligible_events = events.filter(pl.col("inspection_date") < cutoff_date)
    eligible_violations = violations.filter(pl.col("inspection_date") < cutoff_date)
    excluded_count = input_event_count - eligible_events.height

    identity_records: list[dict[str, Any]] = []
    feature_records: list[dict[str, Any]] = []
    audit_records: list[dict[str, Any]] = []

    nyc_same_day_incomplete = 0
    nyc_same_day_conflict = 0
    fl_same_day_incomplete: dict[str, int] = {
        "high_priority": 0,
        "intermediate": 0,
        "basic": 0,
        "total": 0,
    }
    fl_unknown_disposition_event_count = 0

    if eligible_events.height:
        sorted_events = eligible_events.sort(["restaurant_id", "inspection_date", "inspection_id"])
        events_by_restaurant: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in sorted_events.to_dicts():
            events_by_restaurant[row["restaurant_id"]].append(row)

        violations_by_key: dict[tuple[str, date], list[dict[str, Any]]] = defaultdict(list)
        if eligible_violations.height:
            for row in eligible_violations.sort(["restaurant_id", "inspection_date"]).to_dicts():
                violations_by_key[(row["restaurant_id"], row["inspection_date"])].append(row)

        for restaurant_id, rows in events_by_restaurant.items():
            jurisdiction = rows[0]["jurisdiction"]
            by_date: dict[date, list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                by_date[row["inspection_date"]].append(row)
            sorted_dates = sorted(by_date)
            first_date = sorted_dates[0]

            history_depth = 0
            prior_day_count = 0
            previous_date: date | None = None

            nyc_score_acc = _MeasureAccumulator(additive=False)
            nyc_critical_acc = _MeasureAccumulator(additive=True)
            nyc_prev_day_grade: str | None = None

            fl_high_acc = _MeasureAccumulator(additive=True)
            fl_inter_acc = _MeasureAccumulator(additive=True)
            fl_basic_acc = _MeasureAccumulator(additive=True)
            fl_total_acc = _MeasureAccumulator(additive=True)
            fl_inspections_with_hp = 0
            fl_follow_up_total = 0
            fl_temp_closure_total = 0
            fl_prev_day_disposition: str | None = None

            viol_total_citations = 0
            viol_inspection_ids_seen: set[str] = set()
            code_stats: dict[str, dict[str, Any]] = {}
            any_violation_history = False

            for current_date in sorted_dates:
                today_rows = by_date[current_date]
                today_violations = violations_by_key.get((restaurant_id, current_date), [])

                nyc_snap = nyc_score_acc.snapshot()
                nyc_crit_snap = nyc_critical_acc.snapshot()
                fl_high_snap = fl_high_acc.snapshot()
                fl_inter_snap = fl_inter_acc.snapshot()
                fl_basic_snap = fl_basic_acc.snapshot()
                fl_total_snap = fl_total_acc.snapshot()
                top_code = _select_top_code(code_stats)

                for row in today_rows:
                    inspection_id = row["inspection_id"]
                    days_since = (current_date - previous_date).days if previous_date else None

                    identity_records.append(
                        {
                            "inspection_id": inspection_id,
                            "restaurant_id": restaurant_id,
                            "jurisdiction": jurisdiction,
                            "inspection_date": current_date,
                            "previous_inspection_date": previous_date,
                            "inspection_type": row.get("inspection_type"),
                            "dba": row.get("dba"),
                            "cuisine_description": row.get("cuisine_description"),
                            "boro_raw": row.get("boro_raw"),
                            "zipcode": row.get("zipcode"),
                            "month": current_date.month,
                            "season": _season_for_month(current_date.month),
                        }
                    )

                    feature_records.append(
                        {
                            "inspection_id": inspection_id,
                            "history_depth": history_depth,
                            "prior_inspection_day_count": prior_day_count,
                            "days_since_previous_inspection_date": days_since,
                            "missing_history": history_depth == 0,
                            "prior_violation_total_citation_count": (
                                viol_total_citations if any_violation_history else None
                            ),
                            "prior_inspections_with_any_violation": (
                                len(viol_inspection_ids_seen) if any_violation_history else None
                            ),
                            "distinct_prior_violation_code_count": (
                                len(code_stats) if any_violation_history else None
                            ),
                            "prior_top_violation_code_count": (
                                code_stats[top_code]["citations"] if top_code else None
                            ),
                            "prior_top_violation_inspection_count": (
                                len(code_stats[top_code]["inspection_ids"]) if top_code else None
                            ),
                            "nyc_previous_day_score": nyc_snap["previous_day_value"],
                            "nyc_previous_day_score_complete": nyc_snap["previous_day_complete"],
                            "nyc_prior_valid_score_count": nyc_snap["valid_count"],
                            "nyc_score_prior_mean": nyc_snap["mean"],
                            "nyc_score_prior_max": nyc_snap["max"],
                            "nyc_score_prior_variance": nyc_snap["variance"],
                            "nyc_score_variance_available": nyc_snap["variance_available"],
                            "nyc_score_prior_time_trend": nyc_snap["trend"],
                            "nyc_score_trend_available": nyc_snap["trend_available"],
                            "nyc_previous_day_critical_violation_count": nyc_crit_snap[
                                "previous_day_value"
                            ],
                            "nyc_previous_day_critical_violation_count_complete": nyc_crit_snap[
                                "previous_day_complete"
                            ],
                            "nyc_prior_critical_violation_total": (
                                int(nyc_crit_snap["sum"])
                                if nyc_crit_snap["sum"] is not None
                                else None
                            ),
                            "fl_previous_day_high_priority_count": fl_high_snap[
                                "previous_day_value"
                            ],
                            "fl_previous_day_high_priority_count_complete": fl_high_snap[
                                "previous_day_complete"
                            ],
                            "fl_previous_day_intermediate_count": fl_inter_snap[
                                "previous_day_value"
                            ],
                            "fl_previous_day_intermediate_count_complete": fl_inter_snap[
                                "previous_day_complete"
                            ],
                            "fl_previous_day_basic_count": fl_basic_snap["previous_day_value"],
                            "fl_previous_day_basic_count_complete": fl_basic_snap[
                                "previous_day_complete"
                            ],
                            "fl_previous_day_total_violation_count": fl_total_snap[
                                "previous_day_value"
                            ],
                            "fl_previous_day_total_violation_count_complete": fl_total_snap[
                                "previous_day_complete"
                            ],
                            "fl_total_violation_prior_mean": fl_total_snap["mean"],
                            "fl_total_violation_prior_max": fl_total_snap["max"],
                            "fl_total_violation_prior_variance": fl_total_snap["variance"],
                            "fl_total_violation_variance_available": fl_total_snap[
                                "variance_available"
                            ],
                            "fl_total_violation_prior_time_trend": fl_total_snap["trend"],
                            "fl_total_violation_trend_available": fl_total_snap["trend_available"],
                            "fl_high_priority_prior_mean": fl_high_snap["mean"],
                            "fl_high_priority_prior_max": fl_high_snap["max"],
                            "fl_prior_inspections_with_high_priority": (
                                fl_inspections_with_hp
                                if jurisdiction == "florida" and history_depth
                                else None
                            ),
                            "fl_prior_high_priority_total": (
                                int(fl_high_snap["sum"])
                                if fl_high_snap["sum"] is not None
                                else None
                            ),
                            "fl_prior_valid_high_priority_count": fl_high_snap["valid_count"],
                            "fl_prior_valid_intermediate_count": fl_inter_snap["valid_count"],
                            "fl_prior_valid_basic_count": fl_basic_snap["valid_count"],
                            "fl_prior_valid_total_count": fl_total_snap["valid_count"],
                            "fl_prior_follow_up_required_count": (
                                fl_follow_up_total
                                if jurisdiction == "florida" and history_depth
                                else None
                            ),
                            "fl_prior_temporary_closure_count": (
                                fl_temp_closure_total
                                if jurisdiction == "florida" and history_depth
                                else None
                            ),
                        }
                    )

                    feature_dict = feature_records[-1]
                    for key in list(feature_dict):
                        if jurisdiction != "nyc" and key.startswith("nyc_"):
                            feature_dict[key] = None
                        if jurisdiction != "florida" and key.startswith("fl_"):
                            feature_dict[key] = None

                    audit_records.append(
                        {
                            "inspection_id": inspection_id,
                            "nyc_previous_day_grade": nyc_prev_day_grade
                            if jurisdiction == "nyc"
                            else None,
                            "fl_previous_day_disposition_status": (
                                fl_prev_day_disposition if jurisdiction == "florida" else None
                            ),
                            "prior_top_violation_code": top_code,
                        }
                    )

                # --- fold today's contribution into running state ------------
                elapsed_days = float((current_date - first_date).days)

                if jurisdiction == "nyc":
                    score_values = [
                        None if r.get("score_conflict") else r.get("score") for r in today_rows
                    ]
                    was_complete = nyc_score_acc.fold_day(elapsed_days, score_values)
                    if not was_complete:
                        non_null = [v for v in score_values if v is not None]
                        if len(non_null) == len(score_values) and len(score_values) > 1:
                            nyc_same_day_conflict += 1
                        else:
                            nyc_same_day_incomplete += 1
                    critical_values = [r.get("critical_violation_count") for r in today_rows]
                    nyc_critical_acc.fold_day(elapsed_days, critical_values)

                    trusted_grades = [
                        r.get("grade")
                        for r in today_rows
                        if r.get("grade") and not r.get("grade_conflict")
                    ]
                    nyc_prev_day_grade = (
                        trusted_grades[0] if len(set(trusted_grades)) == 1 else None
                    )

                if jurisdiction == "florida":
                    hp_values = [r.get("high_priority_count") for r in today_rows]
                    fl_high_acc.fold_day(elapsed_days, hp_values)
                    if any(v is None for v in hp_values):
                        fl_same_day_incomplete["high_priority"] += 1

                    inter_values = [r.get("intermediate_count") for r in today_rows]
                    fl_inter_acc.fold_day(elapsed_days, inter_values)
                    if any(v is None for v in inter_values):
                        fl_same_day_incomplete["intermediate"] += 1

                    basic_values = [r.get("basic_count") for r in today_rows]
                    fl_basic_acc.fold_day(elapsed_days, basic_values)
                    if any(v is None for v in basic_values):
                        fl_same_day_incomplete["basic"] += 1

                    total_values = [r.get("violation_count") for r in today_rows]
                    fl_total_acc.fold_day(elapsed_days, total_values)
                    if any(v is None for v in total_values):
                        fl_same_day_incomplete["total"] += 1

                    for r in today_rows:
                        hp = r.get("high_priority_count")
                        if hp is not None and hp > 0:
                            fl_inspections_with_hp += 1
                        status = r.get("disposition_status")
                        if status == "follow_up_required":
                            fl_follow_up_total += 1
                        elif status == "temporary_closure":
                            fl_temp_closure_total += 1
                        elif status not in _KNOWN_FL_DISPOSITION_STATUSES:
                            fl_unknown_disposition_event_count += 1

                    distinct_dispositions = {
                        r.get("disposition_status")
                        for r in today_rows
                        if r.get("disposition_status")
                    }
                    fl_prev_day_disposition = (
                        next(iter(distinct_dispositions))
                        if len(distinct_dispositions) == 1
                        else None
                    )

                for v in today_violations:
                    code = v.get("violation_code")
                    viol_inspection_id = v.get("inspection_id")
                    if not code or not viol_inspection_id:
                        continue
                    count = v.get("count") or 0
                    stats = code_stats.setdefault(code, {"citations": 0, "inspection_ids": set()})
                    stats["citations"] += count
                    stats["inspection_ids"].add(viol_inspection_id)
                    viol_total_citations += count
                    viol_inspection_ids_seen.add(viol_inspection_id)
                    any_violation_history = True
                if today_violations:
                    any_violation_history = True

                history_depth += len(today_rows)
                prior_day_count += 1
                previous_date = current_date

    identity = finalize_event_frame(identity_records, IDENTITY_SCHEMA).sort("inspection_id")
    features = finalize_event_frame(feature_records, FEATURE_SCHEMA).sort("inspection_id")
    audit = finalize_event_frame(audit_records, AUDIT_SCHEMA).sort("inspection_id")

    assert_model_matrix_is_safe(features)

    build_report = TemporalFeatureBuildReport(
        input_event_count=input_event_count,
        input_violation_count=input_violation_count,
        eligible_prediction_event_count=identity.height,
        excluded_at_or_after_cutoff_count=excluded_count,
        nyc_same_day_score_incomplete_day_count=nyc_same_day_incomplete,
        nyc_same_day_score_conflict_day_count=nyc_same_day_conflict,
        fl_same_day_incomplete_day_counts=fl_same_day_incomplete,
        fl_unknown_disposition_event_count=fl_unknown_disposition_event_count,
        cutoff_date=cutoff_date,
        generated_at=now,
    )
    availability_report = _build_availability_report()

    return TemporalFeatureResult(
        identity=identity,
        features=features,
        audit=audit,
        availability_report=availability_report,
        build_report=build_report,
    )


def _build_availability_report() -> FeatureAvailabilityReport:
    entries = [FeatureManifestEntry(**meta) for meta in _FEATURE_MANIFEST_METADATA]
    names = {entry.feature_name for entry in entries}
    assert names == MODEL_FEATURE_ALLOWLIST, "feature manifest drifted from MODEL_FEATURE_ALLOWLIST"
    return FeatureAvailabilityReport(entries=entries)


def _entry(
    name: str,
    jurisdiction: Literal["nyc", "florida", "shared"],
    source_columns: list[str],
    window: str,
    min_valid: int,
    null_behavior: str,
    safety: str,
    production: str,
    dtype: str,
) -> dict[str, Any]:
    return {
        "feature_name": name,
        "jurisdiction": jurisdiction,
        "source_columns": source_columns,
        "historical_window": window,
        "min_valid_observations": min_valid,
        "null_behavior": null_behavior,
        "temporal_safety_explanation": safety,
        "production_availability_explanation": production,
        "data_type": dtype,
    }


_SAFE_ALL = "Computed only from events/violations strictly before the prediction event's date."
_PROD_ALL = "Available at prediction time: derived only from the restaurant's own recorded past."

_FEATURE_MANIFEST_METADATA: list[dict[str, Any]] = [
    _entry(
        "history_depth",
        "shared",
        ["inspection_date"],
        "all strictly-prior eligible events",
        0,
        "0 when no prior inspection exists",
        _SAFE_ALL,
        _PROD_ALL,
        "Int32",
    ),
    _entry(
        "prior_inspection_day_count",
        "shared",
        ["inspection_date"],
        "all strictly-prior eligible events",
        0,
        "0 when no prior inspection exists",
        _SAFE_ALL,
        _PROD_ALL,
        "Int32",
    ),
    _entry(
        "days_since_previous_inspection_date",
        "shared",
        ["inspection_date"],
        "most recent strictly-prior date",
        1,
        "null when no prior inspection exists",
        _SAFE_ALL,
        _PROD_ALL,
        "Int32",
    ),
    _entry(
        "missing_history",
        "shared",
        ["inspection_date"],
        "all strictly-prior eligible events",
        0,
        "never null",
        _SAFE_ALL,
        _PROD_ALL,
        "Boolean",
    ),
    _entry(
        "prior_violation_total_citation_count",
        "shared",
        ["violation_events.count"],
        "all strictly-prior eligible violations",
        0,
        "null only when history_depth == 0; a trustworthy 0 when history exists with no citations",
        _SAFE_ALL,
        _PROD_ALL,
        "Int32",
    ),
    _entry(
        "prior_inspections_with_any_violation",
        "shared",
        ["violation_events.inspection_id"],
        "all strictly-prior eligible violations",
        0,
        "null only when history_depth == 0",
        _SAFE_ALL,
        _PROD_ALL,
        "Int32",
    ),
    _entry(
        "distinct_prior_violation_code_count",
        "shared",
        ["violation_events.violation_code"],
        "all strictly-prior eligible violations",
        0,
        "null only when history_depth == 0",
        _SAFE_ALL,
        _PROD_ALL,
        "Int32",
    ),
    _entry(
        "prior_top_violation_code_count",
        "shared",
        ["violation_events.violation_code", "count"],
        "all strictly-prior eligible violations",
        0,
        "null when no prior violation exists",
        _SAFE_ALL,
        _PROD_ALL,
        "Int32",
    ),
    _entry(
        "prior_top_violation_inspection_count",
        "shared",
        ["violation_events.violation_code", "inspection_id"],
        "all strictly-prior eligible violations",
        0,
        "null when no prior violation exists",
        _SAFE_ALL,
        _PROD_ALL,
        "Int32",
    ),
    _entry(
        "nyc_previous_day_score",
        "nyc",
        ["score", "score_conflict"],
        "most recent strictly-prior date",
        1,
        "null when no prior inspection, a score_conflict=True row, or same-day disagreement",
        "score_conflict=True rows are excluded from history regardless of the displayed score.",
        _PROD_ALL,
        "Float64",
    ),
    _entry(
        "nyc_previous_day_score_complete",
        "nyc",
        ["score", "score_conflict"],
        "most recent strictly-prior date",
        1,
        "never null",
        _SAFE_ALL,
        _PROD_ALL,
        "Boolean",
    ),
    _entry(
        "nyc_prior_valid_score_count",
        "nyc",
        ["score", "score_conflict"],
        "all strictly-prior eligible events",
        0,
        "never null",
        _SAFE_ALL,
        _PROD_ALL,
        "Int32",
    ),
    _entry(
        "nyc_score_prior_mean",
        "nyc",
        ["score", "score_conflict"],
        "all strictly-prior eligible events",
        1,
        "null when zero valid observations",
        _SAFE_ALL,
        _PROD_ALL,
        "Float64",
    ),
    _entry(
        "nyc_score_prior_max",
        "nyc",
        ["score", "score_conflict"],
        "all strictly-prior eligible events",
        1,
        "null when zero valid observations",
        _SAFE_ALL,
        _PROD_ALL,
        "Float64",
    ),
    _entry(
        "nyc_score_prior_variance",
        "nyc",
        ["score", "score_conflict"],
        "all strictly-prior eligible events",
        MIN_VALID_FOR_VARIANCE,
        f"null when fewer than {MIN_VALID_FOR_VARIANCE} valid observations",
        _SAFE_ALL,
        _PROD_ALL,
        "Float64",
    ),
    _entry(
        "nyc_score_variance_available",
        "nyc",
        ["score", "score_conflict"],
        "all strictly-prior eligible events",
        MIN_VALID_FOR_VARIANCE,
        "never null",
        _SAFE_ALL,
        _PROD_ALL,
        "Boolean",
    ),
    _entry(
        "nyc_score_prior_time_trend",
        "nyc",
        ["score", "score_conflict", "inspection_date"],
        "all strictly-prior eligible events",
        MIN_VALID_FOR_TREND,
        f"null when fewer than {MIN_VALID_FOR_TREND} valid observations or zero time variance",
        "Regressed on elapsed calendar days, not row index; scaled to change per 365 days.",
        _PROD_ALL,
        "Float64",
    ),
    _entry(
        "nyc_score_trend_available",
        "nyc",
        ["score", "score_conflict", "inspection_date"],
        "all strictly-prior eligible events",
        MIN_VALID_FOR_TREND,
        "never null",
        _SAFE_ALL,
        _PROD_ALL,
        "Boolean",
    ),
    _entry(
        "nyc_previous_day_critical_violation_count",
        "nyc",
        ["critical_violation_count"],
        "most recent strictly-prior date",
        1,
        "null when no prior inspection exists",
        _SAFE_ALL,
        _PROD_ALL,
        "Int32",
    ),
    _entry(
        "nyc_previous_day_critical_violation_count_complete",
        "nyc",
        ["critical_violation_count"],
        "most recent strictly-prior date",
        1,
        "never null",
        _SAFE_ALL,
        _PROD_ALL,
        "Boolean",
    ),
    _entry(
        "nyc_prior_critical_violation_total",
        "nyc",
        ["critical_violation_count"],
        "all strictly-prior eligible events",
        1,
        "null when zero valid observations",
        _SAFE_ALL,
        _PROD_ALL,
        "Int32",
    ),
    _entry(
        "fl_previous_day_high_priority_count",
        "florida",
        ["high_priority_count"],
        "most recent strictly-prior date",
        1,
        "null when any same-day contributing event's value is missing/malformed",
        _SAFE_ALL,
        _PROD_ALL,
        "Int32",
    ),
    _entry(
        "fl_previous_day_high_priority_count_complete",
        "florida",
        ["high_priority_count"],
        "most recent strictly-prior date",
        1,
        "never null",
        _SAFE_ALL,
        _PROD_ALL,
        "Boolean",
    ),
    _entry(
        "fl_previous_day_intermediate_count",
        "florida",
        ["intermediate_count"],
        "most recent strictly-prior date",
        1,
        "null when any same-day contributing event's value is missing/malformed",
        _SAFE_ALL,
        _PROD_ALL,
        "Int32",
    ),
    _entry(
        "fl_previous_day_intermediate_count_complete",
        "florida",
        ["intermediate_count"],
        "most recent strictly-prior date",
        1,
        "never null",
        _SAFE_ALL,
        _PROD_ALL,
        "Boolean",
    ),
    _entry(
        "fl_previous_day_basic_count",
        "florida",
        ["basic_count"],
        "most recent strictly-prior date",
        1,
        "null when any same-day contributing event's value is missing/malformed",
        _SAFE_ALL,
        _PROD_ALL,
        "Int32",
    ),
    _entry(
        "fl_previous_day_basic_count_complete",
        "florida",
        ["basic_count"],
        "most recent strictly-prior date",
        1,
        "never null",
        _SAFE_ALL,
        _PROD_ALL,
        "Boolean",
    ),
    _entry(
        "fl_previous_day_total_violation_count",
        "florida",
        ["violation_count"],
        "most recent strictly-prior date",
        1,
        "null when any same-day contributing event's value is missing/malformed",
        _SAFE_ALL,
        _PROD_ALL,
        "Int32",
    ),
    _entry(
        "fl_previous_day_total_violation_count_complete",
        "florida",
        ["violation_count"],
        "most recent strictly-prior date",
        1,
        "never null",
        _SAFE_ALL,
        _PROD_ALL,
        "Boolean",
    ),
    _entry(
        "fl_total_violation_prior_mean",
        "florida",
        ["violation_count"],
        "all strictly-prior eligible events",
        1,
        "null when zero valid observations",
        _SAFE_ALL,
        _PROD_ALL,
        "Float64",
    ),
    _entry(
        "fl_total_violation_prior_max",
        "florida",
        ["violation_count"],
        "all strictly-prior eligible events",
        1,
        "null when zero valid observations",
        _SAFE_ALL,
        _PROD_ALL,
        "Float64",
    ),
    _entry(
        "fl_total_violation_prior_variance",
        "florida",
        ["violation_count"],
        "all strictly-prior eligible events",
        MIN_VALID_FOR_VARIANCE,
        f"null when fewer than {MIN_VALID_FOR_VARIANCE} valid observations",
        _SAFE_ALL,
        _PROD_ALL,
        "Float64",
    ),
    _entry(
        "fl_total_violation_variance_available",
        "florida",
        ["violation_count"],
        "all strictly-prior eligible events",
        MIN_VALID_FOR_VARIANCE,
        "never null",
        _SAFE_ALL,
        _PROD_ALL,
        "Boolean",
    ),
    _entry(
        "fl_total_violation_prior_time_trend",
        "florida",
        ["violation_count", "inspection_date"],
        "all strictly-prior eligible events",
        MIN_VALID_FOR_TREND,
        f"null when fewer than {MIN_VALID_FOR_TREND} valid observations or zero time variance",
        "Regressed on elapsed calendar days, not row index; scaled to change per 365 days.",
        _PROD_ALL,
        "Float64",
    ),
    _entry(
        "fl_total_violation_trend_available",
        "florida",
        ["violation_count", "inspection_date"],
        "all strictly-prior eligible events",
        MIN_VALID_FOR_TREND,
        "never null",
        _SAFE_ALL,
        _PROD_ALL,
        "Boolean",
    ),
    _entry(
        "fl_high_priority_prior_mean",
        "florida",
        ["high_priority_count"],
        "all strictly-prior eligible events",
        1,
        "null when zero valid observations",
        _SAFE_ALL,
        _PROD_ALL,
        "Float64",
    ),
    _entry(
        "fl_high_priority_prior_max",
        "florida",
        ["high_priority_count"],
        "all strictly-prior eligible events",
        1,
        "null when zero valid observations",
        _SAFE_ALL,
        _PROD_ALL,
        "Float64",
    ),
    _entry(
        "fl_prior_inspections_with_high_priority",
        "florida",
        ["high_priority_count"],
        "all strictly-prior eligible events",
        0,
        "null only when history_depth == 0",
        _SAFE_ALL,
        _PROD_ALL,
        "Int32",
    ),
    _entry(
        "fl_prior_high_priority_total",
        "florida",
        ["high_priority_count"],
        "all strictly-prior eligible events",
        1,
        "null when zero valid observations",
        _SAFE_ALL,
        _PROD_ALL,
        "Int32",
    ),
    _entry(
        "fl_prior_valid_high_priority_count",
        "florida",
        ["high_priority_count"],
        "all strictly-prior eligible events",
        0,
        "never null",
        _SAFE_ALL,
        _PROD_ALL,
        "Int32",
    ),
    _entry(
        "fl_prior_valid_intermediate_count",
        "florida",
        ["intermediate_count"],
        "all strictly-prior eligible events",
        0,
        "never null",
        _SAFE_ALL,
        _PROD_ALL,
        "Int32",
    ),
    _entry(
        "fl_prior_valid_basic_count",
        "florida",
        ["basic_count"],
        "all strictly-prior eligible events",
        0,
        "never null",
        _SAFE_ALL,
        _PROD_ALL,
        "Int32",
    ),
    _entry(
        "fl_prior_valid_total_count",
        "florida",
        ["violation_count"],
        "all strictly-prior eligible events",
        0,
        "never null",
        _SAFE_ALL,
        _PROD_ALL,
        "Int32",
    ),
    _entry(
        "fl_prior_follow_up_required_count",
        "florida",
        ["disposition_status"],
        "all strictly-prior eligible events",
        0,
        "null only when history_depth == 0",
        "Only the exact recognized 'follow_up_required' status counts; unknown/null never counts.",
        _PROD_ALL,
        "Int32",
    ),
    _entry(
        "fl_prior_temporary_closure_count",
        "florida",
        ["disposition_status"],
        "all strictly-prior eligible events",
        0,
        "null only when history_depth == 0",
        "Only the exact recognized 'temporary_closure' status counts; unknown/null never counts.",
        _PROD_ALL,
        "Int32",
    ),
]
