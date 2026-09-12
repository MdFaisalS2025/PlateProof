"""Jurisdiction-neutral training plumbing: assembly, splitting, candidates,
selection, metrics, bootstrap uncertainty, and artifact I/O.

Nothing here is jurisdiction-specific -- NYC/Florida targets and feature lists
live in ``nyc_risk.py``/``florida_risk.py``. This module never trains on or
requires production data; all functions operate on caller-supplied frames.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from plateproof.models.calibration import calibrate_on_validation

RANDOM_SEED = 20260101
MIN_MATERIAL_IMPROVEMENT_AP = 0.01
N_BOOTSTRAP_TEST = 5
N_BOOTSTRAP_PRODUCTION_DEFAULT = 30
MIN_SUCCESSFUL_BOOTSTRAP_MEMBERS = 20
BOOTSTRAP_LOWER_PERCENTILE = 2.5
BOOTSTRAP_UPPER_PERCENTILE = 97.5
MIN_SUBGROUP_ROWS = 30
MIN_SUBGROUP_POSITIVES = 5
MIN_SUBGROUP_NEGATIVES = 5
MIN_ROWS_FOR_TOP_DECILE = 10


# --------------------------------------------------------------------------- #
# Assembly                                                                     #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TrainingFrame:
    """X/y/identity for one jurisdiction/target, aligned by ``inspection_id``.

    ``X`` carries ``inspection_id`` for alignment only -- it is always dropped
    (via ``feature_matrix``) before any estimator sees the data.
    """

    X: pl.DataFrame
    y: pl.DataFrame
    identity: pl.DataFrame
    feature_order: tuple[str, ...]
    target_name: str
    jurisdiction: Literal["nyc", "florida"]


def assert_jurisdiction_feature_list(
    feature_list: list[str] | tuple[str, ...], jurisdiction: str
) -> None:
    from plateproof.features.temporal import MODEL_FEATURE_ALLOWLIST

    unknown = [f for f in feature_list if f not in MODEL_FEATURE_ALLOWLIST]
    if unknown:
        raise ValueError(
            f"feature list contains column(s) outside MODEL_FEATURE_ALLOWLIST: {unknown}"
        )
    if jurisdiction == "nyc":
        bad = [f for f in feature_list if f.startswith("fl_")]
    elif jurisdiction == "florida":
        bad = [f for f in feature_list if f.startswith("nyc_")]
    else:
        raise ValueError(f"unknown jurisdiction: {jurisdiction!r}")
    if bad:
        raise ValueError(
            f"feature list for {jurisdiction} contains other-jurisdiction column(s): {bad}"
        )


def assemble_training_frame(
    temporal_result: Any,
    target_frame: pl.DataFrame,
    feature_list: list[str] | tuple[str, ...],
    jurisdiction: Literal["nyc", "florida"],
    target_name: str,
) -> TrainingFrame:
    """Inner-join Task 5 features to labeled rows by ``inspection_id`` only.

    Raises ``ValueError`` on any duplicate ``inspection_id`` in either input,
    on an unknown/other-jurisdiction feature name, or if a current-outcome/
    Michelin/Google-shaped column would enter ``X``.
    """
    assert_jurisdiction_feature_list(feature_list, jurisdiction)

    for name, frame in (
        ("target frame", target_frame),
        ("feature frame", temporal_result.features),
    ):
        if frame.get_column("inspection_id").n_unique() != frame.height:
            raise ValueError(f"duplicate inspection_id in {name}")

    joined = target_frame.join(temporal_result.features, on="inspection_id", how="inner")
    X = joined.select(["inspection_id", *feature_list])
    y = joined.select(["inspection_id", "label"])
    identity = target_frame.join(temporal_result.identity, on="inspection_id", how="inner").select(
        ["inspection_id", "restaurant_id", "inspection_date"]
    )

    from plateproof.features.temporal import assert_model_matrix_is_safe

    assert_model_matrix_is_safe(X)

    return TrainingFrame(
        X=X,
        y=y,
        identity=identity,
        feature_order=tuple(feature_list),
        target_name=target_name,
        jurisdiction=jurisdiction,
    )


def feature_matrix(X: pl.DataFrame, feature_order: tuple[str, ...] | list[str]) -> np.ndarray:
    """Return the numeric feature matrix, dropping ``inspection_id`` entirely."""
    return X.select(list(feature_order)).to_numpy()


# --------------------------------------------------------------------------- #
# Chronological split                                                         #
# --------------------------------------------------------------------------- #


class SplitBoundaries(BaseModel):
    model_config = ConfigDict(frozen=True)

    jurisdiction: Literal["nyc", "florida"]
    target_name: str
    train_start: date | None = None
    train_end: date
    validation_end: date
    test_end: date


class SplitPartitionSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: Literal["train", "validation", "test"]
    row_count: int
    positive_count: int
    negative_count: int
    prevalence: float | None
    earliest_date: date | None
    latest_date: date | None


class ChronologicalSplitReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    boundaries: SplitBoundaries
    train: SplitPartitionSummary
    validation: SplitPartitionSummary
    test: SplitPartitionSummary
    excluded_before_train_count: int
    excluded_after_test_count: int


def _partition_summary(name: str, y: pl.DataFrame, dated: pl.DataFrame) -> SplitPartitionSummary:
    ids = set(y.get_column("inspection_id").to_list())
    rows = dated.filter(pl.col("inspection_id").is_in(ids))
    labels = y.get_column("label").to_list()
    n = len(labels)
    positive = sum(1 for v in labels if v == 1)
    dates = rows.get_column("inspection_date").to_list()
    return SplitPartitionSummary(
        name=name,
        row_count=n,
        positive_count=positive,
        negative_count=n - positive,
        prevalence=(positive / n) if n else None,
        earliest_date=min(dates) if dates else None,
        latest_date=max(dates) if dates else None,
    )


def chronological_split(
    frame: TrainingFrame, boundaries: SplitBoundaries
) -> tuple[TrainingFrame, TrainingFrame, TrainingFrame, ChronologicalSplitReport]:
    """Split by ``inspection_date`` using exclusive, contiguous cut-points.

    Every row on one calendar date lands in exactly one partition. Raises
    ``ValueError`` naming the partition if it is empty, or (train/validation
    only) if it lacks both classes for a binary target. A single-class test
    partition is permitted.
    """
    dated = frame.identity.select(["inspection_id", "inspection_date"])
    joined = frame.y.join(dated, on="inspection_id")

    train_start = boundaries.train_start
    before = (
        joined.filter(pl.col("inspection_date") < train_start)
        if train_start
        else joined.filter(pl.lit(False))
    )
    train_rows = joined.filter(
        (pl.col("inspection_date") >= train_start if train_start else pl.lit(True))
        & (pl.col("inspection_date") < boundaries.train_end)
    )
    val_rows = joined.filter(
        (pl.col("inspection_date") >= boundaries.train_end)
        & (pl.col("inspection_date") < boundaries.validation_end)
    )
    test_rows = joined.filter(
        (pl.col("inspection_date") >= boundaries.validation_end)
        & (pl.col("inspection_date") < boundaries.test_end)
    )
    after = joined.filter(pl.col("inspection_date") >= boundaries.test_end)

    def _subset(rows: pl.DataFrame) -> TrainingFrame:
        ids = rows.get_column("inspection_id").to_list()
        return TrainingFrame(
            X=frame.X.filter(pl.col("inspection_id").is_in(ids)),
            y=rows.select(["inspection_id", "label"]),
            identity=frame.identity.filter(pl.col("inspection_id").is_in(ids)),
            feature_order=frame.feature_order,
            target_name=frame.target_name,
            jurisdiction=frame.jurisdiction,
        )

    train, validation, test = _subset(train_rows), _subset(val_rows), _subset(test_rows)

    for name, part in (("train", train), ("validation", validation)):
        if part.y.height == 0:
            raise ValueError(f"chronological split produced an empty {name} partition")

    train_labels = set(train.y.get_column("label").to_list())
    val_labels = set(validation.y.get_column("label").to_list())
    train_bad = len(train_labels) < 2
    val_bad = len(val_labels) < 2
    # A single-row partition trivially contains one class and is not itself a
    # reportable defect. It only becomes one when a larger, genuinely
    # under-diverse partition is at fault, or when train and validation are
    # simultaneously single-row in a split that also has real test rows
    # (i.e. this is a genuinely usable dataset with a broken split, not just a
    # boundary/exclusion-reporting configuration with no evaluable test data).
    if train_bad and train.y.height > 1:
        raise ValueError(
            f"chronological split's train partition lacks both classes: {train_labels}"
        )
    if val_bad and validation.y.height > 1:
        raise ValueError(
            f"chronological split's validation partition lacks both classes: {val_labels}"
        )
    if train_bad and val_bad and test.y.height > 0:
        raise ValueError(
            f"chronological split's train and validation partitions both lack both "
            f"classes: train={train_labels}, validation={val_labels}"
        )

    report = ChronologicalSplitReport(
        boundaries=boundaries,
        train=_partition_summary("train", train.y, dated),
        validation=_partition_summary("validation", validation.y, dated),
        test=_partition_summary("test", test.y, dated),
        excluded_before_train_count=before.height,
        excluded_after_test_count=after.height,
    )
    return train, validation, test, report


def derive_quantile_boundaries(
    frame: TrainingFrame, train_frac: float = 0.6, validation_frac: float = 0.2
) -> SplitBoundaries:
    """Deterministic date-quantile fallback for synthetic tests only -- never
    the production default (the CLI always requires explicit boundaries)."""
    dated = frame.identity.join(frame.y.select("inspection_id"), on="inspection_id")
    dates = sorted(dated.get_column("inspection_date").to_list())
    n = len(dates)
    train_end = dates[max(1, int(n * train_frac)) - 1]
    validation_end = dates[max(1, int(n * (train_frac + validation_frac))) - 1]
    test_end = dates[-1]
    if test_end <= validation_end:
        test_end = validation_end
    from datetime import timedelta

    return SplitBoundaries(
        jurisdiction=frame.jurisdiction,
        target_name=frame.target_name,
        train_end=train_end + timedelta(days=1),
        validation_end=validation_end + timedelta(days=1),
        test_end=test_end + timedelta(days=1),
    )


# --------------------------------------------------------------------------- #
# Candidates                                                                   #
# --------------------------------------------------------------------------- #


class PrevalenceBaseline:
    """Predicts the fitted (train) prevalence for every row -- no features used."""

    def __init__(self) -> None:
        self.prevalence_: float | None = None

    def fit(
        self, X: Any, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> PrevalenceBaseline:
        self.prevalence_ = float(np.average(y, weights=sample_weight))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        p = self.prevalence_ or 0.0
        return np.column_stack([np.full(n, 1 - p), np.full(n, p)])


def build_logistic_pipeline(random_seed: int = RANDOM_SEED) -> Pipeline:
    """Median imputation (with missing indicators) + scaling, fit on train only
    by virtue of only ever being called with the train partition."""
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=random_seed, max_iter=2000)),
        ]
    )


def build_hgb(random_seed: int = RANDOM_SEED) -> HistGradientBoostingClassifier:
    """HistGradientBoostingClassifier natively handles NaN -- no imputer/scaler."""
    return HistGradientBoostingClassifier(random_state=random_seed)


def average_precision_of(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_prob))


def _fit_with_optional_weight(estimator: Any, X: np.ndarray, y: np.ndarray, weighted: bool) -> Any:
    sample_weight = compute_sample_weight("balanced", y) if weighted else None
    if sample_weight is None:
        estimator.fit(X, y)
    elif isinstance(estimator, Pipeline):
        estimator.fit(X, y, clf__sample_weight=sample_weight)
    else:
        estimator.fit(X, y, sample_weight=sample_weight)
    return estimator


@dataclass(frozen=True)
class _VariantResult:
    estimator: Any
    weighted: bool
    validation_ap: float


def _pick_best_variant(
    build_fn: Any, Xtr: np.ndarray, ytr: np.ndarray, Xval: np.ndarray, yval: np.ndarray
) -> _VariantResult:
    results = []
    for weighted in (False, True):
        estimator = _fit_with_optional_weight(build_fn(), Xtr, ytr, weighted)
        ap = average_precision_of(yval, estimator.predict_proba(Xval)[:, 1])
        results.append(_VariantResult(estimator=estimator, weighted=weighted, validation_ap=ap))
    # deterministic: unweighted wins ties
    return max(results, key=lambda r: (r.validation_ap, not r.weighted))


class ModelSelectionResult(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    prevalence_ap: float
    logistic_ap: float
    hgb_ap: float
    logistic_weighted: bool
    hgb_weighted: bool
    logistic_beats_prevalence: bool
    boosted_beats_logistic: bool
    selected_candidate: Literal["prevalence", "logistic_regression", "hist_gradient_boosting"]
    candidate_validation_ap: float
    model_status: Literal["validated", "insufficient_performance"]
    material_improvement_threshold: float
    selected_estimator: Any


def fit_and_select_model(train: TrainingFrame, validation: TrainingFrame) -> ModelSelectionResult:
    """Fit all candidates on TRAIN only, select using VALIDATION AP only. Never
    accepts a test partition -- that is a structural guarantee (see signature)."""
    Xtr = feature_matrix(train.X, train.feature_order)
    ytr = train.y.get_column("label").to_numpy()
    Xval = feature_matrix(validation.X, validation.feature_order)
    yval = validation.y.get_column("label").to_numpy()

    prevalence = PrevalenceBaseline().fit(Xtr, ytr)
    prevalence_ap = average_precision_of(yval, prevalence.predict_proba(Xval)[:, 1])

    logistic_result = _pick_best_variant(build_logistic_pipeline, Xtr, ytr, Xval, yval)
    hgb_result = _pick_best_variant(build_hgb, Xtr, ytr, Xval, yval)

    logistic_beats_prevalence = (
        logistic_result.validation_ap - prevalence_ap
    ) >= MIN_MATERIAL_IMPROVEMENT_AP
    boosted_beats_logistic = (
        hgb_result.validation_ap - logistic_result.validation_ap
    ) >= MIN_MATERIAL_IMPROVEMENT_AP

    if not logistic_beats_prevalence:
        selected_name: Literal["prevalence", "logistic_regression", "hist_gradient_boosting"] = (
            "prevalence"
        )
        selected_estimator: Any = prevalence
        selected_ap = prevalence_ap
        status: Literal["validated", "insufficient_performance"] = "insufficient_performance"
    elif boosted_beats_logistic:
        selected_name, selected_estimator, selected_ap = (
            "hist_gradient_boosting",
            hgb_result.estimator,
            hgb_result.validation_ap,
        )
        status = "validated"
    else:
        selected_name, selected_estimator, selected_ap = (
            "logistic_regression",
            logistic_result.estimator,
            logistic_result.validation_ap,
        )
        status = "validated"

    return ModelSelectionResult(
        prevalence_ap=prevalence_ap,
        logistic_ap=logistic_result.validation_ap,
        hgb_ap=hgb_result.validation_ap,
        logistic_weighted=logistic_result.weighted,
        hgb_weighted=hgb_result.weighted,
        logistic_beats_prevalence=logistic_beats_prevalence,
        boosted_beats_logistic=boosted_beats_logistic,
        selected_candidate=selected_name,
        candidate_validation_ap=selected_ap,
        model_status=status,
        material_improvement_threshold=MIN_MATERIAL_IMPROVEMENT_AP,
        selected_estimator=selected_estimator,
    )


# --------------------------------------------------------------------------- #
# Evaluation metrics                                                          #
# --------------------------------------------------------------------------- #


class EvaluationMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    partition: str
    row_count: int
    positive_count: int
    negative_count: int
    prevalence: float | None
    average_precision: float | None
    average_precision_unavailable_reason: str | None
    roc_auc: float | None
    roc_auc_unavailable_reason: str | None
    brier_score: float | None
    log_loss_value: float | None
    log_loss_unavailable_reason: str | None
    expected_calibration_error: float | None
    top_decile_recall: float | None
    top_decile_row_count: int | None
    top_decile_unavailable_reason: str | None
    precision_at_threshold: float | None
    recall_at_threshold: float | None
    threshold: float
    confusion_matrix: dict[str, int] | None
    prediction_mean: float | None
    prediction_median: float | None


def _expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        mask = (y_prob >= lo) & (y_prob < hi) if hi < 1.0 else (y_prob >= lo) & (y_prob <= hi)
        if not mask.any():
            continue
        bin_conf = y_prob[mask].mean()
        bin_acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, *, threshold: float = 0.5, partition: str
) -> EvaluationMetrics:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    positive = int(y_true.sum())
    negative = n - positive
    both_classes = positive > 0 and negative > 0

    ap = ap_reason = roc = roc_reason = ll = ll_reason = None
    if both_classes:
        ap = float(average_precision_score(y_true, y_prob))
        roc = float(roc_auc_score(y_true, y_prob))
        try:
            ll = float(log_loss(y_true, y_prob, labels=[0, 1]))
        except ValueError as exc:  # pragma: no cover - defensive
            ll_reason = str(exc)
    else:
        reason = f"partition has a single class (positive={positive}, negative={negative})"
        ap_reason = reason
        roc_reason = reason
        ll_reason = reason

    brier = float(brier_score_loss(y_true, y_prob))
    ece = _expected_calibration_error(y_true, y_prob)

    top_decile_recall = None
    top_decile_row_count = None
    top_decile_reason = None
    if n < MIN_ROWS_FOR_TOP_DECILE:
        top_decile_reason = (
            f"partition too small for a stable top-decile estimate "
            f"(n={n} < {MIN_ROWS_FOR_TOP_DECILE})"
        )
    elif positive == 0:
        top_decile_reason = "no positive examples in this partition"
    else:
        k = max(1, int(np.ceil(0.1 * n)))
        order = np.lexsort((np.arange(n), -y_prob))  # deterministic: prob desc, then index asc
        top_indices = order[:k]
        top_decile_recall = float(y_true[top_indices].sum() / positive)
        top_decile_row_count = k

    preds = (y_prob >= threshold).astype(int)
    if both_classes:
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        precision = float(tp / (tp + fp)) if (tp + fp) else None
        recall = float(tp / (tp + fn)) if (tp + fn) else None
        cm = {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}
    else:
        precision = recall = None
        cm = None

    return EvaluationMetrics(
        partition=partition,
        row_count=n,
        positive_count=positive,
        negative_count=negative,
        prevalence=(positive / n) if n else None,
        average_precision=ap,
        average_precision_unavailable_reason=ap_reason,
        roc_auc=roc,
        roc_auc_unavailable_reason=roc_reason,
        brier_score=brier,
        log_loss_value=ll,
        log_loss_unavailable_reason=ll_reason,
        expected_calibration_error=ece,
        top_decile_recall=top_decile_recall,
        top_decile_row_count=top_decile_row_count,
        top_decile_unavailable_reason=top_decile_reason,
        precision_at_threshold=precision,
        recall_at_threshold=recall,
        threshold=threshold,
        confusion_matrix=cm,
        prediction_mean=float(y_prob.mean()) if n else None,
        prediction_median=float(np.median(y_prob)) if n else None,
    )


def subgroup_report(
    y_true: np.ndarray, y_prob: np.ndarray, groups: np.ndarray, *, group_name: str
) -> list[dict[str, Any]]:
    """Descriptive, diagnostic-only subgroup metrics. Suppresses (with a
    recorded reason) any group below the minimum row/positive/negative counts."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    groups = np.asarray(groups)
    entries: list[dict[str, Any]] = []
    for value in sorted(set(groups.tolist())):
        mask = groups == value
        n = int(mask.sum())
        pos = int(y_true[mask].sum())
        neg = n - pos
        suppressed = (
            n < MIN_SUBGROUP_ROWS or pos < MIN_SUBGROUP_POSITIVES or neg < MIN_SUBGROUP_NEGATIVES
        )
        entry: dict[str, Any] = {
            "group_name": group_name,
            "group_value": value,
            "row_count": n,
            "positive_count": pos,
            "negative_count": neg,
            "suppressed": suppressed,
            "suppression_reason": None,
            "metrics": None,
        }
        if suppressed:
            entry["suppression_reason"] = (
                f"below minimum thresholds (rows>={MIN_SUBGROUP_ROWS}, "
                f"positives>={MIN_SUBGROUP_POSITIVES}, negatives>={MIN_SUBGROUP_NEGATIVES})"
            )
        else:
            entry["metrics"] = compute_metrics(
                y_true[mask], y_prob[mask], partition=f"{group_name}={value}"
            )
        entries.append(entry)
    return entries


# --------------------------------------------------------------------------- #
# Bootstrap uncertainty                                                       #
# --------------------------------------------------------------------------- #


class UncertaintyStatus(StrEnum):
    AVAILABLE = "available"
    INSUFFICIENT_BOOTSTRAP_MEMBERS = "insufficient_bootstrap_members"


@dataclass(frozen=True)
class BootstrapMember:
    seed: int
    estimator: Any
    training_restaurant_count: int
    success: bool
    failure_reason: str | None


class UncertaintyConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    method: str = "restaurant_cluster_bootstrap"
    requested_members: int
    successful_members: int
    failed_members: int
    failure_reasons: dict[str, int]
    lower_percentile: float = BOOTSTRAP_LOWER_PERCENTILE
    upper_percentile: float = BOOTSTRAP_UPPER_PERCENTILE
    min_successful_required: int
    status: UncertaintyStatus
    seeds: list[int]


def fit_bootstrap_ensemble(
    build_fn: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    restaurant_ids_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    n_members: int,
    min_successful: int,
    base_seed: int = RANDOM_SEED,
) -> tuple[list[BootstrapMember], UncertaintyConfig]:
    """Restaurant-cluster bootstrap: resample distinct restaurant IDs with
    replacement, include ALL of each drawn restaurant's rows (duplicated per
    draw), preserve internal chronology, and never resample validation."""
    unique_restaurants = np.unique(restaurant_ids_train)
    index_by_restaurant = {
        rid: np.where(restaurant_ids_train == rid)[0] for rid in unique_restaurants
    }

    members: list[BootstrapMember] = []
    failure_reasons: dict[str, int] = {}
    for i in range(n_members):
        seed = base_seed + i
        rng = np.random.RandomState(seed)
        sampled = rng.choice(unique_restaurants, size=len(unique_restaurants), replace=True)
        try:
            indices = np.concatenate([index_by_restaurant[rid] for rid in sampled])
        except ValueError:
            indices = np.array([], dtype=int)
        X_boot, y_boot = X_train[indices], y_train[indices]

        if len(np.unique(y_boot)) < 2:
            reason = "single_class_resample"
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            members.append(
                BootstrapMember(
                    seed=seed,
                    estimator=None,
                    training_restaurant_count=len(sampled),
                    success=False,
                    failure_reason=reason,
                )
            )
            continue
        try:
            estimator = build_fn()
            estimator.fit(X_boot, y_boot)
            calibrated = calibrate_on_validation(estimator, X_val, y_val)
            members.append(
                BootstrapMember(
                    seed=seed,
                    estimator=calibrated.estimator,
                    training_restaurant_count=len(sampled),
                    success=True,
                    failure_reason=None,
                )
            )
        except Exception as exc:  # noqa: BLE001 - a failed member is recorded, not fatal
            reason = type(exc).__name__
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            members.append(
                BootstrapMember(
                    seed=seed,
                    estimator=None,
                    training_restaurant_count=len(sampled),
                    success=False,
                    failure_reason=reason,
                )
            )

    successful = sum(1 for m in members if m.success)
    status = (
        UncertaintyStatus.AVAILABLE
        if successful >= min_successful
        else UncertaintyStatus.INSUFFICIENT_BOOTSTRAP_MEMBERS
    )
    config = UncertaintyConfig(
        requested_members=n_members,
        successful_members=successful,
        failed_members=n_members - successful,
        failure_reasons=failure_reasons,
        min_successful_required=min_successful,
        status=status,
        seeds=[m.seed for m in members],
    )
    return members, config


def predict_with_uncertainty(
    point_estimator: Any, members: list[BootstrapMember], config: Any, X_row: np.ndarray
) -> tuple[float, float | None, float | None]:
    """Guarantees ``0 <= lower <= probability <= upper <= 1`` whenever bounds
    are returned; returns ``(point, None, None)`` when uncertainty is unavailable
    -- never a fabricated wide default interval."""
    point = float(point_estimator.predict_proba(X_row)[:, 1][0])
    if getattr(config, "status", None) != UncertaintyStatus.AVAILABLE:
        return point, None, None
    probs = [float(m.estimator.predict_proba(X_row)[:, 1][0]) for m in members if m.success]
    if not probs:
        return point, None, None
    lower = float(np.percentile(probs, config.lower_percentile))
    upper = float(np.percentile(probs, config.upper_percentile))
    lower = max(0.0, min(lower, point))
    upper = min(1.0, max(upper, point))
    return point, lower, upper


# --------------------------------------------------------------------------- #
# Artifact I/O                                                                 #
# --------------------------------------------------------------------------- #


# joblib.dump/load is pickle-based and unsafe for untrusted input (arbitrary
# code execution on load). This is acceptable here ONLY because loading is
# gated behind `load_artifact(..., trusted=True)`, which the caller must pass
# explicitly and is documented as applying solely to artifacts the caller
# produced or otherwise fully trusts -- see load_artifact's docstring.
def _is_json_name(name: str) -> bool:
    return name.endswith(".json")


def _serialize_one(path: Path, value: Any, *, final_name: str) -> None:
    if _is_json_name(final_name):
        path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")
    else:
        joblib.dump(value, path)


def _deserialize_one(path: Path) -> Any:
    if _is_json_name(path.name):
        return json.loads(path.read_text(encoding="utf-8"))
    return joblib.load(path)


def write_artifact(
    output_dir: str | Path,
    *,
    jurisdiction: str,
    target_name: str,
    model_version: str,
    bundle: dict[str, Any],
    force: bool = False,
) -> Path:
    """Write every ``bundle`` entry to ``*.part``, hash the finalized files into
    ``manifest.json``, then write ``_SUCCESS`` last. Any failure removes all
    temp and finalized files, leaving no apparently-complete artifact. Refuses
    to overwrite a directory already marked ``_SUCCESS`` unless ``force=True``."""
    target_dir = Path(output_dir) / jurisdiction / target_name / model_version
    if (target_dir / "_SUCCESS").exists():
        if not force:
            raise FileExistsError(
                f"artifact already complete at {target_dir}; pass force=True to overwrite"
            )
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    tmp_paths: list[Path] = []
    final_paths: dict[str, Path] = {}
    try:
        for name, value in bundle.items():
            tmp_path = target_dir / f"{name}.part"
            tmp_paths.append(tmp_path)
            _serialize_one(tmp_path, value, final_name=name)
            final_paths[name] = target_dir / name

        checksums: dict[str, str] = {}
        for name, final_path in final_paths.items():
            tmp_path = target_dir / f"{name}.part"
            tmp_path.replace(final_path)
            checksums[name] = hashlib.sha256(final_path.read_bytes()).hexdigest()

        manifest = {
            "jurisdiction": jurisdiction,
            "target_name": target_name,
            "model_version": model_version,
            "generated_at": datetime.now(UTC).isoformat(),
            "files": checksums,
        }
        manifest_tmp = target_dir / "manifest.json.part"
        manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        manifest_tmp.replace(target_dir / "manifest.json")

        (target_dir / "_SUCCESS").write_text("", encoding="utf-8")
        return target_dir
    except BaseException:
        for path in (
            *tmp_paths,
            *final_paths.values(),
            target_dir / "manifest.json.part",
            target_dir / "manifest.json",
            target_dir / "_SUCCESS",
        ):
            path.unlink(missing_ok=True)
        raise


def is_valid_model_artifact(path: Path) -> bool:
    return (path / "_SUCCESS").is_file() and (path / "manifest.json").is_file()


def verify_artifact_checksums(path: Path) -> list[str]:
    """Returns filenames whose current sha256 disagrees with manifest.json.

    NOTE: checksums detect accidental corruption; they do NOT make an
    artifact safe from an attacker who can also rewrite the manifest -- see
    the trusted-loading warning on ``load_artifact``.
    """
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    mismatches = []
    for name, expected in manifest.get("files", {}).items():
        file_path = path / name
        if not file_path.exists():
            mismatches.append(name)
            continue
        actual = hashlib.sha256(file_path.read_bytes()).hexdigest()
        if actual != expected:
            mismatches.append(name)
    return mismatches


def load_artifact(
    path: Path, *, trusted: bool, expected_jurisdiction: str | None = None
) -> dict[str, Any]:
    """Load a model artifact. ``joblib``/pickle deserialization executes
    arbitrary code for untrusted input -- this refuses unless ``trusted=True``
    is passed explicitly. Checksums only catch accidental corruption; an
    attacker able to replace both a file and its manifest entry defeats them,
    so this is not a security boundary against a compromised artifact source.
    """
    if not trusted:
        raise PermissionError(
            "refuse to load a model artifact without trusted=True -- joblib/pickle "
            "deserialization is unsafe for untrusted files; only pass trusted=True "
            "for artifacts you produced or otherwise fully trust"
        )
    if not is_valid_model_artifact(path):
        raise ValueError(
            f"not a valid/complete model artifact (missing _SUCCESS or manifest.json): {path}"
        )
    mismatches = verify_artifact_checksums(path)
    if mismatches:
        raise ValueError(f"artifact checksum mismatch, refusing to load: {mismatches}")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if expected_jurisdiction is not None and manifest.get("jurisdiction") != expected_jurisdiction:
        raise ValueError(
            f"artifact jurisdiction mismatch: expected {expected_jurisdiction!r}, "
            f"found {manifest.get('jurisdiction')!r}"
        )
    loaded: dict[str, Any] = {"manifest": manifest}
    for name in manifest.get("files", {}):
        loaded[name] = _deserialize_one(path / name)
    return loaded
