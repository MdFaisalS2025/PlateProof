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
import uuid
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

from plateproof.models.calibration import (
    CalibrationOutcome,
    CalibrationStatus,
    calibrate_on_validation,
)
from plateproof.models.risk_bands import RiskBandThresholds

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

    for name, part in (("train", train), ("validation", validation), ("test", test)):
        if part.y.height == 0:
            raise ValueError(f"chronological split produced an empty {name} partition")

    train_labels = set(train.y.get_column("label").to_list())
    val_labels = set(validation.y.get_column("label").to_list())
    # Train and validation must always contain both classes -- a partition is
    # not usable merely because it has at least one row. A single-class test
    # partition is permitted: test is never used for fitting or selection.
    if len(train_labels) < 2:
        raise ValueError(
            f"chronological split's train partition lacks both classes: {train_labels}"
        )
    if len(val_labels) < 2:
        raise ValueError(
            f"chronological split's validation partition lacks both classes: {val_labels}"
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


CANDIDATE_BUILDERS: dict[str, Any] = {
    "prevalence": PrevalenceBaseline,
    "logistic_regression": build_logistic_pipeline,
    "hist_gradient_boosting": build_hgb,
}

_PREPROCESSING_DESCRIPTIONS: dict[str, str] = {
    "prevalence": "none -- predicts the fitted train prevalence for every row",
    "logistic_regression": "median imputation with missing indicators, then standard scaling",
    "hist_gradient_boosting": "none -- native NaN handling, no imputer or scaler",
}


class SelectedModelConfiguration(BaseModel):
    """The frozen selection decision, sufficient to reproduce the exact
    selected candidate (estimator type, weighting, preprocessing, seed) for
    every bootstrap refit -- never a generic unweighted rebuild."""

    model_config = ConfigDict(frozen=True)

    candidate_name: Literal["prevalence", "logistic_regression", "hist_gradient_boosting"]
    weighted: bool
    random_seed: int
    feature_order: tuple[str, ...]
    preprocessing_description: str
    selection_metric: str
    material_improvement_threshold: float


def build_selected_model_configuration(
    selection: Any, feature_order: tuple[str, ...], random_seed: int = RANDOM_SEED
) -> SelectedModelConfiguration:
    weighted = {
        "prevalence": False,
        "logistic_regression": selection.logistic_weighted,
        "hist_gradient_boosting": selection.hgb_weighted,
    }[selection.selected_candidate]
    return SelectedModelConfiguration(
        candidate_name=selection.selected_candidate,
        weighted=weighted,
        random_seed=random_seed,
        feature_order=feature_order,
        preprocessing_description=_PREPROCESSING_DESCRIPTIONS[selection.selected_candidate],
        selection_metric="validation_average_precision",
        material_improvement_threshold=selection.material_improvement_threshold,
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


def aligned_restaurant_ids(
    inspection_ids: list[str] | np.ndarray, identity: pl.DataFrame
) -> np.ndarray:
    """Return one restaurant_id per ``inspection_ids`` entry, in that exact
    order, by looking each one up in ``identity``.

    This never relies on ``identity`` and the caller's id sequence sharing a
    row order -- the lookup is by ``inspection_id`` value, not position.
    Raises ``ValueError`` if ``identity`` has a duplicate ``inspection_id``, or
    if any requested id is absent from it.
    """
    ids = list(inspection_ids)
    if identity.get_column("inspection_id").n_unique() != identity.height:
        raise ValueError("identity frame contains duplicate inspection_id values")
    mapping = dict(
        zip(
            identity.get_column("inspection_id").to_list(),
            identity.get_column("restaurant_id").to_list(),
            strict=True,
        )
    )
    missing = [i for i in ids if i not in mapping]
    if missing:
        raise ValueError(
            f"inspection_id(s) missing from identity frame, cannot align restaurant "
            f"ids: {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    return np.array([mapping[i] for i in ids])


def assert_bootstrap_inputs_aligned(
    X: np.ndarray,
    y: np.ndarray,
    inspection_ids: list[str] | np.ndarray,
    restaurant_ids: np.ndarray,
) -> None:
    """Guard called before bootstrapping: every one of ``X``, ``y``,
    ``inspection_ids``, and ``restaurant_ids`` must be the same length, and
    ``inspection_ids`` must contain no duplicate and no missing entry."""
    ids = list(inspection_ids)
    lengths = {
        "X": len(X),
        "y": len(y),
        "inspection_ids": len(ids),
        "restaurant_ids": len(restaurant_ids),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f"misaligned bootstrap input lengths: {lengths}")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate inspection_id in bootstrap training inputs")


@dataclass(frozen=True)
class BootstrapMember:
    member_index: int
    seed: int
    estimator: Any
    training_restaurant_count: int
    training_row_count: int
    success: bool
    calibration_status: str | None
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
    weighted: bool = False,
    base_seed: int = RANDOM_SEED,
) -> tuple[list[BootstrapMember], UncertaintyConfig]:
    """Restaurant-cluster bootstrap: resample distinct restaurant IDs with
    replacement, include ALL of each drawn restaurant's rows (duplicated per
    draw), preserve internal chronology, and never resample validation.

    ``weighted`` must match the selected candidate's chosen weighting
    (``SelectedModelConfiguration.weighted``) -- every member reproduces the
    same weighting decision that was actually selected, never a generic
    unweighted refit.
    """
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
                    member_index=i,
                    seed=seed,
                    estimator=None,
                    training_restaurant_count=len(sampled),
                    training_row_count=len(indices),
                    success=False,
                    calibration_status=None,
                    failure_reason=reason,
                )
            )
            continue
        try:
            estimator = build_fn()
            estimator = _fit_with_optional_weight(estimator, X_boot, y_boot, weighted)
            calibrated = calibrate_on_validation(estimator, X_val, y_val)
            members.append(
                BootstrapMember(
                    member_index=i,
                    seed=seed,
                    estimator=calibrated.estimator,
                    training_restaurant_count=len(sampled),
                    training_row_count=len(indices),
                    success=True,
                    calibration_status=calibrated.report.status.value,
                    failure_reason=None,
                )
            )
        except Exception as exc:  # noqa: BLE001 - a failed member is recorded, not fatal
            reason = type(exc).__name__
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            members.append(
                BootstrapMember(
                    member_index=i,
                    seed=seed,
                    estimator=None,
                    training_restaurant_count=len(sampled),
                    training_row_count=len(indices),
                    success=False,
                    calibration_status=None,
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


def serialize_bootstrap_members(members: list[BootstrapMember]) -> list[dict[str, Any]]:
    """Persist every member (success or failure) with its full metadata --
    never just a bare list of anonymous estimators. A failed member's
    ``estimator`` field is ``None``; it is still present in the output with
    its seed, index, and failure reason."""
    return [
        {
            "member_index": m.member_index,
            "seed": m.seed,
            "success": m.success,
            "estimator": m.estimator,
            "calibration_status": m.calibration_status,
            "training_restaurant_count": m.training_restaurant_count,
            "training_row_count": m.training_row_count,
            "failure_reason": m.failure_reason,
        }
        for m in members
    ]


def deserialize_bootstrap_members(records: list[dict[str, Any]]) -> list[BootstrapMember]:
    return [
        BootstrapMember(
            member_index=r["member_index"],
            seed=r["seed"],
            estimator=r["estimator"],
            training_restaurant_count=r["training_restaurant_count"],
            training_row_count=r["training_row_count"],
            success=r["success"],
            calibration_status=r["calibration_status"],
            failure_reason=r["failure_reason"],
        )
        for r in records
    ]


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
# Deployment readiness                                                        #
# --------------------------------------------------------------------------- #


class ModelReadinessStatus(StrEnum):
    """``_SUCCESS`` means only "artifact write completed and checksums are
    valid." It never means "safe for production." This status is the
    explicit, separate signal for that."""

    READY = "ready"
    INSUFFICIENT_PERFORMANCE = "insufficient_performance"
    UNCALIBRATED = "uncalibrated"
    INSUFFICIENT_UNCERTAINTY = "insufficient_uncertainty"
    INSUFFICIENT_DATA = "insufficient_data"
    EVALUATION_ONLY = "evaluation_only"


_CALIBRATED_STATUSES = frozenset(
    {CalibrationStatus.CALIBRATED_SIGMOID.value, CalibrationStatus.CALIBRATED_ISOTONIC.value}
)


def determine_readiness(
    *,
    model_status: str,
    calibration_status: str,
    uncertainty_status: str,
    feature_order: tuple[str, ...],
    split_report: ChronologicalSplitReport,
    test_metrics_computed: bool,
) -> tuple[ModelReadinessStatus, str]:
    """A model may be READY only when every one of these independently holds:
    it materially beat the prevalence baseline, calibration succeeded, the
    required number of bootstrap members succeeded, the feature schema is
    present, train/validation/test partitions all passed integrity checks
    (nonempty; train/validation contain both classes -- enforced upstream by
    ``chronological_split``), and test metrics were actually computed.
    """
    if not feature_order:
        return ModelReadinessStatus.INSUFFICIENT_DATA, "no feature schema is present"
    for name, summary in (
        ("train", split_report.train),
        ("validation", split_report.validation),
        ("test", split_report.test),
    ):
        if summary.row_count == 0:
            return ModelReadinessStatus.INSUFFICIENT_DATA, f"the {name} partition is empty"
    if model_status != "validated":
        return (
            ModelReadinessStatus.INSUFFICIENT_PERFORMANCE,
            "the selected candidate did not materially beat the prevalence baseline",
        )
    if calibration_status not in _CALIBRATED_STATUSES:
        return (
            ModelReadinessStatus.UNCALIBRATED,
            f"calibration did not succeed (status={calibration_status})",
        )
    if uncertainty_status != UncertaintyStatus.AVAILABLE.value:
        return (
            ModelReadinessStatus.INSUFFICIENT_UNCERTAINTY,
            "too few bootstrap members succeeded to report an uncertainty interval",
        )
    if not test_metrics_computed:
        return ModelReadinessStatus.EVALUATION_ONLY, "test metrics were not computed"
    return ModelReadinessStatus.READY, "all readiness checks passed"


# --------------------------------------------------------------------------- #
# Risk policy per jurisdiction                                                #
# --------------------------------------------------------------------------- #

ARTIFACT_SCHEMA_VERSION = "1.0.0"
SOURCE_PROVENANCE_UNAVAILABLE: dict[str, Any] = {"status": "unavailable"}

_NYC_HISTORY_SUFFICIENCY_RULE: dict[str, Any] = {
    "jurisdiction": "nyc",
    "rules": ["history_depth >= 1", "nyc_prior_valid_score_count >= 1"],
}
_FLORIDA_HISTORY_SUFFICIENCY_RULE: dict[str, Any] = {
    "jurisdiction": "florida",
    "rules": ["history_depth >= 1", "fl_prior_valid_high_priority_count >= 1"],
}


def history_sufficiency_rule_for(jurisdiction: str) -> dict[str, Any]:
    """Jurisdiction-specific sufficient-history rule, matching
    ``plateproof.models.risk_bands.has_sufficient_history``. Deliberately
    keyed by jurisdiction with no shared/derived fallback, so a Florida
    artifact can never end up carrying NYC's rule or vice versa."""
    if jurisdiction == "nyc":
        return _NYC_HISTORY_SUFFICIENCY_RULE
    if jurisdiction == "florida":
        return _FLORIDA_HISTORY_SUFFICIENCY_RULE
    raise ValueError(f"unknown jurisdiction: {jurisdiction!r}")


def library_versions() -> dict[str, str]:
    import sys

    import numpy
    import polars
    import sklearn

    return {
        "python": sys.version.split()[0],
        "scikit_learn": sklearn.__version__,
        "polars": polars.__version__,
        "numpy": numpy.__version__,
    }


def feature_dtypes(feature_order: tuple[str, ...]) -> dict[str, str]:
    """Feature name -> declared dtype, from Task 5's ``FEATURE_SCHEMA``. A
    name absent from that schema (only possible for a non-production,
    synthetic-test feature list) is recorded as ``"unknown"`` rather than
    raising, since this also runs against test fixtures."""
    from plateproof.features.temporal import FEATURE_SCHEMA

    return {
        name: str(FEATURE_SCHEMA[name]) if name in FEATURE_SCHEMA else "unknown"
        for name in feature_order
    }


# --------------------------------------------------------------------------- #
# Model card                                                                   #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelCardContext:
    model_version: str
    jurisdiction: str
    target_name: str
    target_definition: str
    eligible_inspection_types: list[str]
    excluded_inspection_types: list[str]
    feature_categories: list[str]
    split_boundaries: SplitBoundaries
    selection: Any
    selected_config: SelectedModelConfiguration
    calibration_status: str
    risk_band_thresholds: RiskBandThresholds
    uncertainty_config: UncertaintyConfig
    train_metrics: EvaluationMetrics
    validation_metrics: EvaluationMetrics
    test_metrics: EvaluationMetrics
    subgroup_limitations: str
    data_snapshot_note: str
    known_limitations: list[str]
    readiness_status: ModelReadinessStatus
    readiness_reason: str


def generate_model_card(ctx: ModelCardContext) -> str:
    """Markdown model card. States plainly, at the top, when the artifact is
    not approved for production prediction -- an insufficient-performance or
    uncalibrated model's card is never silent about that."""
    lines: list[str] = [
        f"# PlateProof Model Card -- {ctx.jurisdiction.upper()} / {ctx.target_name}",
        "",
        f"- Model version: `{ctx.model_version}`",
        f"- Jurisdiction: `{ctx.jurisdiction}`",
        "",
        "## Deployment status",
        f"- Status: `{ctx.readiness_status.value}`",
        f"- Reason: {ctx.readiness_reason}",
    ]
    if ctx.readiness_status != ModelReadinessStatus.READY:
        lines += [
            "",
            f"> **This artifact is NOT APPROVED FOR PRODUCTION prediction** "
            f"(status: `{ctx.readiness_status.value}`). {ctx.readiness_reason}.",
        ]
    lines += [
        "",
        "## Target definition",
        ctx.target_definition,
        "",
        "### Eligible inspection types",
        *(f"- {t}" for t in ctx.eligible_inspection_types),
        "",
        "### Excluded inspection types",
        *(f"- {t}" for t in ctx.excluded_inspection_types),
        "",
        "## Feature categories",
        *(f"- {c}" for c in ctx.feature_categories),
        "",
        "## Temporal leakage protections",
        "Every feature is computed from information strictly earlier than its own prediction "
        "event's inspection date, restricted to an explicit model-feature allowlist, with "
        "same-day events collapsed before any historical shift/window computation.",
        "",
        "## Chronological split",
        f"- Train end: {ctx.split_boundaries.train_end}",
        f"- Validation end: {ctx.split_boundaries.validation_end}",
        f"- Test end: {ctx.split_boundaries.test_end}",
        "",
        "## Candidate comparison",
        f"- Prevalence baseline validation AP: {ctx.selection.prevalence_ap:.4f}",
        f"- Logistic regression validation AP: {ctx.selection.logistic_ap:.4f} "
        f"(weighted={ctx.selection.logistic_weighted})",
        f"- Hist gradient boosting validation AP: {ctx.selection.hgb_ap:.4f} "
        f"(weighted={ctx.selection.hgb_weighted})",
        f"- Selected candidate: **{ctx.selection.selected_candidate}**",
        "",
        "## Selected estimator",
        f"- Candidate: {ctx.selected_config.candidate_name}",
        f"- Weighted: {ctx.selected_config.weighted}",
        f"- Preprocessing: {ctx.selected_config.preprocessing_description}",
        f"- Random seed: {ctx.selected_config.random_seed}",
        "",
        "## Calibration",
        f"- Status: {ctx.calibration_status}",
        "- Calibrated on the validation partition only.",
        "",
        "## Risk bands",
        f"- Low: probability < {ctx.risk_band_thresholds.low_upper_bound}",
        f"- Moderate: {ctx.risk_band_thresholds.low_upper_bound} <= probability < "
        f"{ctx.risk_band_thresholds.moderate_upper_bound}",
        f"- High: probability >= {ctx.risk_band_thresholds.moderate_upper_bound}",
        "- Insufficient history: no probability or interval is produced.",
        f"- {ctx.risk_band_thresholds.policy}",
        "",
        "## Bootstrap uncertainty",
        f"- Method: {ctx.uncertainty_config.method}",
        f"- Successful members: {ctx.uncertainty_config.successful_members} / "
        f"{ctx.uncertainty_config.requested_members} "
        f"(minimum required: {ctx.uncertainty_config.min_successful_required})",
        f"- Status: {ctx.uncertainty_config.status.value}",
        "",
        "## Evaluation metrics",
        f"- Train average precision: {ctx.train_metrics.average_precision}",
        f"- Validation average precision: {ctx.validation_metrics.average_precision}",
        f"- Test average precision: {ctx.test_metrics.average_precision}",
        "",
        "## Subgroup limitations",
        ctx.subgroup_limitations,
        "",
        "## Data snapshot provenance",
        ctx.data_snapshot_note,
        "",
        "## Known limitations",
        *(f"- {k}" for k in ctx.known_limitations),
        "",
        "## Important notices",
        "- Non-causal: this model is predictive and associative only, and does not identify a "
        "cause of any outcome.",
        "- This score is a PlateProof prediction, not an official inspection result.",
        "- Michelin recognition, where shown elsewhere in the product, is contextual dining "
        "metadata only, plays no role in this model, and does not imply food safety.",
        "- This model was never trained, tested, validated, or fine-tuned using Google Maps "
        "content, and no Google integration is required to use it.",
    ]
    return "\n".join(lines)


def assemble_production_bundle(
    *,
    jurisdiction: Literal["nyc", "florida"],
    target_name: str,
    model_version: str,
    feature_order: tuple[str, ...],
    target_definition: str,
    eligible_inspection_types: list[str],
    excluded_inspection_types: list[str],
    feature_categories: list[str],
    split_boundaries: SplitBoundaries,
    split_report: ChronologicalSplitReport,
    target_build_report: Any | None,
    selection: Any,
    selected_config: SelectedModelConfiguration,
    calibration_outcome: CalibrationOutcome,
    train_metrics: EvaluationMetrics,
    validation_metrics: EvaluationMetrics,
    test_metrics: EvaluationMetrics,
    test_metrics_computed: bool,
    members: list[BootstrapMember],
    uncertainty_config: UncertaintyConfig,
    subgroup_limitations: str,
    data_snapshot_note: str,
    known_limitations: list[str],
    source_provenance: dict[str, Any] | None,
    random_seeds: dict[str, int],
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Assemble the complete production artifact bundle for
    ``write_artifact``. Everything Task 6 requires the bundle to carry is
    produced here: readiness status, risk-band policy, model card, versions,
    and every intermediate report -- so the CLI never has to remember to
    include a required field, and tests can exercise this without a full
    training run.
    """
    readiness_status, readiness_reason = determine_readiness(
        model_status=selection.model_status,
        calibration_status=calibration_outcome.report.status.value,
        uncertainty_status=uncertainty_config.status.value,
        feature_order=feature_order,
        split_report=split_report,
        test_metrics_computed=test_metrics_computed,
    )
    risk_band_thresholds = RiskBandThresholds(
        jurisdiction=jurisdiction, model_version=model_version
    )
    history_rule = history_sufficiency_rule_for(jurisdiction)

    card_context = ModelCardContext(
        model_version=model_version,
        jurisdiction=jurisdiction,
        target_name=target_name,
        target_definition=target_definition,
        eligible_inspection_types=eligible_inspection_types,
        excluded_inspection_types=excluded_inspection_types,
        feature_categories=feature_categories,
        split_boundaries=split_boundaries,
        selection=selection,
        selected_config=selected_config,
        calibration_status=calibration_outcome.report.status.value,
        risk_band_thresholds=risk_band_thresholds,
        uncertainty_config=uncertainty_config,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        subgroup_limitations=subgroup_limitations,
        data_snapshot_note=data_snapshot_note,
        known_limitations=known_limitations,
        readiness_status=readiness_status,
        readiness_reason=readiness_reason,
    )

    return {
        "point_estimator.joblib": calibration_outcome.estimator,
        "bootstrap_members.joblib": serialize_bootstrap_members(members),
        "feature_order.json": list(feature_order),
        "feature_dtypes.json": feature_dtypes(feature_order),
        "jurisdiction.json": jurisdiction,
        "target_definition.json": {"target_name": target_name, "definition": target_definition},
        "selection_report.json": selection.model_dump(exclude={"selected_estimator"}),
        "selected_configuration.json": selected_config.model_dump(),
        "calibration_report.json": {
            "status": calibration_outcome.report.status.value,
            "method": calibration_outcome.report.method,
            "brier_before": calibration_outcome.report.brier_before,
            "brier_after": calibration_outcome.report.brier_after,
            "validation_row_count": calibration_outcome.report.validation_row_count,
        },
        "uncertainty_config.json": uncertainty_config.model_dump(),
        "split_report.json": split_report.model_dump(mode="json"),
        "target_build_report.json": target_build_report.model_dump()
        if target_build_report is not None
        else None,
        "metrics.json": {
            "train": train_metrics.model_dump(),
            "validation": validation_metrics.model_dump(),
            "test": test_metrics.model_dump(),
        },
        "risk_band_thresholds.json": risk_band_thresholds.model_dump(),
        "history_sufficiency_rule.json": history_rule,
        "deployment_status.json": {
            "status": readiness_status.value,
            "reason": readiness_reason,
        },
        "environment.json": {**library_versions(), "random_seeds": random_seeds},
        "source_provenance.json": source_provenance
        if source_provenance is not None
        else dict(SOURCE_PROVENANCE_UNAVAILABLE),
        "artifact_schema_version.json": ARTIFACT_SCHEMA_VERSION,
        "model_card.md": generate_model_card(card_context),
        "generated_at.json": (generated_at or datetime.now(UTC)).isoformat(),
    }


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


def _is_text_name(name: str) -> bool:
    return name.endswith((".md", ".txt"))


def _serialize_one(path: Path, value: Any, *, final_name: str) -> None:
    if _is_json_name(final_name):
        path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8")
    elif _is_text_name(final_name):
        path.write_text(str(value), encoding="utf-8")
    else:
        joblib.dump(value, path)


def _deserialize_one(path: Path) -> Any:
    if _is_json_name(path.name):
        return json.loads(path.read_text(encoding="utf-8"))
    if _is_text_name(path.name):
        return path.read_text(encoding="utf-8")
    return joblib.load(path)


def _write_bundle_to_dir(
    target_dir: Path,
    *,
    jurisdiction: str,
    target_name: str,
    model_version: str,
    bundle: dict[str, Any],
) -> None:
    """Write every ``bundle`` entry to ``*.part``, hash the finalized files into
    ``manifest.json``, then write ``_SUCCESS`` last. Any failure removes all
    temp and finalized files, leaving no apparently-complete artifact in
    ``target_dir``. ``target_dir`` must not already exist as a completed
    artifact -- callers are responsible for directory placement/overwrite
    policy (see ``write_artifact``)."""
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


def write_artifact(
    output_dir: str | Path,
    *,
    jurisdiction: str,
    target_name: str,
    model_version: str,
    bundle: dict[str, Any],
    force: bool = False,
) -> Path:
    """Write a complete, checksummed artifact to
    ``<output_dir>/<jurisdiction>/<target_name>/<model_version>``.

    Refuses to overwrite a directory already marked ``_SUCCESS`` unless
    ``force=True``. When forcing a replacement, the new artifact is written
    to and fully validated in a sibling temporary directory *before* the
    previous complete artifact is touched; only a validated replacement is
    swapped into the final location (Windows-safe: rename, never an
    in-place directory delete-then-write). If writing, validating, or
    swapping the replacement fails, the previous complete artifact is left
    exactly as it was.
    """
    target_dir = Path(output_dir) / jurisdiction / target_name / model_version
    already_complete = (target_dir / "_SUCCESS").exists()

    if not already_complete:
        _write_bundle_to_dir(
            target_dir,
            jurisdiction=jurisdiction,
            target_name=target_name,
            model_version=model_version,
            bundle=bundle,
        )
        return target_dir

    if not force:
        raise FileExistsError(
            f"artifact already complete at {target_dir}; pass force=True to overwrite"
        )

    token = uuid.uuid4().hex
    replacement_dir = target_dir.parent / f"{model_version}.replacing-{token}"
    backup_dir = target_dir.parent / f"{model_version}.previous-{token}"
    try:
        _write_bundle_to_dir(
            replacement_dir,
            jurisdiction=jurisdiction,
            target_name=target_name,
            model_version=model_version,
            bundle=bundle,
        )
        if not is_valid_model_artifact(replacement_dir) or verify_artifact_checksums(
            replacement_dir
        ):
            raise ValueError(f"replacement artifact failed validation at {replacement_dir}")
    except BaseException:
        shutil.rmtree(replacement_dir, ignore_errors=True)
        raise

    target_dir.rename(backup_dir)
    try:
        replacement_dir.rename(target_dir)
    except BaseException:
        # Restore the previous complete artifact -- the swap itself failed,
        # so the caller must not be left with neither a valid old nor new one.
        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
        backup_dir.rename(target_dir)
        shutil.rmtree(replacement_dir, ignore_errors=True)
        raise
    shutil.rmtree(backup_dir, ignore_errors=True)
    return target_dir


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
    path: Path,
    *,
    trusted: bool,
    expected_jurisdiction: str | None = None,
    require_ready: bool = True,
) -> dict[str, Any]:
    """Load a model artifact. ``joblib``/pickle deserialization executes
    arbitrary code for untrusted input -- this refuses unless ``trusted=True``
    is passed explicitly. Checksums only catch accidental corruption; an
    attacker able to replace both a file and its manifest entry defeats them,
    so this is not a security boundary against a compromised artifact source.

    ``_SUCCESS`` and valid checksums mean only that the artifact write
    completed intact -- never that the model is safe for production. When the
    artifact records a ``deployment_status.json`` and ``require_ready=True``
    (the default), this refuses to load anything but a ``ready`` model. Pass
    ``require_ready=False`` only for deliberate audit/evaluation use of a
    known non-ready artifact. An artifact with no recorded deployment status
    (e.g. a non-model artifact in tests) is unaffected by this check.
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

    status_record = loaded.get("deployment_status.json")
    if require_ready and isinstance(status_record, dict) and "status" in status_record:
        if status_record["status"] != ModelReadinessStatus.READY.value:
            raise ValueError(
                f"refusing to load a non-ready model artifact "
                f"(status={status_record['status']!r}, reason={status_record.get('reason')!r}); "
                "pass require_ready=False for deliberate audit/evaluation use only"
            )
    return loaded
