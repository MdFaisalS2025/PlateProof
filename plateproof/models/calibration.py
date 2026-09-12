"""Post-hoc probability calibration on the validation partition only.

Uses ``sklearn.frozen.FrozenEstimator`` (verified present in the installed
scikit-learn 1.9.1) to wrap an already-fitted base estimator so
``CalibratedClassifierCV`` calibrates without refitting or ever touching the
train partition -- this replaces the deprecated ``cv="prefit"`` API.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss

MIN_VALIDATION_ROWS_FOR_CALIBRATION = 30
MIN_VALIDATION_ROWS_FOR_ISOTONIC = 1000
MIN_CLASS_ROWS_FOR_ISOTONIC = 400


class CalibrationStatus(StrEnum):
    CALIBRATED_SIGMOID = "calibrated_sigmoid"
    CALIBRATED_ISOTONIC = "calibrated_isotonic"
    UNCALIBRATED_INSUFFICIENT_DATA = "uncalibrated_insufficient_data"
    UNCALIBRATED_SINGLE_CLASS = "uncalibrated_single_class"


@dataclass(frozen=True)
class CalibrationReport:
    status: CalibrationStatus
    method: str | None
    brier_before: float
    brier_after: float
    validation_row_count: int


class _RawFallbackEstimator:
    """Wraps a fitted estimator so ``.predict_proba`` works unchanged when
    calibration is skipped -- callers never need to branch on status."""

    def __init__(self, estimator: Any) -> None:
        self._estimator = estimator

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._estimator.predict_proba(X)  # type: ignore[no-any-return]


@dataclass(frozen=True)
class CalibrationOutcome:
    report: CalibrationReport
    estimator: Any


def calibrate_on_validation(
    fitted_estimator: Any, X_val: np.ndarray, y_val: np.ndarray
) -> CalibrationOutcome:
    """Calibrate ``fitted_estimator`` (already fit on train) using validation
    only. Falls back to the raw estimator with an explicit uncalibrated
    status when validation is too small or single-class -- never fabricates
    a calibration."""
    y_val = np.asarray(y_val)
    n = len(y_val)
    raw_probs = fitted_estimator.predict_proba(X_val)[:, 1]
    brier_before = (
        float(brier_score_loss(y_val, raw_probs))
        if len(set(y_val.tolist())) > 1
        else float(brier_score_loss(y_val, raw_probs))
    )

    if len(set(y_val.tolist())) < 2:
        report = CalibrationReport(
            status=CalibrationStatus.UNCALIBRATED_SINGLE_CLASS,
            method=None,
            brier_before=brier_before,
            brier_after=brier_before,
            validation_row_count=n,
        )
        return CalibrationOutcome(report=report, estimator=_RawFallbackEstimator(fitted_estimator))

    if n < MIN_VALIDATION_ROWS_FOR_CALIBRATION:
        report = CalibrationReport(
            status=CalibrationStatus.UNCALIBRATED_INSUFFICIENT_DATA,
            method=None,
            brier_before=brier_before,
            brier_after=brier_before,
            validation_row_count=n,
        )
        return CalibrationOutcome(report=report, estimator=_RawFallbackEstimator(fitted_estimator))

    class_counts = np.bincount(y_val.astype(int))
    use_isotonic = (
        n >= MIN_VALIDATION_ROWS_FOR_ISOTONIC and class_counts.min() >= MIN_CLASS_ROWS_FOR_ISOTONIC
    )
    method = "isotonic" if use_isotonic else "sigmoid"

    calibrated = CalibratedClassifierCV(estimator=FrozenEstimator(fitted_estimator), method=method)
    calibrated.fit(X_val, y_val)
    calibrated_probs = calibrated.predict_proba(X_val)[:, 1]
    brier_after = float(brier_score_loss(y_val, calibrated_probs))

    status = (
        CalibrationStatus.CALIBRATED_ISOTONIC
        if use_isotonic
        else CalibrationStatus.CALIBRATED_SIGMOID
    )
    report = CalibrationReport(
        status=status,
        method=method,
        brier_before=brier_before,
        brier_after=brier_after,
        validation_row_count=n,
    )
    return CalibrationOutcome(report=report, estimator=calibrated)
