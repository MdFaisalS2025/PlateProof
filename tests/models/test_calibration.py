"""Tests for validation-only calibration."""

from __future__ import annotations

from typing import Any

import numpy as np


def _fit_base(Xtr: Any, ytr: Any) -> Any:
    from plateproof.models.training import build_logistic_pipeline

    pipeline = build_logistic_pipeline()
    pipeline.fit(Xtr, ytr)
    return pipeline


def test_calibrates_on_validation_only_sigmoid_by_default() -> None:
    from plateproof.models.calibration import CalibrationStatus, calibrate_on_validation

    rng = np.random.RandomState(0)
    Xtr = rng.rand(200, 2)
    ytr = (Xtr[:, 0] + rng.rand(200) * 0.3 > 0.6).astype(int)
    base = _fit_base(Xtr, ytr)
    Xval = rng.rand(60, 2)
    yval = (Xval[:, 0] + rng.rand(60) * 0.3 > 0.6).astype(int)
    outcome = calibrate_on_validation(base, Xval, yval)
    assert outcome.report.status == CalibrationStatus.CALIBRATED_SIGMOID
    assert outcome.report.brier_before is not None
    assert outcome.report.brier_after is not None


def test_isotonic_used_when_validation_large_enough() -> None:
    from plateproof.models.calibration import CalibrationStatus, calibrate_on_validation

    rng = np.random.RandomState(1)
    Xtr = rng.rand(500, 2)
    ytr = (Xtr[:, 0] + rng.rand(500) * 0.3 > 0.6).astype(int)
    base = _fit_base(Xtr, ytr)
    Xval = rng.rand(1200, 2)
    yval = (Xval[:, 0] + rng.rand(1200) * 0.3 > 0.6).astype(int)
    outcome = calibrate_on_validation(base, Xval, yval)
    assert outcome.report.status == CalibrationStatus.CALIBRATED_ISOTONIC


def test_uncalibrated_when_validation_too_small() -> None:
    from plateproof.models.calibration import CalibrationStatus, calibrate_on_validation

    rng = np.random.RandomState(2)
    Xtr = rng.rand(100, 2)
    ytr = (Xtr[:, 0] > 0.5).astype(int)
    base = _fit_base(Xtr, ytr)
    Xval = rng.rand(10, 2)
    yval = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    outcome = calibrate_on_validation(base, Xval, yval)
    assert outcome.report.status == CalibrationStatus.UNCALIBRATED_INSUFFICIENT_DATA
    assert outcome.report.method is None
    # raw fallback: predict_proba still callable, not calibrated
    assert outcome.estimator.predict_proba(Xval).shape == (10, 2)


def test_uncalibrated_when_validation_single_class() -> None:
    from plateproof.models.calibration import CalibrationStatus, calibrate_on_validation

    rng = np.random.RandomState(3)
    Xtr = rng.rand(100, 2)
    ytr = (Xtr[:, 0] > 0.5).astype(int)
    base = _fit_base(Xtr, ytr)
    Xval = rng.rand(50, 2)
    yval = np.zeros(50, dtype=int)
    outcome = calibrate_on_validation(base, Xval, yval)
    assert outcome.report.status == CalibrationStatus.UNCALIBRATED_SINGLE_CLASS


def test_calibration_never_touches_test_data() -> None:
    import inspect

    from plateproof.models.calibration import calibrate_on_validation

    params = inspect.signature(calibrate_on_validation).parameters
    assert "X_test" not in params and "y_test" not in params
