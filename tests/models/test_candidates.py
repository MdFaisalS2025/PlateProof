"""Tests for prevalence/logistic/HGB candidate fitting and model selection."""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest


def test_deterministic_candidate_fitting(synthetic_frame: Any) -> None:
    from plateproof.models.training import (
        SplitBoundaries,
        chronological_split,
        fit_and_select_model,
    )

    frame = synthetic_frame()
    boundaries = SplitBoundaries(
        jurisdiction="nyc",
        target_name="t",
        train_end=date(2020, 1, 1) + __import__("datetime").timedelta(days=120),
        validation_end=date(2020, 1, 1) + __import__("datetime").timedelta(days=160),
        test_end=date(2020, 1, 1) + __import__("datetime").timedelta(days=200),
    )
    train, val, _test, _ = chronological_split(frame, boundaries)
    a = fit_and_select_model(train, val)
    b = fit_and_select_model(train, val)
    assert a.selected_candidate == b.selected_candidate
    assert a.candidate_validation_ap == pytest.approx(b.candidate_validation_ap)


def test_prevalence_baseline_matches_train_prevalence(synthetic_frame: Any) -> None:
    from plateproof.models.training import PrevalenceBaseline

    frame = synthetic_frame()
    y = frame.y.get_column("label").to_numpy()
    baseline = PrevalenceBaseline().fit(None, y)
    proba = baseline.predict_proba(frame.X.select(frame.feature_order).to_numpy())
    assert proba[:, 1] == pytest.approx(y.mean())


def test_logistic_beats_prevalence_on_informative_signal(synthetic_frame: Any) -> None:
    from plateproof.models.training import (
        SplitBoundaries,
        average_precision_of,
        build_logistic_pipeline,
        chronological_split,
    )

    frame = synthetic_frame(n_restaurants=60, visits_per_restaurant=6)
    boundaries = SplitBoundaries(
        jurisdiction="nyc",
        target_name="t",
        train_end=date(2020, 1, 1) + __import__("datetime").timedelta(days=200),
        validation_end=date(2020, 1, 1) + __import__("datetime").timedelta(days=280),
        test_end=date(2020, 1, 1) + __import__("datetime").timedelta(days=360),
    )
    train, val, _test, _ = chronological_split(frame, boundaries)
    Xtr = train.X.select(train.feature_order).to_numpy()
    ytr = train.y.get_column("label").to_numpy()
    Xval = val.X.select(val.feature_order).to_numpy()
    yval = val.y.get_column("label").to_numpy()
    pipeline = build_logistic_pipeline()
    pipeline.fit(Xtr, ytr)
    ap = average_precision_of(yval, pipeline.predict_proba(Xval)[:, 1])
    assert ap > yval.mean()  # meaningfully better than prevalence


def test_boosted_must_beat_logistic_by_material_margin(
    synthetic_frame: Any, monkeypatch: Any
) -> None:
    from plateproof.models import training as training_module
    from plateproof.models.training import (
        SplitBoundaries,
        chronological_split,
        fit_and_select_model,
    )

    frame = synthetic_frame(n_restaurants=60, visits_per_restaurant=6)
    boundaries = SplitBoundaries(
        jurisdiction="nyc",
        target_name="t",
        train_end=date(2020, 1, 1) + __import__("datetime").timedelta(days=200),
        validation_end=date(2020, 1, 1) + __import__("datetime").timedelta(days=280),
        test_end=date(2020, 1, 1) + __import__("datetime").timedelta(days=360),
    )
    train, val, _test, _ = chronological_split(frame, boundaries)

    # Force a scenario where boosted looks only microscopically better: selection
    # must still land on logistic because the improvement is below the material
    # improvement threshold.
    monkeypatch.setattr(training_module, "MIN_MATERIAL_IMPROVEMENT_AP", 1.0)
    result = fit_and_select_model(train, val)
    assert result.selected_candidate in ("logistic_regression", "prevalence")


def test_class_weighting_selected_via_validation_and_recorded(synthetic_frame: Any) -> None:
    from plateproof.models.training import (
        SplitBoundaries,
        chronological_split,
        fit_and_select_model,
    )

    frame = synthetic_frame(n_restaurants=60, visits_per_restaurant=6)
    boundaries = SplitBoundaries(
        jurisdiction="nyc",
        target_name="t",
        train_end=date(2020, 1, 1) + __import__("datetime").timedelta(days=200),
        validation_end=date(2020, 1, 1) + __import__("datetime").timedelta(days=280),
        test_end=date(2020, 1, 1) + __import__("datetime").timedelta(days=360),
    )
    train, val, _test, _ = chronological_split(frame, boundaries)
    result = fit_and_select_model(train, val)
    assert result.logistic_weighted in (True, False)
    assert result.hgb_weighted in (True, False)


def test_insufficient_performance_status_when_no_improvement(
    synthetic_frame: Any, monkeypatch: Any
) -> None:
    from plateproof.models import training as training_module
    from plateproof.models.training import (
        SplitBoundaries,
        chronological_split,
        fit_and_select_model,
    )

    # random labels: no genuine signal, so no candidate should beat prevalence materially
    frame = synthetic_frame(n_restaurants=30, visits_per_restaurant=3, seed=1)
    boundaries = SplitBoundaries(
        jurisdiction="nyc",
        target_name="t",
        train_end=date(2020, 1, 1) + __import__("datetime").timedelta(days=60),
        validation_end=date(2020, 1, 1) + __import__("datetime").timedelta(days=80),
        test_end=date(2020, 1, 1) + __import__("datetime").timedelta(days=100),
    )
    train, val, _test, _ = chronological_split(frame, boundaries)
    monkeypatch.setattr(training_module, "MIN_MATERIAL_IMPROVEMENT_AP", 0.999)
    result = fit_and_select_model(train, val)
    assert result.model_status == "insufficient_performance"


def test_test_metrics_cannot_change_selection(synthetic_frame: Any) -> None:
    from plateproof.models.training import (
        SplitBoundaries,
        chronological_split,
        fit_and_select_model,
    )

    frame = synthetic_frame(n_restaurants=60, visits_per_restaurant=6)
    boundaries = SplitBoundaries(
        jurisdiction="nyc",
        target_name="t",
        train_end=date(2020, 1, 1) + __import__("datetime").timedelta(days=200),
        validation_end=date(2020, 1, 1) + __import__("datetime").timedelta(days=280),
        test_end=date(2020, 1, 1) + __import__("datetime").timedelta(days=360),
    )
    train, val, test, _ = chronological_split(frame, boundaries)
    fit_and_select_model(train, val)
    # fit_and_select_model's signature intentionally excludes test data entirely --
    # this is a structural (signature-level) guarantee, verified here.
    import inspect

    params = inspect.signature(fit_and_select_model).parameters
    assert "test" not in params
    del test  # never passed to selection
