"""Tests for the persisted restaurant-cluster bootstrap uncertainty ensemble."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest


def _toy_data(n_restaurants: int = 20, rows_per: int = 4, seed: int = 0) -> Any:
    rng = np.random.RandomState(seed)
    restaurant_ids = np.repeat(np.arange(n_restaurants), rows_per)
    X = rng.rand(n_restaurants * rows_per, 2)
    y = (X[:, 0] + rng.rand(n_restaurants * rows_per) * 0.3 > 0.6).astype(int)
    return restaurant_ids, X, y


def test_bootstrap_is_deterministic_for_fixed_seed() -> None:
    from plateproof.models.training import build_logistic_pipeline, fit_bootstrap_ensemble

    restaurant_ids, X, y = _toy_data()
    Xval, yval = X[:20], y[:20]
    members_a, config_a = fit_bootstrap_ensemble(
        build_logistic_pipeline, X, y, restaurant_ids, Xval, yval, n_members=5, min_successful=3
    )
    members_b, config_b = fit_bootstrap_ensemble(
        build_logistic_pipeline, X, y, restaurant_ids, Xval, yval, n_members=5, min_successful=3
    )
    probs_a = [m.estimator.predict_proba(Xval[:1])[0, 1] for m in members_a if m.success]
    probs_b = [m.estimator.predict_proba(Xval[:1])[0, 1] for m in members_b if m.success]
    assert probs_a == pytest.approx(probs_b)
    assert config_a.successful_members == config_b.successful_members


def test_bootstrap_preserves_restaurant_clusters(monkeypatch: Any) -> None:
    from plateproof.models import training as training_module

    restaurant_ids, X, y = _toy_data(n_restaurants=10, rows_per=3)
    captured: list[Any] = []
    real_fit = training_module.build_logistic_pipeline

    class _Capturing:
        def __init__(self) -> None:
            self._inner = real_fit()

        def fit(self, X_boot: Any, y_boot: Any, **kwargs: Any) -> Any:
            captured.append(X_boot.shape[0])
            return self._inner.fit(X_boot, y_boot, **kwargs)

        def predict_proba(self, X: Any) -> Any:
            return self._inner.predict_proba(X)

    training_module.fit_bootstrap_ensemble(
        _Capturing, X, y, restaurant_ids, X[:6], y[:6], n_members=3, min_successful=1
    )
    # Every resample's row count is a multiple of rows_per_restaurant (3) since
    # whole restaurants are included, never partial/individual rows.
    assert all(count % 3 == 0 for count in captured)


def test_interval_ordering_and_bounds() -> None:
    from plateproof.models.training import (
        build_logistic_pipeline,
        fit_bootstrap_ensemble,
        predict_with_uncertainty,
    )

    restaurant_ids, X, y = _toy_data(n_restaurants=30, rows_per=4)
    Xval, yval = X[:30], y[:30]
    point_model = build_logistic_pipeline().fit(X, y)
    members, config = fit_bootstrap_ensemble(
        build_logistic_pipeline, X, y, restaurant_ids, Xval, yval, n_members=5, min_successful=3
    )
    for row in X[:10]:
        point, lower, upper = predict_with_uncertainty(
            point_model, members, config, row.reshape(1, -1)
        )
        assert 0.0 <= lower <= point <= upper <= 1.0


def test_too_few_successful_members_yields_insufficient_status() -> None:
    from plateproof.models.training import UncertaintyStatus, fit_bootstrap_ensemble

    restaurant_ids, X, y = _toy_data(n_restaurants=6, rows_per=2)
    Xval, yval = X[:6], y[:6]

    def _always_fails() -> Any:
        class _Bad:
            def fit(self, X: Any, y: Any, **kwargs: Any) -> Any:
                raise RuntimeError("forced failure")

        return _Bad()

    members, config = fit_bootstrap_ensemble(
        _always_fails, X, y, restaurant_ids, Xval, yval, n_members=5, min_successful=3
    )
    assert config.status == UncertaintyStatus.INSUFFICIENT_BOOTSTRAP_MEMBERS
    assert config.successful_members == 0
    assert config.failed_members == 5


def test_insufficient_uncertainty_never_fabricates_wide_default_interval() -> None:
    from plateproof.models.training import (
        UncertaintyStatus,
        build_logistic_pipeline,
        predict_with_uncertainty,
    )

    restaurant_ids, X, y = _toy_data(n_restaurants=6, rows_per=2)
    point_model = build_logistic_pipeline().fit(X, y)

    class _FakeConfig:
        status = UncertaintyStatus.INSUFFICIENT_BOOTSTRAP_MEMBERS

    point, lower, upper = predict_with_uncertainty(point_model, [], _FakeConfig(), X[:1])
    assert lower is None
    assert upper is None
    assert point is not None
