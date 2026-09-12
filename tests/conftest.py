"""Shared test fixtures.

Keeps configuration tests deterministic: they must not be influenced by the
developer's shell environment or by a repository-local ``.env`` file. The fixture
is opt-in (not autouse) so it does not change the working directory for tests
that legitimately rely on repository-relative paths.
"""

import os
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from plateproof.core.config import get_settings


@pytest.fixture
def isolated_settings_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[None]:
    """Strip ``PLATEPROOF_*`` env vars, drop the cache, and run from a clean cwd."""
    for key in list(os.environ):
        if key.startswith("PLATEPROOF_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# --------------------------------------------------------------------------- #
# Task 7 serving-layer builders -- shared by tests/serving and tests/api.     #
# Everything here is a tiny fictional fixture written to a temporary         #
# directory; no production data, no network access.                          #
# --------------------------------------------------------------------------- #


def _restaurant_row(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "restaurant_id": "nyc:1",
        "jurisdiction": "nyc",
        "source_id": "1",
        "name": "Anna's Kitchen",
        "normalized_name": "annas kitchen",
        "address": "100 Broadway",
        "city": "Manhattan",
        "region": "NY",
        "postal_code": "10001",
        "latitude": 40.7,
        "longitude": -73.9,
        "cuisine": "American",
        "latest_inspection_date": date(2025, 6, 1),
        "source_snapshot_date": date(2026, 1, 1),
        "source_retrieved_at_utc": datetime(2026, 1, 1, tzinfo=UTC),
        "source_filename": None,
        "source_url": None,
    }
    base.update(overrides)
    return base


def _inspection_row(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "inspection_id": "nyc:1:2025-06-01",
        "restaurant_id": "nyc:1",
        "jurisdiction": "nyc",
        "inspection_date": date(2025, 6, 1),
        "inspection_type": "Cycle Inspection / Initial Inspection",
        "score": 13.0,
        "grade": "A",
        "critical_violation_count": 1,
        "action": "Violations were cited.",
        "high_priority_count": None,
        "intermediate_count": None,
        "basic_count": None,
        "total_violation_count": None,
        "disposition_status": None,
        "source_snapshot_date": date(2026, 1, 1),
    }
    base.update(overrides)
    return base


def _violation_row(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "violation_event_id": "nyc:v:1",
        "inspection_id": "nyc:1:2025-06-01",
        "restaurant_id": "nyc:1",
        "inspection_date": date(2025, 6, 1),
        "violation_code": "04L",
        "violation_description": "Evidence of mice",
        "severity": "critical",
        "count": 1,
        "corrected_on_site": None,
    }
    base.update(overrides)
    return base


@pytest.fixture
def processed_dir(tmp_path: Path) -> Path:
    d = tmp_path / "data" / "processed"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def write_restaurants(processed_dir: Path) -> Any:
    def _write(rows: list[dict[str, Any]]) -> Path:
        path = processed_dir / "restaurants.parquet"
        pl.DataFrame(rows if rows else [_restaurant_row()][:0]).write_parquet(path)
        return path

    return _write


@pytest.fixture
def write_inspections(processed_dir: Path) -> Any:
    def _write(rows: list[dict[str, Any]]) -> Path:
        path = processed_dir / "inspection_events.parquet"
        pl.DataFrame(rows).write_parquet(path)
        return path

    return _write


@pytest.fixture
def write_violations(processed_dir: Path) -> Any:
    def _write(rows: list[dict[str, Any]]) -> Path:
        path = processed_dir / "violation_events.parquet"
        pl.DataFrame(rows).write_parquet(path)
        return path

    return _write


@pytest.fixture
def write_michelin(processed_dir: Path) -> Any:
    def _write(
        restaurants: list[dict[str, Any]],
        distinctions: list[dict[str, Any]],
        matches: list[dict[str, Any]],
    ) -> None:
        pl.DataFrame(restaurants).write_parquet(processed_dir / "michelin_restaurants.parquet")
        pl.DataFrame(distinctions).write_parquet(
            processed_dir / "michelin_distinction_events.parquet"
        )
        pl.DataFrame(matches).write_parquet(processed_dir / "restaurant_michelin_matches.parquet")

    return _write


@pytest.fixture
def restaurant_row() -> Any:
    return _restaurant_row


@pytest.fixture
def inspection_row() -> Any:
    return _inspection_row


@pytest.fixture
def violation_row() -> Any:
    return _violation_row


def _build_ready_artifact(
    tmp_path: Path,
    *,
    jurisdiction: str,
    target_name: str,
    feature_order: tuple[str, ...],
    model_version: str = "v1",
    n_restaurants: int = 60,
    visits_per_restaurant: int = 6,
    seed: int = 20260101,
) -> Path:
    """Build and write a small, fully synthetic *ready* model artifact for
    ``jurisdiction``/``target_name``, using the real feature-name schema so
    scoring/serving code can select those columns from a real as-of feature
    frame. Mirrors the Task 6 synthetic end-to-end recipe."""
    from datetime import date as _date
    from datetime import timedelta

    import numpy as np

    from plateproof.models.calibration import calibrate_on_validation
    from plateproof.models.training import (
        CANDIDATE_BUILDERS,
        SplitBoundaries,
        TrainingFrame,
        aligned_restaurant_ids,
        assemble_production_bundle,
        assert_bootstrap_inputs_aligned,
        build_selected_model_configuration,
        chronological_split,
        compute_metrics,
        feature_matrix,
        fit_and_select_model,
        fit_bootstrap_ensemble,
        write_artifact,
    )

    rng = np.random.RandomState(seed)
    start = _date(2020, 1, 1)
    rows_id, rows_restaurant, rows_date, rows_label = [], [], [], []
    feature_columns: dict[str, list[float]] = {name: [] for name in feature_order}
    counter = 0
    for r in range(n_restaurants):
        restaurant_id = f"{jurisdiction}:synthetic{r}"
        for _v in range(visits_per_restaurant):
            signal = rng.rand()
            label = int(rng.rand() < (0.1 + 0.7 * signal))
            rows_id.append(f"syn-{counter}")
            rows_restaurant.append(restaurant_id)
            rows_date.append(start + timedelta(days=counter))
            rows_label.append(label)
            for i, name in enumerate(feature_order):
                # deterministic, feature-index-dependent signal so the model
                # has something real to learn from
                value = signal if i == 0 else rng.rand()
                feature_columns[name].append(value)
            counter += 1

    X = pl.DataFrame({"inspection_id": rows_id, **feature_columns})
    y = pl.DataFrame({"inspection_id": rows_id, "label": rows_label})
    identity = pl.DataFrame(
        {
            "inspection_id": rows_id,
            "restaurant_id": rows_restaurant,
            "inspection_date": rows_date,
        }
    )
    frame = TrainingFrame(
        X=X,
        y=y,
        identity=identity,
        feature_order=feature_order,
        target_name=target_name,
        jurisdiction=jurisdiction,  # type: ignore[arg-type]
    )

    boundaries = SplitBoundaries(
        jurisdiction=jurisdiction,  # type: ignore[arg-type]
        target_name=target_name,
        train_end=start + timedelta(days=int(counter * 0.55)),
        validation_end=start + timedelta(days=int(counter * 0.75)),
        test_end=start + timedelta(days=counter),
    )
    train, validation, test, split_report = chronological_split(frame, boundaries)

    selection = fit_and_select_model(train, validation)
    selected_config = build_selected_model_configuration(selection, feature_order, random_seed=123)

    Xtrain = feature_matrix(train.X, train.feature_order)
    ytrain = train.y.get_column("label").to_numpy()
    Xval = feature_matrix(validation.X, validation.feature_order)
    yval = validation.y.get_column("label").to_numpy()
    Xtest = feature_matrix(test.X, test.feature_order)
    ytest = test.y.get_column("label").to_numpy()

    calibration = calibrate_on_validation(selection.selected_estimator, Xval, yval)
    train_metrics = compute_metrics(
        ytrain, calibration.estimator.predict_proba(Xtrain)[:, 1], partition="train"
    )
    validation_metrics = compute_metrics(
        yval, calibration.estimator.predict_proba(Xval)[:, 1], partition="validation"
    )
    test_metrics = compute_metrics(
        ytest, calibration.estimator.predict_proba(Xtest)[:, 1], partition="test"
    )

    train_ids = train.X.get_column("inspection_id").to_list()
    restaurant_ids = aligned_restaurant_ids(train_ids, train.identity)
    assert_bootstrap_inputs_aligned(Xtrain, ytrain, train_ids, restaurant_ids)

    build_fn = CANDIDATE_BUILDERS[selected_config.candidate_name]
    members, uncertainty_config = fit_bootstrap_ensemble(
        build_fn,
        Xtrain,
        ytrain,
        restaurant_ids,
        Xval,
        yval,
        n_members=25,
        min_successful=15,
        weighted=selected_config.weighted,
        base_seed=selected_config.random_seed,
    )

    bundle = assemble_production_bundle(
        jurisdiction=jurisdiction,  # type: ignore[arg-type]
        target_name=target_name,
        model_version=model_version,
        feature_order=feature_order,
        target_definition="synthetic fixture target for serving tests",
        eligible_inspection_types=["synthetic"],
        excluded_inspection_types=[],
        feature_categories=["synthetic"],
        split_boundaries=boundaries,
        split_report=split_report,
        target_build_report=None,
        selection=selection,
        selected_config=selected_config,
        calibration_outcome=calibration,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        members=members,
        uncertainty_config=uncertainty_config,
        subgroup_limitations="none (synthetic fixture)",
        data_snapshot_note="unavailable (synthetic fixture)",
        known_limitations=["synthetic data only"],
        source_provenance=None,
        random_seeds={"training": selected_config.random_seed},
    )
    assert bundle["deployment_status.json"]["status"] == "ready", bundle["deployment_status.json"]

    output_root = tmp_path / "artifacts"
    output = write_artifact(
        output_root,
        jurisdiction=jurisdiction,
        target_name=target_name,
        model_version=model_version,
        bundle=bundle,
    )
    return output


@pytest.fixture
def build_ready_artifact() -> Any:
    return _build_ready_artifact
