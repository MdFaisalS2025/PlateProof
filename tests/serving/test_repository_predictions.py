"""Tests for the repository's prediction-selection logic: never MAX(generated_at)
alone -- jurisdiction, target, model_version, artifact_schema_version, and
readiness must all match the currently active model, and a matching row
outside the staleness window is reported as stale, not silently served.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl
import pytest


def _settings(processed_dir: Path, prediction_dir: Path, **overrides: Any) -> Any:
    from plateproof.core.config import Settings

    return Settings(
        _env_file=None,
        processed_data_dir=processed_dir,
        prediction_table_path=prediction_dir,
        **overrides,
    )


def _prediction_row(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "prediction_id": "pred:1",
        "restaurant_id": "nyc:1",
        "jurisdiction": "nyc",
        "target_name": "nyc_next_initial_score_ge_14",
        "model_version": "v1",
        "artifact_schema_version": "1.0.0",
        "as_of_date": date(2026, 1, 1),
        "generated_at": datetime(2026, 1, 1, tzinfo=UTC),
        "probability": 0.3,
        "lower_bound": 0.1,
        "upper_bound": 0.5,
        "risk_band": "moderate",
        "insufficient_history_reason": None,
        "calibration_status": "calibrated_sigmoid",
        "uncertainty_status": "available",
        "readiness_status": "ready",
    }
    base.update(overrides)
    return base


@pytest.fixture
def prediction_dir(tmp_path: Path) -> Path:
    d = tmp_path / "predictions"
    d.mkdir()
    return d


def _write_predictions(prediction_dir: Path, rows: list[dict[str, Any]]) -> None:
    pl.DataFrame(rows).write_parquet(prediction_dir / "predictions.parquet")


def test_no_prediction_table_returns_none(processed_dir: Path, prediction_dir: Path) -> None:
    from plateproof.serving.repository import open_repository

    repo = open_repository(_settings(processed_dir, prediction_dir))
    result = repo.latest_prediction(
        restaurant_id="nyc:1",
        jurisdiction="nyc",
        target_name="nyc_next_initial_score_ge_14",
        active_model_version="v1",
        active_artifact_schema_version="1.0.0",
        staleness_days=90,
        today=date(2026, 1, 15),
    )
    assert result.row is None
    assert result.stale_row_exists is False


def test_matching_ready_prediction_is_selected(processed_dir: Path, prediction_dir: Path) -> None:
    from plateproof.serving.repository import open_repository

    _write_predictions(prediction_dir, [_prediction_row()])
    repo = open_repository(_settings(processed_dir, prediction_dir))
    result = repo.latest_prediction(
        restaurant_id="nyc:1",
        jurisdiction="nyc",
        target_name="nyc_next_initial_score_ge_14",
        active_model_version="v1",
        active_artifact_schema_version="1.0.0",
        staleness_days=90,
        today=date(2026, 1, 15),
    )
    assert result.row is not None
    assert result.row["prediction_id"] == "pred:1"


def test_wrong_target_prediction_is_excluded(processed_dir: Path, prediction_dir: Path) -> None:
    from plateproof.serving.repository import open_repository

    _write_predictions(prediction_dir, [_prediction_row(target_name="nyc_next_score_ge_28")])
    repo = open_repository(_settings(processed_dir, prediction_dir))
    result = repo.latest_prediction(
        restaurant_id="nyc:1",
        jurisdiction="nyc",
        target_name="nyc_next_initial_score_ge_14",
        active_model_version="v1",
        active_artifact_schema_version="1.0.0",
        staleness_days=90,
        today=date(2026, 1, 15),
    )
    assert result.row is None


def test_stale_prediction_is_excluded_but_reported(
    processed_dir: Path, prediction_dir: Path
) -> None:
    from plateproof.serving.repository import open_repository

    _write_predictions(prediction_dir, [_prediction_row(as_of_date=date(2025, 1, 1))])
    repo = open_repository(_settings(processed_dir, prediction_dir))
    result = repo.latest_prediction(
        restaurant_id="nyc:1",
        jurisdiction="nyc",
        target_name="nyc_next_initial_score_ge_14",
        active_model_version="v1",
        active_artifact_schema_version="1.0.0",
        staleness_days=90,
        today=date(2026, 1, 15),
    )
    assert result.row is None
    assert result.stale_row_exists is True


def test_newer_incompatible_model_version_prediction_is_excluded(
    processed_dir: Path, prediction_dir: Path
) -> None:
    from plateproof.serving.repository import open_repository

    _write_predictions(prediction_dir, [_prediction_row(model_version="v0-old")])
    repo = open_repository(_settings(processed_dir, prediction_dir))
    result = repo.latest_prediction(
        restaurant_id="nyc:1",
        jurisdiction="nyc",
        target_name="nyc_next_initial_score_ge_14",
        active_model_version="v1",
        active_artifact_schema_version="1.0.0",
        staleness_days=90,
        today=date(2026, 1, 15),
    )
    assert result.row is None


def test_newest_as_of_date_selected_among_valid_candidates(
    processed_dir: Path, prediction_dir: Path
) -> None:
    from plateproof.serving.repository import open_repository

    _write_predictions(
        prediction_dir,
        [
            _prediction_row(prediction_id="pred:old", as_of_date=date(2026, 1, 1)),
            _prediction_row(prediction_id="pred:new", as_of_date=date(2026, 1, 10)),
        ],
    )
    repo = open_repository(_settings(processed_dir, prediction_dir))
    result = repo.latest_prediction(
        restaurant_id="nyc:1",
        jurisdiction="nyc",
        target_name="nyc_next_initial_score_ge_14",
        active_model_version="v1",
        active_artifact_schema_version="1.0.0",
        staleness_days=90,
        today=date(2026, 1, 15),
    )
    assert result.row is not None
    assert result.row["prediction_id"] == "pred:new"


def test_model_registry_public_fields_only(processed_dir: Path, prediction_dir: Path) -> None:
    from plateproof.serving.repository import open_repository

    pl.DataFrame(
        [
            {
                "jurisdiction": "nyc",
                "target_name": "nyc_next_initial_score_ge_14",
                "model_version": "v1",
                "artifact_schema_version": "1.0.0",
                "deployment_status": "ready",
                "registered_at": date(2026, 1, 1),
                "source_snapshot_date": date(2026, 1, 1),
                "artifact_path": "C:/secret/local/path",
            }
        ]
    ).write_parquet(prediction_dir / "model_registry.parquet")
    repo = open_repository(_settings(processed_dir, prediction_dir))
    entry = repo.model_registry_entry("nyc")
    assert entry is not None
    assert "artifact_path" not in entry
    assert entry["model_version"] == "v1"
