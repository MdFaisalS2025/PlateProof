"""Tests for the shared prediction-lookup service used by both the FastAPI
route and the Streamlit page. Reads only precomputed tables and sanitized
metadata -- never deserializes or runs a model artifact.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl


def _settings(**overrides: Any) -> Any:
    from plateproof.core.config import Settings

    return Settings(_env_file=None, **overrides)


def _write_registry_and_prediction(
    prediction_dir: Path,
    *,
    registry_model_version: str,
    registry_target_name: str = "nyc_next_initial_score_ge_14",
    registry_schema_version: str = "1.0.0",
    prediction_row: dict[str, Any] | None = None,
) -> None:
    pl.DataFrame(
        [
            {
                "jurisdiction": "nyc",
                "target_name": registry_target_name,
                "model_version": registry_model_version,
                "artifact_schema_version": registry_schema_version,
                "deployment_status": "ready",
                "registered_at": date.today(),
                "source_snapshot_date": date.today(),
                "artifact_path": "irrelevant",
            }
        ]
    ).write_parquet(prediction_dir / "model_registry.parquet")
    if prediction_row is not None:
        pl.DataFrame([prediction_row]).write_parquet(prediction_dir / "predictions.parquet")


def test_no_registry_entry_is_no_ready_model(processed_dir: Path, tmp_path: Path) -> None:
    from plateproof.serving.model_registry_service import ModelMetadataReader
    from plateproof.serving.prediction_service import resolve_prediction
    from plateproof.serving.repository import open_repository

    repo = open_repository(_settings(processed_data_dir=processed_dir))
    reader = ModelMetadataReader(_settings())
    result = resolve_prediction(
        repo, reader, restaurant_id="nyc:1", jurisdiction="nyc", staleness_days=90
    )
    assert result.status == "no_ready_model"


def test_registry_ready_but_no_configured_artifact_is_no_ready_model(
    processed_dir: Path, tmp_path: Path
) -> None:
    from plateproof.serving.model_registry_service import ModelMetadataReader
    from plateproof.serving.prediction_service import resolve_prediction
    from plateproof.serving.repository import open_repository

    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    _write_registry_and_prediction(prediction_dir, registry_model_version="v1")

    repo = open_repository(
        _settings(processed_data_dir=processed_dir, prediction_table_path=prediction_dir)
    )
    reader = ModelMetadataReader(_settings())  # no artifact path configured at all
    result = resolve_prediction(
        repo, reader, restaurant_id="nyc:1", jurisdiction="nyc", staleness_days=90
    )
    assert result.status == "no_ready_model"


def test_registry_and_artifact_disagreement_fails_closed(
    processed_dir: Path, tmp_path: Path, build_ready_artifact: Any
) -> None:
    """The registry table says predictions were scored against model
    version 'v-old', but the currently configured artifact is a DIFFERENT
    version -- this must never be served as available, even though a
    "ready" registry entry and a "ready" configured artifact both exist."""
    from plateproof.models.nyc_risk import NYC_FEATURE_LIST
    from plateproof.serving.model_registry_service import ModelMetadataReader
    from plateproof.serving.prediction_service import resolve_prediction
    from plateproof.serving.repository import open_repository

    artifact = build_ready_artifact(
        tmp_path,
        jurisdiction="nyc",
        target_name="nyc_next_initial_score_ge_14",
        feature_order=NYC_FEATURE_LIST,
        model_version="v-current",
    )
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    _write_registry_and_prediction(
        prediction_dir,
        registry_model_version="v-old",
        prediction_row={
            "prediction_id": "pred:1",
            "restaurant_id": "nyc:1",
            "jurisdiction": "nyc",
            "target_name": "nyc_next_initial_score_ge_14",
            "model_version": "v-old",
            "artifact_schema_version": "1.0.0",
            "as_of_date": date.today(),
            "generated_at": date.today(),
            "probability": 0.4,
            "lower_bound": 0.2,
            "upper_bound": 0.6,
            "risk_band": "moderate",
            "insufficient_history_reason": None,
            "calibration_status": "calibrated_sigmoid",
            "uncertainty_status": "available",
            "readiness_status": "ready",
        },
    )

    repo = open_repository(
        _settings(processed_data_dir=processed_dir, prediction_table_path=prediction_dir)
    )
    reader = ModelMetadataReader(_settings(nyc_model_artifact_path=artifact))
    result = resolve_prediction(
        repo, reader, restaurant_id="nyc:1", jurisdiction="nyc", staleness_days=90
    )
    assert result.status == "stale"
    assert result.row is None


def test_registry_and_artifact_agreement_with_valid_prediction_is_available(
    processed_dir: Path, tmp_path: Path, build_ready_artifact: Any
) -> None:
    from plateproof.models.nyc_risk import NYC_FEATURE_LIST
    from plateproof.serving.model_registry_service import ModelMetadataReader
    from plateproof.serving.prediction_service import resolve_prediction
    from plateproof.serving.repository import open_repository

    artifact = build_ready_artifact(
        tmp_path,
        jurisdiction="nyc",
        target_name="nyc_next_initial_score_ge_14",
        feature_order=NYC_FEATURE_LIST,
        model_version="v1",
    )
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    schema_version = json.loads(
        (artifact / "artifact_schema_version.json").read_text(encoding="utf-8")
    )
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    _write_registry_and_prediction(
        prediction_dir,
        registry_model_version=manifest["model_version"],
        registry_schema_version=schema_version,
        prediction_row={
            "prediction_id": "pred:1",
            "restaurant_id": "nyc:1",
            "jurisdiction": "nyc",
            "target_name": "nyc_next_initial_score_ge_14",
            "model_version": manifest["model_version"],
            "artifact_schema_version": schema_version,
            "as_of_date": date.today(),
            "generated_at": date.today(),
            "probability": 0.4,
            "lower_bound": 0.2,
            "upper_bound": 0.6,
            "risk_band": "moderate",
            "insufficient_history_reason": None,
            "calibration_status": "calibrated_sigmoid",
            "uncertainty_status": "available",
            "readiness_status": "ready",
        },
    )

    repo = open_repository(
        _settings(processed_data_dir=processed_dir, prediction_table_path=prediction_dir)
    )
    reader = ModelMetadataReader(_settings(nyc_model_artifact_path=artifact))
    result = resolve_prediction(
        repo, reader, restaurant_id="nyc:1", jurisdiction="nyc", staleness_days=90
    )
    assert result.status == "available"
    assert result.row is not None
    assert result.row["risk_band"] == "moderate"


def test_resolve_prediction_never_calls_load_artifact(
    processed_dir: Path, tmp_path: Path, build_ready_artifact: Any, monkeypatch: Any
) -> None:
    from plateproof.models import training as training_module
    from plateproof.models.nyc_risk import NYC_FEATURE_LIST
    from plateproof.serving.model_registry_service import ModelMetadataReader
    from plateproof.serving.prediction_service import resolve_prediction
    from plateproof.serving.repository import open_repository

    artifact = build_ready_artifact(
        tmp_path,
        jurisdiction="nyc",
        target_name="nyc_next_initial_score_ge_14",
        feature_order=NYC_FEATURE_LIST,
    )

    def _forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("resolve_prediction must never call load_artifact")

    monkeypatch.setattr(training_module, "load_artifact", _forbidden)

    repo = open_repository(_settings(processed_data_dir=processed_dir))
    reader = ModelMetadataReader(_settings(nyc_model_artifact_path=artifact))
    result = resolve_prediction(
        repo, reader, restaurant_id="nyc:1", jurisdiction="nyc", staleness_days=90
    )
    assert result.status == "no_ready_model"  # no registry table configured -- still safe
