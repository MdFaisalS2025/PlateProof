"""Tests for the model-card/artifact cache used by the API and Streamlit.
Never re-deserializes per call, never exposes a filesystem path publicly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _settings(**overrides: Any) -> Any:
    from plateproof.core.config import Settings

    return Settings(_env_file=None, **overrides)


def test_no_configured_path_means_model_unavailable() -> None:
    from plateproof.serving.model_registry_service import ModelCache

    cache = ModelCache(_settings())
    assert cache.get("nyc") is None


def test_ready_artifact_is_loaded_and_cached(tmp_path: Path, build_ready_artifact: Any) -> None:
    from plateproof.models.nyc_risk import NYC_FEATURE_LIST
    from plateproof.serving.model_registry_service import ModelCache

    artifact = build_ready_artifact(
        tmp_path,
        jurisdiction="nyc",
        target_name="nyc_next_initial_score_ge_14",
        feature_order=NYC_FEATURE_LIST,
    )
    cache = ModelCache(_settings(nyc_model_artifact_path=artifact))
    loaded_first = cache.get("nyc")
    assert loaded_first is not None
    assert loaded_first.jurisdiction == "nyc"
    assert loaded_first.deployment_status == "ready"

    loaded_second = cache.get("nyc")
    assert loaded_second is loaded_first  # cached, not reloaded


def test_non_ready_artifact_reports_unavailable(tmp_path: Path) -> None:
    from plateproof.models.training import write_artifact
    from plateproof.serving.model_registry_service import ModelCache

    bundle = {
        "model.joblib": {"fake": "estimator"},
        "deployment_status.json": {"status": "uncalibrated", "reason": "test"},
    }
    artifact = write_artifact(
        tmp_path / "artifacts",
        jurisdiction="nyc",
        target_name="nyc_next_initial_score_ge_14",
        model_version="v1",
        bundle=bundle,
    )
    cache = ModelCache(_settings(nyc_model_artifact_path=artifact))
    assert cache.get("nyc") is None
    assert cache.last_error_reason("nyc") is not None


def test_wrong_jurisdiction_configured_path_reports_unavailable(
    tmp_path: Path, build_ready_artifact: Any
) -> None:
    from plateproof.models.nyc_risk import NYC_FEATURE_LIST
    from plateproof.serving.model_registry_service import ModelCache

    artifact = build_ready_artifact(
        tmp_path,
        jurisdiction="nyc",
        target_name="nyc_next_initial_score_ge_14",
        feature_order=NYC_FEATURE_LIST,
    )
    # deliberately configured under the wrong jurisdiction's setting
    cache = ModelCache(_settings(florida_model_artifact_path=artifact))
    assert cache.get("florida") is None


def test_feature_schema_mismatch_reports_unavailable(
    tmp_path: Path, build_ready_artifact: Any
) -> None:
    from plateproof.serving.model_registry_service import ModelCache

    artifact = build_ready_artifact(
        tmp_path,
        jurisdiction="nyc",
        target_name="nyc_next_initial_score_ge_14",
        feature_order=("not_a_real_feature", "also_not_real"),
    )
    cache = ModelCache(_settings(nyc_model_artifact_path=artifact))
    assert cache.get("nyc") is None


def test_public_summary_never_includes_artifact_path(
    tmp_path: Path, build_ready_artifact: Any
) -> None:
    from plateproof.models.nyc_risk import NYC_FEATURE_LIST
    from plateproof.serving.model_registry_service import ModelCache

    artifact = build_ready_artifact(
        tmp_path,
        jurisdiction="nyc",
        target_name="nyc_next_initial_score_ge_14",
        feature_order=NYC_FEATURE_LIST,
    )
    cache = ModelCache(_settings(nyc_model_artifact_path=artifact))
    loaded = cache.get("nyc")
    assert loaded is not None
    summary = loaded.public_summary()
    assert str(artifact) not in str(summary)
    assert "artifact_path" not in summary
    assert summary["jurisdiction"] == "nyc"
    assert summary["deployment_status"] == "ready"


def test_get_card_allows_non_ready_when_explicitly_requested(tmp_path: Path) -> None:
    from plateproof.models.training import write_artifact
    from plateproof.serving.model_registry_service import ModelCache

    bundle = {
        "model.joblib": {"fake": "estimator"},
        "deployment_status.json": {"status": "uncalibrated", "reason": "test"},
        "target_definition.json": {"target_name": "nyc_next_initial_score_ge_14"},
        "model_card.md": "# Not ready card",
    }
    artifact = write_artifact(
        tmp_path / "artifacts",
        jurisdiction="nyc",
        target_name="nyc_next_initial_score_ge_14",
        model_version="v1",
        bundle=bundle,
    )
    cache = ModelCache(_settings(nyc_model_artifact_path=artifact))

    assert cache.get_card("nyc", allow_non_ready=False) is None
    audit = cache.get_card("nyc", allow_non_ready=True)
    assert audit is not None
    assert audit.deployment_status == "uncalibrated"
    assert "Not ready card" in audit.model_card_markdown
