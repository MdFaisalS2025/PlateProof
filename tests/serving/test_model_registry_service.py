"""Tests for the metadata-only model reader used by the API and Streamlit.

This reader is the ONLY thing the running web application ever learns about
a Task 6 model artifact. It never deserializes ``point_estimator.joblib`` or
``bootstrap_members.joblib`` -- it only reads and JSON/text-parses the small
set of metadata files a production artifact bundle carries (see
``plateproof.models.training.assemble_production_bundle``), verifying
checksums as raw bytes only (hashing, never deserializing).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _settings(**overrides: Any) -> Any:
    from plateproof.core.config import Settings

    return Settings(_env_file=None, **overrides)


def test_no_configured_path_means_model_unavailable() -> None:
    from plateproof.serving.model_registry_service import ModelMetadataReader

    reader = ModelMetadataReader(_settings())
    assert reader.get("nyc") is None


def test_ready_artifact_metadata_is_read_and_cached(
    tmp_path: Path, build_ready_artifact: Any
) -> None:
    from plateproof.models.nyc_risk import NYC_FEATURE_LIST
    from plateproof.serving.model_registry_service import ModelMetadataReader

    artifact = build_ready_artifact(
        tmp_path,
        jurisdiction="nyc",
        target_name="nyc_next_initial_score_ge_14",
        feature_order=NYC_FEATURE_LIST,
    )
    reader = ModelMetadataReader(_settings(nyc_model_artifact_path=artifact))
    loaded_first = reader.get("nyc")
    assert loaded_first is not None
    assert loaded_first.jurisdiction == "nyc"
    assert loaded_first.deployment_status == "ready"
    assert loaded_first.target_name == "nyc_next_initial_score_ge_14"

    loaded_second = reader.get("nyc")
    assert loaded_second is loaded_first  # cached, not re-read


def test_non_ready_artifact_reports_unavailable(tmp_path: Path) -> None:
    from plateproof.models.training import write_artifact
    from plateproof.serving.model_registry_service import ModelMetadataReader

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
    reader = ModelMetadataReader(_settings(nyc_model_artifact_path=artifact))
    assert reader.get("nyc") is None
    assert reader.last_error_reason("nyc") is not None


def test_wrong_jurisdiction_configured_path_reports_unavailable(
    tmp_path: Path, build_ready_artifact: Any
) -> None:
    from plateproof.models.nyc_risk import NYC_FEATURE_LIST
    from plateproof.serving.model_registry_service import ModelMetadataReader

    artifact = build_ready_artifact(
        tmp_path,
        jurisdiction="nyc",
        target_name="nyc_next_initial_score_ge_14",
        feature_order=NYC_FEATURE_LIST,
    )
    # deliberately configured under the wrong jurisdiction's setting
    reader = ModelMetadataReader(_settings(florida_model_artifact_path=artifact))
    assert reader.get("florida") is None


def test_feature_schema_mismatch_reports_unavailable(
    tmp_path: Path, build_ready_artifact: Any
) -> None:
    from plateproof.serving.model_registry_service import ModelMetadataReader

    artifact = build_ready_artifact(
        tmp_path,
        jurisdiction="nyc",
        target_name="nyc_next_initial_score_ge_14",
        feature_order=("not_a_real_feature", "also_not_real"),
    )
    reader = ModelMetadataReader(_settings(nyc_model_artifact_path=artifact))
    assert reader.get("nyc") is None


def test_public_summary_never_includes_artifact_path(
    tmp_path: Path, build_ready_artifact: Any
) -> None:
    from plateproof.models.nyc_risk import NYC_FEATURE_LIST
    from plateproof.serving.model_registry_service import ModelMetadataReader

    artifact = build_ready_artifact(
        tmp_path,
        jurisdiction="nyc",
        target_name="nyc_next_initial_score_ge_14",
        feature_order=NYC_FEATURE_LIST,
    )
    reader = ModelMetadataReader(_settings(nyc_model_artifact_path=artifact))
    loaded = reader.get("nyc")
    assert loaded is not None
    summary = loaded.public_summary()
    assert str(artifact) not in str(summary)
    assert "artifact_path" not in summary
    assert summary["jurisdiction"] == "nyc"
    assert summary["deployment_status"] == "ready"


def test_get_card_allows_non_ready_when_explicitly_requested(tmp_path: Path) -> None:
    from plateproof.models.training import write_artifact
    from plateproof.serving.model_registry_service import ModelMetadataReader

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
    reader = ModelMetadataReader(_settings(nyc_model_artifact_path=artifact))

    assert reader.get_card("nyc", allow_non_ready=False) is None
    audit = reader.get_card("nyc", allow_non_ready=True)
    assert audit is not None
    assert audit.deployment_status == "uncalibrated"
    assert "Not ready card" in audit.model_card_markdown


# --------------------------------------------------------------------------- #
# Offline-only boundary: this reader must never deserialize an estimator     #
# --------------------------------------------------------------------------- #


def test_reader_never_imports_or_calls_load_artifact(
    tmp_path: Path, build_ready_artifact: Any, monkeypatch: Any
) -> None:
    from plateproof.models import training as training_module
    from plateproof.models.nyc_risk import NYC_FEATURE_LIST
    from plateproof.serving.model_registry_service import ModelMetadataReader

    artifact = build_ready_artifact(
        tmp_path,
        jurisdiction="nyc",
        target_name="nyc_next_initial_score_ge_14",
        feature_order=NYC_FEATURE_LIST,
    )

    def _forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("ModelMetadataReader must never call load_artifact")

    monkeypatch.setattr(training_module, "load_artifact", _forbidden)

    reader = ModelMetadataReader(_settings(nyc_model_artifact_path=artifact))
    metadata = reader.get("nyc")
    assert metadata is not None
    assert metadata.deployment_status == "ready"


def test_reader_never_deserializes_the_point_estimator_file(
    tmp_path: Path, build_ready_artifact: Any
) -> None:
    """Replace the real point estimator with bytes that would raise if ever
    passed to joblib.load, keeping its manifest checksum consistent (so this
    test isolates "never opens this file for deserialization" from the
    separate, already-covered "checksum mismatch fails closed" behavior).
    The reader must still succeed -- proving it hashes the bytes for
    integrity only and never deserializes them."""
    from plateproof.models.nyc_risk import NYC_FEATURE_LIST
    from plateproof.serving.model_registry_service import ModelMetadataReader

    artifact = build_ready_artifact(
        tmp_path,
        jurisdiction="nyc",
        target_name="nyc_next_initial_score_ge_14",
        feature_order=NYC_FEATURE_LIST,
    )
    poisoned = b"not a real joblib payload at all"
    (artifact / "point_estimator.joblib").write_bytes(poisoned)
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"]["point_estimator.joblib"] = hashlib.sha256(poisoned).hexdigest()
    (artifact / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    reader = ModelMetadataReader(_settings(nyc_model_artifact_path=artifact))
    metadata = reader.get("nyc")
    assert metadata is not None
    assert metadata.deployment_status == "ready"


def test_metadata_checksum_corruption_is_rejected_safely(
    tmp_path: Path, build_ready_artifact: Any
) -> None:
    """Tampering with a metadata file's bytes without updating its manifest
    checksum must be detected and treated as unavailable, not crash and not
    silently serve stale/wrong content."""
    from plateproof.models.nyc_risk import NYC_FEATURE_LIST
    from plateproof.serving.model_registry_service import ModelMetadataReader

    artifact = build_ready_artifact(
        tmp_path,
        jurisdiction="nyc",
        target_name="nyc_next_initial_score_ge_14",
        feature_order=NYC_FEATURE_LIST,
    )
    (artifact / "deployment_status.json").write_text(
        json.dumps({"status": "ready", "reason": "tampered"}), encoding="utf-8"
    )

    reader = ModelMetadataReader(_settings(nyc_model_artifact_path=artifact))
    assert reader.get("nyc") is None
    assert reader.last_error_reason("nyc") is not None


def test_missing_metadata_file_produces_honest_unavailable_response(
    tmp_path: Path, build_ready_artifact: Any
) -> None:
    from plateproof.models.nyc_risk import NYC_FEATURE_LIST
    from plateproof.serving.model_registry_service import ModelMetadataReader

    artifact = build_ready_artifact(
        tmp_path,
        jurisdiction="nyc",
        target_name="nyc_next_initial_score_ge_14",
        feature_order=NYC_FEATURE_LIST,
    )
    (artifact / "_SUCCESS").unlink()

    reader = ModelMetadataReader(_settings(nyc_model_artifact_path=artifact))
    assert reader.get("nyc") is None
    assert reader.last_error_reason("nyc") is not None
