"""Hardens ``ModelMetadataReader``/``_read_metadata`` against malformed or
adversarial artifact content: every expected failure mode must return
``(None, reason)``, never raise, and the reader must never even open a
``.joblib``/``.pickle`` file -- only offline scoring does full-artifact
verification (see ``plateproof.models.training.load_artifact``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


def _settings(**overrides: Any) -> Any:
    from plateproof.core.config import Settings

    return Settings(_env_file=None, **overrides)


def _read_metadata(path: Path, *, jurisdiction: str = "nyc", require_ready: bool = True) -> Any:
    from plateproof.serving.model_registry_service import _read_metadata as _impl

    return _impl(path, expected_jurisdiction=jurisdiction, require_ready=require_ready)


# --------------------------------------------------------------------------- #
# 1 / 4 / 5 / 6: malformed manifest/metadata content fails closed, never raises
# --------------------------------------------------------------------------- #


def test_malformed_manifest_json_fails_closed(tmp_path: Path) -> None:
    (tmp_path / "_SUCCESS").write_text("", encoding="utf-8")
    (tmp_path / "manifest.json").write_text("{not valid json", encoding="utf-8")

    metadata, reason = _read_metadata(tmp_path)
    assert metadata is None
    assert reason is not None
    assert "{not valid json" not in reason


def test_manifest_invalid_utf8_fails_closed(tmp_path: Path) -> None:
    (tmp_path / "_SUCCESS").write_text("", encoding="utf-8")
    (tmp_path / "manifest.json").write_bytes(b"\xff\xfe\x00not utf-8")

    metadata, reason = _read_metadata(tmp_path)
    assert metadata is None
    assert reason is not None


def test_manifest_wrong_top_level_type_fails_closed(tmp_path: Path) -> None:
    (tmp_path / "_SUCCESS").write_text("", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")

    metadata, reason = _read_metadata(tmp_path)
    assert metadata is None
    assert reason is not None


def test_manifest_missing_files_mapping_fails_closed(tmp_path: Path) -> None:
    (tmp_path / "_SUCCESS").write_text("", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"jurisdiction": "nyc", "model_version": "v1"}), encoding="utf-8"
    )

    metadata, reason = _read_metadata(tmp_path)
    assert metadata is None
    assert reason is not None


def test_manifest_files_mapping_wrong_type_fails_closed(tmp_path: Path) -> None:
    (tmp_path / "_SUCCESS").write_text("", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"jurisdiction": "nyc", "model_version": "v1", "files": "not-a-mapping"}),
        encoding="utf-8",
    )

    metadata, reason = _read_metadata(tmp_path)
    assert metadata is None
    assert reason is not None


def test_malformed_allowlisted_metadata_json_fails_closed(
    tmp_path: Path, build_ready_artifact: Any
) -> None:
    """A metadata file inside the closed allowlist (deployment_status.json)
    that is itself malformed JSON must fail closed, not raise -- even though
    its checksum matches (the corruption was present when the artifact was
    written, e.g. a disk fault, not a subsequent tamper)."""
    import hashlib

    from plateproof.models.nyc_risk import NYC_FEATURE_LIST

    artifact = build_ready_artifact(
        tmp_path,
        jurisdiction="nyc",
        target_name="nyc_next_initial_score_ge_14",
        feature_order=NYC_FEATURE_LIST,
    )
    broken = b"{not valid json"
    (artifact / "deployment_status.json").write_bytes(broken)
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"]["deployment_status.json"] = hashlib.sha256(broken).hexdigest()
    (artifact / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    metadata, reason = _read_metadata(artifact)
    assert metadata is None
    assert reason is not None
    assert "{not valid json" not in reason


# --------------------------------------------------------------------------- #
# 7 / 8 / 9: manifest-listed filenames must be safe before any file operation
# --------------------------------------------------------------------------- #


def _artifact_with_poisoned_manifest_entry(
    tmp_path: Path, build_ready_artifact: Any, *, poisoned_name: str
) -> Path:
    from plateproof.models.nyc_risk import NYC_FEATURE_LIST

    artifact = build_ready_artifact(
        tmp_path,
        jurisdiction="nyc",
        target_name="nyc_next_initial_score_ge_14",
        feature_order=NYC_FEATURE_LIST,
    )
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    # Rename the deployment_status.json entry to a poisoned name so the
    # reader is forced to consider it when validating the "files" mapping.
    manifest["files"][poisoned_name] = manifest["files"].pop("deployment_status.json")
    (artifact / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return artifact


def test_absolute_manifest_path_is_rejected_without_reading_the_target(
    tmp_path: Path, build_ready_artifact: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path / "outside_secret.txt"
    outside.write_text("must never be read", encoding="utf-8")
    poisoned_name = str(outside)  # an absolute path used as a manifest "filename"
    artifact = _artifact_with_poisoned_manifest_entry(
        tmp_path, build_ready_artifact, poisoned_name=poisoned_name
    )

    original_read_bytes = Path.read_bytes

    def _guarded_read_bytes(self: Path) -> bytes:
        assert self != outside, "must never read a manifest-supplied absolute path"
        return original_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", _guarded_read_bytes)

    metadata, reason = _read_metadata(artifact)
    # deployment_status.json's checksum entry is now unreachable under its
    # expected name, so the artifact is honestly reported not-ready/unavailable.
    assert metadata is None
    assert reason is not None


def test_parent_traversal_manifest_path_is_rejected_without_reading_the_target(
    tmp_path: Path, build_ready_artifact: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = _artifact_with_poisoned_manifest_entry(
        tmp_path, build_ready_artifact, poisoned_name="../outside_secret.txt"
    )
    outside = artifact.parent / "outside_secret.txt"
    outside.write_text("must never be read", encoding="utf-8")

    original_read_bytes = Path.read_bytes

    def _guarded_read_bytes(self: Path) -> bytes:
        assert self.resolve() != outside.resolve(), "must never read a path escaping the artifact"
        return original_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", _guarded_read_bytes)

    metadata, reason = _read_metadata(artifact)
    assert metadata is None
    assert reason is not None


def test_drive_qualified_manifest_path_is_rejected(
    tmp_path: Path, build_ready_artifact: Any
) -> None:
    artifact = _artifact_with_poisoned_manifest_entry(
        tmp_path, build_ready_artifact, poisoned_name="C:secret.txt"
    )
    metadata, reason = _read_metadata(artifact)
    assert metadata is None
    assert reason is not None


def test_manifest_path_with_separator_is_rejected(
    tmp_path: Path, build_ready_artifact: Any
) -> None:
    artifact = _artifact_with_poisoned_manifest_entry(
        tmp_path, build_ready_artifact, poisoned_name="subdir/deployment_status.json"
    )
    metadata, reason = _read_metadata(artifact)
    assert metadata is None
    assert reason is not None


def test_symlink_escape_is_rejected(tmp_path: Path, build_ready_artifact: Any) -> None:
    outside = tmp_path / "outside_secret.txt"
    outside.write_text('{"status": "ready"}', encoding="utf-8")

    from plateproof.models.nyc_risk import NYC_FEATURE_LIST

    artifact = build_ready_artifact(
        tmp_path,
        jurisdiction="nyc",
        target_name="nyc_next_initial_score_ge_14",
        feature_order=NYC_FEATURE_LIST,
    )
    link_path = artifact / "deployment_status.json"
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    expected_checksum = manifest["files"]["deployment_status.json"]
    link_path.unlink()
    try:
        link_path.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("platform/user cannot create symlinks")

    # Keep the manifest checksum as originally recorded -- the symlink target
    # doesn't match it, so even a checksum-only check would still reject it,
    # but the point of this test is that the symlink itself is never followed.
    manifest["files"]["deployment_status.json"] = expected_checksum
    (artifact / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    metadata, reason = _read_metadata(artifact)
    assert metadata is None
    assert reason is not None


# --------------------------------------------------------------------------- #
# 10: a file disappearing/changing mid-validation fails closed
# --------------------------------------------------------------------------- #


def test_manifest_listed_file_missing_from_disk_fails_closed(
    tmp_path: Path, build_ready_artifact: Any
) -> None:
    from plateproof.models.nyc_risk import NYC_FEATURE_LIST

    artifact = build_ready_artifact(
        tmp_path,
        jurisdiction="nyc",
        target_name="nyc_next_initial_score_ge_14",
        feature_order=NYC_FEATURE_LIST,
    )
    (artifact / "deployment_status.json").unlink()

    metadata, reason = _read_metadata(artifact)
    assert metadata is None
    assert reason is not None


def test_manifest_listed_directory_where_file_required_fails_closed(
    tmp_path: Path, build_ready_artifact: Any
) -> None:
    from plateproof.models.nyc_risk import NYC_FEATURE_LIST

    artifact = build_ready_artifact(
        tmp_path,
        jurisdiction="nyc",
        target_name="nyc_next_initial_score_ge_14",
        feature_order=NYC_FEATURE_LIST,
    )
    status_path = artifact / "deployment_status.json"
    checksum = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))["files"][
        "deployment_status.json"
    ]
    status_path.unlink()
    status_path.mkdir()

    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"]["deployment_status.json"] = checksum
    (artifact / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    metadata, reason = _read_metadata(artifact)
    assert metadata is None
    assert reason is not None


# --------------------------------------------------------------------------- #
# Oversized metadata document
# --------------------------------------------------------------------------- #


def test_oversized_metadata_document_fails_closed(
    tmp_path: Path, build_ready_artifact: Any
) -> None:
    import hashlib

    from plateproof.models.nyc_risk import NYC_FEATURE_LIST

    artifact = build_ready_artifact(
        tmp_path,
        jurisdiction="nyc",
        target_name="nyc_next_initial_score_ge_14",
        feature_order=NYC_FEATURE_LIST,
    )
    huge = b'{"status": "ready", "padding": "' + b"x" * (2 * 1024 * 1024) + b'"}'
    (artifact / "deployment_status.json").write_bytes(huge)
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"]["deployment_status.json"] = hashlib.sha256(huge).hexdigest()
    (artifact / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    metadata, reason = _read_metadata(artifact)
    assert metadata is None
    assert reason is not None


# --------------------------------------------------------------------------- #
# Wrong JSON field types: fields with harmless defaults fall back safely;
# fields that gate readiness fail closed instead of guessing.
# --------------------------------------------------------------------------- #


def _rewrite_checksummed_file(artifact: Path, name: str, value: Any) -> None:
    import hashlib

    raw = json.dumps(value).encode("utf-8")
    (artifact / name).write_bytes(raw)
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"][name] = hashlib.sha256(raw).hexdigest()
    (artifact / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_wrong_type_display_fields_fall_back_to_safe_defaults(
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
    _rewrite_checksummed_file(artifact, "target_definition.json", {"target_name": 42})
    _rewrite_checksummed_file(artifact, "artifact_schema_version.json", {"not": "a string"})

    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    manifest["model_version"] = 42
    manifest["generated_at"] = 42
    (artifact / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    reader = ModelMetadataReader(_settings(nyc_model_artifact_path=artifact))
    metadata = reader.get_card("nyc", allow_non_ready=True)
    assert metadata is not None
    assert metadata.target_name == ""
    assert metadata.artifact_schema_version == "unknown"
    assert metadata.model_version == ""
    assert metadata.registered_at == ""


def test_wrong_type_deployment_status_fails_closed(
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
    _rewrite_checksummed_file(artifact, "deployment_status.json", {"status": 42})

    reader = ModelMetadataReader(_settings(nyc_model_artifact_path=artifact))
    assert reader.get("nyc") is None


def test_wrong_type_feature_order_fails_closed(tmp_path: Path, build_ready_artifact: Any) -> None:
    from plateproof.models.nyc_risk import NYC_FEATURE_LIST
    from plateproof.serving.model_registry_service import ModelMetadataReader

    artifact = build_ready_artifact(
        tmp_path,
        jurisdiction="nyc",
        target_name="nyc_next_initial_score_ge_14",
        feature_order=NYC_FEATURE_LIST,
    )
    _rewrite_checksummed_file(artifact, "feature_order.json", {"not": "a list"})

    reader = ModelMetadataReader(_settings(nyc_model_artifact_path=artifact))
    assert reader.get("nyc") is None


# --------------------------------------------------------------------------- #
# 11: the web reader never opens or hashes joblib/pickle files
# --------------------------------------------------------------------------- #


def test_reader_never_opens_estimator_or_bootstrap_files(
    tmp_path: Path, build_ready_artifact: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from plateproof.models.nyc_risk import NYC_FEATURE_LIST
    from plateproof.serving.model_registry_service import ModelMetadataReader

    artifact = build_ready_artifact(
        tmp_path,
        jurisdiction="nyc",
        target_name="nyc_next_initial_score_ge_14",
        feature_order=NYC_FEATURE_LIST,
    )

    original_read_bytes = Path.read_bytes

    def _guarded_read_bytes(self: Path) -> bytes:
        if self.suffix in (".joblib", ".pickle", ".pkl"):
            raise AssertionError(f"web metadata reader must never open {self.name}")
        return original_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", _guarded_read_bytes)

    reader = ModelMetadataReader(_settings(nyc_model_artifact_path=artifact))
    metadata = reader.get("nyc")
    assert metadata is not None
    assert metadata.deployment_status == "ready"
