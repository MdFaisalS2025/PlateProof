"""Tests for atomic, checksummed, trusted-load-gated model artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _bundle() -> dict[str, Any]:
    return {
        "model.joblib": {"fake": "estimator"},
        "manifest_extra.json": {"note": "example artifact content"},
    }


def test_write_artifact_produces_success_and_manifest(tmp_path: Path) -> None:
    from plateproof.models.training import is_valid_model_artifact, write_artifact

    output = write_artifact(
        tmp_path, jurisdiction="nyc", target_name="t", model_version="v1", bundle=_bundle()
    )
    assert is_valid_model_artifact(output)
    assert (output / "_SUCCESS").exists()
    assert (output / "manifest.json").exists()


def test_success_written_last(tmp_path: Path, monkeypatch: Any) -> None:
    from plateproof.models import training as training_module

    calls: list[str] = []
    original_write_text = Path.write_text

    def _tracking_write_text(self: Path, *args: Any, **kwargs: Any) -> Any:
        calls.append(self.name)
        return original_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", _tracking_write_text)
    training_module.write_artifact(
        tmp_path, jurisdiction="nyc", target_name="t", model_version="v1", bundle=_bundle()
    )
    assert calls[-1] == "_SUCCESS"


def test_refuses_to_overwrite_completed_artifact_without_force(tmp_path: Path) -> None:
    import pytest

    from plateproof.models.training import write_artifact

    output_dir = tmp_path / "nyc" / "t" / "v1"
    write_artifact(
        tmp_path, jurisdiction="nyc", target_name="t", model_version="v1", bundle=_bundle()
    )
    with pytest.raises(FileExistsError):
        write_artifact(
            tmp_path, jurisdiction="nyc", target_name="t", model_version="v1", bundle=_bundle()
        )
    assert output_dir.exists()


def test_force_allows_overwrite(tmp_path: Path) -> None:
    from plateproof.models.training import write_artifact

    write_artifact(
        tmp_path, jurisdiction="nyc", target_name="t", model_version="v1", bundle=_bundle()
    )
    output = write_artifact(
        tmp_path,
        jurisdiction="nyc",
        target_name="t",
        model_version="v1",
        bundle=_bundle(),
        force=True,
    )
    assert output.exists()


def test_checksum_mismatch_detected(tmp_path: Path) -> None:
    from plateproof.models.training import verify_artifact_checksums, write_artifact

    output = write_artifact(
        tmp_path, jurisdiction="nyc", target_name="t", model_version="v1", bundle=_bundle()
    )
    (output / "manifest_extra.json").write_text('{"note": "tampered"}', encoding="utf-8")
    mismatches = verify_artifact_checksums(output)
    assert "manifest_extra.json" in mismatches


def test_load_artifact_requires_trusted_true(tmp_path: Path) -> None:
    import pytest

    from plateproof.models.training import load_artifact, write_artifact

    output = write_artifact(
        tmp_path, jurisdiction="nyc", target_name="t", model_version="v1", bundle=_bundle()
    )
    with pytest.raises(PermissionError):
        load_artifact(output, trusted=False)
    loaded = load_artifact(output, trusted=True)  # must not raise
    assert loaded is not None


def test_load_artifact_refuses_corrupt_checksum(tmp_path: Path) -> None:
    import pytest

    from plateproof.models.training import load_artifact, write_artifact

    output = write_artifact(
        tmp_path, jurisdiction="nyc", target_name="t", model_version="v1", bundle=_bundle()
    )
    (output / "manifest_extra.json").write_text('{"note": "tampered"}', encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_artifact(output, trusted=True)


def test_manifest_tampering_alone_is_not_declared_secure(tmp_path: Path) -> None:
    """An attacker who can replace both the artifact and its manifest defeats
    the checksum. This is documented, not silently claimed otherwise."""
    import json

    from plateproof.models.training import load_artifact, write_artifact

    output = write_artifact(
        tmp_path, jurisdiction="nyc", target_name="t", model_version="v1", bundle=_bundle()
    )
    # simulate an attacker replacing both the file and its recorded checksum
    tampered_bytes = json.dumps({"note": "malicious payload"}).encode("utf-8")
    (output / "manifest_extra.json").write_bytes(tampered_bytes)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    import hashlib

    manifest["files"]["manifest_extra.json"] = hashlib.sha256(tampered_bytes).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    # checksums now agree -- load succeeds despite the tampering, proving
    # checksums alone are not a security boundary (documented in the module).
    loaded = load_artifact(output, trusted=True)
    assert loaded is not None


def test_nyc_and_florida_artifacts_cannot_be_interchanged(tmp_path: Path) -> None:
    import pytest

    from plateproof.models.training import load_artifact, write_artifact

    nyc_output = write_artifact(
        tmp_path,
        jurisdiction="nyc",
        target_name="nyc_next_initial_score_ge_14",
        model_version="v1",
        bundle=_bundle(),
    )
    write_artifact(
        tmp_path,
        jurisdiction="florida",
        target_name="florida_next_routine_high_priority_or_follow_up",
        model_version="v1",
        bundle=_bundle(),
    )
    loaded = load_artifact(nyc_output, trusted=True)
    manifest = loaded["manifest"]
    assert manifest["jurisdiction"] == "nyc"
    with pytest.raises(ValueError):
        load_artifact(nyc_output, trusted=True, expected_jurisdiction="florida")
