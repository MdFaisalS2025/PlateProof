"""Reads sanitized, non-executable model metadata for the API and Streamlit
UI. This module is the ONLY thing the running web application ever learns
about a Task 6 model artifact.

Trust boundary
--------------
* Offline administrative scoring (``plateproof/serving/scoring.py``, driven
  by ``scripts/score_predictions.py``) is the only code in this project
  permitted to call ``plateproof.models.training.load_artifact`` and
  therefore to deserialize ``point_estimator.joblib`` /
  ``bootstrap_members.joblib``. That is an explicit, administrator-triggered,
  offline action against a path the administrator configured.
* The public application (every FastAPI route and every Streamlit page)
  consumes only sanitized metadata (via this module) and precomputed
  prediction rows (via ``plateproof.serving.repository``). It never
  deserializes or executes a model artifact, and it must keep working
  correctly even if ``load_artifact``/``joblib.load`` are broken, patched,
  or removed entirely -- it simply never calls them.

This reader validates artifact integrity for only the closed metadata
allowlist below (``_SUCCESS`` presence and sha256 checksums, computed the
same way ``plateproof.models.training`` does) using raw byte hashing --
hashing never executes file content, unlike ``joblib.load``/``pickle.load``.
Unlike ``plateproof.models.training.verify_artifact_checksums`` (which
verifies every file in a production bundle, including the estimator and
bootstrap members, as part of offline scoring's full-artifact validation),
this reader deliberately validates only its metadata subset: it never
opens, reads, hashes, or otherwise touches ``point_estimator.joblib``,
``bootstrap_members.joblib``, or any other manifest entry outside the
allowlist. Do not read this module as validating the complete production
model artifact -- that is offline scoring's responsibility alone.

Every failure mode this reader can encounter -- malformed or oversized
JSON, invalid UTF-8, an unsafe or missing manifest-listed filename, a
symlink, a file that changes or disappears mid-check, an ordinary
filesystem error -- is handled explicitly and returns ``(None, reason)``.
Nothing here ever raises out to a caller, and no local path or raw
exception text is ever included in a returned reason.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal

from plateproof.core.config import Settings
from plateproof.models.florida_risk import FL_FEATURE_LIST
from plateproof.models.nyc_risk import NYC_FEATURE_LIST

Jurisdiction = Literal["nyc", "florida"]

_EXPECTED_FEATURES: dict[str, tuple[str, ...]] = {
    "nyc": NYC_FEATURE_LIST,
    "florida": FL_FEATURE_LIST,
}

# The only files this reader will ever open and parse. Every other file
# named in manifest.json (in particular point_estimator.joblib and
# bootstrap_members.joblib) is checksummed as raw bytes only -- this set is
# the complete, closed allowlist of what gets a json.loads/read_text call.
_JSON_METADATA_FILES = frozenset(
    {
        "target_definition.json",
        "artifact_schema_version.json",
        "deployment_status.json",
        "feature_order.json",
    }
)
_TEXT_METADATA_FILES = frozenset({"model_card.md"})
_ALLOWLISTED_METADATA_FILES = _JSON_METADATA_FILES | _TEXT_METADATA_FILES

# A single metadata document has no legitimate reason to be large -- this is
# generous headroom over real usage and exists only to bound how much an
# adversarial or corrupted file makes this reader read into memory.
_MAX_METADATA_BYTES = 1_000_000


@dataclass(frozen=True)
class ModelMetadata:
    """Sanitized, non-executable model metadata. Contains nothing an
    attacker could use to reconstruct model internals, and never a local
    filesystem path."""

    jurisdiction: Jurisdiction
    target_name: str
    model_version: str
    artifact_schema_version: str
    deployment_status: str
    registered_at: str
    model_card_markdown: str

    def public_summary(self) -> dict[str, Any]:
        """Everything a public route may return -- explicitly excludes any
        local filesystem path."""
        return {
            "jurisdiction": self.jurisdiction,
            "target_name": self.target_name,
            "model_version": self.model_version,
            "artifact_schema_version": self.artifact_schema_version,
            "deployment_status": self.deployment_status,
            "registered_at": self.registered_at,
        }


def _is_valid_artifact_dir(path: Path) -> bool:
    success = path / "_SUCCESS"
    manifest = path / "manifest.json"
    return (
        success.is_file()
        and not success.is_symlink()
        and manifest.is_file()
        and not manifest.is_symlink()
    )


def _is_safe_manifest_filename(name: object) -> bool:
    """A manifest-listed filename is safe to join beneath the artifact
    directory only if it is a single, ordinary path component: no
    separators (so no traversal and no subdirectories are even
    expressible), no absolute/drive/UNC qualification, no NUL byte, and
    not empty or a bare ``.``/``..``."""
    if not isinstance(name, str) or not name or "\x00" in name:
        return False
    if name in (".", ".."):
        return False
    if "/" in name or "\\" in name:
        return False
    if PureWindowsPath(name).drive or PureWindowsPath(name).root:
        return False
    if PurePosixPath(name).is_absolute():
        return False
    return True


def _safe_metadata_path(directory: Path, name: str) -> Path | None:
    """Returns the path to ``name`` beneath ``directory`` only if it is
    provably safe to open: a bare, non-traversing filename that is not a
    symlink or reparse point, is a regular file, and resolves to a path
    actually contained in ``directory``. Returns ``None`` for anything
    else -- including a file that disappears or changes type between
    checks -- rather than ever raising."""
    if not _is_safe_manifest_filename(name):
        return None
    candidate = directory / name
    try:
        if candidate.is_symlink() or not candidate.is_file():
            return None
        resolved_dir = directory.resolve(strict=True)
        resolved_candidate = candidate.resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        return None
    if not resolved_candidate.is_relative_to(resolved_dir):
        return None
    return candidate


def _read_json_file(path: Path) -> tuple[Any, str | None]:
    """Reads and parses one already-path-checked file. Returns ``(value,
    None)`` on success or ``(None, reason)`` on any expected failure --
    never raises, and the reason never includes a path or raw exception
    text."""
    try:
        raw = path.read_bytes()
    except OSError:
        return None, "metadata file could not be read"
    if len(raw) > _MAX_METADATA_BYTES:
        return None, "metadata file exceeds the maximum allowed size"
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None, "metadata file is not valid UTF-8"
    try:
        return json.loads(text), None
    except json.JSONDecodeError:
        return None, "metadata file is not valid JSON"


def _read_text_file(path: Path) -> tuple[str, str | None]:
    """Same contract as :func:`_read_json_file` for a plain-text file."""
    try:
        raw = path.read_bytes()
    except OSError:
        return "", "metadata file could not be read"
    if len(raw) > _MAX_METADATA_BYTES:
        return "", "metadata file exceeds the maximum allowed size"
    try:
        return raw.decode("utf-8"), None
    except UnicodeDecodeError:
        return "", "metadata file is not valid UTF-8"


def _verify_allowlisted_checksums(path: Path, files: Any) -> list[str] | None:
    """Checksum-verifies only the entries in :data:`_ALLOWLISTED_METADATA_FILES`
    that are present in the manifest's ``files`` mapping. Returns ``None``
    if ``files`` itself is not a mapping (a distinct, equally fail-closed
    outcome from "some files mismatched"). Deliberately never looks at,
    hashes, or opens ``point_estimator.joblib``, ``bootstrap_members.joblib``,
    or any other non-allowlisted entry -- see the module docstring."""
    if not isinstance(files, dict):
        return None
    mismatches: list[str] = []
    for name in _ALLOWLISTED_METADATA_FILES:
        expected = files.get(name)
        if expected is None:
            continue  # not part of this bundle; treated as absent, not corrupt
        if not isinstance(expected, str):
            mismatches.append(name)
            continue
        safe_path = _safe_metadata_path(path, name)
        if safe_path is None:
            mismatches.append(name)
            continue
        try:
            actual = hashlib.sha256(safe_path.read_bytes()).hexdigest()
        except OSError:
            mismatches.append(name)
            continue
        if actual != expected:
            mismatches.append(name)
    return mismatches


def _read_metadata(
    path: Path, *, expected_jurisdiction: str, require_ready: bool
) -> tuple[ModelMetadata | None, str | None]:
    """Returns ``(metadata, None)`` on success or ``(None, reason)`` on any
    failure -- never raises, never reads/parses anything outside
    :data:`_ALLOWLISTED_METADATA_FILES`, and never includes a local path or
    raw exception text in ``reason``."""
    if not _is_valid_artifact_dir(path):
        return None, "artifact is missing _SUCCESS or manifest.json"

    manifest, error = _read_json_file(path / "manifest.json")
    if error is not None:
        return None, f"manifest {error}"
    if not isinstance(manifest, dict):
        return None, "manifest has an unexpected structure"

    files = manifest.get("files")
    mismatches = _verify_allowlisted_checksums(path, files)
    if mismatches is None:
        return None, "manifest 'files' mapping is malformed"
    if mismatches:
        return None, "artifact checksum verification failed"

    if manifest.get("jurisdiction") != expected_jurisdiction:
        return None, "artifact jurisdiction mismatch"

    files_dict = files if isinstance(files, dict) else {}

    def _read_json(name: str) -> tuple[Any, str | None]:
        if name not in _JSON_METADATA_FILES or name not in files_dict:
            return None, None
        safe_path = _safe_metadata_path(path, name)
        if safe_path is None:
            return None, "a required metadata file is unavailable"
        return _read_json_file(safe_path)

    def _read_text(name: str) -> tuple[str, str | None]:
        if name not in _TEXT_METADATA_FILES or name not in files_dict:
            return "", None
        safe_path = _safe_metadata_path(path, name)
        if safe_path is None:
            return "", "a required metadata file is unavailable"
        return _read_text_file(safe_path)

    deployment_status_raw, error = _read_json("deployment_status.json")
    if error is not None:
        return None, error
    deployment_status_obj = deployment_status_raw if isinstance(deployment_status_raw, dict) else {}
    deployment_status = deployment_status_obj.get("status", "unknown")
    if not isinstance(deployment_status, str):
        deployment_status = "unknown"
    if require_ready and deployment_status != "ready":
        return None, f"artifact is not ready (status={deployment_status})"

    if require_ready:
        feature_order_raw, error = _read_json("feature_order.json")
        if error is not None:
            return None, error
        if not isinstance(feature_order_raw, list) or not all(
            isinstance(item, str) for item in feature_order_raw
        ):
            return None, "artifact feature schema is malformed"
        feature_order = tuple(feature_order_raw)
        expected = _EXPECTED_FEATURES[expected_jurisdiction]
        if not feature_order or feature_order != tuple(expected):
            return None, "artifact feature schema does not match Task 7's expectations"

    target_definition_raw, error = _read_json("target_definition.json")
    if error is not None:
        return None, error
    target_definition = target_definition_raw if isinstance(target_definition_raw, dict) else {}
    target_name = target_definition.get("target_name", "")
    if not isinstance(target_name, str):
        target_name = ""

    artifact_schema_version_raw, error = _read_json("artifact_schema_version.json")
    if error is not None:
        return None, error
    artifact_schema_version = (
        artifact_schema_version_raw if isinstance(artifact_schema_version_raw, str) else "unknown"
    )

    model_version = manifest.get("model_version", "")
    if not isinstance(model_version, str):
        model_version = ""
    registered_at = manifest.get("generated_at", "")
    if not isinstance(registered_at, str):
        registered_at = ""

    model_card_markdown, error = _read_text("model_card.md")
    if error is not None:
        return None, error

    metadata = ModelMetadata(
        jurisdiction=expected_jurisdiction,  # type: ignore[arg-type]
        target_name=target_name,
        model_version=model_version,
        artifact_schema_version=artifact_schema_version,
        deployment_status=deployment_status,
        registered_at=registered_at,
        model_card_markdown=model_card_markdown,
    )
    return metadata, None


class ModelMetadataReader:
    """One process-lifetime cache entry per jurisdiction. Never calls
    ``load_artifact``, ``joblib.load``, or ``pickle.load`` -- see the module
    docstring for the full trust-boundary explanation."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._cache: dict[str, ModelMetadata | None] = {}
        self._error_reasons: dict[str, str] = {}

    def _configured_path(self, jurisdiction: Jurisdiction) -> Path | None:
        if jurisdiction == "nyc":
            return self._settings.nyc_model_artifact_path
        return self._settings.florida_model_artifact_path

    def last_error_reason(self, jurisdiction: Jurisdiction) -> str | None:
        return self._error_reasons.get(jurisdiction)

    def get(self, jurisdiction: Jurisdiction) -> ModelMetadata | None:
        """The cached, ``ready``-only metadata used for health/prediction
        gating. Never re-reads after the first call for this jurisdiction."""
        if jurisdiction in self._cache:
            return self._cache[jurisdiction]
        metadata = self._load(jurisdiction, require_ready=True)
        self._cache[jurisdiction] = metadata
        return metadata

    def get_card(
        self, jurisdiction: Jurisdiction, *, allow_non_ready: bool
    ) -> ModelMetadata | None:
        """For the model-card route only. ``allow_non_ready=True`` is an
        explicit, administrator-gated audit path (``Settings
        .expose_non_ready_model_cards``) -- not cached, since it is expected
        to be rare (admin/debug use)."""
        if not allow_non_ready:
            return self.get(jurisdiction)
        return self._load(jurisdiction, require_ready=False)

    def _load(self, jurisdiction: Jurisdiction, *, require_ready: bool) -> ModelMetadata | None:
        path = self._configured_path(jurisdiction)
        if path is None:
            self._error_reasons[jurisdiction] = "no artifact path configured"
            return None

        metadata, error = _read_metadata(
            path, expected_jurisdiction=jurisdiction, require_ready=require_ready
        )
        if error is not None:
            self._error_reasons[jurisdiction] = error
        return metadata
