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

This reader validates artifact integrity (``_SUCCESS`` presence and sha256
checksums, exactly as ``plateproof.models.training`` does) using only raw
byte hashing -- hashing never executes file content, unlike
``joblib.load``/``pickle.load``. It then parses only a fixed, explicit set of
JSON/Markdown metadata files. ``point_estimator.joblib`` and
``bootstrap_members.joblib`` are checksum-verified as raw bytes like every
other file, but their content is never opened, parsed, or deserialized by
this module under any circumstance.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
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
    return (path / "_SUCCESS").is_file() and (path / "manifest.json").is_file()


def _checksum_mismatches(path: Path, manifest: dict[str, Any]) -> list[str]:
    """Byte-level sha256 verification only -- hashing raw bytes never
    executes or interprets file content, so this is safe to run against
    every file in the manifest, including the joblib ones."""
    mismatches: list[str] = []
    for name, expected in manifest.get("files", {}).items():
        file_path = path / name
        if not file_path.is_file():
            mismatches.append(name)
            continue
        actual = hashlib.sha256(file_path.read_bytes()).hexdigest()
        if actual != expected:
            mismatches.append(name)
    return mismatches


def _read_metadata(
    path: Path, *, expected_jurisdiction: str, require_ready: bool
) -> tuple[ModelMetadata | None, str | None]:
    """Returns ``(metadata, None)`` on success or ``(None, reason)`` on any
    failure -- never raises, and never reads/parses anything outside
    ``_JSON_METADATA_FILES`` / ``_TEXT_METADATA_FILES``."""
    if not _is_valid_artifact_dir(path):
        return None, "artifact is missing _SUCCESS or manifest.json"

    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    mismatches = _checksum_mismatches(path, manifest)
    if mismatches:
        return None, f"artifact checksum mismatch: {mismatches}"

    if manifest.get("jurisdiction") != expected_jurisdiction:
        return None, "artifact jurisdiction mismatch"

    files = manifest.get("files", {})

    def _read_json(name: str) -> Any:
        if name not in _JSON_METADATA_FILES or name not in files:
            return None
        return json.loads((path / name).read_text(encoding="utf-8"))

    def _read_text(name: str) -> str:
        if name not in _TEXT_METADATA_FILES or name not in files:
            return ""
        return (path / name).read_text(encoding="utf-8")

    deployment_status = (_read_json("deployment_status.json") or {}).get("status", "unknown")
    if require_ready and deployment_status != "ready":
        return None, f"artifact is not ready (status={deployment_status})"

    if require_ready:
        feature_order = tuple(_read_json("feature_order.json") or ())
        expected = _EXPECTED_FEATURES[expected_jurisdiction]
        if not feature_order or feature_order != tuple(expected):
            return None, "artifact feature schema does not match Task 7's expectations"

    target_definition = _read_json("target_definition.json") or {}
    metadata = ModelMetadata(
        jurisdiction=expected_jurisdiction,  # type: ignore[arg-type]
        target_name=target_definition.get("target_name", ""),
        model_version=manifest.get("model_version", ""),
        artifact_schema_version=_read_json("artifact_schema_version.json") or "unknown",
        deployment_status=deployment_status,
        registered_at=manifest.get("generated_at", ""),
        model_card_markdown=_read_text("model_card.md"),
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
