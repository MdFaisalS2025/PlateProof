"""Loads and caches a single, validated, ``ready`` Task 6 model artifact per
jurisdiction, using only administrator-configured local paths -- never a
path supplied by a request. Deserializes each artifact at most once per
process; routes and Streamlit pages hold a reference to the cached
:class:`LoadedModel`, never call ``load_artifact`` themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from plateproof.core.config import Settings
from plateproof.models.florida_risk import FL_FEATURE_LIST
from plateproof.models.nyc_risk import NYC_FEATURE_LIST
from plateproof.models.risk_bands import RiskBandThresholds
from plateproof.models.training import BootstrapMember, UncertaintyConfig, load_artifact

Jurisdiction = Literal["nyc", "florida"]

_EXPECTED_FEATURES: dict[str, tuple[str, ...]] = {
    "nyc": NYC_FEATURE_LIST,
    "florida": FL_FEATURE_LIST,
}


@dataclass(frozen=True)
class LoadedModel:
    jurisdiction: Jurisdiction
    target_name: str
    model_version: str
    artifact_schema_version: str
    deployment_status: str
    point_estimator: Any
    members: list[BootstrapMember]
    uncertainty_config: UncertaintyConfig | None
    risk_band_thresholds: RiskBandThresholds | None
    feature_order: tuple[str, ...]
    feature_dtypes: dict[str, str]
    model_card_markdown: str
    registered_at: str

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


class ModelCache:
    """One process-lifetime cache entry per jurisdiction. ``get`` is cheap
    after the first call -- it never re-runs ``load_artifact``."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._cache: dict[str, LoadedModel | None] = {}
        self._error_reasons: dict[str, str] = {}

    def _configured_path(self, jurisdiction: Jurisdiction) -> Path | None:
        if jurisdiction == "nyc":
            return self._settings.nyc_model_artifact_path
        return self._settings.florida_model_artifact_path

    def last_error_reason(self, jurisdiction: Jurisdiction) -> str | None:
        return self._error_reasons.get(jurisdiction)

    def get(self, jurisdiction: Jurisdiction) -> LoadedModel | None:
        """The cached, ``ready``-only model used for scoring/health/serving.
        Never re-runs ``load_artifact`` after the first call."""
        if jurisdiction in self._cache:
            return self._cache[jurisdiction]
        model = self._load(jurisdiction, require_ready=True)
        self._cache[jurisdiction] = model
        return model

    def get_card(self, jurisdiction: Jurisdiction, *, allow_non_ready: bool) -> LoadedModel | None:
        """For the model-card route only. ``allow_non_ready=True`` is an
        explicit, administrator-gated audit path (``Settings
        .expose_non_ready_model_cards``) -- it is never used for scoring and
        is not cached, since it is expected to be rare (admin/debug use)."""
        if not allow_non_ready:
            return self.get(jurisdiction)
        return self._load(jurisdiction, require_ready=False)

    def _load(self, jurisdiction: Jurisdiction, *, require_ready: bool) -> LoadedModel | None:
        path = self._configured_path(jurisdiction)
        if path is None:
            self._error_reasons[jurisdiction] = "no artifact path configured"
            return None

        try:
            loaded = load_artifact(
                path,
                trusted=True,
                expected_jurisdiction=jurisdiction,
                require_ready=require_ready,
            )
        except Exception as exc:  # noqa: BLE001 - recorded, never re-raised into a request
            self._error_reasons[jurisdiction] = f"{type(exc).__name__}: could not load artifact"
            return None

        feature_order = tuple(loaded.get("feature_order.json") or ())
        if require_ready:
            expected = _EXPECTED_FEATURES[jurisdiction]
            if not feature_order or feature_order != tuple(expected):
                self._error_reasons[jurisdiction] = (
                    "artifact feature schema does not match Task 7's expectations"
                )
                return None

        return LoadedModel(
            jurisdiction=jurisdiction,
            target_name=(loaded.get("target_definition.json") or {}).get("target_name", ""),
            model_version=loaded["manifest"]["model_version"],
            artifact_schema_version=loaded.get("artifact_schema_version.json", "unknown"),
            deployment_status=(loaded.get("deployment_status.json") or {}).get("status", "unknown"),
            point_estimator=loaded.get("point_estimator.joblib"),
            members=[],  # bootstrap members not needed for card/health display
            uncertainty_config=(
                UncertaintyConfig(**loaded["uncertainty_config.json"])
                if "uncertainty_config.json" in loaded
                else None
            ),
            risk_band_thresholds=(
                RiskBandThresholds(**loaded["risk_band_thresholds.json"])
                if "risk_band_thresholds.json" in loaded
                else None
            ),
            feature_order=feature_order,
            feature_dtypes=loaded.get("feature_dtypes.json", {}),
            model_card_markdown=loaded.get("model_card.md", ""),
            registered_at=loaded["manifest"]["generated_at"],
        )
