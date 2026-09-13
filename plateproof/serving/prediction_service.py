"""Shared prediction-lookup logic used by both the FastAPI prediction route
and the Streamlit inspection-history page -- business rules live here once,
not duplicated per surface.

Reads only the precomputed ``predictions``/``model_registry`` tables (via
``Repository``) and sanitized artifact metadata (via
``ModelMetadataReader``). Never deserializes or runs a model artifact, and
never computes a prediction.

The processed ``model_registry`` table -- written by the offline scoring
step alongside ``predictions.parquet`` -- is the authoritative source for
which target/model_version/artifact_schema_version a given restaurant's
precomputed prediction should be validated against. The currently
*configured* artifact (read via ``ModelMetadataReader``) is cross-checked
against it: if they disagree (e.g. an administrator has swapped in a new
model artifact but not yet re-run offline scoring), this fails closed --
the precomputed predictions are treated as out of sync with the active
model, never served as though they still applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from plateproof.serving.model_registry_service import ModelMetadataReader
from plateproof.serving.repository import Repository

PredictionStatus = Literal[
    "available", "no_ready_model", "not_scored", "stale", "insufficient_history"
]


@dataclass(frozen=True)
class PredictionAvailability:
    status: PredictionStatus
    detail: str
    row: dict[str, Any] | None = None
    insufficient_history_reason: str | None = None


_NO_READY_MODEL = "No ready model is currently configured for this jurisdiction."
_REGISTRY_ARTIFACT_DISAGREE = (
    "The precomputed predictions no longer match the currently active model."
)
_STALE = (
    "The most recent prediction for this restaurant is older than the configured freshness window."
)
_NOT_SCORED = "No precomputed prediction is available for this restaurant yet."
_INSUFFICIENT_HISTORY = "This restaurant does not yet have enough recorded history for a forecast."


def resolve_prediction(
    repository: Repository,
    model_metadata: ModelMetadataReader,
    *,
    restaurant_id: str,
    jurisdiction: str,
    staleness_days: int,
) -> PredictionAvailability:
    """Prefer the processed ``model_registry`` table as the authoritative
    active-model source for prediction lookup; fail closed (never guess) if
    it disagrees with the currently configured artifact's own metadata."""
    registry_entry = repository.model_registry_entry(jurisdiction)
    if registry_entry is None or registry_entry.get("deployment_status") != "ready":
        return PredictionAvailability(status="no_ready_model", detail=_NO_READY_MODEL)

    metadata = model_metadata.get(jurisdiction)  # type: ignore[arg-type]
    if metadata is None:
        return PredictionAvailability(status="no_ready_model", detail=_NO_READY_MODEL)

    if (
        registry_entry["target_name"] != metadata.target_name
        or registry_entry["model_version"] != metadata.model_version
        or registry_entry["artifact_schema_version"] != metadata.artifact_schema_version
    ):
        return PredictionAvailability(status="stale", detail=_REGISTRY_ARTIFACT_DISAGREE)

    lookup = repository.latest_prediction(
        restaurant_id=restaurant_id,
        jurisdiction=jurisdiction,
        target_name=registry_entry["target_name"],
        active_model_version=registry_entry["model_version"],
        active_artifact_schema_version=registry_entry["artifact_schema_version"],
        staleness_days=staleness_days,
    )
    if lookup.row is None:
        if lookup.stale_row_exists:
            return PredictionAvailability(status="stale", detail=_STALE)
        return PredictionAvailability(status="not_scored", detail=_NOT_SCORED)

    row = lookup.row
    if row["risk_band"] == "insufficient_history" or row["probability"] is None:
        return PredictionAvailability(
            status="insufficient_history",
            detail=_INSUFFICIENT_HISTORY,
            insufficient_history_reason=row.get("insufficient_history_reason"),
        )

    return PredictionAvailability(status="available", detail="", row=row)
