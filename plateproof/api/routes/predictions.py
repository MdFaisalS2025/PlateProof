"""GET /restaurants/{restaurant_id}/prediction.

Reads only the precomputed ``predictions`` table -- never deserializes or
runs a model artifact in the request path. Prediction unavailability
(no ready model, no precomputed score, insufficient history, stale score)
is always a typed, successful HTTP 200 response; 503 is reserved for an
actually-unavailable required datastore.
"""

from __future__ import annotations

from typing import Literal, cast

from fastapi import APIRouter, Depends

from plateproof.api.dependencies import get_app_settings, get_model_cache, get_repository
from plateproof.api.schemas import PredictionAvailable, PredictionResponse, PredictionUnavailable
from plateproof.core.config import Settings
from plateproof.serving.errors import RestaurantNotFoundError
from plateproof.serving.model_registry_service import ModelCache
from plateproof.serving.repository import Repository

router = APIRouter()


@router.get("/restaurants/{restaurant_id}/prediction", response_model=PredictionResponse)
def get_prediction(
    restaurant_id: str,
    repository: Repository = Depends(get_repository),
    model_cache: ModelCache = Depends(get_model_cache),
    settings: Settings = Depends(get_app_settings),
) -> PredictionAvailable | PredictionUnavailable:
    restaurant = repository.get_restaurant(restaurant_id)
    if restaurant is None:
        raise RestaurantNotFoundError(restaurant_id)
    jurisdiction: str = restaurant["jurisdiction"]
    validated_jurisdiction = cast(Literal["nyc", "florida"], jurisdiction)

    model = model_cache.get(validated_jurisdiction)
    if model is None:
        return PredictionUnavailable(
            restaurant_id=restaurant_id,
            jurisdiction=jurisdiction,
            reason="no_ready_model",
            detail="No ready model is currently configured for this jurisdiction.",
        )

    lookup = repository.latest_prediction(
        restaurant_id=restaurant_id,
        jurisdiction=jurisdiction,
        target_name=model.target_name,
        active_model_version=model.model_version,
        active_artifact_schema_version=model.artifact_schema_version,
        staleness_days=settings.prediction_staleness_days,
    )
    if lookup.row is None:
        if lookup.stale_row_exists:
            return PredictionUnavailable(
                restaurant_id=restaurant_id,
                jurisdiction=jurisdiction,
                reason="stale",
                detail="The most recent prediction for this restaurant is older than the "
                "configured freshness window.",
            )
        return PredictionUnavailable(
            restaurant_id=restaurant_id,
            jurisdiction=jurisdiction,
            reason="not_scored",
            detail="No precomputed prediction is available for this restaurant yet.",
        )

    row = lookup.row
    if row["risk_band"] == "insufficient_history" or row["probability"] is None:
        return PredictionUnavailable(
            restaurant_id=restaurant_id,
            jurisdiction=jurisdiction,
            reason="insufficient_history",
            detail="This restaurant does not yet have enough recorded history for a forecast.",
            insufficient_history_reason=row.get("insufficient_history_reason"),
        )

    return PredictionAvailable(
        restaurant_id=restaurant_id,
        jurisdiction=jurisdiction,
        target_name=row["target_name"],
        probability=row["probability"],
        lower_bound=row["lower_bound"],
        upper_bound=row["upper_bound"],
        risk_band=row["risk_band"],
        model_version=row["model_version"],
        generated_at=row["generated_at"],
        as_of_date=row["as_of_date"],
        uncertainty_status=row["uncertainty_status"],
        calibration_status=row["calibration_status"],
        readiness_status=row["readiness_status"],
    )
