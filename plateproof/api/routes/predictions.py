"""GET /restaurants/{restaurant_id}/prediction.

Reads only the precomputed ``predictions``/``model_registry`` tables and
sanitized artifact metadata (via
``plateproof.serving.prediction_service.resolve_prediction``) -- never
deserializes or runs a model artifact in the request path. Prediction
unavailability (no ready model, no precomputed score, insufficient history,
stale score) is always a typed, successful HTTP 200 response; 503 is
reserved for an actually-unavailable required datastore.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from plateproof.api.dependencies import get_app_settings, get_model_metadata, get_repository
from plateproof.api.schemas import PredictionAvailable, PredictionResponse, PredictionUnavailable
from plateproof.core.config import Settings
from plateproof.serving.errors import RestaurantNotFoundError
from plateproof.serving.model_registry_service import ModelMetadataReader
from plateproof.serving.prediction_service import resolve_prediction
from plateproof.serving.repository import Repository

router = APIRouter()


@router.get("/restaurants/{restaurant_id}/prediction", response_model=PredictionResponse)
def get_prediction(
    restaurant_id: str,
    repository: Repository = Depends(get_repository),
    model_metadata: ModelMetadataReader = Depends(get_model_metadata),
    settings: Settings = Depends(get_app_settings),
) -> PredictionAvailable | PredictionUnavailable:
    restaurant = repository.get_restaurant(restaurant_id)
    if restaurant is None:
        raise RestaurantNotFoundError(restaurant_id)
    jurisdiction: str = restaurant["jurisdiction"]

    availability = resolve_prediction(
        repository,
        model_metadata,
        restaurant_id=restaurant_id,
        jurisdiction=jurisdiction,
        staleness_days=settings.prediction_staleness_days,
    )

    if availability.status != "available":
        return PredictionUnavailable(
            restaurant_id=restaurant_id,
            jurisdiction=jurisdiction,
            reason=availability.status,
            detail=availability.detail,
            insufficient_history_reason=availability.insufficient_history_reason,
        )

    row = availability.row
    assert row is not None  # "available" always carries a row
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
