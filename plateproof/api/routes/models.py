"""GET /models/{jurisdiction}/card. Returns only ready-model cards by
default; a non-ready evaluation card is served only when the administrator
has explicitly set ``expose_non_ready_model_cards=True``.

This route exposes only sanitized, non-executable metadata and Markdown
read via ``plateproof.serving.model_registry_service.ModelMetadataReader``
-- it never deserializes the point estimator or bootstrap ensemble.
"""

from __future__ import annotations

from typing import Literal, cast

from fastapi import APIRouter, Depends

from plateproof.api.dependencies import get_app_settings, get_model_metadata
from plateproof.api.schemas import ModelCardResponse
from plateproof.core.config import Settings
from plateproof.serving.errors import InvalidQueryError, ModelCardNotFoundError
from plateproof.serving.model_registry_service import ModelMetadataReader

router = APIRouter()

_JURISDICTIONS = frozenset({"nyc", "florida"})


@router.get("/models/{jurisdiction}/card", response_model=ModelCardResponse)
def get_model_card(
    jurisdiction: str,
    model_metadata: ModelMetadataReader = Depends(get_model_metadata),
    settings: Settings = Depends(get_app_settings),
) -> ModelCardResponse:
    if jurisdiction not in _JURISDICTIONS:
        raise InvalidQueryError(f"unsupported jurisdiction: {jurisdiction!r}")
    validated_jurisdiction = cast(Literal["nyc", "florida"], jurisdiction)

    metadata = model_metadata.get_card(
        validated_jurisdiction, allow_non_ready=settings.expose_non_ready_model_cards
    )
    if metadata is None:
        raise ModelCardNotFoundError(f"no model card available for {jurisdiction!r}")

    return ModelCardResponse(
        jurisdiction=validated_jurisdiction,
        target_name=metadata.target_name,
        model_version=metadata.model_version,
        readiness_status=metadata.deployment_status,
        markdown=metadata.model_card_markdown,
        registered_at=metadata.registered_at,
    )
