"""GET /health -- reports component availability without leaking local
paths, secrets, or exception detail. The whole service is never marked
unavailable merely because Michelin, Google, or a model is absent.

Model availability is determined from sanitized metadata only (see
``plateproof.serving.model_registry_service``) -- this route never
deserializes or executes a Task 6 model artifact.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from plateproof.api.dependencies import get_app_settings, get_model_metadata, get_repository
from plateproof.api.schemas import HealthComponent, HealthResponse
from plateproof.copilot.wiring import local_ai_health_status
from plateproof.core.config import Settings
from plateproof.serving.model_registry_service import ModelMetadataReader
from plateproof.serving.repository import Repository

router = APIRouter()

try:
    import plateproof as _plateproof

    _VERSION = _plateproof.__version__
except Exception:  # pragma: no cover - defensive only
    _VERSION = "unknown"

_REQUIRED_OK = {"ok"}
_DEGRADING = {"unavailable", "empty"}


@router.get("/health", response_model=HealthResponse)
def get_health(
    repository: Repository = Depends(get_repository),
    model_metadata: ModelMetadataReader = Depends(get_model_metadata),
    settings: Settings = Depends(get_app_settings),
) -> HealthResponse:
    snapshot = repository.health_snapshot()
    nyc_model_status = "ok" if model_metadata.get("nyc") is not None else "unavailable"
    florida_model_status = "ok" if model_metadata.get("florida") is not None else "unavailable"
    # Task 10: "google" reports whether the link-only Google Maps feature
    # (plateproof.serving.display.google_maps_search_link) is enabled --
    # never Google API reachability. There is no Google API in this
    # feature to be reachable or not: PlateProof's server makes no network
    # call to Google at any point.
    google_status = "ok" if settings.google_integration_enabled else "disabled"
    # No live network probe of the local model server happens here -- that
    # would make this shared health endpoint slow and flaky.
    # local_ai_health_status distinguishes "disabled" (not enabled),
    # "configured_unverified" (enabled, every setting structurally valid,
    # but not live-probed), and "unavailable" (enabled but misconfigured
    # -- a helper could not even be constructed) without ever connecting
    # to Ollama.
    local_ai_status, _local_ai_detail = local_ai_health_status(settings)

    components = [
        HealthComponent(name="datastore", status=snapshot["datastore"]),
        HealthComponent(name="nyc_data", status=snapshot["nyc_data"]),
        HealthComponent(name="florida_data", status=snapshot["florida_data"]),
        HealthComponent(name="michelin", status=snapshot["michelin"]),
        HealthComponent(name="nyc_model", status=nyc_model_status),
        HealthComponent(name="florida_model", status=florida_model_status),
        HealthComponent(name="google", status=google_status),
        HealthComponent(name="local_ai", status=local_ai_status),
    ]

    if snapshot["datastore"] == "unavailable":
        overall = "unavailable"
    elif any(c.status in _DEGRADING for c in components):
        overall = "degraded"
    else:
        overall = "ok"

    return HealthResponse(status=overall, version=_VERSION, components=components)
