"""``POST /copilot/query`` -- the real Task 8B Copilot endpoint. Restaurant-
scoped, read-only: never trains, scores, or deserializes a model artifact,
never rebuilds the graph per request, never writes to DuckDB/Parquet, and
never accepts a client-supplied jurisdiction (always derived from the
resolved restaurant's own graph node, exactly like
``plateproof.copilot.service.CopilotService`` itself).

Restaurant existence is checked via ``Repository`` -- the same source
Task 7's other restaurant-scoped routes use -- before ``CopilotService`` is
ever consulted, so 404 semantics stay identical across the whole API. A
``GraphScaleExceededError`` while building the graph is the one condition
that becomes a 503 (required infrastructure genuinely unavailable); every
other outcome -- grounded, refused, or local-model unavailable/disabled --
is always a typed 200.
"""

from __future__ import annotations

from collections.abc import Mapping

from fastapi import APIRouter, Depends

from plateproof.api.copilot_projection import project_answer
from plateproof.api.dependencies import (
    get_app_settings,
    get_corpus_store,
    get_graph_service,
    get_intent_helper,
    get_model_metadata,
    get_repository,
)
from plateproof.api.schemas import CopilotQueryRequest, CopilotQueryResponse
from plateproof.copilot.corpus import CorpusStore
from plateproof.copilot.generators.base import IntentHelper
from plateproof.copilot.service import CopilotService
from plateproof.core.config import Settings
from plateproof.graph.builder import GraphService
from plateproof.graph.models import GraphScaleExceededError
from plateproof.serving.errors import DatastoreUnavailableError, RestaurantNotFoundError
from plateproof.serving.model_registry_service import ModelMetadataReader
from plateproof.serving.prediction_service import resolve_prediction
from plateproof.serving.repository import Repository

router = APIRouter()

_MAX_CLAIMS = 20
_MAX_CITATIONS = 30
_MAX_EXCERPT_LENGTH = 1_000


@router.post("/copilot/query", response_model=CopilotQueryResponse)
def copilot_query(
    payload: CopilotQueryRequest,
    repository: Repository = Depends(get_repository),
    model_metadata: ModelMetadataReader = Depends(get_model_metadata),
    settings: Settings = Depends(get_app_settings),
    graph_service: GraphService = Depends(get_graph_service),
    corpus_store: CorpusStore | None = Depends(get_corpus_store),
    intent_helper: IntentHelper | None = Depends(get_intent_helper),
) -> CopilotQueryResponse:
    if repository.get_restaurant(payload.restaurant_id) is None:
        raise RestaurantNotFoundError(payload.restaurant_id)

    try:
        graph_result = graph_service.get()
    except GraphScaleExceededError as exc:
        raise DatastoreUnavailableError(
            "the documented-inspection graph could not be built"
        ) from exc

    def _forecast_lookup(restaurant_id: str, jurisdiction: str) -> Mapping[str, object] | None:
        availability = resolve_prediction(
            repository,
            model_metadata,
            restaurant_id=restaurant_id,
            jurisdiction=jurisdiction,
            staleness_days=settings.prediction_staleness_days,
        )
        return availability.row if availability.status == "available" else None

    service = CopilotService(
        graph_result.graph,
        corpus_store,
        forecast_lookup=_forecast_lookup,
        intent_helper=intent_helper,
        max_question_length=settings.copilot_max_question_length,
    )
    answer = service.answer(restaurant_id=payload.restaurant_id, question=payload.question)
    return project_answer(
        answer,
        max_claims=_MAX_CLAIMS,
        max_citations=_MAX_CITATIONS,
        max_excerpt_length=_MAX_EXCERPT_LENGTH,
    )
