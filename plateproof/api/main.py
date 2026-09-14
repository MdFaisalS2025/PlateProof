"""FastAPI app factory for the PlateProof service.

``create_app`` builds a fresh, isolated app: its own repository (a DuckDB
connection over the configured Parquet tables) and its own model metadata
reader. Tests construct an app against a temporary :class:`Settings`
instance instead of relying on any module-level singleton, so isolated
repositories and fake prediction data never leak between tests.

Neither the repository nor the model metadata reader ever deserializes or
executes a Task 6 model artifact -- see
``plateproof.serving.model_registry_service`` for the trust boundary.
"""

from __future__ import annotations

from fastapi import FastAPI

from plateproof.api.errors import register_exception_handlers
from plateproof.api.routes import copilot, deferred, health, models, predictions, restaurants
from plateproof.copilot.wiring import build_corpus_store, build_intent_helper
from plateproof.core.config import Settings, get_settings
from plateproof.graph.builder import GraphService
from plateproof.serving.model_registry_service import ModelMetadataReader
from plateproof.serving.repository import open_repository


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()

    app = FastAPI(
        title="PlateProof API",
        description="Evidence-based restaurant inspection intelligence for NYC and Florida.",
    )
    app.state.settings = resolved_settings
    app.state.repository = open_repository(resolved_settings)
    app.state.model_metadata = ModelMetadataReader(resolved_settings)
    # GraphService construction is cheap (no I/O) -- the actual graph build
    # happens lazily on first request and is then cached by source
    # fingerprint (see plateproof.graph.builder.GraphService), never
    # rebuilt on every request. The corpus and the optional intent helper
    # are each built exactly once here, at app-construction time.
    app.state.graph_service = GraphService(
        resolved_settings.resolve_path(resolved_settings.processed_data_dir)
    )
    app.state.corpus_store = build_corpus_store(resolved_settings)
    app.state.intent_helper = build_intent_helper(resolved_settings)

    app.include_router(health.router)
    app.include_router(restaurants.router)
    app.include_router(predictions.router)
    app.include_router(models.router)
    app.include_router(copilot.router)
    app.include_router(deferred.router)

    register_exception_handlers(app)
    return app
