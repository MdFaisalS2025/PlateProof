"""FastAPI app factory for the PlateProof service.

``create_app`` builds a fresh, isolated app: its own repository (a DuckDB
connection over the configured Parquet tables) and its own model cache.
Tests construct an app against a temporary :class:`Settings` instance
instead of relying on any module-level singleton, so isolated repositories
and fake prediction data never leak between tests.
"""

from __future__ import annotations

from fastapi import FastAPI

from plateproof.api.errors import register_exception_handlers
from plateproof.api.routes import deferred, health, models, predictions, restaurants
from plateproof.core.config import Settings, get_settings
from plateproof.serving.model_registry_service import ModelCache
from plateproof.serving.repository import open_repository


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()

    app = FastAPI(
        title="PlateProof API",
        description="Evidence-based restaurant inspection intelligence for NYC and Florida.",
    )
    app.state.settings = resolved_settings
    app.state.repository = open_repository(resolved_settings)
    app.state.model_cache = ModelCache(resolved_settings)

    app.include_router(health.router)
    app.include_router(restaurants.router)
    app.include_router(predictions.router)
    app.include_router(models.router)
    app.include_router(deferred.router)

    register_exception_handlers(app)
    return app
