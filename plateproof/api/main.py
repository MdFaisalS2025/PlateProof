"""FastAPI app factory for the PlateProof service.

``create_app`` builds a fresh, isolated app: its own repository (a DuckDB
connection over the configured Parquet tables) and its own model metadata
reader. Tests construct an app against a temporary :class:`Settings`
instance instead of relying on any module-level singleton, so isolated
repositories and fake prediction data never leak between tests.

Neither the repository nor the model metadata reader ever deserializes or
executes a Task 6 model artifact -- see
``plateproof.serving.model_registry_service`` for the trust boundary.

Task 9B: this process owns exactly one document worker pool
(``app.state.document_worker_pool``), constructed eagerly here (spawning a
worker process is lazy -- see ``WorkerPool.submit`` -- so constructing the
pool object itself is cheap) and shut down via the app's ``lifespan``
context manager below. This pool is entirely separate from any Streamlit
process's own pool (``app.theme.document_worker_pool``) -- pool-size limits
are per application process, never a machine-wide ceiling; see
``plateproof/api/routes/documents.py`` and ``README.md`` for the
deployment-sizing consequence.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from plateproof.api.errors import register_exception_handlers
from plateproof.api.routes import copilot, documents, health, models, predictions, restaurants
from plateproof.copilot.wiring import build_corpus_store, build_intent_helper
from plateproof.core.config import Settings, get_settings
from plateproof.documents.worker.pool import WorkerPool, WorkerPoolConfig
from plateproof.graph.builder import GraphService
from plateproof.serving.model_registry_service import ModelMetadataReader
from plateproof.serving.repository import open_repository


def _build_document_worker_pool(settings: Settings) -> WorkerPool:
    return WorkerPool(
        config=WorkerPoolConfig(
            pool_size=settings.documents_worker_pool_size,
            page_timeout_seconds=settings.documents_worker_page_timeout_seconds,
            total_timeout_seconds=settings.documents_worker_total_timeout_seconds,
            kill_grace_seconds=settings.documents_worker_kill_grace_seconds,
            admission_timeout_seconds=settings.documents_worker_admission_timeout_seconds,
        )
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()

    @asynccontextmanager
    async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            # Explicit process-level shutdown: terminate/join/kill any live
            # worker processes and close every pipe handle, mirroring the
            # per-job cleanup guarantee WorkerPool already provides.
            app.state.document_worker_pool.shutdown()

    app = FastAPI(
        title="PlateProof API",
        description="Evidence-based restaurant inspection intelligence for NYC and Florida.",
        lifespan=_lifespan,
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
    app.state.document_worker_pool = _build_document_worker_pool(resolved_settings)

    app.include_router(health.router)
    app.include_router(restaurants.router)
    app.include_router(predictions.router)
    app.include_router(models.router)
    app.include_router(copilot.router)
    app.include_router(documents.router)

    register_exception_handlers(app)
    return app
