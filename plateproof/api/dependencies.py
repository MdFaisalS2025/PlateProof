"""FastAPI dependency providers. Every route depends on these, never on a
module-level global -- ``create_app`` builds a fresh repository and model
metadata reader per app instance, so tests can construct an isolated app
against a temporary datastore without monkeypatching production state.
"""

from __future__ import annotations

from fastapi import Request

from plateproof.copilot.corpus import CorpusStore
from plateproof.copilot.generators.base import IntentHelper
from plateproof.core.config import Settings
from plateproof.documents.worker.pool import WorkerPool
from plateproof.graph.builder import GraphService
from plateproof.serving.model_registry_service import ModelMetadataReader
from plateproof.serving.repository import Repository


def get_app_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_repository(request: Request) -> Repository:
    repository: Repository = request.app.state.repository
    return repository


def get_model_metadata(request: Request) -> ModelMetadataReader:
    model_metadata: ModelMetadataReader = request.app.state.model_metadata
    return model_metadata


def get_graph_service(request: Request) -> GraphService:
    graph_service: GraphService = request.app.state.graph_service
    return graph_service


def get_corpus_store(request: Request) -> CorpusStore | None:
    corpus_store: CorpusStore | None = request.app.state.corpus_store
    return corpus_store


def get_intent_helper(request: Request) -> IntentHelper | None:
    intent_helper: IntentHelper | None = request.app.state.intent_helper
    return intent_helper


def get_document_worker_pool(request: Request) -> WorkerPool:
    """This process's own document worker pool (Task 9B). Pool limits are
    per application process -- a separate FastAPI process, and a separate
    Streamlit process (see ``app.theme.document_worker_pool``), each own
    their own pool; see the module docstring of
    ``plateproof.api.routes.documents`` for the deployment-sizing
    consequence of that."""
    pool: WorkerPool = request.app.state.document_worker_pool
    return pool
