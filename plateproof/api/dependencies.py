"""FastAPI dependency providers. Every route depends on these, never on a
module-level global -- ``create_app`` builds a fresh repository and model
metadata reader per app instance, so tests can construct an isolated app
against a temporary datastore without monkeypatching production state.
"""

from __future__ import annotations

from fastapi import Request

from plateproof.core.config import Settings
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
