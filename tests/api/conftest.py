"""Builds an isolated FastAPI TestClient per test, against a temporary
Settings instance -- no shared/global app state between tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def make_client() -> Any:
    def _make(**settings_overrides: Any) -> Any:
        from fastapi.testclient import TestClient

        from plateproof.api.main import create_app
        from plateproof.core.config import Settings

        settings = Settings(_env_file=None, **settings_overrides)
        app = create_app(settings)
        return TestClient(app)

    return _make


@pytest.fixture
def client(processed_dir: Path, make_client: Any) -> Any:
    return make_client(processed_data_dir=processed_dir)
