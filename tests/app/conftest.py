"""AppTest fixtures for the Streamlit pages. Each test points
PLATEPROOF_PROCESSED_DATA_DIR (and friends) at a temporary directory via
environment variables, since Streamlit's AppTest executes the page script
directly and pages read settings via ``plateproof.core.config.get_settings``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

_APP_DIR = Path(__file__).resolve().parent.parent.parent / "app"


@pytest.fixture
def app_env(monkeypatch: pytest.MonkeyPatch, processed_dir: Path) -> Any:
    from plateproof.core.config import get_settings

    def _set(**overrides: str) -> None:
        monkeypatch.setenv("PLATEPROOF_PROCESSED_DATA_DIR", str(processed_dir))
        for key, value in overrides.items():
            monkeypatch.setenv(f"PLATEPROOF_{key.upper()}", value)
        get_settings.cache_clear()

    _set()
    yield _set
    get_settings.cache_clear()


@pytest.fixture
def app_path() -> Any:
    def _path(*parts: str) -> str:
        return str(_APP_DIR.joinpath(*parts))

    return _path
