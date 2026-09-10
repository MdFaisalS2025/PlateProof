"""Shared test fixtures.

Keeps configuration tests deterministic: they must not be influenced by the
developer's shell environment or by a repository-local ``.env`` file. The fixture
is opt-in (not autouse) so it does not change the working directory for tests
that legitimately rely on repository-relative paths.
"""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from plateproof.core.config import get_settings


@pytest.fixture
def isolated_settings_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[None]:
    """Strip ``PLATEPROOF_*`` env vars, drop the cache, and run from a clean cwd."""
    for key in list(os.environ):
        if key.startswith("PLATEPROOF_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
