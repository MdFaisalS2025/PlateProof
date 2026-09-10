"""Typed application settings sourced from the environment.

Configuration is read from ``PLATEPROOF_``-prefixed environment variables, with an
optional ``.env`` file for local development. Reconciliation between this env-based
layer and ``configs/base.yaml`` is deferred (see Task 1 notes); no YAML is loaded
here.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Process-wide configuration with safe, offline defaults.

    Field names mirror the keys documented in ``.env.example``. Google and local
    LLM integrations default to disabled so the MVP runs without paid services.
    """

    model_config = SettingsConfigDict(
        env_prefix="PLATEPROOF_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    env: str = "development"
    data_dir: Path = Path("data")
    database_path: Path = Path("data/processed/plateproof.duckdb")

    google_integration_enabled: bool = False
    google_maps_api_key: str | None = None
    google_oauth_client_id: str | None = None
    google_oauth_client_secret: str | None = None

    local_llm_enabled: bool = False
    local_llm_base_url: str = "http://localhost:11434"
    local_llm_model: str | None = None


@lru_cache
def get_settings() -> Settings:
    """Return the cached :class:`Settings` singleton for this process."""
    return Settings()
