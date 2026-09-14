"""Typed application settings sourced from the environment.

Configuration is read from ``PLATEPROOF_``-prefixed environment variables, with an
optional ``.env`` file for local development. Reconciliation between this env-based
layer and ``configs/base.yaml`` is deferred (see Task 1 notes); no YAML is loaded
here.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

#: The application's base directory (the repository root, three levels above
#: this file: plateproof/core/config.py). Relative Settings paths resolve
#: against this, never against a caller's unpredictable current working
#: directory.
APP_BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent


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
    # --- Task 8B: optional local intent assistance + Copilot API/UI ---- #
    local_llm_connect_timeout_seconds: float = 2.0
    local_llm_read_timeout_seconds: float = 5.0
    local_llm_max_response_bytes: int = 65_536
    local_llm_min_confidence: float = 0.6
    copilot_max_question_length: int = 500
    guidance_corpus_manifest_path: Path | None = Path("data/reference/guidance/manifest.json")

    # --- Task 7: API/UI serving --------------------------------------- #
    processed_data_dir: Path = Path("data/processed")
    nyc_model_artifact_path: Path | None = None
    florida_model_artifact_path: Path | None = None
    prediction_table_path: Path | None = None
    expose_non_ready_model_cards: bool = False
    max_page_size: int = 50
    prediction_staleness_days: int = 90

    def resolve_path(self, path: Path) -> Path:
        """Resolve ``path`` against :data:`APP_BASE_DIR` when relative;
        return an absolute path unchanged. Every local filesystem setting
        (data directory, database path, artifact paths) must be resolved
        this way rather than against ``Path.cwd()``, since a Streamlit page
        or a script invoked from an arbitrary directory must not silently
        read/write a different location than the FastAPI process does.
        """
        return path if path.is_absolute() else (APP_BASE_DIR / path)


@lru_cache
def get_settings() -> Settings:
    """Return the cached :class:`Settings` singleton for this process."""
    return Settings()
