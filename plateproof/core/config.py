"""Typed application settings sourced from the environment.

Configuration is read from ``PLATEPROOF_``-prefixed environment variables, with an
optional ``.env`` file for local development. Reconciliation between this env-based
layer and ``configs/base.yaml`` is deferred (see Task 1 notes); no YAML is loaded
here.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from plateproof.copilot.question_validation import ABSOLUTE_MAX_QUESTION_LENGTH
from plateproof.documents.limits import (
    ABSOLUTE_MAX_ADMISSION_TIMEOUT_SECONDS,
    ABSOLUTE_MAX_KILL_GRACE_SECONDS,
    ABSOLUTE_MAX_PAGE_TIMEOUT_SECONDS,
    ABSOLUTE_MAX_PAGES,
    ABSOLUTE_MAX_PIXELS_PER_PAGE,
    ABSOLUTE_MAX_POOL_SIZE,
    ABSOLUTE_MAX_TEXT_BYTES_PER_DOCUMENT,
    ABSOLUTE_MAX_TOTAL_PREVIEW_BYTES,
    ABSOLUTE_MAX_TOTAL_TIMEOUT_SECONDS,
    ABSOLUTE_MAX_UPLOAD_BYTES,
    DEFAULT_ADMISSION_TIMEOUT_SECONDS,
    DEFAULT_KILL_GRACE_SECONDS,
    DEFAULT_MAX_PAGES,
    DEFAULT_MAX_PIXELS_PER_PAGE,
    DEFAULT_MAX_TEXT_BYTES_PER_DOCUMENT,
    DEFAULT_MAX_TOTAL_PREVIEW_BYTES,
    DEFAULT_MAX_UPLOAD_BYTES,
    DEFAULT_PAGE_TIMEOUT_SECONDS,
    DEFAULT_POOL_SIZE,
    DEFAULT_TOTAL_TIMEOUT_SECONDS,
)

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

    #: Task 10: enables PlateProof's link-only Google Maps feature
    #: (plateproof.serving.display.google_maps_search_link) -- a plain,
    #: no-API-key, no-billing outbound search URL, never a call to a paid
    #: Google API. There is deliberately no API-key/OAuth-client setting
    #: here: this feature needs neither, and the previous placeholders for
    #: a possible future Places/Business Profile integration were removed
    #: after a repository-wide reference check found they were unused
    #: anywhere in the codebase.
    google_integration_enabled: bool = False

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

    # --- Task 9A: document extraction core (core safety settings, ------ #
    # not deferred to Task 9B) ------------------------------------------ #
    documents_max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES
    documents_max_pages: int = DEFAULT_MAX_PAGES
    documents_max_pixels_per_page: int = DEFAULT_MAX_PIXELS_PER_PAGE
    documents_max_text_bytes: int = DEFAULT_MAX_TEXT_BYTES_PER_DOCUMENT
    documents_worker_pool_size: int = DEFAULT_POOL_SIZE
    documents_worker_page_timeout_seconds: float = DEFAULT_PAGE_TIMEOUT_SECONDS
    documents_worker_total_timeout_seconds: float = DEFAULT_TOTAL_TIMEOUT_SECONDS
    documents_worker_kill_grace_seconds: float = DEFAULT_KILL_GRACE_SECONDS
    documents_max_total_preview_bytes: int = DEFAULT_MAX_TOTAL_PREVIEW_BYTES
    documents_worker_admission_timeout_seconds: float = DEFAULT_ADMISSION_TIMEOUT_SECONDS

    @field_validator("documents_max_upload_bytes")
    @classmethod
    def _validate_documents_max_upload_bytes(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("documents_max_upload_bytes must be positive")
        if value > ABSOLUTE_MAX_UPLOAD_BYTES:
            raise ValueError(
                f"documents_max_upload_bytes must not exceed {ABSOLUTE_MAX_UPLOAD_BYTES}"
            )
        return value

    @field_validator("documents_max_pages")
    @classmethod
    def _validate_documents_max_pages(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("documents_max_pages must be positive")
        if value > ABSOLUTE_MAX_PAGES:
            raise ValueError(f"documents_max_pages must not exceed {ABSOLUTE_MAX_PAGES}")
        return value

    @field_validator("documents_max_pixels_per_page")
    @classmethod
    def _validate_documents_max_pixels_per_page(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("documents_max_pixels_per_page must be positive")
        if value > ABSOLUTE_MAX_PIXELS_PER_PAGE:
            raise ValueError(
                f"documents_max_pixels_per_page must not exceed {ABSOLUTE_MAX_PIXELS_PER_PAGE}"
            )
        return value

    @field_validator("documents_max_text_bytes")
    @classmethod
    def _validate_documents_max_text_bytes(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("documents_max_text_bytes must be positive")
        if value > ABSOLUTE_MAX_TEXT_BYTES_PER_DOCUMENT:
            raise ValueError(
                f"documents_max_text_bytes must not exceed {ABSOLUTE_MAX_TEXT_BYTES_PER_DOCUMENT}"
            )
        return value

    @field_validator("documents_worker_pool_size")
    @classmethod
    def _validate_documents_worker_pool_size(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("documents_worker_pool_size must be positive")
        if value > ABSOLUTE_MAX_POOL_SIZE:
            raise ValueError(f"documents_worker_pool_size must not exceed {ABSOLUTE_MAX_POOL_SIZE}")
        return value

    @field_validator("documents_worker_page_timeout_seconds")
    @classmethod
    def _validate_documents_worker_page_timeout_seconds(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("documents_worker_page_timeout_seconds must be positive")
        if value > ABSOLUTE_MAX_PAGE_TIMEOUT_SECONDS:
            raise ValueError(
                "documents_worker_page_timeout_seconds must not exceed "
                f"{ABSOLUTE_MAX_PAGE_TIMEOUT_SECONDS}"
            )
        return value

    @field_validator("documents_worker_total_timeout_seconds")
    @classmethod
    def _validate_documents_worker_total_timeout_seconds(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("documents_worker_total_timeout_seconds must be positive")
        if value > ABSOLUTE_MAX_TOTAL_TIMEOUT_SECONDS:
            raise ValueError(
                "documents_worker_total_timeout_seconds must not exceed "
                f"{ABSOLUTE_MAX_TOTAL_TIMEOUT_SECONDS}"
            )
        return value

    @field_validator("documents_worker_kill_grace_seconds")
    @classmethod
    def _validate_documents_worker_kill_grace_seconds(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("documents_worker_kill_grace_seconds must be positive")
        if value > ABSOLUTE_MAX_KILL_GRACE_SECONDS:
            raise ValueError(
                "documents_worker_kill_grace_seconds must not exceed "
                f"{ABSOLUTE_MAX_KILL_GRACE_SECONDS}"
            )
        return value

    @field_validator("documents_max_total_preview_bytes")
    @classmethod
    def _validate_documents_max_total_preview_bytes(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("documents_max_total_preview_bytes must be positive")
        if value > ABSOLUTE_MAX_TOTAL_PREVIEW_BYTES:
            raise ValueError(
                "documents_max_total_preview_bytes must not exceed "
                f"{ABSOLUTE_MAX_TOTAL_PREVIEW_BYTES}"
            )
        return value

    @field_validator("documents_worker_admission_timeout_seconds")
    @classmethod
    def _validate_documents_worker_admission_timeout_seconds(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("documents_worker_admission_timeout_seconds must be positive")
        if value > ABSOLUTE_MAX_ADMISSION_TIMEOUT_SECONDS:
            raise ValueError(
                "documents_worker_admission_timeout_seconds must not exceed "
                f"{ABSOLUTE_MAX_ADMISSION_TIMEOUT_SECONDS}"
            )
        return value

    @model_validator(mode="after")
    def _validate_documents_total_timeout_not_less_than_page_timeout(self) -> "Settings":
        if self.documents_worker_total_timeout_seconds < self.documents_worker_page_timeout_seconds:
            raise ValueError(
                "documents_worker_total_timeout_seconds must be at least "
                "documents_worker_page_timeout_seconds"
            )
        return self

    @field_validator("copilot_max_question_length")
    @classmethod
    def _validate_copilot_max_question_length(cls, value: int) -> int:
        """A core, always-relevant Copilot parameter (not an optional
        local-AI setting) -- an invalid value here fails Settings
        construction loudly and early, rather than letting a nonsensical
        bound silently reach the API/service/UI. Must stay within the
        fixed absolute public-safety ceiling; see
        ``plateproof.copilot.question_validation.ABSOLUTE_MAX_QUESTION_LENGTH``."""
        if value <= 0:
            raise ValueError("copilot_max_question_length must be positive")
        if value > ABSOLUTE_MAX_QUESTION_LENGTH:
            raise ValueError(
                f"copilot_max_question_length must not exceed {ABSOLUTE_MAX_QUESTION_LENGTH}"
            )
        return value

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
