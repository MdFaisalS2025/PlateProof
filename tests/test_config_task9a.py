"""RED-first tests for Task 9A's new Settings fields (independent-review
correction, Finding 6): core document-processing safety settings are added
to the typed configuration now, not deferred to Task 9B. Every bounded
numeric setting must fail loudly and early at Settings construction if it
exceeds its fixed absolute ceiling -- these are core, always-relevant
safety parameters, not optional/degrade-gracefully settings."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from plateproof.core.config import Settings
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
    DEFAULT_MAX_PAGES,
    DEFAULT_MAX_PIXELS_PER_PAGE,
    DEFAULT_MAX_TEXT_BYTES_PER_DOCUMENT,
    DEFAULT_MAX_TOTAL_PREVIEW_BYTES,
    DEFAULT_MAX_UPLOAD_BYTES,
    DEFAULT_PAGE_TIMEOUT_SECONDS,
    DEFAULT_POOL_SIZE,
    DEFAULT_TOTAL_TIMEOUT_SECONDS,
)


def test_task_9a_settings_have_safe_defaults() -> None:
    settings = Settings(_env_file=None)
    assert settings.documents_max_upload_bytes == DEFAULT_MAX_UPLOAD_BYTES
    assert settings.documents_max_pages == DEFAULT_MAX_PAGES
    assert settings.documents_max_pixels_per_page == DEFAULT_MAX_PIXELS_PER_PAGE
    assert settings.documents_max_text_bytes == DEFAULT_MAX_TEXT_BYTES_PER_DOCUMENT
    assert settings.documents_worker_pool_size == DEFAULT_POOL_SIZE
    assert settings.documents_worker_page_timeout_seconds == DEFAULT_PAGE_TIMEOUT_SECONDS
    assert settings.documents_worker_total_timeout_seconds == DEFAULT_TOTAL_TIMEOUT_SECONDS
    assert settings.documents_max_total_preview_bytes == DEFAULT_MAX_TOTAL_PREVIEW_BYTES
    assert settings.documents_worker_admission_timeout_seconds == DEFAULT_ADMISSION_TIMEOUT_SECONDS


def test_task_9a_settings_are_overridable_within_ceilings() -> None:
    settings = Settings(
        _env_file=None,
        documents_max_upload_bytes=1_000_000,
        documents_max_pages=3,
        documents_worker_pool_size=1,
    )
    assert settings.documents_max_upload_bytes == 1_000_000
    assert settings.documents_max_pages == 3
    assert settings.documents_worker_pool_size == 1


@pytest.mark.parametrize(
    "field,ceiling",
    [
        ("documents_max_upload_bytes", ABSOLUTE_MAX_UPLOAD_BYTES),
        ("documents_max_pages", ABSOLUTE_MAX_PAGES),
        ("documents_max_pixels_per_page", ABSOLUTE_MAX_PIXELS_PER_PAGE),
        ("documents_max_text_bytes", ABSOLUTE_MAX_TEXT_BYTES_PER_DOCUMENT),
        ("documents_worker_pool_size", ABSOLUTE_MAX_POOL_SIZE),
        ("documents_worker_page_timeout_seconds", ABSOLUTE_MAX_PAGE_TIMEOUT_SECONDS),
        ("documents_worker_total_timeout_seconds", ABSOLUTE_MAX_TOTAL_TIMEOUT_SECONDS),
        ("documents_worker_kill_grace_seconds", ABSOLUTE_MAX_KILL_GRACE_SECONDS),
        ("documents_max_total_preview_bytes", ABSOLUTE_MAX_TOTAL_PREVIEW_BYTES),
        ("documents_worker_admission_timeout_seconds", ABSOLUTE_MAX_ADMISSION_TIMEOUT_SECONDS),
    ],
)
def test_setting_above_its_absolute_ceiling_is_rejected(field: str, ceiling: float) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: ceiling + 1})


@pytest.mark.parametrize(
    "field",
    [
        "documents_max_upload_bytes",
        "documents_max_pages",
        "documents_max_pixels_per_page",
        "documents_max_text_bytes",
        "documents_worker_pool_size",
        "documents_worker_page_timeout_seconds",
        "documents_worker_total_timeout_seconds",
        "documents_worker_kill_grace_seconds",
        "documents_max_total_preview_bytes",
        "documents_worker_admission_timeout_seconds",
    ],
)
def test_setting_zero_or_negative_is_rejected(field: str) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: 0})
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: -1})


def test_total_timeout_less_than_page_timeout_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            documents_worker_page_timeout_seconds=30.0,
            documents_worker_total_timeout_seconds=10.0,
        )
