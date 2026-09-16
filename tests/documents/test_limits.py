"""Tests for the centralized, fixed absolute ceilings module. Separate
JSON/document/preview frame limits (Finding 3) and the worker-config
validation ceilings (Finding 6) both depend on these constants being
distinct and internally consistent."""

from __future__ import annotations


def test_document_frame_ceiling_is_larger_than_json_frame_ceiling() -> None:
    from plateproof.documents.limits import ABSOLUTE_MAX_UPLOAD_BYTES, MAX_JSON_FRAME_BYTES

    assert ABSOLUTE_MAX_UPLOAD_BYTES > MAX_JSON_FRAME_BYTES


def test_default_upload_bytes_does_not_exceed_absolute_ceiling() -> None:
    from plateproof.documents.limits import ABSOLUTE_MAX_UPLOAD_BYTES, DEFAULT_MAX_UPLOAD_BYTES

    assert DEFAULT_MAX_UPLOAD_BYTES <= ABSOLUTE_MAX_UPLOAD_BYTES


def test_preview_frame_ceiling_is_smaller_than_document_frame_ceiling() -> None:
    from plateproof.documents.limits import (
        ABSOLUTE_MAX_PREVIEW_FRAME_BYTES,
        ABSOLUTE_MAX_UPLOAD_BYTES,
    )

    assert ABSOLUTE_MAX_PREVIEW_FRAME_BYTES < ABSOLUTE_MAX_UPLOAD_BYTES


def test_pool_size_ceiling_and_default_are_consistent() -> None:
    from plateproof.documents.limits import ABSOLUTE_MAX_POOL_SIZE, DEFAULT_POOL_SIZE, MIN_POOL_SIZE

    assert MIN_POOL_SIZE >= 1
    assert MIN_POOL_SIZE <= DEFAULT_POOL_SIZE <= ABSOLUTE_MAX_POOL_SIZE


def test_total_timeout_default_is_not_smaller_than_page_timeout_default() -> None:
    from plateproof.documents.limits import (
        DEFAULT_PAGE_TIMEOUT_SECONDS,
        DEFAULT_TOTAL_TIMEOUT_SECONDS,
    )

    assert DEFAULT_TOTAL_TIMEOUT_SECONDS >= DEFAULT_PAGE_TIMEOUT_SECONDS
