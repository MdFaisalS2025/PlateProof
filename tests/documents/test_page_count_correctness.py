"""Finding 4: service.py constructed UploadMetadata(page_count=0) before
handling a successful response and never replaced it -- successful drafts
reported zero pages even when pages were actually processed."""

from __future__ import annotations

from typing import Any


def _multi_page_echo_worker(conn: Any) -> None:
    from plateproof.documents.worker.protocol import (
        ProtocolViolationError,
        recv_bytes_frame,
        recv_frame,
        send_frame,
    )

    while True:
        try:
            recv_frame(conn)
            recv_bytes_frame(conn, max_length=64_000_000)
        except ProtocolViolationError:
            return
        send_frame(
            conn,
            {
                "protocol_version": 1,
                "message_type": "job_response",
                "ocr_available": False,
                "pages": [
                    {
                        "page_number": i + 1,
                        "width_px": 10,
                        "height_px": 10,
                        "used_ocr": False,
                        "ocr_attempted": False,
                        "text_blocks": [],
                    }
                    for i in range(3)
                ],
            },
        )


def _empty_pages_worker(conn: Any) -> None:
    from plateproof.documents.worker.protocol import (
        ProtocolViolationError,
        recv_bytes_frame,
        recv_frame,
        send_frame,
    )

    while True:
        try:
            recv_frame(conn)
            recv_bytes_frame(conn, max_length=64_000_000)
        except ProtocolViolationError:
            return
        send_frame(
            conn,
            {
                "protocol_version": 1,
                "message_type": "job_response",
                "ocr_available": False,
                "pages": [],
            },
        )


def _minimal_pdf_bytes() -> bytes:
    return b"%PDF-1.4\n%rest of a pdf..."


def _timeout_worker(conn: Any) -> None:
    import time

    time.sleep(600)


def test_successful_draft_page_count_matches_worker_response() -> None:
    from plateproof.documents.service import extract_document
    from plateproof.documents.worker.pool import WorkerPool, WorkerPoolConfig

    pool = WorkerPool(config=WorkerPoolConfig(pool_size=1), worker_main=_multi_page_echo_worker)
    try:
        draft = extract_document(
            _minimal_pdf_bytes(),
            expected_jurisdiction="nyc",
            restaurant_id="nyc:1",
            expected_restaurant_name="Joe's Pizza",
            pool=pool,
        )
        assert draft is not None
        assert draft.upload.page_count == 3
        assert len(draft.pages) == 3
    finally:
        pool.shutdown()


def test_page_numbering_is_contiguous_and_starts_at_one() -> None:
    from plateproof.documents.service import extract_document
    from plateproof.documents.worker.pool import WorkerPool, WorkerPoolConfig

    pool = WorkerPool(config=WorkerPoolConfig(pool_size=1), worker_main=_multi_page_echo_worker)
    try:
        draft = extract_document(
            _minimal_pdf_bytes(),
            expected_jurisdiction="nyc",
            restaurant_id="nyc:1",
            expected_restaurant_name="Joe's Pizza",
            pool=pool,
        )
        assert draft is not None
        assert [p.page_number for p in draft.pages] == [1, 2, 3]
        assert draft.upload.page_count == len(draft.pages)
    finally:
        pool.shutdown()


def test_empty_successful_response_is_rejected_as_invalid_not_completed() -> None:
    """A structurally-valid job_response with zero pages is not a legitimate
    outcome -- it must be treated as invalid worker output, never silently
    converted into a completed draft with page_count=0."""
    from plateproof.documents.models import ExtractionDraft
    from plateproof.documents.service import extract_document
    from plateproof.documents.worker.pool import WorkerPool, WorkerPoolConfig

    pool = WorkerPool(config=WorkerPoolConfig(pool_size=1), worker_main=_empty_pages_worker)
    try:
        draft = extract_document(
            _minimal_pdf_bytes(),
            expected_jurisdiction="nyc",
            restaurant_id="nyc:1",
            expected_restaurant_name="Joe's Pizza",
            pool=pool,
        )
        assert isinstance(draft, ExtractionDraft)
        assert draft.processing_status != "completed"
        assert draft.confirmable is False
    finally:
        pool.shutdown()


def test_failed_outcome_page_count_is_zero_because_actually_unknown() -> None:
    from plateproof.documents.service import extract_document
    from plateproof.documents.worker.pool import WorkerPool, WorkerPoolConfig

    pool = WorkerPool(
        config=WorkerPoolConfig(pool_size=1, total_timeout_seconds=1.0, page_timeout_seconds=1.0),
        worker_main=_timeout_worker,
    )
    try:
        draft = extract_document(
            _minimal_pdf_bytes(),
            expected_jurisdiction="nyc",
            restaurant_id="nyc:1",
            expected_restaurant_name="Joe's Pizza",
            pool=pool,
        )
        assert draft is not None
        assert draft.upload.page_count == 0
        assert draft.pages == ()
    finally:
        pool.shutdown()
