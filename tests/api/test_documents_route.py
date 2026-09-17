"""RED-first tests for POST /owners/documents/extract -- the real Task 9B
endpoint that replaces the deferred 501. Restaurant identity/jurisdiction
are always resolved server-side; a client-supplied jurisdiction is never
even accepted as a field. Every worker outcome after a successful
``extract_document(...)`` call is a typed 200 -- never collapsed into an
HTTP error and never silently reported as completed when it failed.

Uses fake, fast ``worker_main`` callables (the same style
``tests/documents/test_service.py`` already uses) injected via FastAPI's
own dependency-override mechanism, rather than the real PDFium/RapidOCR
worker -- this file tests the ROUTE's own logic (status codes, restaurant
resolution, file/form handling, response projection), not Task 9A's
already-tested extraction behavior.
"""

from __future__ import annotations

import io
from datetime import date
from typing import Any

import pytest


def _minimal_pdf(text: str) -> bytes:
    content = f"BT /F1 24 Tf 20 100 Td ({text}) Tj ET".encode()
    return (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        b"4 0 obj\n<< /Length "
        + str(len(content)).encode()
        + b" >>\nstream\n"
        + content
        + b"\nendstream\nendobj\n"
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        b"xref\n0 1\n0000000000 65535 f \ntrailer\n<< /Size 1 /Root 1 0 R >>\nstartxref\n0\n%%EOF\n"
    )


def _echo_success_worker(conn: Any) -> None:
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
                        "page_number": 1,
                        "width_px": 200,
                        "height_px": 200,
                        "used_ocr": False,
                        "ocr_attempted": False,
                        "text_blocks": [
                            {
                                "text": (
                                    "Restaurant Name: Anna's Kitchen\n"
                                    "Inspection Date: 01/15/2024\nScore: 14"
                                ),
                                "source": "embedded_text",
                                "ocr_confidence": None,
                                "bounding_box": None,
                            }
                        ],
                    }
                ],
            },
        )


def _encrypted_document_worker(conn: Any) -> None:
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
                "message_type": "job_error",
                "error_kind": "encrypted_document",
            },
        )


def _timeout_worker(conn: Any) -> None:
    import time

    time.sleep(600)


def _write_restaurant(
    write_restaurants: Any,
    restaurant_row: Any,
    *,
    restaurant_id: str = "nyc:1",
    jurisdiction: str = "nyc",
    name: str = "Anna's Kitchen",
) -> None:
    write_restaurants(
        [
            restaurant_row(
                restaurant_id=restaurant_id,
                jurisdiction=jurisdiction,
                name=name,
                latest_inspection_date=date(2025, 6, 1),
            )
        ]
    )


@pytest.fixture
def documents_app(processed_dir: Any) -> Any:
    """Builds a real app + TestClient, overriding only the document worker
    pool dependency with a fast, fake-``worker_main``-backed pool -- the
    route's own code (dependency wiring, restaurant lookup, form/file
    handling, projection) all run for real."""
    from fastapi.testclient import TestClient

    from plateproof.api.dependencies import get_document_worker_pool
    from plateproof.api.main import create_app
    from plateproof.core.config import Settings
    from plateproof.documents.worker.pool import WorkerPool, WorkerPoolConfig

    def _make(worker_main: Any = _echo_success_worker, **settings_overrides: Any) -> Any:
        settings = Settings(_env_file=None, processed_data_dir=processed_dir, **settings_overrides)
        app = create_app(settings)
        pool = WorkerPool(
            config=WorkerPoolConfig(
                pool_size=1, page_timeout_seconds=20.0, total_timeout_seconds=40.0
            ),
            worker_main=worker_main,
        )
        app.dependency_overrides[get_document_worker_pool] = lambda: pool
        client = TestClient(app)
        client.__pool_to_shutdown = pool  # type: ignore[attr-defined]
        return client

    made: list[Any] = []

    def _tracking_make(*args: Any, **kwargs: Any) -> Any:
        client = _make(*args, **kwargs)
        made.append(client)
        return client

    yield _tracking_make

    for client in made:
        client.__pool_to_shutdown.shutdown()  # type: ignore[attr-defined]
        client.app.state.document_worker_pool.shutdown()


def _post_document(
    client: Any, *, restaurant_id: str | None = "nyc:1", filename: str = "doc.pdf", data: bytes
) -> Any:
    files = {"file": (filename, io.BytesIO(data), "application/pdf")}
    form_data = {} if restaurant_id is None else {"restaurant_id": restaurant_id}
    return client.post("/owners/documents/extract", data=form_data, files=files)


def test_successful_extraction_returns_completed_draft(
    documents_app: Any, write_restaurants: Any, restaurant_row: Any
) -> None:
    _write_restaurant(write_restaurants, restaurant_row)
    client = documents_app()
    response = _post_document(client, data=_minimal_pdf("Score: 14"))
    assert response.status_code == 200
    body = response.json()
    assert body["processing_status"] == "completed"
    assert body["jurisdiction_expected"] == "nyc"
    assert body["restaurant_id"] == "nyc:1"
    assert body["candidates"]["score"]["value"] == 14.0


def test_jurisdiction_is_never_accepted_from_the_client(
    documents_app: Any, write_restaurants: Any, restaurant_row: Any
) -> None:
    """Even if a client sends a jurisdiction field, it has no effect --
    the route only ever reads restaurant_id and file from the form."""
    _write_restaurant(write_restaurants, restaurant_row, jurisdiction="nyc")
    client = documents_app()
    files = {"file": ("doc.pdf", io.BytesIO(_minimal_pdf("Score: 14")), "application/pdf")}
    response = client.post(
        "/owners/documents/extract",
        data={"restaurant_id": "nyc:1", "jurisdiction": "florida"},
        files=files,
    )
    assert response.status_code == 200
    assert response.json()["jurisdiction_expected"] == "nyc"


def test_unknown_restaurant_returns_404(documents_app: Any) -> None:
    client = documents_app()
    response = _post_document(client, restaurant_id="nyc:does-not-exist", data=_minimal_pdf("x"))
    assert response.status_code == 404
    assert "does-not-exist" not in response.text or "traceback" not in response.text.lower()


def test_missing_restaurant_id_returns_422(documents_app: Any) -> None:
    client = documents_app()
    response = _post_document(client, restaurant_id=None, data=_minimal_pdf("x"))
    assert response.status_code == 422


def test_missing_file_returns_422(
    documents_app: Any, write_restaurants: Any, restaurant_row: Any
) -> None:
    _write_restaurant(write_restaurants, restaurant_row)
    client = documents_app()
    response = client.post("/owners/documents/extract", data={"restaurant_id": "nyc:1"})
    assert response.status_code == 422


def test_two_file_parts_rejected_with_413(
    documents_app: Any, write_restaurants: Any, restaurant_row: Any
) -> None:
    _write_restaurant(write_restaurants, restaurant_row)
    client = documents_app()
    response = client.post(
        "/owners/documents/extract",
        data={"restaurant_id": "nyc:1"},
        files=[
            ("file", ("a.pdf", io.BytesIO(_minimal_pdf("a")), "application/pdf")),
            ("extra", ("b.pdf", io.BytesIO(_minimal_pdf("b")), "application/pdf")),
        ],
    )
    assert response.status_code == 413


def test_oversized_restaurant_id_rejected_with_413(
    documents_app: Any, write_restaurants: Any, restaurant_row: Any
) -> None:
    _write_restaurant(write_restaurants, restaurant_row)
    client = documents_app()
    response = _post_document(
        client, restaurant_id="nyc:" + "x" * 200, data=_minimal_pdf("Score: 14")
    )
    assert response.status_code == 413


def test_unsupported_media_type_rejected_with_415(
    documents_app: Any, write_restaurants: Any, restaurant_row: Any
) -> None:
    _write_restaurant(write_restaurants, restaurant_row)
    client = documents_app()
    response = _post_document(client, data=b"not a real document at all")
    assert response.status_code == 415


def test_oversized_upload_rejected_with_413(
    documents_app: Any, write_restaurants: Any, restaurant_row: Any
) -> None:
    _write_restaurant(write_restaurants, restaurant_row)
    client = documents_app(documents_max_upload_bytes=100)
    response = _post_document(client, data=b"%PDF-1.4\n" + b"x" * 500)
    assert response.status_code == 413


def test_worker_job_error_is_a_typed_200_never_a_completed_draft(
    documents_app: Any, write_restaurants: Any, restaurant_row: Any
) -> None:
    """A worker-reported processing failure (encrypted document) is never
    turned into a completed draft -- it stays a typed 200 with
    processing_status="failed" and the specific warning code."""
    _write_restaurant(write_restaurants, restaurant_row)
    client = documents_app(worker_main=_encrypted_document_worker)
    response = _post_document(client, data=_minimal_pdf("Score: 14"))
    assert response.status_code == 200
    body = response.json()
    assert body["processing_status"] == "failed"
    assert body["confirmable"] is False
    assert any(w["code"] == "encrypted_document" for w in body["warnings"])


def test_worker_timeout_is_a_typed_200_failed_draft(
    documents_app: Any, write_restaurants: Any, restaurant_row: Any
) -> None:
    _write_restaurant(write_restaurants, restaurant_row)
    client = documents_app(
        worker_main=_timeout_worker,
        documents_worker_page_timeout_seconds=1.0,
        documents_worker_total_timeout_seconds=1.0,
    )
    response = _post_document(client, data=_minimal_pdf("Score: 14"))
    assert response.status_code == 200
    body = response.json()
    assert body["processing_status"] == "failed"
    assert any(w["code"] == "worker_timeout" for w in body["warnings"])


def test_response_never_contains_a_filesystem_path_or_exception_text(
    documents_app: Any, write_restaurants: Any, restaurant_row: Any
) -> None:
    _write_restaurant(write_restaurants, restaurant_row)
    client = documents_app(worker_main=_encrypted_document_worker)
    response = _post_document(client, data=_minimal_pdf("Score: 14"))
    text = response.text
    assert "Traceback" not in text
    assert "site-packages" not in text
    assert ".py" not in text


def test_response_projects_bounded_preview_not_raw_upload_bytes(
    documents_app: Any, write_restaurants: Any, restaurant_row: Any
) -> None:
    """Whatever preview bytes appear in the response are base64 and
    strictly smaller than the raw uploaded document -- proving they are
    the (absent, in this fake-worker case) worker-generated preview, never
    a re-encoding of the original upload."""
    _write_restaurant(write_restaurants, restaurant_row)
    client = documents_app()
    upload = _minimal_pdf("Score: 14")
    response = _post_document(client, data=upload)
    body = response.json()
    for page in body["pages"]:
        if page["preview_png_base64"] is not None:
            assert len(page["preview_png_base64"]) < len(upload)


def test_upload_file_is_closed_after_successful_request(
    documents_app: Any, write_restaurants: Any, restaurant_row: Any, monkeypatch: Any
) -> None:
    from starlette.datastructures import UploadFile

    _write_restaurant(write_restaurants, restaurant_row)
    client = documents_app()

    close_calls: list[int] = []
    original_close = UploadFile.close

    async def _tracked_close(self: Any) -> None:
        close_calls.append(1)
        await original_close(self)

    monkeypatch.setattr(UploadFile, "close", _tracked_close)
    response = _post_document(client, data=_minimal_pdf("Score: 14"))
    assert response.status_code == 200
    assert close_calls == [1]


def test_upload_file_is_closed_after_restaurant_not_found(
    documents_app: Any, monkeypatch: Any
) -> None:
    from starlette.datastructures import UploadFile

    client = documents_app()
    close_calls: list[int] = []
    original_close = UploadFile.close

    async def _tracked_close(self: Any) -> None:
        close_calls.append(1)
        await original_close(self)

    monkeypatch.setattr(UploadFile, "close", _tracked_close)
    response = _post_document(client, restaurant_id="nyc:missing", data=_minimal_pdf("x"))
    assert response.status_code == 404
    assert close_calls == [1]


def test_upload_file_is_closed_after_worker_timeout(
    documents_app: Any, write_restaurants: Any, restaurant_row: Any, monkeypatch: Any
) -> None:
    from starlette.datastructures import UploadFile

    _write_restaurant(write_restaurants, restaurant_row)
    client = documents_app(
        worker_main=_timeout_worker,
        documents_worker_page_timeout_seconds=1.0,
        documents_worker_total_timeout_seconds=1.0,
    )
    close_calls: list[int] = []
    original_close = UploadFile.close

    async def _tracked_close(self: Any) -> None:
        close_calls.append(1)
        await original_close(self)

    monkeypatch.setattr(UploadFile, "close", _tracked_close)
    response = _post_document(client, data=_minimal_pdf("Score: 14"))
    assert response.status_code == 200
    assert close_calls == [1]


def test_starlette_version_guards_against_the_known_rollover_cve(documents_app: Any) -> None:
    """Guards against a future dependency downgrade reintroducing
    CVE-2025-54121's blocking-rollover behavior."""
    import importlib.metadata

    from packaging.version import Version

    assert Version(importlib.metadata.version("starlette")) >= Version("0.47.2")


def test_uvicorn_has_no_request_body_size_option(documents_app: Any) -> None:
    """Documents (as a test, not just a comment) that Uvicorn's own CLI
    exposes no request-body-size limit in the installed version -- the
    Layer-1 gateway precondition this route's module docstring and
    README.md describe is not something Uvicorn itself can substitute
    for."""
    import uvicorn.config

    config_fields = {f for f in dir(uvicorn.config.Config) if not f.startswith("_")}
    assert not any("body" in f.lower() and "size" in f.lower() for f in config_fields)
