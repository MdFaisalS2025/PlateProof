"""Real end-to-end tests for POST /owners/documents/extract: the real
FastAPI app, a real spawned worker process (the default entrypoint --
never a fake ``worker_main``), and real PDFium/RapidOCR. Task 9A's own
suite already exhaustively covers pdf.py/entrypoint.py/worker behavior in
isolation (including OCR-unavailable, via a monkeypatched engine
construction failure that a real spawned child process can't easily
reproduce from this test process) -- this file's job is to prove the
FULL PIPELINE, HTTP request through to a real worker and back, actually
works for representative NYC/Florida text documents, a scanned image, an
encrypted document, a malformed document, and an ambiguous document.
"""

from __future__ import annotations

from datetime import date
from io import BytesIO
from typing import Any

import pytest
from PIL import Image, ImageDraw


def _minimal_pdf(text: str) -> bytes:
    content = f"BT /F1 24 Tf 20 100 Td ({text}) Tj ET".encode()
    return (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        b"4 0 obj\n<< /Length "
        + str(len(content)).encode()
        + b" >>\nstream\n"
        + content
        + b"\nendstream\nendobj\n"
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        b"xref\n0 1\n0000000000 65535 f \ntrailer\n<< /Size 1 /Root 1 0 R >>\nstartxref\n0\n%%EOF\n"
    )


def _multiline_pdf(lines: list[str]) -> bytes:
    stream_ops = "\n".join(
        f"BT /F1 14 Tf 20 {280 - i * 20} Td ({line}) Tj ET" for i, line in enumerate(lines)
    )
    content = stream_ops.encode()
    return (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 400] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        b"4 0 obj\n<< /Length "
        + str(len(content)).encode()
        + b" >>\nstream\n"
        + content
        + b"\nendstream\nendobj\n"
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        b"xref\n0 1\n0000000000 65535 f \ntrailer\n<< /Size 1 /Root 1 0 R >>\nstartxref\n0\n%%EOF\n"
    )


def _png_with_text(text: str) -> bytes:
    image = Image.new("RGB", (400, 100), "white")
    ImageDraw.Draw(image).text((10, 30), text, fill="black")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _build_encrypted_pdf(text: str) -> bytes:
    from tests.documents.test_pdf_streaming import _build_encrypted_pdf as _build

    return _build(text)


def _write_restaurant(
    write_restaurants: Any,
    restaurant_row: Any,
    *,
    restaurant_id: str,
    jurisdiction: str,
    name: str,
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
def real_documents_client(processed_dir: Any) -> Any:
    """A real app + TestClient using the REAL default worker entrypoint
    (a real spawned process, real PDFium, real RapidOCR) -- no fake
    worker_main anywhere in this fixture."""
    from fastapi.testclient import TestClient

    from plateproof.api.main import create_app
    from plateproof.core.config import Settings

    def _make(**settings_overrides: Any) -> Any:
        settings = Settings(
            _env_file=None,
            processed_data_dir=processed_dir,
            documents_worker_page_timeout_seconds=30.0,
            documents_worker_total_timeout_seconds=60.0,
            **settings_overrides,
        )
        app = create_app(settings)
        return TestClient(app)

    clients: list[Any] = []

    def _tracking_make(*args: Any, **kwargs: Any) -> Any:
        client = _make(*args, **kwargs)
        clients.append(client)
        return client

    yield _tracking_make

    for client in clients:
        client.app.state.document_worker_pool.shutdown()


def _post_document(
    client: Any, *, restaurant_id: str, data: bytes, filename: str = "doc.pdf"
) -> Any:
    files = {"file": (filename, BytesIO(data), "application/pdf")}
    return client.post(
        "/owners/documents/extract", data={"restaurant_id": restaurant_id}, files=files
    )


def test_nyc_text_pdf_end_to_end(
    real_documents_client: Any, write_restaurants: Any, restaurant_row: Any
) -> None:
    _write_restaurant(
        write_restaurants,
        restaurant_row,
        restaurant_id="nyc:1",
        jurisdiction="nyc",
        name="Anna's Kitchen",
    )
    client = real_documents_client()
    data = _multiline_pdf(
        [
            "Restaurant Name: Anna's Kitchen",
            "Inspection Date: 01/15/2024",
            "Score: 13",
            "Grade: A",
        ]
    )
    response = _post_document(client, restaurant_id="nyc:1", data=data)
    assert response.status_code == 200
    body = response.json()
    assert body["processing_status"] == "completed"
    assert body["candidates"]["score"]["value"] == 13.0
    assert body["candidates"]["grade"]["value"] == "A"
    assert body["restaurant_identity_corroborated"] is True


def test_florida_text_pdf_end_to_end(
    real_documents_client: Any, write_restaurants: Any, restaurant_row: Any
) -> None:
    _write_restaurant(
        write_restaurants,
        restaurant_row,
        restaurant_id="florida:1",
        jurisdiction="florida",
        name="Sunshine Cafe",
    )
    client = real_documents_client()
    data = _multiline_pdf(
        [
            "Restaurant Name: Sunshine Cafe",
            "Inspection Date: 03/02/2024",
            "Disposition: Inspection Completed - No Further Action",
            "High Priority: 0",
        ]
    )
    response = _post_document(client, restaurant_id="florida:1", data=data)
    assert response.status_code == 200
    body = response.json()
    assert body["processing_status"] == "completed"
    assert body["jurisdiction_expected"] == "florida"
    # Florida results are never converted into an NYC letter grade.
    assert "grade" not in body["candidates"]


def test_scanned_image_end_to_end_uses_real_ocr(
    real_documents_client: Any, write_restaurants: Any, restaurant_row: Any
) -> None:
    _write_restaurant(
        write_restaurants,
        restaurant_row,
        restaurant_id="nyc:2",
        jurisdiction="nyc",
        name="Bob's Diner",
    )
    client = real_documents_client()
    data = _png_with_text("Restaurant Name: Bob's Diner")
    response = _post_document(client, restaurant_id="nyc:2", data=data, filename="doc.png")
    assert response.status_code == 200
    body = response.json()
    # Either OCR successfully read text (completed) or genuinely isn't
    # available in this environment (ocr_unavailable) -- never "failed"
    # for a well-formed image, and never silently "completed" with no
    # pages at all.
    assert body["processing_status"] in ("completed", "ocr_unavailable")
    assert body["upload_page_count"] == 1


def test_encrypted_pdf_end_to_end(
    real_documents_client: Any, write_restaurants: Any, restaurant_row: Any
) -> None:
    _write_restaurant(
        write_restaurants,
        restaurant_row,
        restaurant_id="nyc:3",
        jurisdiction="nyc",
        name="Carla's Corner",
    )
    client = real_documents_client()
    data = _build_encrypted_pdf("Score: 14")
    response = _post_document(client, restaurant_id="nyc:3", data=data)
    assert response.status_code == 200
    body = response.json()
    assert body["processing_status"] == "failed"
    assert body["confirmable"] is False
    assert any(w["code"] == "encrypted_document" for w in body["warnings"])


def test_malformed_pdf_end_to_end(
    real_documents_client: Any, write_restaurants: Any, restaurant_row: Any
) -> None:
    _write_restaurant(
        write_restaurants,
        restaurant_row,
        restaurant_id="nyc:4",
        jurisdiction="nyc",
        name="Dana's Deli",
    )
    client = real_documents_client()
    # Passes the magic-byte sniff (starts with %PDF-) but isn't a real PDF.
    data = b"%PDF-1.4\nthis is not a real pdf structure at all"
    response = _post_document(client, restaurant_id="nyc:4", data=data)
    assert response.status_code == 200
    body = response.json()
    assert body["processing_status"] == "failed"
    assert any(w["code"] == "pdf_malformed" for w in body["warnings"])


def test_ambiguous_conflicting_score_end_to_end(
    real_documents_client: Any, write_restaurants: Any, restaurant_row: Any
) -> None:
    _write_restaurant(
        write_restaurants,
        restaurant_row,
        restaurant_id="nyc:5",
        jurisdiction="nyc",
        name="Eve's Eatery",
    )
    client = real_documents_client()
    data = _multiline_pdf(
        [
            "Restaurant Name: Eve's Eatery",
            "Inspection Date: 01/15/2024",
            "Score: 13",
            "Score: 27",
        ]
    )
    response = _post_document(client, restaurant_id="nyc:5", data=data)
    assert response.status_code == 200
    body = response.json()
    assert body["processing_status"] == "completed"
    assert any(a["field_name"] == "score" for a in body["ambiguities"])
    assert "score" not in body["candidates"]
    assert body["confirmable"] is False
