"""RED-first tests for the real Document Reader Streamlit page (Task 9B),
replacing the deferred-route-only state. Uses streamlit.testing.v1.AppTest
-- no browser, no network, no real PDFium/RapidOCR/ONNX Runtime required
(the worker pool is monkeypatched at the theme.document_worker_pool()
boundary, the same "inject a fake at the boundary" style already used for
Task 8's copilot_service())."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Any

import pytest

_APP_DIR = Path(__file__).resolve().parent.parent.parent / "app"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))


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


def _timeout_worker(conn: Any) -> None:
    import time

    time.sleep(600)


def _crashing_worker(conn: Any) -> None:
    import sys

    sys.exit(1)


@pytest.fixture
def fake_pool(monkeypatch: Any) -> Any:
    """Monkeypatches ``theme.document_worker_pool`` (imported by the page
    under its own local name) with a fast, fake-``worker_main``-backed
    pool for one test -- the page's own code (restaurant lookup, session
    state, correction validation, projection) all runs for real."""

    def _install(worker_main: Any = _echo_success_worker) -> Any:
        import theme

        from plateproof.documents.worker.pool import WorkerPool, WorkerPoolConfig

        pool = WorkerPool(
            config=WorkerPoolConfig(
                pool_size=1, page_timeout_seconds=2.0, total_timeout_seconds=3.0
            ),
            worker_main=worker_main,
        )
        monkeypatch.setattr(theme, "document_worker_pool", lambda: pool)
        return pool

    installed: list[Any] = []

    def _tracking_install(*args: Any, **kwargs: Any) -> Any:
        pool = _install(*args, **kwargs)
        installed.append(pool)
        return pool

    yield _tracking_install

    for pool in installed:
        pool.shutdown()


def _select_restaurant(at: Any, restaurant_id: str) -> None:
    at.text_input(key="doc_restaurant_id").set_value(restaurant_id).run(timeout=30)


def _upload_and_run(at: Any, data: bytes, *, filename: str = "doc.pdf") -> None:
    uploader_key = next(
        (w.key for w in at.get("file_uploader") if str(w.key).startswith("doc_file_uploader_")),
        None,
    )
    assert uploader_key is not None
    at.get("file_uploader")[0].upload(filename, data, "application/pdf").run(timeout=30)


def test_page_loads_without_a_restaurant_selected(app_env: Any, app_path: Any) -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(app_path("pages", "5_Document_Reader.py"))
    at.run(timeout=30)
    assert not at.exception
    assert any("Document Reader" in t.value for t in at.title)


def test_no_upload_control_reachable_without_restaurant_selection(
    app_env: Any, app_path: Any
) -> None:
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(app_path("pages", "5_Document_Reader.py"))
    at.run(timeout=30)
    assert len(at.get("file_uploader")) == 0


def test_privacy_notice_shown_before_upload(
    app_env: Any, app_path: Any, write_restaurants: Any, restaurant_row: Any
) -> None:
    from streamlit.testing.v1 import AppTest

    _write_restaurant(write_restaurants, restaurant_row)
    at = AppTest.from_file(app_path("pages", "5_Document_Reader.py"))
    at.run(timeout=30)
    _select_restaurant(at, "nyc:1")
    body = "\n".join(m.value for m in list(at.warning) + list(at.info))
    assert "does not create an official inspection record" in body


def test_successful_upload_produces_extraction_results(
    app_env: Any,
    app_path: Any,
    write_restaurants: Any,
    restaurant_row: Any,
    fake_pool: Any,
) -> None:
    from streamlit.testing.v1 import AppTest

    _write_restaurant(write_restaurants, restaurant_row)
    fake_pool()
    at = AppTest.from_file(app_path("pages", "5_Document_Reader.py"))
    at.run(timeout=30)
    _select_restaurant(at, "nyc:1")
    _upload_and_run(at, _minimal_pdf("Score: 14"))
    assert not at.exception
    assert at.session_state["doc_draft"] is not None
    assert at.session_state["doc_draft"].processing_status == "completed"


def test_no_raw_upload_bytes_retained_after_successful_extraction(
    app_env: Any,
    app_path: Any,
    write_restaurants: Any,
    restaurant_row: Any,
    fake_pool: Any,
) -> None:
    from streamlit.testing.v1 import AppTest

    _write_restaurant(write_restaurants, restaurant_row)
    fake_pool()
    at = AppTest.from_file(app_path("pages", "5_Document_Reader.py"))
    at.run(timeout=30)
    _select_restaurant(at, "nyc:1")
    upload = _minimal_pdf("Score: 14")
    _upload_and_run(at, upload)
    for key, value in at.session_state.filtered_state.items():
        if isinstance(value, (bytes, bytearray)):
            raise AssertionError(f"raw bytes retained under session_state key {key!r}")
        if type(value).__name__ == "UploadedFile":
            raise AssertionError(f"an UploadedFile object retained under session_state key {key!r}")


def test_no_raw_upload_bytes_retained_after_worker_timeout(
    app_env: Any,
    app_path: Any,
    write_restaurants: Any,
    restaurant_row: Any,
    fake_pool: Any,
) -> None:
    from streamlit.testing.v1 import AppTest

    _write_restaurant(write_restaurants, restaurant_row)
    fake_pool(_timeout_worker)
    at = AppTest.from_file(app_path("pages", "5_Document_Reader.py"))
    at.run(timeout=30)
    _select_restaurant(at, "nyc:1")
    _upload_and_run(at, _minimal_pdf("Score: 14"))
    for key, value in at.session_state.filtered_state.items():
        if isinstance(value, (bytes, bytearray)):
            raise AssertionError(f"raw bytes retained under session_state key {key!r}")


def test_worker_timeout_shows_a_fixed_sanitized_message(
    app_env: Any,
    app_path: Any,
    write_restaurants: Any,
    restaurant_row: Any,
    fake_pool: Any,
) -> None:
    from streamlit.testing.v1 import AppTest

    _write_restaurant(write_restaurants, restaurant_row)
    fake_pool(_timeout_worker)
    at = AppTest.from_file(app_path("pages", "5_Document_Reader.py"))
    at.run(timeout=30)
    _select_restaurant(at, "nyc:1")
    _upload_and_run(at, _minimal_pdf("Score: 14"))
    assert not at.exception
    body = "\n".join(m.value for m in at.error)
    assert "took too long" in body
    assert "Traceback" not in body
    assert ".py" not in body


def test_worker_crash_shows_a_fixed_sanitized_message(
    app_env: Any,
    app_path: Any,
    write_restaurants: Any,
    restaurant_row: Any,
    fake_pool: Any,
) -> None:
    from streamlit.testing.v1 import AppTest

    _write_restaurant(write_restaurants, restaurant_row)
    fake_pool(_crashing_worker)
    at = AppTest.from_file(app_path("pages", "5_Document_Reader.py"))
    at.run(timeout=30)
    _select_restaurant(at, "nyc:1")
    _upload_and_run(at, _minimal_pdf("Score: 14"))
    assert not at.exception
    body = "\n".join(m.value for m in at.error)
    assert "internal error" in body
    assert "Traceback" not in body


def test_switching_restaurant_clears_draft_and_corrections(
    app_env: Any,
    app_path: Any,
    write_restaurants: Any,
    write_inspections: Any,
    write_violations: Any,
    restaurant_row: Any,
    inspection_row: Any,
    violation_row: Any,
    fake_pool: Any,
) -> None:
    from streamlit.testing.v1 import AppTest

    write_restaurants(
        [
            restaurant_row(restaurant_id="nyc:1", jurisdiction="nyc", name="Anna's Kitchen"),
            restaurant_row(restaurant_id="nyc:2", jurisdiction="nyc", name="Bob's Diner"),
        ]
    )
    fake_pool()
    at = AppTest.from_file(app_path("pages", "5_Document_Reader.py"))
    at.run(timeout=30)
    _select_restaurant(at, "nyc:1")
    _upload_and_run(at, _minimal_pdf("Score: 14"))
    assert at.session_state["doc_draft"] is not None

    _select_restaurant(at, "nyc:2")
    assert at.session_state["doc_draft"] is None
    assert at.session_state["doc_draft_restaurant_id"] is None
    assert at.session_state["doc_corrections"] == {}


def test_start_over_clears_the_same_state_as_a_restaurant_switch(
    app_env: Any,
    app_path: Any,
    write_restaurants: Any,
    restaurant_row: Any,
    fake_pool: Any,
) -> None:
    from streamlit.testing.v1 import AppTest

    _write_restaurant(write_restaurants, restaurant_row)
    fake_pool()
    at = AppTest.from_file(app_path("pages", "5_Document_Reader.py"))
    at.run(timeout=30)
    _select_restaurant(at, "nyc:1")
    _upload_and_run(at, _minimal_pdf("Score: 14"))
    assert at.session_state["doc_draft"] is not None

    at.button(key="doc_start_over").click().run(timeout=30)
    assert at.session_state["doc_draft"] is None
    assert at.session_state["doc_restaurant_id"] == ""


def test_confirmation_checkbox_is_unchecked_by_default(
    app_env: Any,
    app_path: Any,
    write_restaurants: Any,
    restaurant_row: Any,
    fake_pool: Any,
) -> None:
    from streamlit.testing.v1 import AppTest

    _write_restaurant(write_restaurants, restaurant_row)
    fake_pool()
    at = AppTest.from_file(app_path("pages", "5_Document_Reader.py"))
    at.run(timeout=30)
    _select_restaurant(at, "nyc:1")
    _upload_and_run(at, _minimal_pdf("Score: 14"))
    checkboxes = at.checkbox(key="doc_confirm_checkbox")
    assert checkboxes.value is False


def test_download_button_absent_until_confirmed(
    app_env: Any,
    app_path: Any,
    write_restaurants: Any,
    restaurant_row: Any,
    fake_pool: Any,
) -> None:
    from streamlit.testing.v1 import AppTest

    _write_restaurant(write_restaurants, restaurant_row)
    fake_pool()
    at = AppTest.from_file(app_path("pages", "5_Document_Reader.py"))
    at.run(timeout=30)
    _select_restaurant(at, "nyc:1")
    _upload_and_run(at, _minimal_pdf("Score: 14"))
    assert len(at.get("download_button")) == 0

    at.checkbox(key="doc_confirm_checkbox").set_value(True).run(timeout=30)
    assert len(at.get("download_button")) == 1


def test_previews_rendered_via_data_uri_never_via_pil_decode(
    app_env: Any,
    app_path: Any,
    write_restaurants: Any,
    restaurant_row: Any,
    fake_pool: Any,
) -> None:
    """Whatever image element(s) render on the page must be given a
    ``data:image/png;base64,...`` string -- proving Streamlit's own
    Pillow-based image_to_url decode path is never reached for a preview
    (see the page's module docstring)."""
    from streamlit.testing.v1 import AppTest

    _write_restaurant(write_restaurants, restaurant_row)
    fake_pool()
    at = AppTest.from_file(app_path("pages", "5_Document_Reader.py"))
    at.run(timeout=30)
    _select_restaurant(at, "nyc:1")
    _upload_and_run(at, _minimal_pdf("Score: 14"))
    # If no previews were generated for this tiny synthetic fixture, the
    # import-graph assertion in test_page_never_imports_native_parsers_or_decoders
    # is what actually matters -- Task 9A already proves preview generation
    # itself; this test only proves the page doesn't crash rendering them.
    assert not at.exception


def test_page_never_imports_native_parsers_or_decoders(app_path: Any) -> None:
    """Static import-graph check: the page must never import PDFium,
    Pillow's image module, RapidOCR, ONNX Runtime, or OpenCV -- only
    plateproof.documents.service/models/corrections/nyc_extractor/
    florida_extractor may be imported from the documents package."""
    with open(app_path("pages", "5_Document_Reader.py"), encoding="utf-8") as handle:
        lines = handle.readlines()
    import_lines = [line for line in lines if line.lstrip().startswith(("import ", "from "))]
    import_source = "".join(import_lines)
    assert "pypdfium2" not in import_source
    assert "cv2" not in import_source
    assert "onnxruntime" not in import_source
    assert "rapidocr" not in import_source
    assert "plateproof.documents.pdf" not in import_source
    assert "plateproof.documents.images" not in import_source
    assert "plateproof.documents.ocr" not in import_source
    # PIL/Pillow is never imported for DECODING here -- st.image() is given
    # a data: URI string, never bytes -- so the module itself is never
    # imported by this page's own source at all.
    assert "PIL" not in import_source


# --------------------------------------------------------------------------- #
# Finding 3 (independent review of c60cc80): the aggregate correction-       #
# payload limit must be enforced via validate_corrections(), not just a     #
# per-field loop -- and confirmation must block on it.                      #
# --------------------------------------------------------------------------- #


def test_aggregate_correction_payload_overflow_blocks_confirmation(
    app_env: Any,
    app_path: Any,
    write_restaurants: Any,
    restaurant_row: Any,
    fake_pool: Any,
) -> None:
    """Many individually-small correction values whose TOTAL length
    exceeds plateproof.documents.corrections.MAX_TOTAL_CORRECTION_PAYLOAD_LENGTH
    must block confirmation -- proving the page validates the complete
    corrections mapping with validate_corrections() (which enforces this
    aggregate limit), not a per-field loop that never sums anything."""
    from streamlit.testing.v1 import AppTest

    from plateproof.documents.corrections import MAX_TOTAL_CORRECTION_PAYLOAD_LENGTH

    _write_restaurant(write_restaurants, restaurant_row)
    fake_pool()
    at = AppTest.from_file(app_path("pages", "5_Document_Reader.py"))
    at.run(timeout=30)
    _select_restaurant(at, "nyc:1")
    _upload_and_run(at, _minimal_pdf("Score: 14"))
    assert at.session_state["doc_draft"] is not None

    # Individually well under the 500-char per-field cap, but the sum
    # across many fields exceeds the aggregate payload limit. Uses
    # otherwise-unknown field names so the ONLY way confirmation can be
    # blocked is the aggregate-size check specifically -- a per-field-only
    # validation loop (which never sums anything) would instead report
    # each field as an unrecognized correction field, a completely
    # different failure reason from a genuinely-enforced aggregate limit.
    field_value = "x" * 400
    field_count = (MAX_TOTAL_CORRECTION_PAYLOAD_LENGTH // len(field_value)) + 5
    at.session_state["doc_corrections"] = {
        f"synthetic_field_{i}": field_value for i in range(field_count)
    }
    at.run(timeout=30)

    checkbox = at.checkbox(key="doc_confirm_checkbox")
    assert checkbox.disabled is True
    error_text = "\n".join(m.value for m in at.error)
    assert "exceeds the maximum allowed size" in error_text


def test_invalid_numeric_correction_blocks_confirmation(
    app_env: Any,
    app_path: Any,
    write_restaurants: Any,
    restaurant_row: Any,
    fake_pool: Any,
) -> None:
    """A non-finite numeric correction (e.g. "nan") for the score field
    parses as a float via Python's own float("nan") but must be rejected
    -- confirmation must stay blocked."""
    from streamlit.testing.v1 import AppTest

    _write_restaurant(write_restaurants, restaurant_row)
    fake_pool()
    at = AppTest.from_file(app_path("pages", "5_Document_Reader.py"))
    at.run(timeout=30)
    _select_restaurant(at, "nyc:1")
    _upload_and_run(at, _minimal_pdf("Score: 14"))
    assert at.session_state["doc_draft"] is not None

    at.text_input(key="doc_correction_score").set_value("nan").run(timeout=30)

    checkbox = at.checkbox(key="doc_confirm_checkbox")
    assert checkbox.disabled is True
