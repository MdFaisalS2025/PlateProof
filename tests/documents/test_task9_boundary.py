"""Cross-cutting boundary tests: Task 9 never touches Task 6/8, and OCR
unavailability never breaks embedded-text PDF extraction end to end."""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_FORBIDDEN_PREFIXES = (
    "plateproof.models",
    "plateproof.graph",
    "plateproof.copilot",
)


def _imported_module_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_documents_package_never_imports_task6_or_task8_modules() -> None:
    documents_dir = _REPO_ROOT / "plateproof" / "documents"
    for path in documents_dir.rglob("*.py"):
        imported = _imported_module_names(path)
        for name in imported:
            for forbidden in _FORBIDDEN_PREFIXES:
                assert not name.startswith(forbidden), f"{path} imports {name!r}"


def test_documents_package_never_imports_serving_write_paths() -> None:
    """plateproof.serving.scoring is the one module trusted to deserialize
    a model artifact -- Task 9 must never import it."""
    documents_dir = _REPO_ROOT / "plateproof" / "documents"
    for path in documents_dir.rglob("*.py"):
        imported = _imported_module_names(path)
        assert "plateproof.serving.scoring" not in imported


def test_end_to_end_embedded_text_pdf_completes_through_real_spawned_worker() -> None:
    """Real worker pool (a genuine spawned OS process), real entrypoint,
    real pdf.py -- no test doubles anywhere in this call. An embedded-text
    PDF must produce a completed, restaurant-corroborated draft without
    ever needing OCR, proving the full process-isolation redesign works
    end to end, not just in unit tests with injected doubles."""
    from plateproof.documents.service import extract_document
    from plateproof.documents.worker.pool import WorkerPool, WorkerPoolConfig

    # Avoids an apostrophe: PDF Standard Encoding maps code 0x27 to a curly
    # quote (U+2019), not ASCII U+0027 -- a PDF-encoding quirk of this
    # hand-written fixture, not a product bug. Plain ASCII sidesteps it.
    content = b"BT /F1 24 Tf 20 100 Td (Restaurant Name: Joes Pizza) Tj ET"
    pdf_bytes = (
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

    pool = WorkerPool(config=WorkerPoolConfig(pool_size=1))
    try:
        draft = extract_document(
            pdf_bytes,
            expected_jurisdiction="nyc",
            restaurant_id="nyc:1",
            expected_restaurant_name="Joes Pizza",
            pool=pool,
        )
    finally:
        pool.shutdown()

    assert draft is not None
    assert draft.processing_status == "completed"
    assert draft.candidates["restaurant_name"].value == "Joes Pizza"
    assert draft.restaurant_identity_corroborated is True
    assert draft.pages[0].used_ocr is False
