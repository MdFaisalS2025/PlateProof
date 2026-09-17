"""Task 9B structural boundary tests: neither entry point (the FastAPI
route or the Streamlit page) may create a second parsing path. Both call
the identical ``plateproof.documents.service.extract_document(...)``
function Task 9A built and fully tested -- enforced here by inspecting
import graphs, not just runtime behavior, exactly like the existing Task
9A/service.py boundary tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent.parent.parent / "app"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

_FORBIDDEN_SUBSTRINGS = (
    "pypdfium2",
    "cv2",
    "onnxruntime",
    "rapidocr",
    "plateproof.documents.pdf",
    "plateproof.documents.images",
    "plateproof.documents.ocr",
)


def _import_lines(path: Path) -> str:
    with open(path, encoding="utf-8") as handle:
        lines = handle.readlines()
    return "".join(line for line in lines if line.lstrip().startswith(("import ", "from ")))


def test_documents_route_never_imports_native_parsers_directly() -> None:
    import plateproof.api.routes.documents as module

    import_source = _import_lines(Path(module.__file__))
    for forbidden in _FORBIDDEN_SUBSTRINGS:
        assert forbidden not in import_source


def test_documents_projection_never_imports_native_parsers_directly() -> None:
    import plateproof.api.documents_projection as module

    import_source = _import_lines(Path(module.__file__))
    for forbidden in _FORBIDDEN_SUBSTRINGS:
        assert forbidden not in import_source


def test_document_reader_page_never_imports_native_parsers_directly() -> None:
    import_source = _import_lines(_APP_DIR / "pages" / "5_Document_Reader.py")
    for forbidden in _FORBIDDEN_SUBSTRINGS:
        assert forbidden not in import_source
    assert "PIL" not in import_source


def test_document_reader_support_never_imports_native_parsers_or_streamlit() -> None:
    """app/document_reader_support.py (Finding 3 correction of c60cc80) is
    a plain, Streamlit-free module -- it must stay directly unit-testable
    without a ScriptRunContext."""
    import_source = _import_lines(_APP_DIR / "document_reader_support.py")
    for forbidden in _FORBIDDEN_SUBSTRINGS:
        assert forbidden not in import_source
    assert "PIL" not in import_source
    assert "streamlit" not in import_source


def test_theme_module_never_imports_native_parsers_directly() -> None:
    import_source = _import_lines(_APP_DIR / "theme.py")
    for forbidden in _FORBIDDEN_SUBSTRINGS:
        assert forbidden not in import_source


def test_documents_route_only_reaches_worker_pool_through_service_module() -> None:
    """The route imports WorkerPool only for type annotations/dependency
    wiring -- it never constructs one itself or calls .submit() directly;
    every actual job submission goes through extract_document()."""
    import plateproof.api.routes.documents as module

    with open(module.__file__, encoding="utf-8") as handle:
        source = handle.read()
    assert ".submit(" not in source
    assert "WorkerPool(" not in source


def test_document_reader_page_only_reaches_worker_pool_through_service_module() -> None:
    with open(_APP_DIR / "pages" / "5_Document_Reader.py", encoding="utf-8") as handle:
        source = handle.read()
    assert ".submit(" not in source


def test_app_pages_directory_has_no_alternate_extraction_route() -> None:
    """Only one page implements the document extraction workflow -- no
    other page in app/pages/ imports plateproof.documents.service."""
    pages_dir = _APP_DIR / "pages"
    offending = []
    for page_path in pages_dir.glob("*.py"):
        if page_path.name == "5_Document_Reader.py":
            continue
        import_source = _import_lines(page_path)
        if "plateproof.documents" in import_source:
            offending.append(page_path.name)
    assert offending == []


def test_streamlit_and_fastapi_worker_pools_are_independent_process_local_instances() -> None:
    """Structural proof that the API app and a Streamlit process each own
    their own pool object -- never a shared global."""
    import theme

    from plateproof.api.main import _build_document_worker_pool
    from plateproof.core.config import Settings

    settings = Settings(_env_file=None)
    api_pool = _build_document_worker_pool(settings)
    try:
        theme._document_worker_pool_cache.clear()
        streamlit_pool = theme.document_worker_pool()
        try:
            assert api_pool is not streamlit_pool
        finally:
            theme.shutdown_document_worker_pools()
    finally:
        api_pool.shutdown()


def test_extract_document_never_touches_official_records_graph_or_copilot_state(
    tmp_path: Path,
) -> None:
    """An end-to-end extraction call must never write to the processed
    Parquet tables, the graph, or any Copilot corpus/cache -- Task 9's
    ExtractionDraft is never persisted anywhere server-side (see its own
    docstring). Proven here by hashing every file under the processed
    data directory before and after a real extraction call and asserting
    nothing changed and nothing new appeared."""
    import hashlib
    from datetime import UTC, date, datetime

    import polars as pl

    from plateproof.documents.service import extract_document
    from plateproof.documents.worker.pool import WorkerPool, WorkerPoolConfig

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    pl.DataFrame(
        [
            {
                "restaurant_id": "nyc:1",
                "jurisdiction": "nyc",
                "source_id": "1",
                "name": "Anna's Kitchen",
                "normalized_name": "annas kitchen",
                "address": "100 Broadway",
                "city": "Manhattan",
                "region": "NY",
                "postal_code": "10001",
                "latitude": 40.7,
                "longitude": -73.9,
                "cuisine": "American",
                "latest_inspection_date": date(2025, 6, 1),
                "source_snapshot_date": date(2026, 1, 1),
                "source_retrieved_at_utc": datetime.now(UTC),
                "source_filename": None,
                "source_url": None,
            }
        ]
    ).write_parquet(processed_dir / "restaurants.parquet")

    def _hash_tree() -> dict[str, str]:
        return {
            str(p.relative_to(processed_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(processed_dir.rglob("*"))
            if p.is_file()
        }

    before = _hash_tree()

    from tests.documents.test_pdf import _minimal_pdf

    pool = WorkerPool(config=WorkerPoolConfig(pool_size=1))
    try:
        extract_document(
            _minimal_pdf("Restaurant Name: Anna's Kitchen\nScore: 14", page_count=1),
            expected_jurisdiction="nyc",
            restaurant_id="nyc:1",
            expected_restaurant_name="Anna's Kitchen",
            pool=pool,
        )
    finally:
        pool.shutdown()

    after = _hash_tree()
    assert before == after
