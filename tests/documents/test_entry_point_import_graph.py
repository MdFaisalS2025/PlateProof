"""Import-graph scaffolding test for the Task 9A/9B boundary.

Written in 9A, before ``app/pages/5_Document_Reader.py`` or
``plateproof/api/routes/documents.py`` exist, precisely so it can never be
retrofitted around an already-violating implementation. When 9B creates
those files, this test starts enforcing immediately: neither entry point
may import ``pdf.py``/``images.py``/``ocr/*``/the wire protocol/the worker
entrypoint directly -- both must route everything through
``plateproof.documents.service.extract_document(...)``, which alone owns a
``plateproof.documents.worker.pool.WorkerPool`` instance. Constructing a
``WorkerPool`` (its public, safe API) is the one thing an entry point is
expected to do itself, per the plan's "each application process owns its
own worker pool" design -- only the parser-touching submodules are
forbidden.
"""

from __future__ import annotations

import ast
from pathlib import Path

_FORBIDDEN_MODULE_PREFIXES = (
    "plateproof.documents.pdf",
    "plateproof.documents.images",
    "plateproof.documents.ocr",
    "plateproof.documents.worker.entrypoint",
    "plateproof.documents.worker.protocol",
)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_ENTRY_POINT_FILES = (
    _REPO_ROOT / "plateproof" / "api" / "routes" / "documents.py",
    _REPO_ROOT / "app" / "pages" / "5_Document_Reader.py",
)
_THEME_FILE = _REPO_ROOT / "app" / "theme.py"


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


def _assert_no_forbidden_imports(path: Path) -> None:
    imported = _imported_module_names(path)
    for name in imported:
        for forbidden in _FORBIDDEN_MODULE_PREFIXES:
            assert not name.startswith(forbidden), (
                f"{path} imports {name!r}, which bypasses the worker-isolation boundary "
                "established in Task 9A"
            )


def test_9b_entry_points_never_import_native_parsers_directly() -> None:
    """Neither Task 9B entry point exists yet -- when either is created,
    this test starts enforcing on it automatically."""
    for entry_point in _ENTRY_POINT_FILES:
        if entry_point.exists():
            _assert_no_forbidden_imports(entry_point)


def test_theme_module_never_imports_native_parsers_directly() -> None:
    if _THEME_FILE.exists():
        _assert_no_forbidden_imports(_THEME_FILE)


def test_service_module_is_the_only_place_worker_pool_is_constructed_from_documents_package() -> (
    None
):
    """draft_builder.py, the extractors, corrections.py, and validation.py
    must never construct or import WorkerPool themselves -- only service.py
    (and the worker package's own internals) may."""
    documents_dir = _REPO_ROOT / "plateproof" / "documents"
    allowed_importers = {documents_dir / "service.py", documents_dir / "worker" / "pool.py"}
    for path in documents_dir.rglob("*.py"):
        if path in allowed_importers or path.parent.name == "worker" or path.parent.name == "ocr":
            continue
        imported = _imported_module_names(path)
        assert "plateproof.documents.worker.pool" not in imported, (
            f"{path} imports worker.pool directly"
        )
