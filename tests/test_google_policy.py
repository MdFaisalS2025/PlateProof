"""Task 10 policy barrier: nothing under ``plateproof/ingestion``,
``plateproof/features``, or ``plateproof/models`` may import anything from
``plateproof.serving`` (which is where the Task 10 Google-link helpers
live, alongside every other presentation-layer concern) -- this makes "no
Google content ever reaches training data" true by construction, not by
convention, mirroring ``tests/documents/test_entry_point_import_graph.py``'s
approach for Task 9's worker-isolation boundary.

``tests/features/test_temporal.py::test_allowlist_rejects_google_column``
already covers the complementary runtime guard
(``assert_model_matrix_is_safe`` rejects a ``google``-prefixed column even
if it were somehow allowlisted); this file is the static, import-level
guarantee that no such column-producing code could exist in the first
place.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_GUARDED_DIRS = (
    _REPO_ROOT / "plateproof" / "ingestion",
    _REPO_ROOT / "plateproof" / "features",
    _REPO_ROOT / "plateproof" / "models",
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


def test_ingestion_feature_and_model_code_never_imports_the_serving_layer() -> None:
    for directory in _GUARDED_DIRS:
        for path in directory.rglob("*.py"):
            imported = _imported_module_names(path)
            for name in imported:
                assert not name.startswith("plateproof.serving"), (
                    f"{path} imports {name!r} -- the presentation/serving layer (where "
                    "Task 10's Google-link helpers live) must never be reachable from "
                    "ingestion, feature, or model code"
                )


def test_guarded_directories_actually_exist() -> None:
    """A sanity check on the test itself: if these directories were ever
    renamed, the test above would silently stop checking anything."""
    for directory in _GUARDED_DIRS:
        assert directory.is_dir(), f"expected directory not found: {directory}"
