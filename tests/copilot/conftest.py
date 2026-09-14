"""Shared fixtures for tests/copilot. All corpus content here is fictional
-- the real, reviewed starter corpus lives under
``data/reference/guidance/`` and is exercised by
``tests/copilot/test_real_corpus.py`` instead.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

_FICTIONAL_TEXT = (
    "Fictional guidance: keep hot food above the documented safe temperature "
    "at all times during service."
)


def _write_document_file(
    guidance_dir: Path,
    *,
    document_id: str,
    jurisdiction: str,
    passages: list[dict[str, Any]],
) -> tuple[str, str]:
    """Writes ``documents/<document_id>.json`` and returns
    ``(relative_path, sha256_of_file_bytes)``."""
    doc_payload = {"document_id": document_id, "jurisdiction": jurisdiction, "passages": passages}
    raw = json.dumps(doc_payload).encode("utf-8")
    documents_dir = guidance_dir / "documents"
    documents_dir.mkdir(parents=True, exist_ok=True)
    path = documents_dir / f"{document_id}.json"
    path.write_bytes(raw)
    return f"documents/{document_id}.json", hashlib.sha256(raw).hexdigest()


@pytest.fixture
def guidance_dir(tmp_path: Path) -> Path:
    d = tmp_path / "guidance"
    d.mkdir()
    return d


@pytest.fixture
def make_fictional_passage() -> Any:
    def _make(
        *,
        passage_id: str = "fictional-doc#overview",
        text: str = _FICTIONAL_TEXT,
        section_locator: str = "Section 1",
        applicable_violation_codes: list[str] | None = None,
        topics: list[str] | None = None,
        permitted_uses: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "passage_id": passage_id,
            "section_locator": section_locator,
            "text": text,
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "applicable_violation_codes": applicable_violation_codes or [],
            "topics": topics or ["fictional_topic"],
            "permitted_uses": permitted_uses or ["definition"],
        }

    return _make


@pytest.fixture
def write_guidance_corpus(guidance_dir: Path, make_fictional_passage: Any) -> Any:
    def _write(
        *,
        documents: list[dict[str, Any]] | None = None,
        manifest_overrides: dict[str, Any] | None = None,
    ) -> Path:
        """Writes a complete, valid fictional corpus (unless overridden) and
        returns the manifest path. ``documents`` overrides the default
        single-document corpus; each item is a dict of manifest-entry
        overrides plus an optional ``passages`` list (defaults to one
        fictional passage)."""
        if documents is None:
            documents = [{"document_id": "fictional-doc", "jurisdiction": "nyc"}]

        manifest_documents = []
        for doc_spec in documents:
            doc_spec = dict(doc_spec)
            document_id = doc_spec.pop("document_id")
            jurisdiction = doc_spec.pop("jurisdiction")
            passages = doc_spec.pop("passages", None) or [
                make_fictional_passage(passage_id=f"{document_id}#overview")
            ]
            relative_path, file_sha256 = _write_document_file(
                guidance_dir, document_id=document_id, jurisdiction=jurisdiction, passages=passages
            )
            entry = {
                "document_id": document_id,
                "jurisdiction": jurisdiction,
                "title": doc_spec.pop("title", "Fictional Guidance Document"),
                "authority": doc_spec.pop("authority", "Fictional Testing Authority"),
                "source_url": doc_spec.pop("source_url", "https://www.nyc.gov/fictional-test-path"),
                "access_date": doc_spec.pop("access_date", "2026-01-01"),
                "effective_date": doc_spec.pop("effective_date", None),
                "revision_date": doc_spec.pop("revision_date", None),
                "superseded": doc_spec.pop("superseded", False),
                "superseded_by": doc_spec.pop("superseded_by", None),
                "provenance_note": doc_spec.pop("provenance_note", "Fictional test fixture."),
                "file": doc_spec.pop("file", relative_path),
                "sha256": doc_spec.pop("sha256", file_sha256),
            }
            entry.update(doc_spec)  # any remaining explicit overrides win
            manifest_documents.append(entry)

        manifest = {
            "manifest_version": "1.0.0",
            "generated_at": "2026-01-01T00:00:00Z",
            "documents": manifest_documents,
        }
        if manifest_overrides:
            manifest.update(manifest_overrides)
        manifest_path = guidance_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path

    return _write
