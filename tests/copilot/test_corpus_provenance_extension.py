"""RED-first tests for Task 8B's corpus provenance propagation:
CorpusPassage (and therefore the Citation retrieval builds from it) must
carry access_date/effective_date/revision_date through from the manifest
document entry -- they already exist on GuidanceManifestDocumentEntry but
were not flattened onto CorpusPassage in Task 8A."""

from __future__ import annotations

from datetime import date
from typing import Any

from plateproof.copilot.corpus import load_corpus


def test_corpus_passage_carries_manifest_provenance_dates(write_guidance_corpus: Any) -> None:
    manifest_path = write_guidance_corpus(
        documents=[
            {
                "document_id": "fictional-doc",
                "jurisdiction": "nyc",
                "access_date": "2026-02-01",
                "effective_date": "2025-05-01",
                "revision_date": "2025-06-01",
            }
        ]
    )
    result = load_corpus(manifest_path)
    assert result.store is not None
    passage = result.store.passages[0]
    assert passage.access_date == date(2026, 2, 1)
    assert passage.effective_date == date(2025, 5, 1)
    assert passage.revision_date == date(2025, 6, 1)


def test_corpus_passage_effective_and_revision_dates_default_to_none(
    write_guidance_corpus: Any,
) -> None:
    manifest_path = write_guidance_corpus()
    result = load_corpus(manifest_path)
    assert result.store is not None
    passage = result.store.passages[0]
    assert passage.access_date == date(2026, 1, 1)
    assert passage.effective_date is None
    assert passage.revision_date is None
