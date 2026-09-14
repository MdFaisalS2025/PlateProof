"""RED-first test for Task 8B: a citation built from a guidance passage
must carry its full reviewed provenance (issuing authority, section
locator, access/effective/revision dates) -- previously only title/url/
excerpt/jurisdiction/superseded were populated, and as_of_date was always
None even though the corpus records an access_date."""

from __future__ import annotations

from datetime import date
from typing import Any

from plateproof.copilot import retrieval
from plateproof.copilot.corpus import load_corpus


def test_guidance_citation_carries_full_provenance(
    write_guidance_corpus: Any, make_fictional_passage: Any
) -> None:
    manifest_path = write_guidance_corpus(
        documents=[
            {
                "document_id": "fictional-doc",
                "jurisdiction": "nyc",
                "access_date": "2026-02-01",
                "effective_date": "2025-05-01",
                "revision_date": "2025-06-01",
                "authority": "Fictional Testing Authority",
                "passages": [
                    make_fictional_passage(
                        passage_id="fictional-doc#overview",
                        applicable_violation_codes=["04L"],
                    )
                ],
            }
        ]
    )
    result = load_corpus(manifest_path)
    assert result.store is not None
    items = retrieval.passages_for_codes(result.store, "nyc", ["04L"])
    assert items
    citation = items[0].citation
    assert citation.issuing_authority == "Fictional Testing Authority"
    assert citation.section_locator == "Section 1"
    assert citation.access_date == date(2026, 2, 1)
    assert citation.effective_date == date(2025, 5, 1)
    assert citation.revision_date == date(2025, 6, 1)
    assert citation.as_of_date == date(2026, 2, 1)
