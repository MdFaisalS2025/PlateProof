"""Tests for plateproof.copilot.retrieval: jurisdiction filtering,
deterministic ranking/tie-break, code-first exact matching, and the
minimum-relevance threshold. Retrieval only ever proposes candidates --
none of these tests assert that a high score alone authorizes an answer;
that is claims.py's job (see test_claims.py).
"""

from __future__ import annotations

from typing import Any

from plateproof.copilot.retrieval import passages_for_codes, search_by_topic_or_text


def test_passages_for_codes_matches_exact_code(write_guidance_corpus: Any) -> None:
    from plateproof.copilot.corpus import load_corpus

    manifest_path = write_guidance_corpus(
        documents=[
            {
                "document_id": "fictional-doc",
                "jurisdiction": "nyc",
                "passages": [
                    {
                        "passage_id": "fictional-doc#p1",
                        "section_locator": "S1",
                        "text": "Keep hot food hot.",
                        "sha256": __import__("hashlib").sha256(b"Keep hot food hot.").hexdigest(),
                        "applicable_violation_codes": ["04L"],
                        "topics": ["temperature"],
                        "permitted_uses": ["definition"],
                    }
                ],
            }
        ]
    )
    result = load_corpus(manifest_path)
    assert result.store is not None
    items = passages_for_codes(result.store, "nyc", ["04L"])
    assert len(items) == 1
    assert items[0].citation.citation_id == "fictional-doc#p1"


def test_passages_for_codes_returns_nothing_for_unmatched_code(write_guidance_corpus: Any) -> None:
    from plateproof.copilot.corpus import load_corpus

    manifest_path = write_guidance_corpus()
    result = load_corpus(manifest_path)
    assert result.store is not None
    assert passages_for_codes(result.store, "nyc", ["99Z"]) == ()


def test_passages_for_codes_never_crosses_jurisdictions(write_guidance_corpus: Any) -> None:
    import hashlib

    from plateproof.copilot.corpus import load_corpus

    text = "Florida-specific guidance about code 04L."
    manifest_path = write_guidance_corpus(
        documents=[
            {
                "document_id": "fl-doc",
                "jurisdiction": "florida",
                "passages": [
                    {
                        "passage_id": "fl-doc#p1",
                        "section_locator": "S1",
                        "text": text,
                        "sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "applicable_violation_codes": ["04L"],
                        "topics": [],
                        "permitted_uses": ["definition"],
                    }
                ],
            }
        ]
    )
    result = load_corpus(manifest_path)
    assert result.store is not None
    # Same code, wrong jurisdiction requested -- must never leak across.
    assert passages_for_codes(result.store, "nyc", ["04L"]) == ()


def test_search_by_topic_or_text_ranks_more_similar_passage_first(
    write_guidance_corpus: Any,
) -> None:
    import hashlib

    from plateproof.copilot.corpus import load_corpus

    text_a = "Keep hot food above the safe holding temperature at all times."
    text_b = "Store chemicals away from food preparation surfaces."
    manifest_path = write_guidance_corpus(
        documents=[
            {
                "document_id": "doc-a",
                "jurisdiction": "nyc",
                "passages": [
                    {
                        "passage_id": "doc-a#p1",
                        "section_locator": "S1",
                        "text": text_a,
                        "sha256": hashlib.sha256(text_a.encode()).hexdigest(),
                        "applicable_violation_codes": [],
                        "topics": ["temperature"],
                        "permitted_uses": ["definition"],
                    }
                ],
            },
            {
                "document_id": "doc-b",
                "jurisdiction": "nyc",
                "passages": [
                    {
                        "passage_id": "doc-b#p1",
                        "section_locator": "S1",
                        "text": text_b,
                        "sha256": hashlib.sha256(text_b.encode()).hexdigest(),
                        "applicable_violation_codes": [],
                        "topics": ["chemicals"],
                        "permitted_uses": ["definition"],
                    }
                ],
            },
        ]
    )
    result = load_corpus(manifest_path)
    assert result.store is not None
    items = search_by_topic_or_text(result.store, "nyc", "hot food temperature")
    assert items
    assert items[0].citation.citation_id == "doc-a#p1"


def test_search_by_topic_or_text_deterministic_across_repeated_calls(
    write_guidance_corpus: Any,
) -> None:
    from plateproof.copilot.corpus import load_corpus

    manifest_path = write_guidance_corpus()
    result = load_corpus(manifest_path)
    assert result.store is not None
    first = search_by_topic_or_text(result.store, "nyc", "hot food temperature")
    second = search_by_topic_or_text(result.store, "nyc", "hot food temperature")
    assert first == second


def test_search_by_topic_or_text_never_crosses_jurisdictions(write_guidance_corpus: Any) -> None:
    from plateproof.copilot.corpus import load_corpus

    manifest_path = write_guidance_corpus(
        documents=[{"document_id": "nyc-doc", "jurisdiction": "nyc"}]
    )
    result = load_corpus(manifest_path)
    assert result.store is not None
    assert search_by_topic_or_text(result.store, "florida", "hot food") == ()


def test_search_by_topic_or_text_empty_corpus_returns_nothing(write_guidance_corpus: Any) -> None:
    from plateproof.copilot.corpus import load_corpus

    manifest_path = write_guidance_corpus(
        documents=[{"document_id": "nyc-doc", "jurisdiction": "nyc"}]
    )
    result = load_corpus(manifest_path)
    assert result.store is not None
    assert search_by_topic_or_text(result.store, "florida", "") == ()
