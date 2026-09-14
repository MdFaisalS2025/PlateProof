"""Verifies the committed, reviewed starter corpus under
``data/reference/guidance`` -- the actual production content, not a test
fixture -- loads cleanly through the same secure loader the application
uses, and audits clean with zero findings.
"""

from __future__ import annotations

from pathlib import Path

_GUIDANCE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "reference" / "guidance"


def test_real_starter_corpus_loads_successfully() -> None:
    from plateproof.copilot.corpus import CorpusLoadOutcome, load_corpus

    result = load_corpus(_GUIDANCE_DIR / "manifest.json")
    assert result.outcome == CorpusLoadOutcome.LOADED, result.rejection_reason
    assert result.store is not None
    assert len(result.store.passages) == 5


def test_real_starter_corpus_covers_nyc_florida_and_federal() -> None:
    from plateproof.copilot.corpus import load_corpus

    result = load_corpus(_GUIDANCE_DIR / "manifest.json")
    assert result.store is not None
    jurisdictions = {p.jurisdiction for p in result.store.passages}
    assert jurisdictions == {"nyc", "florida", "federal"}


def test_real_starter_corpus_audits_clean() -> None:
    from plateproof.copilot.corpus import audit_corpus

    report = audit_corpus(_GUIDANCE_DIR / "manifest.json")
    assert report.passed, report.findings
    assert report.documents_examined == 3
    assert report.passages_examined == 5


def test_real_starter_corpus_urls_are_all_allowlisted() -> None:
    from plateproof.copilot.corpus import load_corpus, validate_official_url

    result = load_corpus(_GUIDANCE_DIR / "manifest.json")
    assert result.store is not None
    for passage in result.store.passages:
        assert validate_official_url(passage.source_url), passage.source_url


def test_real_starter_corpus_has_no_direct_code_mapping_without_support() -> None:
    """Per the source-policy README: no passage in the current starter
    corpus claims a direct violation-code mapping -- only topic-level
    associations, since no reviewed source states an explicit code link."""
    from plateproof.copilot.corpus import load_corpus

    result = load_corpus(_GUIDANCE_DIR / "manifest.json")
    assert result.store is not None
    for passage in result.store.passages:
        assert passage.applicable_violation_codes == ()
        assert len(passage.topics) > 0


def test_federal_passages_are_never_retrievable_through_any_jurisdiction_query() -> None:
    """Documented policy (independent-review correction item 4/10):
    federal guidance is loaded into the corpus for completeness/audit but
    is never retrievable through any Task 8A intent, since every
    retrieval call filters by the restaurant's own nyc/florida
    jurisdiction, never "federal"."""
    from plateproof.copilot.corpus import load_corpus
    from plateproof.copilot.retrieval import passages_for_codes, passages_for_topics

    result = load_corpus(_GUIDANCE_DIR / "manifest.json")
    assert result.store is not None
    federal_passages = [p for p in result.store.passages if p.jurisdiction == "federal"]
    assert federal_passages  # the FDA document is genuinely in the corpus

    # Never reachable via passages_for_jurisdiction("federal") from the
    # normal nyc/florida-only call sites in CopilotService...
    assert result.store.passages_for_jurisdiction("nyc") == () or all(
        p.jurisdiction == "nyc" for p in result.store.passages_for_jurisdiction("nyc")
    )
    assert result.store.passages_for_jurisdiction("florida") == () or all(
        p.jurisdiction == "florida" for p in result.store.passages_for_jurisdiction("florida")
    )
    # ...and no exact-match retrieval for "federal" ever surfaces them
    # through a caller that (incorrectly) tried to pass it as a
    # jurisdiction -- the type system already forbids this
    # (RestaurantJurisdiction excludes "federal"), and this is the
    # runtime confirmation that the underlying data-shape agrees.
    for topics in ({"food_code_overview"},):
        assert passages_for_topics(result.store, "nyc", tuple(topics)) == ()
        assert passages_for_topics(result.store, "florida", tuple(topics)) == ()
    assert passages_for_codes(result.store, "nyc", ()) == ()
