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
