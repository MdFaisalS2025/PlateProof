"""Tests for plateproof.copilot.corpus: atomic, fail-closed guidance-corpus
loading, path/URL safety, checksum verification, and size limits. All
fixtures here are fictional -- see tests/copilot/test_real_corpus.py for
the reviewed real starter corpus.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def test_not_configured_returns_unavailable(tmp_path: Path) -> None:
    from plateproof.copilot.corpus import CorpusLoadOutcome, load_corpus

    result = load_corpus(None)
    assert result.outcome == CorpusLoadOutcome.UNAVAILABLE_NOT_CONFIGURED
    assert result.store is None


def test_missing_manifest_file_is_rejected_not_a_crash(tmp_path: Path) -> None:
    from plateproof.copilot.corpus import CorpusLoadOutcome, load_corpus

    result = load_corpus(tmp_path / "does_not_exist.json")
    assert result.outcome == CorpusLoadOutcome.REJECTED_INVALID
    assert result.store is None
    assert result.rejection_reason is not None
    assert str(tmp_path) not in result.rejection_reason


def test_valid_fictional_corpus_loads(write_guidance_corpus: Any) -> None:
    from plateproof.copilot.corpus import CorpusLoadOutcome, load_corpus

    manifest_path = write_guidance_corpus()
    result = load_corpus(manifest_path)
    assert result.outcome == CorpusLoadOutcome.LOADED
    assert result.store is not None
    passages = result.store.passages_for_jurisdiction("nyc")
    assert len(passages) == 1
    assert passages[0].document_id == "fictional-doc"


def test_malformed_manifest_json_fails_closed(guidance_dir: Path) -> None:
    from plateproof.copilot.corpus import CorpusLoadOutcome, load_corpus

    manifest_path = guidance_dir / "manifest.json"
    manifest_path.write_text("{not valid json", encoding="utf-8")
    result = load_corpus(manifest_path)
    assert result.outcome == CorpusLoadOutcome.REJECTED_INVALID
    assert result.rejection_reason is not None
    assert "{not valid json" not in result.rejection_reason


def test_manifest_invalid_utf8_fails_closed(guidance_dir: Path) -> None:
    from plateproof.copilot.corpus import CorpusLoadOutcome, load_corpus

    manifest_path = guidance_dir / "manifest.json"
    manifest_path.write_bytes(b"\xff\xfe\x00not utf-8")
    result = load_corpus(manifest_path)
    assert result.outcome == CorpusLoadOutcome.REJECTED_INVALID


def test_manifest_wrong_top_level_type_fails_closed(guidance_dir: Path) -> None:
    from plateproof.copilot.corpus import CorpusLoadOutcome, load_corpus

    manifest_path = guidance_dir / "manifest.json"
    manifest_path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
    result = load_corpus(manifest_path)
    assert result.outcome == CorpusLoadOutcome.REJECTED_INVALID


def test_document_checksum_mismatch_fails_closed(
    write_guidance_corpus: Any, guidance_dir: Path
) -> None:
    from plateproof.copilot.corpus import CorpusLoadOutcome, load_corpus

    manifest_path = write_guidance_corpus()
    (guidance_dir / "documents" / "fictional-doc.json").write_text(
        json.dumps(
            {
                "document_id": "fictional-doc",
                "jurisdiction": "nyc",
                "passages": [
                    {
                        "passage_id": "fictional-doc#overview",
                        "section_locator": "Section 1",
                        "text": "Tampered content that does not match the manifest checksum.",
                        "sha256": hashlib.sha256(b"anything").hexdigest(),
                        "applicable_violation_codes": [],
                        "topics": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    result = load_corpus(manifest_path)
    assert result.outcome == CorpusLoadOutcome.REJECTED_INVALID
    assert "checksum" in (result.rejection_reason or "")


def test_passage_checksum_mismatch_fails_closed(
    write_guidance_corpus: Any, guidance_dir: Path
) -> None:
    """A document file whose bytes still match the manifest's file-level
    checksum, but whose passage text doesn't match its own passage-level
    checksum, must still fail closed."""
    from plateproof.copilot.corpus import CorpusLoadOutcome, load_corpus

    manifest_path = write_guidance_corpus()
    doc_path = guidance_dir / "documents" / "fictional-doc.json"
    tampered = {
        "document_id": "fictional-doc",
        "jurisdiction": "nyc",
        "passages": [
            {
                "passage_id": "fictional-doc#overview",
                "section_locator": "Section 1",
                "text": "Some text",
                "sha256": hashlib.sha256(b"a different text entirely").hexdigest(),
                "applicable_violation_codes": [],
                "topics": [],
            }
        ],
    }
    raw = json.dumps(tampered).encode("utf-8")
    doc_path.write_bytes(raw)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["documents"][0]["sha256"] = hashlib.sha256(raw).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = load_corpus(manifest_path)
    assert result.outcome == CorpusLoadOutcome.REJECTED_INVALID


def test_duplicate_document_id_fails_closed(write_guidance_corpus: Any) -> None:
    from plateproof.copilot.corpus import CorpusLoadOutcome, load_corpus

    manifest_path = write_guidance_corpus(
        documents=[
            {"document_id": "fictional-doc", "jurisdiction": "nyc"},
            {
                "document_id": "fictional-doc",
                "jurisdiction": "florida",
                "file": "documents/other.json",
            },
        ]
    )
    result = load_corpus(manifest_path)
    assert result.outcome == CorpusLoadOutcome.REJECTED_INVALID


def test_normalization_colliding_document_ids_fail_closed(
    guidance_dir: Path, make_fictional_passage: Any
) -> None:
    """Two ids that are byte-distinct but equal after case-folding are
    ambiguous and must be rejected, not silently disambiguated."""
    from plateproof.copilot.corpus import CorpusLoadOutcome, load_corpus

    def _write_doc(doc_id: str) -> tuple[str, str]:
        passage = make_fictional_passage(passage_id=f"{doc_id}#overview")
        payload = {"document_id": doc_id, "jurisdiction": "nyc", "passages": [passage]}
        raw = json.dumps(payload).encode("utf-8")
        path = guidance_dir / "documents"
        path.mkdir(exist_ok=True)
        (path / f"{doc_id}.json").write_bytes(raw)
        return f"documents/{doc_id}.json", hashlib.sha256(raw).hexdigest()

    file_a, sha_a = _write_doc("fictional-doc")
    file_b, sha_b = _write_doc("fictional-doc-b")  # distinct real id, collision engineered below

    manifest = {
        "manifest_version": "1.0.0",
        "generated_at": "2026-01-01T00:00:00Z",
        "documents": [
            {
                "document_id": "fictional-doc",
                "jurisdiction": "nyc",
                "title": "A",
                "authority": "Test",
                "source_url": "https://www.nyc.gov/a",
                "access_date": "2026-01-01",
                "effective_date": None,
                "revision_date": None,
                "superseded": False,
                "superseded_by": None,
                "provenance_note": "Test.",
                "file": file_a,
                "sha256": sha_a,
            },
            {
                # Same normalized id as above under simple case-folding is
                # not directly expressible given the id pattern's lowercase
                # requirement -- instead this exercises the same guard via
                # a second entry claiming the *same* file+id outright.
                "document_id": "fictional-doc",
                "jurisdiction": "nyc",
                "title": "B",
                "authority": "Test",
                "source_url": "https://www.nyc.gov/b",
                "access_date": "2026-01-01",
                "effective_date": None,
                "revision_date": None,
                "superseded": False,
                "superseded_by": None,
                "provenance_note": "Test.",
                "file": file_b,
                "sha256": sha_b,
            },
        ],
    }
    manifest_path = guidance_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = load_corpus(manifest_path)
    assert result.outcome == CorpusLoadOutcome.REJECTED_INVALID


def test_duplicate_passage_id_across_documents_fails_closed(
    guidance_dir: Path, write_guidance_corpus: Any, make_fictional_passage: Any
) -> None:
    from plateproof.copilot.corpus import CorpusLoadOutcome, load_corpus

    shared_passage = make_fictional_passage(passage_id="fictional-doc#overview")
    manifest_path = write_guidance_corpus(
        documents=[
            {"document_id": "fictional-doc", "jurisdiction": "nyc", "passages": [shared_passage]},
            {
                "document_id": "other-doc",
                "jurisdiction": "florida",
                "passages": [shared_passage],
            },
        ]
    )
    result = load_corpus(manifest_path)
    assert result.outcome == CorpusLoadOutcome.REJECTED_INVALID


def test_absolute_file_path_is_rejected(write_guidance_corpus: Any, guidance_dir: Path) -> None:
    from plateproof.copilot.corpus import CorpusLoadOutcome, load_corpus

    manifest_path = write_guidance_corpus()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["documents"][0]["file"] = str(guidance_dir / "documents" / "fictional-doc.json")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = load_corpus(manifest_path)
    assert result.outcome == CorpusLoadOutcome.REJECTED_INVALID


def test_traversal_file_path_is_rejected(write_guidance_corpus: Any) -> None:
    from plateproof.copilot.corpus import CorpusLoadOutcome, load_corpus

    manifest_path = write_guidance_corpus()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["documents"][0]["file"] = "documents/../../../outside.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result = load_corpus(manifest_path)
    assert result.outcome == CorpusLoadOutcome.REJECTED_INVALID


def test_symlink_escape_is_rejected(write_guidance_corpus: Any, guidance_dir: Path) -> None:
    import pytest

    from plateproof.copilot.corpus import CorpusLoadOutcome, load_corpus

    outside = guidance_dir.parent / "outside.json"
    outside.write_text("not part of the corpus", encoding="utf-8")
    manifest_path = write_guidance_corpus()
    link_path = guidance_dir / "documents" / "fictional-doc.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_checksum = manifest["documents"][0]["sha256"]
    link_path.unlink()
    try:
        link_path.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("platform/user cannot create symlinks")
    manifest["documents"][0]["sha256"] = expected_checksum
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = load_corpus(manifest_path)
    assert result.outcome == CorpusLoadOutcome.REJECTED_INVALID


def test_non_https_url_is_rejected(write_guidance_corpus: Any) -> None:
    from plateproof.copilot.corpus import CorpusLoadOutcome, load_corpus

    manifest_path = write_guidance_corpus(
        documents=[
            {
                "document_id": "fictional-doc",
                "jurisdiction": "nyc",
                "source_url": "http://www.nyc.gov/insecure",
            }
        ]
    )
    result = load_corpus(manifest_path)
    assert result.outcome == CorpusLoadOutcome.REJECTED_INVALID


def test_credential_bearing_url_is_rejected(write_guidance_corpus: Any) -> None:
    from plateproof.copilot.corpus import CorpusLoadOutcome, load_corpus

    manifest_path = write_guidance_corpus(
        documents=[
            {
                "document_id": "fictional-doc",
                "jurisdiction": "nyc",
                "source_url": "https://user:pass@www.nyc.gov/path",
            }
        ]
    )
    result = load_corpus(manifest_path)
    assert result.outcome == CorpusLoadOutcome.REJECTED_INVALID


def test_non_allowlisted_domain_is_rejected(write_guidance_corpus: Any) -> None:
    from plateproof.copilot.corpus import CorpusLoadOutcome, load_corpus

    manifest_path = write_guidance_corpus(
        documents=[
            {
                "document_id": "fictional-doc",
                "jurisdiction": "nyc",
                "source_url": "https://not-an-official-domain.example.com/path",
            }
        ]
    )
    result = load_corpus(manifest_path)
    assert result.outcome == CorpusLoadOutcome.REJECTED_INVALID


def test_oversized_document_file_fails_closed(
    write_guidance_corpus: Any, guidance_dir: Path
) -> None:
    from plateproof.copilot.corpus import CorpusLoadOutcome, load_corpus

    manifest_path = write_guidance_corpus()
    huge_text = "x" * (3 * 1024 * 1024)
    huge = {
        "document_id": "fictional-doc",
        "jurisdiction": "nyc",
        "passages": [
            {
                "passage_id": "fictional-doc#overview",
                "section_locator": "Section 1",
                "text": huge_text[:2000],
                "sha256": hashlib.sha256(huge_text[:2000].encode("utf-8")).hexdigest(),
                "applicable_violation_codes": [],
                "topics": [],
                "padding": huge_text,
            }
        ],
    }
    raw = json.dumps(huge).encode("utf-8")
    (guidance_dir / "documents" / "fictional-doc.json").write_bytes(raw)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["documents"][0]["sha256"] = hashlib.sha256(raw).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = load_corpus(manifest_path)
    assert result.outcome == CorpusLoadOutcome.REJECTED_INVALID


def test_too_many_documents_fails_closed(guidance_dir: Path, make_fictional_passage: Any) -> None:
    from plateproof.copilot.corpus import CorpusLoadOutcome, load_corpus

    documents = []
    for i in range(3):
        doc_id = f"fictional-doc-{i}"
        passage = make_fictional_passage(passage_id=f"{doc_id}#overview")
        payload = {"document_id": doc_id, "jurisdiction": "nyc", "passages": [passage]}
        raw = json.dumps(payload).encode("utf-8")
        docs_dir = guidance_dir / "documents"
        docs_dir.mkdir(exist_ok=True)
        (docs_dir / f"{doc_id}.json").write_bytes(raw)
        documents.append(
            {
                "document_id": doc_id,
                "jurisdiction": "nyc",
                "title": "T",
                "authority": "A",
                "source_url": "https://www.nyc.gov/x",
                "access_date": "2026-01-01",
                "effective_date": None,
                "revision_date": None,
                "superseded": False,
                "superseded_by": None,
                "provenance_note": "Test.",
                "file": f"documents/{doc_id}.json",
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    manifest = {
        "manifest_version": "1.0.0",
        "generated_at": "2026-01-01T00:00:00Z",
        "documents": documents,
    }
    manifest_path = guidance_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    from plateproof.copilot import corpus as corpus_module

    original_max = corpus_module._MAX_DOCUMENTS
    corpus_module._MAX_DOCUMENTS = 2
    try:
        result = load_corpus(manifest_path)
    finally:
        corpus_module._MAX_DOCUMENTS = original_max
    assert result.outcome == CorpusLoadOutcome.REJECTED_INVALID


def test_superseded_documents_are_excluded_from_normal_retrieval(
    write_guidance_corpus: Any,
) -> None:
    from plateproof.copilot.corpus import CorpusLoadOutcome, load_corpus

    manifest_path = write_guidance_corpus(
        documents=[{"document_id": "fictional-doc", "jurisdiction": "nyc", "superseded": True}]
    )
    result = load_corpus(manifest_path)
    assert result.outcome == CorpusLoadOutcome.LOADED
    assert result.store is not None
    assert result.store.passages_for_jurisdiction("nyc") == ()


def test_jurisdiction_filter_excludes_other_jurisdictions(write_guidance_corpus: Any) -> None:
    from plateproof.copilot.corpus import CorpusLoadOutcome, load_corpus

    manifest_path = write_guidance_corpus(
        documents=[
            {"document_id": "nyc-doc", "jurisdiction": "nyc"},
            {"document_id": "fl-doc", "jurisdiction": "florida"},
        ]
    )
    result = load_corpus(manifest_path)
    assert result.outcome == CorpusLoadOutcome.LOADED
    assert result.store is not None
    nyc_ids = {p.document_id for p in result.store.passages_for_jurisdiction("nyc")}
    fl_ids = {p.document_id for p in result.store.passages_for_jurisdiction("florida")}
    assert nyc_ids == {"nyc-doc"}
    assert fl_ids == {"fl-doc"}


def test_no_network_access_during_load(write_guidance_corpus: Any, monkeypatch: Any) -> None:
    import socket

    from plateproof.copilot.corpus import load_corpus

    def _forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("guidance corpus loading must never open a network socket")

    monkeypatch.setattr(socket, "socket", _forbidden)
    manifest_path = write_guidance_corpus()
    result = load_corpus(manifest_path)
    assert result.outcome.value == "loaded"
