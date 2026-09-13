"""Tests for scripts/review_guidance_corpus.py."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def test_review_passes_for_the_real_committed_corpus(capsys: Any) -> None:
    from scripts.review_guidance_corpus import main

    exit_code = main([])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "PASS" in captured.out


def test_review_fails_for_a_corpus_with_a_checksum_mismatch(
    write_guidance_corpus: Any, guidance_dir: Path, capsys: Any
) -> None:
    import hashlib
    import json

    from scripts.review_guidance_corpus import main

    manifest_path = write_guidance_corpus()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["documents"][0]["sha256"] = hashlib.sha256(b"wrong").hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    exit_code = main(["--manifest-path", str(manifest_path)])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "FAIL" in captured.out
    assert "sha256" in captured.out
