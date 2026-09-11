"""Tests for load_source_metadata."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest


def test_parses_valid_sidecar(metadata_json: Path) -> None:
    from plateproof.ingestion.nyc import load_source_metadata

    meta = load_source_metadata(metadata_json)
    assert meta.dataset_id == "43nn-pn8j"
    assert meta.source_sha256.startswith("0000")
    assert meta.retrieved_at_utc == datetime(2026, 9, 9, 14, 30, tzinfo=UTC)
    assert meta.app_token_used is False


def test_missing_file_raises_filenotfound(tmp_path: Path) -> None:
    from plateproof.ingestion.nyc import load_source_metadata

    with pytest.raises(FileNotFoundError):
        load_source_metadata(tmp_path / "absent.json")


def test_missing_key_raises_valueerror(tmp_path: Path, metadata_json: Path) -> None:
    from plateproof.ingestion.nyc import load_source_metadata

    payload = json.loads(metadata_json.read_text(encoding="utf-8"))
    del payload["source_sha256"]
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="source_sha256"):
        load_source_metadata(broken)
