"""Tests for the hand-maintained Florida source manifest and snapshot metadata.

No network access -- the manifest is a static, versioned data structure.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest


def test_manifest_entries_are_well_formed() -> None:
    from plateproof.ingestion.florida import FLORIDA_SOURCE_MANIFEST

    seen: set[tuple[str, int | None]] = set()
    for entry in FLORIDA_SOURCE_MANIFEST:
        assert entry.url.startswith("https://")
        assert entry.format in ("csv", "xlsx", "xls")
        key = (entry.fiscal_year, entry.district)
        assert key not in seen, f"duplicate manifest entry for {key}"
        seen.add(key)


def test_manifest_includes_seven_current_csv_districts() -> None:
    from plateproof.ingestion.florida import FLORIDA_SOURCE_MANIFEST

    current = [
        e for e in FLORIDA_SOURCE_MANIFEST if e.fiscal_year == "current" and e.format == "csv"
    ]
    assert {e.district for e in current} == set(range(1, 8))


def test_manifest_deferred_legacy_formats_carry_a_note() -> None:
    from plateproof.ingestion.florida import FLORIDA_SOURCE_MANIFEST

    deferred = [e for e in FLORIDA_SOURCE_MANIFEST if e.format == "xls"]
    assert deferred, "expected at least one deferred legacy .xls manifest entry"
    for entry in deferred:
        assert entry.note and "deferred" in entry.note.lower()


def test_manifest_includes_recent_xlsx_statewide_years() -> None:
    from plateproof.ingestion.florida import FLORIDA_SOURCE_MANIFEST

    xlsx_years = {e.fiscal_year for e in FLORIDA_SOURCE_MANIFEST if e.format == "xlsx"}
    assert xlsx_years, "expected at least one supported historical XLSX fiscal year"


def test_load_florida_snapshot_metadata_parses_fixture(tmp_path: Path) -> None:
    from plateproof.ingestion.florida import load_florida_snapshot_metadata

    payload = {
        "dataset_id": "fl_dbpr_food_service_inspections",
        "retrieved_at_utc": "2026-02-01T05:00:00Z",
        "requested_fiscal_years": ["current"],
        "requested_districts": [1, 2],
        "files": [
            {
                "url": "https://example/1fdinspi.csv",
                "fiscal_year": "current",
                "district": 1,
                "format": "csv",
                "local_filename": "1fdinspi.csv",
                "retrieved_at_utc": "2026-02-01T05:00:00Z",
                "sha256": "0" * 64,
                "byte_size": 100,
                "row_count": 5,
                "encoding": "utf-8",
            }
        ],
        "downloader_version": "0.1.0",
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    meta = load_florida_snapshot_metadata(path)
    assert meta.retrieved_at_utc == datetime(2026, 2, 1, 5, 0, tzinfo=UTC)
    assert len(meta.files) == 1
    assert meta.files[0].sha256.startswith("0000")


def test_load_florida_snapshot_metadata_missing_file_raises(tmp_path: Path) -> None:
    from plateproof.ingestion.florida import load_florida_snapshot_metadata

    with pytest.raises(FileNotFoundError):
        load_florida_snapshot_metadata(tmp_path / "absent.json")
