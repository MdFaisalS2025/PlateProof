"""Shared helpers for NYC ingestion tests.

No fixture here is autouse and nothing in this package reaches the network.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def soda_csv() -> Path:
    return FIXTURES / "nyc_sample_soda.csv"


@pytest.fixture
def display_csv() -> Path:
    return FIXTURES / "nyc_sample_display.csv"


@pytest.fixture
def metadata_json() -> Path:
    return FIXTURES / "nyc_metadata.json"


@pytest.fixture
def golden() -> dict[str, Any]:
    raw = json.loads((FIXTURES / "nyc_sample_expected.json").read_text(encoding="utf-8"))
    for event in raw["events"]:
        for key in ("inspection_date", "grade_date"):
            event[key] = date.fromisoformat(event[key]) if event[key] else None
    for violation in raw["violations"]:
        violation["inspection_date"] = date.fromisoformat(violation["inspection_date"])
    raw["report"]["snapshot_date"] = date.fromisoformat(raw["report"]["snapshot_date"])
    return raw


@pytest.fixture
def write_csv() -> Callable[[Path, list[str], list[list[str]]], Path]:
    def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> Path:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
        return path

    return _write_csv
