"""Tests for scripts/build_processed_tables.py using the existing tiny
ingestion test fixtures. Never downloads anything."""

from __future__ import annotations

from pathlib import Path

import polars as pl

_FIXTURES = Path(__file__).parent.parent / "ingestion" / "fixtures"


def test_builds_nyc_and_florida_tables(tmp_path: Path) -> None:
    from scripts.build_processed_tables import main

    output = tmp_path / "processed"
    exit_code = main(
        [
            "--nyc-events",
            str(_FIXTURES / "nyc_sample_soda.csv"),
            "--florida-events",
            str(_FIXTURES / "fl_sample_current.csv"),
            "--output",
            str(output),
        ]
    )
    assert exit_code == 0
    restaurants = pl.read_parquet(output / "restaurants.parquet")
    assert restaurants.height > 0
    assert set(restaurants.get_column("jurisdiction").unique().to_list()) <= {"nyc", "florida"}
    assert (output / "inspection_events.parquet").exists()
    assert (output / "violation_events.parquet").exists()


def test_nyc_only_does_not_require_florida(tmp_path: Path) -> None:
    from scripts.build_processed_tables import main

    output = tmp_path / "processed"
    exit_code = main(
        ["--nyc-events", str(_FIXTURES / "nyc_sample_soda.csv"), "--output", str(output)]
    )
    assert exit_code == 0
    restaurants = pl.read_parquet(output / "restaurants.parquet")
    assert set(restaurants.get_column("jurisdiction").unique().to_list()) == {"nyc"}
