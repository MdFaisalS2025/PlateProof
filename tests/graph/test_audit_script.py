"""Tests for scripts/audit_graph.py -- called directly via main(), never
subprocess, matching the project's existing CLI test pattern."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl


def test_audit_passes_for_a_clean_graph(tmp_path: Path, restaurant_row: Any, capsys: Any) -> None:
    from scripts.audit_graph import main

    pl.DataFrame([restaurant_row(restaurant_id="nyc:1")]).write_parquet(
        tmp_path / "restaurants.parquet"
    )
    exit_code = main(["--processed-data-dir", str(tmp_path)])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "PASS" in captured.out


def test_audit_fails_for_a_graph_with_conflicts(
    tmp_path: Path, restaurant_row: Any, capsys: Any
) -> None:
    from scripts.audit_graph import main

    pl.DataFrame(
        [
            restaurant_row(restaurant_id="nyc:1", name="Anna's Kitchen"),
            restaurant_row(restaurant_id="nyc:1", name="Anna's Kitchen And Bar"),
        ]
    ).write_parquet(tmp_path / "restaurants.parquet")
    exit_code = main(["--processed-data-dir", str(tmp_path)])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "FAIL" in captured.out


def test_audit_reports_scale_exceeded(tmp_path: Path, restaurant_row: Any, capsys: Any) -> None:
    from scripts.audit_graph import main

    pl.DataFrame([restaurant_row(restaurant_id=f"nyc:{i}") for i in range(5)]).write_parquet(
        tmp_path / "restaurants.parquet"
    )
    exit_code = main(["--processed-data-dir", str(tmp_path), "--max-nodes", "3"])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "scale_exceeded" in captured.out
