"""Tests for build_graph_from_processed_dir and GraphService's
fingerprint-based cache invalidation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import polars as pl


def test_build_graph_from_processed_dir_tolerates_missing_tables(tmp_path: Path) -> None:
    from plateproof.graph.builder import build_graph_from_processed_dir

    result = build_graph_from_processed_dir(tmp_path)
    assert result.graph.number_of_nodes == 0


def test_build_graph_from_processed_dir_reads_available_tables(
    tmp_path: Path, restaurant_row: Any
) -> None:
    from plateproof.graph.builder import build_graph_from_processed_dir

    pl.DataFrame([restaurant_row(restaurant_id="nyc:1")]).write_parquet(
        tmp_path / "restaurants.parquet"
    )
    result = build_graph_from_processed_dir(tmp_path)
    assert result.graph.has_node("nyc:1")


def test_build_graph_from_processed_dir_tolerates_a_corrupted_file(tmp_path: Path) -> None:
    from plateproof.graph.builder import build_graph_from_processed_dir

    (tmp_path / "restaurants.parquet").write_bytes(b"not a real parquet file")
    result = build_graph_from_processed_dir(tmp_path)
    assert result.graph.number_of_nodes == 0


def test_graph_service_rebuilds_only_when_fingerprint_changes(
    tmp_path: Path, restaurant_row: Any, monkeypatch: Any
) -> None:
    from plateproof.graph import builder as builder_module
    from plateproof.graph.builder import GraphService

    pl.DataFrame([restaurant_row(restaurant_id="nyc:1")]).write_parquet(
        tmp_path / "restaurants.parquet"
    )

    call_count = 0
    original = builder_module.build_graph_from_processed_dir

    def _counting_build(*args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(builder_module, "build_graph_from_processed_dir", _counting_build)
    service = GraphService(tmp_path)

    first = service.get()
    second = service.get()
    assert call_count == 1  # unchanged fingerprint -- no rebuild on the second access
    assert first is second

    time.sleep(0.01)
    pl.DataFrame([restaurant_row(restaurant_id="nyc:2")]).write_parquet(
        tmp_path / "restaurants.parquet"
    )
    third = service.get()
    assert call_count == 2  # file changed -- rebuilt
    assert third.graph.has_node("nyc:2")


def test_compute_source_fingerprint_reflects_missing_files(tmp_path: Path) -> None:
    from plateproof.graph.builder import compute_source_fingerprint

    fingerprint = compute_source_fingerprint(tmp_path)
    assert all(mtime == -1 and size == -1 for _, mtime, size in fingerprint.entries)
