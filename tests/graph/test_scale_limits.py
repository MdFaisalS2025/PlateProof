"""Correctness tests for incremental graph scale-limit enforcement
(independent-review correction item 6) -- kept separate from
test_scale_benchmark.py's below-limit performance benchmark.

These prove construction *stops* near the configured boundary instead of
processing the entire oversized input first and checking afterward, that
a failed build never leaves a partial graph cached, and that repeated
references (not genuinely new nodes) never consume capacity.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from plateproof.graph import builder as builder_module
from plateproof.graph.builder import GraphService, build_graph
from plateproof.graph.models import GraphBuildInput, GraphScaleExceededError, GraphScaleLimits

_ROW_COUNT = 10_000


def _many_restaurants(restaurant_row: Any) -> pl.DataFrame:
    # All rows share the same default city/postal_code/cuisine, so only
    # the restaurant node itself grows per row -- location/cuisine nodes
    # are created once and reused.
    return pl.DataFrame([restaurant_row(restaurant_id=f"nyc:{i}") for i in range(_ROW_COUNT)])


def test_node_limit_stops_construction_before_processing_all_rows(
    restaurant_row: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    original = builder_module._add_node_with_conflict_check

    def _counting(*args: Any, **kwargs: Any) -> None:
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(builder_module, "_add_node_with_conflict_check", _counting)

    build_input = GraphBuildInput(
        restaurants=_many_restaurants(restaurant_row),
        inspection_events=None,
        violation_events=None,
        michelin_restaurants=None,
        michelin_distinction_events=None,
        restaurant_michelin_matches=None,
    )
    with pytest.raises(GraphScaleExceededError):
        build_graph(build_input, limits=GraphScaleLimits(max_nodes=5, max_edges=1_000_000))

    assert len(calls) < 20, f"processed {len(calls)} rows out of {_ROW_COUNT} before raising"


def test_edge_limit_stops_construction_before_processing_all_rows(
    restaurant_row: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    original = builder_module._add_node_with_conflict_check

    def _counting(*args: Any, **kwargs: Any) -> None:
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(builder_module, "_add_node_with_conflict_check", _counting)

    build_input = GraphBuildInput(
        restaurants=_many_restaurants(restaurant_row),
        inspection_events=None,
        violation_events=None,
        michelin_restaurants=None,
        michelin_distinction_events=None,
        restaurant_michelin_matches=None,
    )
    # A generous node limit but a tiny edge limit: each restaurant after
    # the first contributes 2 new edges (LOCATED_IN, HAS_CUISINE) since
    # the shared location/cuisine nodes are created once.
    with pytest.raises(GraphScaleExceededError, match="edge"):
        build_graph(build_input, limits=GraphScaleLimits(max_nodes=1_000_000, max_edges=5))

    assert len(calls) < 20, f"processed {len(calls)} rows out of {_ROW_COUNT} before raising"


def test_repeated_node_reference_never_consumes_new_capacity(
    restaurant_row: Any, inspection_row: Any
) -> None:
    """Two inspections for the SAME restaurant must not require two units
    of node capacity for the restaurant itself -- only the genuinely new
    inspection nodes count."""
    build_input = GraphBuildInput(
        restaurants=pl.DataFrame([restaurant_row(restaurant_id="nyc:1")]),
        inspection_events=pl.DataFrame(
            [
                inspection_row(
                    inspection_id="nyc:1:1", restaurant_id="nyc:1", inspection_date=date(2024, 1, 1)
                ),
                inspection_row(
                    inspection_id="nyc:1:2", restaurant_id="nyc:1", inspection_date=date(2025, 1, 1)
                ),
            ]
        ),
        violation_events=None,
        michelin_restaurants=None,
        michelin_distinction_events=None,
        restaurant_michelin_matches=None,
    )
    # 1 restaurant + 1 location + 1 cuisine + 2 inspections = 5 nodes.
    # If the restaurant node were miscounted as consuming capacity twice
    # (once per inspection referencing it), this would need 6.
    result = build_graph(build_input, limits=GraphScaleLimits(max_nodes=5, max_edges=1_000_000))
    assert result.report.node_count == 5


def test_failed_build_never_caches_a_partial_graph(
    tmp_path: Path, restaurant_row: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    pl.DataFrame([restaurant_row(restaurant_id="nyc:1")]).write_parquet(
        tmp_path / "restaurants.parquet"
    )
    service = GraphService(
        tmp_path, limits=GraphScaleLimits(max_nodes=1_000_000, max_edges=1_000_000)
    )
    good_result = service.get()
    assert good_result.graph.has_node("nyc:1")

    def _always_fails(*args: Any, **kwargs: Any) -> Any:
        raise GraphScaleExceededError("simulated failure")

    monkeypatch.setattr(builder_module, "build_graph_from_processed_dir", _always_fails)
    # Force a fingerprint change so .get() actually attempts a rebuild
    # rather than returning the cached result without calling the builder.
    pl.DataFrame(
        [restaurant_row(restaurant_id="nyc:1"), restaurant_row(restaurant_id="nyc:2")]
    ).write_parquet(tmp_path / "restaurants.parquet")

    with pytest.raises(GraphScaleExceededError):
        service.get()

    # The previous good result must still be exactly what a caller gets
    # from internal state -- never a partially-built replacement.
    assert service._cached is good_result  # noqa: SLF001


def test_below_limit_graph_remains_deterministic_under_shuffle(
    restaurant_row: Any, inspection_row: Any
) -> None:
    """Sanity check that incremental limit-checking didn't change the
    final graph's shape or its order-independence -- the fuller
    determinism proof lives in test_builder.py."""
    import random

    restaurants = [restaurant_row(restaurant_id=f"nyc:{i}") for i in range(20)]
    shuffled = restaurants[:]
    random.Random(7).shuffle(shuffled)

    limits = GraphScaleLimits(max_nodes=1_000_000, max_edges=1_000_000)
    result_a = build_graph(
        GraphBuildInput(
            restaurants=pl.DataFrame(restaurants),
            inspection_events=None,
            violation_events=None,
            michelin_restaurants=None,
            michelin_distinction_events=None,
            restaurant_michelin_matches=None,
        ),
        limits=limits,
    )
    result_b = build_graph(
        GraphBuildInput(
            restaurants=pl.DataFrame(shuffled),
            inspection_events=None,
            violation_events=None,
            michelin_restaurants=None,
            michelin_distinction_events=None,
            restaurant_michelin_matches=None,
        ),
        limits=limits,
    )
    assert result_a.report.node_count == result_b.report.node_count
    assert result_a.report.edge_count == result_b.report.edge_count
