"""Representative-scale timing/memory benchmark for plateproof.graph.builder.
Uses ``tracemalloc`` (stdlib, cross-platform) rather than ``resource``
(unavailable on Windows). Marked ``benchmark`` -- still collected and run by
default (nothing in this project's config deselects markers), but callers
who want a fast subset can run ``pytest -m "not benchmark"``.
"""

from __future__ import annotations

import time
import tracemalloc
from datetime import date, timedelta
from typing import Any

import polars as pl
import pytest

from plateproof.graph.models import GraphScaleLimits

_N_RESTAURANTS = 500
_INSPECTIONS_PER_RESTAURANT = 10
_VIOLATIONS_PER_INSPECTION = 3


def _synthetic_frames() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    restaurants: list[dict[str, Any]] = []
    inspections: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    start = date(2020, 1, 1)
    for r in range(_N_RESTAURANTS):
        restaurant_id = f"nyc:{r}"
        restaurants.append(
            {
                "restaurant_id": restaurant_id,
                "jurisdiction": "nyc",
                "source_id": str(r),
                "name": f"Synthetic {r}",
                "normalized_name": f"synthetic {r}",
                "city": "Manhattan",
                "region": "NY",
                "postal_code": "10001",
                "cuisine": "American",
            }
        )
        for i in range(_INSPECTIONS_PER_RESTAURANT):
            inspection_id = f"{restaurant_id}:{i}"
            inspection_date = start + timedelta(days=i * 30)
            inspections.append(
                {
                    "inspection_id": inspection_id,
                    "restaurant_id": restaurant_id,
                    "jurisdiction": "nyc",
                    "inspection_date": inspection_date,
                    "inspection_type": "Cycle Inspection",
                    "score": 10.0,
                    "grade": "A",
                    "high_priority_count": 0,
                    "intermediate_count": 0,
                    "basic_count": 0,
                    "critical_violation_count": 1,
                }
            )
            for v in range(_VIOLATIONS_PER_INSPECTION):
                violations.append(
                    {
                        "violation_event_id": f"{inspection_id}:v{v}",
                        "inspection_id": inspection_id,
                        "restaurant_id": restaurant_id,
                        "jurisdiction": "nyc",
                        "inspection_date": inspection_date,
                        "violation_code": f"0{v}L",
                        "violation_code_norm": f"0{v}L",
                        "violation_description": "Synthetic violation",
                        "severity": "critical",
                        "count": 1,
                        "corrected_on_site": False,
                    }
                )
    return pl.DataFrame(restaurants), pl.DataFrame(inspections), pl.DataFrame(violations)


@pytest.mark.benchmark
def test_build_graph_scales_within_time_and_memory_bounds() -> None:
    from plateproof.graph.builder import build_graph
    from plateproof.graph.models import GraphBuildInput

    restaurants, inspections, violations = _synthetic_frames()
    build_input = GraphBuildInput(
        restaurants=restaurants,
        inspection_events=inspections,
        violation_events=violations,
        michelin_restaurants=None,
        michelin_distinction_events=None,
        restaurant_michelin_matches=None,
    )

    tracemalloc.start()
    start = time.perf_counter()
    result = build_graph(
        build_input, limits=GraphScaleLimits(max_nodes=1_000_000, max_edges=5_000_000)
    )
    elapsed = time.perf_counter() - start
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    expected_inspections = _N_RESTAURANTS * _INSPECTIONS_PER_RESTAURANT
    expected_violations = expected_inspections * _VIOLATIONS_PER_INSPECTION
    assert result.report.node_counts_by_type["restaurant"] == _N_RESTAURANTS
    assert result.report.node_counts_by_type["inspection"] == expected_inspections
    assert result.report.node_counts_by_type["violation_occurrence"] == expected_violations

    # Generous bounds for CI/dev-machine variance -- these exist to catch a
    # future accidental O(n^2) regression, not to certify production-scale
    # (full-NYC) performance, which Task 8A does not claim. The bound must
    # tolerate running under `pytest --cov`, which line-traces every call
    # and was observed to add ~4x wall-clock overhead to this test alone.
    assert elapsed < 60.0, f"build took {elapsed:.2f}s for {_N_RESTAURANTS} restaurants"
    assert peak < 1_500_000_000, f"peak traced memory was {peak} bytes"
