"""Shared fixtures for tests/graph. Reuses the top-level ``restaurant_row``/
``inspection_row``/``violation_row`` builders from ``tests/conftest.py`` so
graph fixtures stay in sync with the exact processed-table shape Task 7
already established.
"""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest

from plateproof.graph.models import GraphBuildInput


@pytest.fixture
def make_build_input() -> Any:
    def _make(
        *,
        restaurants: list[dict[str, Any]] | None = None,
        inspections: list[dict[str, Any]] | None = None,
        violations: list[dict[str, Any]] | None = None,
        michelin_restaurants: list[dict[str, Any]] | None = None,
        michelin_distinction_events: list[dict[str, Any]] | None = None,
        restaurant_michelin_matches: list[dict[str, Any]] | None = None,
    ) -> GraphBuildInput:
        def _frame(rows: list[dict[str, Any]] | None) -> pl.DataFrame | None:
            if rows is None:
                return None
            return pl.DataFrame(rows)

        return GraphBuildInput(
            restaurants=_frame(restaurants),
            inspection_events=_frame(inspections),
            violation_events=_frame(violations),
            michelin_restaurants=_frame(michelin_restaurants),
            michelin_distinction_events=_frame(michelin_distinction_events),
            restaurant_michelin_matches=_frame(restaurant_michelin_matches),
        )

    return _make
