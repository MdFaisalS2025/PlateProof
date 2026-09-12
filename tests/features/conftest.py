"""Shared builders for temporal-feature tests.

Nothing here is autouse; nothing touches the network or real data files.
Builders reuse ``finalize_event_frame`` so fixtures stay schema-consistent
with the real NYC/Florida ingestion output.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import polars as pl
import pytest


def _event_defaults() -> dict[str, Any]:
    return {
        "inspection_id": None,
        "restaurant_id": None,
        "source_id": None,
        "jurisdiction": None,
        "inspection_date": None,
        "inspection_type": "Routine",
        "inspection_type_raw": "Routine",
        "action": None,
        "action_conflict": False,
        "action_conflict_values": None,
        "score": None,
        "score_conflict": False,
        "score_conflict_values": None,
        "grade": None,
        "grade_conflict": False,
        "grade_conflict_values": None,
        "grade_date": None,
        "violation_count": None,
        "critical_violation_count": None,
        "high_priority_count": None,
        "intermediate_count": None,
        "basic_count": None,
        "dba": "Test Restaurant",
        "boro_raw": None,
        "building": None,
        "street": None,
        "zipcode": None,
        "cuisine_description": None,
        "latitude": None,
        "longitude": None,
        "source_dataset": "test",
        "source_snapshot_date": None,
        "source_retrieved_at_utc": None,
        "source_sha256": None,
        "ingested_at": datetime(2024, 1, 1, tzinfo=UTC),
        "pipeline_version": "test",
        "disposition": None,
        "disposition_status": None,
        "native_inspection_group_id": None,
        "native_visit_sequence": None,
    }


def _violation_defaults() -> dict[str, Any]:
    return {
        "violation_event_id": None,
        "inspection_id": None,
        "restaurant_id": None,
        "jurisdiction": None,
        "inspection_date": None,
        "violation_code": None,
        "violation_code_norm": None,
        "violation_description": None,
        "violation_description_norm": None,
        "critical_flag_raw": None,
        "severity": None,
        "corrected_on_site": None,
        "count": 1,
        "source_dataset": "test",
        "source_snapshot_date": None,
        "source_retrieved_at_utc": None,
        "source_sha256": None,
        "ingested_at": datetime(2024, 1, 1, tzinfo=UTC),
        "pipeline_version": "test",
    }


@pytest.fixture
def make_event() -> Any:
    def _make(**overrides: Any) -> dict[str, Any]:
        row = _event_defaults()
        row.update(overrides)
        if row["inspection_id"] is None:
            row["inspection_id"] = f"{row['restaurant_id']}:{row['inspection_date']}:{id(row)}"
        if row["source_id"] is None and row["restaurant_id"]:
            row["source_id"] = str(row["restaurant_id"]).split(":", 1)[-1]
        return row

    return _make


@pytest.fixture
def make_violation() -> Any:
    def _make(**overrides: Any) -> dict[str, Any]:
        row = _violation_defaults()
        row.update(overrides)
        if row["violation_code_norm"] is None and row["violation_code"]:
            row["violation_code_norm"] = str(row["violation_code"]).casefold()
        if row["violation_event_id"] is None:
            row["violation_event_id"] = f"{row['inspection_id']}:{row['violation_code']}"
        return row

    return _make


@pytest.fixture
def events_frame() -> Any:
    def _frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
        from plateproof.features.inspection_events import (
            INSPECTION_EVENT_SCHEMA,
            finalize_event_frame,
        )

        return finalize_event_frame(rows, INSPECTION_EVENT_SCHEMA)

    return _frame


@pytest.fixture
def violations_frame() -> Any:
    def _frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
        from plateproof.features.inspection_events import (
            VIOLATION_EVENT_SCHEMA,
            finalize_event_frame,
        )

        return finalize_event_frame(rows, VIOLATION_EVENT_SCHEMA)

    return _frame


@pytest.fixture
def d() -> Any:
    """date(...) shorthand for compact test fixtures."""
    return date
