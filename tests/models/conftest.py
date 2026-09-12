"""Shared builders for Task 6 model tests.

Nothing here is autouse; nothing touches the network or real production data.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import polars as pl
import pytest

UTC = UTC


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
def synthetic_frame() -> Any:
    def _build(
        n_restaurants: int = 40,
        visits_per_restaurant: int = 5,
        seed: int = 20260101,
        start: Any = None,
    ) -> Any:
        from datetime import date, timedelta

        import numpy as np
        import polars as pl

        from plateproof.models.training import TrainingFrame

        start = start or date(2020, 1, 1)
        rng = np.random.RandomState(seed)
        rows_id, rows_restaurant, rows_date, rows_signal, rows_label = [], [], [], [], []
        counter = 0
        for r in range(n_restaurants):
            restaurant_id = f"nyc:synthetic{r}"
            for _v in range(visits_per_restaurant):
                signal = rng.rand()
                prob = 0.1 + 0.7 * signal
                label = int(rng.rand() < prob)
                rows_id.append(f"syn-{counter}")
                rows_restaurant.append(restaurant_id)
                rows_date.append(start + timedelta(days=counter))
                rows_signal.append(signal)
                rows_label.append(label)
                counter += 1
        noise = rng.rand(counter)
        X = pl.DataFrame(
            {
                "inspection_id": rows_id,
                "signal_feature": rows_signal,
                "noise_feature": noise,
            }
        )
        y = pl.DataFrame({"inspection_id": rows_id, "label": rows_label})
        identity = pl.DataFrame(
            {
                "inspection_id": rows_id,
                "restaurant_id": rows_restaurant,
                "inspection_date": rows_date,
            }
        )
        return TrainingFrame(
            X=X,
            y=y,
            identity=identity,
            feature_order=("signal_feature", "noise_feature"),
            target_name="synthetic_target",
            jurisdiction="nyc",
        )

    return _build


@pytest.fixture
def temporal_result() -> Any:
    def _build(events: pl.DataFrame, violations: pl.DataFrame, cutoff: Any) -> Any:
        from plateproof.features.temporal import build_temporal_features

        return build_temporal_features(events, violations, cutoff)

    return _build
