"""Tests for scripts/score_predictions.py. Small fictional fixtures only."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def test_as_of_date_is_required(capsys: Any) -> None:
    import pytest

    from scripts.score_predictions import main

    with pytest.raises(SystemExit):
        main(
            [
                "--jurisdiction",
                "nyc",
                "--artifact-path",
                "x",
                "--events",
                "e.parquet",
                "--violations",
                "v.parquet",
                "--output",
                "o",
            ]
        )
    assert "as-of-date" in capsys.readouterr().err.lower()


def test_end_to_end_cli_scoring(tmp_path: Path, build_ready_artifact: Any) -> None:
    from datetime import UTC, date, datetime

    import polars as pl

    from plateproof.features.inspection_events import (
        INSPECTION_EVENT_SCHEMA,
        VIOLATION_EVENT_SCHEMA,
        finalize_event_frame,
    )
    from plateproof.models.nyc_risk import NYC_FEATURE_LIST
    from scripts.score_predictions import main

    artifact = build_ready_artifact(
        tmp_path,
        jurisdiction="nyc",
        target_name="nyc_next_initial_score_ge_14",
        feature_order=NYC_FEATURE_LIST,
    )

    event = {
        "inspection_id": "nyc:cli1:2024-01-01",
        "restaurant_id": "nyc:cli1",
        "source_id": "cli1",
        "jurisdiction": "nyc",
        "inspection_date": date(2024, 1, 1),
        "inspection_type": "Cycle Inspection / Initial Inspection",
        "score": 10.0,
        "score_conflict": False,
        "critical_violation_count": 0,
        "dba": "CLI Test Restaurant",
        "source_dataset": "test",
        "source_snapshot_date": date(2026, 1, 1),
        "source_retrieved_at_utc": datetime(2026, 1, 1, tzinfo=UTC),
        "ingested_at": datetime(2026, 1, 1, tzinfo=UTC),
        "pipeline_version": "test",
    }
    events_path = tmp_path / "events.parquet"
    violations_path = tmp_path / "violations.parquet"
    finalize_event_frame([event], INSPECTION_EVENT_SCHEMA).write_parquet(events_path)
    finalize_event_frame([], VIOLATION_EVENT_SCHEMA).write_parquet(violations_path)

    output_dir = tmp_path / "processed"
    exit_code = main(
        [
            "--jurisdiction",
            "nyc",
            "--artifact-path",
            str(artifact),
            "--events",
            str(events_path),
            "--violations",
            str(violations_path),
            "--as-of-date",
            "2025-01-01",
            "--output",
            str(output_dir),
        ]
    )
    assert exit_code == 0
    predictions = pl.read_parquet(output_dir / "predictions.parquet")
    assert predictions.height == 1
    registry = pl.read_parquet(output_dir / "model_registry.parquet")
    assert registry.filter(pl.col("jurisdiction") == "nyc").height == 1
