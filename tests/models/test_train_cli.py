"""Tests for scripts/train.py. No production training run is exercised here."""

from __future__ import annotations

from pathlib import Path

import pytest


def _write_nyc_fixture_csv(tmp_path: Path) -> Path:
    import shutil

    src = Path(__file__).parent.parent / "ingestion" / "fixtures" / "nyc_sample_soda.csv"
    dest = tmp_path / "nyc_sample_soda.csv"
    shutil.copy(src, dest)
    return dest


def test_dry_run_does_not_write_artifact(tmp_path: Path) -> None:
    from scripts.train import main

    events_path = _write_nyc_fixture_csv(tmp_path)
    output = tmp_path / "models"
    exit_code = main(
        [
            "--jurisdiction",
            "nyc",
            "--target",
            "nyc_next_initial_score_ge_14",
            "--events",
            str(events_path),
            "--output",
            str(output),
            "--dry-run",
        ]
    )
    assert exit_code == 0
    assert not output.exists()


def test_michelin_flag_rejected() -> None:
    from scripts.train import main

    with pytest.raises(SystemExit):
        main(
            [
                "--jurisdiction",
                "nyc",
                "--target",
                "nyc_next_initial_score_ge_14",
                "--events",
                "x.csv",
                "--output",
                "o",
                "--dry-run",
                "--michelin",
            ]
        )


def test_missing_split_boundaries_rejected_for_production_run(tmp_path: Path) -> None:
    from scripts.train import main

    events_path = _write_nyc_fixture_csv(tmp_path)
    exit_code = main(
        [
            "--jurisdiction",
            "nyc",
            "--target",
            "nyc_next_initial_score_ge_14",
            "--events",
            str(events_path),
            "--output",
            str(tmp_path / "models"),
        ]
    )
    assert exit_code != 0


def test_wrong_target_for_jurisdiction_rejected(tmp_path: Path) -> None:
    from scripts.train import main

    events_path = _write_nyc_fixture_csv(tmp_path)
    exit_code = main(
        [
            "--jurisdiction",
            "nyc",
            "--target",
            "florida_next_routine_high_priority_or_follow_up",
            "--events",
            str(events_path),
            "--output",
            str(tmp_path / "models"),
            "--dry-run",
        ]
    )
    assert exit_code != 0
