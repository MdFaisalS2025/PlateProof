"""Tests for scripts/train.py. No production data is used here -- the full
pipeline is only ever exercised against a small, fully synthetic fixture."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


def _write_nyc_fixture_csv(tmp_path: Path) -> Path:
    import shutil

    src = Path(__file__).parent.parent / "ingestion" / "fixtures" / "nyc_sample_soda.csv"
    dest = tmp_path / "nyc_sample_soda.csv"
    shutil.copy(src, dest)
    return dest


def _write_nyc_ready_fixture_csv(tmp_path: Path) -> Path:
    """16 restaurants, each with 3 prior visits plus one qualifying initial
    inspection, alternating scores -- both classes present in every partition
    when split at the boundaries used by the tests below. Small and fully
    synthetic (not downloaded production data)."""
    import shutil

    src = Path(__file__).parent / "fixtures" / "nyc_cli_ready_sample.csv"
    dest = tmp_path / "nyc_cli_ready_sample.csv"
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


def _run_full_pipeline(
    tmp_path: Path,
    *,
    train_end: str,
    validation_end: str,
    test_end: str,
    extra_args: list[str] | None = None,
) -> tuple[int, Path]:
    from scripts.train import main

    events_path = _write_nyc_ready_fixture_csv(tmp_path)
    output = tmp_path / "models"
    args = [
        "--jurisdiction",
        "nyc",
        "--target",
        "nyc_next_initial_score_ge_14",
        "--events",
        str(events_path),
        "--output",
        str(output),
        "--train-end",
        train_end,
        "--validation-end",
        validation_end,
        "--test-end",
        test_end,
        # strictly after the fixture's last event date, so the CLI's default
        # cutoff (= max event date, exclusive) never drops the last event
        "--cutoff-date",
        "2024-09-01",
    ]
    exit_code = main(args + (extra_args or []))
    return exit_code, output


def test_empty_test_partition_is_rejected_by_the_cli(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(ValueError, match="test"):
        _run_full_pipeline(
            tmp_path, train_end="2024-05-01", validation_end="2024-08-01", test_end="2024-08-01"
        )


def test_single_class_train_is_rejected_by_the_cli(tmp_path: Path) -> None:
    """Restaurant 0's initial inspection (label 0) is the only qualifying
    inspection in this train window -- the corrected chronological_split
    must reject a single-row, single-class train partition even via the
    real CLI path."""
    import pytest

    with pytest.raises(ValueError, match="train"):
        _run_full_pipeline(
            tmp_path, train_end="2024-02-01", validation_end="2024-07-20", test_end="2024-08-01"
        )


def test_cli_artifact_is_complete_and_honestly_non_ready_when_uncalibrated(
    tmp_path: Path, capsys: Any
) -> None:
    """This fixture is far too small to reach the calibration row-count
    minimum (30 validation rows) -- exactly the honest non-ready case Task 6
    must handle correctly: the artifact is still complete and fully
    documented, but is not silently called production-ready anywhere."""
    from plateproof.models.training import load_artifact

    exit_code, output_root = _run_full_pipeline(
        tmp_path, train_end="2024-05-01", validation_end="2024-07-01", test_end="2024-08-01"
    )
    assert exit_code == 0
    console_output = capsys.readouterr().out
    assert "NOT approved for production" in console_output
    assert "uncalibrated" in console_output

    artifact_dir = output_root / "nyc" / "nyc_next_initial_score_ge_14" / "v1"

    # risk scoring must reject it by default
    with pytest.raises(ValueError, match="non-ready"):
        load_artifact(artifact_dir, trusted=True)

    # the explicit audit/evaluation override still works
    loaded = load_artifact(
        artifact_dir, trusted=True, expected_jurisdiction="nyc", require_ready=False
    )
    assert loaded["deployment_status.json"]["status"] == "uncalibrated"

    # the model card clearly states the reason it is not approved
    assert "not approved for production" in loaded["model_card.md"].lower()
    assert "uncalibrated" in loaded["model_card.md"].lower()

    # risk thresholds included
    assert loaded["risk_band_thresholds.json"]["jurisdiction"] == "nyc"
    assert loaded["history_sufficiency_rule.json"]["jurisdiction"] == "nyc"

    # feature schema and dtypes included
    from plateproof.models.nyc_risk import NYC_FEATURE_LIST

    assert loaded["feature_order.json"] == list(NYC_FEATURE_LIST)
    assert set(loaded["feature_dtypes.json"]) == set(NYC_FEATURE_LIST)

    # bootstrap metadata included (not just anonymous estimators)
    members = loaded["bootstrap_members.joblib"]
    assert all("member_index" in m and "seed" in m for m in members)

    # source provenance included or explicitly marked unavailable
    provenance = loaded["source_provenance.json"]
    assert provenance["source_snapshot_date"] != ""
    assert all(v for v in provenance.values())


def test_validate_only_runs_the_full_pipeline_but_writes_nothing(tmp_path: Path) -> None:
    """--validate-only must run selection, calibration, and bootstrapping (so
    the console summary is meaningful) without ever writing an artifact."""
    exit_code, output_root = _run_full_pipeline(
        tmp_path,
        train_end="2024-05-01",
        validation_end="2024-07-01",
        test_end="2024-08-01",
        extra_args=["--validate-only"],
    )
    assert exit_code == 0
    artifact_dir = output_root / "nyc" / "nyc_next_initial_score_ge_14" / "v1"
    assert not artifact_dir.exists()  # validate-only never writes


def test_forced_replacement_failure_preserves_previous_cli_artifact(
    tmp_path: Path, monkeypatch: Any
) -> None:
    import pytest

    from plateproof.models import training as training_module
    from plateproof.models.training import load_artifact

    exit_code, output_root = _run_full_pipeline(
        tmp_path, train_end="2024-05-01", validation_end="2024-07-01", test_end="2024-08-01"
    )
    assert exit_code == 0
    artifact_dir = output_root / "nyc" / "nyc_next_initial_score_ge_14" / "v1"
    original_card = (artifact_dir / "model_card.md").read_text(encoding="utf-8")

    real_serialize_one = training_module._serialize_one
    call_count = {"n": 0}

    def _flaky_serialize_one(path: Any, value: Any, *, final_name: str) -> None:
        call_count["n"] += 1
        if call_count["n"] > 2:
            raise RuntimeError("simulated disk failure")
        real_serialize_one(path, value, final_name=final_name)

    monkeypatch.setattr(training_module, "_serialize_one", _flaky_serialize_one)

    with pytest.raises(RuntimeError, match="simulated disk failure"):
        _run_full_pipeline(
            tmp_path,
            train_end="2024-05-01",
            validation_end="2024-07-01",
            test_end="2024-08-01",
            extra_args=["--force"],
        )

    loaded = load_artifact(artifact_dir, trusted=True, require_ready=False)
    assert loaded["model_card.md"] == original_card
