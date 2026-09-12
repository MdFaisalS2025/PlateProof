"""Offline scoring CLI: the only process allowed to deserialize or run a
Task 6 model artifact. Writes ``predictions.parquet`` and
``model_registry.parquet`` for the FastAPI/Streamlit read-only serving layer.

Never downloads data and never accepts an implicit "today" as-of date --
``--as-of-date`` is required on every invocation.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import polars as pl

from plateproof.serving.scoring import run_offline_scoring, write_predictions_and_registry

_PRIMARY_TARGETS: dict[str, str] = {
    "nyc": "nyc_next_initial_score_ge_14",
    "florida": "florida_next_routine_high_priority_or_follow_up",
}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline-score every restaurant in --events as of --as-of-date."
    )
    parser.add_argument("--jurisdiction", required=True, choices=["nyc", "florida"])
    parser.add_argument(
        "--target",
        required=False,
        help="Defaults to the jurisdiction's primary target if omitted.",
    )
    parser.add_argument("--artifact-path", required=True, help="Path to a ready Task 6 artifact.")
    parser.add_argument("--events", required=True, help="Path to a Parquet inspection_events file.")
    parser.add_argument(
        "--violations", required=True, help="Path to a Parquet violation_events file."
    )
    parser.add_argument(
        "--as-of-date",
        required=True,
        type=date.fromisoformat,
        help="Required explicit as-of date (YYYY-MM-DD). Never defaults to today.",
    )
    parser.add_argument("--output", required=True, help="Processed-data directory to write into.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    target_name = args.target or _PRIMARY_TARGETS[args.jurisdiction]

    events = pl.read_parquet(args.events)
    violations = pl.read_parquet(args.violations)

    result = run_offline_scoring(
        artifact_path=Path(args.artifact_path),
        jurisdiction=args.jurisdiction,
        target_name=target_name,
        events=events,
        violations=violations,
        as_of_date=args.as_of_date,
    )
    write_predictions_and_registry(Path(args.output), result.predictions, result.registry_entry)
    print(
        f"scored {result.report.restaurant_count} restaurants "
        f"({result.report.sufficient_history_count} with sufficient history, "
        f"{result.report.insufficient_history_count} insufficient) "
        f"as of {args.as_of_date.isoformat()}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
