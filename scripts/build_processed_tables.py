"""Build Task 7's processed Parquet tables from already-downloaded Task 2/3/4
outputs. Never downloads anything -- every input is an explicit local path.

Writes: ``restaurants.parquet``, ``inspection_events.parquet``,
``violation_events.parquet``, and (only when a Michelin seed is given)
``michelin_restaurants.parquet`` / ``michelin_distinction_events.parquet``.

This script deliberately does **not** produce
``restaurant_michelin_matches.parquet``: that table is the output of Task 4's
auditable entity-resolution matcher (``plateproof.matching.entity_resolution``)
run against a specific restaurant snapshot plus a human reviewer's decisions
on the review-queue range -- it is not something to (re-)run unattended
inside a table-build script. Populate it separately once matches have been
reviewed.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from plateproof.serving.processed_tables import build_florida_restaurants, build_nyc_restaurants


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build processed restaurant/inspection/violation Parquet tables "
        "from already-downloaded NYC/Florida/Michelin extracts."
    )
    parser.add_argument("--nyc-events", help="Path to a raw NYC DOHMH extract CSV.")
    parser.add_argument(
        "--florida-events", nargs="*", default=[], help="Path(s) to raw Florida DBPR extracts."
    )
    parser.add_argument("--michelin-seed", help="Path to the hand-curated Michelin seed CSV.")
    parser.add_argument("--output", required=True, help="Processed-data output directory.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)

    restaurants_frames = []
    inspection_frames = []
    violation_frames = []

    if args.nyc_events:
        from plateproof.ingestion.nyc import build_nyc_inspection_events, load_nyc_raw

        raw = load_nyc_raw(args.nyc_events)
        result = build_nyc_inspection_events(raw, ingested_at=now)
        nyc_restaurants, nyc_report = build_nyc_restaurants(result.inspection_events)
        print(
            f"NYC: {nyc_report.restaurant_count} restaurants "
            f"({len(nyc_report.conflicting_restaurant_ids)} same-date conflicts)"
        )
        restaurants_frames.append(nyc_restaurants)
        inspection_frames.append(result.inspection_events)
        violation_frames.append(result.violation_events)

    if args.florida_events:
        from plateproof.ingestion.florida import (
            FloridaExtractSource,
            build_florida_inspection_events,
            load_florida_extracts,
        )

        sources = [FloridaExtractSource(path=Path(p)) for p in args.florida_events]
        raw = load_florida_extracts(sources)
        fl_result = build_florida_inspection_events(raw, ingested_at=now)
        fl_restaurants, fl_report = build_florida_restaurants(fl_result.inspection_events, raw)
        print(
            f"Florida: {fl_report.restaurant_count} restaurants "
            f"({len(fl_report.conflicting_restaurant_ids)} same-date conflicts)"
        )
        restaurants_frames.append(fl_restaurants)
        inspection_frames.append(fl_result.inspection_events)
        violation_frames.append(fl_result.violation_events)

    if restaurants_frames:
        import polars as pl

        pl.concat(restaurants_frames).write_parquet(output_dir / "restaurants.parquet")
        pl.concat(inspection_frames).write_parquet(output_dir / "inspection_events.parquet")
        pl.concat(violation_frames).write_parquet(output_dir / "violation_events.parquet")

    if args.michelin_seed:
        import polars as pl

        from plateproof.ingestion.michelin import load_michelin_seed

        michelin_result = load_michelin_seed(args.michelin_seed)
        pl.DataFrame([r.model_dump() for r in michelin_result.restaurants]).write_parquet(
            output_dir / "michelin_restaurants.parquet"
        )
        pl.DataFrame([e.model_dump() for e in michelin_result.distinction_events]).write_parquet(
            output_dir / "michelin_distinction_events.parquet"
        )
        print(
            f"Michelin: {michelin_result.report.restaurant_count} restaurants, "
            f"{michelin_result.report.distinction_event_count} distinction events"
        )

    print(str(output_dir))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
