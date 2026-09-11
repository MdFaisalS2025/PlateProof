"""Jurisdiction-neutral schemas and invariants for inspection and violation events.

This module deliberately contains no jurisdiction-specific normalization and no
feature engineering. It defines the canonical output shape that every
jurisdiction pipeline (NYC, Florida, ...) must conform to, plus one invariant
check. Prediction targets, lag/rolling features, and leakage-sensitive
transformations live elsewhere and are out of scope for ingestion.
"""

from __future__ import annotations

from typing import Any

import polars as pl

INSPECTION_KEY: tuple[str, str, str] = ("restaurant_id", "inspection_date", "inspection_type")

INSPECTION_EVENT_SCHEMA: dict[str, pl.DataType] = {
    "inspection_id": pl.String(),
    "restaurant_id": pl.String(),
    "source_id": pl.String(),
    "jurisdiction": pl.String(),
    "inspection_date": pl.Date(),
    "inspection_type": pl.String(),
    "inspection_type_raw": pl.String(),
    "action": pl.String(),
    "action_conflict": pl.Boolean(),
    "action_conflict_values": pl.List(pl.String()),
    "score": pl.Float64(),
    "score_conflict": pl.Boolean(),
    "score_conflict_values": pl.List(pl.Float64()),
    "grade": pl.String(),
    "grade_conflict": pl.Boolean(),
    "grade_conflict_values": pl.List(pl.String()),
    "grade_date": pl.Date(),
    "violation_count": pl.Int32(),
    "critical_violation_count": pl.Int32(),
    "high_priority_count": pl.Int32(),
    "intermediate_count": pl.Int32(),
    "basic_count": pl.Int32(),
    "dba": pl.String(),
    "boro_raw": pl.String(),
    "building": pl.String(),
    "street": pl.String(),
    "zipcode": pl.String(),
    "cuisine_description": pl.String(),
    "latitude": pl.Float64(),
    "longitude": pl.Float64(),
    "source_dataset": pl.String(),
    "source_snapshot_date": pl.Date(),
    "source_retrieved_at_utc": pl.Datetime("us", "UTC"),
    "source_sha256": pl.String(),
    "ingested_at": pl.Datetime("us", "UTC"),
    "pipeline_version": pl.String(),
    # Optional, jurisdiction-specific fields below. Null unless the source jurisdiction
    # populates them; adding a jurisdiction must never repurpose an existing column.
    "disposition": pl.String(),
    "disposition_status": pl.String(),
    "native_inspection_group_id": pl.String(),
    "native_visit_sequence": pl.Int32(),
}

VIOLATION_EVENT_SCHEMA: dict[str, pl.DataType] = {
    "violation_event_id": pl.String(),
    "inspection_id": pl.String(),
    "restaurant_id": pl.String(),
    "jurisdiction": pl.String(),
    "inspection_date": pl.Date(),
    "violation_code": pl.String(),
    "violation_code_norm": pl.String(),
    "violation_description": pl.String(),
    "violation_description_norm": pl.String(),
    "critical_flag_raw": pl.String(),
    "severity": pl.String(),
    "corrected_on_site": pl.Boolean(),
    # How many times this violation category was cited on this inspection. NYC's
    # extract is one row per physical citation, so it is always 1; Florida's extract
    # is a count matrix, so it carries the source count directly.
    "count": pl.Int32(),
    "source_dataset": pl.String(),
    "source_snapshot_date": pl.Date(),
    "source_retrieved_at_utc": pl.Datetime("us", "UTC"),
    "source_sha256": pl.String(),
    "ingested_at": pl.Datetime("us", "UTC"),
    "pipeline_version": pl.String(),
}


def assert_unique_inspection_key(events: pl.DataFrame) -> None:
    """Verify the output invariant: one row per ``INSPECTION_KEY`` in ``events``.

    This checks a post-aggregation property only. The pipeline groups source rows
    by that key before this runs, so every source row sharing the key has already
    been merged into a single published inspection event. This function therefore
    cannot detect whether two genuinely distinct same-type inspections occurred on
    one calendar day; such ambiguity is surfaced through the ``*_conflict``
    columns instead. It raises only if aggregation failed to collapse a key.
    """
    duplicates = events.group_by(list(INSPECTION_KEY)).len().filter(pl.col("len") > 1).drop("len")
    if duplicates.height:
        raise ValueError(
            "inspection_events violates the one-row-per-"
            f"{INSPECTION_KEY} invariant for: {duplicates.rows()}"
        )


def assert_unique_inspection_id(events: pl.DataFrame) -> None:
    """Verify one row per ``inspection_id``.

    This is the correct primary-key invariant for any jurisdiction, whether
    ``inspection_id`` is synthesized from ``INSPECTION_KEY`` (NYC, where the two
    checks are equivalent) or native to the source (Florida's Inspection Visit ID,
    where two rows can legitimately share ``INSPECTION_KEY`` while still being
    distinct, genuinely separate visits). Raises ValueError listing the offending
    ids if any repeats.
    """
    duplicates = events.group_by("inspection_id").len().filter(pl.col("len") > 1).drop("len")
    if duplicates.height:
        raise ValueError(
            "inspection_events violates the one-row-per-inspection_id invariant for: "
            f"{duplicates.get_column('inspection_id').to_list()}"
        )


def finalize_event_frame(
    records: list[dict[str, Any]], schema: dict[str, pl.DataType]
) -> pl.DataFrame:
    """Build a typed frame from ``records``, filling any schema key absent from
    every record with null instead of requiring each jurisdiction's ingestion code
    to enumerate keys it has nothing to contribute for."""
    keys = list(schema)
    present = {key for record in records for key in record}
    missing = [key for key in keys if key not in present]
    if missing and records:
        records = [{**{key: None for key in missing}, **record} for record in records]
    return pl.DataFrame(records, schema=schema)
