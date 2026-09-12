"""Build Task 7's processed ``restaurants`` table from Task 2/3 outputs.

This module never downloads data and never re-derives an official identity
relationship that Task 2/3 already established -- it only reshapes their
already-produced ``inspection_events`` frames (and, for Florida, the Task 3
staging frame) into one row per restaurant.

NYC restaurant identity is fully derivable from ``inspection_events`` alone:
every descriptive field (name, borough, street, zip, cuisine, coordinates)
is already carried on each NYC event row. Florida's full street address and
city are **not** on ``inspection_events`` -- Task 3 (see
``plateproof.ingestion.florida``) deliberately keeps them only on its
auditable staging frame (the ``raw`` output of ``load_florida_extracts``).
Florida restaurants are therefore built by joining that staging frame back
to the normalized events on the exact ``inspection_id`` relationship Task 3
already computed (``florida:<inspection_visit_id>``), never by
re-deriving ``florida:<license_number>`` independently and never by
assuming two license numbers that merely look similar are the same
restaurant.
"""

from __future__ import annotations

from typing import Literal

import polars as pl
from pydantic import BaseModel, ConfigDict

RESTAURANT_SCHEMA: dict[str, pl.DataType] = {
    "restaurant_id": pl.String(),
    "jurisdiction": pl.String(),
    "source_id": pl.String(),
    "name": pl.String(),
    "normalized_name": pl.String(),
    "address": pl.String(),
    "city": pl.String(),
    "region": pl.String(),
    "postal_code": pl.String(),
    "latitude": pl.Float64(),
    "longitude": pl.Float64(),
    "cuisine": pl.String(),
    "latest_inspection_date": pl.Date(),
    "source_snapshot_date": pl.Date(),
    "source_retrieved_at_utc": pl.Datetime("us", "UTC"),
    "source_filename": pl.String(),
    "source_url": pl.String(),
}


class RestaurantConflict(BaseModel):
    """A same-latest-date disagreement on a descriptive field. Reported, not
    silently resolved -- one value is still deterministically chosen for the
    row (see module docstring), but the disagreement is not hidden."""

    model_config = ConfigDict(frozen=True)

    restaurant_id: str
    field: Literal["name", "address"]
    latest_date: str
    values: list[str]


class RestaurantBuildReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    input_row_count: int
    restaurant_count: int
    conflicting_restaurant_ids: list[str]
    conflicts: list[RestaurantConflict]


def _latest_rows_per_restaurant(events: pl.DataFrame) -> pl.DataFrame:
    """Every row whose ``inspection_date`` equals its restaurant's max, in a
    deterministic order (latest date, then ``inspection_id`` ascending as
    the tie-breaker -- never ``source_snapshot_date``, which many rows can
    share)."""
    latest_dates = events.group_by("restaurant_id").agg(
        pl.col("inspection_date").max().alias("latest_inspection_date")
    )
    joined = events.join(latest_dates, on="restaurant_id", how="inner").filter(
        pl.col("inspection_date") == pl.col("latest_inspection_date")
    )
    return joined.sort(["restaurant_id", "inspection_id"])


def _name_address_conflicts(
    latest_rows: pl.DataFrame, address_expr: pl.Expr
) -> tuple[set[str], list[RestaurantConflict]]:
    conflicting_ids: set[str] = set()
    conflicts: list[RestaurantConflict] = []
    with_address = latest_rows.with_columns(address_expr.alias("__address"))
    for restaurant_id, group in with_address.group_by("restaurant_id"):
        rid = str(restaurant_id[0]) if isinstance(restaurant_id, tuple) else str(restaurant_id)
        if group.height < 2:
            continue
        latest_date = str(group.get_column("latest_inspection_date")[0])
        names = sorted({n for n in group.get_column("dba").to_list() if n is not None})
        addresses = sorted({a for a in group.get_column("__address").to_list() if a is not None})
        if len(names) > 1:
            conflicting_ids.add(rid)
            conflicts.append(
                RestaurantConflict(
                    restaurant_id=rid, field="name", latest_date=latest_date, values=names
                )
            )
        if len(addresses) > 1:
            conflicting_ids.add(rid)
            conflicts.append(
                RestaurantConflict(
                    restaurant_id=rid, field="address", latest_date=latest_date, values=addresses
                )
            )
    return conflicting_ids, conflicts


def build_nyc_restaurants(events: pl.DataFrame) -> tuple[pl.DataFrame, RestaurantBuildReport]:
    """One row per NYC restaurant, from its latest known establishment
    snapshot. ``events`` must be NYC-only ``inspection_events`` rows."""
    if events.height == 0:
        return pl.DataFrame(schema=RESTAURANT_SCHEMA), RestaurantBuildReport(
            input_row_count=0, restaurant_count=0, conflicting_restaurant_ids=[], conflicts=[]
        )

    latest_rows = _latest_rows_per_restaurant(events)
    address_expr = pl.concat_str(
        [pl.col("building").fill_null(""), pl.col("street").fill_null("")], separator=" "
    ).str.strip_chars()
    conflicting_ids, conflicts = _name_address_conflicts(latest_rows, address_expr)

    # deterministic pick: first row per restaurant after the (date, inspection_id) sort
    chosen = latest_rows.group_by("restaurant_id", maintain_order=False).first()
    restaurants = chosen.select(
        pl.col("restaurant_id"),
        pl.lit("nyc").alias("jurisdiction"),
        pl.col("source_id"),
        pl.col("dba").alias("name"),
        pl.col("dba").alias("normalized_name"),  # normalized below
        address_expr.alias("address"),
        pl.col("boro_raw").alias("city"),
        pl.lit("NY").alias("region"),
        pl.col("zipcode").alias("postal_code"),
        pl.col("latitude"),
        pl.col("longitude"),
        pl.col("cuisine_description").alias("cuisine"),
        pl.col("latest_inspection_date"),
        pl.col("source_snapshot_date"),
        pl.col("source_retrieved_at_utc"),
        pl.lit(None, dtype=pl.String).alias("source_filename"),
        pl.lit(None, dtype=pl.String).alias("source_url"),
    ).sort("restaurant_id")

    from plateproof.matching.normalize import normalize_name

    restaurants = restaurants.with_columns(
        pl.col("name")
        .map_elements(lambda n: normalize_name(n) if n else n, return_dtype=pl.String)
        .alias("normalized_name")
    )

    report = RestaurantBuildReport(
        input_row_count=events.height,
        restaurant_count=restaurants.height,
        conflicting_restaurant_ids=sorted(conflicting_ids),
        conflicts=conflicts,
    )
    return restaurants.cast(RESTAURANT_SCHEMA), report  # type: ignore[arg-type]


def build_florida_restaurants(
    events: pl.DataFrame, staging: pl.DataFrame
) -> tuple[pl.DataFrame, RestaurantBuildReport]:
    """One row per Florida restaurant, joining Task 3's staging frame (which
    alone carries the full street address and city) back to normalized
    ``inspection_events`` via the exact ``inspection_id`` relationship Task 3
    already computed. ``events`` must be Florida-only ``inspection_events``
    rows; ``staging`` is the ``raw`` frame from ``load_florida_extracts``.

    Never infers that a prefixed license number (``HR12345``) and a numeric
    one (``12345``) are the same restaurant merely because they share
    digits -- identity always comes from ``inspection_events.restaurant_id``,
    which already encodes the exact license number.
    """
    if events.height == 0:
        return pl.DataFrame(schema=RESTAURANT_SCHEMA), RestaurantBuildReport(
            input_row_count=0, restaurant_count=0, conflicting_restaurant_ids=[], conflicts=[]
        )

    staging_with_id = staging.with_columns(
        (pl.lit("florida:") + pl.col("inspection_visit_id").cast(pl.String)).alias("inspection_id")
    )
    joined = events.join(
        staging_with_id.select(
            [
                "inspection_id",
                pl.col("dba").alias("staging_dba"),
                pl.col("location_address"),
                pl.col("location_city"),
                pl.col("location_zip"),
                pl.col("source_file"),
                pl.col("source_url"),
            ]
        ),
        on="inspection_id",
        how="inner",
    )

    latest_dates = joined.group_by("restaurant_id").agg(
        pl.col("inspection_date").max().alias("latest_inspection_date")
    )
    latest_rows = (
        joined.join(latest_dates, on="restaurant_id", how="inner")
        .filter(pl.col("inspection_date") == pl.col("latest_inspection_date"))
        .sort(["restaurant_id", "inspection_id"])
    )

    conflicting_ids: set[str] = set()
    conflicts: list[RestaurantConflict] = []
    for restaurant_id, group in latest_rows.group_by("restaurant_id"):
        rid = str(restaurant_id[0]) if isinstance(restaurant_id, tuple) else str(restaurant_id)
        if group.height < 2:
            continue
        latest_date = str(group.get_column("latest_inspection_date")[0])
        names = sorted({n for n in group.get_column("staging_dba").to_list() if n is not None})
        addresses = sorted(
            {a for a in group.get_column("location_address").to_list() if a is not None}
        )
        if len(names) > 1:
            conflicting_ids.add(rid)
            conflicts.append(
                RestaurantConflict(
                    restaurant_id=rid, field="name", latest_date=latest_date, values=names
                )
            )
        if len(addresses) > 1:
            conflicting_ids.add(rid)
            conflicts.append(
                RestaurantConflict(
                    restaurant_id=rid, field="address", latest_date=latest_date, values=addresses
                )
            )

    chosen = latest_rows.group_by("restaurant_id", maintain_order=False).first()
    restaurants = chosen.select(
        pl.col("restaurant_id"),
        pl.lit("florida").alias("jurisdiction"),
        pl.col("source_id"),
        pl.col("staging_dba").alias("name"),
        pl.col("staging_dba").alias("normalized_name"),
        pl.col("location_address").alias("address"),
        pl.col("location_city").alias("city"),
        pl.lit("FL").alias("region"),
        pl.col("location_zip").alias("postal_code"),
        pl.lit(None, dtype=pl.Float64).alias("latitude"),
        pl.lit(None, dtype=pl.Float64).alias("longitude"),
        pl.lit(None, dtype=pl.String).alias("cuisine"),
        pl.col("latest_inspection_date"),
        pl.col("source_snapshot_date"),
        pl.col("source_retrieved_at_utc"),
        pl.col("source_file").alias("source_filename"),
        pl.col("source_url"),
    ).sort("restaurant_id")

    from plateproof.matching.normalize import normalize_name

    restaurants = restaurants.with_columns(
        pl.col("name")
        .map_elements(lambda n: normalize_name(n) if n else n, return_dtype=pl.String)
        .alias("normalized_name")
    )

    report = RestaurantBuildReport(
        input_row_count=events.height,
        restaurant_count=restaurants.height,
        conflicting_restaurant_ids=sorted(conflicting_ids),
        conflicts=conflicts,
    )
    return restaurants.cast(RESTAURANT_SCHEMA), report  # type: ignore[arg-type]
