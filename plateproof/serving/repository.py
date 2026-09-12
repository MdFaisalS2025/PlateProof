"""The single place all SQL for Task 7 lives. Routes and Streamlit pages call
typed methods here; neither touches DuckDB/Parquet directly.

Every processed table is optional at startup -- a missing or unreadable
Parquet file simply means that table is not registered, never a crash. All
user-supplied text (search query, restaurant_id) reaches SQL only through
bound parameters; a fixed set of column identifiers picked entirely by this
module's own code is the only thing ever interpolated into SQL text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import duckdb

from plateproof.core.config import Settings
from plateproof.matching.normalize import normalize_name
from plateproof.serving.errors import InvalidPaginationError, InvalidQueryError

_JURISDICTIONS = frozenset({"nyc", "florida"})
_RESTAURANT_ID_PATTERN = re.compile(r"^(nyc|florida):.+$")

# Fixed table catalog. No request or configuration value can add to or
# rename this set -- it exists so a table lookup is always one of these
# exact identifiers, never a caller-influenced string.
_PROCESSED_TABLES = (
    "restaurants",
    "inspection_events",
    "violation_events",
    "michelin_restaurants",
    "michelin_distinction_events",
    "restaurant_michelin_matches",
)


def _is_valid_restaurant_id(restaurant_id: str) -> bool:
    return bool(_RESTAURANT_ID_PATTERN.match(restaurant_id))


def _resolve_and_check_containment(resolved_base: Path, table: str) -> Path:
    """The path for one of :data:`_PROCESSED_TABLES` is always
    ``<resolved_base>/<table>.parquet`` -- constructed here, never taken from
    a caller. This still asserts containment as a defense-in-depth
    invariant: a misconfigured base directory (e.g. containing ``..``) must
    not resolve outside itself.

    ``resolved_base`` must already be fully resolved (``Path.resolve()``)
    *once* by the caller and reused for every table. Resolving the base
    directory separately per table is unsafe: for a leaf path that does not
    yet exist (e.g. an optional table's Parquet file has not been written),
    Windows can leave incidental artifacts in the un-resolved base portion
    (observed in practice: a trailing space in a configured directory, which
    the OS silently normalizes away only when resolving a path that fully
    exists) un-normalized, so the same base directory can appear to "escape
    itself" for some table names and not others.
    """
    candidate = (resolved_base / f"{table}.parquet").resolve()
    if not candidate.is_relative_to(resolved_base):
        raise ValueError(f"processed table path escaped its base directory: {table!r}")
    return candidate


@dataclass(frozen=True)
class PredictionLookup:
    row: dict[str, Any] | None
    stale_row_exists: bool


@dataclass(frozen=True)
class SearchResult:
    results: list[dict[str, Any]]
    total: int
    limit: int
    offset: int
    filters_applied: dict[str, bool] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    optional_data_status: dict[str, str] = field(default_factory=dict)


_FL_CATEGORY_CODE_PATTERN = re.compile(r"^\d{2}$")


def _display_severity(
    jurisdiction: str, violation_code: str | None, severity: str | None
) -> str | None:
    """Florida's aggregate numbered categories (two-digit codes) whose
    individual violation could not be classified into high/intermediate/basic
    are recorded as ``severity == "other"`` by Task 3. Presented as an
    explicit "classification unavailable" note -- never the bare word
    "other" -- for Florida's numbered categories specifically. This is
    presentation wording only: it never changes the stored value, and it is
    never applied to a generic "other" violation from any other jurisdiction
    or a non-numbered Florida code.
    """
    if (
        jurisdiction == "florida"
        and severity == "other"
        and violation_code is not None
        and _FL_CATEGORY_CODE_PATTERN.match(violation_code)
    ):
        return "classification_unavailable_at_category_level"
    return severity


class Repository:
    def __init__(
        self, conn: duckdb.DuckDBPyConnection, registered: set[str], max_page_size: int
    ) -> None:
        self._conn = conn
        self._registered = registered
        self._max_page_size = max_page_size

    def _has(self, table: str) -> bool:
        return table in self._registered

    def _scalar(self, sql: str, params: list[Any]) -> Any:
        row = self._conn.execute(sql, params).fetchone()
        return row[0] if row is not None else None

    # --- health --------------------------------------------------------- #

    def health_snapshot(self) -> dict[str, str]:
        if not self._has("restaurants"):
            datastore = "empty"
        else:
            datastore = "ok"
        nyc = (
            "ok"
            if self._has("restaurants")
            and self._scalar("select count(*) from restaurants where jurisdiction = ?", ["nyc"]) > 0
            else "unavailable"
        )
        florida = (
            "ok"
            if self._has("restaurants")
            and self._scalar("select count(*) from restaurants where jurisdiction = ?", ["florida"])
            > 0
            else "unavailable"
        )
        michelin = "ok" if self._has("michelin_restaurants") else "disabled"
        return {
            "datastore": datastore,
            "nyc_data": nyc,
            "florida_data": florida,
            "michelin": michelin,
        }

    # --- search / detail -------------------------------------------------- #

    def search_restaurants(
        self,
        *,
        query: str = "",
        jurisdiction: str | None = None,
        michelin_category: str | None = None,
        michelin_guide_year: int | None = None,
        limit: int,
        offset: int,
    ) -> SearchResult:
        if limit < 1 or limit > self._max_page_size:
            raise InvalidPaginationError(f"limit must be between 1 and {self._max_page_size}")
        if offset < 0:
            raise InvalidPaginationError("offset must be >= 0")
        if jurisdiction is not None and jurisdiction not in _JURISDICTIONS:
            raise InvalidQueryError(f"unsupported jurisdiction: {jurisdiction!r}")

        warnings: list[str] = []
        optional_data_status: dict[str, str] = {}
        filters_applied: dict[str, bool] = {}

        if not self._has("restaurants"):
            return SearchResult(
                results=[],
                total=0,
                limit=limit,
                offset=offset,
                filters_applied={"michelin_category": False} if michelin_category else {},
                warnings=warnings,
                optional_data_status=optional_data_status,
            )

        conditions: list[str] = []
        params: list[Any] = []

        if query:
            normalized = normalize_name(query)
            conditions.append("normalized_name LIKE ?")
            params.append(f"%{normalized}%")
        if jurisdiction is not None:
            conditions.append("jurisdiction = ?")
            params.append(jurisdiction)

        michelin_ids: set[str] | None = None
        if michelin_category is not None:
            if not (self._has("michelin_restaurants") and self._has("restaurant_michelin_matches")):
                optional_data_status["michelin"] = "unavailable"
                warnings.append(
                    "Michelin data is not configured; the michelin_category filter was not applied."
                )
                filters_applied["michelin_category"] = False
            else:
                optional_data_status["michelin"] = "ok"
                filters_applied["michelin_category"] = True
                michelin_ids = self._matching_official_ids_for_category(
                    michelin_category, michelin_guide_year
                )
                if not michelin_ids:
                    return SearchResult(
                        results=[],
                        total=0,
                        limit=limit,
                        offset=offset,
                        filters_applied=filters_applied,
                        warnings=warnings,
                        optional_data_status=optional_data_status,
                    )
                placeholders = ", ".join("?" for _ in michelin_ids)
                conditions.append(f"restaurant_id IN ({placeholders})")
                params.extend(sorted(michelin_ids))

        where_sql = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        total = self._scalar(f"SELECT COUNT(*) FROM restaurants {where_sql}", params)
        rows = self._conn.execute(
            f"""
            SELECT * FROM restaurants {where_sql}
            ORDER BY normalized_name, restaurant_id
            LIMIT ? OFFSET ?
            """,
            [*params, limit, offset],
        ).pl()
        results = rows.to_dicts()
        return SearchResult(
            results=results,
            total=int(total),
            limit=limit,
            offset=offset,
            filters_applied=filters_applied,
            warnings=warnings,
            optional_data_status=optional_data_status,
        )

    def _matching_official_ids_for_category(
        self, category: str, guide_year: int | None
    ) -> set[str]:
        if guide_year is not None:
            rows = self._conn.execute(
                """
                SELECT DISTINCT m.official_restaurant_id
                FROM restaurant_michelin_matches m
                JOIN michelin_distinction_events d
                  ON d.michelin_restaurant_id = m.michelin_restaurant_id
                WHERE m.decision = 'accept' AND d.distinction = ? AND d.guide_year = ?
                """,
                [category, guide_year],
            ).fetchall()
        else:
            # "latest documented guide edition" per matched Michelin restaurant
            rows = self._conn.execute(
                """
                WITH latest AS (
                    SELECT michelin_restaurant_id, MAX(guide_year) AS latest_guide_year
                    FROM michelin_distinction_events
                    GROUP BY michelin_restaurant_id
                )
                SELECT DISTINCT m.official_restaurant_id
                FROM restaurant_michelin_matches m
                JOIN michelin_distinction_events d
                  ON d.michelin_restaurant_id = m.michelin_restaurant_id
                JOIN latest l
                  ON l.michelin_restaurant_id = d.michelin_restaurant_id
                 AND l.latest_guide_year = d.guide_year
                WHERE m.decision = 'accept' AND d.distinction = ?
                """,
                [category],
            ).fetchall()
        return {r[0] for r in rows}

    def get_restaurant(self, restaurant_id: str) -> dict[str, Any] | None:
        if not _is_valid_restaurant_id(restaurant_id):
            raise InvalidQueryError(f"malformed restaurant id: {restaurant_id!r}")
        if not self._has("restaurants"):
            return None
        row = self._conn.execute(
            "SELECT * FROM restaurants WHERE restaurant_id = ?", [restaurant_id]
        ).pl()
        if row.height == 0:
            return None
        return row.to_dicts()[0]

    def list_inspections(self, restaurant_id: str) -> list[dict[str, Any]]:
        if not _is_valid_restaurant_id(restaurant_id):
            raise InvalidQueryError(f"malformed restaurant id: {restaurant_id!r}")
        if not self._has("inspection_events"):
            return []
        rows = self._conn.execute(
            """
            SELECT * FROM inspection_events
            WHERE restaurant_id = ?
            ORDER BY inspection_date DESC, inspection_id DESC
            """,
            [restaurant_id],
        ).pl()
        return rows.to_dicts()

    def list_violations(
        self, restaurant_id: str, inspection_id: str | None = None
    ) -> list[dict[str, Any]]:
        if not _is_valid_restaurant_id(restaurant_id):
            raise InvalidQueryError(f"malformed restaurant id: {restaurant_id!r}")
        if not self._has("violation_events"):
            return []
        conditions = ["restaurant_id = ?"]
        params: list[Any] = [restaurant_id]
        if inspection_id is not None:
            conditions.append("inspection_id = ?")
            params.append(inspection_id)
        rows = self._conn.execute(
            f"""
            SELECT * FROM violation_events
            WHERE {" AND ".join(conditions)}
            ORDER BY inspection_date DESC, violation_code
            """,
            params,
        ).pl()
        records = rows.to_dicts()
        jurisdiction = "florida" if restaurant_id.startswith("florida:") else "nyc"
        for record in records:
            record["severity"] = _display_severity(
                jurisdiction, record.get("violation_code"), record.get("severity")
            )
        return records

    def michelin_history(self, restaurant_id: str) -> list[dict[str, Any]]:
        if not (self._has("michelin_restaurants") and self._has("restaurant_michelin_matches")):
            return []
        rows = self._conn.execute(
            """
            SELECT d.*
            FROM restaurant_michelin_matches m
            JOIN michelin_distinction_events d
              ON d.michelin_restaurant_id = m.michelin_restaurant_id
            WHERE m.decision = 'accept' AND m.official_restaurant_id = ?
            ORDER BY d.guide_year DESC
            """,
            [restaurant_id],
        ).pl()
        return rows.to_dicts()

    # --- predictions / model registry ------------------------------------ #

    def latest_prediction(
        self,
        *,
        restaurant_id: str,
        jurisdiction: str,
        target_name: str,
        active_model_version: str,
        active_artifact_schema_version: str,
        staleness_days: int,
        today: date | None = None,
    ) -> PredictionLookup:
        """Never selects by ``MAX(generated_at)`` alone: a row is a valid,
        public candidate only when its jurisdiction, target, model version,
        and artifact schema version all match the currently active model,
        its readiness status is ``ready``, and its ``as_of_date`` is within
        the staleness window. Among valid candidates, prefers the newest
        ``as_of_date``, then the newest ``generated_at``, then the
        prediction id as a final stable tie-break. A same-model row that
        exists but falls outside the staleness window is reported via
        ``stale_row_exists`` rather than silently treated as "never
        scored".
        """
        if not self._has("predictions"):
            return PredictionLookup(row=None, stale_row_exists=False)

        today = today or date.today()
        cutoff = today - timedelta(days=staleness_days)
        match_params = [
            restaurant_id,
            jurisdiction,
            target_name,
            active_model_version,
            active_artifact_schema_version,
        ]

        valid = self._conn.execute(
            """
            SELECT * FROM predictions
            WHERE restaurant_id = ? AND jurisdiction = ? AND target_name = ?
              AND model_version = ? AND artifact_schema_version = ?
              AND readiness_status = 'ready'
              AND as_of_date >= ?
            ORDER BY as_of_date DESC, generated_at DESC, prediction_id DESC
            LIMIT 1
            """,
            [*match_params, cutoff],
        ).pl()
        if valid.height:
            return PredictionLookup(row=valid.to_dicts()[0], stale_row_exists=False)

        stale = self._conn.execute(
            """
            SELECT COUNT(*) FROM predictions
            WHERE restaurant_id = ? AND jurisdiction = ? AND target_name = ?
              AND model_version = ? AND artifact_schema_version = ?
              AND readiness_status = 'ready'
            """,
            match_params,
        ).fetchone()
        stale_exists = bool(stale and stale[0] > 0)
        return PredictionLookup(row=None, stale_row_exists=stale_exists)

    def model_registry_entry(self, jurisdiction: str) -> dict[str, Any] | None:
        """Public registry fields only -- never ``artifact_path``, which
        exists in the underlying table for administrator use only."""
        if not self._has("model_registry"):
            return None
        rows = self._conn.execute(
            """
            SELECT jurisdiction, target_name, model_version, artifact_schema_version,
                   deployment_status, registered_at, source_snapshot_date
            FROM model_registry
            WHERE jurisdiction = ?
            """,
            [jurisdiction],
        ).pl()
        if rows.height == 0:
            return None
        return rows.to_dicts()[0]


def open_repository(settings: Settings) -> Repository:
    """Open (or, for tests, create a fresh in-memory) DuckDB connection and
    register a view for every processed table whose Parquet file exists and
    is readable. A missing or unsupported file is simply not registered --
    never a startup failure."""
    conn = duckdb.connect(":memory:")
    base_dir = settings.resolve_path(settings.processed_data_dir)
    registered: set[str] = set()
    if base_dir.is_dir():
        resolved_base = base_dir.resolve()
        for table in _PROCESSED_TABLES:
            path = _resolve_and_check_containment(resolved_base, table)
            if not path.is_file():
                continue
            try:
                conn.read_parquet(str(path)).create_view(table)
                registered.add(table)
            except duckdb.Error:
                continue
    if settings.prediction_table_path is not None:
        prediction_dir = settings.resolve_path(settings.prediction_table_path)
        for table in ("predictions", "model_registry"):
            path = prediction_dir / f"{table}.parquet"
            if not path.is_file():
                continue
            try:
                conn.read_parquet(str(path)).create_view(table)
                registered.add(table)
            except duckdb.Error:
                continue
    return Repository(conn, registered=registered, max_page_size=settings.max_page_size)
