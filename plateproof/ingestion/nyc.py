"""NYC DOHMH restaurant-inspection ingestion: load, normalize, build events.

Source: DOHMH New York City Restaurant Inspection Results, NYC Open Data dataset
``43nn-pn8j``. The canonical internal input schema uses the lowercase SODA field
names; bulk-export display-name headers are normalized to them.

CAMIS note: the dataset metadata describes CAMIS as a "10-digit integer", but the
live feed returns 8-character values (e.g. ``50190663``). CAMIS is therefore kept
exactly as received after trimming -- never zero-padded -- and only checked for
being non-empty and numeric.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, ValidationError

import plateproof
from plateproof.features.inspection_events import (
    INSPECTION_EVENT_SCHEMA,
    VIOLATION_EVENT_SCHEMA,
    assert_unique_inspection_key,
)

NYC_DATASET_ID = "43nn-pn8j"
NYC_SOURCE_DATASET = "nyc_dohmh_43nn-pn8j"
NYC_LANDING_PAGE = "https://data.cityofnewyork.us/d/43nn-pn8j"
NYC_RESOURCE_CSV_URL = "https://data.cityofnewyork.us/resource/43nn-pn8j.csv"
NYC_VIEWS_METADATA_URL = "https://data.cityofnewyork.us/api/views/43nn-pn8j.json"

VIOLATION_ID_DIGEST_HEX = 32

NYC_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "camis",
        "inspection_date",
        "inspection_type",
        "action",
        "violation_code",
        "violation_description",
        "critical_flag",
        "score",
        "grade",
        "grade_date",
        "record_date",
    }
)

NYC_OPTIONAL_COLUMNS: frozenset[str] = frozenset(
    {
        "dba",
        "boro",
        "building",
        "street",
        "zipcode",
        "phone",
        "cuisine_description",
        "latitude",
        "longitude",
        "community_board",
        "council_district",
        "census_tract",
        "bin",
        "bbl",
        "nta",
        "location",
    }
)

NYC_IGNORED_COLUMNS: frozenset[str] = frozenset(
    {
        ":@computed_region_f5dn_yrer",
        ":@computed_region_yeji_bk3q",
        ":@computed_region_sbqj_enih",
        ":@computed_region_92fq_4b7q",
    }
)

NYC_DISPLAY_TO_SODA: dict[str, str] = {
    "CAMIS": "camis",
    "DBA": "dba",
    "BORO": "boro",
    "BUILDING": "building",
    "STREET": "street",
    "ZIPCODE": "zipcode",
    "PHONE": "phone",
    "CUISINE DESCRIPTION": "cuisine_description",
    "INSPECTION DATE": "inspection_date",
    "INSPECTION TYPE": "inspection_type",
    "ACTION": "action",
    "VIOLATION CODE": "violation_code",
    "VIOLATION DESCRIPTION": "violation_description",
    "CRITICAL FLAG": "critical_flag",
    "SCORE": "score",
    "GRADE": "grade",
    "GRADE DATE": "grade_date",
    "RECORD DATE": "record_date",
    "LATITUDE": "latitude",
    "LONGITUDE": "longitude",
    "COMMUNITY BOARD": "community_board",
    "COUNCIL DISTRICT": "council_district",
    "CENSUS TRACT": "census_tract",
    "BIN": "bin",
    "BBL": "bbl",
    "NTA": "nta",
    "LOCATION": "location",
}

_KNOWN_COLUMNS: frozenset[str] = NYC_REQUIRED_COLUMNS | NYC_OPTIONAL_COLUMNS
_DATE_COLUMNS = ("inspection_date", "grade_date", "record_date")
_FLOAT_COLUMNS = ("score", "latitude", "longitude")
_PLACEHOLDER_DATE = date(1900, 1, 1)
_SEVERITY_BY_FLAG = {
    "Critical": "critical",
    "Not Critical": "noncritical",
    "Not Applicable": "not_applicable",
}


class NYCSourceMetadata(BaseModel):
    """Provenance for one downloaded NYC extract, read from its sidecar file."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    dataset_id: str
    source_url: str
    landing_page: str | None = None
    retrieved_at_utc: datetime
    source_sha256: str
    socrata_rows_updated_at: str | None = None
    soql_where: str | None = None
    app_token_used: bool
    downloader_version: str


class NYCIngestionReport(BaseModel):
    """Deterministic, auditable summary of one NYC ingestion run."""

    model_config = ConfigDict(frozen=True)

    input_row_count: int
    output_inspection_count: int
    output_violation_count: int
    placeholder_date_rows_removed: int
    invalid_camis_rows_removed: int
    unparseable_date_rows_removed: int
    duplicate_violation_rows_removed: int
    score_conflict_group_count: int
    action_conflict_group_count: int
    grade_conflict_group_count: int
    snapshot_date: date | None
    ingestion_timestamp: datetime
    source_sha256: str | None = None
    retrieved_at_utc: datetime | None = None


@dataclass(frozen=True)
class NYCIngestionResult:
    """The two descriptive event tables plus the run report."""

    inspection_events: pl.DataFrame
    violation_events: pl.DataFrame
    report: NYCIngestionReport


def load_source_metadata(path: str | Path) -> NYCSourceMetadata:
    """Parse a downloader ``metadata.json`` sidecar.

    Raises ``FileNotFoundError`` if the file is absent and ``ValueError`` (naming
    the offending field) if a required key is missing or malformed.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        return NYCSourceMetadata.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"invalid NYC source metadata at {path}: {exc}") from exc


# --------------------------------------------------------------------------- #
# Loading                                                                      #
# --------------------------------------------------------------------------- #


def _canonical_name(column: str) -> str:
    if column in NYC_DISPLAY_TO_SODA:
        return NYC_DISPLAY_TO_SODA[column]
    upper = column.strip().upper()
    if upper in NYC_DISPLAY_TO_SODA:
        return NYC_DISPLAY_TO_SODA[upper]
    lowered = column.strip().lower().replace(" ", "_")
    if lowered in _KNOWN_COLUMNS:
        return lowered
    return column


def _normalize_headers(frame: pl.DataFrame) -> pl.DataFrame:
    drop: list[str] = []
    rename: dict[str, str] = {}
    claimed: dict[str, str] = {}
    for column in frame.columns:
        if column in NYC_IGNORED_COLUMNS:
            drop.append(column)
            continue
        target = _canonical_name(column)
        if target in claimed:
            raise ValueError(
                "NYC extract has a duplicate header mapping: "
                f"'{claimed[target]}' and '{column}' both map to canonical column '{target}'"
            )
        claimed[target] = column
        if target != column:
            rename[column] = target
    if drop:
        frame = frame.drop(drop)
    if rename:
        frame = frame.rename(rename)
    return frame


def _clean_string(column: str) -> pl.Expr:
    stripped = pl.col(column).cast(pl.String).str.strip_chars()
    return pl.when(stripped.str.len_chars() == 0).then(None).otherwise(stripped).alias(column)


def _parse_float(column: str) -> pl.Expr:
    stripped = pl.col(column).cast(pl.String).str.strip_chars()
    non_empty = pl.when(stripped.str.len_chars() == 0).then(None).otherwise(stripped)
    return non_empty.cast(pl.Float64, strict=False).alias(column)


def _parse_date(column: str) -> pl.Expr:
    stripped = pl.col(column).cast(pl.String).str.strip_chars()
    non_empty = pl.when(stripped.str.len_chars() == 0).then(None).otherwise(stripped)
    iso = non_empty.str.slice(0, 10).str.to_date("%Y-%m-%d", strict=False)
    slashed = non_empty.str.to_date("%m/%d/%Y", strict=False)
    return pl.coalesce([iso, slashed]).alias(column)


def load_nyc_raw(path: str | Path) -> pl.DataFrame:
    """Read a NYC extract CSV into a typed, one-row-per-source-citation frame.

    Accepts either lowercase SODA headers or bulk display-name headers. Dates
    become ``pl.Date`` (ISO or ``m/d/Y``); ``score``/``latitude``/``longitude``
    become ``pl.Float64``; every other column is trimmed ``pl.String`` with empty
    strings mapped to null. CAMIS is trimmed only. Placeholder and otherwise
    invalid rows are kept here and removed in :func:`build_nyc_inspection_events`.

    Raises ``ValueError`` if a required column is absent or if two input headers
    map to the same canonical name. Absent optional columns are added as null.
    Unexpected columns are preserved; ignored computed-region columns are dropped.
    """
    frame = pl.read_csv(Path(path), infer_schema_length=0, has_header=True)
    frame = _normalize_headers(frame)

    missing = NYC_REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"NYC extract is missing required column(s): {sorted(missing)}")

    for optional in sorted(NYC_OPTIONAL_COLUMNS - set(frame.columns)):
        frame = frame.with_columns(pl.lit(None, dtype=pl.String).alias(optional))

    typed: list[pl.Expr] = []
    for column in frame.columns:
        if column in _DATE_COLUMNS:
            typed.append(_parse_date(column))
        elif column in _FLOAT_COLUMNS:
            typed.append(_parse_float(column))
        else:
            typed.append(_clean_string(column))
    frame = frame.with_columns(typed)

    return frame.select(sorted(frame.columns))


# --------------------------------------------------------------------------- #
# Event construction                                                           #
# --------------------------------------------------------------------------- #


def _norm_ws(value: str) -> str:
    return " ".join(value.split())


def _as_date(value: Any) -> date | None:
    return value if isinstance(value, date) else None


def _camis_is_valid(value: Any) -> bool:
    return isinstance(value, str) and value != "" and value.isdigit()


def _severity(flag: Any) -> str:
    if isinstance(flag, str):
        return _SEVERITY_BY_FLAG.get(flag.strip(), "unknown")
    return "unknown"


def _first_present(rows: list[dict[str, Any]], field: str) -> Any:
    for row in rows:
        value = row.get(field)
        if value is not None and value != "":
            return value
    return None


def _ordinal(value: Any) -> int:
    parsed = _as_date(value)
    return parsed.toordinal() if parsed is not None else 0


def _resolve_conflict(pairs: list[tuple[Any, date | None]]) -> tuple[Any, bool, list[Any] | None]:
    """Conservative resolution of one per-inspection scalar field.

    * no non-null values -> (None, False, None)
    * exactly one distinct value -> (value, False, None)
    * many, one uniquely on the latest record_date -> (value, True, sorted distinct)
    * many, tied on the latest record_date -> (None, True, sorted distinct)
    """
    distinct = sorted({value for value, _ in pairs})
    if not distinct:
        return None, False, None
    if len(distinct) == 1:
        return distinct[0], False, None
    latest = max((rd for _, rd in pairs if rd is not None), default=None)
    at_latest = sorted({value for value, rd in pairs if latest is not None and rd == latest})
    if len(at_latest) == 1:
        return at_latest[0], True, distinct
    return None, True, distinct


def _stage_violations(kept: list[dict[str, Any]]) -> list[dict[str, Any]]:
    staged: list[dict[str, Any]] = []
    for row in kept:
        code = row.get("violation_code")
        if not isinstance(code, str) or code.strip() == "":
            continue
        code_clean = code.strip()
        description = row.get("violation_description")
        desc_clean = (
            description.strip() if isinstance(description, str) and description.strip() else None
        )
        staged.append(
            {
                "inspection_id": row["__inspection_id"],
                "restaurant_id": row["__restaurant_id"],
                "inspection_date": row["inspection_date"],
                "violation_code": code_clean,
                "violation_code_norm": _norm_ws(code_clean).casefold(),
                "violation_description": desc_clean,
                "violation_description_norm": (
                    _norm_ws(desc_clean).casefold() if desc_clean is not None else None
                ),
                "critical_flag_raw": row.get("critical_flag"),
                "record_date": row.get("record_date"),
                "row_index": row["__row"],
            }
        )
    staged.sort(
        key=lambda item: (
            item["inspection_id"],
            item["violation_code_norm"],
            item["violation_description_norm"] or "",
            -_ordinal(item["record_date"]),
            item["critical_flag_raw"] or "",
            item["row_index"],
        )
    )
    return staged


def _emit_violations(
    kept: list[dict[str, Any]], provenance: dict[str, Any]
) -> tuple[list[dict[str, Any]], int]:
    staged = _stage_violations(kept)
    seen: set[tuple[Any, Any, Any]] = set()
    emitted: list[dict[str, Any]] = []
    for item in staged:
        key = (
            item["inspection_id"],
            item["violation_code_norm"],
            item["violation_description_norm"],
        )
        if key in seen:
            continue
        seen.add(key)
        digest_source = (
            f"{item['inspection_id']}\x1f{item['violation_code_norm']}"
            f"\x1f{item['violation_description_norm'] or ''}"
        )
        digest = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()[:VIOLATION_ID_DIGEST_HEX]
        emitted.append(
            {
                "violation_event_id": f"nyc:v:{digest}",
                "inspection_id": item["inspection_id"],
                "restaurant_id": item["restaurant_id"],
                "jurisdiction": "nyc",
                "inspection_date": item["inspection_date"],
                "violation_code": item["violation_code"],
                "violation_code_norm": item["violation_code_norm"],
                "violation_description": item["violation_description"],
                "violation_description_norm": item["violation_description_norm"],
                "critical_flag_raw": item["critical_flag_raw"],
                "severity": _severity(item["critical_flag_raw"]),
                "corrected_on_site": None,
                **provenance,
            }
        )
    return emitted, len(staged) - len(emitted)


def build_nyc_inspection_events(
    raw: pl.DataFrame,
    *,
    ingested_at: datetime | None = None,
    source_metadata: NYCSourceMetadata | None = None,
) -> NYCIngestionResult:
    """Build descriptive NYC inspection-event and violation-event tables.

    Removes invalid-CAMIS, null/unparseable-date, and ``1900-01-01`` rows (each
    counted). Groups the remainder by ``(camis, inspection_date, normalized
    inspection_type)`` into one published inspection event, preserving the
    published score without summing, deduplicating violations before assigning
    stable digest ids, and resolving score/action/grade conflicts conservatively.

    ``ingested_at`` is captured once (``datetime.now(timezone.utc)`` when not
    supplied) and reused for every row and the report. ``source_metadata``
    supplies download provenance distinct from the row-level ``record_date``
    snapshot. Output is deterministic for identical ``raw`` + ``ingested_at`` +
    ``source_metadata``. No feature engineering or prediction targets are built.
    """
    now = ingested_at if ingested_at is not None else datetime.now(UTC)
    retrieved_at = source_metadata.retrieved_at_utc if source_metadata is not None else None
    source_sha256 = source_metadata.source_sha256 if source_metadata is not None else None
    version = plateproof.__version__
    provenance: dict[str, Any] = {
        "source_dataset": NYC_SOURCE_DATASET,
        "source_retrieved_at_utc": retrieved_at,
        "source_sha256": source_sha256,
        "ingested_at": now,
        "pipeline_version": version,
    }

    input_row_count = raw.height
    snapshot_date = _as_date(raw.select(pl.col("record_date").max()).item())
    provenance_with_snapshot = {**provenance, "source_snapshot_date": snapshot_date}

    rows: list[dict[str, Any]] = raw.to_dicts()
    for index, row in enumerate(rows):
        row["__row"] = index

    after_camis = [row for row in rows if _camis_is_valid(row.get("camis"))]
    invalid_camis_removed = len(rows) - len(after_camis)
    after_date = [row for row in after_camis if isinstance(row.get("inspection_date"), date)]
    unparseable_removed = len(after_camis) - len(after_date)
    kept = [row for row in after_date if row.get("inspection_date") != _PLACEHOLDER_DATE]
    placeholder_removed = len(after_date) - len(kept)

    for row in kept:
        camis = str(row["camis"])
        itype_norm = _norm_ws(str(row.get("inspection_type") or ""))
        inspection_date = row["inspection_date"]
        row["__itype_norm"] = itype_norm
        row["__restaurant_id"] = f"nyc:{camis}"
        row["__inspection_id"] = f"nyc:{camis}:{inspection_date.isoformat()}:{itype_norm}"

    violations, duplicate_violation_rows_removed = _emit_violations(kept, provenance_with_snapshot)

    counts: dict[str, tuple[int, int]] = {}
    for violation in violations:
        inspection_id = violation["inspection_id"]
        total, critical = counts.get(inspection_id, (0, 0))
        counts[inspection_id] = (
            total + 1,
            critical + (1 if violation["severity"] == "critical" else 0),
        )

    groups: dict[tuple[str, date, str], list[dict[str, Any]]] = {}
    for row in kept:
        key = (row["__restaurant_id"], row["inspection_date"], row["__itype_norm"])
        groups.setdefault(key, []).append(row)

    event_records: list[dict[str, Any]] = []
    score_conflicts = action_conflicts = grade_conflicts = 0
    for key in sorted(groups):
        restaurant_id, inspection_date, itype_norm = key
        members = groups[key]
        ordered = sorted(members, key=lambda row: (-_ordinal(row.get("record_date")), row["__row"]))
        inspection_id = ordered[0]["__inspection_id"]

        score, score_conflict, score_values = _resolve_conflict(
            [
                (row["score"], _as_date(row.get("record_date")))
                for row in members
                if row["score"] is not None
            ]
        )
        action, action_conflict, action_values = _resolve_conflict(
            [
                (row["action"], _as_date(row.get("record_date")))
                for row in members
                if row.get("action") not in (None, "", "Missing")
            ]
        )
        grade, grade_conflict, grade_values = _resolve_conflict(
            [
                (row["grade"], _as_date(row.get("record_date")))
                for row in members
                if row.get("grade") not in (None, "")
            ]
        )
        grade_dates = [
            row["grade_date"]
            for row in members
            if grade is not None
            and row.get("grade") == grade
            and isinstance(row.get("grade_date"), date)
        ]
        grade_date = max(grade_dates) if grade_dates else None

        score_conflicts += int(score_conflict)
        action_conflicts += int(action_conflict)
        grade_conflicts += int(grade_conflict)
        total, critical = counts.get(inspection_id, (0, 0))

        event_records.append(
            {
                "inspection_id": inspection_id,
                "restaurant_id": restaurant_id,
                "source_id": str(ordered[0]["camis"]),
                "jurisdiction": "nyc",
                "inspection_date": inspection_date,
                "inspection_type": itype_norm,
                "inspection_type_raw": _first_present(ordered, "inspection_type"),
                "action": action,
                "action_conflict": action_conflict,
                "action_conflict_values": action_values,
                "score": score,
                "score_conflict": score_conflict,
                "score_conflict_values": score_values,
                "grade": grade,
                "grade_conflict": grade_conflict,
                "grade_conflict_values": grade_values,
                "grade_date": grade_date,
                "violation_count": total,
                "critical_violation_count": critical,
                "high_priority_count": None,
                "intermediate_count": None,
                "basic_count": None,
                "dba": _first_present(ordered, "dba"),
                "boro_raw": _first_present(ordered, "boro"),
                "building": _first_present(ordered, "building"),
                "street": _first_present(ordered, "street"),
                "zipcode": _first_present(ordered, "zipcode"),
                "cuisine_description": _first_present(ordered, "cuisine_description"),
                "latitude": _first_present(ordered, "latitude"),
                "longitude": _first_present(ordered, "longitude"),
                "source_snapshot_date": snapshot_date,
                **provenance,
            }
        )

    events = pl.DataFrame(event_records, schema=INSPECTION_EVENT_SCHEMA).sort(
        ["restaurant_id", "inspection_date", "inspection_type"]
    )
    event_violations = pl.DataFrame(violations, schema=VIOLATION_EVENT_SCHEMA).sort(
        ["inspection_id", "violation_event_id"]
    )
    assert_unique_inspection_key(events)

    report = NYCIngestionReport(
        input_row_count=input_row_count,
        output_inspection_count=events.height,
        output_violation_count=event_violations.height,
        placeholder_date_rows_removed=placeholder_removed,
        invalid_camis_rows_removed=invalid_camis_removed,
        unparseable_date_rows_removed=unparseable_removed,
        duplicate_violation_rows_removed=duplicate_violation_rows_removed,
        score_conflict_group_count=score_conflicts,
        action_conflict_group_count=action_conflicts,
        grade_conflict_group_count=grade_conflicts,
        snapshot_date=snapshot_date,
        ingestion_timestamp=now,
        source_sha256=source_sha256,
        retrieved_at_utc=retrieved_at,
    )
    return NYCIngestionResult(
        inspection_events=events, violation_events=event_violations, report=report
    )
