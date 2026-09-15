"""Florida DBPR restaurant-inspection ingestion: load, normalize, build events.

Scope: Florida DBPR **Division of Hotels and Restaurants, public food-service
program only** -- rows where ``Inspection Class == "Food"`` and ``License Type
Code`` is one of the public food-service codes (2010, 2012-2016). This
deliberately excludes DBPR's own Lodging program (2001-2009) and never touches
anything regulated by Florida DOH or FDACS (this file only ever contains DBPR's
own licensees in the first place).

Source: DBPR "Restaurants/Food Service Public Records"
(https://www2.myfloridalicense.com/hotels-restaurants/public-records/). Two
verified layouts are supported:

* current-fiscal-year district CSV extracts (``{district}fdinspi.csv``), and
* recent historical statewide XLSX archives (``fdinspi_{fy}.xlsx``).

Legacy ``.xls``-format district archives are explicitly deferred (recorded in
``FLORIDA_SOURCE_MANIFEST`` with a note, not parsed).

License Number note: the inspection extract's License Number is a plain,
possibly-alphanumeric string (observed numeric in samples, e.g. ``2300027``)
and is preserved exactly. Florida's separate *license* extract (``hrfood*.csv``,
out of scope here) has been observed to format the same identifier with a
rank-code prefix (e.g. ``SEA2300159``) -- a real, documented discrepancy left
for Task 4 entity resolution, not reconciled here.
"""

from __future__ import annotations

import hashlib
import io
import json
import warnings
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

import polars as pl
from pydantic import BaseModel, ConfigDict, ValidationError

import plateproof
from plateproof.features.inspection_events import (
    INSPECTION_EVENT_SCHEMA,
    VIOLATION_EVENT_SCHEMA,
    assert_unique_inspection_id,
    finalize_event_frame,
)

FL_DATASET_ID = "fl_dbpr_food_service_inspections"
FL_SOURCE_DATASET = "florida_dbpr_food_service_inspections"
FL_PUBLIC_RECORDS_PAGE = "https://www2.myfloridalicense.com/hotels-restaurants/public-records/"

VIOLATION_ID_DIGEST_HEX = 32

#: DBPR public food-service license type codes (readme.pdf, "PUBLIC FOOD SERVICE
#: LICENSE TYPE AND RANK CODES"). 2001-2009 is the disjoint Lodging program and
#: is intentionally excluded.
FL_FOOD_SERVICE_LICENSE_TYPE_CODES: frozenset[str] = frozenset(
    {"2010", "2012", "2013", "2014", "2015", "2016"}
)

FL_REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "license_type_code",
        "license_number",
        "inspection_number",
        "visit_number",
        "inspection_class",
        "inspection_type",
        "inspection_disposition",
        "inspection_date",
        "total_violations",
        "high_priority_count",
        "intermediate_count",
        "basic_count",
        "inspection_visit_id",
        *(f"violation_{i:02d}" for i in range(1, 59)),
    }
)

FL_OPTIONAL_COLUMNS: frozenset[str] = frozenset(
    {
        "district",
        "county_number",
        "county_name",
        "dba",
        "location_address",
        "location_city",
        "location_zip",
        "pda_status",
        "critical_violations_legacy",
        "noncritical_violations_legacy",
        "license_id",
        # Present only in the historical statewide XLSX layout; meaning not
        # verified against any official documentation. Preserved, never used.
        "ht_violations_unverified",
    }
)

_KNOWN_COLUMNS: frozenset[str] = FL_REQUIRED_COLUMNS | FL_OPTIONAL_COLUMNS
_AUDIT_COLUMNS = ("source_file", "source_url", "source_encoding")

#: Verified real header (current CSV extract, fetched directly from
#: 1fdinspi.csv). Leading spaces on some headers are real and are stripped
#: before lookup, not encoded here.
FL_CSV_DISPLAY_TO_CANONICAL: dict[str, str] = {
    "District": "district",
    "County Number": "county_number",
    "County Name": "county_name",
    "License Type Code": "license_type_code",
    "License Number": "license_number",
    "Business (DBA-Does Business As) Name": "dba",
    "Location Address": "location_address",
    "Location City": "location_city",
    "Location Zip Code": "location_zip",
    "Inspection Number": "inspection_number",
    "Visit Number": "visit_number",
    "Inspection Class": "inspection_class",
    "Inspection Type": "inspection_type",
    "Inspection Disposition": "inspection_disposition",
    "Inspection Date": "inspection_date",
    "Number of Critical Violations": "critical_violations_legacy",
    "Number of Noncritical Violations": "noncritical_violations_legacy",
    "Number of Total Violations": "total_violations",
    "Number of High Priority Violations": "high_priority_count",
    "Number of Intermediate Violations": "intermediate_count",
    "Number of Basic Violations": "basic_count",
    "PDA Status": "pda_status",
    **{f"Violation {i:02d}": f"violation_{i:02d}" for i in range(1, 59)},
    "License ID": "license_id",
    "Inspection Visit ID": "inspection_visit_id",
}

#: Verified real header (historical statewide XLSX, fdinspi_2425.xlsx, fetched
#: and inspected locally -- never committed).
FL_XLSX_DISPLAY_TO_CANONICAL: dict[str, str] = {
    "DISTRICT": "district",
    "COUNTYCODE": "county_number",
    "CNTY_DESC": "county_name",
    "PROFESSION": "license_type_code",
    "LICENSE_NO": "license_number",
    "DBA_NAME": "dba",
    "LOC_ADDRESS": "location_address",
    "LOC_CITY": "location_city",
    "LOC_ZIP": "location_zip",
    "INSP_NO": "inspection_number",
    "VISIT_NO": "visit_number",
    "INSPCLASS": "inspection_class",
    "INSPTYPE": "inspection_type",
    "DISPOSITION": "inspection_disposition",
    "INSP_DATE": "inspection_date",
    "CRIT_VIOL": "critical_violations_legacy",
    "NONCRIT_VIOL": "noncritical_violations_legacy",
    "VIOLATIONS": "total_violations",
    "HIGH_VIOL": "high_priority_count",
    "INTERMED_VIOL": "intermediate_count",
    "BASIC_VIOL": "basic_count",
    "HT_VIOL": "ht_violations_unverified",
    "PDA": "pda_status",
    **{f"V_{i:02d}": f"violation_{i:02d}" for i in range(1, 59)},
    "LIC_ID": "license_id",
    "INSP_VST_ID": "inspection_visit_id",
}

#: The 14 verified real disposition values from DBPR's own "Inspection
#: Dispositions" webpage, grouped into DBPR's own three categories plus the
#: fallback "unknown" for anything not in this list. Ascii hyphens throughout;
#: input is normalized (unicode dashes -> "-") before lookup.
FL_DISPOSITION_STATUS: dict[str, str] = {
    "Inspection Completed - No Further Action": "met_standards",
    "Callback - Complied": "met_standards",
    "Admin. Complaint Callback Complied": "met_standards",
    "Emergency Order Callback Complied": "met_standards",
    "Warning Issued": "follow_up_required",
    "Callback - Extension given, pending": "follow_up_required",
    "Callback - Administrative complaint recommended": "follow_up_required",
    "Administrative complaint recommended": "follow_up_required",
    "Admin. Complaint Callback Not Complied": "follow_up_required",
    "Administrative Complaint Time Extension": "follow_up_required",
    "Emergency Order Callback Time Extension": "follow_up_required",
    "Emergency order recommended": "temporary_closure",
    "Administrative determination recommended": "temporary_closure",
    "Emergency Order Callback Not Complied": "temporary_closure",
}

#: Real DBPR-published titles for violation categories 1-58 (source: DBPR
#: department-wide "Licensee Download Files" layout reference,
#: https://www2.myfloridalicense.com/sto/documents/readme.pdf, "Food Service
#: Violations are numbered 1-58"). These titles use the pre-2013
#: critical/non-critical scheme's vocabulary in places but carry no current-tier
#: classification; map_florida_violation_classification is never derived from
#: them -- see the module docstring and correction #8.
FLORIDA_VIOLATION_CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "01": "Approved source",
    "02": "Original container: properly labeled, date marking, consumer advisory",
    "03": "Food Out of Temperature",
    "04": "Facilities to maintain product temperature",
    "05": "Thermometers provided and conspicuously placed",
    "06": "Potentially hazardous food properly thawed",
    "07": "Unwrapped or potentially hazardous food not re-served",
    "08": "Food protection, cross-contamination",
    "09": "Foods handled with minimum contact",
    "10": "In use food dispensing utensils properly stored",
    "11": "Personnel with infections restricted",
    "12": "Hands washed and clean, good hygienic practices, eating/drinking/smoking",
    "13": "Clean clothes, hair restraints",
    "14": "Food contact surfaces designed, constructed, maintained, installed, located",
    "15": "Non-food contact surfaces designed, constructed, maintained, installed, located",
    "16": "Dishwashing facilities designed, constructed, operated",
    "17": "Thermometers, gauges, test kits provided",
    "18": "Pre-flushed, scraped, soaked",
    "19": "Wash, rinse water clean, proper temperature",
    "20": "Sanitizing concentration or temperature",
    "21": "Wiping cloths clean, used properly, stored",
    "22": "Food contact surfaces of equipment and utensils clean",
    "23": "Non-food contact surfaces clean",
    "24": "Storage/handling of clean equipment, utensils",
    "25": "Single service items properly stored, handled, dispensed",
    "26": "Single service articles not re-used",
    "27": "Water source safe, hot and cold under pressure",
    "28": "Sewage and wastewater disposed properly",
    "29": "Plumbing installed and maintained",
    "30": "Cross-connection, back siphonage, backflow",
    "31": "Toilet and hand-washing facilities, number, convenient, designed, installed",
    "32": (
        "Restrooms with self-closing doors, fixtures operate properly, facility clean, "
        "supplied with hand-soap, disposable towels or hand drying devices, tissue, "
        "covered waste receptacles"
    ),
    "33": (
        "Containers covered, adequate number, insect and rodent proof, emptied at "
        "proper intervals, clean"
    ),
    "34": "Outside storage area clean, enclosure properly constructed",
    "35": (
        "Presence of insects/rodents. Animals prohibited. Outer openings protected "
        "from insects, rodent proof"
    ),
    "36": "Floors properly constructed, clean, drained, coved",
    "37": "Walls, ceilings, and attached equipment, constructed, clean",
    "38": "Lighting provided as required. Fixtures shielded",
    "39": "Rooms and equipment - vented as required",
    "40": "Employee lockers provided and used, clean",
    "41": "Toxic items properly stored, labeled and used properly",
    "42": (
        "Premises maintained, free of litter, unnecessary articles. Cleaning and "
        "maintenance equipment properly stored. Kitchen restricted to authorized personnel"
    ),
    "43": "Complete separation from living/sleeping area, laundry",
    "44": "Clean and soiled linen segregated and properly stored",
    "45": "Fire extinguishers - proper and sufficient (FOR REPORTING PURPOSES ONLY)",
    "46": "Exiting system - adequate, good repair (FOR REPORTING PURPOSES ONLY)",
    "47": "Electrical wiring - adequate, good repair (FOR REPORTING PURPOSES ONLY)",
    "48": "Gas appliances - properly installed, maintained (FOR REPORTING PURPOSES ONLY)",
    "49": "Flammable/combustible materials - properly stored (FOR REPORTING PURPOSES ONLY)",
    "50": "Current license properly displayed",
    "51": "Other conditions sanitary and safe operation",
    "52": "False/misleading statements published or advertised relating to food/beverage",
    "53": "Food management certification valid / Employee training verification",
    "54": "Florida Clean Indoor Air Act",
    "55": "Automatic Gratuity Notice",
    "56": (
        "Copy of Chapter 509, Florida Statutes, available "
        "(no longer used per Chapter 2008-55, Laws of Florida)"
    ),
    "57": "Hospitality Education Program information provided (information only, not a violation)",
    "58": "Smoke Free (information only, no longer used)",
}


def normalize_violation_code(raw: str) -> str | None:
    """Map a free-text Florida violation-category number (as it might appear
    handwritten or printed on a paper inspection form, e.g. ``"12"`` or
    ``"#12"``) to the canonical two-digit zero-padded code
    (``"01"``-``"58"``) used internally -- see ``FLORIDA_VIOLATION_CATEGORY_DESCRIPTIONS``.
    Unlike NYC's, Florida's canonical code is synthesized from a fixed
    column index rather than parsed from free text in the ingestion
    pipeline, so there is no existing private function to wrap; this is the
    equivalent single, small, additive normalizer Task 9 extraction reuses
    instead of duplicating the 1-58 bounds check. Returns ``None`` for
    anything that is not an integer in that range.
    """
    stripped = raw.strip().lstrip("#").strip()
    try:
        number = int(stripped)
    except ValueError:
        return None
    if not (1 <= number <= 58):
        return None
    return f"{number:02d}"


def map_florida_violation_classification(
    raw: str | None,
) -> Literal["high_priority", "intermediate", "basic", "other"]:
    """Normalize a raw Florida severity token to the shared four-bucket vocabulary.

    Unrecognized, blank, or ``None`` input maps to ``"other"``. The obsolete
    pre-2013 critical/non-critical markers are deliberately NOT recognized here
    (see correction #8) -- there is no verified mapping from them to the current
    High Priority/Intermediate/Basic tiers, so guessing one would misclassify.
    """
    if not isinstance(raw, str):
        return "other"
    token = raw.strip().casefold()
    if token == "high priority":
        return "high_priority"
    if token == "intermediate":
        return "intermediate"
    if token == "basic":
        return "basic"
    return "other"


@dataclass(frozen=True)
class FloridaExtractSource:
    """One input file to load: its local path, plus optional download provenance."""

    path: Path
    source_url: str | None = None
    fiscal_year: str | None = None
    district: int | None = None


@dataclass(frozen=True)
class FloridaSourceFileSpec:
    """One entry in the hand-maintained manifest of known DBPR file locations."""

    fiscal_year: str
    district: int | None
    url: str
    format: Literal["csv", "xlsx", "xls"]
    note: str | None = None


#: Hand-maintained manifest of verified DBPR file locations. DBPR publishes no
#: index/API, so this must be updated manually as new fiscal years appear.
FLORIDA_SOURCE_MANIFEST: tuple[FloridaSourceFileSpec, ...] = (
    *(
        FloridaSourceFileSpec(
            fiscal_year="current",
            district=d,
            format="csv",
            url=f"https://www2.myfloridalicense.com/sto/file_download/extracts/{d}fdinspi.csv",
        )
        for d in range(1, 8)
    ),
    FloridaSourceFileSpec(
        fiscal_year="2526",
        district=None,
        format="xlsx",
        url="https://www2.myfloridalicense.com/hr/inspections/fdinspi_2526.xlsx",
    ),
    FloridaSourceFileSpec(
        fiscal_year="2425",
        district=None,
        format="xlsx",
        url="https://www2.myfloridalicense.com/hr/inspections/fdinspi_2425.xlsx",
    ),
    FloridaSourceFileSpec(
        fiscal_year="2324",
        district=None,
        format="xlsx",
        url="https://www2.myfloridalicense.com/hr/inspections/fdinspi_2324.xlsx",
    ),
    FloridaSourceFileSpec(
        fiscal_year="2223",
        district=None,
        format="xlsx",
        url="https://www2.myfloridalicense.com/sto/file_download/hr/fdinspi_2223.xlsx",
    ),
    FloridaSourceFileSpec(
        fiscal_year="2122",
        district=None,
        format="xlsx",
        url="https://www2.myfloridalicense.com/sto/file_download/hr/fdinspi_2122.xlsx",
    ),
    *(
        FloridaSourceFileSpec(
            fiscal_year="2021",
            district=d,
            format="csv",
            url=f"https://www2.myfloridalicense.com/sto/file_download/hr/{d}fdinspi_2021.csv",
        )
        for d in range(1, 8)
    ),
    *(
        FloridaSourceFileSpec(
            fiscal_year="1920",
            district=d,
            format="xls",
            url=(
                f"https://www2.myfloridalicense.com/sto/file_download/extracts/{d}fdinspi_1920.xls"
            ),
            note=(
                "deferred: legacy .xls parser not implemented in Task 3; URL pattern "
                "verified for districts 1-2, extrapolated for 3-7 from the same path"
            ),
        )
        for d in range(1, 8)
    ),
)


class FloridaFileMetadata(BaseModel):
    """Provenance for one downloaded Florida file."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    url: str
    fiscal_year: str
    district: int | None
    format: str
    local_filename: str
    retrieved_at_utc: datetime
    sha256: str
    byte_size: int
    row_count: int
    encoding: str


class FloridaSnapshotMetadata(BaseModel):
    """Provenance for one Florida download run, potentially spanning many files."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    dataset_id: str = FL_DATASET_ID
    retrieved_at_utc: datetime
    requested_fiscal_years: list[str]
    requested_districts: list[int]
    files: list[FloridaFileMetadata]
    downloader_version: str
    status: str = "complete"
    skipped_files: list[dict[str, Any]] = []
    terms: str = "DBPR Public Records — Chapter 119, F.S."


def load_florida_snapshot_metadata(path: str | Path) -> FloridaSnapshotMetadata:
    """Parse a downloader ``manifest.json``.

    Raises ``FileNotFoundError`` if absent, ``ValueError`` (naming the field) if
    a required key is missing or malformed.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        return FloridaSnapshotMetadata.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"invalid Florida snapshot metadata at {path}: {exc}") from exc


class FloridaIngestionReport(BaseModel):
    """Deterministic, auditable summary of one Florida ingestion run."""

    model_config = ConfigDict(frozen=True)

    input_row_count: int
    output_inspection_count: int
    output_violation_count: int
    files_loaded_count: int
    non_food_service_rows_removed: int
    missing_license_number_rows_removed: int
    missing_inspection_visit_id_rows_removed: int
    invalid_inspection_identity_rows_removed: int
    unparseable_date_rows_removed: int
    identical_duplicate_visit_rows_removed: int
    conflicting_visit_id_rows_removed: int
    conflicting_visit_ids: list[str]
    total_vs_hib_mismatch_count: int
    total_vs_category_sum_mismatch_count: int
    malformed_count_field_rows: int
    unknown_disposition_count: int
    snapshot_date: date | None
    ingestion_timestamp: datetime
    source_encodings: dict[str, str]


@dataclass(frozen=True)
class FloridaIngestionResult:
    """The two descriptive event tables plus the run report.

    The auditable staging representation (correction #7) is the ``raw`` frame
    passed to :func:`build_florida_inspection_events` -- the output of
    :func:`load_florida_extracts` -- which already preserves every original
    DBPR column under a documented, reproducible canonical name. It is not
    duplicated here.
    """

    inspection_events: pl.DataFrame
    violation_events: pl.DataFrame
    report: FloridaIngestionReport


# --------------------------------------------------------------------------- #
# Decoding                                                                     #
# --------------------------------------------------------------------------- #


def _decode_bytes(data: bytes) -> tuple[str, str]:
    """Decode CSV bytes deterministically: UTF-8 (BOM-aware) then Windows-1252.

    Never uses ``errors="replace"``: a byte sequence decodable by neither
    encoding raises ``ValueError`` naming the failure, rather than silently
    corrupting names, addresses, or identifiers.
    """
    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig"), "utf-8-sig"
    try:
        return data.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        pass
    try:
        return data.decode("windows-1252"), "windows-1252"
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"could not decode file as UTF-8 or Windows-1252 (refusing to guess): {exc}"
        ) from exc


# --------------------------------------------------------------------------- #
# Loading                                                                      #
# --------------------------------------------------------------------------- #


def _canonical_name(column: str, display_map: dict[str, str]) -> str:
    stripped = column.strip()
    if stripped in display_map:
        return display_map[stripped]
    upper = stripped.upper()
    if upper in display_map:
        return display_map[upper]
    lowered = stripped.lower().replace(" ", "_")
    if lowered in _KNOWN_COLUMNS:
        return lowered
    return stripped


def _normalize_headers(frame: pl.DataFrame, display_map: dict[str, str]) -> pl.DataFrame:
    rename: dict[str, str] = {}
    claimed: dict[str, str] = {}
    for column in frame.columns:
        target = _canonical_name(column, display_map)
        if target in claimed:
            raise ValueError(
                "Florida extract has a duplicate header mapping: "
                f"'{claimed[target]}' and '{column}' both map to canonical column '{target}'"
            )
        claimed[target] = column
        if target != column:
            rename[column] = target
    return frame.rename(rename) if rename else frame


def _clean_string(column: str) -> pl.Expr:
    stripped = pl.col(column).cast(pl.String).str.strip_chars()
    return pl.when(stripped.str.len_chars() == 0).then(None).otherwise(stripped).alias(column)


def _finalize_columns(frame: pl.DataFrame, display_map: dict[str, str]) -> pl.DataFrame:
    frame = _normalize_headers(frame, display_map)
    missing = FL_REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Florida extract is missing required column(s): {sorted(missing)}")
    for optional in sorted(FL_OPTIONAL_COLUMNS - set(frame.columns)):
        frame = frame.with_columns(pl.lit(None, dtype=pl.String).alias(optional))
    frame = frame.with_columns([_clean_string(c) for c in frame.columns])
    return frame.select(sorted(frame.columns))


def _read_csv_source(source: FloridaExtractSource) -> pl.DataFrame:
    data = Path(source.path).read_bytes()
    text, encoding = _decode_bytes(data)
    frame = pl.read_csv(io.StringIO(text), infer_schema_length=0, has_header=True)
    frame = _finalize_columns(frame, FL_CSV_DISPLAY_TO_CANONICAL)
    return frame.with_columns(
        pl.lit(Path(source.path).name).alias("source_file"),
        pl.lit(source.source_url).alias("source_url"),
        pl.lit(encoding).alias("source_encoding"),
    )


def _read_xlsx_source(source: FloridaExtractSource) -> pl.DataFrame:
    with warnings.catch_warnings():
        # fastexcel's arrow bridge warns about a pl.from_arrow return-type change
        # in a future polars major version; irrelevant to how we call read_excel.
        warnings.simplefilter("ignore", FutureWarning)
        frame = pl.read_excel(Path(source.path))
    frame = frame.with_columns([pl.col(c).cast(pl.String) for c in frame.columns])
    frame = _finalize_columns(frame, FL_XLSX_DISPLAY_TO_CANONICAL)
    return frame.with_columns(
        pl.lit(Path(source.path).name).alias("source_file"),
        pl.lit(source.source_url).alias("source_url"),
        pl.lit("xlsx").alias("source_encoding"),
    )


def load_florida_extracts(sources: list[FloridaExtractSource]) -> pl.DataFrame:
    """Read one or more Florida DBPR inspection extracts into one typed,
    deterministically concatenated **staging** frame -- one row per source
    inspection visit, canonical column names, every original DBPR field
    preserved as a trimmed string under its documented canonical name, plus
    ``source_file``/``source_url``/``source_encoding`` audit columns.

    Accepts a mix of current-format CSV and historical statewide XLSX sources
    (dispatched by file extension). Sources are sorted by filename before
    concatenation so output order never depends on caller-supplied order.
    Performs no filtering, deduplication, or type parsing beyond trimming --
    see :func:`build_florida_inspection_events` for those. Raises ``ValueError``
    if any single source is missing a required column, has a duplicate header
    mapping, or cannot be decoded.
    """
    ordered = sorted(sources, key=lambda s: Path(s.path).name)
    frames = []
    for source in ordered:
        suffix = Path(source.path).suffix.lower()
        if suffix == ".xlsx":
            frames.append(_read_xlsx_source(source))
        elif suffix == ".csv":
            frames.append(_read_csv_source(source))
        else:
            raise ValueError(f"unsupported Florida extract format: {source.path}")
    if not frames:
        columns = sorted({*FL_REQUIRED_COLUMNS, *FL_OPTIONAL_COLUMNS, *_AUDIT_COLUMNS})
        return pl.DataFrame(schema={c: pl.String() for c in columns})
    return pl.concat(frames, how="vertical")


# --------------------------------------------------------------------------- #
# Event construction                                                          #
# --------------------------------------------------------------------------- #

_MATERIAL_FIELDS: tuple[str, ...] = (
    "license_number",
    "inspection_number",
    "__visit_number",
    "inspection_class",
    "inspection_type",
    "inspection_disposition",
    "__inspection_date",
    "__total_violations",
    "__high_priority_count",
    "__intermediate_count",
    "__basic_count",
    *(f"__violation_{i:02d}" for i in range(1, 59)),
)


def _parse_date(value: Any) -> date | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _parse_nonneg_int(value: Any) -> tuple[int | None, bool]:
    """Parse a nonnegative integer count.

    Returns ``(value, was_malformed)``. Blank/None is "not provided", not
    malformed: ``(None, False)``. Non-numeric or negative is malformed and is
    never silently coerced to zero: ``(None, True)``.
    """
    if not isinstance(value, str) or not value.strip():
        return None, False
    try:
        parsed = int(value.strip())
    except ValueError:
        return None, True
    if parsed < 0:
        return None, True
    return parsed, False


def _normalize_disposition(value: str) -> str:
    return value.replace("–", "-").replace("—", "-").strip()


def _is_food_service(row: dict[str, Any]) -> bool:
    inspection_class = row.get("inspection_class")
    license_type = row.get("license_type_code")
    return (
        isinstance(inspection_class, str)
        and inspection_class.strip().casefold() == "food"
        and isinstance(license_type, str)
        and license_type.strip() in FL_FOOD_SERVICE_LICENSE_TYPE_CODES
    )


def build_florida_inspection_events(
    raw: pl.DataFrame,
    *,
    ingested_at: datetime | None = None,
    source_metadata: FloridaSnapshotMetadata | None = None,
    on_conflicting_duplicate: Literal["drop_and_report", "raise"] = "drop_and_report",
) -> FloridaIngestionResult:
    """Build descriptive Florida inspection-event and violation-event tables.

    Scopes to the public food-service program (``inspection_class == "Food"``
    and a food-service ``license_type_code``); rejects rows missing an identity
    field (license number, inspection visit id, inspection number/visit number,
    a parseable inspection date); deduplicates rows sharing an
    ``inspection_visit_id`` (identical rows collapse deterministically,
    conflicting ones are dropped and reported -- or raise, if requested);
    preserves DBPR's authoritative ``total_violations``/H/I/B counts verbatim
    (never recomputed to force agreement with the numbered violation columns)
    while reporting any disagreement; and never silently turns a malformed or
    negative count into zero.

    ``ingested_at`` is captured once and reused for every row and the report.
    ``source_metadata`` supplies ``source_retrieved_at_utc``/``source_sha256``
    (per row, matched by that row's ``source_file``) since Florida's extract
    carries no row-level extract-date column; ``source_snapshot_date`` is
    documented here as the *retrieval* date, not a DBPR-published date.
    """
    now = ingested_at if ingested_at is not None else datetime.now(UTC)
    version = plateproof.__version__

    files_by_name = {
        f.local_filename: f for f in (source_metadata.files if source_metadata else [])
    }
    retrieved_at = source_metadata.retrieved_at_utc if source_metadata is not None else None
    snapshot_date = retrieved_at.date() if retrieved_at is not None else None

    input_row_count = raw.height
    encodings: dict[str, str] = {}
    if input_row_count:
        for file_name, encoding in (
            raw.select("source_file", "source_encoding").unique().iter_rows()
        ):
            if file_name is not None:
                encodings[file_name] = encoding
    files_loaded_count = (
        len({f for f in raw.get_column("source_file").to_list() if f}) if input_row_count else 0
    )

    rows: list[dict[str, Any]] = raw.to_dicts()
    for index, row in enumerate(rows):
        row["__row"] = index
        row["__inspection_date"] = _parse_date(row.get("inspection_date"))
        row["__total_violations"], total_malformed = _parse_nonneg_int(row.get("total_violations"))
        row["__high_priority_count"], hp_malformed = _parse_nonneg_int(
            row.get("high_priority_count")
        )
        row["__intermediate_count"], im_malformed = _parse_nonneg_int(row.get("intermediate_count"))
        row["__basic_count"], bs_malformed = _parse_nonneg_int(row.get("basic_count"))
        row["__visit_number"], visit_malformed = _parse_nonneg_int(row.get("visit_number"))
        category_malformed = False
        for i in range(1, 59):
            key = f"violation_{i:02d}"
            parsed, malformed = _parse_nonneg_int(row.get(key))
            row[f"__{key}"] = parsed
            category_malformed = category_malformed or malformed
        row["__malformed"] = any(
            (
                total_malformed,
                hp_malformed,
                im_malformed,
                bs_malformed,
                visit_malformed,
                category_malformed,
            )
        )

    after_scope = [r for r in rows if _is_food_service(r)]
    non_food_service_removed = len(rows) - len(after_scope)

    after_license = [r for r in after_scope if isinstance(r.get("license_number"), str)]
    missing_license_removed = len(after_scope) - len(after_license)

    after_visit_id = [r for r in after_license if isinstance(r.get("inspection_visit_id"), str)]
    missing_visit_id_removed = len(after_license) - len(after_visit_id)

    after_identity = [
        r
        for r in after_visit_id
        if isinstance(r.get("inspection_number"), str) and r["__visit_number"] is not None
    ]
    invalid_identity_removed = len(after_visit_id) - len(after_identity)

    after_date = [r for r in after_identity if r["__inspection_date"] is not None]
    unparseable_date_removed = len(after_identity) - len(after_date)

    # --- deduplicate / resolve conflicts on inspection_visit_id -------------
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in after_date:
        groups.setdefault(row["inspection_visit_id"], []).append(row)

    survivors: list[dict[str, Any]] = []
    identical_duplicates_removed = 0
    conflicting_rows_removed = 0
    conflicting_ids: list[str] = []
    for visit_id in sorted(groups):
        members = sorted(groups[visit_id], key=lambda r: (r["source_file"] or "", r["__row"]))
        if len(members) == 1:
            survivors.append(members[0])
            continue
        signatures = {tuple(m.get(f) for f in _MATERIAL_FIELDS) for m in members}
        if len(signatures) == 1:
            survivors.append(members[0])
            identical_duplicates_removed += len(members) - 1
        else:
            conflicting_rows_removed += len(members)
            conflicting_ids.append(visit_id)
            if on_conflicting_duplicate == "raise":
                raise ValueError(
                    f"Florida extract has conflicting rows for Inspection Visit ID "
                    f"'{visit_id}': {len(members)} rows disagree on material fields"
                )

    # --- build events ---------------------------------------------------
    event_records: list[dict[str, Any]] = []
    violation_records: list[dict[str, Any]] = []
    total_vs_hib_mismatch = 0
    total_vs_category_mismatch = 0
    malformed_rows = 0
    unknown_disposition = 0

    for row in sorted(survivors, key=lambda r: r["inspection_visit_id"]):
        if row["__malformed"]:
            malformed_rows += 1

        total = row["__total_violations"]
        high = row["__high_priority_count"]
        inter = row["__intermediate_count"]
        basic = row["__basic_count"]
        category_values = [row[f"__violation_{i:02d}"] for i in range(1, 59)]

        if total is not None and None not in (high, inter, basic) and total != high + inter + basic:
            total_vs_hib_mismatch += 1
        if total is not None and all(v is not None for v in category_values):
            if total != sum(v for v in category_values if v is not None):
                total_vs_category_mismatch += 1

        disposition = row.get("inspection_disposition")
        disposition_status = "unknown"
        if isinstance(disposition, str):
            normalized = _normalize_disposition(disposition)
            disposition_status = FL_DISPOSITION_STATUS.get(normalized, "unknown")
        if disposition_status == "unknown":
            unknown_disposition += 1

        license_number = row["license_number"]
        inspection_visit_id = row["inspection_visit_id"]
        inspection_id = f"florida:{inspection_visit_id}"
        restaurant_id = f"florida:{license_number}"
        file_meta = files_by_name.get(row.get("source_file") or "")
        row_sha256 = file_meta.sha256 if file_meta is not None else None

        event_records.append(
            {
                "inspection_id": inspection_id,
                "restaurant_id": restaurant_id,
                "source_id": license_number,
                "jurisdiction": "florida",
                "inspection_date": row["__inspection_date"],
                "inspection_type": row.get("inspection_type"),
                "inspection_type_raw": row.get("inspection_type"),
                "disposition": disposition,
                "disposition_status": disposition_status,
                "native_inspection_group_id": row.get("inspection_number"),
                "native_visit_sequence": row["__visit_number"],
                "violation_count": total,
                "critical_violation_count": None,
                "high_priority_count": high,
                "intermediate_count": inter,
                "basic_count": basic,
                "dba": row.get("dba"),
                "zipcode": row.get("location_zip"),
                "source_dataset": FL_SOURCE_DATASET,
                "source_snapshot_date": snapshot_date,
                "source_retrieved_at_utc": retrieved_at,
                "source_sha256": row_sha256,
                "ingested_at": now,
                "pipeline_version": version,
            }
        )

        for i in range(1, 59):
            count = row[f"__violation_{i:02d}"]
            if not count:
                continue
            code = f"{i:02d}"
            digest_source = f"{inspection_id}\x1f{code}"
            digest = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()[
                :VIOLATION_ID_DIGEST_HEX
            ]
            violation_records.append(
                {
                    "violation_event_id": f"florida:v:{digest}",
                    "inspection_id": inspection_id,
                    "restaurant_id": restaurant_id,
                    "jurisdiction": "florida",
                    "inspection_date": row["__inspection_date"],
                    "violation_code": code,
                    "violation_code_norm": code,
                    "violation_description": FLORIDA_VIOLATION_CATEGORY_DESCRIPTIONS.get(code),
                    "violation_description_norm": None,
                    "critical_flag_raw": None,
                    "severity": map_florida_violation_classification(None),
                    "corrected_on_site": None,
                    "count": count,
                    "source_dataset": FL_SOURCE_DATASET,
                    "source_snapshot_date": snapshot_date,
                    "source_retrieved_at_utc": retrieved_at,
                    "source_sha256": row_sha256,
                    "ingested_at": now,
                    "pipeline_version": version,
                }
            )

    events = finalize_event_frame(event_records, INSPECTION_EVENT_SCHEMA).sort(
        ["restaurant_id", "inspection_date", "inspection_id"]
    )
    violations = finalize_event_frame(violation_records, VIOLATION_EVENT_SCHEMA).sort(
        ["inspection_id", "violation_event_id"]
    )
    assert_unique_inspection_id(events)

    report = FloridaIngestionReport(
        input_row_count=input_row_count,
        output_inspection_count=events.height,
        output_violation_count=violations.height,
        files_loaded_count=files_loaded_count,
        non_food_service_rows_removed=non_food_service_removed,
        missing_license_number_rows_removed=missing_license_removed,
        missing_inspection_visit_id_rows_removed=missing_visit_id_removed,
        invalid_inspection_identity_rows_removed=invalid_identity_removed,
        unparseable_date_rows_removed=unparseable_date_removed,
        identical_duplicate_visit_rows_removed=identical_duplicates_removed,
        conflicting_visit_id_rows_removed=conflicting_rows_removed,
        conflicting_visit_ids=conflicting_ids,
        total_vs_hib_mismatch_count=total_vs_hib_mismatch,
        total_vs_category_sum_mismatch_count=total_vs_category_mismatch,
        malformed_count_field_rows=malformed_rows,
        unknown_disposition_count=unknown_disposition,
        snapshot_date=snapshot_date,
        ingestion_timestamp=now,
        source_encodings=encodings,
    )
    return FloridaIngestionResult(
        inspection_events=events, violation_events=violations, report=report
    )
