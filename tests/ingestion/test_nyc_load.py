"""Tests for load_nyc_raw: header normalization, typing, column policy."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import polars as pl
import pytest

_MIN_HEADER = [
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
]
_MIN_ROW = [
    "50190663",
    "2024-05-10T00:00:00.000",
    "Cycle Inspection / Initial Inspection",
    "Violations were cited in the following area(s).",
    "04L",
    "Evidence of mice",
    "Critical",
    "13",
    "A",
    "2024-05-10T00:00:00.000",
    "2024-07-01T00:00:00.000",
]

CsvWriter = Callable[[Path, list[str], list[list[str]]], Path]


def test_reads_soda_headers(soda_csv: Path) -> None:
    from plateproof.ingestion.nyc import load_nyc_raw

    frame = load_nyc_raw(soda_csv)
    assert frame.height == 14
    assert {"camis", "inspection_date", "violation_code"} <= set(frame.columns)


def test_camis_preserved_exactly_not_padded(soda_csv: Path) -> None:
    from plateproof.ingestion.nyc import load_nyc_raw

    frame = load_nyc_raw(soda_csv)
    values = set(frame.get_column("camis").drop_nulls().to_list())
    assert "50190663" in values
    assert all(len(v) == len(v.strip()) for v in values)
    assert not any(v.startswith("00") for v in values)


def test_required_column_missing_raises_named_error(tmp_path: Path, write_csv: CsvWriter) -> None:
    from plateproof.ingestion.nyc import load_nyc_raw

    header = [c for c in _MIN_HEADER if c != "score"]
    row = [v for c, v in zip(_MIN_HEADER, _MIN_ROW, strict=True) if c != "score"]
    path = write_csv(tmp_path / "no_score.csv", header, [row])
    with pytest.raises(ValueError, match="score"):
        load_nyc_raw(path)


def test_missing_optional_column_becomes_null_column(tmp_path: Path, write_csv: CsvWriter) -> None:
    from plateproof.ingestion.nyc import load_nyc_raw

    path = write_csv(tmp_path / "min.csv", _MIN_HEADER, [_MIN_ROW])
    frame = load_nyc_raw(path)
    for optional in ("dba", "boro", "latitude", "community_board", "bbl", "location"):
        assert optional in frame.columns
        assert frame.get_column(optional).null_count() == frame.height


def test_required_only_extract_loads(tmp_path: Path, write_csv: CsvWriter) -> None:
    from plateproof.ingestion.nyc import load_nyc_raw

    path = write_csv(tmp_path / "min.csv", _MIN_HEADER, [_MIN_ROW])
    frame = load_nyc_raw(path)
    assert frame.height == 1
    assert frame.get_column("inspection_date").dtype == pl.Date
    assert frame.get_column("score").dtype == pl.Float64


def test_display_headers_normalize_to_same_schema(soda_csv: Path, display_csv: Path) -> None:
    from plateproof.ingestion.nyc import load_nyc_raw

    soda = load_nyc_raw(soda_csv)
    display = load_nyc_raw(display_csv)
    assert set(soda.columns) == set(display.columns)
    assert soda.schema == display.schema


def test_display_slash_date_format_parses(display_csv: Path) -> None:
    from plateproof.ingestion.nyc import load_nyc_raw

    frame = load_nyc_raw(display_csv)
    parsed = frame.get_column("record_date").drop_nulls()
    assert parsed.dtype == pl.Date
    assert parsed.len() >= 13


def test_duplicate_header_mapping_raises(tmp_path: Path, write_csv: CsvWriter) -> None:
    from plateproof.ingestion.nyc import load_nyc_raw

    header = ["camis", "CAMIS", *[c for c in _MIN_HEADER if c != "camis"]]
    row = [
        "50190663",
        "50190663",
        *[v for c, v in zip(_MIN_HEADER, _MIN_ROW, strict=True) if c != "camis"],
    ]
    path = write_csv(tmp_path / "dupe.csv", header, [row])
    with pytest.raises(ValueError, match="camis"):
        load_nyc_raw(path)


def test_unexpected_column_is_preserved(tmp_path: Path, write_csv: CsvWriter) -> None:
    from plateproof.ingestion.nyc import load_nyc_raw

    header = [*_MIN_HEADER, "weird_extra_col"]
    path = write_csv(tmp_path / "extra.csv", header, [[*_MIN_ROW, "keep-me"]])
    frame = load_nyc_raw(path)
    assert "weird_extra_col" in frame.columns
    assert frame.get_column("weird_extra_col").to_list() == ["keep-me"]


def test_ignored_computed_region_column_is_dropped(tmp_path: Path, write_csv: CsvWriter) -> None:
    from plateproof.ingestion.nyc import load_nyc_raw

    header = [*_MIN_HEADER, ":@computed_region_f5dn_yrer"]
    path = write_csv(tmp_path / "computed.csv", header, [[*_MIN_ROW, "42"]])
    frame = load_nyc_raw(path)
    assert ":@computed_region_f5dn_yrer" not in frame.columns


def test_placeholder_and_invalid_rows_are_kept_by_loader(soda_csv: Path) -> None:
    from plateproof.ingestion.nyc import load_nyc_raw

    frame = load_nyc_raw(soda_csv)
    assert frame.height == 14  # loader removes nothing
