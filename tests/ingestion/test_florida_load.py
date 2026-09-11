"""Tests for load_florida_extracts: headers, encoding, XLSX, staging fidelity."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest


def test_reads_real_csv_header_shape(fl_current_csv: Path) -> None:
    from plateproof.ingestion.florida import FloridaExtractSource, load_florida_extracts

    frame = load_florida_extracts([FloridaExtractSource(path=fl_current_csv)])
    assert frame.height == 16
    assert {"license_number", "inspection_visit_id", "inspection_disposition"} <= set(frame.columns)


def test_leading_space_headers_normalized(fl_current_csv: Path) -> None:
    from plateproof.ingestion.florida import FloridaExtractSource, load_florida_extracts

    frame = load_florida_extracts([FloridaExtractSource(path=fl_current_csv)])
    # " License Type Code", " License Number", " Location Zip Code",
    # " Number of Total Violations" all carry a leading space in the real file.
    for canonical in ("license_type_code", "license_number", "location_zip", "total_violations"):
        assert canonical in frame.columns


def test_required_column_missing_raises_named_error(tmp_path: Path) -> None:
    from plateproof.ingestion.florida import FloridaExtractSource, load_florida_extracts

    header = ["District", " License Number", "Inspection Date"]
    path = tmp_path / "missing.csv"
    path.write_text("\n".join([",".join(header), "D1,1000000,01/01/2026"]), encoding="utf-8")
    with pytest.raises(ValueError, match="inspection_visit_id"):
        load_florida_extracts([FloridaExtractSource(path=path)])


def test_required_only_extract_loads(fl_required_only_csv: Path) -> None:
    from plateproof.ingestion.florida import FloridaExtractSource, load_florida_extracts

    frame = load_florida_extracts([FloridaExtractSource(path=fl_required_only_csv)])
    assert frame.height == 1
    for optional in ("district", "dba", "location_address", "pda_status"):
        assert optional in frame.columns
        assert frame.get_column(optional).null_count() == 1


def test_staging_preserves_original_values_under_renamed_headers(fl_current_csv: Path) -> None:
    from plateproof.ingestion.florida import FloridaExtractSource, load_florida_extracts

    frame = load_florida_extracts([FloridaExtractSource(path=fl_current_csv)])
    row = frame.filter(pl.col("inspection_visit_id") == "9000001").to_dicts()[0]
    assert row["license_number"] == "1000001"
    assert row["dba"] == "Anna's Diner"
    assert row["inspection_disposition"] == "Warning Issued"
    assert row["violation_04"] == "1"
    assert row["total_violations"] == "3"


def test_staging_stamps_source_file_and_url(fl_current_csv: Path) -> None:
    from plateproof.ingestion.florida import FloridaExtractSource, load_florida_extracts

    frame = load_florida_extracts(
        [FloridaExtractSource(path=fl_current_csv, source_url="https://example/fl.csv")]
    )
    assert set(frame.get_column("source_file").to_list()) == {"fl_sample_current.csv"}
    assert set(frame.get_column("source_url").to_list()) == {"https://example/fl.csv"}


def test_multiple_files_concatenate_deterministically_by_filename(
    fl_current_csv: Path, fl_overlap_csv: Path
) -> None:
    from plateproof.ingestion.florida import FloridaExtractSource, load_florida_extracts

    forward = load_florida_extracts(
        [FloridaExtractSource(path=fl_current_csv), FloridaExtractSource(path=fl_overlap_csv)]
    )
    reversed_order = load_florida_extracts(
        [FloridaExtractSource(path=fl_overlap_csv), FloridaExtractSource(path=fl_current_csv)]
    )
    assert (
        forward.get_column("source_file").to_list()
        == reversed_order.get_column("source_file").to_list()
    )
    assert forward.height == 17  # 16 + 1


def test_utf8_bom_decodes_correctly(fl_utf8_bom_csv: Path) -> None:
    from plateproof.ingestion.florida import FloridaExtractSource, load_florida_extracts

    frame = load_florida_extracts([FloridaExtractSource(path=fl_utf8_bom_csv)])
    assert frame.get_column("dba").to_list() == ["Café Münchner O'Brien's"]
    assert frame.get_column("source_encoding").to_list() == ["utf-8-sig"]
    # The BOM marker itself must not leak into the first column's values.
    assert frame.get_column("district").to_list() == ["D1"]


def test_windows1252_fallback_decodes_correctly(fl_windows1252_csv: Path) -> None:
    from plateproof.ingestion.florida import FloridaExtractSource, load_florida_extracts

    frame = load_florida_extracts([FloridaExtractSource(path=fl_windows1252_csv)])
    assert frame.get_column("dba").to_list() == ["Mary’s Café — Downtown"]
    assert frame.get_column("source_encoding").to_list() == ["windows-1252"]


def test_decode_failure_raises_clear_error(fl_decode_failure_csv: Path) -> None:
    from plateproof.ingestion.florida import FloridaExtractSource, load_florida_extracts

    with pytest.raises(ValueError, match="decode"):
        load_florida_extracts([FloridaExtractSource(path=fl_decode_failure_csv)])


def test_no_silent_replace_ever_used_for_decoding(fl_windows1252_csv: Path) -> None:
    from plateproof.ingestion.florida import FloridaExtractSource, load_florida_extracts

    frame = load_florida_extracts([FloridaExtractSource(path=fl_windows1252_csv)])
    dba = frame.get_column("dba").to_list()[0]
    assert "�" not in dba  # no U+FFFD replacement character


def test_xlsx_source_reads_into_same_canonical_schema(
    fl_historical_xlsx: Path, fl_current_csv: Path
) -> None:
    from plateproof.ingestion.florida import FloridaExtractSource, load_florida_extracts

    xlsx_frame = load_florida_extracts([FloridaExtractSource(path=fl_historical_xlsx)])
    csv_frame = load_florida_extracts([FloridaExtractSource(path=fl_current_csv)])
    assert set(xlsx_frame.columns) == set(csv_frame.columns)
    assert xlsx_frame.height == 2
    row = xlsx_frame.filter(pl.col("inspection_visit_id") == "8800001").to_dicts()[0]
    assert row["license_number"] == "3000001"
    assert row["total_violations"] == "2"
    assert row["violation_06"] == "1"
    assert row["source_encoding"] == "xlsx"


def test_xlsx_unverified_ht_violations_column_preserved_not_interpreted(
    fl_historical_xlsx: Path,
) -> None:
    from plateproof.ingestion.florida import FloridaExtractSource, load_florida_extracts

    frame = load_florida_extracts([FloridaExtractSource(path=fl_historical_xlsx)])
    assert "ht_violations_unverified" in frame.columns
