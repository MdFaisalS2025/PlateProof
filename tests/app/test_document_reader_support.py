"""Unit tests for app.document_reader_support -- the pure, Streamlit-free
JSON-contract helpers behind the Document Reader page's downloadable
record (independent-review correction of ``c60cc80``, Finding 3).

No Streamlit import anywhere in this file or in the module under test --
these are plain, directly-importable functions, unlike the page itself
(which cannot be imported outside a real ``ScriptRunContext``)."""

from __future__ import annotations

import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

_APP_DIR = Path(__file__).resolve().parent.parent.parent / "app"
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))


def _evidence_span(**overrides: Any) -> Any:
    from plateproof.documents.models import EvidenceSpan

    defaults: dict[str, Any] = dict(
        page=1, excerpt="Score: 14", bounding_box=None, source="embedded_text"
    )
    defaults.update(overrides)
    return EvidenceSpan(**defaults)


def _candidate(**overrides: Any) -> Any:
    from plateproof.documents.models import Candidate, ConfidenceComponents

    defaults: dict[str, Any] = dict(
        field_name="score",
        value=14.0,
        display_value="14",
        evidence=(_evidence_span(),),
        confidence=ConfidenceComponents(
            ocr_confidence=None, pattern_confidence=0.9, corroboration_confidence=None, overall=0.9
        ),
        confidence_label="high",
    )
    defaults.update(overrides)
    return Candidate(**defaults)


def _violation(**overrides: Any) -> Any:
    from plateproof.documents.models import ViolationRowCandidate

    defaults: dict[str, Any] = dict(
        raw_code_text="04L",
        code="04L",
        description="Evidence of mice",
        critical=True,
        evidence=(_evidence_span(excerpt="Violation Code: 04L"),),
        confidence_label="high",
    )
    defaults.update(overrides)
    return ViolationRowCandidate(**defaults)


def _draft(**overrides: Any) -> Any:
    from plateproof.documents.models import ExtractionDraft, OcrEngineInfo, UploadMetadata

    defaults: dict[str, Any] = dict(
        draft_id="draft-1",
        jurisdiction_expected="nyc",
        jurisdiction_detected="nyc",
        jurisdiction_mismatch=False,
        restaurant_id="nyc:1",
        restaurant_identity_corroborated=True,
        upload=UploadMetadata(detected_media_type="application/pdf", byte_size=1234, page_count=1),
        pages=(),
        candidates={"score": _candidate()},
        violations=(_violation(),),
        ambiguities=(),
        missing_fields=(),
        warnings=(),
        ocr_engine=OcrEngineInfo(engine_name="rapidocr", available=True, unavailable_reason=None),
        processing_status="completed",
        generated_at=datetime.now(UTC),
        confirmable=True,
    )
    defaults.update(overrides)
    return ExtractionDraft(**defaults)


def _validated_correction(**overrides: Any) -> Any:
    from plateproof.documents.models import ValidatedCorrection

    defaults: dict[str, Any] = dict(
        raw_display_value="20",
        parsed_value=20.0,
        parse_status="parsed",
        rejection_reason=None,
    )
    defaults.update(overrides)
    return ValidatedCorrection(**defaults)


def test_machine_value_preserved_separately_from_user_correction() -> None:
    """The machine's own proposed value (14.0) must appear unchanged
    alongside a DIFFERENT user correction (20.0) -- never overwritten."""
    from document_reader_support import build_downloadable_record

    draft = _draft()
    corrections = {"score": _validated_correction(raw_display_value="20", parsed_value=20.0)}
    record = build_downloadable_record(draft, corrections)

    assert record["machine_candidates"]["score"]["value"] == 14.0
    assert record["user_corrections"]["score"]["parsed_value"] == 20.0


def test_correction_keeps_raw_text_typed_value_and_status_distinct() -> None:
    from document_reader_support import build_downloadable_record

    draft = _draft()
    corrections = {
        "score": _validated_correction(
            raw_display_value="  20 ", parsed_value=20.0, parse_status="parsed"
        )
    }
    record = build_downloadable_record(draft, corrections)

    entry = record["user_corrections"]["score"]
    assert entry["raw_display_value"] == "  20 "
    assert entry["parsed_value"] == 20.0
    assert entry["parse_status"] == "parsed"
    assert entry["rejection_reason"] is None


def test_rejected_correction_preserves_rejection_reason_and_no_parsed_value() -> None:
    from document_reader_support import build_downloadable_record

    draft = _draft()
    corrections = {
        "score": _validated_correction(
            raw_display_value="nan",
            parsed_value=None,
            parse_status="rejected",
            rejection_reason="value is not a finite number",
        )
    }
    record = build_downloadable_record(draft, corrections)

    entry = record["user_corrections"]["score"]
    assert entry["raw_display_value"] == "nan"
    assert entry["parsed_value"] is None
    assert entry["parse_status"] == "rejected"
    assert entry["rejection_reason"] == "value is not a finite number"


def test_machine_candidate_includes_bounded_evidence() -> None:
    from document_reader_support import build_downloadable_record

    draft = _draft()
    record = build_downloadable_record(draft, {})

    evidence = record["machine_candidates"]["score"]["evidence"]
    assert len(evidence) == 1
    assert evidence[0]["page"] == 1
    assert evidence[0]["excerpt"] == "Score: 14"
    assert evidence[0]["source"] == "embedded_text"


def test_violation_rows_include_evidence() -> None:
    from document_reader_support import build_downloadable_record

    draft = _draft()
    record = build_downloadable_record(draft, {})

    assert len(record["violations"]) == 1
    violation = record["violations"][0]
    assert violation["code"] == "04L"
    assert len(violation["evidence"]) == 1
    assert violation["evidence"][0]["excerpt"] == "Violation Code: 04L"


def test_record_never_contains_raw_bytes_or_preview_png() -> None:
    """Recursively scans the whole record for any bytes value -- a preview
    PNG or any other binary blob must never appear in the downloadable
    artifact."""
    from document_reader_support import build_downloadable_record

    draft = _draft()
    record = build_downloadable_record(draft, {"score": _validated_correction()})

    def _scan(value: Any) -> None:
        assert not isinstance(value, (bytes, bytearray))
        if isinstance(value, dict):
            for v in value.values():
                _scan(v)
        elif isinstance(value, list):
            for v in value:
                _scan(v)

    _scan(record)


def test_confirmed_at_is_timezone_aware_utc() -> None:
    from document_reader_support import build_downloadable_record

    draft = _draft()
    record = build_downloadable_record(draft, {})

    parsed = datetime.fromisoformat(record["confirmed_at"])
    assert parsed.tzinfo is not None
    assert parsed.utcoffset().total_seconds() == 0


def test_record_is_deterministically_json_serializable_with_sorted_keys() -> None:
    import json

    from document_reader_support import build_downloadable_record

    draft = _draft()
    record = build_downloadable_record(draft, {"score": _validated_correction()})

    first = json.dumps(record, indent=2, sort_keys=True)
    second = json.dumps(record, indent=2, sort_keys=True)
    assert first == second


def test_record_status_and_disclaimer_present() -> None:
    from document_reader_support import build_downloadable_record

    draft = _draft()
    record = build_downloadable_record(draft, {})

    assert record["record_status"] == "user_submitted"
    assert "not an official inspection record" in record["disclaimer"]


def test_asdict_json_safe_converts_date_and_drops_bytes() -> None:
    from document_reader_support import asdict_json_safe

    assert asdict_json_safe(date(2024, 1, 15)) == "2024-01-15"
    assert asdict_json_safe(b"raw bytes") is None
    assert asdict_json_safe({"a": b"x", "b": date(2024, 1, 1)}) == {"a": None, "b": "2024-01-01"}
