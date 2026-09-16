"""Tests for the frozen, kw_only-only typed contracts in plateproof.documents.models."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

import pytest


def test_all_models_are_frozen_and_kw_only() -> None:
    from plateproof.documents import models

    checked = 0
    for name in dir(models):
        obj = getattr(models, name)
        if dataclasses.is_dataclass(obj) and isinstance(obj, type):
            params = obj.__dataclass_params__
            assert params.frozen is True, f"{name} must be frozen"
            assert params.kw_only is True, f"{name} must be kw_only"
            checked += 1
    assert checked >= 10


def test_bounding_box_construction_is_keyword_only() -> None:
    from plateproof.documents.models import BoundingBox

    box = BoundingBox(x0=1.0, y0=2.0, x1=3.0, y1=4.0)
    assert box.x1 > box.x0
    with pytest.raises(TypeError):
        BoundingBox(1.0, 2.0, 3.0, 4.0)  # type: ignore[misc]


def test_confirmed_document_record_status_defaults_to_user_submitted() -> None:
    from plateproof.documents.models import ConfirmedDocumentRecord

    record = ConfirmedDocumentRecord(
        restaurant_id="nyc:123",
        confirmed_at=datetime.now(UTC),
        machine_candidates={},
        user_corrections={},
        disclaimer="This is a user-submitted record, not an official inspection result.",
    )
    assert record.record_status == "user_submitted"
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.record_status = "official"  # type: ignore[misc]


def test_candidate_confidence_label_is_closed_enum() -> None:
    from plateproof.documents.models import Candidate, ConfidenceComponents

    candidate = Candidate(
        field_name="score",
        value=14.0,
        display_value="14",
        evidence=(),
        confidence=ConfidenceComponents(
            ocr_confidence=0.9, pattern_confidence=0.9, corroboration_confidence=None, overall=0.9
        ),
        confidence_label="high",
    )
    assert candidate.confidence_label in ("high", "medium", "low")


def test_extraction_draft_processing_status_is_closed_enum() -> None:
    from plateproof.documents.models import (
        ExtractionDraft,
        OcrEngineInfo,
        UploadMetadata,
    )

    draft = ExtractionDraft(
        draft_id="draft-1",
        jurisdiction_expected="nyc",
        jurisdiction_detected="nyc",
        jurisdiction_mismatch=False,
        restaurant_id="nyc:123",
        restaurant_identity_corroborated=True,
        upload=UploadMetadata(detected_media_type="application/pdf", byte_size=100, page_count=1),
        pages=(),
        candidates={},
        violations=(),
        ambiguities=(),
        missing_fields=(),
        warnings=(),
        ocr_engine=OcrEngineInfo(engine_name="rapidocr", available=True, unavailable_reason=None),
        processing_status="completed",
        generated_at=datetime.now(UTC),
        confirmable=False,
    )
    assert draft.processing_status in ("completed", "ocr_unavailable", "failed")
