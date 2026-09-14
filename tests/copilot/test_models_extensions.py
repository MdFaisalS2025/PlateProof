"""RED-first tests for Task 8B's additive extensions to
plateproof.copilot.models: richer Citation provenance fields, the new
CopilotAnswer.local_helper_status field, and RefusalReason.INVALID_QUESTION.
All new fields are additive/optional so existing Task 8A construction
sites keep working unchanged."""

from __future__ import annotations

from datetime import UTC, date, datetime

from plateproof.copilot.models import (
    Citation,
    CopilotAnswer,
    EvidenceType,
    Refusal,
    RefusalReason,
)


def test_citation_still_constructs_with_only_the_original_task_8a_fields() -> None:
    """Backward compatibility: no existing call site should need to change."""
    citation = Citation(
        citation_id="record:nyc:1:latest_inspection",
        evidence_type=EvidenceType.RESTAURANT_RECORD,
        title="PlateProof documented record",
        url=None,
        excerpt="Documented inspection.",
        jurisdiction="nyc",
        as_of_date=date(2025, 1, 1),
    )
    assert citation.issuing_authority is None
    assert citation.section_locator is None
    assert citation.access_date is None
    assert citation.effective_date is None
    assert citation.revision_date is None


def test_citation_accepts_full_guidance_provenance() -> None:
    citation = Citation(
        citation_id="fictional-doc#overview",
        evidence_type=EvidenceType.GUIDANCE_PASSAGE,
        title="Fictional Guidance Document",
        url="https://www.nyc.gov/fictional-test-path",
        excerpt="Fictional guidance text.",
        jurisdiction="nyc",
        as_of_date=date(2026, 1, 1),
        issuing_authority="Fictional Testing Authority",
        section_locator="Section 1",
        access_date=date(2026, 1, 1),
        effective_date=date(2025, 6, 1),
        revision_date=date(2025, 7, 1),
    )
    assert citation.issuing_authority == "Fictional Testing Authority"
    assert citation.section_locator == "Section 1"
    assert citation.effective_date == date(2025, 6, 1)
    assert citation.revision_date == date(2025, 7, 1)


def test_copilot_answer_defaults_local_helper_status_to_not_consulted() -> None:
    answer = CopilotAnswer(
        restaurant_id="nyc:1",
        jurisdiction="nyc",
        intent=None,
        answer_text="text",
        claims=(),
        citations=(),
        generator_mode="deterministic",
        grounding_status="refused",
        refusal=Refusal(reason=RefusalReason.UNKNOWN_INTENT, detail="detail"),
        warnings=(),
        generated_at=datetime.now(UTC),
    )
    assert answer.local_helper_status == "not_consulted"


def test_copilot_answer_accepts_explicit_local_helper_status() -> None:
    answer = CopilotAnswer(
        restaurant_id="nyc:1",
        jurisdiction="nyc",
        intent=None,
        answer_text="text",
        claims=(),
        citations=(),
        generator_mode="local_llm_assisted",
        grounding_status="grounded",
        refusal=None,
        warnings=(),
        generated_at=datetime.now(UTC),
        local_helper_status="accepted",
    )
    assert answer.local_helper_status == "accepted"


def test_refusal_reason_has_invalid_question_member() -> None:
    assert RefusalReason.INVALID_QUESTION == "invalid_question"
