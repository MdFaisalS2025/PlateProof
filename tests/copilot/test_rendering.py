"""Tests for plateproof.copilot.rendering wording, in particular the
guidance-unavailable correction (independent-review item 5): the
statement must describe PlateProof's own corpus state, never imply that
no official guidance exists anywhere.
"""

from __future__ import annotations

from datetime import date

from plateproof.copilot.claims import guidance_unavailable_claim
from plateproof.copilot.rendering import render_claim


def test_guidance_unavailable_wording_is_scoped_to_plateproofs_corpus() -> None:
    claim = guidance_unavailable_claim(
        jurisdiction="nyc", violation_code="04L", topic=None, as_of_date=date(2025, 1, 1)
    )
    text = render_claim(claim)
    assert "PlateProof does not currently have reviewed official guidance mapped to" in text
    assert "documented code 04L" in text
    # Must never claim guidance doesn't exist anywhere -- only that
    # PlateProof's own reviewed corpus has nothing mapped.
    assert "no reviewed official guidance is currently available" not in text.lower()
    assert "does not exist" not in text.lower()


def test_guidance_unavailable_wording_uses_topic_when_no_code_given() -> None:
    claim = guidance_unavailable_claim(
        jurisdiction="nyc", violation_code=None, topic="high_priority", as_of_date=date(2025, 1, 1)
    )
    text = render_claim(claim)
    assert "topic 'high_priority'" in text
    assert "PlateProof does not currently have reviewed official guidance mapped to" in text
