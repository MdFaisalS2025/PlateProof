"""The deterministic renderer: the ONLY code that turns
:class:`~plateproof.copilot.models.Claim` objects into ``answer_text``.
Templates are fixed, reviewed Python functions keyed by
``(claim_type, render_template_id)`` -- never generated at runtime, never
influenced by freeform model output. This is what makes every factual
sentence traceable: it was built from a claim's ``values``, which were
themselves copied verbatim from graph/corpus data by
``plateproof.copilot.claims``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from plateproof.copilot.models import Claim, ClaimType


def _render_restaurant_identity_v1(v: Mapping[str, object]) -> str:
    parts = [f"{v.get('name') or 'This restaurant'} is documented in {v.get('jurisdiction')}."]
    if v.get("city"):
        location = str(v["city"])
        if v.get("postal_code"):
            location += f", {v['postal_code']}"
        parts.append(f"Documented location: {location}.")
    if v.get("cuisine"):
        parts.append(f"Documented cuisine: {v['cuisine']}.")
    return " ".join(parts)


def _render_latest_inspection_v1(v: Mapping[str, object]) -> str:
    same_day_raw = v.get("same_day_count") or 1
    same_day = same_day_raw if isinstance(same_day_raw, int) else 1
    suffix = f" ({same_day} inspections recorded that day)" if same_day > 1 else ""
    inspection_date = v["inspection_date"]
    parts = [f"The most recently documented inspection was on {inspection_date}{suffix}."]
    if v.get("inspection_type"):
        parts.append(f"Inspection type: {v['inspection_type']}.")
    if v.get("score") is not None:
        parts.append(f"Score: {v['score']}.")
    if v.get("grade"):
        parts.append(f"Grade: {v['grade']}.")
    return " ".join(parts)


def _render_recurring_violation_v1(v: Mapping[str, object]) -> str:
    description_date = v.get("description_date")
    date_phrase = (
        f"most recently on {description_date}" if description_date else "on an undocumented date"
    )
    return (
        f"Violation code {v['violation_code']} has been documented on "
        f"{v['occurrence_count']} separate inspection(s) (cited {v['total_cited_count']} "
        f"time(s) in total), {date_phrase}."
    )


def _render_violation_frequency_v1(v: Mapping[str, object]) -> str:
    return (
        f"Violation code {v['violation_code']}: documented on {v['occurrence_count']} "
        f"inspection(s), {v['total_cited_count']} total citation(s)."
    )


def _render_history_trend_v1(v: Mapping[str, object]) -> str:
    return (
        f"{v['inspection_count']} inspections are documented between {v['earliest_date']} "
        f"and {v['latest_date']}. Earliest documented score/grade: "
        f"{v.get('earliest_score')}/{v.get('earliest_grade')}. Latest documented "
        f"score/grade: {v.get('latest_score')}/{v.get('latest_grade')}."
    )


def _render_michelin_context_v1(v: Mapping[str, object]) -> str:
    return (
        f"Michelin {v['distinction']} recognition is documented in the {v['guide_year']} "
        f"{v['guide_name']}. This is contextual recognition and does not indicate food "
        "safety or predict inspection outcomes."
    )


def _render_forecast_availability_v1(v: Mapping[str, object]) -> str:
    return (
        f"A PlateProof forecast is available: {v.get('risk_band', 'unavailable')} predicted "
        f"risk band (model version {v.get('model_version')}, generated "
        f"{v.get('generated_at')}). This is a statistical estimate, not a guarantee of any "
        "future inspection outcome."
    )


def _render_guidance_for_code_v1(v: Mapping[str, object]) -> str:
    excerpts_raw = v.get("excerpts") or []
    excerpts = excerpts_raw if isinstance(excerpts_raw, list) else []
    subject = v.get("violation_code") or v.get("topic") or "this topic"
    joined = " ".join(str(e) for e in excerpts)
    return f"Official guidance related to {subject}: {joined}"


def _render_guidance_unavailable_v1(v: Mapping[str, object]) -> str:
    subject = v.get("violation_code") or v.get("topic") or "this topic"
    return f"No reviewed official guidance is currently available for {subject}."


_RENDERERS: Mapping[tuple[ClaimType, str], Callable[[Mapping[str, object]], str]] = {
    (ClaimType.RESTAURANT_IDENTITY, "restaurant_identity_v1"): _render_restaurant_identity_v1,
    (ClaimType.LATEST_INSPECTION, "latest_inspection_v1"): _render_latest_inspection_v1,
    (ClaimType.RECURRING_VIOLATION, "recurring_violation_v1"): _render_recurring_violation_v1,
    (ClaimType.VIOLATION_FREQUENCY, "violation_frequency_v1"): _render_violation_frequency_v1,
    (ClaimType.HISTORY_TREND, "history_trend_v1"): _render_history_trend_v1,
    (ClaimType.MICHELIN_CONTEXT, "michelin_context_v1"): _render_michelin_context_v1,
    (ClaimType.FORECAST_AVAILABILITY, "forecast_availability_v1"): _render_forecast_availability_v1,
    (ClaimType.GUIDANCE_FOR_CODE, "guidance_for_code_v1"): _render_guidance_for_code_v1,
    (ClaimType.GUIDANCE_UNAVAILABLE, "guidance_unavailable_v1"): _render_guidance_unavailable_v1,
}


def render_claim(claim: Claim) -> str:
    renderer = _RENDERERS.get((claim.claim_type, claim.render_template_id))
    if renderer is None:
        raise ValueError(
            f"no renderer registered for {claim.claim_type}/{claim.render_template_id}"
        )
    return renderer(claim.values)


def render_answer(claims: tuple[Claim, ...]) -> str:
    return " ".join(render_claim(claim) for claim in claims)
