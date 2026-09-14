"""Shared construction helpers for the pieces every Copilot-serving
surface (the FastAPI app, the Streamlit UI) needs: a validated
``CorpusStore`` (or ``None``) and an optional local intent helper. Both
are built exactly once per process/settings by the caller (``app.state`` in
``plateproof.api.main``, a module-level cache in ``app.theme``) -- never
rebuilt per request.

Every Task 8B local-AI setting is explicitly bounded-checked here
(:func:`validate_local_llm_settings`) -- but, deliberately, an invalid
value never raises and never prevents ``Settings`` itself from
constructing (see ``plateproof.core.config`` -- only the always-relevant
``copilot_max_question_length`` is a hard Settings-construction failure).
An invalid *optional* local-AI configuration instead makes
:func:`build_intent_helper` return ``None`` (the helper is simply not
constructed) and :func:`local_ai_health_status` report a truthful
``"unavailable"`` -- never a crashed or unusable application.
"""

from __future__ import annotations

import math

from plateproof.copilot.corpus import CorpusStore, load_corpus
from plateproof.copilot.generators.base import IntentHelper
from plateproof.copilot.generators.ollama import OllamaIntentHelper, validate_loopback_url
from plateproof.copilot.intent_validation import MAX_POSSIBLE_CONFIDENCE, MIN_POSSIBLE_CONFIDENCE
from plateproof.core.config import Settings

# Generous headroom over any real usage -- these exist only to reject a
# clearly-nonsensical administrator value (a negative/zero/absurdly large
# timeout, an unbounded response size), not to constrain normal tuning.
_MAX_REASONABLE_CONNECT_TIMEOUT_SECONDS = 30.0
_MAX_REASONABLE_READ_TIMEOUT_SECONDS = 60.0
_MAX_REASONABLE_RESPONSE_BYTES = 10_000_000
_MAX_MODEL_NAME_LENGTH = 200


def build_corpus_store(settings: Settings) -> CorpusStore | None:
    """``None`` both when the corpus isn't configured and when it fails
    validation -- either way, guidance-dependent intents fail closed to an
    honest ``GUIDANCE_UNAVAILABLE`` claim rather than the process
    crashing."""
    if settings.guidance_corpus_manifest_path is None:
        return None
    manifest_path = settings.resolve_path(settings.guidance_corpus_manifest_path)
    return load_corpus(manifest_path).store


def _is_valid_model_name(model: str) -> bool:
    if not model or len(model) > _MAX_MODEL_NAME_LENGTH:
        return False
    return all(ord(char) >= 0x20 for char in model)


def validate_local_llm_settings(settings: Settings) -> tuple[bool, str | None]:
    """Validates every bounded Task 8B local-AI setting together. Never
    raises. Called both by :func:`build_intent_helper` (to decide whether
    to construct a helper at all) and by :func:`local_ai_health_status`
    (to report a truthful status without ever opening a connection)."""
    if not settings.local_llm_model or not _is_valid_model_name(settings.local_llm_model):
        return False, "local model name is missing or invalid"

    connect_timeout = settings.local_llm_connect_timeout_seconds
    if (
        not math.isfinite(connect_timeout)
        or connect_timeout <= 0
        or connect_timeout > _MAX_REASONABLE_CONNECT_TIMEOUT_SECONDS
    ):
        return False, "configured connect timeout is invalid"

    read_timeout = settings.local_llm_read_timeout_seconds
    if (
        not math.isfinite(read_timeout)
        or read_timeout <= 0
        or read_timeout > _MAX_REASONABLE_READ_TIMEOUT_SECONDS
    ):
        return False, "configured read timeout is invalid"

    max_bytes = settings.local_llm_max_response_bytes
    if max_bytes <= 0 or max_bytes > _MAX_REASONABLE_RESPONSE_BYTES:
        return False, "configured maximum response size is invalid"

    min_confidence = settings.local_llm_min_confidence
    if not math.isfinite(min_confidence) or not (
        MIN_POSSIBLE_CONFIDENCE <= min_confidence <= MAX_POSSIBLE_CONFIDENCE
    ):
        return False, "configured minimum confidence is invalid"

    endpoint, reason = validate_loopback_url(settings.local_llm_base_url)
    if endpoint is None:
        return False, "configured base URL is invalid"

    return True, None


def build_intent_helper(settings: Settings) -> IntentHelper | None:
    """``None`` unless local assistance is enabled AND every setting
    passes :func:`validate_local_llm_settings` -- an enabled-but-
    misconfigured local AI is treated exactly like "not configured", never
    as a helper that would immediately fail every request."""
    if not settings.local_llm_enabled:
        return None
    valid, _reason = validate_local_llm_settings(settings)
    if not valid:
        return None
    assert settings.local_llm_model is not None
    return OllamaIntentHelper(
        base_url=settings.local_llm_base_url,
        model=settings.local_llm_model,
        connect_timeout=settings.local_llm_connect_timeout_seconds,
        read_timeout=settings.local_llm_read_timeout_seconds,
        max_response_bytes=settings.local_llm_max_response_bytes,
        min_confidence=settings.local_llm_min_confidence,
    )


def local_ai_health_status(settings: Settings) -> tuple[str, str | None]:
    """Truthful, network-free local-AI health status:
    ``"disabled"`` (not enabled) | ``"configured_unverified"`` (enabled,
    every setting is structurally valid, but no live probe was made --
    /health never contacts Ollama) | ``"unavailable"`` (enabled but
    misconfigured -- a helper could not be constructed at all)."""
    if not settings.local_llm_enabled:
        return "disabled", None
    valid, reason = validate_local_llm_settings(settings)
    if not valid:
        return "unavailable", reason
    return "configured_unverified", None
