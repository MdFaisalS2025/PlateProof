"""Question hygiene, shared by every boundary that ever sees a raw
question string: ``CopilotService.answer`` (the service-level backstop),
``plateproof.api.schemas.CopilotQueryRequest`` (the HTTP-transport 422
boundary), and the Ollama prompt builder (defense in depth before an
already-validated string is embedded in a local-model prompt). A single
shared function means these three boundaries can never quietly drift apart
on what counts as an acceptable question.

Normalization is deliberately narrow and deterministic:

* CRLF/CR line endings are normalized to LF.
* A tab is normalized to a single space.
* Any other C0 control character (NUL included) is REJECTED outright, not
  silently stripped -- silently altering what a user typed would hide what
  was actually sent to the deterministic detector or the local model.

A rejection reason is always a short, static, generic string. It never
echoes any part of the offending raw input -- the caller's job is to show
the user a safe, fixed message, never a diagnostic built from what they
typed.
"""

from __future__ import annotations

_ALLOWED_AFTER_NORMALIZATION = {"\n"}


def sanitize_question(question: str, *, max_length: int) -> tuple[str | None, str | None]:
    """Returns ``(sanitized, None)`` on success or ``(None, reason)`` on
    rejection. Length is checked against the original (pre-strip) input so
    a caller's configured bound is exact and not gameable by leading/
    trailing whitespace."""
    if len(question) > max_length:
        return None, "question exceeds the maximum allowed length"

    normalized = question.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    for char in normalized:
        if ord(char) < 0x20 and char not in _ALLOWED_AFTER_NORMALIZATION:
            return None, "question contains a disallowed control character"

    stripped = normalized.strip()
    if not stripped:
        return None, "question is empty"
    return stripped, None
