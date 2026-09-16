"""Non-pickle wire protocol for the parent<->worker document-parsing boundary.

The worker exists to isolate native parsers (PDFium, Pillow, OpenCV,
onnxruntime, RapidOCR) running against hostile, untrusted document bytes. If
any of those had a bug that let a crafted document influence what the worker
sends back, an object-serialization transport (``Connection.send()``/
``recv()``, ``multiprocessing.Queue``, ``ProcessPoolExecutor``, or any
``pickle.loads``/``pickle.load`` call) would let a malicious pickle stream
execute arbitrary code *during unpickling itself* -- before any schema
validation could run. This module therefore never uses those APIs: every
frame crosses the pipe as raw, length-prefixed bytes
(``Connection.send_bytes()``/``recv_bytes()``), decoded with a hardened
``json.loads`` configuration, and validated field-by-field with exact type
checks before becoming one of the closed dataclasses below.

No worker output is ever used to dynamically import or instantiate a class:
every accepted field maps, via a fixed ``if``/``elif`` dispatch over the
closed ``message_type``/field-name set, directly into the hand-written
dataclasses in this module. There is no ``getattr(module, name)``,
``importlib.import_module(worker_supplied_string)``, or ``globals()[name]``
anywhere in this file.

The protocol never carries exception objects, tracebacks, stdout, stderr,
filesystem paths, or the original uploaded filename. A worker-side failure
is a ``job_error`` message with a closed-enum ``error_kind`` only.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from typing import Any, Literal, NoReturn

from plateproof.documents.limits import (
    ABSOLUTE_MAX_PREVIEW_FRAME_BYTES,
    ABSOLUTE_MAX_PREVIEW_PAGES,
    MAX_JSON_FRAME_BYTES,
)

#: Exact protocol version this codebase speaks. A mismatch is a protocol
#: violation, not a negotiation -- there is exactly one supported version.
PROTOCOL_VERSION = 1

#: Primary memory bound for JSON *control* frames (job_request metadata,
#: job_response/job_error/page_progress) -- never the raw document itself
#: (Finding 3: that uses a much larger, separate ceiling -- see
#: ``send_bytes_frame``'s ``max_length`` parameter and
#: ``plateproof.documents.limits.ABSOLUTE_MAX_UPLOAD_BYTES``). Checked
#: against the declared header length *before* the second ``recv_bytes()``
#: call is ever issued.
MAX_WORKER_FRAME_BYTES = MAX_JSON_FRAME_BYTES

#: Structural bound on JSON nesting, enforced by a lexical scan *before*
#: ``json.loads`` is ever called -- not by letting the decoder recurse and
#: catching ``RecursionError`` afterward.
MAX_NESTING_DEPTH = 6

MAX_PAGES_PER_RESPONSE = 100
MAX_TEXT_BLOCKS_PER_PAGE = 500
MAX_TEXT_BLOCK_LENGTH = 5_000

_MESSAGE_TYPES = frozenset(
    {"job_request", "job_response", "job_error", "page_progress", "preview_header"}
)
_ERROR_KINDS = frozenset(
    {
        "pdf_malformed",
        "image_malformed",
        "decode_failed",
        "ocr_failed",
        "ocr_unavailable",
        "unsupported_media_type",
        "page_limit_exceeded",
        "pixel_limit_exceeded",
        "text_limit_exceeded",
        "encrypted_document",
        "internal_error",
    }
)
_JURISDICTIONS = frozenset({"nyc", "florida"})
_MEDIA_TYPES = frozenset({"application/pdf", "image/png", "image/jpeg"})
_TEXT_SOURCES = frozenset({"embedded_text", "ocr"})


class ProtocolViolationError(Exception):
    """Raised for any malformed, truncated, oversized, or otherwise invalid
    frame or message. The message is always one of a small set of static
    reasons -- never the underlying exception's own text, and never any
    fragment of the offending payload. Callers (``worker/pool.py``) catch
    this once and retire the worker that produced it; it is never allowed
    to reach the API/UI layer directly."""


class _ProtocolDecodeError(Exception):
    """Internal only -- raised by the JSON decoder hooks below and always
    converted to :class:`ProtocolViolationError` by :func:`_decode_json_message`."""


# --------------------------------------------------------------------------- #
# Typed messages                                                              #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, kw_only=True)
class WorkerJobRequest:
    expected_jurisdiction: Literal["nyc", "florida"]
    media_type: Literal["application/pdf", "image/png", "image/jpeg"]
    max_pages: int
    max_pixels: int
    ocr_enabled: bool
    page_timeout_seconds: float


@dataclass(frozen=True, kw_only=True)
class WorkerBoundingBox:
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(frozen=True, kw_only=True)
class WorkerTextBlock:
    text: str
    source: Literal["embedded_text", "ocr"]
    ocr_confidence: float | None
    bounding_box: WorkerBoundingBox | None


@dataclass(frozen=True, kw_only=True)
class WorkerPageResult:
    page_number: int
    width_px: int
    height_px: int
    used_ocr: bool
    #: True iff this page had no usable embedded text and OCR was enabled
    #: for the job, so OCR was actually attempted for it -- distinct from
    #: ``used_ocr`` (True only if OCR *succeeded* and contributed text).
    #: ``ocr_attempted and not used_ocr`` means this page needed OCR and
    #: didn't get usable text from it (Finding 5: distinguishes a genuinely
    #: OCR-dependent page from one that simply had no text to report).
    ocr_attempted: bool
    text_blocks: tuple[WorkerTextBlock, ...]


@dataclass(frozen=True, kw_only=True)
class ValidatedPreview:
    """A worker-generated preview image, re-validated on the parent side
    (signature, dimensions, byte length, page range, uniqueness, ordering,
    and total budget) before ever being trusted. Populated by
    ``worker/pool.py`` after receiving and validating the binary frame that
    follows a :class:`WorkerPreviewHeader` -- never present on the instance
    returned directly by :func:`validate_job_response`."""

    page_number: int
    png_bytes: bytes
    width_px: int
    height_px: int


@dataclass(frozen=True, kw_only=True)
class WorkerJobResponse:
    pages: tuple[WorkerPageResult, ...]
    ocr_available: bool
    #: The number of preview_header+binary-frame pairs the worker will send
    #: immediately after this job_response, in page order -- the parent
    #: reads exactly this many, so an extra/missing/duplicate/out-of-order
    #: frame is always detectable as a protocol violation (Finding 7).
    preview_count: int = 0
    #: Populated by worker/pool.py post-receipt; always empty on the
    #: instance validate_job_response itself returns.
    previews: tuple[ValidatedPreview, ...] = ()


@dataclass(frozen=True, kw_only=True)
class WorkerJobError:
    error_kind: str


@dataclass(frozen=True, kw_only=True)
class WorkerTimeout:
    stage: Literal["page", "total"]


@dataclass(frozen=True, kw_only=True)
class WorkerCrashed:
    exit_code: int | None


@dataclass(frozen=True, kw_only=True)
class WorkerInvalidResponse:
    reason: str


@dataclass(frozen=True, kw_only=True)
class WorkerPreviewHeader:
    """Announces the size of the binary preview frame that immediately
    follows this message on the same connection (Finding 7). The parent
    reads the declared ``byte_length`` bytes via ``recv_bytes_frame`` bounded
    by ``ABSOLUTE_MAX_PREVIEW_FRAME_BYTES`` -- never the document ceiling."""

    page_number: int
    byte_length: int


@dataclass(frozen=True, kw_only=True)
class WorkerPageProgress:
    """Sent by the worker immediately after finishing one page, so the
    parent's per-page timeout can be reset from real progress rather than
    trusting the worker's own sense of time."""

    page_number: int


# --------------------------------------------------------------------------- #
# Exact-type helpers -- isinstance is never used for numeric protocol        #
# fields, because bool is a subclass of int in Python.                       #
# --------------------------------------------------------------------------- #


def _exact_int(value: object) -> int | None:
    if type(value) is bool:
        return None
    if type(value) is int:
        return value
    return None


def _exact_finite_number(value: object) -> float | None:
    if type(value) is bool:
        return None
    if type(value) is int or type(value) is float:
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def _exact_bool(value: object) -> bool | None:
    if type(value) is bool:
        return value
    return None


def _exact_str(value: object, *, max_length: int | None = None) -> str | None:
    if type(value) is not str:
        return None
    if max_length is not None and len(value) > max_length:
        return None
    return value


# --------------------------------------------------------------------------- #
# JSON decoding -- hardened configuration                                    #
# --------------------------------------------------------------------------- #


def _reject_non_finite(token: str) -> NoReturn:
    # json.loads invokes parse_constant only for the literal tokens "NaN",
    # "Infinity", "-Infinity" -- there is no valid protocol field this could
    # ever legitimately produce.
    raise _ProtocolDecodeError("non-finite numeric token in protocol payload")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    # object_pairs_hook is invoked once per JSON object at EVERY nesting
    # level, so this rejects a duplicate key anywhere in the document.
    seen: set[str] = set()
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise _ProtocolDecodeError("duplicate key in protocol payload")
        seen.add(key)
        result[key] = value
    return result


def _check_nesting_depth(text: str) -> None:
    """Walk the raw text once, tracking depth on ``{``/``[``/``}``/``]``
    *outside* of string literals, with correct backslash-escape handling.
    Runs before ``json.loads`` is ever called."""
    depth = 0
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            depth += 1
            if depth > MAX_NESTING_DEPTH:
                raise ProtocolViolationError("payload nesting exceeds the maximum allowed depth")
        elif char in "}]":
            depth -= 1


def _decode_json_message(text: str) -> dict[str, Any]:
    _check_nesting_depth(text)
    try:
        parsed = json.loads(
            text,
            parse_constant=_reject_non_finite,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (json.JSONDecodeError, _ProtocolDecodeError, RecursionError) as exc:
        raise ProtocolViolationError("payload could not be decoded as valid protocol JSON") from exc
    if not isinstance(parsed, dict):
        raise ProtocolViolationError("payload top level must be a JSON object")
    return parsed


# --------------------------------------------------------------------------- #
# Framing                                                                     #
# --------------------------------------------------------------------------- #


def send_frame(conn: Any, message: dict[str, Any]) -> None:
    """Send ``message`` as two separate ``send_bytes()`` calls: a 4-byte
    big-endian length header, then the payload. Deterministic encoding
    (sorted keys, fixed separators) so identical messages produce
    byte-identical frames."""
    payload = json.dumps(message, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(payload) > MAX_WORKER_FRAME_BYTES:
        raise ProtocolViolationError("outgoing frame exceeds the maximum allowed size")
    conn.send_bytes(struct.pack(">I", len(payload)))
    conn.send_bytes(payload)


def send_bytes_frame(conn: Any, data: bytes, *, max_length: int = MAX_WORKER_FRAME_BYTES) -> None:
    """Send a raw binary frame using the identical two-message
    header+payload algorithm as :func:`send_frame`, but with no JSON
    encoding -- this frame carries opaque bytes, not structured data.

    ``max_length`` defaults to the small JSON-control-frame ceiling as a
    safe default; a caller sending the document itself or a preview image
    must pass the appropriate larger/smaller ceiling explicitly (Finding 3
    -- JSON, document, and preview frames are three distinct budgets that
    must never be conflated). The sender and receiver must agree on the
    same ``max_length`` for a given frame; the receiver's own ``max_length``
    on :func:`recv_bytes_frame` is authoritative and is never widened by
    anything the sender claims.
    """
    if len(data) > max_length:
        raise ProtocolViolationError("outgoing binary frame exceeds the maximum allowed size")
    conn.send_bytes(struct.pack(">I", len(data)))
    conn.send_bytes(data)


def recv_bytes_frame(conn: Any, *, max_length: int) -> bytes:
    """Receive a raw binary frame. ``max_length`` is checked against the
    declared header length *before* the second ``recv_bytes()`` call, so an
    oversized declared length never causes an oversized allocation attempt."""
    try:
        header = conn.recv_bytes(maxlength=4)
    except (OSError, EOFError) as exc:
        raise ProtocolViolationError("connection error while reading binary frame header") from exc
    if len(header) != 4:
        raise ProtocolViolationError("binary frame header was not exactly 4 bytes")

    try:
        declared_length = struct.unpack(">I", header)[0]
    except struct.error as exc:
        raise ProtocolViolationError("binary frame header could not be decoded") from exc

    if declared_length > max_length:
        raise ProtocolViolationError(
            "declared binary frame length exceeds the maximum allowed size"
        )

    try:
        payload = conn.recv_bytes(maxlength=declared_length)
    except (OSError, EOFError) as exc:
        raise ProtocolViolationError("connection error while reading binary frame payload") from exc
    if len(payload) != declared_length:
        raise ProtocolViolationError(
            "binary frame payload length did not match its declared header"
        )
    return bytes(payload)


def recv_frame(conn: Any) -> dict[str, Any]:
    """Receive one frame: a 4-byte header via ``recv_bytes(maxlength=4)``,
    then the declared-length payload via a second ``recv_bytes()`` call.
    Every failure mode -- short/long header or payload, a connection error,
    invalid UTF-8, or a JSON decode/schema failure -- is normalized to the
    same :class:`ProtocolViolationError` with a static message."""
    try:
        header = conn.recv_bytes(maxlength=4)
    except (OSError, EOFError) as exc:
        raise ProtocolViolationError("connection error while reading frame header") from exc
    if len(header) != 4:
        raise ProtocolViolationError("frame header was not exactly 4 bytes")

    try:
        declared_length = struct.unpack(">I", header)[0]
    except struct.error as exc:
        raise ProtocolViolationError("frame header could not be decoded") from exc

    if declared_length > MAX_WORKER_FRAME_BYTES:
        raise ProtocolViolationError("declared frame length exceeds the maximum allowed size")

    try:
        payload = conn.recv_bytes(maxlength=declared_length)
    except (OSError, EOFError) as exc:
        raise ProtocolViolationError("connection error while reading frame payload") from exc
    if len(payload) != declared_length:
        raise ProtocolViolationError("frame payload length did not match its declared header")

    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ProtocolViolationError("frame payload was not valid UTF-8") from exc

    return _decode_json_message(text)


# --------------------------------------------------------------------------- #
# Schema validation -- fixed dispatch, no dynamic import/instantiation       #
# --------------------------------------------------------------------------- #


def _validate_bounding_box(
    raw: object, *, width_px: int, height_px: int
) -> WorkerBoundingBox | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ProtocolViolationError("bounding_box must be a JSON object or null")
    unexpected = set(raw.keys()) - {"x0", "y0", "x1", "y1"}
    if unexpected:
        raise ProtocolViolationError("bounding_box contains an unexpected field")
    x0 = _exact_finite_number(raw.get("x0"))
    y0 = _exact_finite_number(raw.get("y0"))
    x1 = _exact_finite_number(raw.get("x1"))
    y1 = _exact_finite_number(raw.get("y1"))
    if x0 is None or y0 is None or x1 is None or y1 is None:
        raise ProtocolViolationError("bounding_box coordinates must be finite numbers")
    if not (x0 < x1 and y0 < y1):
        raise ProtocolViolationError("bounding_box coordinates are inverted")
    if x0 < 0 or y0 < 0 or x1 > width_px or y1 > height_px:
        raise ProtocolViolationError("bounding_box falls outside the page's own pixel dimensions")
    return WorkerBoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)


def _validate_text_block(raw: object, *, width_px: int, height_px: int) -> WorkerTextBlock:
    if not isinstance(raw, dict):
        raise ProtocolViolationError("text_block must be a JSON object")
    unexpected = set(raw.keys()) - {"text", "source", "ocr_confidence", "bounding_box"}
    if unexpected:
        raise ProtocolViolationError("text_block contains an unexpected field")

    text = _exact_str(raw.get("text"), max_length=MAX_TEXT_BLOCK_LENGTH)
    if text is None:
        raise ProtocolViolationError("text_block.text must be a bounded string")

    source = raw.get("source")
    if source not in _TEXT_SOURCES:
        raise ProtocolViolationError("text_block.source is not one of the supported closed values")

    raw_confidence = raw.get("ocr_confidence")
    if raw_confidence is None:
        ocr_confidence: float | None = None
    else:
        ocr_confidence = _exact_finite_number(raw_confidence)
        if ocr_confidence is None or not (0.0 <= ocr_confidence <= 1.0):
            raise ProtocolViolationError(
                "text_block.ocr_confidence must be a finite number in [0, 1]"
            )

    bounding_box = _validate_bounding_box(
        raw.get("bounding_box"), width_px=width_px, height_px=height_px
    )
    return WorkerTextBlock(
        text=text, source=source, ocr_confidence=ocr_confidence, bounding_box=bounding_box
    )


def _validate_page_result(raw: object) -> WorkerPageResult:
    if not isinstance(raw, dict):
        raise ProtocolViolationError("page result must be a JSON object")
    unexpected = set(raw.keys()) - {
        "page_number",
        "width_px",
        "height_px",
        "used_ocr",
        "ocr_attempted",
        "text_blocks",
    }
    if unexpected:
        raise ProtocolViolationError("page result contains an unexpected field")

    page_number = _exact_int(raw.get("page_number"))
    if page_number is None or page_number < 1:
        raise ProtocolViolationError("page_number must be a positive integer")

    width_px = _exact_int(raw.get("width_px"))
    height_px = _exact_int(raw.get("height_px"))
    if width_px is None or width_px <= 0 or height_px is None or height_px <= 0:
        raise ProtocolViolationError("width_px/height_px must be positive integers")

    used_ocr = _exact_bool(raw.get("used_ocr"))
    if used_ocr is None:
        raise ProtocolViolationError("used_ocr must be a boolean")

    ocr_attempted = _exact_bool(raw.get("ocr_attempted"))
    if ocr_attempted is None:
        raise ProtocolViolationError("ocr_attempted must be a boolean")

    raw_blocks = raw.get("text_blocks")
    if not isinstance(raw_blocks, list) or len(raw_blocks) > MAX_TEXT_BLOCKS_PER_PAGE:
        raise ProtocolViolationError("text_blocks must be a bounded list")
    text_blocks = tuple(
        _validate_text_block(b, width_px=width_px, height_px=height_px) for b in raw_blocks
    )

    return WorkerPageResult(
        page_number=page_number,
        width_px=width_px,
        height_px=height_px,
        used_ocr=used_ocr,
        ocr_attempted=ocr_attempted,
        text_blocks=text_blocks,
    )


def validate_job_response(raw: dict[str, Any]) -> WorkerJobResponse:
    unexpected = set(raw.keys()) - {
        "protocol_version",
        "message_type",
        "pages",
        "ocr_available",
        "preview_count",
    }
    if unexpected:
        raise ProtocolViolationError("job_response contains an unexpected field")

    raw_pages = raw.get("pages")
    if not isinstance(raw_pages, list) or len(raw_pages) > MAX_PAGES_PER_RESPONSE:
        raise ProtocolViolationError("pages must be a bounded list")

    ocr_available = _exact_bool(raw.get("ocr_available"))
    if ocr_available is None:
        raise ProtocolViolationError("ocr_available must be a boolean")

    preview_count = 0
    if "preview_count" in raw:
        parsed_preview_count = _exact_int(raw.get("preview_count"))
        if parsed_preview_count is None or not (
            0 <= parsed_preview_count <= ABSOLUTE_MAX_PREVIEW_PAGES
        ):
            raise ProtocolViolationError(
                f"preview_count must be an integer in [0, {ABSOLUTE_MAX_PREVIEW_PAGES}]"
            )
        preview_count = parsed_preview_count

    pages = tuple(_validate_page_result(p) for p in raw_pages)
    return WorkerJobResponse(pages=pages, ocr_available=ocr_available, preview_count=preview_count)


def validate_preview_header(raw: dict[str, Any]) -> WorkerPreviewHeader:
    unexpected = set(raw.keys()) - {
        "protocol_version",
        "message_type",
        "page_number",
        "byte_length",
    }
    if unexpected:
        raise ProtocolViolationError("preview_header contains an unexpected field")

    page_number = _exact_int(raw.get("page_number"))
    if page_number is None or page_number < 1:
        raise ProtocolViolationError("page_number must be a positive integer")

    byte_length = _exact_int(raw.get("byte_length"))
    if byte_length is None or not (0 <= byte_length <= ABSOLUTE_MAX_PREVIEW_FRAME_BYTES):
        raise ProtocolViolationError(
            f"byte_length must be an integer in [0, {ABSOLUTE_MAX_PREVIEW_FRAME_BYTES}]"
        )

    return WorkerPreviewHeader(page_number=page_number, byte_length=byte_length)


def validate_job_error(raw: dict[str, Any]) -> WorkerJobError:
    unexpected = set(raw.keys()) - {"protocol_version", "message_type", "error_kind"}
    if unexpected:
        raise ProtocolViolationError("job_error contains an unexpected field")

    error_kind = raw.get("error_kind")
    if error_kind not in _ERROR_KINDS:
        raise ProtocolViolationError("error_kind is not one of the supported closed values")
    return WorkerJobError(error_kind=error_kind)


def validate_job_request(raw: dict[str, Any]) -> WorkerJobRequest:
    unexpected = set(raw.keys()) - {
        "protocol_version",
        "message_type",
        "expected_jurisdiction",
        "media_type",
        "max_pages",
        "max_pixels",
        "ocr_enabled",
        "page_timeout_seconds",
    }
    if unexpected:
        raise ProtocolViolationError("job_request contains an unexpected field")

    expected_jurisdiction = raw.get("expected_jurisdiction")
    if expected_jurisdiction not in _JURISDICTIONS:
        raise ProtocolViolationError(
            "expected_jurisdiction is not one of the supported closed values"
        )

    media_type = raw.get("media_type")
    if media_type not in _MEDIA_TYPES:
        raise ProtocolViolationError("media_type is not one of the supported closed values")

    max_pages = _exact_int(raw.get("max_pages"))
    if max_pages is None or max_pages < 1:
        raise ProtocolViolationError("max_pages must be a positive integer")

    max_pixels = _exact_int(raw.get("max_pixels"))
    if max_pixels is None or max_pixels < 1:
        raise ProtocolViolationError("max_pixels must be a positive integer")

    ocr_enabled = _exact_bool(raw.get("ocr_enabled"))
    if ocr_enabled is None:
        raise ProtocolViolationError("ocr_enabled must be a boolean")

    page_timeout_seconds = _exact_finite_number(raw.get("page_timeout_seconds"))
    if page_timeout_seconds is None or page_timeout_seconds <= 0:
        raise ProtocolViolationError("page_timeout_seconds must be a positive finite number")

    return WorkerJobRequest(
        expected_jurisdiction=expected_jurisdiction,
        media_type=media_type,
        max_pages=max_pages,
        max_pixels=max_pixels,
        ocr_enabled=ocr_enabled,
        page_timeout_seconds=page_timeout_seconds,
    )


def validate_page_progress(raw: dict[str, Any]) -> WorkerPageProgress:
    unexpected = set(raw.keys()) - {"protocol_version", "message_type", "page_number"}
    if unexpected:
        raise ProtocolViolationError("page_progress contains an unexpected field")

    page_number = _exact_int(raw.get("page_number"))
    if page_number is None or page_number < 1:
        raise ProtocolViolationError("page_number must be a positive integer")
    return WorkerPageProgress(page_number=page_number)


def validate_worker_message(
    raw: dict[str, Any],
) -> (
    WorkerJobRequest | WorkerJobResponse | WorkerJobError | WorkerPageProgress | WorkerPreviewHeader
):
    """The single closed dispatch point: ``message_type`` selects exactly one
    of the fixed validators above. There is no dynamic lookup by name
    anywhere in this function."""
    protocol_version = raw.get("protocol_version")
    if _exact_int(protocol_version) != PROTOCOL_VERSION:
        raise ProtocolViolationError("unsupported protocol_version")

    message_type = raw.get("message_type")
    if message_type not in _MESSAGE_TYPES:
        raise ProtocolViolationError("message_type is not one of the supported closed values")

    if message_type == "job_request":
        return validate_job_request(raw)
    if message_type == "job_response":
        return validate_job_response(raw)
    if message_type == "page_progress":
        return validate_page_progress(raw)
    if message_type == "preview_header":
        return validate_preview_header(raw)
    return validate_job_error(raw)
