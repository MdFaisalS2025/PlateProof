"""Shared, parent-process-safe byte validation for uploaded documents.

This module NEVER parses a document -- it only sniffs a small fixed magic-
byte signature and checks size. Real parsing (PDFium, Pillow, OpenCV,
onnxruntime, RapidOCR) happens only inside a worker process (see
``plateproof.documents.worker``). Both the FastAPI route and the Streamlit
page (Task 9B) call the same function here before any byte ever reaches the
worker manager.

A declared filename or MIME type is never trusted -- :func:`validate_upload_bytes`
does not even accept one as a parameter, so there is no code path where either
could influence the result."""

from __future__ import annotations

from dataclasses import dataclass

from plateproof.documents.models import DetectedMediaType

#: Fixed magic-byte signatures for the only three supported formats. Checked
#: in this order; the first match wins (a deliberately crafted polyglot
#: still gets exactly one deterministic classification).
_SIGNATURES: tuple[tuple[bytes, DetectedMediaType], ...] = (
    (b"%PDF-", "application/pdf"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
)


def sniff_media_type(data: bytes) -> DetectedMediaType | None:
    """Return the detected media type from a fixed magic-byte signature, or
    ``None`` if the bytes don't start with any of the three supported
    signatures. TIFF, GIF, and every other format are deliberately
    unrecognized -- Task 9 supports only PDF, PNG, and JPEG."""
    for signature, media_type in _SIGNATURES:
        if data.startswith(signature):
            return media_type
    return None


@dataclass(frozen=True, kw_only=True)
class UploadValidationResult:
    accepted: bool
    detected_media_type: DetectedMediaType | None
    byte_size: int
    rejection_reason: str | None


def validate_upload_bytes(data: bytes, *, max_bytes: int) -> UploadValidationResult:
    """Validate raw upload bytes against size and signature rules only.

    Deliberately takes no filename or declared-content-type parameter --
    neither is ever consulted, so neither can be trusted or spoofed here.
    """
    size = len(data)

    if max_bytes <= 0:
        return UploadValidationResult(
            accepted=False,
            detected_media_type=None,
            byte_size=size,
            rejection_reason="invalid_configured_limit",
        )

    if size == 0:
        return UploadValidationResult(
            accepted=False,
            detected_media_type=None,
            byte_size=size,
            rejection_reason="empty_upload",
        )

    if size > max_bytes:
        return UploadValidationResult(
            accepted=False,
            detected_media_type=None,
            byte_size=size,
            rejection_reason="upload_too_large",
        )

    media_type = sniff_media_type(data)
    if media_type is None:
        return UploadValidationResult(
            accepted=False,
            detected_media_type=None,
            byte_size=size,
            rejection_reason="unsupported_media_type",
        )

    return UploadValidationResult(
        accepted=True, detected_media_type=media_type, byte_size=size, rejection_reason=None
    )
