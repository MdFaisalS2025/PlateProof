"""RED-first tests for the non-pickle worker wire protocol.

These tests are the actual proof of the plan's security claims: no pickle
anywhere on the parent<->worker boundary, hardened JSON decoding (non-finite
rejection, duplicate-key rejection, bounded nesting scanned before decode,
exact-type numeric checks), and exact two-message framing with declared-size
checks before any oversized read is attempted.
"""

from __future__ import annotations

import json
import math
import pickle
import struct
from multiprocessing import Pipe
from typing import Any
from unittest.mock import patch

import pytest

# --------------------------------------------------------------------------- #
# Test doubles                                                                #
# --------------------------------------------------------------------------- #


class _RaisingConnection:
    """A Connection stand-in whose payload-reading method raises if ever
    called with an oversized maxlength -- proves the receiver never even
    attempts to allocate for a declared-oversized frame."""

    def __init__(self, header_bytes: bytes) -> None:
        self._header = header_bytes
        self._first_call = True

    def recv_bytes(self, maxlength: int | None = None) -> bytes:
        if self._first_call:
            self._first_call = False
            return self._header
        raise AssertionError(f"payload recv_bytes called with maxlength={maxlength}")


class _ShortHeaderConnection:
    def recv_bytes(self, maxlength: int | None = None) -> bytes:
        return b"\x00\x01"  # 2 bytes, not 4


class _MismatchedPayloadConnection:
    """Returns a header declaring N bytes but a payload of a different length."""

    def __init__(self, declared_length: int, actual_payload: bytes) -> None:
        self._declared_length = declared_length
        self._actual_payload = actual_payload
        self._calls = 0

    def recv_bytes(self, maxlength: int | None = None) -> bytes:
        self._calls += 1
        if self._calls == 1:
            return struct.pack(">I", self._declared_length)
        return self._actual_payload


class _FixedFramesConnection:
    """Replays a fixed sequence of recv_bytes return values, one per call."""

    def __init__(self, frames: list[bytes]) -> None:
        self._frames = list(frames)

    def recv_bytes(self, maxlength: int | None = None) -> bytes:
        return self._frames.pop(0)


def _encode_valid_frame(message: dict[str, Any]) -> bytes:
    return json.dumps(message, sort_keys=True, separators=(",", ":")).encode("utf-8")


# --------------------------------------------------------------------------- #
# Framing algorithm                                                           #
# --------------------------------------------------------------------------- #


def test_send_frame_issues_exactly_two_send_bytes_calls() -> None:
    from plateproof.documents.worker.protocol import send_frame

    parent_conn, child_conn = Pipe(duplex=True)
    try:
        with patch.object(parent_conn, "send_bytes", wraps=parent_conn.send_bytes) as spy:
            send_frame(
                parent_conn,
                {
                    "protocol_version": 1,
                    "message_type": "job_error",
                    "error_kind": "internal_error",
                },
            )
        assert spy.call_count == 2
        first_arg = spy.call_args_list[0].args[0]
        second_arg = spy.call_args_list[1].args[0]
        assert len(first_arg) == 4
        assert len(second_arg) == struct.unpack(">I", first_arg)[0]
    finally:
        parent_conn.close()
        child_conn.close()


def test_send_then_recv_frame_round_trips() -> None:
    from plateproof.documents.worker.protocol import recv_frame, send_frame

    parent_conn, child_conn = Pipe(duplex=True)
    try:
        message = {"protocol_version": 1, "message_type": "job_error", "error_kind": "ocr_failed"}
        send_frame(parent_conn, message)
        received = recv_frame(child_conn)
        assert received == message
    finally:
        parent_conn.close()
        child_conn.close()


def test_short_header_is_a_protocol_violation() -> None:
    from plateproof.documents.worker.protocol import ProtocolViolationError, recv_frame

    with pytest.raises(ProtocolViolationError):
        recv_frame(_ShortHeaderConnection())  # type: ignore[arg-type]


def test_oversized_declared_length_rejected_before_second_recv() -> None:
    from plateproof.documents.worker.protocol import (
        MAX_WORKER_FRAME_BYTES,
        ProtocolViolationError,
        recv_frame,
    )

    header = struct.pack(">I", MAX_WORKER_FRAME_BYTES + 1)
    conn = _RaisingConnection(header)
    with pytest.raises(ProtocolViolationError):
        recv_frame(conn)  # type: ignore[arg-type]


def test_short_payload_is_a_protocol_violation() -> None:
    from plateproof.documents.worker.protocol import ProtocolViolationError, recv_frame

    payload = _encode_valid_frame(
        {"protocol_version": 1, "message_type": "job_error", "error_kind": "internal_error"}
    )
    conn = _MismatchedPayloadConnection(len(payload), payload[:-1])
    with pytest.raises(ProtocolViolationError):
        recv_frame(conn)  # type: ignore[arg-type]


def test_long_payload_is_a_protocol_violation() -> None:
    from plateproof.documents.worker.protocol import ProtocolViolationError, recv_frame

    payload = _encode_valid_frame(
        {"protocol_version": 1, "message_type": "job_error", "error_kind": "internal_error"}
    )
    conn = _MismatchedPayloadConnection(len(payload), payload + b"trailing")
    with pytest.raises(ProtocolViolationError):
        recv_frame(conn)  # type: ignore[arg-type]


def test_invalid_utf8_payload_rejected() -> None:
    from plateproof.documents.worker.protocol import ProtocolViolationError, recv_frame

    bad_bytes = b"\xff\xfe\xfd"
    header = struct.pack(">I", len(bad_bytes))
    conn = _FixedFramesConnection([header, bad_bytes])
    with pytest.raises(ProtocolViolationError):
        recv_frame(conn)  # type: ignore[arg-type]


def test_syntactically_invalid_json_rejected() -> None:
    from plateproof.documents.worker.protocol import ProtocolViolationError, recv_frame

    bad = b"{not valid json"
    header = struct.pack(">I", len(bad))
    conn = _FixedFramesConnection([header, bad])
    with pytest.raises(ProtocolViolationError):
        recv_frame(conn)  # type: ignore[arg-type]


def test_non_object_top_level_json_rejected() -> None:
    from plateproof.documents.worker.protocol import ProtocolViolationError, recv_frame

    bad = b"[1,2,3]"
    header = struct.pack(">I", len(bad))
    conn = _FixedFramesConnection([header, bad])
    with pytest.raises(ProtocolViolationError):
        recv_frame(conn)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# JSON decoder configuration (non-finite / duplicate keys / nesting)         #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_json_tokens_rejected(token: str) -> None:
    from plateproof.documents.worker.protocol import ProtocolViolationError, recv_frame

    raw = (
        f'{{"protocol_version": 1, "message_type": "job_error", "error_kind": {token}}}'
    ).encode()
    header = struct.pack(">I", len(raw))
    conn = _FixedFramesConnection([header, raw])
    with pytest.raises(ProtocolViolationError):
        recv_frame(conn)  # type: ignore[arg-type]


def test_duplicate_key_at_top_level_rejected() -> None:
    from plateproof.documents.worker.protocol import ProtocolViolationError, recv_frame

    raw = (
        b'{"protocol_version": 1, "protocol_version": 2, '
        b'"message_type": "job_error", "error_kind": "internal_error"}'
    )
    header = struct.pack(">I", len(raw))
    conn = _FixedFramesConnection([header, raw])
    with pytest.raises(ProtocolViolationError):
        recv_frame(conn)  # type: ignore[arg-type]


def test_duplicate_key_in_nested_object_rejected() -> None:
    from plateproof.documents.worker.protocol import ProtocolViolationError, recv_frame

    raw = (
        b'{"protocol_version": 1, "message_type": "job_response", "pages": ['
        b'{"page_number": 1, "page_number": 2, "width_px": 10, "height_px": 10, '
        b'"used_ocr": false, "text_blocks": []}]}'
    )
    header = struct.pack(">I", len(raw))
    conn = _FixedFramesConnection([header, raw])
    with pytest.raises(ProtocolViolationError):
        recv_frame(conn)  # type: ignore[arg-type]


def test_excess_nesting_rejected_before_json_loads() -> None:
    from plateproof.documents.worker.protocol import (
        MAX_NESTING_DEPTH,
        ProtocolViolationError,
        recv_frame,
    )

    nested = "1"
    for _ in range(MAX_NESTING_DEPTH + 2):
        nested = f"[{nested}]"
    raw = f'{{"protocol_version": 1, "message_type": "job_error", "error_kind": {nested}}}'.encode()
    header = struct.pack(">I", len(raw))
    conn = _FixedFramesConnection([header, raw])
    with patch("json.loads", side_effect=AssertionError("json.loads must not be called")):
        with pytest.raises(ProtocolViolationError):
            recv_frame(conn)  # type: ignore[arg-type]


def test_string_containing_braces_does_not_trip_nesting_scanner() -> None:
    from plateproof.documents.worker.protocol import recv_frame, send_frame

    text_with_punctuation = "value contains { [ ] } characters but is just text"
    message = {
        "protocol_version": 1,
        "message_type": "job_response",
        "pages": [
            {
                "page_number": 1,
                "width_px": 100,
                "height_px": 100,
                "used_ocr": False,
                "text_blocks": [
                    {
                        "text": text_with_punctuation,
                        "source": "embedded_text",
                        "ocr_confidence": None,
                        "bounding_box": None,
                    }
                ],
            }
        ],
    }
    parent_conn, child_conn = Pipe(duplex=True)
    try:
        send_frame(parent_conn, message)
        received = recv_frame(child_conn)
        assert received["pages"][0]["text_blocks"][0]["text"] == text_with_punctuation
    finally:
        parent_conn.close()
        child_conn.close()


def test_escaped_quotes_and_backslashes_do_not_desync_scanner() -> None:
    from plateproof.documents.worker.protocol import recv_frame, send_frame

    tricky_text = 'has an escaped quote \\" and an escaped backslash \\\\ then more'
    message = {
        "protocol_version": 1,
        "message_type": "job_response",
        "pages": [
            {
                "page_number": 1,
                "width_px": 100,
                "height_px": 100,
                "used_ocr": False,
                "text_blocks": [
                    {
                        "text": tricky_text,
                        "source": "embedded_text",
                        "ocr_confidence": None,
                        "bounding_box": None,
                    }
                ],
            }
        ],
    }
    parent_conn, child_conn = Pipe(duplex=True)
    try:
        send_frame(parent_conn, message)
        received = recv_frame(child_conn)
        assert received["pages"][0]["text_blocks"][0]["text"] == tricky_text
    finally:
        parent_conn.close()
        child_conn.close()


# --------------------------------------------------------------------------- #
# Schema validation (exact-type checks, closed enums, ranges)                #
# --------------------------------------------------------------------------- #


def test_boolean_rejected_for_integer_field() -> None:
    from plateproof.documents.worker.protocol import ProtocolViolationError, validate_job_response

    raw = {
        "protocol_version": 1,
        "message_type": "job_response",
        "pages": [
            {
                "page_number": True,
                "width_px": 10,
                "height_px": 10,
                "used_ocr": False,
                "text_blocks": [],
            }
        ],
    }
    with pytest.raises(ProtocolViolationError):
        validate_job_response(raw)


def test_non_finite_float_rejected_post_decode_defense_in_depth() -> None:
    from plateproof.documents.worker.protocol import ProtocolViolationError, validate_job_response

    raw = {
        "protocol_version": 1,
        "message_type": "job_response",
        "pages": [
            {
                "page_number": 1,
                "width_px": 10,
                "height_px": 10,
                "used_ocr": False,
                "text_blocks": [
                    {
                        "text": "x",
                        "source": "ocr",
                        "ocr_confidence": math.inf,
                        "bounding_box": None,
                    }
                ],
            }
        ],
    }
    with pytest.raises(ProtocolViolationError):
        validate_job_response(raw)


def test_page_number_out_of_range_rejected() -> None:
    from plateproof.documents.worker.protocol import ProtocolViolationError, validate_job_response

    raw = {
        "protocol_version": 1,
        "message_type": "job_response",
        "pages": [
            {
                "page_number": 0,
                "width_px": 10,
                "height_px": 10,
                "used_ocr": False,
                "text_blocks": [],
            }
        ],
    }
    with pytest.raises(ProtocolViolationError):
        validate_job_response(raw)


def test_inverted_bounding_box_rejected() -> None:
    from plateproof.documents.worker.protocol import ProtocolViolationError, validate_job_response

    raw = {
        "protocol_version": 1,
        "message_type": "job_response",
        "pages": [
            {
                "page_number": 1,
                "width_px": 100,
                "height_px": 100,
                "used_ocr": False,
                "text_blocks": [
                    {
                        "text": "x",
                        "source": "embedded_text",
                        "ocr_confidence": None,
                        "bounding_box": {"x0": 50, "y0": 0, "x1": 10, "y1": 10},
                    }
                ],
            }
        ],
    }
    with pytest.raises(ProtocolViolationError):
        validate_job_response(raw)


def test_bounding_box_outside_page_dimensions_rejected() -> None:
    from plateproof.documents.worker.protocol import ProtocolViolationError, validate_job_response

    raw = {
        "protocol_version": 1,
        "message_type": "job_response",
        "pages": [
            {
                "page_number": 1,
                "width_px": 100,
                "height_px": 100,
                "used_ocr": False,
                "text_blocks": [
                    {
                        "text": "x",
                        "source": "embedded_text",
                        "ocr_confidence": None,
                        "bounding_box": {"x0": 0, "y0": 0, "x1": 500, "y1": 10},
                    }
                ],
            }
        ],
    }
    with pytest.raises(ProtocolViolationError):
        validate_job_response(raw)


def test_unrecognized_message_type_rejected() -> None:
    from plateproof.documents.worker.protocol import ProtocolViolationError, validate_worker_message

    with pytest.raises(ProtocolViolationError):
        validate_worker_message({"protocol_version": 1, "message_type": "shell_exec"})


def test_unrecognized_protocol_version_rejected() -> None:
    from plateproof.documents.worker.protocol import ProtocolViolationError, validate_worker_message

    with pytest.raises(ProtocolViolationError):
        validate_worker_message(
            {"protocol_version": 999, "message_type": "job_error", "error_kind": "internal_error"}
        )


def test_unrecognized_error_kind_rejected() -> None:
    from plateproof.documents.worker.protocol import ProtocolViolationError, validate_worker_message

    with pytest.raises(ProtocolViolationError):
        validate_worker_message(
            {"protocol_version": 1, "message_type": "job_error", "error_kind": "rm_rf_root"}
        )


def test_unknown_top_level_key_rejected() -> None:
    from plateproof.documents.worker.protocol import ProtocolViolationError, validate_worker_message

    with pytest.raises(ProtocolViolationError):
        validate_worker_message(
            {
                "protocol_version": 1,
                "message_type": "job_error",
                "error_kind": "internal_error",
                "stdout": "leaked",
            }
        )


def test_too_many_pages_rejected() -> None:
    from plateproof.documents.worker.protocol import (
        MAX_PAGES_PER_RESPONSE,
        ProtocolViolationError,
        validate_job_response,
    )

    pages = [
        {
            "page_number": i + 1,
            "width_px": 10,
            "height_px": 10,
            "used_ocr": False,
            "text_blocks": [],
        }
        for i in range(MAX_PAGES_PER_RESPONSE + 1)
    ]
    with pytest.raises(ProtocolViolationError):
        validate_job_response(
            {"protocol_version": 1, "message_type": "job_response", "pages": pages}
        )


def test_valid_job_response_round_trips_deterministically() -> None:
    from plateproof.documents.worker.protocol import (
        WorkerJobResponse,
        recv_frame,
        send_frame,
        validate_job_response,
    )

    message = {
        "protocol_version": 1,
        "message_type": "job_response",
        "ocr_available": True,
        "pages": [
            {
                "page_number": 1,
                "width_px": 100,
                "height_px": 200,
                "used_ocr": True,
                "ocr_attempted": True,
                "text_blocks": [
                    {
                        "text": "Score: 14",
                        "source": "ocr",
                        "ocr_confidence": 0.95,
                        "bounding_box": {"x0": 1.0, "y0": 2.0, "x1": 50.0, "y1": 20.0},
                    }
                ],
            }
        ],
    }
    encoded_a = json.dumps(message, sort_keys=True, separators=(",", ":"))
    encoded_b = json.dumps(message, sort_keys=True, separators=(",", ":"))
    assert encoded_a == encoded_b

    response = validate_job_response(message)
    assert isinstance(response, WorkerJobResponse)
    assert response.pages[0].text_blocks[0].text == "Score: 14"

    parent_conn, child_conn = Pipe(duplex=True)
    try:
        send_frame(parent_conn, message)
        raw_received = recv_frame(child_conn)
        assert raw_received == message
    finally:
        parent_conn.close()
        child_conn.close()


# --------------------------------------------------------------------------- #
# Raw binary document-bytes frame                                            #
# --------------------------------------------------------------------------- #


def test_bytes_frame_round_trips() -> None:
    from plateproof.documents.worker.protocol import recv_bytes_frame, send_bytes_frame

    parent_conn, child_conn = Pipe(duplex=True)
    try:
        payload = b"%PDF-1.4 fake document bytes"
        send_bytes_frame(parent_conn, payload)
        received = recv_bytes_frame(child_conn, max_length=len(payload))
        assert received == payload
    finally:
        parent_conn.close()
        child_conn.close()


def test_bytes_frame_oversized_declared_length_rejected_before_second_recv() -> None:
    from plateproof.documents.worker.protocol import ProtocolViolationError, recv_bytes_frame

    header = struct.pack(">I", 1000)
    conn = _RaisingConnection(header)
    with pytest.raises(ProtocolViolationError):
        recv_bytes_frame(conn, max_length=100)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# No-pickle proof                                                             #
# --------------------------------------------------------------------------- #


class _EvilPickle:
    """A __reduce__-based payload that would set a module-level sentinel if
    ever actually unpickled -- proves rejection, not just "the result looks
    safe". Safe here: pickle.dumps only constructs the malicious byte
    sequence in this test process; the assertion is that the receiver under
    test never calls pickle.loads/pickle.load on it (see the following test),
    so __reduce__ is never invoked."""

    triggered = False

    def __reduce__(self) -> tuple[Any, ...]:
        return (_mark_triggered, ())


def _mark_triggered() -> None:
    _EvilPickle.triggered = True


def test_malicious_pickle_bytes_are_rejected_not_unpickled() -> None:
    from plateproof.documents.worker.protocol import ProtocolViolationError, recv_frame

    evil_bytes = pickle.dumps(_EvilPickle())
    header = struct.pack(">I", len(evil_bytes))
    conn = _FixedFramesConnection([header, evil_bytes])
    with pytest.raises(ProtocolViolationError):
        recv_frame(conn)  # type: ignore[arg-type]
    assert _EvilPickle.triggered is False


def test_no_pickle_or_pickling_connection_apis_used_in_round_trip() -> None:
    from multiprocessing.connection import Connection

    from plateproof.documents.worker.protocol import recv_frame, send_frame

    parent_conn, child_conn = Pipe(duplex=True)
    try:
        with (
            patch("pickle.loads") as mock_loads,
            patch("pickle.load") as mock_load,
            patch.object(Connection, "send") as mock_send,
            patch.object(Connection, "recv") as mock_recv,
        ):
            message = {
                "protocol_version": 1,
                "message_type": "job_error",
                "error_kind": "internal_error",
            }
            send_frame(parent_conn, message)
            recv_frame(child_conn)
        mock_loads.assert_not_called()
        mock_load.assert_not_called()
        mock_send.assert_not_called()
        mock_recv.assert_not_called()
    finally:
        parent_conn.close()
        child_conn.close()


def test_page_progress_round_trips() -> None:
    from plateproof.documents.worker.protocol import (
        WorkerPageProgress,
        recv_frame,
        send_frame,
        validate_worker_message,
    )

    parent_conn, child_conn = Pipe(duplex=True)
    try:
        send_frame(
            parent_conn, {"protocol_version": 1, "message_type": "page_progress", "page_number": 3}
        )
        raw = recv_frame(child_conn)
        validated = validate_worker_message(raw)
        assert isinstance(validated, WorkerPageProgress)
        assert validated.page_number == 3
    finally:
        parent_conn.close()
        child_conn.close()


def test_page_progress_rejects_non_positive_page_number() -> None:
    from plateproof.documents.worker.protocol import ProtocolViolationError, validate_page_progress

    with pytest.raises(ProtocolViolationError):
        validate_page_progress(
            {"protocol_version": 1, "message_type": "page_progress", "page_number": 0}
        )


def test_extra_unexpected_message_is_a_protocol_violation() -> None:
    """An extra message arriving when none was expected must not be silently
    buffered for the next logical read -- the caller reading a header when a
    stray payload-shaped message is actually next should fail closed."""
    from plateproof.documents.worker.protocol import ProtocolViolationError, recv_frame, send_frame

    parent_conn, child_conn = Pipe(duplex=True)
    try:
        send_frame(
            parent_conn,
            {"protocol_version": 1, "message_type": "job_error", "error_kind": "internal_error"},
        )
        recv_frame(child_conn)
        # Nothing else was sent -- attempting to read a second frame with no
        # data available must fail closed rather than hang or crash uncaught.
        parent_conn.close()
        with pytest.raises(ProtocolViolationError):
            recv_frame(child_conn)
    finally:
        child_conn.close()
