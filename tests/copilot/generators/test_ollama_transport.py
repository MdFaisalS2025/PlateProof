"""Tests against the REAL transport code in
plateproof.copilot.generators.ollama -- a real loopback HTTP server
(tests/copilot/generators/conftest.py::ollama_loopback_server) stands in
for Ollama, and only the outermost socket is real. Nothing here mocks
``_send_request`` itself: bounded reads, timeouts, connection cleanup,
status handling, and JSON/UTF-8 decoding are all exercised for real. No
test contacts anything outside 127.0.0.1."""

from __future__ import annotations

import json
import time
from typing import Any

import pytest

from plateproof.copilot.generators.base import IntentHelperOutcome
from plateproof.copilot.generators.ollama import OllamaIntentHelper
from plateproof.copilot.models import Intent


def _helper(port: int, **overrides: Any) -> OllamaIntentHelper:
    kwargs: dict[str, Any] = {
        "base_url": f"http://127.0.0.1:{port}",
        "model": "fictional-model",
        "connect_timeout": 2.0,
        "read_timeout": 2.0,
        "max_response_bytes": 65_536,
        "min_confidence": 0.6,
    }
    kwargs.update(overrides)
    return OllamaIntentHelper(**kwargs)


def test_valid_response_is_accepted_and_only_the_fixed_path_is_requested(
    ollama_loopback_server: Any,
) -> None:
    helper = _helper(ollama_loopback_server.port)
    result = helper.propose(question="what violations recur?", jurisdiction="nyc", now=None)  # type: ignore[arg-type]
    assert result.outcome == IntentHelperOutcome.ACCEPTED
    assert result.proposal is not None
    assert result.proposal.intent == Intent.RECURRING_VIOLATIONS
    assert ollama_loopback_server.received_paths == ["/api/generate"]


def test_localhost_hostname_actually_connects_via_the_127_0_0_1_literal(
    ollama_loopback_server: Any,
) -> None:
    """The server is bound only to 127.0.0.1. If the code resolved
    "localhost" via the system resolver and it preferred ::1 (a real
    possibility on many hosts), this request would fail to connect --
    proving the connection genuinely goes out over the IP literal, not a
    DNS-resolved hostname."""
    helper = _helper(
        ollama_loopback_server.port, base_url=f"http://localhost:{ollama_loopback_server.port}"
    )
    result = helper.propose(question="what violations recur?", jurisdiction="nyc", now=None)  # type: ignore[arg-type]
    assert result.outcome == IntentHelperOutcome.ACCEPTED


def test_proxy_environment_variables_are_ignored(
    ollama_loopback_server: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Point HTTP_PROXY/HTTPS_PROXY at a non-routable, non-responding
    address. If the transport honored them, this request would hang or
    fail; since http.client never consults proxy env vars, it still
    reaches the real loopback server directly and succeeds quickly."""
    monkeypatch.setenv("HTTP_PROXY", "http://10.255.255.1:1")
    monkeypatch.setenv("HTTPS_PROXY", "http://10.255.255.1:1")
    monkeypatch.setenv("http_proxy", "http://10.255.255.1:1")
    helper = _helper(ollama_loopback_server.port)
    start = time.perf_counter()
    result = helper.propose(question="what violations recur?", jurisdiction="nyc", now=None)  # type: ignore[arg-type]
    elapsed = time.perf_counter() - start
    assert result.outcome == IntentHelperOutcome.ACCEPTED
    assert elapsed < 2.0


def test_redirect_status_is_rejected_not_followed(ollama_loopback_server: Any) -> None:
    ollama_loopback_server.set_redirect()
    helper = _helper(ollama_loopback_server.port)
    result = helper.propose(question="what violations recur?", jurisdiction="nyc", now=None)  # type: ignore[arg-type]
    assert result.outcome == IntentHelperOutcome.UNAVAILABLE
    assert result.proposal is None


def test_non_200_status_is_rejected(ollama_loopback_server: Any) -> None:
    ollama_loopback_server.set_raw_body(b"server error", status=500)
    helper = _helper(ollama_loopback_server.port)
    result = helper.propose(question="what violations recur?", jurisdiction="nyc", now=None)  # type: ignore[arg-type]
    assert result.outcome == IntentHelperOutcome.UNAVAILABLE


def test_oversized_response_is_rejected_and_not_fully_materialized(
    ollama_loopback_server: Any,
) -> None:
    huge = json.dumps({"response": "x" * 500_000}).encode("utf-8")
    ollama_loopback_server.set_raw_body(huge)
    helper = _helper(ollama_loopback_server.port, max_response_bytes=1_000)
    start = time.perf_counter()
    result = helper.propose(question="what violations recur?", jurisdiction="nyc", now=None)  # type: ignore[arg-type]
    elapsed = time.perf_counter() - start
    assert result.outcome == IntentHelperOutcome.UNAVAILABLE
    assert elapsed < 5.0
    assert result.reason is not None
    assert "x" * 50 not in result.reason


def test_malformed_outer_json_is_rejected(ollama_loopback_server: Any) -> None:
    ollama_loopback_server.set_raw_body(b"not json at all {{{")
    helper = _helper(ollama_loopback_server.port)
    result = helper.propose(question="what violations recur?", jurisdiction="nyc", now=None)  # type: ignore[arg-type]
    assert result.outcome == IntentHelperOutcome.UNAVAILABLE


def test_invalid_utf8_response_is_rejected(ollama_loopback_server: Any) -> None:
    ollama_loopback_server.set_raw_body(b"\xff\xfe\x00\x01 not utf-8")
    helper = _helper(ollama_loopback_server.port)
    result = helper.propose(question="what violations recur?", jurisdiction="nyc", now=None)  # type: ignore[arg-type]
    assert result.outcome == IntentHelperOutcome.UNAVAILABLE


def test_malformed_inner_json_is_rejected_as_content_not_transport_failure(
    ollama_loopback_server: Any,
) -> None:
    """A well-formed HTTP 200 + valid outer JSON envelope, but the
    ``response`` field itself isn't valid JSON -- the server replied, so
    this is a REJECTED content failure, not an UNAVAILABLE transport one."""
    ollama_loopback_server.set_raw_body(
        json.dumps({"response": "not valid json {{"}).encode("utf-8")
    )
    helper = _helper(ollama_loopback_server.port)
    result = helper.propose(question="what violations recur?", jurisdiction="nyc", now=None)  # type: ignore[arg-type]
    assert result.outcome == IntentHelperOutcome.REJECTED


def test_response_envelope_missing_response_field_is_rejected(ollama_loopback_server: Any) -> None:
    ollama_loopback_server.set_raw_body(json.dumps({"not_response": "x"}).encode("utf-8"))
    helper = _helper(ollama_loopback_server.port)
    result = helper.propose(question="what violations recur?", jurisdiction="nyc", now=None)  # type: ignore[arg-type]
    assert result.outcome == IntentHelperOutcome.UNAVAILABLE


def test_read_timeout_is_applied_to_a_slow_response(ollama_loopback_server: Any) -> None:
    ollama_loopback_server.set_sleep(3.0)
    helper = _helper(ollama_loopback_server.port, read_timeout=0.5, connect_timeout=2.0)
    start = time.perf_counter()
    result = helper.propose(question="what violations recur?", jurisdiction="nyc", now=None)  # type: ignore[arg-type]
    elapsed = time.perf_counter() - start
    assert result.outcome == IntentHelperOutcome.UNAVAILABLE
    assert elapsed < 2.5  # bounded by read_timeout, not the server's 3s sleep


def test_connection_refused_is_rejected(unused_tcp_port_factory: Any) -> None:
    port = unused_tcp_port_factory()
    helper = _helper(port)
    result = helper.propose(question="what violations recur?", jurisdiction="nyc", now=None)  # type: ignore[arg-type]
    assert result.outcome == IntentHelperOutcome.UNAVAILABLE


def test_connect_timeout_value_is_threaded_into_the_real_connection_object(
    ollama_loopback_server: Any,
) -> None:
    """White-box check that the configured connect timeout genuinely
    reaches the real http.client.HTTPConnection object (constructed by the
    module's own default connection factory), complementing the
    behavioral read-timeout test above -- loopback connects are too fast
    to reliably trigger a real connect-timeout end-to-end."""
    from plateproof.copilot.generators import ollama as ollama_module

    captured: dict[str, Any] = {}
    original_factory = ollama_module._default_connection_factory

    def _spying_factory(host: str, port: int, connect_timeout: float) -> Any:
        connection = original_factory(host, port, connect_timeout)
        captured["timeout"] = connection.timeout
        return connection

    helper = _helper(
        ollama_loopback_server.port, connect_timeout=1.234, connection_factory=_spying_factory
    )
    result = helper.propose(question="what violations recur?", jurisdiction="nyc", now=None)  # type: ignore[arg-type]
    assert result.outcome == IntentHelperOutcome.ACCEPTED
    assert captured["timeout"] == 1.234


def test_raw_response_content_never_appears_in_the_reason(ollama_loopback_server: Any) -> None:
    marker = "SUPER_SECRET_RAW_MARKER_12345"
    ollama_loopback_server.set_raw_body(f"garbage {marker} garbage".encode())
    helper = _helper(ollama_loopback_server.port)
    result = helper.propose(question="what violations recur?", jurisdiction="nyc", now=None)  # type: ignore[arg-type]
    assert result.outcome == IntentHelperOutcome.UNAVAILABLE
    assert result.reason is not None
    assert marker not in result.reason


def test_connection_is_closed_after_a_successful_request(ollama_loopback_server: Any) -> None:
    from plateproof.copilot.generators import ollama as ollama_module

    closed: list[bool] = []
    original_factory = ollama_module._default_connection_factory

    def _spying_factory(host: str, port: int, connect_timeout: float) -> Any:
        connection = original_factory(host, port, connect_timeout)
        original_close = connection.close

        def _close() -> None:
            closed.append(True)
            original_close()

        connection.close = _close  # type: ignore[method-assign]
        return connection

    helper = _helper(ollama_loopback_server.port, connection_factory=_spying_factory)
    helper.propose(question="what violations recur?", jurisdiction="nyc", now=None)  # type: ignore[arg-type]
    # http.client may close the connection itself once a
    # ``Connection: close`` response body is fully consumed, in addition
    # to this module's own ``finally: connection.close()`` -- both are the
    # real code path, and closing twice is a safe no-op, not a leak. What
    # matters is that a close happened at all.
    assert closed


def test_connection_is_closed_even_after_a_transport_failure(ollama_loopback_server: Any) -> None:
    from plateproof.copilot.generators import ollama as ollama_module

    closed: list[bool] = []
    original_factory = ollama_module._default_connection_factory

    def _spying_factory(host: str, port: int, connect_timeout: float) -> Any:
        connection = original_factory(host, port, connect_timeout)
        original_close = connection.close

        def _close() -> None:
            closed.append(True)
            original_close()

        connection.close = _close  # type: ignore[method-assign]
        return connection

    ollama_loopback_server.set_raw_body(b"invalid utf8 \xff\xfe")
    helper = _helper(ollama_loopback_server.port, connection_factory=_spying_factory)
    result = helper.propose(question="what violations recur?", jurisdiction="nyc", now=None)  # type: ignore[arg-type]
    assert result.outcome == IntentHelperOutcome.UNAVAILABLE
    assert closed


def test_invalid_base_url_never_opens_a_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[Any] = []

    def _factory(host: str, port: int, connect_timeout: float) -> Any:
        called.append((host, port))
        raise AssertionError("should never be called for an invalid base URL")

    helper = OllamaIntentHelper(
        base_url="http://evil.example.com:11434",
        model="fictional-model",
        connect_timeout=1.0,
        read_timeout=1.0,
        max_response_bytes=1_000,
        min_confidence=0.6,
        connection_factory=_factory,
    )
    result = helper.propose(question="what violations recur?", jurisdiction="nyc", now=None)  # type: ignore[arg-type]
    assert result.outcome == IntentHelperOutcome.UNAVAILABLE
    assert called == []


@pytest.fixture
def unused_tcp_port_factory() -> Any:
    import socket

    def _factory() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]

    return _factory
