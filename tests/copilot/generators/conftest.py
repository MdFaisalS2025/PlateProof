"""A real, in-process, 127.0.0.1-only HTTP server used to exercise
``plateproof.copilot.generators.ollama``'s actual transport code (SSRF
boundary, bounded reads, timeouts, connection cleanup) rather than mocking
``_send_request`` itself. This is loopback-only, started and stopped
per-test in a background thread -- it never reaches any external network,
so "no test may contact an external service" still holds."""

from __future__ import annotations

import http.server
import json
import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest


class _Handler(http.server.BaseHTTPRequestHandler):
    # Silence the default per-request stderr logging -- keeps test output
    # clean; nothing security-relevant is suppressed (it's just an access
    # log line).
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        server: _LoopbackOllamaServer = self.server  # type: ignore[assignment]
        server.received_paths.append(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        server.received_bodies.append(self.rfile.read(length))
        behavior = server.behavior

        if behavior["kind"] == "sleep_then_respond":
            time.sleep(behavior["seconds"])

        status = behavior.get("status", 200)
        self.send_response(status)
        if behavior["kind"] == "redirect":
            self.send_header("Location", "http://127.0.0.1:1/elsewhere")
        self.send_header("Content-Type", "application/json")
        body = behavior["body"]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            # The client (bounded reader) is allowed to stop reading and
            # close its side before we finish writing an oversized body.
            pass


class _LoopbackOllamaServer(http.server.HTTPServer):
    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.behavior: dict[str, Any] = {
            "kind": "respond",
            "status": 200,
            "body": json.dumps(
                {"response": json.dumps({"intent": "recurring_violations", "confidence": 0.9})}
            ).encode("utf-8"),
        }
        self.received_paths: list[str] = []
        self.received_bodies: list[bytes] = []

    @property
    def port(self) -> int:
        return self.server_address[1]

    def set_raw_body(self, body: bytes, *, status: int = 200) -> None:
        self.behavior = {"kind": "respond", "status": status, "body": body}

    def set_sleep(self, seconds: float, *, body: bytes = b"{}") -> None:
        self.behavior = {
            "kind": "sleep_then_respond",
            "seconds": seconds,
            "status": 200,
            "body": body,
        }

    def set_redirect(self) -> None:
        self.behavior = {"kind": "redirect", "status": 302, "body": b""}


@pytest.fixture
def ollama_loopback_server() -> Iterator[_LoopbackOllamaServer]:
    server = _LoopbackOllamaServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
