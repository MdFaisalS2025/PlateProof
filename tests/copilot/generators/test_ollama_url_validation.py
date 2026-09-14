"""RED-first tests for plateproof.copilot.generators.ollama.validate_loopback_url
-- the SSRF boundary. Pure function, no sockets involved. Whichever allowed
hostname is configured, the validated endpoint always connects via the
127.0.0.1 IP literal (never a DNS lookup of "localhost")."""

from __future__ import annotations

import pytest

from plateproof.copilot.generators.ollama import validate_loopback_url


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:11434",
        "http://127.0.0.1:11434",
        "http://LOCALHOST:11434",
        "http://localhost:11434/",
        "http://localhost:1024",
        "http://localhost:65535",
    ],
)
def test_valid_loopback_urls_are_accepted_and_connect_via_ip_literal(url: str) -> None:
    endpoint, reason = validate_loopback_url(url)
    assert reason is None
    assert endpoint is not None
    assert endpoint.connect_host == "127.0.0.1"


@pytest.mark.parametrize(
    "url,label",
    [
        ("https://localhost:11434", "https scheme"),
        ("http://example.com:11434", "non-loopback hostname"),
        ("http://notlocalhost:11434", "hostname merely containing localhost-like text"),
        ("http://localhost.evil.com:11434", "hostname suffix trick"),
        ("http://192.168.1.5:11434", "private LAN address"),
        ("http://8.8.8.8:11434", "public IP address"),
        ("http://2130706433:11434", "decimal IP encoding of 127.0.0.1"),
        ("http://0x7f000001:11434", "hex IP encoding of 127.0.0.1"),
        ("http://0177.0.0.1:11434", "octal IP encoding of 127.0.0.1"),
        ("http://[::1]:11434", "IPv6 loopback (not supported)"),
        ("http://user:pass@localhost:11434", "embedded credentials"),
        ("http://localhost:11434/api/generate", "non-empty configured path"),
        ("http://localhost:11434/%2e%2e", "encoded path traversal"),
        ("http://localhost:11434?x=1", "query string"),
        ("http://localhost:11434#frag", "fragment"),
        ("http://localhost", "missing port"),
        ("http://localhost:80", "privileged/low port outside policy"),
        ("http://localhost:99999", "port out of range"),
        ("ftp://localhost:11434", "non-http scheme"),
        ("", "empty string"),
    ],
)
def test_invalid_or_unsafe_urls_are_rejected(url: str, label: str) -> None:
    endpoint, reason = validate_loopback_url(url)
    assert endpoint is None, label
    assert reason is not None, label


def test_non_string_input_is_rejected() -> None:
    endpoint, reason = validate_loopback_url(None)
    assert endpoint is None
    assert reason is not None
