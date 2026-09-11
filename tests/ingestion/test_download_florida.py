"""Tests for scripts/download_florida.py. Nothing here touches the network."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import urllib.error
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

FIXED_TS = datetime(2026, 2, 1, 5, 0, 0, tzinfo=UTC)


def _csv_bytes(n: int) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["a", "b"])
    for i in range(n):
        w.writerow([str(i), "x"])
    return buf.getvalue().encode("utf-8")


class _Resp:
    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self._body = body
        self.headers: dict[str, str] = {}

    def read(self) -> bytes:
        return self._body


class _FakeOpener:
    def __init__(self, by_url: dict[str, bytes]) -> None:
        self._by_url = by_url
        self.calls: list[str] = []

    def __call__(self, url: str, headers: Mapping[str, str]) -> _Resp:
        self.calls.append(url)
        for key, body in self._by_url.items():
            if key in url:
                return _Resp(200, body)
        raise urllib.error.HTTPError(url, 404, "not found", None, io.BytesIO(b""))


def _run(tmp_path: Path, opener: Any, **kwargs: Any) -> Path:
    from plateproof.ingestion.florida import FLORIDA_SOURCE_MANIFEST
    from scripts.download_florida import run_florida_download

    defaults: dict[str, Any] = dict(
        output_dir=tmp_path / "florida",
        fiscal_years=["current"],
        districts=None,
        sample_rows=None,
        opener=opener,
        sleeper=lambda _s: None,
        clock=lambda: FIXED_TS,
        manifest=FLORIDA_SOURCE_MANIFEST,
    )
    defaults.update(kwargs)
    return run_florida_download(**defaults)


def test_complete_snapshot_downloads_all_seven_current_districts(tmp_path: Path) -> None:
    from scripts.download_florida import is_valid_florida_snapshot

    opener = _FakeOpener({"fdinspi.csv": _csv_bytes(5)})
    snapshot = _run(tmp_path, opener)
    assert is_valid_florida_snapshot(snapshot)
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert len(manifest["files"]) == 7
    for entry in manifest["files"]:
        assert (snapshot / entry["local_filename"]).exists()
        assert (
            entry["sha256"]
            == hashlib.sha256((snapshot / entry["local_filename"]).read_bytes()).hexdigest()
        )
        assert entry["row_count"] == 5


def test_requested_unsupported_fiscal_year_fails_before_finalization(tmp_path: Path) -> None:
    import pytest

    from scripts.download_florida import DownloadError

    opener = _FakeOpener({"fdinspi.csv": _csv_bytes(3)})
    with pytest.raises(DownloadError, match="1920"):
        _run(tmp_path, opener, fiscal_years=["1920"])
    assert not (tmp_path / "florida").exists() or not list((tmp_path / "florida").glob("*/*"))


def test_partial_failure_leaves_no_success_and_no_files(tmp_path: Path) -> None:
    import pytest

    from scripts.download_florida import DownloadError

    opener = _FakeOpener({"1fdinspi.csv": _csv_bytes(3)})  # districts 2-7 will 404
    with pytest.raises(DownloadError):
        _run(tmp_path, opener)
    root = tmp_path / "florida"
    for child in root.glob("*/*"):
        assert child.name not in {"_SUCCESS", "manifest.json"}
        assert not child.name.endswith(".part")


def test_allow_skipped_mode_records_skipped_files_and_partial_status(tmp_path: Path) -> None:
    from scripts.download_florida import is_valid_florida_snapshot

    opener = _FakeOpener({"fdinspi.csv": _csv_bytes(2)})
    snapshot = _run(
        tmp_path,
        opener,
        fiscal_years=["current", "1920"],
        allow_skipped=True,
    )
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "partial"
    assert manifest["skipped_files"]
    assert not is_valid_florida_snapshot(snapshot)  # strict check: not a complete snapshot
    assert is_valid_florida_snapshot(snapshot, allow_partial=True)


def test_sample_rows_truncates_after_full_download_docstring_is_honest() -> None:
    from scripts.download_florida import run_florida_download

    assert "after" in (run_florida_download.__doc__ or "").lower()
    assert "transfer" in (run_florida_download.__doc__ or "").lower()


def test_sample_rows_truncates_csv_output(tmp_path: Path) -> None:
    opener = _FakeOpener({"fdinspi.csv": _csv_bytes(100)})
    snapshot = _run(tmp_path, opener, sample_rows=3)
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        assert entry["row_count"] == 3
        with (snapshot / entry["local_filename"]).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        assert len(rows) == 4  # header + 3


def test_retry_then_success(tmp_path: Path) -> None:
    from scripts.download_florida import _fetch

    calls: list[int] = []

    def opener(url: str, headers: Mapping[str, str]) -> _Resp:
        calls.append(1)
        if len(calls) == 1:
            raise urllib.error.HTTPError(url, 500, "err", None, io.BytesIO(b""))
        return _Resp(200, b"ok")

    sleeps: list[float] = []
    status, _headers, body = _fetch("https://x", opener=opener, sleeper=sleeps.append)
    assert status == 200 and body == b"ok"
    assert sleeps == [1]


def test_is_valid_florida_snapshot_requires_success_and_manifest(tmp_path: Path) -> None:
    from scripts.download_florida import is_valid_florida_snapshot

    directory = tmp_path / "snap"
    directory.mkdir()
    assert not is_valid_florida_snapshot(directory)
    (directory / "manifest.json").write_text("{}", encoding="utf-8")
    assert not is_valid_florida_snapshot(directory)
    (directory / "_SUCCESS").write_text("", encoding="utf-8")
    assert is_valid_florida_snapshot(directory)
