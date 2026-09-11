"""Tests for scripts/download_nyc.py. Nothing here touches the network."""

from __future__ import annotations

import csv
import email.message
import io
import json
import urllib.error
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

FIXED_TS = datetime(2026, 9, 9, 14, 30, 0, tzinfo=UTC)
HEADER = ["camis", "inspection_date", "score"]


def _records(count: int) -> list[list[str]]:
    return [[f"5000{i:04d}", "2024-05-10T00:00:00.000", str(i)] for i in range(count)]


def _csv_bytes(header: list[str], rows: list[list[str]]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


class _Resp:
    def __init__(self, status: int, body: bytes, headers: Mapping[str, str] | None = None) -> None:
        self.status = status
        self._body = body
        self.headers = dict(headers or {})

    def read(self) -> bytes:
        return self._body


def _http_error(
    code: int, body: bytes = b"", retry_after: int | None = None
) -> urllib.error.HTTPError:
    msg = email.message.Message()
    if retry_after is not None:
        msg["Retry-After"] = str(retry_after)
    return urllib.error.HTTPError(
        "https://data.cityofnewyork.us/x", code, "err", msg, io.BytesIO(body)
    )


class _ScriptedOpener:
    def __init__(self, behaviours: list[Any]) -> None:
        self._behaviours = list(behaviours)
        self.calls = 0

    def __call__(self, url: str, headers: Mapping[str, str]) -> _Resp:
        self.calls += 1
        item = self._behaviours.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class _FakeApi:
    """Serves records like the Socrata CSV endpoint, honouring $limit / $offset."""

    def __init__(
        self,
        records: list[list[str]],
        *,
        header: list[str] | None = None,
        over_return: int = 0,
        page_headers: list[list[str]] | None = None,
    ) -> None:
        self._records = records
        self._header = header or HEADER
        self._over_return = over_return
        self._page_headers = page_headers
        self.urls: list[str] = []

    def __call__(self, url: str, headers: Mapping[str, str]) -> _Resp:
        from urllib.parse import parse_qs, urlparse

        self.urls.append(url)
        query = parse_qs(urlparse(url).query)
        limit = int(query["$limit"][0])
        offset = int(query["$offset"][0])
        take = limit + self._over_return
        page = self._records[offset : offset + take]
        header = self._header
        if self._page_headers is not None:
            header = self._page_headers[len(self.urls) - 1]
        return _Resp(200, _csv_bytes(header, page))


def _run(tmp_path: Path, api: Any, **kwargs: Any) -> Path:
    from scripts.download_nyc import run_download

    defaults: dict[str, Any] = dict(
        output_dir=tmp_path / "nyc",
        limit=10,
        offset=0,
        max_rows=None,
        where=None,
        app_token=None,
        opener=api,
        sleeper=lambda _seconds: None,
        clock=lambda: FIXED_TS,
        metadata_fetcher=lambda: {"rowsUpdatedAt": "2026-09-08T05:00:00Z"},
    )
    defaults.update(kwargs)
    return run_download(**defaults)


# --- pure helpers ---------------------------------------------------------


def test_build_page_url_includes_order_limit_offset_where() -> None:
    from scripts.download_nyc import NYC_RESOURCE_CSV_URL, build_page_url

    url = build_page_url(NYC_RESOURCE_CSV_URL, limit=10, offset=20, where="score > 10")
    assert "%3Aid" in url and "$order" in url
    assert "$limit=10" in url
    assert "$offset=20" in url
    assert "score" in url and "10" in url


# --- retry policy -------------------------------------------------------


def test_retry_policy_attempts_and_sleeps() -> None:
    from scripts.download_nyc import _fetch

    opener = _ScriptedOpener([_http_error(429), _http_error(500), _Resp(200, b"ok")])
    sleeps: list[float] = []
    status, _headers, body = _fetch(
        "https://data.cityofnewyork.us/x", opener=opener, sleeper=sleeps.append, app_token=None
    )
    assert opener.calls == 3
    assert sleeps == [1, 4]
    assert status == 200 and body == b"ok"


def test_retry_exhausted_raises_after_three_attempts() -> None:
    import pytest

    from scripts.download_nyc import DownloadError, _fetch

    opener = _ScriptedOpener([_http_error(500), _http_error(500), _http_error(500)])
    sleeps: list[float] = []
    with pytest.raises(DownloadError):
        _fetch(
            "https://data.cityofnewyork.us/x", opener=opener, sleeper=sleeps.append, app_token=None
        )
    assert opener.calls == 3
    assert sleeps == [1, 4]


def test_no_retry_on_http_400_mentions_where() -> None:
    import pytest

    from scripts.download_nyc import DownloadError, _fetch

    opener = _ScriptedOpener([_http_error(400, b"Invalid SoQL near WHERE")])
    with pytest.raises(DownloadError, match="where"):
        _fetch(
            "https://data.cityofnewyork.us/x",
            opener=opener,
            sleeper=lambda _s: None,
            app_token=None,
        )
    assert opener.calls == 1


def test_no_retry_on_http_403_mentions_app_token() -> None:
    import pytest

    from scripts.download_nyc import DownloadError, _fetch

    opener = _ScriptedOpener([_http_error(403, b"Forbidden")])
    with pytest.raises(DownloadError, match="PLATEPROOF_SODA_APP_TOKEN"):
        _fetch(
            "https://data.cityofnewyork.us/x",
            opener=opener,
            sleeper=lambda _s: None,
            app_token=None,
        )
    assert opener.calls == 1


def test_retry_on_urlerror_then_succeeds() -> None:
    from scripts.download_nyc import _fetch

    opener = _ScriptedOpener([urllib.error.URLError("temporary failure"), _Resp(200, b"ok")])
    sleeps: list[float] = []
    status, _headers, _body = _fetch(
        "https://data.cityofnewyork.us/x", opener=opener, sleeper=sleeps.append, app_token=None
    )
    assert status == 200
    assert opener.calls == 2
    assert sleeps == [1]


def test_honours_retry_after_header() -> None:
    from scripts.download_nyc import _fetch

    opener = _ScriptedOpener([_http_error(429, retry_after=10), _Resp(200, b"ok")])
    sleeps: list[float] = []
    _fetch("https://data.cityofnewyork.us/x", opener=opener, sleeper=sleeps.append, app_token=None)
    assert sleeps == [10]


def test_retry_after_is_capped() -> None:
    from scripts.download_nyc import RETRY_AFTER_MAX_SECONDS, _fetch

    opener = _ScriptedOpener([_http_error(429, retry_after=9999), _Resp(200, b"ok")])
    sleeps: list[float] = []
    _fetch("https://data.cityofnewyork.us/x", opener=opener, sleeper=sleeps.append, app_token=None)
    assert sleeps == [RETRY_AFTER_MAX_SECONDS]


# --- snapshot assembly -------------------------------------------------


def test_complete_snapshot_has_csv_metadata_and_success(tmp_path: Path) -> None:
    from scripts.download_nyc import is_valid_snapshot

    api = _FakeApi(_records(25))
    snapshot = _run(tmp_path, api, limit=10)
    assert (snapshot / "raw.csv").exists()
    assert (snapshot / "metadata.json").exists()
    assert (snapshot / "_SUCCESS").exists()
    assert is_valid_snapshot(snapshot)

    meta = json.loads((snapshot / "metadata.json").read_text(encoding="utf-8"))
    assert meta["dataset_id"] == "43nn-pn8j"
    assert meta["row_count"] == 25
    assert meta["requested_start_offset"] == 0
    assert len(meta["source_sha256"]) == 64
    assert meta["app_token_used"] is False


def test_header_written_once_across_pages(tmp_path: Path) -> None:
    api = _FakeApi(_records(25))
    snapshot = _run(tmp_path, api, limit=10)
    with (snapshot / "raw.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == HEADER
    assert len(rows) == 26  # 1 header + 25 records
    assert HEADER not in rows[1:]


def test_header_change_between_pages_is_fatal(tmp_path: Path) -> None:
    import pytest

    from scripts.download_nyc import DownloadError

    api = _FakeApi(
        _records(20),
        page_headers=[["camis", "inspection_date", "score"], ["camis", "inspection_date", "grade"]],
    )
    with pytest.raises(DownloadError, match="header"):
        _run(tmp_path, api, limit=10)
    snapshot_root = tmp_path / "nyc"
    for child in snapshot_root.glob("*/*"):
        assert child.name != "_SUCCESS"
        assert not child.name.endswith(".part")


def test_quoted_field_with_comma_and_newline_round_trips(tmp_path: Path) -> None:
    records = [["50000001", "2024-05-10T00:00:00.000", "note: a, b\nsecond line"]]
    api = _FakeApi(records)
    snapshot = _run(tmp_path, api, limit=10)
    with (snapshot / "raw.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[1][2] == "note: a, b\nsecond line"
    meta = json.loads((snapshot / "metadata.json").read_text(encoding="utf-8"))
    assert meta["row_count"] == 1


def test_max_rows_reduces_final_request_limit(tmp_path: Path) -> None:
    api = _FakeApi(_records(100))
    snapshot = _run(tmp_path, api, limit=40, max_rows=50)
    with (snapshot / "raw.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == 51  # header + 50
    assert any("$limit=10" in url and "$offset=40" in url for url in api.urls)
    meta = json.loads((snapshot / "metadata.json").read_text(encoding="utf-8"))
    assert meta["row_count"] == 50
    assert meta["requested_start_offset"] == 0


def test_max_rows_truncates_overlong_page(tmp_path: Path) -> None:
    api = _FakeApi(_records(100), over_return=30)
    snapshot = _run(tmp_path, api, limit=40, max_rows=50)
    with (snapshot / "raw.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == 51
    meta = json.loads((snapshot / "metadata.json").read_text(encoding="utf-8"))
    assert meta["row_count"] == 50


def test_metadata_retrieval_failure_leaves_no_apparent_snapshot(tmp_path: Path) -> None:
    import pytest

    from scripts.download_nyc import DownloadError

    def _boom() -> dict[str, Any]:
        raise DownloadError("socrata metadata unavailable")

    api = _FakeApi(_records(10))
    with pytest.raises(DownloadError):
        _run(tmp_path, api, limit=10, metadata_fetcher=_boom)
    for child in (tmp_path / "nyc").glob("*/*"):
        assert child.name not in {"_SUCCESS", "raw.csv", "metadata.json"}


def test_metadata_serialisation_failure_leaves_no_success(tmp_path: Path) -> None:
    import pytest

    api = _FakeApi(_records(10))
    with pytest.raises((TypeError, ValueError)):
        _run(tmp_path, api, limit=10, metadata_fetcher=lambda: {"rowsUpdatedAt": object()})
    for child in (tmp_path / "nyc").glob("*/*"):
        assert child.name != "_SUCCESS"


def test_is_valid_snapshot_requires_all_three(tmp_path: Path) -> None:
    from scripts.download_nyc import is_valid_snapshot

    directory = tmp_path / "snap"
    directory.mkdir()
    (directory / "raw.csv").write_text("camis\n", encoding="utf-8")
    assert not is_valid_snapshot(directory)
    (directory / "metadata.json").write_text("{}", encoding="utf-8")
    assert not is_valid_snapshot(directory)
    (directory / "_SUCCESS").write_text("", encoding="utf-8")
    assert is_valid_snapshot(directory)


def test_runs_without_app_token(tmp_path: Path) -> None:
    api = _FakeApi(_records(5))
    snapshot = _run(tmp_path, api, limit=10, app_token=None)
    meta = json.loads((snapshot / "metadata.json").read_text(encoding="utf-8"))
    assert meta["app_token_used"] is False
    assert (snapshot / "_SUCCESS").exists()
