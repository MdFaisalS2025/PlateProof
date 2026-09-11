"""Reproducible downloader for the NYC DOHMH restaurant-inspection extract.

Fetches the dataset from the NYC Open Data SODA CSV endpoint page by page, writes
``raw.csv`` plus a ``metadata.json`` provenance sidecar into a timestamped
snapshot directory, and marks the snapshot complete with a ``_SUCCESS`` file.
Downstream ingestion must treat a snapshot as valid only when ``_SUCCESS`` exists.

An app token is optional and free; unauthenticated requests work but are
throttled. Nothing here requires a paid service. Network access is fully
injectable so tests never touch the network.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import plateproof
from plateproof.ingestion.nyc import (
    NYC_DATASET_ID,
    NYC_LANDING_PAGE,
    NYC_RESOURCE_CSV_URL,
    NYC_VIEWS_METADATA_URL,
)

__all__ = [
    "NYC_RESOURCE_CSV_URL",
    "RETRY_AFTER_MAX_SECONDS",
    "DownloadError",
    "build_page_url",
    "is_valid_snapshot",
    "run_download",
    "main",
]

RETRY_AFTER_MAX_SECONDS = 60
MAX_ATTEMPTS = 3
_BACKOFF_SECONDS = (1, 4)  # slept after attempt 1 and attempt 2 only
_TIMEOUT_SECONDS = 60
DEFAULT_PAGE_LIMIT = 50_000
DOWNLOADER_VERSION = plateproof.__version__
_DEV_SETTINGS_URL = "https://data.cityofnewyork.us/profile/edit/developer_settings"

Opener = Callable[[str, Mapping[str, str]], Any]
Sleeper = Callable[[float], None]
Clock = Callable[[], datetime]
MetadataFetcher = Callable[[], dict[str, Any]]


class DownloadError(RuntimeError):
    """Raised for an unrecoverable download problem, with an actionable message."""


def build_page_url(base_url: str, *, limit: int, offset: int, where: str | None) -> str:
    params: dict[str, str] = {"$order": ":id", "$limit": str(limit), "$offset": str(offset)}
    if where:
        params["$where"] = where
    return f"{base_url}?{urlencode(params, safe='$')}"


def _is_retryable_status(status: int) -> bool:
    return status == 429 or 500 <= status <= 599


def _retry_delay(attempt: int, headers: Mapping[str, str]) -> float:
    base = _BACKOFF_SECONDS[attempt - 1]
    retry_after = headers.get("Retry-After")
    if not retry_after:
        return base
    try:
        seconds = int(retry_after)
    except (TypeError, ValueError):
        return min(base, RETRY_AFTER_MAX_SECONDS)
    return max(0, min(seconds, RETRY_AFTER_MAX_SECONDS))


def _status_message(status: int, body: str) -> str:
    if status == 400:
        return (
            "NYC Open Data returned HTTP 400 (bad request). Check the --where SoQL "
            f"filter syntax. Server said: {body}"
        )
    if status == 403:
        return (
            "NYC Open Data returned HTTP 403 (forbidden). Unauthenticated requests are "
            f"throttled. Register a free app token at {_DEV_SETTINGS_URL} and pass it via "
            "the PLATEPROOF_SODA_APP_TOKEN environment variable."
        )
    if status == 429:
        return (
            f"NYC Open Data rate-limited the download (HTTP 429) after {MAX_ATTEMPTS} "
            "attempts. Wait a few minutes or use an app token."
        )
    return f"NYC Open Data returned HTTP {status}. Server said: {body}"


def _read_body(source: Any) -> bytes:
    try:
        data = source.read()
    except Exception:  # noqa: BLE001 - a body we cannot read is simply empty
        return b""
    return data if isinstance(data, bytes) else b""


def _fetch(
    url: str,
    *,
    opener: Opener,
    sleeper: Sleeper,
    app_token: str | None,
) -> tuple[int, dict[str, str], bytes]:
    headers: dict[str, str] = {"Accept": "text/csv"}
    if app_token:
        headers["X-App-Token"] = app_token

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = opener(url, headers)
        except urllib.error.HTTPError as exc:
            status = int(exc.code)
            response_headers = {key: value for key, value in (exc.headers or {}).items()}
            if _is_retryable_status(status) and attempt < MAX_ATTEMPTS:
                sleeper(_retry_delay(attempt, response_headers))
                continue
            raise DownloadError(
                _status_message(status, _read_body(exc).decode("utf-8", "replace"))
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt < MAX_ATTEMPTS:
                sleeper(_BACKOFF_SECONDS[attempt - 1])
                continue
            raise DownloadError(
                f"Could not reach NYC Open Data after {MAX_ATTEMPTS} attempts (last error: {exc})."
            ) from exc

        status = int(response.status)
        response_headers = {
            key: value for key, value in dict(getattr(response, "headers", {})).items()
        }
        body = _read_body(response)
        if status == 200:
            return status, response_headers, body
        if _is_retryable_status(status) and attempt < MAX_ATTEMPTS:
            sleeper(_retry_delay(attempt, response_headers))
            continue
        raise DownloadError(_status_message(status, body.decode("utf-8", "replace")))

    raise DownloadError("Exhausted download attempts without a response.")  # pragma: no cover


def _download_pages(
    *,
    base_url: str,
    limit: int,
    offset: int,
    max_rows: int | None,
    where: str | None,
    app_token: str | None,
    opener: Opener,
    sleeper: Sleeper,
    target: Path,
) -> tuple[list[str], int]:
    header: list[str] | None = None
    written = 0
    consumed = 0
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        while max_rows is None or written < max_rows:
            page_limit = limit if max_rows is None else min(limit, max_rows - written)
            url = build_page_url(base_url, limit=page_limit, offset=offset + consumed, where=where)
            _status, _headers, body = _fetch(
                url, opener=opener, sleeper=sleeper, app_token=app_token
            )
            page = list(csv.reader(io.StringIO(body.decode("utf-8"))))
            if not page:
                break
            page_header, records = page[0], page[1:]
            if header is None:
                header = page_header
                writer.writerow(header)
            elif page_header != header:
                raise DownloadError(
                    f"CSV header changed between pages: first={header}, later={page_header}."
                )
            if not records:
                break
            consumed += len(records)
            if max_rows is not None and written + len(records) > max_rows:
                records = records[: max_rows - written]
            writer.writerows(records)
            written += len(records)
            if len(records) < page_limit:
                break
    if header is None:
        raise DownloadError("NYC Open Data returned no CSV header.")
    return header, written


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _iso_z(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat().replace("+00:00", "Z")


def is_valid_snapshot(directory: str | Path) -> bool:
    """A snapshot is usable only when raw.csv, metadata.json and _SUCCESS all exist."""
    base = Path(directory)
    return all((base / name).is_file() for name in ("raw.csv", "metadata.json", "_SUCCESS"))


def run_download(
    *,
    output_dir: str | Path,
    limit: int = DEFAULT_PAGE_LIMIT,
    offset: int = 0,
    max_rows: int | None = None,
    where: str | None = None,
    app_token: str | None = None,
    opener: Opener,
    sleeper: Sleeper,
    clock: Clock,
    metadata_fetcher: MetadataFetcher,
    base_url: str = NYC_RESOURCE_CSV_URL,
) -> Path:
    """Download the dataset into a timestamped snapshot directory and return it.

    ``raw.csv`` and ``metadata.json`` are written to ``*.part`` paths first; the
    SHA-256 is taken from the completed temporary CSV; the Socrata view metadata
    is fetched before anything is finalized; and ``_SUCCESS`` is written last. Any
    failure removes the temporary and final files so no apparently complete
    snapshot is left behind.
    """
    started = clock()
    snapshot_dir = Path(output_dir) / started.strftime("%Y%m%dT%H%M%SZ")
    tmp_csv = snapshot_dir / "raw.csv.part"
    tmp_meta = snapshot_dir / "metadata.json.part"
    final_csv = snapshot_dir / "raw.csv"
    final_meta = snapshot_dir / "metadata.json"
    marker = snapshot_dir / "_SUCCESS"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    try:
        _header, written = _download_pages(
            base_url=base_url,
            limit=limit,
            offset=offset,
            max_rows=max_rows,
            where=where,
            app_token=app_token,
            opener=opener,
            sleeper=sleeper,
            target=tmp_csv,
        )
        socrata_metadata = metadata_fetcher()
        payload = {
            "dataset_id": NYC_DATASET_ID,
            "source_url": base_url,
            "landing_page": NYC_LANDING_PAGE,
            "retrieved_at_utc": _iso_z(started),
            "soql_where": where,
            "requested_start_offset": offset,
            "row_count": written,
            "max_rows": max_rows,
            "source_sha256": _sha256_file(tmp_csv),
            "socrata_rows_updated_at": socrata_metadata.get("rowsUpdatedAt"),
            "app_token_used": bool(app_token),
            "downloader_version": DOWNLOADER_VERSION,
            "terms": "NYC Open Data Terms of Use",
        }
        serialized = json.dumps(payload, indent=2, sort_keys=True)
        tmp_meta.write_text(serialized, encoding="utf-8")
        tmp_csv.replace(final_csv)
        tmp_meta.replace(final_meta)
        marker.write_text("", encoding="utf-8")
        return snapshot_dir
    except BaseException:
        for path in (tmp_csv, tmp_meta, final_csv, final_meta, marker):
            path.unlink(missing_ok=True)
        raise


def _urlopen(url: str, headers: Mapping[str, str]) -> Any:
    request = urllib.request.Request(url, headers=dict(headers))
    return urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS)  # noqa: S310


def _fetch_view_metadata(opener: Opener, sleeper: Sleeper, app_token: str | None) -> dict[str, Any]:
    _status, _headers, body = _fetch(
        NYC_VIEWS_METADATA_URL, opener=opener, sleeper=sleeper, app_token=app_token
    )
    parsed = json.loads(body.decode("utf-8"))
    return parsed if isinstance(parsed, dict) else {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download the NYC DOHMH inspection extract.")
    parser.add_argument("--output", required=True, help="Directory for the snapshot subfolder.")
    parser.add_argument("--limit", type=int, default=DEFAULT_PAGE_LIMIT, help="Rows per API page.")
    parser.add_argument("--offset", type=int, default=0, help="Starting API offset.")
    parser.add_argument("--max-rows", type=int, default=None, help="Cap on written data records.")
    parser.add_argument("--where", default=None, help="Optional SoQL $where filter.")
    parser.add_argument(
        "--app-token-env",
        default="PLATEPROOF_SODA_APP_TOKEN",
        help="Environment variable holding an optional Socrata app token.",
    )
    args = parser.parse_args(argv)
    app_token = os.environ.get(args.app_token_env) or None

    try:
        snapshot = run_download(
            output_dir=args.output,
            limit=args.limit,
            offset=args.offset,
            max_rows=args.max_rows,
            where=args.where,
            app_token=app_token,
            opener=_urlopen,
            sleeper=time.sleep,
            clock=lambda: datetime.now(UTC),
            metadata_fetcher=lambda: _fetch_view_metadata(_urlopen, time.sleep, app_token),
        )
    except DownloadError as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        return 1
    print(str(snapshot))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
