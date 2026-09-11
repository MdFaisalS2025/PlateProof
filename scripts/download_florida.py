"""Reproducible downloader for Florida DBPR public food-service inspection extracts.

Fetches files named in ``plateproof.ingestion.florida.FLORIDA_SOURCE_MANIFEST``
(the seven current-fiscal-year district CSVs, plus recent historical statewide
XLSX archives) and writes them, together with a ``manifest.json`` provenance
record, into one timestamped snapshot directory. ``_SUCCESS`` is written only
when every requested file was downloaded and finalized -- the complete
requested snapshot is what "success" means here (see ``allow_skipped`` below
for the one, explicit, opt-in exception). Network access is fully injectable so
tests never touch the network. DBPR requires no API key or app token.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import time
import urllib.error
import urllib.request
import warnings
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

import plateproof
from plateproof.ingestion.florida import (
    FL_DATASET_ID,
    FLORIDA_SOURCE_MANIFEST,
    FloridaSourceFileSpec,
    _decode_bytes,
)

__all__ = [
    "DownloadError",
    "is_valid_florida_snapshot",
    "run_florida_download",
    "main",
]

MAX_ATTEMPTS = 3
_BACKOFF_SECONDS = (1, 4)
_TIMEOUT_SECONDS = 60
DOWNLOADER_VERSION = plateproof.__version__

Opener = Callable[[str, Mapping[str, str]], Any]
Sleeper = Callable[[float], None]
Clock = Callable[[], datetime]


class DownloadError(RuntimeError):
    """Raised for an unrecoverable download problem, with an actionable message."""


def _is_retryable_status(status: int) -> bool:
    return status == 429 or 500 <= status <= 599


def _read_body(source: Any) -> bytes:
    try:
        data = source.read()
    except Exception:  # noqa: BLE001 - a body we cannot read is simply empty
        return b""
    return data if isinstance(data, bytes) else b""


def _fetch(url: str, *, opener: Opener, sleeper: Sleeper) -> tuple[int, dict[str, str], bytes]:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = opener(url, {})
        except urllib.error.HTTPError as exc:
            status = int(exc.code)
            if _is_retryable_status(status) and attempt < MAX_ATTEMPTS:
                sleeper(_BACKOFF_SECONDS[attempt - 1])
                continue
            error_text = _read_body(exc).decode("utf-8", "replace")
            raise DownloadError(
                f"Florida DBPR file server returned HTTP {status} for {url}. "
                f"Server said: {error_text}"
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt < MAX_ATTEMPTS:
                sleeper(_BACKOFF_SECONDS[attempt - 1])
                continue
            raise DownloadError(
                f"Could not reach {url} after {MAX_ATTEMPTS} attempts (last error: {exc})."
            ) from exc

        status = int(response.status)
        body = _read_body(response)
        if status == 200:
            return status, dict(getattr(response, "headers", {}) or {}), body
        if _is_retryable_status(status) and attempt < MAX_ATTEMPTS:
            sleeper(_BACKOFF_SECONDS[attempt - 1])
            continue
        raise DownloadError(f"Florida DBPR file server returned HTTP {status} for {url}.")

    raise DownloadError("Exhausted download attempts without a response.")  # pragma: no cover


def _iso_z(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat().replace("+00:00", "Z")


def is_valid_florida_snapshot(directory: str | Path, *, allow_partial: bool = False) -> bool:
    """A snapshot is usable when it has a manifest and a completion marker.

    ``_SUCCESS`` means every requested file is present and valid. A snapshot
    produced with ``allow_skipped=True`` (some requested files intentionally
    omitted) instead carries ``_PARTIAL_SUCCESS`` and is only considered valid
    here when ``allow_partial=True`` is passed explicitly -- a caller that does
    not ask for partial snapshots never silently accepts one.
    """
    base = Path(directory)
    if not (base / "manifest.json").is_file():
        return False
    if (base / "_SUCCESS").is_file():
        return True
    return allow_partial and (base / "_PARTIAL_SUCCESS").is_file()


def _resolve_requested(
    manifest: tuple[FloridaSourceFileSpec, ...],
    fiscal_years: list[str],
    districts: list[int] | None,
) -> tuple[list[FloridaSourceFileSpec], list[FloridaSourceFileSpec]]:
    requested = [
        entry
        for entry in manifest
        if entry.fiscal_year in fiscal_years
        and (entry.district is None or districts is None or entry.district in districts)
    ]
    supported = [e for e in requested if e.format in ("csv", "xlsx")]
    unsupported = [e for e in requested if e.format not in ("csv", "xlsx")]
    return supported, unsupported


def _xlsx_row_count(path: Path) -> int:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return pl.read_excel(path).height


def run_florida_download(
    *,
    output_dir: str | Path,
    fiscal_years: list[str],
    districts: list[int] | None = None,
    sample_rows: int | None = None,
    opener: Opener,
    sleeper: Sleeper,
    clock: Clock,
    manifest: tuple[FloridaSourceFileSpec, ...] = FLORIDA_SOURCE_MANIFEST,
    allow_skipped: bool = False,
) -> Path:
    """Download every requested manifest entry into one snapshot directory.

    ``sample_rows``, when set, truncates each downloaded **CSV** file's data
    rows only *after* that file's complete bytes have already been transferred
    and written to a temporary path -- it is a local development convenience,
    not a network optimization. DBPR's static file server supports no
    row-limited or otherwise partial retrieval for these files, so the full
    remote file is always fetched regardless of this option. XLSX files are
    not truncated in this version.

    If any requested fiscal year resolves only to an unsupported format (e.g.
    the deferred legacy ``.xls`` archives), this raises ``DownloadError``
    *before* downloading anything, unless ``allow_skipped=True`` is passed
    explicitly -- in which case those files are recorded in the manifest's
    ``skipped_files`` and the snapshot's ``status`` is ``"partial"`` (marked
    with ``_PARTIAL_SUCCESS`` instead of ``_SUCCESS``). A snapshot is never
    marked complete while silently omitting a requested file.
    """
    started = clock()
    supported, unsupported = _resolve_requested(manifest, fiscal_years, districts)
    supported_years = {e.fiscal_year for e in supported}
    missing_years = sorted(fy for fy in fiscal_years if fy not in supported_years)
    if missing_years and not allow_skipped:
        raise DownloadError(
            f"Requested fiscal year(s) {missing_years} have no supported (csv/xlsx) file in "
            "the manifest. Pass allow_skipped=True to proceed with an explicitly partial "
            "snapshot, or request a fiscal year with CSV/XLSX coverage."
        )
    skipped_entries = (
        [e for e in unsupported if e.fiscal_year in missing_years] if allow_skipped else []
    )

    snapshot_dir = Path(output_dir) / started.strftime("%Y%m%dT%H%M%SZ")
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    tmp_paths: list[Path] = []
    final_paths: list[Path] = []
    manifest_tmp = snapshot_dir / "manifest.json.part"
    success_marker = snapshot_dir / "_SUCCESS"
    partial_marker = snapshot_dir / "_PARTIAL_SUCCESS"

    try:
        file_records: list[dict[str, Any]] = []
        for entry in supported:
            local_name = Path(entry.url).name
            tmp_path = snapshot_dir / f"{local_name}.part"
            tmp_paths.append(tmp_path)
            final_paths.append(snapshot_dir / local_name)
            _status, _headers, body = _fetch(entry.url, opener=opener, sleeper=sleeper)

            if entry.format == "csv":
                text, encoding = _decode_bytes(body)
                rows = list(csv.reader(io.StringIO(text)))
                header, data = (rows[0], rows[1:]) if rows else ([], [])
                if sample_rows is not None:
                    data = data[:sample_rows]
                with tmp_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    if header:
                        writer.writerow(header)
                    writer.writerows(data)
                row_count = len(data)
            else:  # xlsx: written verbatim; not truncated in this version
                tmp_path.write_bytes(body)
                row_count = _xlsx_row_count(tmp_path)
                encoding = "xlsx"

            file_records.append(
                {
                    "url": entry.url,
                    "fiscal_year": entry.fiscal_year,
                    "district": entry.district,
                    "format": entry.format,
                    "local_filename": local_name,
                    "retrieved_at_utc": _iso_z(started),
                    "sha256": hashlib.sha256(tmp_path.read_bytes()).hexdigest(),
                    "byte_size": tmp_path.stat().st_size,
                    "row_count": row_count,
                    "encoding": encoding,
                }
            )

        status = "partial" if skipped_entries else "complete"
        payload = {
            "dataset_id": FL_DATASET_ID,
            "retrieved_at_utc": _iso_z(started),
            "requested_fiscal_years": fiscal_years,
            "requested_districts": districts or [],
            "files": file_records,
            "downloader_version": DOWNLOADER_VERSION,
            "status": status,
            "skipped_files": [
                {
                    "fiscal_year": e.fiscal_year,
                    "district": e.district,
                    "url": e.url,
                    "format": e.format,
                    "note": e.note,
                }
                for e in skipped_entries
            ],
            "terms": "DBPR Public Records — Chapter 119, F.S.",
        }
        manifest_tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

        for tmp_path, final_path in zip(tmp_paths, final_paths, strict=True):
            tmp_path.replace(final_path)
        manifest_tmp.replace(snapshot_dir / "manifest.json")
        (success_marker if status == "complete" else partial_marker).write_text(
            "", encoding="utf-8"
        )
        return snapshot_dir
    except BaseException:
        for path in (
            *tmp_paths,
            *final_paths,
            manifest_tmp,
            snapshot_dir / "manifest.json",
            success_marker,
            partial_marker,
        ):
            path.unlink(missing_ok=True)
        raise


def _urlopen(url: str, headers: Mapping[str, str]) -> Any:
    request = urllib.request.Request(url, headers=dict(headers))
    return urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS)  # noqa: S310


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Download Florida DBPR food-service extracts.")
    parser.add_argument("--output", required=True, help="Directory for the snapshot subfolder.")
    parser.add_argument("--fiscal-years", required=True, help="Comma-separated, e.g. current,2021")
    parser.add_argument("--districts", default=None, help="Comma-separated district numbers 1-7.")
    parser.add_argument("--sample-rows", type=int, default=None)
    parser.add_argument("--allow-skipped", action="store_true", default=False)
    args = parser.parse_args(argv)

    districts = [int(d) for d in args.districts.split(",")] if args.districts else None
    try:
        snapshot = run_florida_download(
            output_dir=args.output,
            fiscal_years=[fy.strip() for fy in args.fiscal_years.split(",")],
            districts=districts,
            sample_rows=args.sample_rows,
            opener=_urlopen,
            sleeper=time.sleep,
            clock=lambda: datetime.now(UTC),
            allow_skipped=args.allow_skipped,
        )
    except DownloadError as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        return 1
    print(str(snapshot))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
