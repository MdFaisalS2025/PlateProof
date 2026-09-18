"""Bounded, spawn-context, killable worker-process pool.

Each worker process handles exactly one document job at a time -- there is
never more than one PDFium/OCR call in flight inside any single process, and
different processes have entirely separate address space by construction.
``multiprocessing.get_context("spawn")`` is used on every platform (not just
Windows) so a worker never inherits partially-initialized parent state.

Two independently-tracked deadlines are enforced *from the parent process
watching the child* (never trusted from inside the child, which could be
compromised or wedged): a per-page timeout, reset only by a genuine
``page_progress`` message from the worker, and a whole-job total timeout.
On either deadline, or any protocol violation, the worker is terminated via
``terminate()`` -> a short grace period -> `kill()`, and is never reused for
a later job -- a fresh ``spawn``-context process replaces it in that slot the
next time it is needed.
"""

from __future__ import annotations

import math
import multiprocessing
import queue
import struct
import sys
import threading
import time
import zlib
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from multiprocessing.connection import Connection
from multiprocessing.context import SpawnContext
from typing import Any

from plateproof.documents.limits import (
    ABSOLUTE_MAX_ADMISSION_TIMEOUT_SECONDS,
    ABSOLUTE_MAX_KILL_GRACE_SECONDS,
    ABSOLUTE_MAX_PAGE_TIMEOUT_SECONDS,
    ABSOLUTE_MAX_POOL_SIZE,
    ABSOLUTE_MAX_PREVIEW_FRAME_BYTES,
    ABSOLUTE_MAX_PREVIEW_HEIGHT_PX,
    ABSOLUTE_MAX_PREVIEW_PIXELS,
    ABSOLUTE_MAX_PREVIEW_WIDTH_PX,
    ABSOLUTE_MAX_TOTAL_PREVIEW_BYTES,
    ABSOLUTE_MAX_TOTAL_TIMEOUT_SECONDS,
    ABSOLUTE_MAX_UPLOAD_BYTES,
    DEFAULT_ADMISSION_TIMEOUT_SECONDS,
    DEFAULT_KILL_GRACE_SECONDS,
    DEFAULT_PAGE_TIMEOUT_SECONDS,
    DEFAULT_POOL_SIZE,
    DEFAULT_TOTAL_TIMEOUT_SECONDS,
    MIN_ADMISSION_TIMEOUT_SECONDS,
    MIN_KILL_GRACE_SECONDS,
    MIN_POOL_SIZE,
)
from plateproof.documents.worker.protocol import (
    MAX_WORKER_FRAME_BYTES,
    PROTOCOL_VERSION,
    ProtocolViolationError,
    ValidatedPreview,
    WorkerBusy,
    WorkerCrashed,
    WorkerInvalidResponse,
    WorkerJobError,
    WorkerJobRequest,
    WorkerJobResponse,
    WorkerPageProgress,
    WorkerPreviewHeader,
    WorkerTimeout,
    recv_bytes_frame,
    recv_frame,
    send_bytes_frame,
    send_frame,
    validate_worker_message,
)


@contextmanager
def _safe_main_module_for_spawn() -> Iterator[None]:
    """Streamlit's own script runner replaces ``sys.modules["__main__"]``
    with a fake, bare (``__spec__``-less) module wrapping the CURRENTLY
    EXECUTING PAGE SCRIPT on every single rerun
    (``streamlit/runtime/scriptrunner/script_runner.py``), so that pickling
    works inside page code. On Windows, ``multiprocessing``'s spawn
    bootstrap (``multiprocessing.spawn.get_preparation_data``) checks
    whether ``__main__`` has a real module spec; a bare module has none, so
    it falls back to reconstructing the child's ``__main__`` by literally
    re-executing ``sys.modules["__main__"].__file__`` via
    ``runpy.run_path`` -- for a Streamlit page, that file has no real
    ``ScriptRunContext`` and crashes immediately, verified directly against
    a real ``AppTest`` run. Pointing ``__main__`` at this own, real,
    properly-spec'd module for the duration of ``Process.start()`` makes
    the child instead safely re-``import`` it by name -- never re-execute
    arbitrary page code -- and the original ``__main__`` (whatever it was)
    is restored immediately afterward, so nothing about a caller's own
    ``__main__`` is permanently changed. A no-op for a normal API/CLI/test
    process, whose ``__main__`` already has a real spec."""
    original = sys.modules.get("__main__")
    try:
        sys.modules["__main__"] = sys.modules[__name__]
        yield
    finally:
        if original is not None:
            sys.modules["__main__"] = original
        else:
            del sys.modules["__main__"]


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_CHUNK_HEADER_SIZE = 8  # 4-byte big-endian length + 4-byte ASCII type
_PNG_CHUNK_CRC_SIZE = 4
_PNG_IHDR_LENGTH = 13

#: The exact, narrow PNG form our own worker-side encoder (Pillow saving an
#: RGB image with no extra options) actually produces: 8-bit truecolor, no
#: palette/alpha, no compression/filter/interlace variants, and no
#: ancillary chunks (EXIF, text, gamma, ICC profile, etc.) at all. Anything
#: outside this exact shape is rejected -- this is deliberately not a
#: general-purpose PNG decoder.
_SUPPORTED_BIT_DEPTH = 8
_SUPPORTED_COLOR_TYPE = 2
_SUPPORTED_COMPRESSION_METHOD = 0
_SUPPORTED_FILTER_METHOD = 0
_SUPPORTED_INTERLACE_METHOD = 0
_ALLOWED_PNG_CHUNK_TYPES = frozenset({"IHDR", "IDAT", "IEND"})


def _parse_png_chunks(data: bytes) -> list[tuple[str, bytes]] | None:
    """Splits ``data`` into ``(chunk_type, chunk_data)`` pairs after the
    8-byte PNG signature. Returns ``None`` for anything structurally
    invalid: a chunk header that doesn't fit in the remaining bytes, a
    declared chunk length that would run past the end of ``data`` (the
    frame is already capped upstream, but this bounds the *chunk*, never
    trusting a length field on its own), a CRC32 mismatch, or any byte left
    over after the last chunk is consumed (no trailing bytes are ever
    tolerated)."""
    if not data.startswith(_PNG_SIGNATURE):
        return None
    pos = len(_PNG_SIGNATURE)
    total = len(data)
    chunks: list[tuple[str, bytes]] = []
    while pos < total:
        if pos + _PNG_CHUNK_HEADER_SIZE > total:
            return None
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        chunk_type_bytes = data[pos + 4 : pos + 8]
        data_start = pos + _PNG_CHUNK_HEADER_SIZE
        data_end = data_start + length
        crc_end = data_end + _PNG_CHUNK_CRC_SIZE
        if data_end > total or crc_end > total:
            return None
        try:
            chunk_type = chunk_type_bytes.decode("ascii")
        except UnicodeDecodeError:
            return None
        chunk_data = data[data_start:data_end]
        (declared_crc,) = struct.unpack(">I", data[data_end:crc_end])
        computed_crc = zlib.crc32(chunk_type_bytes + chunk_data) & 0xFFFFFFFF
        if declared_crc != computed_crc:
            return None
        chunks.append((chunk_type, chunk_data))
        pos = crc_end
    if pos != total:
        return None
    return chunks


def _validate_preview_png(
    data: bytes, *, max_width: int, max_height: int, max_pixels: int
) -> tuple[int, int] | None:
    """Validates the ENTIRE PNG container structure -- never just a
    signature-plus-IHDR check, which a 24-byte fake payload can pass while
    containing no valid chunks or image data at all (second independent
    review, Finding 1). This is trusted-boundary validation of a
    worker-controlled byte string before it is ever handed to a UI image
    renderer, so it must reject anything that isn't exactly the narrow
    IHDR -> IDAT+ -> IEND shape our own encoder produces: wrong chunk
    order, forbidden ancillary/metadata chunks, bad CRCs, impossible or
    truncated chunk lengths, a missing IEND, or trailing bytes after it.

    Uses only ``struct``/``zlib`` -- never Pillow, PDFium, or any other
    image decoder -- so no untrusted PNG decoding ever happens in this
    (trusted, API/Streamlit-facing) parent process. Returns the validated
    ``(width, height)`` or ``None``.
    """
    chunks = _parse_png_chunks(data)
    if chunks is None or len(chunks) < 3:
        return None
    if any(chunk_type not in _ALLOWED_PNG_CHUNK_TYPES for chunk_type, _ in chunks):
        return None

    first_type, ihdr_data = chunks[0]
    if first_type != "IHDR" or len(ihdr_data) != _PNG_IHDR_LENGTH:
        return None
    width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(
        ">IIBBBBB", ihdr_data
    )
    if (
        bit_depth != _SUPPORTED_BIT_DEPTH
        or color_type != _SUPPORTED_COLOR_TYPE
        or compression != _SUPPORTED_COMPRESSION_METHOD
        or filter_method != _SUPPORTED_FILTER_METHOD
        or interlace != _SUPPORTED_INTERLACE_METHOD
    ):
        return None
    if width <= 0 or height <= 0 or width > max_width or height > max_height:
        return None
    if width * height > max_pixels:
        return None

    last_type, last_data = chunks[-1]
    if last_type != "IEND" or last_data:
        return None
    if any(chunk_type == "IEND" for chunk_type, _ in chunks[:-1]):
        return None  # exactly one IEND, and it must be the final chunk

    middle = chunks[1:-1]
    if any(chunk_type != "IDAT" for chunk_type, _ in middle):
        return None  # only IDAT chunks may appear between IHDR and IEND
    if not any(chunk_data for _, chunk_data in middle):
        return None  # at least one non-empty IDAT chunk is required

    return width, height


#: A validated ``WorkerJobError`` is preserved as its own typed outcome
#: (Finding 5) -- it is a well-formed, closed-enum report from the worker,
#: distinct from ``WorkerInvalidResponse`` (a *malformed* or schema-invalid
#: message the worker never should have sent at all).
WorkerOutcome = (
    WorkerJobResponse
    | WorkerJobError
    | WorkerTimeout
    | WorkerCrashed
    | WorkerInvalidResponse
    | WorkerBusy
)

# Re-exported for backward-compatible import sites; the authoritative values
# now live in plateproof.documents.limits (Finding 6: centralized, absolute
# ceilings shared with Settings validation).
MAX_POOL_SIZE = ABSOLUTE_MAX_POOL_SIZE
MAX_PAGE_TIMEOUT_SECONDS = ABSOLUTE_MAX_PAGE_TIMEOUT_SECONDS
MAX_TOTAL_TIMEOUT_SECONDS = ABSOLUTE_MAX_TOTAL_TIMEOUT_SECONDS
MAX_ADMISSION_TIMEOUT_SECONDS = ABSOLUTE_MAX_ADMISSION_TIMEOUT_SECONDS
KILL_GRACE_SECONDS = DEFAULT_KILL_GRACE_SECONDS


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


@dataclass(frozen=True, kw_only=True)
class WorkerPoolConfig:
    """Validated eagerly at construction time -- an invalid config can never
    be used to construct a :class:`WorkerPool` at all, so a zero/negative
    pool size can never leave a caller blocked forever waiting for a free
    slot (Finding 6)."""

    pool_size: int = DEFAULT_POOL_SIZE
    page_timeout_seconds: float = DEFAULT_PAGE_TIMEOUT_SECONDS
    total_timeout_seconds: float = DEFAULT_TOTAL_TIMEOUT_SECONDS
    kill_grace_seconds: float = DEFAULT_KILL_GRACE_SECONDS
    #: Bounded wait for a free slot in ``submit()`` (Finding 1) -- separate
    #: from the job deadlines above, since admission ("wait for a slot to
    #: exist") and job execution ("wait for a job to finish") are different
    #: things that must not share one budget.
    admission_timeout_seconds: float = DEFAULT_ADMISSION_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        pool_size = _exact_int(self.pool_size)
        if pool_size is None or not (MIN_POOL_SIZE <= pool_size <= ABSOLUTE_MAX_POOL_SIZE):
            raise ValueError(
                f"pool_size must be an integer in [{MIN_POOL_SIZE}, {ABSOLUTE_MAX_POOL_SIZE}]"
            )

        page_timeout = _exact_finite_number(self.page_timeout_seconds)
        if page_timeout is None or not (0 < page_timeout <= ABSOLUTE_MAX_PAGE_TIMEOUT_SECONDS):
            raise ValueError(
                "page_timeout_seconds must be a finite number in "
                f"(0, {ABSOLUTE_MAX_PAGE_TIMEOUT_SECONDS}]"
            )

        total_timeout = _exact_finite_number(self.total_timeout_seconds)
        if total_timeout is None or not (0 < total_timeout <= ABSOLUTE_MAX_TOTAL_TIMEOUT_SECONDS):
            raise ValueError(
                "total_timeout_seconds must be a finite number in "
                f"(0, {ABSOLUTE_MAX_TOTAL_TIMEOUT_SECONDS}]"
            )
        if total_timeout < page_timeout:
            raise ValueError("total_timeout_seconds must be at least page_timeout_seconds")

        kill_grace = _exact_finite_number(self.kill_grace_seconds)
        if kill_grace is None or not (
            MIN_KILL_GRACE_SECONDS <= kill_grace <= ABSOLUTE_MAX_KILL_GRACE_SECONDS
        ):
            raise ValueError(
                f"kill_grace_seconds must be a finite number in "
                f"[{MIN_KILL_GRACE_SECONDS}, {ABSOLUTE_MAX_KILL_GRACE_SECONDS}]"
            )

        admission_timeout = _exact_finite_number(self.admission_timeout_seconds)
        if admission_timeout is None or not (
            MIN_ADMISSION_TIMEOUT_SECONDS
            <= admission_timeout
            <= ABSOLUTE_MAX_ADMISSION_TIMEOUT_SECONDS
        ):
            raise ValueError(
                "admission_timeout_seconds must be a finite number in "
                f"[{MIN_ADMISSION_TIMEOUT_SECONDS}, {ABSOLUTE_MAX_ADMISSION_TIMEOUT_SECONDS}]"
            )


@dataclass
class _WorkerHandle:
    process: Any
    # multiprocessing's typeshed stubs return a distinct PipeConnection type
    # from Pipe() that doesn't structurally unify with Connection in strict
    # mode; both expose the same send_bytes/recv_bytes/poll/close API this
    # module actually uses, so Any is accepted here deliberately.
    conn: Any


def _default_worker_main(
    conn: Connection,
) -> None:  # pragma: no cover - overridden by entrypoint wiring
    from plateproof.documents.worker.entrypoint import worker_main

    worker_main(conn)


class WorkerPool:
    """Owns up to ``config.pool_size`` long-lived worker processes. Each
    ``submit()`` call blocks until a slot is free (bounding concurrent
    process count), lazily spawns a replacement worker for that slot if
    needed, and retires the worker on any non-success outcome."""

    def __init__(
        self,
        *,
        config: WorkerPoolConfig,
        worker_main: Callable[[Connection], None] | None = None,
    ) -> None:
        self._config = config
        self._worker_main = worker_main or _default_worker_main
        self._ctx: SpawnContext = multiprocessing.get_context("spawn")
        self._lock = threading.Lock()
        self._slots: dict[int, _WorkerHandle | None] = dict.fromkeys(range(config.pool_size))
        self._free_slots: queue.Queue[int] = queue.Queue()
        for index in range(config.pool_size):
            self._free_slots.put(index)

    def submit(self, job_request: WorkerJobRequest, document_bytes: bytes) -> WorkerOutcome:
        """Blocks until a slot is free, bounded by
        ``config.admission_timeout_seconds`` (Finding 1) -- a saturated
        pool fails closed with :class:`WorkerBusy` rather than blocking
        indefinitely, which under concurrent uploads would otherwise let
        requests pile up holding threads and document bytes with no time
        bound at all (the per-page/per-document deadlines only start once
        a job is actually admitted, so they never covered this wait).

        ``queue.Queue.get(timeout=...)`` either returns a slot or raises
        ``queue.Empty``, atomically -- there is no third outcome where a
        timed-out caller is later handed a slot anyway, so a request that
        received :class:`WorkerBusy` never goes on to spawn a worker or
        send it a job.
        """
        try:
            index = self._free_slots.get(timeout=self._config.admission_timeout_seconds)
        except queue.Empty:
            return WorkerBusy()
        try:
            handle = self._ensure_worker(index)
            outcome = self._run_job(handle, job_request, document_bytes)
            if not isinstance(outcome, WorkerJobResponse):
                self._retire(index)
            return outcome
        finally:
            self._free_slots.put(index)

    def shutdown(self) -> None:
        with self._lock:
            for index, handle in list(self._slots.items()):
                if handle is not None:
                    self._kill(handle)
                    self._slots[index] = None

    def _ensure_worker(self, index: int) -> _WorkerHandle:
        with self._lock:
            handle = self._slots[index]
            if handle is None or not handle.process.is_alive():
                handle = self._spawn_worker()
                self._slots[index] = handle
            return handle

    def _spawn_worker(self) -> _WorkerHandle:
        parent_conn, child_conn = self._ctx.Pipe(duplex=True)
        with _safe_main_module_for_spawn():
            process = self._ctx.Process(target=self._worker_main, args=(child_conn,), daemon=True)
            process.start()
        # Parent hygiene: close our copy of the child's end immediately so
        # EOF is detected correctly if the child dies.
        child_conn.close()
        return _WorkerHandle(process=process, conn=parent_conn)

    def _run_job(
        self, handle: _WorkerHandle, job_request: WorkerJobRequest, document_bytes: bytes
    ) -> WorkerOutcome:
        conn = handle.conn
        try:
            send_frame(
                conn,
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "message_type": "job_request",
                    **asdict(job_request),
                },
            )
            send_bytes_frame(conn, document_bytes, max_length=ABSOLUTE_MAX_UPLOAD_BYTES)
        except (ProtocolViolationError, OSError, EOFError):
            return self._classify_failure(handle, reason="failed to send job to worker")

        deadline_total = time.monotonic() + self._config.total_timeout_seconds
        last_progress = time.monotonic()

        while True:
            now = time.monotonic()
            remaining_total = deadline_total - now
            remaining_page = (last_progress + self._config.page_timeout_seconds) - now

            if remaining_total <= 0:
                self._kill(handle)
                return WorkerTimeout(stage="total")
            if remaining_page <= 0:
                self._kill(handle)
                return WorkerTimeout(stage="page")

            wait_for = min(remaining_total, remaining_page)
            try:
                ready = conn.poll(wait_for)
            except OSError:
                return self._classify_failure(
                    handle, reason="connection error while waiting for worker"
                )
            if not ready:
                continue

            try:
                raw = recv_frame(conn)
                validated = validate_worker_message(raw)
            except ProtocolViolationError:
                return self._classify_failure(
                    handle, reason="worker sent a malformed or invalid message"
                )

            if isinstance(validated, WorkerPageProgress):
                last_progress = time.monotonic()
                continue
            if isinstance(validated, WorkerJobResponse):
                if validated.preview_count == 0:
                    return validated
                previews, failure = self._receive_previews(handle, validated, deadline_total)
                if failure is not None:
                    return failure
                assert previews is not None
                return replace(validated, previews=previews)
            # A job_error message: a well-formed, closed-enum report from
            # the worker -- preserved as its own typed outcome (Finding 5),
            # never collapsed into the generic "malformed message" outcome.
            assert isinstance(validated, WorkerJobError)
            return validated

    def _receive_previews(
        self, handle: _WorkerHandle, response: WorkerJobResponse, deadline_total: float
    ) -> tuple[tuple[ValidatedPreview, ...] | None, WorkerOutcome | None]:
        """Reads exactly ``response.preview_count`` (header, binary) pairs,
        re-validating each before ever trusting it (Finding 7): page number
        must be one of this response's own pages, strictly increasing
        (catches both duplicates and out-of-order frames), the actual byte
        count must match the declared header exactly, the cumulative byte
        budget must stay bounded, and the bytes must decode as a
        real, size-bounded PNG. Any violation retires the worker and
        returns a fail-closed outcome -- never a partial preview set."""
        conn = handle.conn
        valid_page_numbers = {page.page_number for page in response.pages}
        last_page_number = 0
        total_bytes = 0
        previews: list[ValidatedPreview] = []

        while len(previews) < response.preview_count:
            remaining_total = deadline_total - time.monotonic()
            if remaining_total <= 0:
                self._kill(handle)
                return None, WorkerTimeout(stage="total")
            try:
                ready = conn.poll(remaining_total)
            except OSError:
                return None, self._classify_failure(
                    handle, reason="connection error while waiting for a preview"
                )
            if not ready:
                continue

            try:
                raw = recv_frame(conn)
                header = validate_worker_message(raw)
            except ProtocolViolationError:
                return None, self._classify_failure(
                    handle, reason="worker sent a malformed preview header"
                )
            if not isinstance(header, WorkerPreviewHeader):
                return None, self._classify_failure(
                    handle, reason="unexpected message where a preview header was expected"
                )
            if header.page_number not in valid_page_numbers:
                return None, self._classify_failure(
                    handle, reason="preview page number is not one of this response's pages"
                )
            if header.page_number <= last_page_number:
                return None, self._classify_failure(
                    handle, reason="preview page number is duplicate or out of order"
                )
            last_page_number = header.page_number

            total_bytes += header.byte_length
            if total_bytes > ABSOLUTE_MAX_TOTAL_PREVIEW_BYTES:
                return None, self._classify_failure(
                    handle, reason="total preview byte budget exceeded"
                )

            try:
                png_bytes = recv_bytes_frame(conn, max_length=ABSOLUTE_MAX_PREVIEW_FRAME_BYTES)
            except ProtocolViolationError:
                return None, self._classify_failure(
                    handle, reason="worker sent a malformed preview binary frame"
                )
            if len(png_bytes) != header.byte_length:
                return None, self._classify_failure(
                    handle, reason="preview byte length did not match its declared header"
                )

            dimensions = _validate_preview_png(
                png_bytes,
                max_width=ABSOLUTE_MAX_PREVIEW_WIDTH_PX,
                max_height=ABSOLUTE_MAX_PREVIEW_HEIGHT_PX,
                max_pixels=ABSOLUTE_MAX_PREVIEW_PIXELS,
            )
            if dimensions is None:
                return None, self._classify_failure(
                    handle, reason="preview is not a valid, well-formed PNG"
                )
            width, height = dimensions

            previews.append(
                ValidatedPreview(
                    page_number=header.page_number,
                    png_bytes=png_bytes,
                    width_px=width,
                    height_px=height,
                )
            )

        return tuple(previews), None

    def _classify_failure(
        self, handle: _WorkerHandle, *, reason: str
    ) -> WorkerCrashed | WorkerInvalidResponse:
        # Check whether the worker had already died *on its own* before we
        # kill it ourselves -- otherwise our own terminate() would make every
        # failure look like a crash. A short join gives the OS a moment to
        # finalize the exit code if the process is in the process of exiting
        # right as we observe the connection failure (e.g. EOF racing the
        # process's own reaping).
        handle.process.join(timeout=0.5)
        already_dead = not handle.process.is_alive()
        exit_code = handle.process.exitcode
        self._kill(handle)
        if already_dead and exit_code not in (0, None):
            return WorkerCrashed(exit_code=exit_code)
        return WorkerInvalidResponse(reason=reason)

    def _kill(self, handle: _WorkerHandle) -> None:
        process = handle.process
        if process.is_alive():
            process.terminate()
            process.join(timeout=self._config.kill_grace_seconds)
            if process.is_alive():
                process.kill()
                process.join()
        try:
            handle.conn.close()
        except OSError:
            pass

    def _retire(self, index: int) -> None:
        with self._lock:
            handle = self._slots[index]
            if handle is not None:
                self._kill(handle)
                self._slots[index] = None


__all__ = [
    "DEFAULT_ADMISSION_TIMEOUT_SECONDS",
    "DEFAULT_PAGE_TIMEOUT_SECONDS",
    "DEFAULT_POOL_SIZE",
    "DEFAULT_TOTAL_TIMEOUT_SECONDS",
    "KILL_GRACE_SECONDS",
    "MAX_ADMISSION_TIMEOUT_SECONDS",
    "MAX_PAGE_TIMEOUT_SECONDS",
    "MAX_POOL_SIZE",
    "MAX_TOTAL_TIMEOUT_SECONDS",
    "MAX_WORKER_FRAME_BYTES",
    "WorkerBusy",
    "WorkerOutcome",
    "WorkerPool",
    "WorkerPoolConfig",
]
