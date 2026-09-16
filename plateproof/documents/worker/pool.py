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
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from multiprocessing.connection import Connection
from multiprocessing.context import SpawnContext
from typing import Any

from plateproof.documents.limits import (
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
    DEFAULT_KILL_GRACE_SECONDS,
    DEFAULT_PAGE_TIMEOUT_SECONDS,
    DEFAULT_POOL_SIZE,
    DEFAULT_TOTAL_TIMEOUT_SECONDS,
    MIN_KILL_GRACE_SECONDS,
    MIN_POOL_SIZE,
)
from plateproof.documents.worker.protocol import (
    MAX_WORKER_FRAME_BYTES,
    PROTOCOL_VERSION,
    ProtocolViolationError,
    ValidatedPreview,
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

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _read_png_dimensions(data: bytes) -> tuple[int, int] | None:
    """Reads the width/height straight out of a PNG's IHDR chunk (bytes
    16-24, big-endian) via ``struct`` -- no Pillow/PDFium involved. This is
    trusted parsing of the worker's *own generated* small preview output,
    never the original hostile upload (Finding 7: no parent-side parser
    imports for untrusted content)."""
    if len(data) < 24 or not data.startswith(_PNG_SIGNATURE):
        return None
    try:
        width, height = struct.unpack(">II", data[16:24])
    except struct.error:
        return None
    return width, height


#: A validated ``WorkerJobError`` is preserved as its own typed outcome
#: (Finding 5) -- it is a well-formed, closed-enum report from the worker,
#: distinct from ``WorkerInvalidResponse`` (a *malformed* or schema-invalid
#: message the worker never should have sent at all).
WorkerOutcome = (
    WorkerJobResponse | WorkerJobError | WorkerTimeout | WorkerCrashed | WorkerInvalidResponse
)

# Re-exported for backward-compatible import sites; the authoritative values
# now live in plateproof.documents.limits (Finding 6: centralized, absolute
# ceilings shared with Settings validation).
MAX_POOL_SIZE = ABSOLUTE_MAX_POOL_SIZE
MAX_PAGE_TIMEOUT_SECONDS = ABSOLUTE_MAX_PAGE_TIMEOUT_SECONDS
MAX_TOTAL_TIMEOUT_SECONDS = ABSOLUTE_MAX_TOTAL_TIMEOUT_SECONDS
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
        index = self._free_slots.get()
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

            dimensions = _read_png_dimensions(png_bytes)
            if dimensions is None:
                return None, self._classify_failure(
                    handle, reason="preview is not a valid, decodable PNG"
                )
            width, height = dimensions
            if (
                width <= 0
                or height <= 0
                or width > ABSOLUTE_MAX_PREVIEW_WIDTH_PX
                or height > ABSOLUTE_MAX_PREVIEW_HEIGHT_PX
                or width * height > ABSOLUTE_MAX_PREVIEW_PIXELS
            ):
                return None, self._classify_failure(
                    handle, reason="preview dimensions exceed the maximum allowed size"
                )

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
    "DEFAULT_PAGE_TIMEOUT_SECONDS",
    "DEFAULT_POOL_SIZE",
    "DEFAULT_TOTAL_TIMEOUT_SECONDS",
    "KILL_GRACE_SECONDS",
    "MAX_PAGE_TIMEOUT_SECONDS",
    "MAX_POOL_SIZE",
    "MAX_TOTAL_TIMEOUT_SECONDS",
    "MAX_WORKER_FRAME_BYTES",
    "WorkerOutcome",
    "WorkerPool",
    "WorkerPoolConfig",
]
