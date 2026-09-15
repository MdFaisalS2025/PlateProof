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

import multiprocessing
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from multiprocessing.connection import Connection
from multiprocessing.context import SpawnContext
from typing import Any

from plateproof.documents.worker.protocol import (
    MAX_WORKER_FRAME_BYTES,
    PROTOCOL_VERSION,
    ProtocolViolationError,
    WorkerCrashed,
    WorkerInvalidResponse,
    WorkerJobRequest,
    WorkerJobResponse,
    WorkerPageProgress,
    WorkerTimeout,
    recv_frame,
    send_bytes_frame,
    send_frame,
    validate_worker_message,
)

WorkerOutcome = WorkerJobResponse | WorkerTimeout | WorkerCrashed | WorkerInvalidResponse

#: Default per-application worker-pool sizing (see plan §2/§3b) -- these are
#: process-local defaults, not a machine-wide ceiling; a deployment running
#: several application processes must size against the combined total.
DEFAULT_POOL_SIZE = 2
MAX_POOL_SIZE = 4
DEFAULT_PAGE_TIMEOUT_SECONDS = 20.0
MAX_PAGE_TIMEOUT_SECONDS = 60.0
DEFAULT_TOTAL_TIMEOUT_SECONDS = 60.0
MAX_TOTAL_TIMEOUT_SECONDS = 180.0
#: Fixed constant, not configurable -- a short, bounded grace period between
#: a polite terminate() and a hard kill().
KILL_GRACE_SECONDS = 2.0


@dataclass(frozen=True, kw_only=True)
class WorkerPoolConfig:
    pool_size: int = DEFAULT_POOL_SIZE
    page_timeout_seconds: float = DEFAULT_PAGE_TIMEOUT_SECONDS
    total_timeout_seconds: float = DEFAULT_TOTAL_TIMEOUT_SECONDS
    kill_grace_seconds: float = KILL_GRACE_SECONDS


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
            send_bytes_frame(conn, document_bytes)
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
                return validated
            # A job_error message (typed, closed error_kind) -- still not a
            # trusted "response," but a well-formed report from the worker.
            return WorkerInvalidResponse(reason="worker reported a job error")

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
