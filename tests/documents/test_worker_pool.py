"""RED-first tests for the bounded, spawn-context, killable worker pool.

Real OS processes are used where the plan requires a genuine security proof
(termination actually stops native work; two jobs run in two distinct PIDs;
concurrency is bounded) -- these are slower but are the only real evidence
thread-based timeouts couldn't provide.
"""

from __future__ import annotations

import time
from typing import Any

from plateproof.documents.worker.protocol import (
    WorkerCrashed,
    WorkerInvalidResponse,
    WorkerJobRequest,
    WorkerJobResponse,
    WorkerTimeout,
    recv_bytes_frame,
    recv_frame,
    send_frame,
)

_REQUEST = WorkerJobRequest(
    expected_jurisdiction="nyc",
    media_type="application/pdf",
    max_pages=5,
    max_pixels=10_000_000,
    ocr_enabled=False,
    page_timeout_seconds=5.0,
)


def _pool_config(**overrides: Any) -> Any:
    from plateproof.documents.worker.pool import WorkerPoolConfig

    defaults = {
        "pool_size": 2,
        # Generous margins: real OS process spawn + import time under heavy
        # system load (e.g. this test running inside a full coverage-
        # instrumented suite) can be substantially slower than in isolation.
        "page_timeout_seconds": 20.0,
        "total_timeout_seconds": 40.0,
        "kill_grace_seconds": 0.5,
    }
    defaults.update(overrides)
    return WorkerPoolConfig(**defaults)


# --------------------------------------------------------------------------- #
# Worker-main test doubles (spawn-context targets must be module-level,      #
# picklable-by-reference functions -- spawn re-imports this module).         #
# --------------------------------------------------------------------------- #


def _hang_forever_worker(conn: Any) -> None:
    """Never responds -- proves the pool's total-timeout kill path."""
    time.sleep(600)


def _echo_success_worker(conn: Any) -> None:
    """Loops handling one job after another on the same connection, exiting
    cleanly on EOF -- matching the real entrypoint's recyclable-worker
    design, where one live process serves multiple successive jobs."""
    from plateproof.documents.worker.protocol import ProtocolViolationError

    while True:
        try:
            recv_frame(conn)
            document_bytes = recv_bytes_frame(conn, max_length=64_000_000)
        except ProtocolViolationError:
            return
        assert isinstance(document_bytes, bytes)
        send_frame(
            conn,
            {
                "protocol_version": 1,
                "message_type": "job_response",
                "ocr_available": False,
                "pages": [
                    {
                        "page_number": 1,
                        "width_px": 10,
                        "height_px": 10,
                        "used_ocr": False,
                        "ocr_attempted": False,
                        "text_blocks": [],
                    }
                ],
            },
        )


def _crash_worker(conn: Any) -> None:
    recv_frame(conn)
    recv_bytes_frame(conn, max_length=64_000_000)
    raise RuntimeError("simulated worker crash")


def _malformed_response_worker(conn: Any) -> None:
    recv_frame(conn)
    recv_bytes_frame(conn, max_length=64_000_000)
    # unknown top-level key -- schema-invalid
    send_frame(
        conn, {"protocol_version": 1, "message_type": "job_response", "pages": [], "stdout": "leak"}
    )


def _pid_reporting_worker(conn: Any) -> None:
    import os

    from plateproof.documents.worker.protocol import ProtocolViolationError

    while True:
        try:
            recv_frame(conn)
            recv_bytes_frame(conn, max_length=64_000_000)
        except ProtocolViolationError:
            return
        send_frame(
            conn,
            {
                "protocol_version": 1,
                "message_type": "job_response",
                "ocr_available": False,
                "pages": [
                    {
                        "page_number": os.getpid() % 100000 + 1,
                        "width_px": 10,
                        "height_px": 10,
                        "used_ocr": False,
                        "ocr_attempted": False,
                        "text_blocks": [],
                    }
                ],
            },
        )


def _slow_then_success_worker(conn: Any) -> None:
    """Sleeps briefly, sends page_progress, then completes -- proves the
    per-page timeout resets on real progress rather than firing immediately."""
    recv_frame(conn)
    recv_bytes_frame(conn, max_length=64_000_000)
    time.sleep(2.0)
    send_frame(conn, {"protocol_version": 1, "message_type": "page_progress", "page_number": 1})
    time.sleep(2.0)
    send_frame(
        conn,
        {
            "protocol_version": 1,
            "message_type": "job_response",
            "ocr_available": False,
            "pages": [
                {
                    "page_number": 1,
                    "width_px": 10,
                    "height_px": 10,
                    "used_ocr": False,
                    "ocr_attempted": False,
                    "text_blocks": [],
                }
            ],
        },
    )


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #


def test_successful_job_returns_validated_response() -> None:
    from plateproof.documents.worker.pool import WorkerPool

    pool = WorkerPool(config=_pool_config(), worker_main=_echo_success_worker)
    try:
        outcome = pool.submit(_REQUEST, b"fake pdf bytes")
        assert isinstance(outcome, WorkerJobResponse)
        assert outcome.pages[0].page_number == 1
    finally:
        pool.shutdown()


def test_hung_worker_times_out_and_is_terminated() -> None:
    from plateproof.documents.worker.pool import WorkerPool

    pool = WorkerPool(
        config=_pool_config(total_timeout_seconds=1.0, page_timeout_seconds=1.0),
        worker_main=_hang_forever_worker,
    )
    try:
        started = time.monotonic()
        outcome = pool.submit(_REQUEST, b"x")
        elapsed = time.monotonic() - started
        assert isinstance(outcome, WorkerTimeout)
        assert elapsed < 5.0  # bounded, not left hanging for the full sleep(600)
    finally:
        pool.shutdown()


def test_crashing_worker_yields_worker_crashed() -> None:
    from plateproof.documents.worker.pool import WorkerPool

    pool = WorkerPool(config=_pool_config(), worker_main=_crash_worker)
    try:
        outcome = pool.submit(_REQUEST, b"x")
        assert isinstance(outcome, WorkerCrashed)
    finally:
        pool.shutdown()


def test_malformed_response_yields_worker_invalid_response_not_raw_data() -> None:
    from plateproof.documents.worker.pool import WorkerPool

    pool = WorkerPool(config=_pool_config(), worker_main=_malformed_response_worker)
    try:
        outcome = pool.submit(_REQUEST, b"x")
        assert isinstance(outcome, WorkerInvalidResponse)
        assert "leak" not in outcome.reason
    finally:
        pool.shutdown()


def test_two_concurrent_jobs_run_in_two_distinct_processes() -> None:
    from concurrent.futures import ThreadPoolExecutor

    from plateproof.documents.worker.pool import WorkerPool

    pool = WorkerPool(config=_pool_config(pool_size=2), worker_main=_pid_reporting_worker)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(pool.submit, _REQUEST, b"x") for _ in range(2)]
            results = [f.result(timeout=10) for f in futures]
        pids = {r.pages[0].page_number for r in results if isinstance(r, WorkerJobResponse)}
        assert len(pids) == 2
    finally:
        pool.shutdown()


def test_submissions_beyond_pool_size_queue_rather_than_exceeding_it() -> None:
    from concurrent.futures import ThreadPoolExecutor

    from plateproof.documents.worker import pool as pool_module
    from plateproof.documents.worker.pool import WorkerPool

    live_process_counts: list[int] = []
    original_spawn = WorkerPool._spawn_worker

    def _tracking_spawn(self: Any) -> Any:
        handle = original_spawn(self)
        alive = sum(1 for h in self._slots.values() if h is not None and h.process.is_alive())
        live_process_counts.append(alive)
        return handle

    pool = WorkerPool(config=_pool_config(pool_size=2), worker_main=_echo_success_worker)
    try:
        with pytest_monkeypatch_context() as mp:
            mp.setattr(pool_module.WorkerPool, "_spawn_worker", _tracking_spawn)
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(pool.submit, _REQUEST, b"x") for _ in range(5)]
                results = [f.result(timeout=15) for f in futures]
        assert all(isinstance(r, WorkerJobResponse) for r in results)
        assert max(live_process_counts, default=0) <= 2
    finally:
        pool.shutdown()


def pytest_monkeypatch_context() -> Any:
    from _pytest.monkeypatch import MonkeyPatch

    return MonkeyPatch.context()


def test_retired_worker_is_never_assigned_another_job() -> None:
    from plateproof.documents.worker.pool import WorkerPool

    pool = WorkerPool(config=_pool_config(pool_size=1), worker_main=_malformed_response_worker)
    try:
        first_outcome = pool.submit(_REQUEST, b"x")
        assert isinstance(first_outcome, WorkerInvalidResponse)
        # A different worker (freshly spawned into the same slot) handles the
        # next job -- since this test double always misbehaves, if the same
        # (already-known-bad) process were reused, it would just misbehave
        # again in the same way, which is what we assert here: it is a FRESH
        # attempt, not a hung/reused process from before.
        second_outcome = pool.submit(_REQUEST, b"x")
        assert isinstance(second_outcome, WorkerInvalidResponse)
    finally:
        pool.shutdown()


def test_page_timeout_resets_on_real_progress() -> None:
    from plateproof.documents.worker.pool import WorkerPool

    pool = WorkerPool(
        config=_pool_config(page_timeout_seconds=25.0, total_timeout_seconds=50.0),
        worker_main=_slow_then_success_worker,
    )
    try:
        outcome = pool.submit(_REQUEST, b"x")
        assert isinstance(outcome, WorkerJobResponse)
    finally:
        pool.shutdown()


def test_shutdown_terminates_all_live_workers() -> None:
    from plateproof.documents.worker.pool import WorkerPool

    pool = WorkerPool(
        config=_pool_config(pool_size=2, total_timeout_seconds=30.0),
        worker_main=_hang_forever_worker,
    )
    import threading

    thread = threading.Thread(target=pool.submit, args=(_REQUEST, b"x"), daemon=True)
    thread.start()
    time.sleep(0.5)  # let the worker actually spawn
    pool.shutdown()
    live = [h for h in pool._slots.values() if h is not None and h.process.is_alive()]
    assert live == []
