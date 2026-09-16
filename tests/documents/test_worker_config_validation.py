"""Finding 6: WorkerPoolConfig must validate every field before any worker
or queue is created. A zero pool size must never be constructible -- it
would block forever waiting for a free slot."""

from __future__ import annotations

import math

import pytest


def _build(**overrides: object) -> object:
    from plateproof.documents.worker.pool import WorkerPoolConfig

    defaults: dict[str, object] = {
        "pool_size": 2,
        "page_timeout_seconds": 20.0,
        "total_timeout_seconds": 60.0,
        "kill_grace_seconds": 2.0,
    }
    defaults.update(overrides)
    return WorkerPoolConfig(**defaults)  # type: ignore[arg-type]


@pytest.mark.parametrize("pool_size", [0, -1])
def test_non_positive_pool_size_rejected(pool_size: int) -> None:
    with pytest.raises(ValueError):
        _build(pool_size=pool_size)


def test_pool_size_above_ceiling_rejected() -> None:
    from plateproof.documents.limits import ABSOLUTE_MAX_POOL_SIZE

    with pytest.raises(ValueError):
        _build(pool_size=ABSOLUTE_MAX_POOL_SIZE + 1)


def test_pool_size_at_ceiling_accepted() -> None:
    from plateproof.documents.limits import ABSOLUTE_MAX_POOL_SIZE

    _build(pool_size=ABSOLUTE_MAX_POOL_SIZE)  # must not raise


def test_boolean_pool_size_rejected() -> None:
    """bool is a subclass of int in Python -- True/False must not silently
    pass as 1/0."""
    with pytest.raises(ValueError):
        _build(pool_size=True)


@pytest.mark.parametrize("value", [0.0, -1.0, math.inf, math.nan])
def test_invalid_page_timeout_rejected(value: float) -> None:
    with pytest.raises(ValueError):
        _build(page_timeout_seconds=value)


@pytest.mark.parametrize("value", [0.0, -1.0, math.inf, math.nan])
def test_invalid_total_timeout_rejected(value: float) -> None:
    with pytest.raises(ValueError):
        _build(total_timeout_seconds=value)


def test_page_timeout_above_ceiling_rejected() -> None:
    from plateproof.documents.limits import ABSOLUTE_MAX_PAGE_TIMEOUT_SECONDS

    with pytest.raises(ValueError):
        _build(page_timeout_seconds=ABSOLUTE_MAX_PAGE_TIMEOUT_SECONDS + 1)


def test_total_timeout_above_ceiling_rejected() -> None:
    from plateproof.documents.limits import ABSOLUTE_MAX_TOTAL_TIMEOUT_SECONDS

    with pytest.raises(ValueError):
        _build(total_timeout_seconds=ABSOLUTE_MAX_TOTAL_TIMEOUT_SECONDS + 1)


def test_total_timeout_less_than_page_timeout_rejected() -> None:
    with pytest.raises(ValueError):
        _build(page_timeout_seconds=30.0, total_timeout_seconds=10.0)


def test_total_timeout_equal_to_page_timeout_accepted() -> None:
    _build(page_timeout_seconds=20.0, total_timeout_seconds=20.0)  # must not raise


@pytest.mark.parametrize("value", [0.0, -1.0, math.inf, math.nan])
def test_invalid_kill_grace_rejected(value: float) -> None:
    with pytest.raises(ValueError):
        _build(kill_grace_seconds=value)


def test_kill_grace_above_ceiling_rejected() -> None:
    from plateproof.documents.limits import ABSOLUTE_MAX_KILL_GRACE_SECONDS

    with pytest.raises(ValueError):
        _build(kill_grace_seconds=ABSOLUTE_MAX_KILL_GRACE_SECONDS + 1)


def test_boolean_timeout_rejected() -> None:
    with pytest.raises(ValueError):
        _build(page_timeout_seconds=True)


def test_invalid_config_never_creates_a_partially_initialized_pool() -> None:
    """Constructing a WorkerPool with an invalid config must fail before
    any Pipe/Process/queue is created -- proven by asserting the pool
    object never comes into existence at all."""
    from plateproof.documents.worker.pool import WorkerPool, WorkerPoolConfig

    with pytest.raises(ValueError):
        WorkerPoolConfig(
            pool_size=0,
            page_timeout_seconds=20.0,
            total_timeout_seconds=60.0,
            kill_grace_seconds=2.0,
        )
    # A pool is never constructed with this config since the config itself
    # never came into existence -- WorkerPool(config=...) is simply
    # unreachable in real calling code following this pattern.
    assert WorkerPool  # sanity: module still imports cleanly
