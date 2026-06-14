"""pre-load swap models on a
background thread so the engine worker's chunk-boundary swap hits
a warm `_rvc_pool` cache.

Pre-fix the engine worker called
`self._rvc_pool.get_or_create(target)` inside `_apply_one_swap`. On
a cache miss this costs ~600 ms (ORT session load + cuDNN tune).
During those 600 ms the engine isn't reading the mic; the writer
queue drains; the user hears a glitch.

Post-fix `request_model_swap` puts the target on BOTH `_swap_queue`
(drained by the engine worker at chunk boundary) AND
`_swap_preload_queue` (drained by `_swap_preloader_loop` on a
dedicated thread). The preloader calls `_rvc_pool.get_or_create`
itself; by the time the worker reaches its swap, the cache is
warm and the worker's `get_or_create` is a ~10 ms cache hit.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "src" / "server") not in sys.path:
    sys.path.insert(0, str(REPO / "src" / "server"))


def test_swap_preload_queue_and_thread_attributes_exist() -> None:
    """Structural pin on the new infrastructure surface."""
    from audio import engine

    eng = engine.RealtimeEngine(engine.EngineConfig())
    assert isinstance(eng._swap_preload_queue, queue.Queue)
    assert eng._swap_preload_thread is None  # spawned in _worker_preamble


def test_request_model_swap_enqueues_on_preload_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`request_model_swap` puts the target on `_swap_preload_queue`
    so the preloader thread (if running) primes the cache. Pre-fix
    no preload queue existed."""
    from audio import engine

    eng = engine.RealtimeEngine(engine.EngineConfig())
    target = Path("/tmp/test_voice.onnx")
    eng.request_model_swap(target)

    # The preload queue has the target.
    assert eng._swap_preload_queue.qsize() == 1
    assert eng._swap_preload_queue.get_nowait() == target


def test_swap_preloader_loop_primes_pool_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end. Stub `_rvc_pool.get_or_create` to record calls;
    push a target onto `_swap_preload_queue`; let the preloader
    drain; assert get_or_create was called with the target on the
    PRELOADER thread (not the test thread)."""
    from audio import engine

    eng = engine.RealtimeEngine(engine.EngineConfig())
    calls: list[tuple[Path, str]] = []

    def fake_get(path: Path) -> object:
        calls.append((path, threading.current_thread().name))
        return object()

    monkeypatch.setattr(eng._rvc_pool, "get_or_create", fake_get)

    t = threading.Thread(target=eng._swap_preloader_loop, name="test-preloader", daemon=True)
    t.start()
    try:
        target = Path("/tmp/voiceX.onnx")
        eng._swap_preload_queue.put(target)
        # Wait for the preloader to drain.
        for _ in range(50):
            if calls:
                break
            time.sleep(0.02)
        assert calls, "preloader thread must call _rvc_pool.get_or_create"
        assert calls[0][0] == target
        assert calls[0][1] == "test-preloader", (
            "the get_or_create call must run on the preloader thread, not the caller's thread"
        )
    finally:
        eng._stop_event.set()
        t.join(timeout=2.0)


def test_swap_preloader_swallows_get_or_create_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defensive: if `get_or_create` raises (missing file, ORT
    error), the preloader thread keeps running. The engine worker
    will hit the same failure at the chunk boundary and surface it
    through the existing `_SwapRequest.error` path."""
    from audio import engine

    eng = engine.RealtimeEngine(engine.EngineConfig())
    raises = [True, True, False]

    def flaky_get(path: Path) -> object:
        if raises:
            should_raise = raises.pop(0)
            if should_raise:
                raise FileNotFoundError(str(path))
        return object()

    monkeypatch.setattr(eng._rvc_pool, "get_or_create", flaky_get)

    t = threading.Thread(target=eng._swap_preloader_loop, daemon=True)
    t.start()
    try:
        # Three swaps: first two raise (preloader swallows); third succeeds.
        eng._swap_preload_queue.put(Path("/tmp/missing-a.onnx"))
        eng._swap_preload_queue.put(Path("/tmp/missing-b.onnx"))
        eng._swap_preload_queue.put(Path("/tmp/exists.onnx"))
        # Wait for all 3 to be drained.
        for _ in range(50):
            if eng._swap_preload_queue.qsize() == 0:
                break
            time.sleep(0.02)
        assert eng._swap_preload_queue.qsize() == 0, (
            "preloader must drain all 3 entries even when get_or_create raises"
        )
        # The thread is still alive (didn't die on the exception).
        assert t.is_alive()
    finally:
        eng._stop_event.set()
        t.join(timeout=2.0)


def test_preloader_exits_promptly_on_stop_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The preloader uses a 100 ms get-timeout so the loop wakes to
    re-check `_stop_event` even on an empty queue. `stop()` joins
    with a 1 s timeout; the preloader must exit within that window."""
    from audio import engine

    eng = engine.RealtimeEngine(engine.EngineConfig())
    t = threading.Thread(target=eng._swap_preloader_loop, daemon=True)
    t.start()
    time.sleep(0.05)  # let the loop reach get(timeout)
    t0 = time.monotonic()
    eng._stop_event.set()
    t.join(timeout=1.0)
    elapsed = time.monotonic() - t0

    assert not t.is_alive(), "preloader must exit on _stop_event"
    assert elapsed < 0.5, (
        f"preloader exit must be prompt (< get-timeout + epsilon); got {elapsed:.3f}s"
    )


def test_warmed_pool_makes_chunk_boundary_swap_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The load-bearing claim. With the preloader having primed the
    cache, the engine worker's `_apply_one_swap` call to
    `_rvc_pool.get_or_create` is a cache hit (fast). Pre-fix it was
    a cache miss (~600 ms on the hot path).

    We simulate by mocking `get_or_create` with a slow first call
    and a fast second call. The preloader's first call pays the
    cost; the worker's second call is fast."""
    from audio import engine

    eng = engine.RealtimeEngine(engine.EngineConfig())
    call_count = {"n": 0}

    def slow_first_then_fast(path: Path) -> object:
        call_count["n"] += 1
        if call_count["n"] == 1:
            time.sleep(0.2)  # cache-miss-slow
        # Cache-hit second call: returns immediately
        return object()

    monkeypatch.setattr(eng._rvc_pool, "get_or_create", slow_first_then_fast)

    # Spawn preloader.
    t = threading.Thread(target=eng._swap_preloader_loop, daemon=True)
    t.start()
    try:
        target = Path("/tmp/voiceY.onnx")
        eng._swap_preload_queue.put(target)
        # Wait for the preloader to do its slow call.
        time.sleep(0.3)
        assert call_count["n"] == 1, "preloader should have done one call by now"

        # Now simulate the worker's chunk-boundary call.
        t0 = time.monotonic()
        eng._rvc_pool.get_or_create(target)
        elapsed = time.monotonic() - t0
        assert elapsed < 0.05, (
            f"with the preloader having primed the cache, the worker's "
            f"call must be fast (< 50ms); got {elapsed:.3f}s"
        )
    finally:
        eng._stop_event.set()
        t.join(timeout=2.0)
