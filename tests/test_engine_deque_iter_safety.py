"""commit-040c: cross-thread deque iteration
safety.

Pre-fix the ~11 `_recent_X` rolling-window deques and
`_writer_intervals_ms` on `EngineStats` were appended-to from one
thread (engine worker / writer / GPU keepalive) and iterated from
ANOTHER thread (TUI poll / `woys diag`) via the
`recent_X_samples_ms()` / `writer_interval_samples_ms()` accessors.

`np.array(deque)` and `list(deque)` iterate, and Python's deque
implementation raises `RuntimeError: deque mutated during iteration`
if the underlying deque gains/loses items mid-iter. The reader
thread silently dies on that exception; in the writer-jitter probe
at `engine.py:3125` (pre-fix), the dying thread was the WRITER
itself -- which means audio stops silently. The most serious of the
three F-merged-017 sub-bugs (the review ranks it ahead of the lost-
update class because the user only learns "audio stopped" via their
ears).

Post-fix every append and every iteration goes through the shared
`stats._internal_lock` (= `engine._stats_lock`). These tests pound
on the bug-class directly: one thread appends; another iterates;
no `RuntimeError` may surface.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "src" / "server") not in sys.path:
    sys.path.insert(0, str(REPO / "src" / "server"))


def test_stats_internal_lock_is_aliased_to_engine_stats_lock() -> None:
    """The lock lives on `EngineStats._internal_lock`; the engine
    aliases it as `_stats_lock` so 040a's contract (callers using
    `engine._stats_lock`) keeps working with the same primitive."""
    from audio import engine

    eng = engine.RealtimeEngine(engine.EngineConfig())
    assert eng._stats_lock is eng.stats._internal_lock, (
        "engine._stats_lock must BE stats._internal_lock (same object) "
        "so 040a's locking pattern and 040c's iteration safety share "
        "a single primitive"
    )


def test_writer_interval_snapshot_under_concurrent_append_does_not_raise() -> None:
    """Bug-class test for engine.py:3125 (pre-fix `np.array(deque)`).
    One thread appends to `_writer_intervals_ms` like the writer
    does; another thread iterates like the TUI poll does. Pre-fix
    the iterator would eventually see a mid-mutation deque and raise
    `RuntimeError: deque mutated during iteration`, killing the
    reader thread. Post-fix both sides serialize under
    `_internal_lock` so the reader never observes a torn deque."""
    from audio import engine

    eng = engine.RealtimeEngine(engine.EngineConfig())
    stop = threading.Event()
    errors: list[BaseException] = []

    def appender() -> None:
        while not stop.is_set():
            with eng._stats_lock:
                eng.stats._writer_intervals_ms.append(time.monotonic() * 1000.0)
            # Tight loop -- writer interval can be hundreds of µs.

    def reader() -> None:
        try:
            for _ in range(500):
                snap = eng.stats.writer_interval_samples_ms()
                # Force iteration (the np.array site at
                # engine.py:3125 also iterates).
                _ = list(snap)
        except BaseException as e:
            errors.append(e)

    t_app = threading.Thread(target=appender, name="writer-mimic")
    t_read = threading.Thread(target=reader, name="diag-mimic")
    t_app.start()
    t_read.start()
    t_read.join(timeout=10.0)
    stop.set()
    t_app.join(timeout=2.0)

    assert not errors, (
        f"cross-thread iteration must not raise; got {len(errors)} exception(s): {errors[:3]!r}"
    )
    # Sanity: a non-trivial number of writes happened.
    assert len(eng.stats._writer_intervals_ms) > 0


def test_recent_inference_snapshot_under_concurrent_append_does_not_raise() -> None:
    """Same shape as the writer test but on `_recent_inference` --
    the deque appended-to from the engine worker thread and iterated
    from TUI for inference percentiles."""
    from audio import engine

    eng = engine.RealtimeEngine(engine.EngineConfig())
    stop = threading.Event()
    errors: list[BaseException] = []

    def appender() -> None:
        while not stop.is_set():
            with eng._stats_lock:
                eng.stats._recent_inference.append(42.0)

    def reader() -> None:
        try:
            for _ in range(500):
                snap = eng.stats.inference_samples()
                _ = sum(snap)  # forces iteration
        except BaseException as e:
            errors.append(e)

    t_app = threading.Thread(target=appender, name="engine-mimic")
    t_read = threading.Thread(target=reader, name="tui-mimic")
    t_app.start()
    t_read.start()
    t_read.join(timeout=10.0)
    stop.set()
    t_app.join(timeout=2.0)

    assert not errors


def test_eleven_snapshot_methods_use_internal_lock() -> None:
    """Structural pin: every `_recent_X` / `_writer_intervals_ms`
    rolling-window accessor must hold `_internal_lock`. A future
    refactor that adds a new accessor without the lock would
    silently reintroduce the bug-class for that field."""
    src = Path(__file__).resolve().parent.parent / "src" / "audio" / "engine.py"
    text = src.read_text()

    expected_methods = [
        "inference_samples",
        "total_samples",
        "mic_read_samples_ms",
        "enqueue_lag_samples_ms",
        "cv_samples_ms",
        "rmvpe_samples_ms",
        "rvc_samples_ms",
        "writer_interval_samples_ms",
        "rvc_pre_samples_ms",
        "rvc_run_samples_ms",
        "rvc_post_samples_ms",
    ]
    for name in expected_methods:
        # Find the def line + the next ~8 lines and check the lock is in there.
        idx = text.find(f"def {name}(")
        assert idx > 0, f"snapshot method `{name}` should exist on EngineStats"
        body = text[idx : idx + 400]
        assert "self._internal_lock" in body, (
            f"snapshot method `{name}` must hold _internal_lock around the "
            f"`list(self._recent_X)` iteration -- otherwise cross-thread "
            f"append raises 'deque mutated during iteration'"
        )


def test_concurrent_appends_to_distinct_deques_dont_block_each_other() -> None:
    """Sanity: the lock is one object (commit-040a's _stats_lock IS
    commit-040c's _internal_lock) and serializes everything. Two
    threads appending to DIFFERENT deques still serialize, which is
    fine (the lock is held for microseconds), but verify there is
    no deadlock or extreme contention."""
    from audio import engine

    eng = engine.RealtimeEngine(engine.EngineConfig())
    N = 5000

    def push_inference() -> None:
        for i in range(N):
            with eng._stats_lock:
                eng.stats._recent_inference.append(float(i))

    def push_writer() -> None:
        for i in range(N):
            with eng._stats_lock:
                eng.stats._writer_intervals_ms.append(float(i))

    t1 = threading.Thread(target=push_inference)
    t2 = threading.Thread(target=push_writer)
    t0 = time.monotonic()
    t1.start()
    t2.start()
    t1.join(timeout=10.0)
    t2.join(timeout=10.0)
    elapsed = time.monotonic() - t0

    assert not t1.is_alive() and not t2.is_alive()
    # 10_000 lock-acquire-release pairs should complete in well under
    # a second on any sane machine. If it takes longer, the lock has
    # excessive contention or the test machine is under heavy load.
    assert elapsed < 5.0, (
        f"two threads x 5000 lock-acquire-release pairs should be fast; took {elapsed:.2f}s"
    )
