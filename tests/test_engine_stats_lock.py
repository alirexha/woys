"""commit-040a: `_stats_lock` for the
lost-update and TOCTOU bug-classes on `EngineStats`.

Pre-fix `EngineStats` was mutated lock-free across 5-6 threads. CX3
checked the "the GIL makes it fine" objection per-operation and it
fails for every one:

- `stats.<counter> += 1` is LOAD/BINARY_OP/STORE_ATTR -- the GIL
  guarantees each BYTECODE, not the triple. Textbook lost-update.
- `helper_exit_reasons.append() + len(...) > 10: pop(0)` from two
  threads (engine.py:3084-3086 and :3608-3610 -- *different threads,
  same list, same pattern*) violates the `len <= 10` invariant under
  interleaved scheduling.
- The module docstring at engine.py:33-37 claimed "no shared mutable
  state beyond a few atomic-ish primitives" -- false.

Post-fix `self._stats_lock` (a `threading.RLock`) serializes every
`+=` and every `append+len-check+pop` block. Held for microseconds
on the hot path; negligible overhead.

These tests verify the lock USAGE pattern is correct (i.e., the
expected math holds when callers acquire the lock before mutating).
They do NOT verify that every internal callsite uses the lock --
that's mechanical code review. The bug-class signal is "without the
lock, the math fails"; the post-fix signal is "with the lock, the
math holds".

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "src" / "server") not in sys.path:
    sys.path.insert(0, str(REPO / "src" / "server"))


def test_stats_lock_exists_and_is_an_rlock() -> None:
    """The fix exposes `_stats_lock` as a `threading.RLock`. Future
    re-entrant call patterns (a method holding the lock calling
    another method that takes the lock) must not deadlock, so the
    lock is recursive."""
    from audio import engine

    eng = engine.RealtimeEngine(engine.EngineConfig())
    lock = eng._stats_lock
    # threading.RLock is implemented as a factory; the type returned
    # has _is_owned and acquire/release methods. Probe behaviorally
    # rather than asserting the exact type name.
    lock.acquire()
    try:
        # Re-entrant acquire must not deadlock.
        assert lock.acquire(blocking=False) is True, (
            "_stats_lock must be re-entrant (threading.RLock)"
        )
        lock.release()
    finally:
        lock.release()


def test_stats_xruns_lost_update_class_is_protected_under_lock() -> None:
    """Bug-class A. Four threads each call `with lock: stats.xruns +=
    1` 100_000 times. With the lock in place, the final value equals
    `N * threads`. Without the lock the read-modify-write race causes
    `lost-update` -- the final value would be LESS than `N * threads`
    and tests would flake on this assertion."""
    from audio import engine

    eng = engine.RealtimeEngine(engine.EngineConfig())
    N = 100_000
    THREADS = 4

    def hammer() -> None:
        for _ in range(N):
            with eng._stats_lock:
                eng.stats.xruns += 1

    threads = [threading.Thread(target=hammer, name=f"hammer-{i}") for i in range(THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30.0)

    assert eng.stats.xruns == N * THREADS, (
        f"with _stats_lock held around every +=, {THREADS} threads "
        f"x {N} increments must yield {N * THREADS}; got {eng.stats.xruns} "
        f"(lost-update class: pre-fix this number would be smaller)"
    )


def test_helper_exit_reasons_len_invariant_holds_under_concurrent_append() -> None:
    """Bug-class B. The cap-at-10 ring (`append + if len > 10: pop(0)`)
    fires from two threads in production: the helper-stderr drain
    thread (engine.py:3084-3086) and the pacat watchdog thread
    (:3636-3638). Without serialization, two appends followed by two
    pops can leave the list at 11. With `_stats_lock`, len <= 10
    always."""
    from audio import engine

    eng = engine.RealtimeEngine(engine.EngineConfig())
    APPENDS_PER_THREAD = 500
    THREADS = 4

    def hammer(tid: int) -> None:
        for i in range(APPENDS_PER_THREAD):
            with eng._stats_lock:
                eng.stats.helper_exit_reasons.append(f"t{tid}-i{i}")
                if len(eng.stats.helper_exit_reasons) > 10:
                    eng.stats.helper_exit_reasons.pop(0)

    threads = [
        threading.Thread(target=hammer, args=(i,), name=f"helper-{i}") for i in range(THREADS)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30.0)

    assert len(eng.stats.helper_exit_reasons) == 10, (
        f"the cap-at-10 ring must hold its invariant under {THREADS} "
        f"concurrent appenders; got {len(eng.stats.helper_exit_reasons)} entries"
    )


def test_stats_lock_is_held_briefly_does_not_deadlock_with_lifecycle_lock() -> None:
    """The engine has two locks (commit-038's `_lifecycle_lock` and
    commit-040a's `_stats_lock`). They are independent and may be held
    concurrently by different threads. A thread holding `_stats_lock`
    and a thread holding `_lifecycle_lock` must NOT deadlock on each
    other.

    The lifecycle lock is acquired in start() / stop() only; the
    stats lock is acquired on every `+=`. There's no production path
    today that takes both, but verify they don't fight if someone
    ever does."""
    from audio import engine

    eng = engine.RealtimeEngine(engine.EngineConfig())

    stats_done = threading.Event()
    lifecycle_done = threading.Event()

    def stats_holder() -> None:
        with eng._stats_lock:
            # Pretend a tight critical section.
            stats_done.set()
            # Let the other thread grab _lifecycle_lock now.
            lifecycle_done.wait(timeout=1.0)

    def lifecycle_holder() -> None:
        # Wait for stats_holder to be inside _stats_lock.
        assert stats_done.wait(timeout=1.0)
        with eng._lifecycle_lock:
            lifecycle_done.set()

    t1 = threading.Thread(target=stats_holder)
    t2 = threading.Thread(target=lifecycle_holder)
    t1.start()
    t2.start()
    t1.join(timeout=3.0)
    t2.join(timeout=3.0)

    assert not t1.is_alive() and not t2.is_alive(), (
        "_stats_lock and _lifecycle_lock must not deadlock when held "
        "concurrently by different threads"
    )
    assert stats_done.is_set() and lifecycle_done.is_set()
