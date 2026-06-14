"""bounded timestamped thread-safe error
ring on `EngineStats`.

Pre-fix `last_error: str | None` was a single clobberable string with
~28 write sites across 6 threads, no lock, no history. A real failure
cascade (`subprocess died -> sessions reloading -> pacat respawn
failed`) overwrote `last_error` repeatedly so the user saw only the
LAST symptom in `woys diag`. The codebase already knew about the
problem -- engine.py:868 calls out sessions "with only last_error
(easily clobbered) as evidence"; engine.py:3207-3208 deliberately
routes the watchdog message to a SEPARATE `helper_exit_reasons` list
"rather than clobber last_error wholesale". Acknowledged-but-
ungeneralized.

Post-fix: a `deque(maxlen=20)` of `(monotonic_ts, thread_name, msg)`
on `EngineStats.error_history`, written by `RealtimeEngine.record_
error()` under `_stats_lock`. `last_error` stays as a
back-compat mirror of the newest entry. `recent_errors(n)` returns
the latest N entries for `woys diag` / TUI consumption.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import sys
import threading
import time
from itertools import pairwise
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "src" / "server") not in sys.path:
    sys.path.insert(0, str(REPO / "src" / "server"))


def test_record_error_appends_to_history_and_mirrors_last_error() -> None:
    """record_error() pushes a timestamped entry onto error_history
    AND sets last_error for back-compat readers (cli.py: s.last_error
    is read in 6 places)."""
    from audio import engine

    eng = engine.RealtimeEngine(engine.EngineConfig())
    assert eng.stats.last_error is None
    assert len(eng.stats.error_history) == 0

    eng.record_error("first failure")
    assert eng.stats.last_error == "first failure"
    assert len(eng.stats.error_history) == 1
    ts, thread_name, msg = eng.stats.error_history[0]
    assert isinstance(ts, float)
    assert thread_name == threading.current_thread().name
    assert msg == "first failure"


def test_error_history_caps_at_20_entries() -> None:
    """deque(maxlen=20) -- a long-running session with many failures
    must NOT grow without bound. Pre-fix `helper_exit_reasons` did this
    by hand with `if len > 10: pop(0)`; deque(maxlen) is the cleaner
    primitive."""
    from audio import engine

    eng = engine.RealtimeEngine(engine.EngineConfig())
    for i in range(50):
        eng.record_error(f"failure-{i}")
    assert len(eng.stats.error_history) == 20
    # The OLDEST 30 must have been evicted; entry 0 should be #30.
    msgs = [e[2] for e in eng.stats.error_history]
    assert msgs[0] == "failure-30"
    assert msgs[-1] == "failure-49"
    # Back-compat last_error mirrors the newest.
    assert eng.stats.last_error == "failure-49"


def test_recent_errors_returns_latest_n_entries() -> None:
    """recent_errors(n) is the documented consumer surface for
    `woys diag` / TUI status views. Returns up to n entries, oldest
    first (within the n-tail window)."""
    from audio import engine

    eng = engine.RealtimeEngine(engine.EngineConfig())
    for i in range(7):
        eng.record_error(f"failure-{i}")

    recent3 = eng.recent_errors(3)
    assert len(recent3) == 3
    assert [e[2] for e in recent3] == ["failure-4", "failure-5", "failure-6"]
    # Asking for more than the history has just returns everything.
    recent20 = eng.recent_errors(20)
    assert len(recent20) == 7
    assert [e[2] for e in recent20] == [f"failure-{i}" for i in range(7)]


def test_record_error_from_two_threads_within_1ms_retains_both() -> None:
    """Bug-class test (the review's stated test). Two threads push
    errors from each other within a millisecond. Pre-fix `last_error`
    is a single string -- thread B's write OVERWRITES thread A's, so
    the user diagnosing a cascade sees only one symptom. Post-fix the
    ring retains BOTH entries with their own timestamps and thread
    names."""
    from audio import engine

    eng = engine.RealtimeEngine(engine.EngineConfig())
    barrier = threading.Barrier(2)
    msgs_pushed: list[tuple[str, str]] = []
    lock = threading.Lock()

    def pusher(label: str) -> None:
        barrier.wait()
        eng.record_error(f"err from {label}")
        with lock:
            msgs_pushed.append((label, threading.current_thread().name))

    t1 = threading.Thread(target=pusher, args=("A",), name="cascade-A")
    t2 = threading.Thread(target=pusher, args=("B",), name="cascade-B")
    t1.start()
    t2.start()
    t1.join(timeout=2.0)
    t2.join(timeout=2.0)

    history_msgs = [e[2] for e in eng.stats.error_history]
    assert "err from A" in history_msgs, (
        "F-merged-015 bug-class: thread A's error must be retained "
        "even though thread B raced to push one too"
    )
    assert "err from B" in history_msgs, "same for thread B"
    assert len(eng.stats.error_history) == 2

    # Each entry's thread_name field reflects its actual writer thread.
    thread_names = {e[1] for e in eng.stats.error_history}
    assert thread_names == {"cascade-A", "cascade-B"}


def test_record_error_serializes_under_stats_lock() -> None:
    """The ring is mutated under `_stats_lock` so concurrent writers
    don't corrupt the deque. Hammer with many threads, many pushes;
    the final length matches what we expect AND every entry parses
    cleanly."""
    from audio import engine

    eng = engine.RealtimeEngine(engine.EngineConfig())
    THREADS = 4
    PUSHES_PER_THREAD = 100

    def hammer(tid: int) -> None:
        for i in range(PUSHES_PER_THREAD):
            eng.record_error(f"tid={tid}-i={i}")

    threads = [threading.Thread(target=hammer, args=(i,)) for i in range(THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)

    # Total writes = THREADS * PUSHES_PER_THREAD = 400, but deque
    # caps at 20 so only the last 20 survive.
    assert len(eng.stats.error_history) == 20
    for entry in eng.stats.error_history:
        ts, thread_name, msg = entry
        assert isinstance(ts, float)
        assert isinstance(thread_name, str)
        assert msg.startswith("tid=")


def test_timestamps_are_monotonic_and_non_decreasing() -> None:
    """Each entry's timestamp comes from `time.monotonic()` so it's
    monotonic by construction. Important when reconstructing a
    cascade -- the user should be able to order errors in time."""
    from audio import engine

    eng = engine.RealtimeEngine(engine.EngineConfig())
    for i in range(5):
        eng.record_error(f"err {i}")
        time.sleep(0.001)  # 1 ms between writes

    timestamps = [e[0] for e in eng.stats.error_history]
    for prev, curr in pairwise(timestamps):
        assert curr >= prev, "monotonic timestamps must be non-decreasing"
