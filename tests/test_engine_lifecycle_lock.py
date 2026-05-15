"""review F-merged-018: serialize `start()` and `stop()` under a
`_lifecycle_lock`, with an idempotence guard inside `stop()`.

CX3 corrected the verdict's reachability narrative: the "two concurrent
start() from TUI autostart + socket command" race is NOT reachable
(autostart and action_toggle_engine run on the single event-loop
thread; no control command calls start()). The reachable hazards are:

1. **Concurrent / interleaved stop()** (signal-handler path + action_quit
   + CLI teardown -- see F-CX3-01). Pre-fix two callers both passed
   `self._inf_client is not None` and called `_inf_client.stop()` twice;
   `contextlib.suppress(Exception)` hid the double-teardown error.
2. **stop() racing start()'s multi-second warmup window** (SIGTERM during
   autostart). Pre-fix start() could finish populating `_inf_client`
   AFTER stop() had observed `_inf_client is None` -- so the engine
   came up "running" after the stop signal landed.

Post-fix one `_lifecycle_lock` held across the WHOLE body of both
methods serializes them; the `_stopped` flag short-circuits a redundant
`stop()` so resources are released exactly once.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

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


def test_lifecycle_lock_exists() -> None:
    """The fix introduces `_lifecycle_lock`. Future maintainers must
    not accidentally regress to checking `is_alive()` outside it."""
    from audio import engine

    eng = engine.RealtimeEngine(engine.EngineConfig())
    assert isinstance(eng._lifecycle_lock, type(threading.Lock())), (
        "RealtimeEngine must expose a `_lifecycle_lock` so start() / "
        "stop() can serialize against each other"
    )


def test_stopped_flag_starts_false_and_flips_after_first_stop() -> None:
    """`_stopped` is the idempotence guard. Constructed-not-running is
    False (so the first stop() runs cleanup -- the pre-existing
    `test_stop_releases_in_process_sessions` contract). After one
    stop() it flips True; subsequent stop() short-circuits."""
    from audio import engine

    eng = engine.RealtimeEngine(engine.EngineConfig())
    assert eng._stopped is False
    eng.stop(timeout=0.1)
    assert eng._stopped is True


def test_concurrent_stop_calls_run_inf_client_stop_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bug-class test. Two threads concurrently call `stop()` on the
    same engine. Pre-fix both passed `self._inf_client is not None` and
    both called `_inf_client.stop()` -- the second call hit a half-torn-
    down client. Post-fix the lock + `_stopped` guard make exactly one
    `_inf_client.stop()` call land."""
    from audio import engine

    eng = engine.RealtimeEngine(engine.EngineConfig())

    stop_calls: list[float] = []
    sleep_in_stop = threading.Event()

    class _SlowFakeInfClient:
        """Fake InferenceClient whose stop() sleeps long enough to let
        a second stop() arrive while the first is still inside the
        lock -- the exact race the lock prevents."""

        def stop(self, *, timeout_s: float = 2.0) -> None:
            stop_calls.append(time.monotonic())
            # Sleep INSIDE the lock so a second stop() that races
            # has time to arrive and assert against the lock.
            time.sleep(0.1)
            sleep_in_stop.set()

    eng._inf_client = _SlowFakeInfClient()  # type: ignore[assignment]
    # Suppress the rest of stop()'s work (GC restore, clock-lock revert)
    # so the test isolates _inf_client.stop().
    monkeypatch.setattr(eng._rvc_pool, "evict_all", lambda: None)
    monkeypatch.setattr(eng, "_revert_gpu_clock_lock", lambda: None)

    def _stop_call() -> None:
        eng.stop(timeout=1.0)

    t1 = threading.Thread(target=_stop_call, name="stop-A")
    t2 = threading.Thread(target=_stop_call, name="stop-B")
    t1.start()
    # Make sure t2 actually competes with t1, not just gets sequenced
    # after it finishes. Give t1 a few ms to enter stop().
    time.sleep(0.01)
    t2.start()
    t1.join(timeout=2.0)
    t2.join(timeout=2.0)

    assert not t1.is_alive() and not t2.is_alive(), (
        "both stop() callers must complete; one was deadlocked"
    )
    assert len(stop_calls) == 1, (
        f"_inf_client.stop() must run exactly once (the second caller "
        f"hits the `_stopped` short-circuit); got {len(stop_calls)} calls"
    )
    assert eng._stopped is True


def test_stop_during_start_warmup_waits_then_tears_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bug-class test (CX3's corrected scenario). A stop() arriving
    during start()'s multi-second warmup window must wait for warmup
    to finish before tearing down. Pre-fix stop() observed
    `_inf_client is None` (start() hadn't yet assigned it) and skipped
    teardown -- so start() then completed and the engine came up
    "running" after the stop signal."""
    from audio import engine

    eng = engine.RealtimeEngine(engine.EngineConfig())
    # Force the in-process path so start() runs _ensure_sessions +
    # _warmup_realtime_pipeline (which we monkey-patch to be slow).
    eng.cfg.inference_subprocess = False
    eng.cfg.eager_warmup = False

    # Phase-A: stub the heavy / unsafe pieces of start().
    monkeypatch.setattr(
        engine,
        "gc",
        type(
            "FakeGc",
            (),
            {  # type: ignore[arg-type]
                "isenabled": staticmethod(lambda: False),
                "disable": staticmethod(lambda: None),
                "enable": staticmethod(lambda: None),
                "collect": staticmethod(lambda: 0),
            },
        ),
    )
    monkeypatch.setattr(eng, "_install_signal_handlers", lambda: None)
    monkeypatch.setattr(eng, "_warn_if_default_sink_hijacked", lambda: None)
    monkeypatch.setattr(eng, "_resolve_anti_jitter_flags", lambda: (False, False))
    monkeypatch.setattr(eng, "_apply_gpu_clock_lock", lambda: None)
    monkeypatch.setattr(eng, "_revert_gpu_clock_lock", lambda: None)
    monkeypatch.setattr(eng, "_ensure_sessions", lambda: None)
    monkeypatch.setattr(eng._rvc_pool, "evict_all", lambda: None)

    # Stub _run_loop so the engine thread exits immediately when
    # _stop_event is set -- otherwise the test would hang waiting on
    # join().
    def _short_run_loop() -> None:
        eng._stop_event.wait(timeout=2.0)

    monkeypatch.setattr(eng, "_run_loop", _short_run_loop)

    # The load-bearing slow step: warmup takes 200 ms. While start() is
    # inside it, stop() can race -- but with the lock, stop() blocks
    # until start() completes (after which `_inf_client is None` is
    # still true on the in-process path, but the engine thread is
    # already running and stop() correctly joins it).
    warmup_in_flight = threading.Event()
    warmup_done = threading.Event()

    def _slow_warmup(*_a: object, **_kw: object) -> None:
        warmup_in_flight.set()
        time.sleep(0.2)
        warmup_done.set()

    monkeypatch.setattr(eng, "_warmup_realtime_pipeline", _slow_warmup)

    stop_started = threading.Event()
    stop_done = threading.Event()

    def _do_stop() -> None:
        stop_started.set()
        eng.stop(timeout=1.0)
        stop_done.set()

    start_thread = threading.Thread(target=eng.start, name="start-A")
    start_thread.start()
    assert warmup_in_flight.wait(timeout=1.0), "start() must reach the warmup step within 1 s"
    # warmup_in_flight is set -- start() is INSIDE _lifecycle_lock,
    # holding it. Now fire stop() concurrently.
    stop_thread = threading.Thread(target=_do_stop, name="stop-A")
    stop_thread.start()
    assert stop_started.wait(timeout=0.5)
    # The lock invariant: stop_done must NOT fire until warmup_done.
    # We give stop() 100 ms head-start; warmup is still 100+ ms away
    # from done, so stop_done must still be unset.
    time.sleep(0.05)
    assert not stop_done.is_set(), (
        "stop() must block on _lifecycle_lock while start()'s warmup "
        "is in flight; pre-fix stop() raced through with _inf_client "
        "still None and the engine came up running afterwards"
    )
    # Wait for both to finish.
    start_thread.join(timeout=2.0)
    stop_thread.join(timeout=2.0)
    assert warmup_done.is_set()
    assert stop_done.is_set()
    assert eng._stopped is True, "after stop() runs, _stopped must be True"

    # The single woys-engine thread must have been joined cleanly. A
    # second start()-then-immediate-stop() (lifecycle re-use) must not
    # leave a dangling thread either -- exercise idempotency.
    alive_engine_threads = [
        t for t in threading.enumerate() if t.name == "woys-engine" and t.is_alive()
    ]
    assert len(alive_engine_threads) == 0, (
        f"after stop(), no woys-engine thread must be alive; got {alive_engine_threads}"
    )


def test_redundant_stop_after_clean_stop_short_circuits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The `_stopped` guard is idempotent. The second sequential stop()
    on the same engine must not re-tear-down -- if we did, we'd be
    calling `_inf_client.stop()` on a None (or worse, a freshly-reset
    client created by an interleaving start)."""
    from audio import engine

    eng = engine.RealtimeEngine(engine.EngineConfig())
    monkeypatch.setattr(eng._rvc_pool, "evict_all", lambda: None)

    stops: list[float] = []

    class _Counter:
        def stop(self, *, timeout_s: float = 2.0) -> None:
            stops.append(time.monotonic())

    eng._inf_client = _Counter()  # type: ignore[assignment]
    eng.stop(timeout=0.1)
    eng.stop(timeout=0.1)
    eng.stop(timeout=0.1)

    assert len(stops) == 1, (
        f"_inf_client.stop() must run only on the FIRST stop() call; got {len(stops)}"
    )
