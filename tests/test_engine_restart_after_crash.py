"""start() after a run that ended without stop() must tear that run down.

A failed warmup, a crash in the chunk loop, or a self-stop all end the
worker without stop() having run. The TUI's toggle then calls start()
straight away (the engine reads as not running), and start() used to
build on top of the leftovers:

* it saved `gc.isenabled()` -- already False from the first start -- as
  the state to restore, so GC stayed off for the rest of the process;
* it saved the engine's own signal handler as the "prior" one, so the
  eventual stop() re-installed the engine handler and a later SIGTERM
  recursed until RecursionError instead of exiting;
* the previous run's inference child was never told to stop.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import gc
import signal
from collections.abc import Iterator

import pytest

from audio import engine


def _prior_handler(_signum: int, _frame: object) -> None:
    pass


@pytest.fixture
def process_state() -> Iterator[None]:
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    gc_was_enabled = gc.isenabled()
    signal.signal(signal.SIGTERM, _prior_handler)
    gc.enable()
    yield None
    for sig, handler in saved.items():
        signal.signal(sig, handler)
    if gc_was_enabled:
        gc.enable()
    else:
        gc.disable()


def _failing_engine(monkeypatch: pytest.MonkeyPatch) -> engine.RealtimeEngine:
    eng = engine.RealtimeEngine(engine.EngineConfig())

    def boom() -> None:
        raise FileNotFoundError("contentvec model not found")

    monkeypatch.setattr(eng, "_worker_preamble", boom)
    return eng


def _start_and_wait(eng: engine.RealtimeEngine) -> None:
    eng.start()
    assert eng._thread is not None
    eng._thread.join(timeout=5.0)
    assert eng.stats.crashed is True


def test_retry_after_failed_warmup_restores_gc_and_signals(
    monkeypatch: pytest.MonkeyPatch, process_state: None
) -> None:
    eng = _failing_engine(monkeypatch)
    _start_and_wait(eng)
    _start_and_wait(eng)
    eng.stop(timeout=1.0)
    assert gc.isenabled(), "GC must be back on after the final stop()"
    assert signal.getsignal(signal.SIGTERM) is _prior_handler


def test_restart_saves_the_real_prior_handler(
    monkeypatch: pytest.MonkeyPatch, process_state: None
) -> None:
    eng = _failing_engine(monkeypatch)
    _start_and_wait(eng)
    _start_and_wait(eng)
    try:
        assert eng._prior_signal_handlers.get(signal.SIGTERM) is _prior_handler
    finally:
        eng.stop(timeout=1.0)


def test_restart_after_crash_stops_the_previous_inference_child(
    monkeypatch: pytest.MonkeyPatch, process_state: None
) -> None:
    eng = engine.RealtimeEngine(engine.EngineConfig())
    stopped: list[str] = []

    class _Child:
        def __init__(self, name: str) -> None:
            self.name = name

        def stop(self, timeout_s: float = 2.0) -> None:
            stopped.append(self.name)

    runs = iter(["first", "second"])

    def preamble_spawns_child_then_fails() -> None:
        eng._inf_client = _Child(next(runs))  # type: ignore[assignment]
        raise RuntimeError("rvc probe failed")

    monkeypatch.setattr(eng, "_worker_preamble", preamble_spawns_child_then_fails)
    _start_and_wait(eng)
    assert stopped == []
    _start_and_wait(eng)
    assert stopped == ["first"], "the crashed run's child must be stopped before the restart"
    eng.stop(timeout=1.0)
    assert stopped == ["first", "second"]
