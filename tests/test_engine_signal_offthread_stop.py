"""SIGTERM must still work after the engine is stopped off the main thread.

The TUI stops the engine on a worker thread. `signal.signal()` only works
on the main thread, so the restore of the prior (Textual / CLI) handlers
silently failed -- and then the saved map was cleared anyway. The engine's
handler stayed installed with nothing to hand over to: the next SIGTERM
restored nothing, re-raised into itself and recursed until RecursionError,
and the process did not exit. A restart then saved the engine's own
handler as the "prior", with the same result.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import gc
import os
import signal
import threading
import time
from collections.abc import Iterator

import pytest

from audio import engine

_hits: list[int] = []


def _prior_handler(signum: int, _frame: object) -> None:
    _hits.append(signum)


@pytest.fixture
def process_state() -> Iterator[None]:
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    gc_was_enabled = gc.isenabled()
    signal.signal(signal.SIGTERM, _prior_handler)
    _hits.clear()
    yield None
    for sig, handler in saved.items():
        signal.signal(sig, handler)
    if gc_was_enabled:
        gc.enable()


def _engine(monkeypatch: pytest.MonkeyPatch) -> engine.RealtimeEngine:
    eng = engine.RealtimeEngine(engine.EngineConfig())
    monkeypatch.setattr(eng, "_worker_main", lambda: None)
    return eng


def _stop_off_main_thread(eng: engine.RealtimeEngine) -> None:
    t = threading.Thread(target=eng.stop, name="woys-tui-stop")
    t.start()
    t.join(timeout=5.0)


def _deliver_sigterm() -> None:
    os.kill(os.getpid(), signal.SIGTERM)
    deadline = time.monotonic() + 2.0
    while not _hits and time.monotonic() < deadline:
        time.sleep(0.01)


def test_sigterm_after_offthread_stop_reaches_the_prior_handler(
    monkeypatch: pytest.MonkeyPatch, process_state: None
) -> None:
    eng = _engine(monkeypatch)
    eng.start()
    _stop_off_main_thread(eng)
    assert eng._prior_signal_handlers.get(signal.SIGTERM) is _prior_handler, (
        "an off-thread stop must keep the saved handlers for a later restore"
    )
    _deliver_sigterm()  # pre-fix: RecursionError
    assert _hits == [signal.SIGTERM]


def test_restart_after_offthread_stop_keeps_the_real_prior(
    monkeypatch: pytest.MonkeyPatch, process_state: None
) -> None:
    eng = _engine(monkeypatch)
    eng.start()
    _stop_off_main_thread(eng)
    eng.start()
    try:
        assert eng._prior_signal_handlers.get(signal.SIGTERM) is _prior_handler
    finally:
        eng.stop(timeout=1.0)
    assert signal.getsignal(signal.SIGTERM) is _prior_handler


def test_handler_without_a_saved_prior_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch, process_state: None
) -> None:
    """Nothing to restore: the re-raise must hit SIG_DFL, not this handler."""
    eng = engine.RealtimeEngine(engine.EngineConfig())
    kills: list[int] = []
    monkeypatch.setattr(os, "kill", lambda _pid, sig: kills.append(sig))
    signal.signal(signal.SIGTERM, eng._signal_handler_revert_lock)
    eng._prior_signal_handlers = {}

    eng._signal_handler_revert_lock(signal.SIGTERM, None)

    assert kills == [signal.SIGTERM]
    assert signal.getsignal(signal.SIGTERM) == signal.SIG_DFL
