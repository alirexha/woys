"""review F-17-06 (P1): the headless path must not be blind to a
worker-thread crash.

If `_run_loop` exits via an unhandled exception, `EngineStats.crashed` must
be set (distinct from `running == False`, which a clean `stop()` also
produces). `cmd_engine`'s loop reads `eng.stats.crashed` to break early and
exit non-zero -- pre-fix it checked only `stop["now"]` / `deadline` and
printed frozen `chunks=0` stats for the full `--seconds`.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "src" / "server") not in sys.path:
    sys.path.insert(0, str(REPO / "src" / "server"))

from audio import engine  # noqa: E402


def test_engine_stats_crashed_defaults_false() -> None:
    """A fresh engine has not crashed."""
    eng = engine.RealtimeEngine(engine.EngineConfig())
    assert eng.stats.crashed is False


def test_run_loop_sets_crashed_on_unhandled_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bug-class test. An unhandled exception inside `_run_loop` must
    leave `EngineStats.crashed` True (and not propagate -- the except
    handler catches it). Pre-fix `crashed` did not exist, so accessing it
    raises AttributeError; the headless loop had no crash signal to read.
    """
    eng = engine.RealtimeEngine(engine.EngineConfig())

    def _boom() -> object:
        raise RuntimeError("pacat blew up")

    # `_open_pacat` is the first call inside `_run_loop`'s guarded block;
    # making it raise drives the engine straight into the except handler.
    monkeypatch.setattr(eng, "_open_pacat", _boom)
    assert eng.stats.crashed is False

    eng._run_loop()  # must NOT propagate -- the except handler catches it

    assert eng.stats.crashed is True, "_run_loop crash must set stats.crashed"
    assert eng.stats.running is False
    assert eng.stats.last_error and "RuntimeError" in eng.stats.last_error


def test_start_resets_crashed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A prior crash must not stick across a fresh `start()`. `start()`
    clears `crashed` before spawning the worker (verified without actually
    running the engine: stub the heavy steps)."""
    eng = engine.RealtimeEngine(engine.EngineConfig())
    eng.stats.crashed = True  # simulate a prior crashed run

    # Stub everything start() does after the `crashed = False` reset so the
    # test needs no GPU / PipeWire and never spawns the real worker thread.
    monkeypatch.setattr(eng, "_install_signal_handlers", lambda: None)
    monkeypatch.setattr(eng, "_warn_if_default_sink_hijacked", lambda: None)
    monkeypatch.setattr(eng, "_resolve_anti_jitter_flags", lambda: (False, False))
    monkeypatch.setattr(eng, "_ensure_sessions", lambda: None)
    monkeypatch.setattr(eng, "_warmup_realtime_pipeline", lambda: None)

    class _DummyThread:
        def start(self) -> None:
            pass

        def is_alive(self) -> bool:
            return False

    monkeypatch.setattr(engine.threading, "Thread", lambda *a, **k: _DummyThread())

    eng.start()
    assert eng.stats.crashed is False, "start() must clear a stale crashed flag"
