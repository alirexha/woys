"""A crash inside `_run_loop` must take its helper threads down with it.

The loop's `finally` joins the pacat writer / watchdog / stderr threads,
but those threads only exit once `_stop_event` is set -- and only stop()
used to set it. After a crash nobody had called stop(), so the joins
timed out, the thread references were dropped, and the writer spun hot
on a `None` queue for the rest of the session. A TUI restart then added
a second writer and watchdog racing for the new session's queue.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
import types
from typing import Any

import pytest

from audio import engine

_HELPER_NAMES = ("woys-pacat-writer", "woys-pacat-watchdog", "woys-pacat-stderr")


def _helpers_alive() -> list[str]:
    return [t.name for t in threading.enumerate() if t.name in _HELPER_NAMES and t.is_alive()]


class _BrokenInputStream:
    def __init__(self, *_a: Any, **_k: Any) -> None:
        raise RuntimeError("PortAudio: device unavailable")


def _crashing_engine(monkeypatch: pytest.MonkeyPatch) -> engine.RealtimeEngine:
    fake_sd = types.ModuleType("sounddevice")
    fake_sd.InputStream = _BrokenInputStream  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    eng = engine.RealtimeEngine(engine.EngineConfig(realtime_priority=False))
    monkeypatch.setattr(
        eng,
        "_open_pacat",
        lambda: subprocess.Popen(
            ["cat"], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        ),
    )
    return eng


def test_run_loop_crash_stops_helper_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _helpers_alive() == [], "leftover helper threads from another test"
    eng = _crashing_engine(monkeypatch)
    try:
        eng._run_loop()
        assert eng.stats.crashed is True
        deadline = time.monotonic() + 2.0
        while _helpers_alive() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert eng._stop_event.is_set(), "a crashed loop must signal its helpers to exit"
        assert _helpers_alive() == []
    finally:
        eng._stop_event.set()
        deadline = time.monotonic() + 2.0
        while _helpers_alive() and time.monotonic() < deadline:
            time.sleep(0.02)


def test_writer_exits_when_its_queue_is_gone() -> None:
    """Belt and braces: a writer whose session queue was torn down exits
    instead of polling a `None` queue in a tight loop."""
    eng = engine.RealtimeEngine(engine.EngineConfig(realtime_priority=False))
    eng._writer_queue = None
    t = threading.Thread(target=eng._writer_loop, daemon=True)
    t.start()
    t.join(timeout=1.0)
    alive = t.is_alive()
    eng._stop_event.set()
    t.join(timeout=1.0)
    assert not alive, "writer kept running with no queue to drain"
