"""The engine's own safety stops must read as a dead engine.

Three sites stop the engine from inside: the inference circuit breaker
(50 consecutive dropped chunks), the playback-helper respawn cap, and a
failed subprocess model swap. They used to set only `_stop_event`, so the
worker exited while `stats.running` stayed True and `stats.crashed` stayed
False: the TUI kept showing RUNNING, `woys engine --seconds 0` printed
frozen stats forever, a finite `woys engine` run exited 0, and a swap
queued afterwards was never resolved.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import gc
import signal
import subprocess
import sys
import threading
import time
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from audio import engine


@pytest.fixture
def process_state() -> Iterator[None]:
    prior = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    gc_was_enabled = gc.isenabled()
    yield None
    for sig, handler in prior.items():
        signal.signal(sig, handler)
    if gc_was_enabled:
        gc.enable()


class _LoudInputStream:
    def __init__(self, *_a: Any, channels: int = 1, **_k: Any) -> None:
        self._channels = channels

    def __enter__(self) -> _LoudInputStream:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def read(self, n: int) -> tuple[np.ndarray, bool]:  # type: ignore[type-arg]
        time.sleep(0.001)
        return np.full((n, self._channels), 0.3, dtype=np.float32), False


def test_circuit_breaker_marks_engine_crashed() -> None:
    eng = engine.RealtimeEngine(engine.EngineConfig())

    def boom(_audio: object) -> object:
        raise ValueError("cuda oom (simulated)")

    eng._process_streaming_16k = boom  # type: ignore[method-assign,assignment]
    for _ in range(50):
        eng._safe_process_streaming_16k(np.zeros(160, dtype=np.float32))
    assert eng._stop_event.is_set()
    assert eng.stats.crashed is True


def test_failed_subprocess_swap_marks_engine_crashed() -> None:
    from audio.inference_client import InferenceError

    eng = engine.RealtimeEngine(engine.EngineConfig())

    class _DeadChild:
        def swap_model(self, _target: Path) -> tuple[int, bool]:
            raise InferenceError("child died")

    eng._inf_client = _DeadChild()  # type: ignore[assignment]
    req = eng.request_model_swap(Path("/models/x.onnx"))
    eng._maybe_swap_model()
    eng._inf_client = None
    assert req.completion.is_set() and req.error is not None
    assert eng._stop_event.is_set()
    assert eng.stats.crashed is True


def test_respawn_cap_marks_engine_crashed(monkeypatch: pytest.MonkeyPatch) -> None:
    eng = engine.RealtimeEngine(engine.EngineConfig())

    class _DeadProc:
        returncode = 1

        def poll(self) -> int:
            return 1

    eng._pacat_proc = _DeadProc()  # type: ignore[assignment]

    def always_fails() -> object:
        raise RuntimeError("helper permanently broken")

    monkeypatch.setattr(eng, "_open_pacat", always_fails)
    monkeypatch.setattr(eng._pacat_dead_event, "wait", lambda timeout=None: None)
    monkeypatch.setattr(engine.time, "sleep", lambda *_a: None)
    t = threading.Thread(target=eng._watchdog_loop, daemon=True)
    t.start()
    t.join(timeout=5.0)
    eng._stop_event.set()
    t.join(timeout=2.0)
    assert eng.stats.crashed is True


def test_self_stopped_engine_is_not_left_running(
    monkeypatch: pytest.MonkeyPatch, process_state: None
) -> None:
    """End to end through start(): the circuit breaker trips, the worker
    exits, and the engine must then read as crashed and not running, and
    swaps queued after it must resolve instead of parking."""
    fake_sd = types.ModuleType("sounddevice")
    fake_sd.InputStream = _LoudInputStream  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    eng = engine.RealtimeEngine(
        engine.EngineConfig(
            inference_subprocess=False, input_gate_dbfs=-200.0, realtime_priority=False
        )
    )
    monkeypatch.setattr(eng, "_worker_preamble", lambda: None)
    monkeypatch.setattr(
        eng,
        "_open_pacat",
        lambda: subprocess.Popen(
            ["cat"], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        ),
    )

    def boom(_audio: object) -> object:
        raise ValueError("cuda oom (simulated)")

    monkeypatch.setattr(eng, "_process_streaming_16k", boom)
    eng.start()
    try:
        assert eng._thread is not None
        eng._thread.join(timeout=10.0)
        assert not eng._thread.is_alive()
        assert eng.stats.running is False
        assert eng.stats.crashed is True
        req = eng.request_model_swap(Path("/models/x.onnx"))
        assert req.completion.wait(0.5), "a swap queued after a self-stop must not park"
        assert req.error is not None
    finally:
        eng.stop(timeout=1.0)
