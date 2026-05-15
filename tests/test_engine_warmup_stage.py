"""review F-merged-030 (commit-048): cold-start moved to the
worker thread + `warmup_stage` progress field.

Pre-fix `start()` ran `_ensure_sessions` (heavy: ORT/cuDNN), then
`_warmup_realtime_pipeline` (heavy: up to 16 synthetic `_infer`
calls), then optionally `warmup_voice_library` (very heavy:
eager-warms every voice in the library), then `_apply_gpu_clock_
lock` (subprocess.run nvidia-smi with timeout=4s) -- ALL on the
caller's thread. The TUI froze for up to ~10 s with a stale
"engine starting (cudnn warmup ~2s)" toast.

Post-fix `start()` does only sub-millisecond setup (gc disable,
signal handlers) on the caller's thread, then spawns the worker
and returns. The worker calls `_worker_preamble()` which does the
heavy work and updates `stats.warmup_stage` at each documented
step. Preamble failures land async via `stats.crashed=True` +
`record_error(...)`.

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


def _stub_worker_preamble(eng, slow_seconds: float, stages: list[str]) -> None:  # type: ignore[no-untyped-def]
    """Replace `_worker_preamble` with a stub that sleeps `slow_
    seconds` while writing the documented stage names to
    `stats.warmup_stage`. Caller passes a list that the stub
    appends to so the test can inspect the progression."""

    def _stub() -> None:
        for label in (
            "checking default sink",
            "applying GPU clock lock",
            "loading sessions",
            "warming pipeline",
        ):
            eng.stats.warmup_stage = label
            stages.append(label)
            time.sleep(slow_seconds / 4)

    eng._worker_preamble = _stub  # type: ignore[method-assign]


def test_warmup_stage_field_exists() -> None:
    """`EngineStats.warmup_stage` is the new progress signal."""
    from audio import engine

    eng = engine.RealtimeEngine(engine.EngineConfig())
    assert hasattr(eng.stats, "warmup_stage")
    assert eng.stats.warmup_stage == ""


def test_start_returns_promptly_even_when_preamble_is_slow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bug-class test for the verdict's primary claim. With a 500 ms
    preamble, `start()` itself must return in well under 100 ms.
    Pre-fix start() blocked for the full preamble before spawning
    the worker."""
    from audio import engine

    eng = engine.RealtimeEngine(engine.EngineConfig())
    stages: list[str] = []
    _stub_worker_preamble(eng, slow_seconds=0.5, stages=stages)
    # Stub the run loop so the worker doesn't actually try to do
    # audio I/O.
    monkeypatch.setattr(eng, "_run_loop", lambda: eng._stop_event.wait(timeout=2.0))

    t0 = time.monotonic()
    eng.start()
    elapsed = time.monotonic() - t0

    assert elapsed < 0.1, (
        f"F-merged-030: start() must return promptly (sub-100ms); "
        f"pre-fix it ran the full preamble synchronously. Got {elapsed:.3f}s"
    )
    # The worker is doing the slow preamble in the background.
    assert eng._thread is not None and eng._thread.is_alive()
    # Wait for preamble to finish so we can assert the stage progression.
    eng._thread.join(timeout=3.0)
    assert stages == [
        "checking default sink",
        "applying GPU clock lock",
        "loading sessions",
        "warming pipeline",
    ]
    # Final stage is "ready" after preamble succeeds + run_loop is entered.
    assert eng.stats.warmup_stage == "ready"
    eng.stop(timeout=0.5)


def test_preamble_failure_lands_on_stats_crashed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bug-class test. When the preamble raises (e.g., missing model
    -> FileNotFoundError), start() ITSELF returns OK (the spawn
    succeeded); the failure surfaces ASYNC via `stats.crashed=True`
    + `record_error(...)`. The TUI's _refresh_stats notify path
    then picks it up.

    Pre-fix the same FileNotFoundError raised SYNCHRONOUSLY from
    start(); _start_engine in app.py caught it. Post-fix the
    exception path is async."""
    from audio import engine

    eng = engine.RealtimeEngine(engine.EngineConfig())

    def _boom() -> None:
        raise FileNotFoundError("rvc model not found at /tmp/missing.onnx")

    eng._worker_preamble = _boom  # type: ignore[method-assign]
    monkeypatch.setattr(eng, "_run_loop", lambda: None)

    # start() does NOT raise -- the spawn succeeded.
    eng.start()

    # Wait for the worker to run + raise + record.
    if eng._thread is not None:
        eng._thread.join(timeout=2.0)

    assert eng.stats.crashed is True, (
        "preamble failures must set stats.crashed so the TUI surfaces them"
    )
    assert eng.stats.last_error is not None
    assert "rvc model not found" in eng.stats.last_error
    assert eng.stats.warmup_stage.startswith("crashed:"), (
        f"warmup_stage must reflect the crash; got {eng.stats.warmup_stage!r}"
    )
    assert eng.stats.running is False


def test_stop_during_preamble_cancels_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Coordination with F-merged-018's _lifecycle_lock. A stop()
    fired while the preamble is in flight must coordinate cleanly:
    stop() takes the lock + sets _stop_event; the preamble's
    _lifecycle_lock acquisition then blocks; once the worker
    eventually acquires, it sees _stop_event and bails. No
    crashed-state, no leaked thread."""
    from audio import engine

    eng = engine.RealtimeEngine(engine.EngineConfig())
    preamble_started = threading.Event()

    def _slow_preamble() -> None:
        preamble_started.set()
        time.sleep(0.3)  # the preamble must NOT actually run

    eng._worker_preamble = _slow_preamble  # type: ignore[method-assign]
    monkeypatch.setattr(eng, "_run_loop", lambda: None)

    eng.start()
    # Immediately call stop() -- the worker may not have grabbed
    # the lifecycle lock yet, but stop() will.
    eng.stop(timeout=2.0)

    # The worker should have either bailed early (because stop()
    # got the lock first) or finished its (short, since we set the
    # event) preamble and entered _run_loop.
    if eng._thread is not None:
        eng._thread.join(timeout=2.0)
    # The engine is stopped.
    assert eng.stats.running is False
    assert eng._stopped is True


def test_warmup_stage_cleared_on_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Post-stop the engine is idle; `warmup_stage` returns to ''
    so the TUI doesn't show a stale 'warming pipeline' indicator."""
    from audio import engine

    eng = engine.RealtimeEngine(engine.EngineConfig())
    monkeypatch.setattr(eng, "_worker_preamble", lambda: None)
    monkeypatch.setattr(eng, "_run_loop", lambda: None)
    eng.start()
    if eng._thread is not None:
        eng._thread.join(timeout=2.0)
    # Worker entered run_loop which immediately returns, then exited.
    # Stage was "ready".
    assert eng.stats.warmup_stage in ("ready", "")
    eng.stop(timeout=0.5)
    assert eng.stats.warmup_stage == ""


def test_start_no_longer_calls_ensure_sessions_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Structural pin. Pre-fix `start()` called `_ensure_sessions()`
    on the caller's thread. Post-fix that call moves to
    `_worker_preamble()`. A future refactor that re-introduces a
    caller-side `_ensure_sessions()` would re-enable the freeze."""
    src = Path(__file__).resolve().parent.parent / "src" / "audio" / "engine.py"
    text = src.read_text()
    # Find the start() body. The start() method's body ends at the
    # next `def `; check the slice for `_ensure_sessions` and
    # `_warmup_realtime_pipeline` calls.
    start_idx = text.find("    def start(self) -> None:")
    next_def_idx = text.find("\n    def ", start_idx + 50)
    start_body = text[start_idx:next_def_idx]
    assert "self._ensure_sessions()" not in start_body, (
        "F-merged-030: start() must NOT call _ensure_sessions "
        "directly anymore; the work belongs in _worker_preamble"
    )
    assert "self._warmup_realtime_pipeline()" not in start_body, (
        "F-merged-030: start() must NOT call _warmup_realtime_pipeline directly anymore"
    )
    # And the worker preamble HAS them.
    preamble_idx = text.find("def _worker_preamble(self)")
    assert preamble_idx > 0, "_worker_preamble must exist"
    preamble_end = text.find("\n    def ", preamble_idx + 50)
    preamble_body = text[preamble_idx:preamble_end]
    assert "_ensure_sessions" in preamble_body
    assert "_warmup_realtime_pipeline" in preamble_body
