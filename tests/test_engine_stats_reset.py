"""Each engine run starts with fresh per-session stats.

`RealtimeEngine.stats` is created once and start() only reset `crashed`
and `warmup_stage`. After the first stop->start, `chunks_processed` was
already >= 10, so the TUI's warmup indicator (`running and
chunks_processed < 10`) never showed again, and late_chunks / max_* /
averages / drop counters mixed two sessions.

What must survive a restart: the error history, and the GPU clock-lock
state (a lock whose revert failed is recovered on the next start).

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import gc
import signal
from collections.abc import Iterator

import pytest

from audio import engine


@pytest.fixture
def process_state() -> Iterator[None]:
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    gc_was_enabled = gc.isenabled()
    yield None
    for sig, handler in saved.items():
        signal.signal(sig, handler)
    if gc_was_enabled:
        gc.enable()


def test_start_resets_per_session_stats(
    monkeypatch: pytest.MonkeyPatch, process_state: None
) -> None:
    eng = engine.RealtimeEngine(engine.EngineConfig())
    monkeypatch.setattr(eng, "_worker_main", lambda: None)
    stats = eng.stats
    # A previous session's leftovers.
    stats.chunks_processed = 400
    stats.late_chunks = 7
    stats.max_inference_ms = 91.0
    stats.dropped_chunks = 2
    stats._recent_inference.append(30.0)
    eng.record_error("inference dropped chunk #1: RuntimeError: transient")
    stats.gpu_clock_lock_active = True
    stats.gpu_clock_lock_revert_failed = True

    eng.start()
    try:
        assert eng.stats is stats, "readers hold the stats object; reset it in place"
        assert eng._stats_lock is stats._internal_lock
        assert stats.chunks_processed == 0
        assert stats.late_chunks == 0
        assert stats.max_inference_ms == 0.0
        assert stats.dropped_chunks == 0
        assert list(stats._recent_inference) == []
        assert stats.running is True
        assert stats.warmup_stage == "starting"
        # Kept across runs.
        assert [m for _t, _th, m in eng.recent_errors(5)] == [
            "inference dropped chunk #1: RuntimeError: transient"
        ]
        assert stats.gpu_clock_lock_active is True
        assert stats.gpu_clock_lock_revert_failed is True
    finally:
        eng.stop(timeout=1.0)
