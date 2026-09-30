"""Worker-thread failures must always surface as a crashed engine.

onnxruntime raises its own exception classes (InvalidProtobuf, Fail,
NoSuchFile, ...) that derive straight from `Exception`, not from
RuntimeError or OSError. A corrupt or truncated model therefore has to
land on the same `stats.crashed` / `record_error` path as a missing one,
otherwise the worker dies silently and the engine reads "loading
sessions" forever.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import gc
import signal
from collections.abc import Iterator

import pytest

from audio import engine


class _OrtLikeError(Exception):
    """Stand-in for onnxruntime's InvalidProtobuf (a plain Exception)."""


def _restore_process_state(prior: dict[int, object], gc_was_enabled: bool) -> None:
    for sig, handler in prior.items():
        signal.signal(sig, handler)  # type: ignore[arg-type]
    if gc_was_enabled:
        gc.enable()


@pytest.fixture
def process_state() -> Iterator[None]:
    prior = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    gc_was_enabled = gc.isenabled()
    yield None
    _restore_process_state(prior, gc_was_enabled)


def test_non_runtime_error_in_preamble_marks_engine_crashed(
    monkeypatch: pytest.MonkeyPatch, process_state: None
) -> None:
    eng = engine.RealtimeEngine(engine.EngineConfig())

    def _corrupt_model() -> None:
        raise _OrtLikeError("INVALID_PROTOBUF : Load model from rvc.onnx failed")

    monkeypatch.setattr(eng, "_worker_preamble", _corrupt_model)
    eng.start()
    assert eng._thread is not None
    eng._thread.join(timeout=5.0)
    try:
        assert not eng._thread.is_alive()
        assert eng.stats.crashed is True
        assert eng.stats.running is False
        assert eng.stats.last_error is not None
        assert "_OrtLikeError" in eng.stats.last_error
        assert eng.stats.warmup_stage.startswith("crashed")
    finally:
        eng.stop(timeout=1.0)
