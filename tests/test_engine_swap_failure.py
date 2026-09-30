"""A failed in-process model swap must not take the engine down.

The in-process (default) branch of `_apply_one_swap` wrote
`cfg.rvc_model` first and then loaded the session with no error handling.
A corrupt / incompatible / vanished model raised out of the chunk loop:
the engine crashed, the `_SwapRequest` never resolved (the TUI parked the
full 10 s and reported no swap error), `cfg.rvc_model` pointed at the
bad file, and any later request drained in the same batch was stranded.

Post-fix the failure lands on `req.error` + `record_error`, the old voice
keeps playing, and the rest of the batch is applied.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from audio import engine


class _FakeInput:
    type = "tensor(float)"
    shape = (1, "T", 768)


class _FakeSession:
    def __init__(self, name: str) -> None:
        self.name = name

    def get_inputs(self) -> list[_FakeInput]:
        return [_FakeInput()]


class _CorruptModelError(Exception):
    """Stand-in for onnxruntime's InvalidProtobuf (a plain Exception)."""


OLD = Path("/models/old.onnx")
BAD = Path("/models/bad.onnx")
NEW = Path("/models/new.onnx")


def _engine(monkeypatch: pytest.MonkeyPatch) -> tuple[engine.RealtimeEngine, _FakeSession]:
    eng = engine.RealtimeEngine(engine.EngineConfig(rvc_model=OLD))
    old = _FakeSession("old")
    eng._rvc = old  # type: ignore[assignment]
    eng._rvc_output_sr = 40_000
    sessions = {OLD: old, NEW: _FakeSession("new")}

    def get_or_create(path: Path) -> Any:
        if Path(path) == BAD:
            raise _CorruptModelError("INVALID_PROTOBUF : Load model from bad.onnx failed")
        return sessions[Path(path)]

    monkeypatch.setattr(eng._rvc_pool, "get_or_create", get_or_create)
    monkeypatch.setattr(eng, "_cached_rvc_sr", lambda _p: 40_000)
    return eng, old


def test_failed_swap_resolves_request_and_keeps_old_voice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eng, old = _engine(monkeypatch)
    req = eng.request_model_swap(BAD)

    eng._maybe_swap_model()  # pre-fix: raises out of the chunk loop

    assert req.completion.is_set()
    assert isinstance(req.error, _CorruptModelError)
    assert req not in eng._outstanding_swaps
    assert eng.cfg.rvc_model == OLD, "cfg must not point at a model that never loaded"
    assert eng._rvc is old
    assert eng.stats.last_error is not None and "bad.onnx" in eng.stats.last_error


def test_failed_swap_does_not_strand_the_rest_of_the_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eng, _old = _engine(monkeypatch)
    bad = eng.request_model_swap(BAD)
    good = eng.request_model_swap(NEW)

    eng._maybe_swap_model()

    assert bad.completion.is_set() and bad.error is not None
    assert good.completion.is_set() and good.error is None
    assert eng._outstanding_swaps == []
    assert eng.cfg.rvc_model == NEW
    assert eng._rvc is not None and eng._rvc.name == "new"  # type: ignore[attr-defined]


def test_failed_rate_probe_restores_the_old_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """The session loads but the output-rate probe fails: the half-swapped
    session must be rolled back, not left in place under the old rate."""
    eng, old = _engine(monkeypatch)
    eng._is_half = True

    def probe_fails(_p: Path) -> int:
        raise RuntimeError("failed to probe the RVC model's output sample rate")

    monkeypatch.setattr(eng, "_cached_rvc_sr", probe_fails)
    req = eng.request_model_swap(NEW)

    eng._maybe_swap_model()

    assert req.completion.is_set() and isinstance(req.error, RuntimeError)
    assert eng._rvc is old
    assert eng._is_half is True
    assert eng.cfg.rvc_model == OLD


def test_failed_swap_after_sola_flush_leaves_a_usable_output_resampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The swap flushes the SOLA tail through `_resampler_out`, which
    finalizes that soxr stream. If the new model then fails to load, the
    old voice keeps playing, so it needs a fresh output resampler or the
    next chunk dies with "Input after last input"."""
    eng, _old = _engine(monkeypatch)
    assert eng._sola is not None
    eng._rebuild_sola_for_rate(40_000)
    assert eng._sola is not None
    eng._sola._prev_tail = np.ones(64, dtype=np.float32)
    eng._resampler_out = engine._StreamResampler(40_000, eng.cfg.sink_rate)
    eng._resampler_out.process(np.zeros(400, dtype=np.float32))
    req = eng.request_model_swap(BAD)

    eng._maybe_swap_model()

    assert req.error is not None
    out = eng._resampler_out.process(np.zeros(4000, dtype=np.float32))
    assert out.dtype == np.float32
