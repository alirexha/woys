"""review F-merged-016 (P1): `_probe_rvc_output_sr` must not silently
guess the RVC model's output sample rate.

Pre-fix it returned 16 kHz on `except Exception` ("possibly chipmunk", the
comment admitted) and treated an unrecognised sample count as raw Hz on
the unknown-rate branch. Either guess pitch-shifts the whole session and
poisons `_rvc_sr_cache`, with no `last_error` and no counter. CX2 corrected
the mechanism: the realistic trigger is a transient GPU/cuDNN error on the
cold first pass, where swallowing is most dangerous -- so the fix is to
re-raise.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "src" / "server") not in sys.path:
    sys.path.insert(0, str(REPO / "src" / "server"))

from audio import engine  # noqa: E402


class _FakeInput:
    shape: ClassVar[list[object]] = [1, None, 768]


class _FakeRvc:
    """Minimal stand-in for the loaded RVC ORT session."""

    def __init__(self, *, run_result: object) -> None:
        self._run_result = run_result

    def get_inputs(self) -> list[_FakeInput]:
        return [_FakeInput()]

    def run(self, *_a: object, **_k: object) -> object:
        if isinstance(self._run_result, BaseException):
            raise self._run_result
        return self._run_result


def _engine_with_rvc(run_result: object) -> engine.RealtimeEngine:
    eng = engine.RealtimeEngine(engine.EngineConfig())
    eng._rvc = _FakeRvc(run_result=run_result)  # type: ignore[assignment]
    eng._is_half = False
    return eng


def test_probe_rvc_output_sr_raises_instead_of_returning_16k() -> None:
    """A probe failure must raise -- not silently return 16 kHz. Pre-fix
    the `except` returned 16000, so `pytest.raises` fails with DID NOT
    RAISE."""
    eng = _engine_with_rvc(RuntimeError("transient cuDNN error"))
    with pytest.raises(RuntimeError, match="probe"):
        eng._probe_rvc_output_sr()


def test_probe_rvc_output_sr_raises_on_unrecognised_rate() -> None:
    """A sample count that matches no known RVC training rate must raise --
    not be returned as raw Hz."""
    # 99_999 samples for a 1 s input matches none of 16/22.05/24/32/40/44.1/48k.
    eng = _engine_with_rvc([np.zeros(99_999, dtype=np.float32)])
    with pytest.raises(RuntimeError, match="no known RVC training rate"):
        eng._probe_rvc_output_sr()


def test_probe_rvc_output_sr_returns_known_rate_on_clean_probe() -> None:
    """The happy path still works: ~40 kHz of output samples -> 40000."""
    eng = _engine_with_rvc([np.zeros(40_000, dtype=np.float32)])
    assert eng._probe_rvc_output_sr() == 40_000
