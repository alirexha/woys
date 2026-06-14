"""`_make_session` must hard-fail the silent
CUDA->CPU fallback.

Pre-fix `_make_session` appended `CPUExecutionProvider` as an unconditional
landing pad and never checked `get_providers()`. A broken onnxruntime-gpu
wheel / NVIDIA driver / missing `ort.preload_dlls()` therefore produced a
working-looking but unusable engine -- realtime RVC on CPU runs ~10-50x over
the chunk-period latency budget -- with no error surfaced anywhere. Five
areas converged on this independently; it is the strongest signal in the
audit.

These tests stub `ort.InferenceSession` so they need no GPU and no model
files -- they pin the *decision* `_make_session` makes about a CPU-bound
session, which is the bug class.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from audio import engine  # noqa: E402


class _FakeCpuOnlySession:
    """Stand-in for an ORT session that bound CPU-only (the failure mode)."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def get_providers(self) -> list[str]:
        return ["CPUExecutionProvider"]


def _patch(monkeypatch: pytest.MonkeyPatch, available: list[str]) -> None:
    monkeypatch.setattr(engine.ort, "get_available_providers", lambda: available)
    monkeypatch.setattr(engine.ort, "InferenceSession", _FakeCpuOnlySession)


def test_make_session_hard_fails_silent_cpu_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """CUDA EP installed but the session bound CPU-only -> raise, don't return.

    This is the bug-class test. On pre-fix code `_make_session` returns the
    CPU-only session normally and `pytest.raises` fails with DID NOT RAISE.
    """
    _patch(monkeypatch, ["CUDAExecutionProvider", "CPUExecutionProvider"])

    with pytest.raises(RuntimeError) as exc:
        engine._make_session(Path("/nonexistent/model.onnx"), use_tensorrt=False)

    assert type(exc.value).__name__ == "CpuFallbackError"
    assert "CPU" in str(exc.value)


def test_make_session_allows_cpu_when_explicitly_opted_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """allow_cpu_fallback=True is a deliberate override -> no raise."""
    _patch(monkeypatch, ["CUDAExecutionProvider", "CPUExecutionProvider"])

    sess = engine._make_session(
        Path("/nonexistent/model.onnx"), use_tensorrt=False, allow_cpu_fallback=True
    )
    assert engine._session_is_cpu_only(sess)


def test_make_session_cpu_only_build_is_not_a_silent_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No CUDA EP in the ORT build at all -> CPU is the environment, not a
    silent *fallback*; `_make_session` must not raise. (The no-GPU condition
    is surfaced elsewhere by `woys info` -- F-merged-013.)"""
    _patch(monkeypatch, ["CPUExecutionProvider"])

    sess = engine._make_session(Path("/nonexistent/model.onnx"), use_tensorrt=False)
    assert engine._session_is_cpu_only(sess)
