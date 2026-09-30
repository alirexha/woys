"""`woys info` must not call a compiled-in CUDA provider "available".

`ort.get_available_providers()` lists the execution providers built into
the onnxruntime wheel, not ones that can run here. On a box with no
NVIDIA GPU or driver, `woys info` still printed "CUDAExecutionProvider:
available" -- on the command the error hints send users to for exactly
that question.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import shutil

import onnxruntime as ort
import pytest

import woys.cli as cli


def _no_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    monkeypatch.setattr(ort, "preload_dlls", lambda *a, **k: None, raising=False)


def test_info_says_built_in_not_available(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _no_tools(monkeypatch)
    monkeypatch.setattr(
        ort,
        "get_available_providers",
        lambda: ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    assert cli.cmd_info() == 0
    out = capsys.readouterr().out
    assert "CUDAExecutionProvider: built into this onnxruntime" in out
    assert ": available" not in out
    assert "no NVIDIA GPU detected" in out


def test_info_flags_a_build_without_cuda(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _no_tools(monkeypatch)
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    cli.cmd_info()
    out = capsys.readouterr().out
    assert "CUDAExecutionProvider: NOT in this onnxruntime build" in out
    assert "onnxruntime-gpu" in out
