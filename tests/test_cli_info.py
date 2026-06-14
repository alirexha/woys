"""`woys info` must report the ONNX Runtime
version and CUDA EP availability.

Pre-fix `cmd_info` printed Python / PipeWire / GPU but never imported
`onnxruntime` -- so a user whose CUDA EP silently fell back to CPU (the
F-merged-001 P0) had no command that revealed it, despite `info` being the
command literally named for runtime diagnostics.

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


def test_cmd_info_reports_onnxruntime_and_cuda_ep(capsys: pytest.CaptureFixture[str]) -> None:
    """`woys info` output must name onnxruntime and the CUDA execution
    provider. Pre-fix neither string appears (cmd_info never imported ORT)."""
    from woys.cli import cmd_info

    rc = cmd_info()
    out = capsys.readouterr().out

    assert rc == 0
    assert "onnxruntime" in out.lower(), "woys info must report the ONNX Runtime version"
    assert "CUDAExecutionProvider" in out, (
        "woys info must report CUDA EP availability -- the diagnostic half "
        "of the F-merged-001 silent-CPU-fallback P0"
    )
    # The active model line is the most common first-run failure surface.
    assert "rvc model" in out.lower()
