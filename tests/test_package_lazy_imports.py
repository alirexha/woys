"""Short CLI commands must not pay for the engine, CUDA or Textual.

`woys toggle` / `woys pitch +1` / `woys status` only send a few bytes over
the control socket, and they are the commands a window-manager keybinding
runs. Pre-fix `audio/__init__.py` and `tui/__init__.py` imported
`audio.engine` (numpy, onnxruntime and its CUDA/cuDNN preload) and `tui.app`
(all of Textual) eagerly, so every key press cost ~0.5 s and ~200 MB.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

HEAVY = ("audio.engine", "onnxruntime", "numpy", "textual", "tui.app")


def _loaded_after(code: str) -> set[str]:
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    out = subprocess.run(
        [sys.executable, "-c", f"{code}\nimport sys\nprint(sorted(sys.modules))"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    return set(eval(out.stdout.strip().splitlines()[-1]))


def test_control_client_and_pipewire_imports_stay_light() -> None:
    loaded = _loaded_after("import tui.control\nimport audio.pipewire")
    assert not loaded & set(HEAVY), sorted(loaded & set(HEAVY))


def test_package_exports_still_resolve() -> None:
    loaded = _loaded_after(
        "from audio import RealtimeEngine, EngineConfig, PipeWireError\n"
        "from tui import AppConfig, load_config, WoysApp, VCClientApp\n"
        "assert VCClientApp is WoysApp"
    )
    assert "audio.engine" in loaded and "tui.app" in loaded
