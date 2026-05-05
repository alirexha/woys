"""Sanity checks for the host environment. These run first and gate everything else."""

from __future__ import annotations

import shutil
import subprocess
import sys

import pytest


def test_python_version() -> None:
    """We target 3.11 in the venv; reject system 3.13/3.14."""
    assert sys.version_info[:2] == (3, 11), f"woys targets Python 3.11; got {sys.version_info[:2]}"


def test_pactl_available() -> None:
    assert shutil.which("pactl") is not None, "pactl missing — install pipewire-pulse"


@pytest.mark.pipewire
def test_pipewire_running() -> None:
    out = subprocess.run(["pactl", "info"], capture_output=True, text=True, timeout=3)
    assert out.returncode == 0
    assert "PipeWire" in out.stdout, "PulseAudio detected — PipeWire required"


@pytest.mark.gpu
def test_nvidia_smi() -> None:
    out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=3)
    assert out.returncode == 0
    assert "GPU" in out.stdout
