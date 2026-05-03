"""Shared pytest fixtures for vcclient-cachy."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path.home() / ".local" / "share" / "vcclient-cachy" / "models"
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def models_dir() -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIXTURES_DIR


def _has_gpu() -> bool:
    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=3
        )
        return out.returncode == 0 and "GPU" in out.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _has_pipewire() -> bool:
    if not shutil.which("pactl"):
        return False
    try:
        out = subprocess.run(
            ["pactl", "info"], capture_output=True, text=True, timeout=3
        )
        return out.returncode == 0 and "PipeWire" in out.stdout
    except subprocess.TimeoutExpired:
        return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skip_gpu = pytest.mark.skip(reason="no GPU detected")
    skip_pw = pytest.mark.skip(reason="PipeWire not running")
    has_gpu = _has_gpu()
    has_pw = _has_pipewire()
    for item in items:
        if "gpu" in item.keywords and not has_gpu:
            item.add_marker(skip_gpu)
        if "pipewire" in item.keywords and not has_pw:
            item.add_marker(skip_pw)
