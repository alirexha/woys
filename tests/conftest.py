"""Shared pytest fixtures for woys."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
SERVER_ROOT = PROJECT_ROOT / "src" / "server"
MODELS_DIR = Path.home() / ".local" / "share" / "woys" / "models"
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"

# upstream uses unprefixed imports (from voice_changer.X import Y); inject the
# server dir on sys.path before any test collection so those imports resolve.
# src/ itself goes on too so `import tui.config` resolves without leaning on
# the editable install (which breaks the moment the project dir is renamed).
for _root in (SRC_ROOT, SERVER_ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))


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


@pytest.fixture(autouse=True)
def _isolate_woys_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Hard-isolate the on-disk config for EVERY test.

    Pre-fix, any test that called load_config()/save_config()/cli_profile_*
    with no explicit path wrote the user's REAL ~/.config/woys/config.toml --
    a bare save_config(AppConfig()) reset it to pristine defaults and wiped
    every saved profile (happened 2026-05-15, -06-07, -06-14). Redirect the
    module globals that load_config/save_config resolve at call time so no
    test can reach the real file, even one that forgets to pass a tmp path.
    """
    import tui.config as _cfg

    cfg_dir = tmp_path / "woys-config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(_cfg, "CONFIG_DIR", cfg_dir)
    monkeypatch.setattr(_cfg, "CONFIG_FILE", cfg_dir / "config.toml")


def _has_gpu() -> bool:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=3)
        return out.returncode == 0 and "GPU" in out.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _has_pipewire() -> bool:
    if not shutil.which("pactl"):
        return False
    try:
        out = subprocess.run(["pactl", "info"], capture_output=True, text=True, timeout=3)
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
