"""save_config creates the config directory 0700, as the file header says.

Pre-fix `mkdir(parents=True, exist_ok=True)` used the default mode, so
under the usual umask 022 `~/.config/woys` came out 0755 and other local
users could list it, although the header promises a private 0700 dir.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture
def umask_022() -> Iterator[None]:
    old = os.umask(0o022)
    try:
        yield
    finally:
        os.umask(old)


@pytest.mark.usefixtures("umask_022")
def test_save_config_creates_the_dir_0700(tmp_path: Path) -> None:
    from tui.config import AppConfig, save_config

    cfg_dir = tmp_path / "woys"
    save_config(AppConfig(), cfg_dir / "config.toml")
    assert stat.S_IMODE(cfg_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE((cfg_dir / "config.toml").stat().st_mode) == 0o600


@pytest.mark.usefixtures("umask_022")
def test_save_config_leaves_an_existing_dir_mode_alone(tmp_path: Path) -> None:
    # A WOYS_CONFIG_DIR the user picked may be shared on purpose; only a
    # directory woys itself creates is made private.
    from tui.config import AppConfig, save_config

    cfg_dir = tmp_path / "shared"
    cfg_dir.mkdir(mode=0o750)
    save_config(AppConfig(), cfg_dir / "config.toml")
    assert stat.S_IMODE(cfg_dir.stat().st_mode) == 0o750
