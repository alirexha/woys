"""The config dir honors WOYS_CONFIG_DIR and XDG_CONFIG_HOME.

Pre-fix `tui/config.py` hardcoded `Path.home() / ".config" / "woys"` while its
own file header said `$XDG_CONFIG_HOME/woys/config.toml`, and `woys chain`
already honored XDG_CONFIG_HOME -- so a user with a custom XDG_CONFIG_HOME
got the chain unit in one tree and config.toml in another.

Each case runs in a fresh interpreter with HOME and the XDG/WOYS variables set
explicitly: the value under test is the one `tui.config` computes at import,
so nothing here depends on monkeypatching a module constant.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _config_file_for(env_extra: dict[str, str], home: Path) -> Path:
    env = {k: v for k, v in os.environ.items() if k not in ("XDG_CONFIG_HOME", "WOYS_CONFIG_DIR")}
    env["HOME"] = str(home)
    env["PYTHONPATH"] = str(REPO / "src")
    env.update(env_extra)
    out = subprocess.run(
        [sys.executable, "-c", "import tui.config as c; print(c.CONFIG_FILE)"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    return Path(out.stdout.strip().splitlines()[-1])


def test_default_is_home_dot_config(tmp_path: Path) -> None:
    assert _config_file_for({}, tmp_path) == tmp_path / ".config" / "woys" / "config.toml"


def test_xdg_config_home_is_honored(tmp_path: Path) -> None:
    xdg = tmp_path / "xdg"
    got = _config_file_for({"XDG_CONFIG_HOME": str(xdg)}, tmp_path / "home")
    assert got == xdg / "woys" / "config.toml"


def test_relative_xdg_config_home_is_ignored(tmp_path: Path) -> None:
    """The XDG spec says a relative path is invalid and must be ignored."""
    got = _config_file_for({"XDG_CONFIG_HOME": "rel/xdg"}, tmp_path)
    assert got == tmp_path / ".config" / "woys" / "config.toml"


def test_woys_config_dir_wins(tmp_path: Path) -> None:
    override = tmp_path / "override"
    got = _config_file_for(
        {"WOYS_CONFIG_DIR": str(override), "XDG_CONFIG_HOME": str(tmp_path / "xdg")},
        tmp_path,
    )
    assert got == override / "config.toml"


def test_existing_legacy_config_is_kept_when_xdg_dir_is_new(tmp_path: Path) -> None:
    """A user who set XDG_CONFIG_HOME before this fix has their config (and
    saved profiles) under ~/.config/woys. Keep reading it there rather than
    starting from blank defaults in the XDG tree."""
    home = tmp_path / "home"
    legacy = home / ".config" / "woys" / "config.toml"
    legacy.parent.mkdir(parents=True)
    legacy.write_text('rvc_model = ""\n')
    got = _config_file_for({"XDG_CONFIG_HOME": str(tmp_path / "xdg")}, home)
    assert got == legacy


def test_suite_never_resolves_the_real_config() -> None:
    """conftest points WOYS_CONFIG_DIR at a temp dir before any test module
    imports tui.config, so even an import-time binding of CONFIG_FILE lands
    outside the real ~/.config/woys."""
    import tui.config as cfg_mod

    real_dir = (Path.home() / ".config" / "woys").resolve()
    assert os.environ.get("WOYS_CONFIG_DIR")
    assert real_dir not in cfg_mod.CONFIG_FILE.resolve().parents
    assert real_dir not in Path(os.environ["WOYS_CONFIG_DIR"]).resolve().parents
    assert Path(os.environ["WOYS_CONFIG_DIR"]).resolve() != real_dir
