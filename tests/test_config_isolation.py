"""A test run must NEVER touch the user's real ~/.config/woys/config.toml.

Three real incidents wiped every saved profile (2026-05-15, 2026-06-07,
2026-06-14) because the suite wrote the LIVE config: load_config/save_config
baked CONFIG_FILE in as an import-time default arg, so monkeypatching it --
the only isolation lever a test has -- silently did nothing, and conftest had
no global guard. A bare save_config(AppConfig()) on the default path turned the
live config into pristine defaults: blank model, no _user_overrides, zero
profiles.

The fix has two halves, one test each, plus an end-to-end guard:
  1. load_config/save_config resolve CONFIG_FILE at CALL time (default None),
     so the module global is monkeypatchable.
  2. conftest ships an autouse fixture redirecting CONFIG_FILE to a tmp dir for
     every test, so no test can reach the real file even by accident.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import tui.config as cfg_mod  # noqa: E402

REAL_CONFIG = Path.home() / ".config" / "woys" / "config.toml"


def test_config_funcs_resolve_path_at_call_time() -> None:
    """load_config/save_config must not capture the real CONFIG_FILE as an
    import-time default; the default must be None so the (monkeypatchable)
    module global is read at call time."""
    assert inspect.signature(cfg_mod.load_config).parameters["path"].default is None
    assert inspect.signature(cfg_mod.save_config).parameters["path"].default is None


def test_config_path_is_isolated_during_tests() -> None:
    """The autouse conftest guard must redirect CONFIG_FILE away from the
    user's real ~/.config/woys/config.toml for every test."""
    assert cfg_mod.CONFIG_FILE.resolve() != REAL_CONFIG.resolve()


def test_no_path_save_writes_isolated_file_not_real_config(
    tmp_path: Path, monkeypatch
) -> None:
    """End-to-end guarantee: save_config() with NO explicit path honors the
    patched CONFIG_FILE. This is the exact call shape (cli_profile_save ->
    save_config(cfg)) that wiped the profiles."""
    target = tmp_path / "isolated" / "config.toml"
    monkeypatch.setattr(cfg_mod, "CONFIG_DIR", target.parent)
    monkeypatch.setattr(cfg_mod, "CONFIG_FILE", target)

    cfg_mod.save_config(cfg_mod.AppConfig())  # no path -> must honor the patch

    assert target.exists(), "save_config() ignored the patched CONFIG_FILE"
