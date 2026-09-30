"""A config.toml that fails to load must never be overwritten.

load_config falls back to in-memory defaults when the file is malformed or
unreadable (so woys still starts), and tells the user "the file was NOT
touched". Pre-fix the very next save -- `woys profile save`, `models use`,
or simply quitting the TUI -- wrote those defaults over the file and deleted
every saved profile, the pinned `_user_overrides` and the model path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tui.config import ConfigNotSavedError, load_config, save_config

BAD = 'f0_up_key = 3\n_user_overrides = ["f0_up_key"]\n\n[profiles.alice]\nf0_up_key = 5\n\n[profiles.bob\n'


def test_save_refuses_to_overwrite_malformed_config(tmp_path: Path) -> None:
    cf = tmp_path / "config.toml"
    cf.write_text(BAD)
    cfg = load_config(cf)
    with pytest.raises(ConfigNotSavedError, match="malformed"):
        save_config(cfg, cf)
    assert cf.read_text() == BAD


def test_save_refuses_after_unreadable_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cf = tmp_path / "config.toml"
    cf.write_text("f0_up_key = 3\n")
    real_open = open

    def _deny(path: object, *args: object, **kwargs: object) -> object:
        if Path(str(path)) == cf and args and "r" in str(args[0]):
            raise PermissionError(13, "Permission denied")
        return real_open(path, *args, **kwargs)  # type: ignore[call-overload]

    monkeypatch.setattr("builtins.open", _deny)
    cfg = load_config(cf)
    monkeypatch.undo()
    with pytest.raises(ConfigNotSavedError):
        save_config(cfg, cf)
    assert cf.read_text() == "f0_up_key = 3\n"


def test_cli_profile_save_exits_nonzero_and_keeps_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The exact call shape that lost data: `woys profile save new`."""
    import tui.config as cfg_mod
    from woys import cli

    cf = tmp_path / "config.toml"
    cf.write_text(BAD)
    monkeypatch.setattr(cfg_mod, "CONFIG_FILE", cf)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    rc = cli.main(["profile", "save", "new"])
    assert rc != 0
    assert cf.read_text() == BAD
    assert "malformed" in capsys.readouterr().err


def test_good_config_still_saves(tmp_path: Path) -> None:
    cf = tmp_path / "config.toml"
    cf.write_text("f0_up_key = 3\n")
    cfg = load_config(cf)
    cfg.f0_up_key = 4
    save_config(cfg, cf)
    assert load_config(cf).f0_up_key == 4


def test_missing_config_is_created(tmp_path: Path) -> None:
    cf = tmp_path / "sub" / "config.toml"
    load_config(cf)
    assert cf.exists()


def test_tui_quit_keeps_malformed_config(tmp_path: Path) -> None:
    """Quitting the TUI saves the config; with a file that failed to load
    the save must be refused and the quit must still complete."""
    import asyncio

    import tui.app as app_mod
    import tui.config as cfg_mod

    cf = tmp_path / "config.toml"
    cf.write_text(BAD)
    cfg = load_config(cf)
    app = app_mod.WoysApp(cfg=cfg, no_pw_setup=True)
    exits: list[tuple[object, ...]] = []
    app.push_screen = lambda *a, **kw: None  # type: ignore[method-assign,assignment]
    app.engine.stop = lambda *a, **kw: None  # type: ignore[method-assign]
    app._control.stop = lambda *a, **kw: None  # type: ignore[method-assign]
    app.exit = lambda *a, **kw: exits.append((a, kw))  # type: ignore[method-assign]
    orig = cfg_mod.CONFIG_FILE
    cfg_mod.CONFIG_FILE = cf
    try:
        asyncio.run(app.action_quit())
    finally:
        cfg_mod.CONFIG_FILE = orig
    assert exits, "quit must still exit"
    assert cf.read_text() == BAD
