"""The CLI owns the `[profiles]` table in config.toml.

`woys profile save/delete` edit config.toml while a TUI may be running.
The TUI must not write its startup copy of the profiles back over those
edits, and a profile saved after the TUI started must still be usable
from the TUI (PROFILE socket command, `p` key).
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pytest


def _disk_profiles() -> dict[str, Any]:
    import tui.config as tc

    raw = tomllib.loads(tc.CONFIG_FILE.read_text())
    bag = raw.get("profiles", {})
    assert isinstance(bag, dict)
    return bag


def _seed(profiles: dict[str, dict[str, Any]]) -> None:
    import tui.config as tc

    cfg = tc.AppConfig(autostart_engine=False)
    cfg._extras["profiles"] = profiles
    tc.save_config(cfg)


def _app() -> Any:
    import tui.config as tc
    from tui.app import WoysApp

    app = WoysApp(cfg=tc.load_config(), no_pw_setup=True)
    app.notify = lambda *_a, **_k: None  # type: ignore[method-assign]
    return app


def test_tui_save_keeps_profiles_edited_by_the_cli() -> None:
    from woys.profiles import cli_profile_delete, cli_profile_save

    _seed({"old": {"f0_up_key": 1}})
    app = _app()
    # The CLI edits the file behind the running TUI's back.
    assert cli_profile_save("from_cli") == 0
    assert cli_profile_delete("old", assume_yes=True) == 0

    app.action_pitch_up()
    assert app._save_cfg()

    assert set(_disk_profiles()) == {"from_cli"}


@pytest.mark.asyncio
async def test_tui_quit_keeps_profiles_saved_by_the_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from woys.profiles import cli_profile_save

    runtime = tmp_path / "run"
    runtime.mkdir(mode=0o700)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    _seed({"old": {"f0_up_key": 1}})
    app = _app()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert cli_profile_save("from_cli") == 0
        await pilot.press("q")
        await pilot.pause()

    assert set(_disk_profiles()) == {"old", "from_cli"}


def test_profile_saved_after_tui_start_applies() -> None:
    import tui.config as tc

    _seed({})
    app = _app()
    cfg = tc.load_config()
    cfg._extras["profiles"] = {"new": {"f0_up_key": 5}}
    tc.save_config(cfg)

    app._apply_profile_named("new")

    assert app._active_profile == "new"
    assert app.cfg.f0_up_key == 5


def test_cycle_key_sees_profiles_saved_after_tui_start() -> None:
    import tui.config as tc

    _seed({})
    app = _app()
    cfg = tc.load_config()
    cfg._extras["profiles"] = {"new": {"f0_up_key": 5}}
    tc.save_config(cfg)
    submitted: list[object] = []
    app._jobs.submit = submitted.append  # type: ignore[method-assign]

    app.action_cycle_profile()

    assert len(submitted) == 1
