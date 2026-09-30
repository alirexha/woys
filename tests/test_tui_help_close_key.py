"""The key that closes the help modal must not also run its action.

The help says "press any key to close". Closing it with `t` also
toggled the engine, `q` quit the app, `0` reset the pitch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["t", "q", "0", "p"])
async def test_closing_help_does_not_run_the_key_action(
    key: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tui.app import HelpScreen, WoysApp
    from tui.config import AppConfig

    runtime = tmp_path / "run"
    runtime.mkdir(mode=0o700)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    ran: list[str] = []

    def record(name: str) -> Any:
        return lambda *_a, **_k: ran.append(name)

    monkeypatch.setattr(WoysApp, "action_toggle_engine", record("toggle"))
    monkeypatch.setattr(WoysApp, "action_pitch_reset", record("pitch_reset"))
    monkeypatch.setattr(WoysApp, "action_cycle_profile", record("cycle"))
    cfg = AppConfig(autostart_engine=False, f0_up_key=3)
    app = WoysApp(cfg=cfg, no_pw_setup=True)
    async with app.run_test() as pilot:
        await pilot.press("question_mark")
        await pilot.pause()
        assert any(isinstance(s, HelpScreen) for s in app.screen_stack)

        await pilot.press(key)
        await pilot.pause()

        assert not any(isinstance(s, HelpScreen) for s in app.screen_stack)
        assert ran == []
        assert app.is_running
        assert app.return_code is None
