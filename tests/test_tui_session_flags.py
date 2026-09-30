"""`woys run --autostart` / `--monitor` / `--no-monitor` are session-only.

run_tui used to write the flags into the AppConfig it then saved on
quit, so one `woys run --autostart --monitor` made every later bare
`woys` start the engine and play the converted voice to the speakers.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def headless(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Run WoysApp headless through run_tui and record engine starts."""
    import tui.app as app_mod

    runtime = tmp_path / "run"
    runtime.mkdir(mode=0o700)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    state: dict[str, Any] = {"starts": 0, "keys": [], "engine_monitor": None}

    def fake_start(self: Any) -> bool:
        state["starts"] += 1
        return True

    def run(self: Any, *_a: Any, **_k: Any) -> int:
        async def go() -> None:
            async with self.run_test() as pilot:
                await pilot.pause()
                state["engine_monitor"] = self.engine.cfg.monitor
                for key in state["keys"]:
                    await pilot.press(key)
                await pilot.press("q")
                await pilot.pause()

        asyncio.run(go())
        return 0

    monkeypatch.setattr(app_mod.WoysApp, "_start_engine", fake_start)
    monkeypatch.setattr(app_mod.WoysApp, "run", run)
    return state


def test_cli_flags_are_not_persisted(headless: dict[str, Any]) -> None:
    import tui.config as tc
    from tui.app import run_tui

    tc.save_config(tc.AppConfig())

    assert run_tui(no_pw_setup=True, autostart=True, monitor=True) == 0
    assert headless["starts"] == 1
    assert headless["engine_monitor"] is True
    cfg = tc.load_config()
    assert cfg.autostart_engine is False
    assert cfg.monitor is False

    headless["starts"] = 0
    assert run_tui(no_pw_setup=True, autostart=False, monitor=None) == 0
    assert headless["starts"] == 0
    assert headless["engine_monitor"] is False


def test_no_monitor_overrides_config_for_the_session(headless: dict[str, Any]) -> None:
    import tui.config as tc
    from tui.app import run_tui

    tc.save_config(tc.AppConfig(monitor=True))

    assert run_tui(no_pw_setup=True, monitor=False) == 0
    assert headless["engine_monitor"] is False
    assert tc.load_config().monitor is True


def test_monitor_key_in_a_flag_session_is_persisted(headless: dict[str, Any]) -> None:
    import tui.config as tc
    from tui.app import run_tui

    tc.save_config(tc.AppConfig(monitor=False))
    headless["keys"] = ["m", "m", "m"]  # --monitor on, then off, on, off

    assert run_tui(no_pw_setup=True, monitor=True) == 0
    assert tc.load_config().monitor is False

    headless["keys"] = ["m"]
    assert run_tui(no_pw_setup=True, monitor=False) == 0
    assert tc.load_config().monitor is True
