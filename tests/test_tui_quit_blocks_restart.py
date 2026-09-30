"""Nothing may start the engine again while the TUI is quitting.

engine.stop() runs off the event loop for several seconds during quit,
and the control socket keeps answering. A `woys toggle` (or `t`) in that
window started a fresh engine that quit then left running at exit.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

import pytest


@pytest.mark.asyncio
async def test_toggle_during_quit_does_not_restart_the_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tui.app import WoysApp
    from tui.config import AppConfig

    runtime = tmp_path / "run"
    runtime.mkdir(mode=0o700)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    starts: list[int] = []
    monkeypatch.setattr(WoysApp, "_start_engine", lambda self: starts.append(1) or True)
    app = WoysApp(cfg=AppConfig(autostart_engine=False), no_pw_setup=True)
    stopping = threading.Event()
    release = threading.Event()

    def slow_stop(*_a: Any, **_k: Any) -> None:
        stopping.set()
        release.wait(timeout=5)

    app.engine.stop = slow_stop  # type: ignore[method-assign]
    async with app.run_test() as pilot:
        await pilot.pause()
        quit_task = asyncio.create_task(app.action_quit())
        assert await asyncio.to_thread(stopping.wait, 3)

        toggle_reply = await asyncio.to_thread(app._handle_control, "TOGGLE")
        profile_reply = await asyncio.to_thread(app._handle_control, "PROFILE any")
        app.action_toggle_engine()

        release.set()
        await asyncio.wait_for(quit_task, 5)

    assert starts == []
    assert toggle_reply.startswith("ERR"), toggle_reply
    assert profile_reply.startswith("ERR"), profile_reply
