"""Control-socket QUIT must not deadlock the TUI.

The QUIT handler runs on a ControlServer pool worker. Teardown calls
`ControlServer.stop()`, which joins every pool worker -- including the
one still serving QUIT. If the handler waits for the teardown to finish,
the event loop and that worker wait on each other forever. The handler
must hand the quit to the event loop and return its reply first.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

import pytest


@pytest.mark.asyncio
async def test_socket_quit_replies_before_teardown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tui.app import WoysApp
    from tui.config import AppConfig

    runtime = tmp_path / "run"
    runtime.mkdir(mode=0o700)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))

    app = WoysApp(cfg=AppConfig(autostart_engine=False), no_pw_setup=True)
    handler_returned = threading.Event()
    seen: dict[str, Any] = {}

    def fake_engine_stop(*_a: Any, **_k: Any) -> None:
        seen["engine_stop_thread"] = threading.current_thread() is threading.main_thread()

    async with app.run_test() as pilot:
        await pilot.pause()
        real_control_stop = app._control.stop

        def control_stop_after_reply() -> None:
            # In production this join blocks until the QUIT worker has
            # returned. Bounded here so the old behaviour fails the test
            # instead of hanging it.
            seen["reply_first"] = handler_returned.wait(timeout=3.0)
            real_control_stop()

        app.engine.stop = fake_engine_stop  # type: ignore[method-assign]
        app._control.stop = control_stop_after_reply  # type: ignore[method-assign]

        reply = await asyncio.to_thread(app._handle_control, "QUIT")
        handler_returned.set()
        for _ in range(100):
            if app.return_code is not None:
                break
            await asyncio.sleep(0.05)

    assert reply == "OK quitting"
    assert seen.get("reply_first") is True, "QUIT must reply before ControlServer.stop() joins"
    assert seen.get("engine_stop_thread") is False, "engine.stop must run off the event loop"
    assert app.return_code == 0
