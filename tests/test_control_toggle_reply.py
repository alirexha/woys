"""TOGGLE must reply ERR when the engine fails to start.

`woys toggle` exits 1 only on an ERR reply. The handler used to answer
"OK toggled" no matter what happened, so `woys toggle && notify-send on`
reported success for an engine that never started.
"""

from __future__ import annotations

from typing import Any

import pytest


def _app() -> Any:
    from tui.app import WoysApp
    from tui.config import AppConfig

    app = WoysApp(cfg=AppConfig(autostart_engine=False), no_pw_setup=True)
    app.call_from_thread = lambda fn, *a, **k: fn(*a, **k)  # type: ignore[method-assign]
    app.notify = lambda *_a, **_k: None  # type: ignore[method-assign]
    return app


def test_toggle_replies_err_when_start_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _app()

    def fail_start() -> None:
        raise FileNotFoundError("model missing: /nope.onnx")

    app.engine.start = fail_start  # type: ignore[method-assign]

    reply = app._handle_control("TOGGLE")

    assert reply.startswith("ERR"), reply
    assert "/nope.onnx" in reply


def test_toggle_replies_ok_when_start_succeeds() -> None:
    app = _app()
    app.engine.start = lambda: None  # type: ignore[method-assign]

    reply = app._handle_control("TOGGLE")

    assert reply.startswith("OK toggled"), reply
    assert "start" in reply


def test_toggle_replies_ok_when_stopping() -> None:
    app = _app()
    app.engine.stats.running = True
    app.engine.stop = lambda *_a, **_k: None  # type: ignore[method-assign]

    reply = app._handle_control("TOGGLE")

    assert reply.startswith("OK toggled"), reply
    assert "stop" in reply
