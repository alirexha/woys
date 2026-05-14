"""review F-23-06 (P1): a PipeWire-setup failure in the TUI must be
blocking.

Pre-fix `WoysApp.on_mount` caught the `PipeWireError`, recorded an 8 s
toast, then fell straight through to `_start_engine()` -- so the app
showed a green RUNNING status on a setup with no `woys-mic` device. Bare
`woys` passes `autostart=True`, so this was the *default* invocation: a
clean Hard Rule 2 violation on the product's core function.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "src" / "server") not in sys.path:
    sys.path.insert(0, str(REPO / "src" / "server"))


def _mk_app(monkeypatch: pytest.MonkeyPatch) -> object:
    """A WoysApp wired so on_mount() has no real side effects except the
    PipeWire-setup branch under test."""
    from tui.app import WoysApp
    from tui.config import AppConfig

    cfg = AppConfig()
    cfg.autostart_engine = True
    app = WoysApp(cfg=cfg, no_pw_setup=False)
    monkeypatch.setattr(app, "notify", lambda *a, **k: None)
    monkeypatch.setattr(app._control, "start", lambda: None)
    monkeypatch.setattr(app, "set_interval", lambda *a, **k: None)
    return app


def test_on_mount_does_not_autostart_when_pipewire_setup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bug-class test: a failed `VirtualMic().ensure()` must block
    autostart. Pre-fix on_mount fell through to `_start_engine()`."""
    from audio.pipewire import PipeWireError

    app = _mk_app(monkeypatch)

    def _raise(_self: object) -> None:
        raise PipeWireError("pactl missing")

    monkeypatch.setattr("audio.pipewire.VirtualMic.ensure", _raise)
    started: list[bool] = []
    monkeypatch.setattr(app, "_start_engine", lambda: started.append(True) or True)

    app.on_mount()

    assert started == [], "on_mount must NOT autostart the engine when PipeWire setup failed"
    last_error = app.engine.stats.last_error
    assert last_error and "PipeWire" in last_error and "woys pw setup" in last_error


def test_on_mount_autostarts_when_pipewire_setup_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression guard: a *successful* PipeWire setup must still autostart
    -- the fix must block only on failure, not over-block."""
    app = _mk_app(monkeypatch)
    monkeypatch.setattr("audio.pipewire.VirtualMic.ensure", lambda _self: None)
    started: list[bool] = []
    monkeypatch.setattr(app, "_start_engine", lambda: started.append(True) or True)

    app.on_mount()

    assert started == [True], "on_mount must still autostart when PipeWire setup succeeds"
