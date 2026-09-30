"""A configured voice model that is missing must be reported, not
silently replaced by the default voice."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


@pytest.mark.asyncio
async def test_missing_configured_model_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from audio.engine import DEFAULT_RVC_MODEL
    from tui.app import WoysApp
    from tui.config import AppConfig

    runtime = tmp_path / "run"
    runtime.mkdir(mode=0o700)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    gone = tmp_path / "gone.onnx"
    app = WoysApp(cfg=AppConfig(rvc_model=str(gone), autostart_engine=False), no_pw_setup=True)
    toasts: list[str] = []
    real_notify = app.notify

    def spy(message: str, **kw: Any) -> None:
        toasts.append(message)
        real_notify(message, **kw)

    app.notify = spy  # type: ignore[method-assign]
    async with app.run_test() as pilot:
        await pilot.pause(0.6)

    # Still falls back so a stale config.toml doesn't brick the engine...
    assert app.engine.cfg.rvc_model == DEFAULT_RVC_MODEL
    # ...but says so, in the error history (file log, `woys diag`) and a toast.
    history = [msg for _ts, _thread, msg in app.engine.stats.error_history]
    assert any(str(gone) in m and DEFAULT_RVC_MODEL.name in m for m in history), history
    assert any(str(gone) in t for t in toasts), toasts


def test_profile_with_missing_model_is_reported(tmp_path: Path) -> None:
    from tui.app import WoysApp
    from tui.config import AppConfig

    gone = tmp_path / "gone.onnx"
    cfg = AppConfig()
    cfg._extras["profiles"] = {"p": {"f0_up_key": 1, "rvc_model": str(gone)}}
    app = WoysApp(cfg=cfg, no_pw_setup=True)
    toasts: list[str] = []
    app.notify = lambda message, **_k: toasts.append(message)  # type: ignore[method-assign]

    assert app._apply_profile_named("p") is None

    assert any(str(gone) in t and "not found" in t for t in toasts), toasts
