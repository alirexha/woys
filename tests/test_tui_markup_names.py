"""Profile names are shown as text, never parsed as Textual markup.

A hand-edited config.toml can hold a profile named `x[/bold]`. Toasting
it through notify() with markup on raised MarkupError during render and
took the whole TUI down.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_NAME = "x[/bold]"


@pytest.mark.asyncio
async def test_profile_with_markup_name_does_not_crash_the_tui(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tui.app import WoysApp
    from tui.config import AppConfig

    runtime = tmp_path / "run"
    runtime.mkdir(mode=0o700)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    cfg = AppConfig(autostart_engine=False)
    cfg._extras["profiles"] = {_NAME: {"f0_up_key": 2}}
    app = WoysApp(cfg=cfg, no_pw_setup=True)
    async with app.run_test(notifications=True) as pilot:
        await pilot.press("p")
        for _ in range(10):
            await pilot.pause(0.05)
        assert app.is_running
        assert app._active_profile == _NAME
        await pilot.press("p")
        await pilot.pause(0.1)
        assert app.is_running
    assert app.return_code in (None, 0)


def test_status_panel_escapes_names() -> None:
    from textual.content import Content

    from tui.app import StatusPanel

    text = StatusPanel().render_status(
        running=False,
        model=Path("/m/v[b].onnx"),
        pitch=0,
        profile=_NAME,
        cold_start=False,
        swapping=None,
        error="profile 'x[/bold]' swap failed",
    )
    plain = Content.from_markup(text).plain
    assert _NAME in plain
    assert "v[b].onnx" in plain
    assert "profile 'x[/bold]' swap failed" in plain
