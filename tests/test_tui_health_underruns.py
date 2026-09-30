"""The TUI audio-health row counts native-pw player underruns.

The default backend (native pw helper) reports its dropouts in
`stats.player_underruns`; `stats.xruns` only moves on the pacat
backend. The row read xruns alone and stayed green while the helper
underran.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_render_lat_flags_player_underruns() -> None:
    from tui.app import LatencyPanel

    text = LatencyPanel().render_lat(10.0, 5.0, 100, xruns=0, underruns=3)

    assert "underruns=3" in text
    assert "[red]" in text


def test_render_lat_green_when_clean() -> None:
    from tui.app import LatencyPanel

    text = LatencyPanel().render_lat(10.0, 5.0, 100)

    assert "[green]" in text and "[red]" not in text


@pytest.mark.asyncio
async def test_refresh_shows_player_underruns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tui.app import LatencyPanel, WoysApp
    from tui.config import AppConfig

    runtime = tmp_path / "run"
    runtime.mkdir(mode=0o700)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    app = WoysApp(cfg=AppConfig(autostart_engine=False), no_pw_setup=True)
    app.engine.stats.player_underruns = 7
    async with app.run_test() as pilot:
        await pilot.pause(0.4)
        app._refresh_stats()
        panel = app.query_one("#latency", LatencyPanel)
        assert "underruns=7" in str(panel.render())
