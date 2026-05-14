"""review F-08-09 / F-23-03 (P1): the TUI's `_refresh_stats` must not
swallow engine errors.

Pre-fix `_refresh_stats` wrapped its *entire* body -- including the
`last_error` -> toast escalation -- in a bare `except Exception: pass`,
justified only for the startup window before the widget tree is realized.
So any widget-render hiccup silently swallowed engine errors, and the bare
`pass` ran forever with no log and no counter -- a silent-fallback in the
observability surface itself.

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


def _raise(*_a: object, **_k: object) -> object:
    raise RuntimeError("widget tree not realized")


def test_refresh_stats_toast_fires_even_when_widget_render_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bug-class test: the `last_error` -> toast escalation must fire
    even when `query_one` raises. Pre-fix the toast block sat inside the
    blanket `try`, so a widget hiccup swallowed it -> `notes` stays empty.
    """
    from tui.app import WoysApp
    from tui.config import AppConfig

    app = WoysApp(cfg=AppConfig(), no_pw_setup=True)
    app.engine.stats.last_error = "engine boom 999"
    monkeypatch.setattr(app, "query_one", _raise)  # widget tree "not realized"
    notes: list[tuple[object, ...]] = []
    monkeypatch.setattr(app, "notify", lambda *a, **k: notes.append((a, k)))

    app._refresh_stats()

    assert notes, "the last_error toast must fire even when widget render fails"
    assert "engine boom 999" in str(notes[0])


def test_refresh_stats_counts_render_errors_after_startup_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After the startup window a render failure is logged + counted --
    never a bare `pass`."""
    from tui.app import _REFRESH_STARTUP_TICKS, WoysApp
    from tui.config import AppConfig

    app = WoysApp(cfg=AppConfig(), no_pw_setup=True)
    monkeypatch.setattr(app, "query_one", _raise)
    monkeypatch.setattr(app, "notify", lambda *a, **k: None)

    app._refresh_ticks = _REFRESH_STARTUP_TICKS  # next tick is past the window
    app._refresh_stats()

    assert app._refresh_errors == 1, "a post-startup render failure must be counted"


def test_refresh_stats_silent_during_startup_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Within the startup window a render failure stays silent -- the
    widget tree genuinely isn't realized yet."""
    from tui.app import WoysApp
    from tui.config import AppConfig

    app = WoysApp(cfg=AppConfig(), no_pw_setup=True)
    monkeypatch.setattr(app, "query_one", _raise)
    monkeypatch.setattr(app, "notify", lambda *a, **k: None)

    app._refresh_ticks = 0  # first tick -> inside the startup window
    app._refresh_stats()

    assert app._refresh_errors == 0
