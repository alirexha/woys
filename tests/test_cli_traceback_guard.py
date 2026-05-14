"""review F-merged-022 (P1): the most common first-run failures must
reach a non-developer as one actionable line, not a raw Python traceback or
a crashed TUI mount.

Two surfaces:
  * `woys.cli.main()` wraps command dispatch in a top-level guard.
  * `tui.app.WoysApp._start_engine()` catches `engine.start()` failures and
    surfaces them via `notify()` + `stats.last_error` instead of crashing.

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


def test_main_guards_operational_errors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The bug-class test. An operational error from command dispatch must
    become one actionable stderr line + exit 1.

    Pre-fix `main()` had no guard, so the `FileNotFoundError` propagates out
    of `main()` and `rc = cli.main(...)` raises -- this test errors.
    """
    from woys import cli

    def _boom() -> int:
        raise FileNotFoundError("rvc model not found at /models/missing.onnx")

    monkeypatch.setattr(cli, "cmd_info", _boom)
    monkeypatch.delenv("WOYS_DEBUG", raising=False)

    rc = cli.main(["info"])
    err = capsys.readouterr().err

    assert rc == 1
    assert "error:" in err
    assert "missing.onnx" in err, "the message must name the missing resource"
    assert "models download" in err or "config.toml" in err, "and name a remedy"
    assert "Traceback" not in err, "a non-developer must not see a raw traceback"


def test_main_woys_debug_preserves_traceback(monkeypatch: pytest.MonkeyPatch) -> None:
    """WOYS_DEBUG=1 re-raises so developers still get the traceback. (Also
    passes pre-fix, where everything propagates -- this confirms the guard
    is *conditional*, paired with the bug-class test above.)"""
    from woys import cli

    def _boom() -> int:
        raise RuntimeError("kaboom")

    monkeypatch.setattr(cli, "cmd_info", _boom)
    monkeypatch.setenv("WOYS_DEBUG", "1")

    with pytest.raises(RuntimeError, match="kaboom"):
        cli.main(["info"])


def test_main_does_not_swallow_bug_class_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard catches *operational* errors only. A bug-class exception
    (KeyError/TypeError/...) must still surface as a traceback so it gets
    reported and fixed -- not be silently turned into exit 1."""
    from woys import cli

    def _boom() -> int:
        raise KeyError("internal bug")

    monkeypatch.setattr(cli, "cmd_info", _boom)
    monkeypatch.delenv("WOYS_DEBUG", raising=False)

    with pytest.raises(KeyError):
        cli.main(["info"])


def test_start_engine_surfaces_failure_instead_of_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`_start_engine` must catch an `engine.start()` failure, record
    `stats.last_error`, `notify()`, and return False -- not let the
    exception crash the TUI mount.

    Pre-fix `_start_engine` returned None and did not catch, so
    `app._start_engine()` raises here -- this test errors.
    """
    from tui.app import WoysApp
    from tui.config import AppConfig

    app = WoysApp(cfg=AppConfig(), no_pw_setup=True)

    def _raise() -> None:
        raise FileNotFoundError("rvc model not found at /models/x.onnx")

    monkeypatch.setattr(app.engine, "start", _raise)
    notes: list[tuple[object, ...]] = []
    monkeypatch.setattr(app, "notify", lambda *a, **k: notes.append((a, k)))

    ok = app._start_engine()

    assert ok is False, "_start_engine must report failure, not None/raise"
    assert "x.onnx" in (app.engine.stats.last_error or "")
    assert notes, "the failure must be surfaced to the user via notify()"
