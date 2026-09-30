"""`woys run --no-monitor` turns self-monitor off for a session.

There was only `--monitor`, so a user whose config has monitor on had no
way to launch one session without hearing themselves.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import tui.app
import woys.cli as cli
import woys.logsetup as logsetup


@pytest.mark.parametrize(
    ("flags", "expected"),
    [([], None), (["--monitor"], True), (["--no-monitor"], False)],
)
def test_run_forwards_the_monitor_choice(
    flags: list[str],
    expected: bool | None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seen: dict[str, Any] = {}

    def fake_run_tui(**kwargs: Any) -> int:
        seen.update(kwargs)
        return 0

    monkeypatch.setattr(tui.app, "run_tui", fake_run_tui)
    monkeypatch.setattr(logsetup, "setup_logging", lambda: tmp_path / "woys.log")
    assert cli.main(["run", *flags]) == 0
    assert seen["monitor"] is expected
