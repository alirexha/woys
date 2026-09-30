"""The SLOW dump refuses a symlink planted at its path and writes 0600."""

from __future__ import annotations

import stat
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    from tui.app import WoysApp
    from tui.config import AppConfig

    runtime = tmp_path / "run"
    runtime.mkdir(mode=0o700)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    return WoysApp(cfg=AppConfig(), no_pw_setup=True)


def test_slow_dump_does_not_follow_a_symlink(app: Any, tmp_path: Path) -> None:
    from tui.control import runtime_path

    victim = tmp_path / "victim.txt"
    victim.write_text("keep me\n")
    runtime_path("slow-chunks.txt").symlink_to(victim)

    reply = app._handle_control("SLOW")

    assert reply.startswith("ERR"), reply
    assert victim.read_text() == "keep me\n"


def test_slow_dump_is_private(app: Any) -> None:
    from tui.control import runtime_path

    reply = app._handle_control("SLOW")

    assert reply.startswith("OK wrote 0 entries"), reply
    out = runtime_path("slow-chunks.txt")
    assert stat.S_IMODE(out.stat().st_mode) == 0o600
    assert out.read_text().startswith("# slow chunk log")
