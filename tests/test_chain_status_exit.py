"""`woys chain status` must not exit 0 when it cannot read PipeWire state.

The --help exit-code table says an unready environment exits 2. Pre-fix a
failed `pactl list short modules` printed its error and the command still
returned 0, so a script checking the exit status saw a healthy chain.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from woys import chain


def _ok(*_a: object, **_k: object) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")


def test_status_exits_2_when_modules_cannot_be_listed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def _fail() -> list[tuple[str, str, str]]:
        raise chain.ChainError("pactl list short modules failed (rc=1)")

    monkeypatch.setattr(chain, "_list_modules", _fail)
    monkeypatch.setattr(chain, "_pactl", _ok)
    monkeypatch.setattr(chain, "_systemctl", _ok)
    monkeypatch.setattr(chain, "_alsa_leak_links", lambda: [])
    monkeypatch.setattr(chain, "_systemd_unit_path", lambda: tmp_path / "none.service")
    assert chain.status() == 2


def test_status_exits_0_when_state_is_readable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(chain, "_list_modules", lambda: [])
    monkeypatch.setattr(chain, "_pactl", _ok)
    monkeypatch.setattr(chain, "_systemctl", _ok)
    monkeypatch.setattr(chain, "_alsa_leak_links", lambda: [])
    monkeypatch.setattr(chain, "_systemd_unit_path", lambda: tmp_path / "none.service")
    assert chain.status() == 0
