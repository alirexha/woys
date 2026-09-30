"""`woys chain disable` must not delete the unit when systemctl failed.

It used to ignore the `systemctl --user disable --now` result, delete the
unit file anyway and print "disabled + removed" with exit 0 -- over SSH
with no user bus that left the service running and a dangling
default.target.wants symlink with no unit file behind it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from woys import chain


def _cp(rc: int, stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=rc, stdout="", stderr=stderr)


class _Run:
    def __init__(self, fail: str | None) -> None:
        self.fail = fail
        self.calls: list[list[str]] = []

    def __call__(self, cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(cmd))
        if cmd[0] == "systemctl" and self.fail is not None and self.fail in cmd:
            return _cp(1, "Failed to connect to bus: No medium found")
        return _cp(0)


@pytest.fixture
def unit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr(chain.shutil, "which", lambda name: f"/usr/bin/{name}")
    path = chain._systemd_unit_path()
    path.parent.mkdir(parents=True)
    path.write_text("[Unit]\n")
    return path


def test_disable_keeps_unit_when_systemctl_disable_fails(
    unit: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(chain.subprocess, "run", _Run(fail="disable"))
    rc = chain.disable()
    out = capsys.readouterr()
    assert rc == 2
    assert unit.is_file()
    assert "disabled + removed" not in out.out
    assert "No medium found" in out.err


def test_disable_fails_when_daemon_reload_fails(
    unit: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(chain.subprocess, "run", _Run(fail="daemon-reload"))
    assert chain.disable() == 2
    assert not unit.exists()


def test_disable_removes_unit_on_success(unit: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chain.subprocess, "run", _Run(fail=None))
    assert chain.disable() == 0
    assert not unit.exists()
