"""`woys chain status --check` must fail when it cannot run its probes.

The check is the ExecStartPost of woys-mic.service and woys-chain.service.
It used to be fail-open: a failed `pactl get-default-sink` printed
"default sink '(unknown)' ... OK", and a missing or failing pw-link
printed "no ALSA leak links OK", so the unit went green without checking
the two things it exists to check.
"""

from __future__ import annotations

import subprocess
from typing import Any

import pytest

from woys import chain


def _healthy(monkeypatch: pytest.MonkeyPatch, *, stub_leaks: bool = True) -> None:
    monkeypatch.setattr(chain, "_source_present", lambda name: True)
    monkeypatch.setattr(chain, "_default_sink", lambda: "alsa_output.real-speakers")
    if stub_leaks:
        monkeypatch.setattr(chain, "_alsa_leak_links", lambda: [])


def test_check_fails_when_default_sink_is_unreadable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _healthy(monkeypatch)
    monkeypatch.setattr(chain, "_default_sink", lambda: "")
    assert chain.status(check=True) == 1
    assert "(unknown)' is not a woys null-sink  OK" not in capsys.readouterr().out


def test_check_fails_when_pw_link_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    _healthy(monkeypatch, stub_leaks=False)
    monkeypatch.setattr(chain.shutil, "which", lambda name: None)
    assert chain.status(check=True) == 1


def test_check_fails_when_pw_link_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    _healthy(monkeypatch, stub_leaks=False)
    monkeypatch.setattr(chain.shutil, "which", lambda name: "/usr/bin/pw-link")

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 1, "", "failed to connect: Host is down")

    monkeypatch.setattr(chain.subprocess, "run", fake_run)
    assert chain.status(check=True) == 1


def test_leak_probe_raises_when_pw_link_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chain.shutil, "which", lambda name: "/usr/bin/pw-link")

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 1, "", "failed to connect")

    monkeypatch.setattr(chain.subprocess, "run", fake_run)
    with pytest.raises(chain.ChainError, match="failed to connect"):
        chain._alsa_leak_links()
