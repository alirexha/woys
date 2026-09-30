"""`woys pw setup --rate/--channels` must not be silently ignored.

`VirtualMic.ensure()` returns early when woys-mic is already loaded, so
`woys pw setup --rate 44100` on an existing 48 kHz sink printed "woys-mic
ready" and exited 0 without applying anything. Nonsense values (`--rate
0`) also went straight to pactl.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import pytest

import audio.pipewire as pw
import woys.cli as cli

_LOADED = pw.VirtualMicState(True, True, 11, 12)


def _already_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pw, "get_state", lambda: _LOADED)
    monkeypatch.setattr(pw.VirtualMic, "ensure", lambda self: _LOADED)


def test_explicit_rate_on_a_loaded_sink_warns(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _already_loaded(monkeypatch)
    cli.cmd_pw_setup(44_100, None)
    err = capsys.readouterr().err
    assert "already loaded" in err
    assert "woys pw teardown" in err


def test_plain_setup_on_a_loaded_sink_stays_quiet(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _already_loaded(monkeypatch)
    assert cli.cmd_pw_setup(None, None) == 0
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize(
    "argv",
    [
        ["pw", "setup", "--rate", "0"],
        ["pw", "setup", "--rate", "-48000"],
        ["pw", "setup", "--channels", "0"],
        ["pw", "setup", "--channels", "99"],
    ],
)
def test_out_of_range_values_are_usage_errors(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.build_parser().parse_args(argv)
    assert exc.value.code == 2
