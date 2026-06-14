"""socket-routed CLI commands (toggle / status /
pitch / slow) must return non-zero on `ERR ...` replies, route them to
stderr, and document the convention.

Pre-fix all four commands `print(send_command(...))` then `return 0` —
so `woys toggle && notify-send "on"` would notify even when the toggle
silently failed because the TUI wasn't running. The Unix socket exists
explicitly for WM-shortcut scripting; an exit code that lies about
success is a silent-failure on the tool's primary scripting contract.

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


def _patch_send_command(monkeypatch: pytest.MonkeyPatch, reply: str) -> list[str]:
    """Patch `tui.control.send_command` to return `reply`. Returns the list
    that captures every cmd-string passed in (so the test can assert which
    socket cmd was sent)."""
    sent: list[str] = []

    def _fake(cmd: str, timeout: float = 30.0) -> str:
        sent.append(cmd)
        return reply

    from tui import control

    monkeypatch.setattr(control, "send_command", _fake)
    return sent


@pytest.mark.parametrize(
    "cmd, sock_cmd",
    [("toggle", "TOGGLE"), ("status", "STATUS"), ("slow", "SLOW")],
)
def test_socket_command_returns_zero_on_ok(
    cmd: str,
    sock_cmd: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An OK reply: exit 0, reply on stdout, stderr clean."""
    from woys import cli

    sent = _patch_send_command(monkeypatch, "OK toggled")
    rc = cli.main([cmd])
    out = capsys.readouterr()

    assert rc == 0, f"OK reply must produce exit 0, got {rc}"
    assert sent == [sock_cmd], f"expected one {sock_cmd} send, got {sent}"
    assert "OK toggled" in out.out
    assert "OK toggled" not in out.err


@pytest.mark.parametrize("cmd", ["toggle", "status", "slow"])
def test_socket_command_returns_one_on_err(
    cmd: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The bug-class test. An ERR reply must produce exit 1, route the
    message to stderr, and *not* print it on stdout (otherwise piping
    `woys status | grep running` would silently match on the error string)."""
    from woys import cli

    err_reply = "ERR control socket not found - TUI not running?"
    _patch_send_command(monkeypatch, err_reply)

    rc = cli.main([cmd])
    out = capsys.readouterr()

    assert rc == 1, f"ERR reply must produce exit 1, got {rc}"
    assert err_reply in out.err, "the ERR message must reach stderr"
    assert err_reply not in out.out, (
        "the ERR message must not leak onto stdout — that's the WM-script-"
        "scripting contract the fix enforces"
    )


def test_pitch_command_exit_codes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`woys pitch +2` mirrors the toggle/status convention; the existing
    `return 2` on a non-integer argument is preserved (argparse-style
    usage error, distinct from server-ERR)."""
    from woys import cli

    sent = _patch_send_command(monkeypatch, "OK pitch=+2")
    rc = cli.main(["pitch", "+2"])
    out = capsys.readouterr()
    assert rc == 0
    assert sent == ["PITCH 2"]
    assert "OK pitch=+2" in out.out

    _patch_send_command(monkeypatch, "ERR bad pitch: 'abc'")
    rc = cli.main(["pitch", "abc"])
    out = capsys.readouterr()
    assert rc == 2, "non-integer delta is a usage error -> exit 2"
    assert "pitch must be an integer" in out.err


def test_help_epilog_documents_exit_codes() -> None:
    """The exit-code convention is documented in the parser's epilog so
    a user discovering the contract via `woys --help` finds it."""
    from woys.cli import build_parser

    epilog = build_parser().format_help()

    assert "Exit codes:" in epilog
    assert "0" in epilog and "1" in epilog and "2" in epilog
    assert "ERR" in epilog, "must mention that ERR replies map to exit 1"
