"""Client side of the control socket: every failure is an ERR line.

`send_command` promises a clear ERR string instead of an exception; the
CLI turns ERR into exit code 1. A traceback breaks that contract.
"""

from __future__ import annotations

import socket
import types
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture
def sock_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "control.sock"
    monkeypatch.setattr("tui.control.control_socket_path", lambda: path)
    return path


@pytest.fixture
def silent_server(sock_path: Path) -> Iterator[socket.socket]:
    """Listens (so connect succeeds) but never answers."""
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(sock_path))
    srv.listen(4)
    try:
        yield srv
    finally:
        srv.close()


def test_send_command_timeout_is_an_err_line(silent_server: socket.socket) -> None:
    from tui.control import send_command

    reply = send_command("STATUS", timeout=0.2)

    assert reply.startswith("ERR"), reply
    assert "timed out" in reply.lower() or "timeout" in reply.lower()
    # Not one of the "TUI not running" strings the CLI persists on.
    assert not reply.startswith("ERR control socket")


def test_send_command_other_socket_errors_are_err_lines(
    sock_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tui import control

    sock_path.write_text("")

    class _Sock:
        def __init__(self, *_a: object) -> None:
            pass

        def __enter__(self) -> _Sock:
            return self

        def __exit__(self, *_a: object) -> None:
            pass

        def settimeout(self, _t: float) -> None:
            pass

        def connect(self, _p: str) -> None:
            raise PermissionError(13, "Permission denied")

    fake_socket_mod = types.SimpleNamespace(
        socket=_Sock, AF_UNIX=socket.AF_UNIX, SOCK_STREAM=socket.SOCK_STREAM
    )
    monkeypatch.setattr(control, "socket", fake_socket_mod)

    reply = control.send_command("STATUS", timeout=0.2)

    assert reply.startswith("ERR"), reply
    assert "Permission denied" in reply
