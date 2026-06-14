"""newline-terminated framing on
both sides of the control socket.

Pre-fix:
- Server (`tui/control.py:238`): `conn.recv(256)` — a single fixed-size
  recv. Any command longer than 256 B was silently truncated.
- Client (`tui/control.py:278`): `s.recv(512)` — same problem, on the
  reply side. The STATUS reply was already ~250 B and growing; the
  review noted truncation was a *when*, not *if*.

The docstring at the top of `control.py` promised
"one newline-terminated command per connection" / "single short reply
line", but the implementation didn't honor it.

Post-fix `_recv_line` (server) and `_recv_reply` (client) loop on recv
until a `\\n` byte or a configured MAX cap. The MAX caps return clear
error states ("ERR command too long" server-side; "...(truncated)"
suffix client-side) so the failure mode is visible instead of silent.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import contextlib
import socket
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "src" / "server") not in sys.path:
    sys.path.insert(0, str(REPO / "src" / "server"))


def _start_server(sock_path: Path, handler):  # type: ignore[no-untyped-def]
    """Spin up a ControlServer on `sock_path` with the given handler.
    Returns the server (caller must call .stop())."""
    from tui.control import ControlServer

    srv = ControlServer(handler, path=sock_path)
    srv.start()
    # Give the listen-accept thread a moment.
    for _ in range(20):
        if sock_path.exists():
            break
        time.sleep(0.01)
    return srv


def _direct_send(sock_path: Path, cmd: str, *, timeout: float = 2.0) -> str:
    """Send `cmd` directly (no `send_command` wrapper) and read reply
    via socket.recv(MAX) at MAX_REPLY_BYTES + slack so we can detect
    truncation independently of the client-side framing fix.
    """
    from tui.control import MAX_REPLY_BYTES

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect(str(sock_path))
        s.sendall((cmd + "\n").encode("utf-8"))
        chunks: list[bytes] = []
        # Read until peer closes or we get a newline. Bound the read so
        # a buggy server can't hang the test.
        s.settimeout(timeout)
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                chunk = s.recv(min(8192, MAX_REPLY_BYTES + 1024))
            except TimeoutError:
                break
            if not chunk:
                break
            chunks.append(chunk)
            if b"\n" in chunk:
                break
        return b"".join(chunks).decode("utf-8", errors="replace").strip("\n")


# --- server-side framing (commands) ---------------------------------------


def test_server_receives_full_command_longer_than_256_bytes(tmp_path: Path) -> None:
    """Bug-class half-A: a command > 256 bytes used to be silently
    truncated by the pre-fix `conn.recv(256)`. The server-side handler
    must now see the FULL command."""
    sock = tmp_path / "c.sock"
    received: list[str] = []

    def handler(cmd: str) -> str:
        received.append(cmd)
        return f"OK len={len(cmd)}"

    srv = _start_server(sock, handler)
    try:
        # 1024-byte command. Use a body without newlines.
        body = "X" * 1018
        cmd = f"PITCH {body}"  # 1024 chars
        assert len(cmd) > 256
        reply = _direct_send(sock, cmd)
        assert reply.startswith("OK"), f"expected OK reply; got: {reply!r}"
        assert received and len(received[0]) == 1024, (
            f"server should have received the full 1024-byte cmd; "
            f"got {len(received[0]) if received else 'nothing'}"
        )
    finally:
        srv.stop()


def test_server_rejects_command_over_max_bytes_with_clear_error(
    tmp_path: Path,
) -> None:
    """A pathologically large command (no newline within MAX_COMMAND_BYTES)
    must get a clean `ERR command too long` reply -- not a hang, not a
    silent truncate, not a stack trace in the server log."""
    from tui.control import MAX_COMMAND_BYTES

    sock = tmp_path / "c.sock"
    received: list[str] = []

    def handler(cmd: str) -> str:
        received.append(cmd)
        return "OK"

    srv = _start_server(sock, handler)
    try:
        # Send MAX_COMMAND_BYTES + 1 bytes without a newline. The server
        # should see the recv loop overflow and reply with the error.
        # _direct_send appends a "\n" at the end -- bypass that by
        # sending the raw bytes inline.
        oversized = b"A" * (MAX_COMMAND_BYTES + 4)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(5.0)
            s.connect(str(sock))
            with contextlib.suppress(BrokenPipeError):
                # The server may close before we finish sending; that's
                # fine -- we'll still get the error reply if it landed.
                s.sendall(oversized)
            chunks: list[bytes] = []
            try:
                while True:
                    chunk = s.recv(4096)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    if b"\n" in chunk:
                        break
            except (TimeoutError, OSError):
                pass
            reply = b"".join(chunks).decode("utf-8", errors="replace").strip()

        assert "ERR command too long" in reply, (
            f"oversized command must produce the framing-overflow error; got: {reply!r}"
        )
        assert not received, (
            "handler must NOT be invoked when the command overflows -- "
            "it has not been fully received and is unparseable"
        )
    finally:
        srv.stop()


# --- client-side framing (replies) ----------------------------------------


def test_client_receives_full_reply_longer_than_512_bytes(tmp_path: Path) -> None:
    """Bug-class half-B: server emits a reply > 512 bytes. The pre-fix
    `s.recv(512)` truncated it; the client must now see the FULL reply."""
    from tui.control import send_command

    sock = tmp_path / "c.sock"

    def handler(_cmd: str) -> str:
        return "OK " + ("Y" * 2000)  # 2003 byte reply

    srv = _start_server(sock, handler)
    try:
        # send_command uses the canonical XDG_RUNTIME_DIR path, so we
        # monkeypatch by way of `control_socket_path`. Simpler: call
        # send_command after pointing the env var.
        import os

        os.environ["XDG_RUNTIME_DIR"] = str(tmp_path)
        # send_command resolves $XDG_RUNTIME_DIR/woys/control.sock; our
        # server is at tmp_path/c.sock, so move the server-side path.
        srv.stop()
        sock = tmp_path / "woys" / "control.sock"
        sock.parent.mkdir(parents=True, exist_ok=True)
        srv = _start_server(sock, handler)

        reply = send_command("STATUS", timeout=3.0)
        assert reply.startswith("OK")
        # 2000 'Y's plus "OK " = 2003 chars
        y_count = reply.count("Y")
        assert y_count == 2000, (
            f"client must receive the full 2003-byte reply; got "
            f"{y_count} Y characters (truncation expected pre-fix at "
            f"~509 Y's after 'OK '). reply len: {len(reply)}"
        )
    finally:
        srv.stop()


# --- helper unit tests on _recv_line / _recv_reply directly ---------------


class _FakeSocket:
    """Stand-in for socket.socket that returns pre-baked chunks. Lets
    us exercise the framing helpers' edge cases without spinning up a
    real server."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)

    def recv(self, _n: int) -> bytes:
        if not self._chunks:
            return b""
        return self._chunks.pop(0)


def test_recv_line_returns_decoded_line_stripped_of_newline() -> None:
    from tui.control import _recv_line

    fake = _FakeSocket([b"PITCH +2\n"])
    line = _recv_line(fake)  # type: ignore[arg-type]
    assert line == "PITCH +2"


def test_recv_line_handles_split_chunks() -> None:
    """A command that arrives in two recv() calls must be reassembled."""
    from tui.control import _recv_line

    fake = _FakeSocket([b"PITCH ", b"+", b"22\n"])
    line = _recv_line(fake)  # type: ignore[arg-type]
    assert line == "PITCH +22"


def test_recv_line_returns_none_on_immediate_eof() -> None:
    from tui.control import _recv_line

    fake = _FakeSocket([])
    assert _recv_line(fake) is None  # type: ignore[arg-type]


def test_recv_line_raises_when_max_bytes_exceeded() -> None:
    """No newline within max_bytes -> ValueError. The server's _loop
    catches this and returns 'ERR command too long'."""
    from tui.control import _recv_line

    fake = _FakeSocket([b"A" * 16, b"A" * 16, b"A" * 16])
    with pytest.raises(ValueError, match=r"exceeded.*bytes"):
        _recv_line(fake, max_bytes=32)  # type: ignore[arg-type]


def test_recv_reply_returns_full_line_across_chunks() -> None:
    from tui.control import _recv_reply

    fake = _FakeSocket([b"OK ", b"running=True ", b"pitch=0\n"])
    reply = _recv_reply(fake)  # type: ignore[arg-type]
    assert reply == "OK running=True pitch=0"


def test_recv_reply_marks_truncation_when_max_exceeded() -> None:
    """If the server emits a > MAX_REPLY_BYTES reply without a newline,
    the client returns what it has plus a clear ' ... (truncated)'
    suffix so the caller knows the reply is incomplete."""
    from tui.control import _recv_reply

    huge = b"X" * 100
    fake = _FakeSocket([huge])
    reply = _recv_reply(fake, max_bytes=40)  # type: ignore[arg-type]
    assert reply.endswith("... (truncated)"), (
        f"truncated reply must carry the marker; got: {reply!r}"
    )


def test_recv_reply_returns_what_it_has_on_eof_without_newline() -> None:
    """Edge case: the server crashed mid-reply (or wrote without a
    trailing newline). The client returns what arrived rather than
    hanging."""
    from tui.control import _recv_reply

    fake = _FakeSocket([b"partial-reply"])  # no newline
    reply = _recv_reply(fake)  # type: ignore[arg-type]
    assert reply == "partial-reply"
