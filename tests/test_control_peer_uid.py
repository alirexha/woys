"""review F-05-01: SO_PEERCRED UID check on the control socket.

Pre-fix `ControlServer._loop` did `accept()` and dispatched the
command without checking the peer's credentials. The 0600 socket-
file perms + XDG_RUNTIME_DIR (mode 0700) closed CROSS-UID access,
but SAME-UID processes (a malicious pip dep in the same venv, a
game mod, a misbehaving script) could `connect()` and issue
TOGGLE / PITCH / MODEL / PROFILE / QUIT with real impact -- the
socket is reachable by design (the WM-shortcut control path).

Post-fix the accept-loop reads `SO_PEERCRED` and rejects any UID
that does not match the server's. The check is a regression guard
for the 0600 + XDG_RUNTIME_DIR layer; the filesystem permissions
remain the load-bearing cross-UID guard.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import os
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
    from tui.control import ControlServer

    srv = ControlServer(handler, path=sock_path)
    srv.start()
    for _ in range(20):
        if sock_path.exists():
            break
        time.sleep(0.01)
    return srv


def _send_and_recv(sock_path: Path, cmd: str, *, timeout: float = 2.0) -> str:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect(str(sock_path))
        s.sendall((cmd + "\n").encode("utf-8"))
        chunks: list[bytes] = []
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                chunk = s.recv(4096)
            except TimeoutError:
                break
            if not chunk:
                break
            chunks.append(chunk)
            if b"\n" in chunk:
                break
        return b"".join(chunks).decode("utf-8", errors="replace").strip()


def test_same_uid_connection_dispatches_normally(tmp_path: Path) -> None:
    """Sanity: a connection from the server's own UID dispatches as
    usual. This is the production case."""
    sock = tmp_path / "c.sock"
    received: list[str] = []

    def handler(cmd: str) -> str:
        received.append(cmd)
        return "OK ack"

    srv = _start_server(sock, handler)
    try:
        reply = _send_and_recv(sock, "TOGGLE")
        assert reply == "OK ack"
        assert received == ["TOGGLE"]
    finally:
        srv.stop()


def test_foreign_uid_connection_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bug-class test. Patch `os.getuid()` to return a UID that does
    NOT match the peer's (which is the real test-process UID, the
    same as the server -- but monkeypatched `getuid` reports a
    different one). The server compares peer-uid to
    `os.getuid()`; the mismatch triggers ERR unauthorized + no
    handler dispatch."""
    sock = tmp_path / "c.sock"
    handler_calls: list[str] = []

    def handler(cmd: str) -> str:
        handler_calls.append(cmd)
        return "OK ack"

    srv = _start_server(sock, handler)
    try:
        # Make the server's `os.getuid()` report a non-matching UID.
        # The peer's UID (read via SO_PEERCRED) is the real test-
        # process UID. The comparison `peer != fake_uid` triggers
        # the rejection.
        real_uid = os.getuid()
        fake_uid = real_uid + 9999

        monkeypatch.setattr("tui.control.os.getuid", lambda: fake_uid)

        reply = _send_and_recv(sock, "QUIT")
        assert "ERR unauthorized" in reply, (
            f"foreign-UID connection must be rejected; got: {reply!r}"
        )
        # The handler must NOT have been invoked.
        assert handler_calls == [], (
            f"foreign-UID dispatch must not reach the handler; got calls: {handler_calls}"
        )
    finally:
        srv.stop()


def test_check_peer_uid_returns_true_on_same_uid_pair() -> None:
    """Direct test of the helper. `_check_peer_uid` on a socketpair
    where both ends are the test process must return True."""
    from tui.control import _check_peer_uid

    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        assert _check_peer_uid(a) is True
        assert _check_peer_uid(b) is True
    finally:
        a.close()
        b.close()


def test_check_peer_uid_returns_false_when_uids_differ(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Force a UID mismatch by patching `os.getuid()` and assert the
    helper rejects."""
    from tui.control import _check_peer_uid

    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    real_uid = os.getuid()
    monkeypatch.setattr("tui.control.os.getuid", lambda: real_uid + 7777)
    try:
        assert _check_peer_uid(a) is False
    finally:
        a.close()
        b.close()


def test_check_peer_uid_accepts_when_so_peercred_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defensive: very old kernels / non-Linux systems may not
    support SO_PEERCRED. We log a warning and accept (the
    filesystem perms remain the load-bearing guard). Pin the
    behavior so a future change is intentional."""
    from tui.control import _check_peer_uid

    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)

    class _BoomSocket:
        def getsockopt(self, *_a: object, **_kw: object) -> bytes:
            raise OSError("simulated kernel without SO_PEERCRED")

    try:
        assert _check_peer_uid(_BoomSocket()) is True  # type: ignore[arg-type]
    finally:
        a.close()
        b.close()
