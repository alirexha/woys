"""broad except around the
control handler so an unexpected exception class doesn't silently
kill the listener thread.

Pre-fix `ControlServer._loop` caught only
`(TimeoutError, OSError)` around the conn-handling block. If the
handler raised anything else (e.g., `MODEL <slug>` → `find_by_name`
→ `ValueError`, or any future protocol-extension exception), the
exception propagated out of the with-conn block, was NOT caught by
the outer except, and silently killed the `woys-control` listener
thread. Every subsequent `woys toggle` then hung to its client
timeout (~30 s).

Post-fix any `Exception` from the handler routes to an
`ERR internal: <ClsName>: <msg>` reply so the client sees the
failure and the listener thread keeps running.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import socket
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:
    sys.path.append(str(REPO / "src"))
if str(REPO / "src" / "server") not in sys.path:
    sys.path.append(str(REPO / "src" / "server"))


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


def test_handler_raises_value_error_routes_to_err_internal(tmp_path: Path) -> None:
    """Bug-class test. A handler that raises ValueError (a real
    in-the-wild example: MODEL <unknown-slug> -> find_by_name() ->
    ValueError) returns ERR internal to the client. Pre-fix the
    listener thread died silently."""
    sock = tmp_path / "c.sock"

    def boom_handler(cmd: str) -> str:
        if cmd.startswith("BOOM"):
            raise ValueError("simulated handler failure")
        return "OK"

    srv = _start_server(sock, boom_handler)
    try:
        reply = _send_and_recv(sock, "BOOM")
        assert "ERR internal" in reply, (
            f"F-CX6-01: a handler exception must produce an ERR internal "
            f"reply (the client gets a failure signal); got: {reply!r}"
        )
        assert "ValueError" in reply
        assert "simulated handler failure" in reply
    finally:
        srv.stop()


def test_listener_thread_survives_handler_exception(tmp_path: Path) -> None:
    """Bug-class test. After a handler exception, the listener
    thread must still be alive and a subsequent connection must
    succeed. Pre-fix the thread died -- subsequent `woys toggle`
    hung."""
    sock = tmp_path / "c.sock"
    counter = {"n": 0}

    def flaky_handler(cmd: str) -> str:
        counter["n"] += 1
        if counter["n"] == 1:
            raise RuntimeError("first call crashes")
        return f"OK count={counter['n']}"

    srv = _start_server(sock, flaky_handler)
    try:
        # First call: handler raises; ERR internal reply.
        reply1 = _send_and_recv(sock, "PING1")
        assert "ERR internal" in reply1
        # Second call: handler runs normally. Pre-fix the listener
        # was dead and this connect would hang to timeout.
        reply2 = _send_and_recv(sock, "PING2", timeout=2.0)
        assert reply2 == "OK count=2", f"listener must survive the exception; got {reply2!r}"
    finally:
        srv.stop()


def test_normal_handler_path_unchanged(tmp_path: Path) -> None:
    """Back-compat: a handler that returns normally is dispatched
    + reply sent + listener continues."""
    sock = tmp_path / "c.sock"

    def ok_handler(cmd: str) -> str:
        return "OK normal"

    srv = _start_server(sock, ok_handler)
    try:
        reply = _send_and_recv(sock, "TOGGLE")
        assert reply == "OK normal"
    finally:
        srv.stop()
