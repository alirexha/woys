"""Unix-socket control channel round-trips."""

from __future__ import annotations

import time
from pathlib import Path

from tui.control import ControlServer, control_socket_path, send_command


def test_socket_path_lives_in_runtime_dir(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/tmp")
    p = control_socket_path()
    assert str(p).startswith("/tmp")
    assert p.name == "control.sock"


def test_socket_path_fallback_when_xdg_unset(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """B52 / test-011: when XDG_RUNTIME_DIR is unset (locked-down CI / minimal
    Linux setups), `control_socket_path` must fall back to a per-uid path
    under /tmp instead of crashing."""
    import os

    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    p = control_socket_path()
    assert p.name == "control.sock"
    assert str(p).startswith("/tmp/woys-")
    assert str(os.getuid()) in str(p)
    # And `runtime_path()` should resolve under the same fallback root.
    from tui.control import runtime_path

    rp = runtime_path("slow-chunks.txt")
    assert rp.name == "slow-chunks.txt"
    assert str(rp).startswith("/tmp/woys-")


def test_round_trip_toggle_and_pitch(tmp_path: Path) -> None:
    state = {"running": False, "pitch": 0}

    def handler(cmd: str) -> str:
        if cmd == "TOGGLE":
            state["running"] = not state["running"]
            return f"OK toggled (running={state['running']})"
        if cmd.startswith("PITCH "):
            try:
                state["pitch"] += int(cmd.split(maxsplit=1)[1])
            except (IndexError, ValueError):
                return "ERR"
            return f"OK pitch={state['pitch']}"
        if cmd == "STATUS":
            return f"OK r={state['running']} p={state['pitch']}"
        return f"ERR unknown: {cmd!r}"

    sock_path = tmp_path / "test.sock"
    srv = ControlServer(handler, path=sock_path)
    srv.start()
    try:
        # Wait for the listener to bind.
        for _ in range(20):
            if sock_path.exists():
                break
            time.sleep(0.05)
        assert sock_path.exists()

        # Use a direct socket since send_command() resolves a different default path.
        import socket

        def call(cmd: str) -> str:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.settimeout(1.0)
                s.connect(str(sock_path))
                s.sendall((cmd + "\n").encode())
                return s.recv(256).decode().strip()

        assert "OK toggled" in call("TOGGLE")
        assert "OK pitch=2" in call("PITCH 2")
        assert "OK pitch=1" in call("PITCH -1")
        st = call("STATUS")
        assert "r=True" in st and "p=1" in st
    finally:
        srv.stop()
    assert not sock_path.exists()


def test_send_command_when_no_server(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # Ensure the helper fails gracefully if no server is running. Pin the socket
    # path to a tmp dir so the test is deterministic even when a real woys
    # engine is live on the host (otherwise the test connects to it).
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    out = send_command("STATUS", timeout=0.2)
    assert "ERR" in out or "not found" in out
