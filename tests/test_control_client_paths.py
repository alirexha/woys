"""Control-socket client paths against real sockets and a real server.

Everything else that reaches send_command / submit_and_wait stubs them
with canned strings, so the stale/refused mapping, the connect retry,
the JOB error terminator and the job-table GC were never exercised.
"""

from __future__ import annotations

import socket
import time
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def sock_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "control.sock"
    monkeypatch.setattr("tui.control.control_socket_path", lambda: path)
    return path


def _patch_sleep(monkeypatch: pytest.MonkeyPatch, fake: Any) -> None:
    """Swap control.py's `time` for one whose sleep() is `fake`, leaving
    the real time module (and every other thread) alone."""
    from tui import control

    monkeypatch.setattr(
        control,
        "time",
        types.SimpleNamespace(sleep=fake, time=time.time, monotonic=time.monotonic),
    )


def test_dead_socket_file_maps_to_refused_after_retries(
    sock_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tui import control

    # A socket file left behind by a killed TUI: bound, never listening.
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(str(sock_path))
    s.close()
    sleeps: list[float] = []
    _patch_sleep(monkeypatch, sleeps.append)

    reply = control.send_command("STATUS", timeout=0.5)

    assert reply == "ERR control socket refused - TUI not accepting connections?"
    assert len(sleeps) == 3  # retried, with a pause each time


def test_socket_vanishing_before_connect_maps_to_stale(tmp_path: Path, monkeypatch: Any) -> None:
    from tui import control

    missing = tmp_path / "gone.sock"

    class _RacyPath:
        """exists() says yes, then the file is gone by connect()."""

        def exists(self) -> bool:
            return True

        def __str__(self) -> str:
            return str(missing)

    monkeypatch.setattr(control, "control_socket_path", _RacyPath)

    assert control.send_command("STATUS") == "ERR control socket stale - TUI not running?"


def test_no_socket_file_maps_to_not_found(sock_path: Path) -> None:
    from tui import control

    assert control.send_command("STATUS") == "ERR control socket not found - TUI not running?"


def test_connect_retry_reaches_a_server_that_comes_up_late(
    sock_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tui import control

    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(str(sock_path))  # file exists, nobody listening yet
    srv: list[Any] = []

    def sleep_then_start(_seconds: float) -> None:
        if not srv:
            s.close()
            sock_path.unlink()
            server = control.ControlServer(lambda cmd: f"OK {cmd}", path=sock_path)
            server.start()
            srv.append(server)

    _patch_sleep(monkeypatch, sleep_then_start)
    try:
        assert control.send_command("STATUS", timeout=2.0) == "OK STATUS"
    finally:
        for server in srv:
            server.stop()


@pytest.fixture
def job_server(sock_path: Path) -> Iterator[Any]:
    from tui.control import ControlServer, JobRegistry

    jobs = JobRegistry()

    def fail() -> None:
        raise RuntimeError("swap rejected")

    def handler(cmd: str) -> str:
        if cmd == "FAIL":
            return f"OK job={jobs.submit(fail)}"
        if cmd == "WORK":
            return f"OK job={jobs.submit(lambda: None)}"
        if cmd.startswith("JOB "):
            return jobs.status_line(cmd[len("JOB ") :])
        return "ERR unknown"

    srv = ControlServer(handler, path=sock_path)
    srv.start()
    try:
        yield jobs
    finally:
        srv.stop()


def test_submit_and_wait_stops_at_state_error(job_server: Any) -> None:
    from tui.control import submit_and_wait

    t0 = time.monotonic()
    reply = submit_and_wait("FAIL", overall_timeout=10.0)

    assert "state=error" in reply
    assert "RuntimeError: swap rejected" in reply
    assert time.monotonic() - t0 < 2.0  # not the 10 s timeout


def test_submit_and_wait_stops_at_state_done(job_server: Any) -> None:
    from tui.control import submit_and_wait

    assert "state=done" in submit_and_wait("WORK", overall_timeout=10.0)


def test_job_registry_gc_drops_finished_jobs() -> None:
    from tui.control import JobRegistry

    jobs = JobRegistry(ttl_seconds=0.0)
    jid = jobs.submit(lambda: None)
    deadline = time.monotonic() + 2.0
    while "state=done" not in jobs.status_line(jid):
        assert time.monotonic() < deadline
        time.sleep(0.01)
    time.sleep(0.01)

    jobs.submit(lambda: None)  # submit runs the GC

    assert jobs.status_line(jid) == "ERR unknown job"
