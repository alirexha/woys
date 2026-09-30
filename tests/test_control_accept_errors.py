"""A transient accept() error must not kill the control listener.

Pre-fix any OSError from accept() (EMFILE under fd pressure,
ECONNABORTED) ended the listener loop without a log line. The socket
file stayed, so every later `woys ...` call hung or was told the TUI
wasn't running while it was.
"""

from __future__ import annotations

import errno
import logging
import socket
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest


class _FlakyListener:
    """Wraps the real listening socket; the first accept() fails."""

    def __init__(self, real: socket.socket, err: int) -> None:
        self._real = real
        self._err = err
        self.failed = False

    def accept(self) -> Any:
        if not self.failed:
            self.failed = True
            raise OSError(self._err, "simulated accept failure")
        return self._real.accept()

    def close(self) -> None:
        self._real.close()


@pytest.fixture
def records() -> Iterator[list[logging.LogRecord]]:
    """Records from woys.control, captured on the logger itself: logsetup
    turns off propagation for the woys namespace, so caplog can miss them."""
    got: list[logging.LogRecord] = []

    class _Keep(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            got.append(record)

    log = logging.getLogger("woys.control")
    handler = _Keep(level=logging.WARNING)
    prior = log.level
    log.addHandler(handler)
    log.setLevel(logging.WARNING)
    try:
        yield got
    finally:
        log.removeHandler(handler)
        log.setLevel(prior)


def _call(path: Path, cmd: str) -> str:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as c:
        c.settimeout(3.0)
        c.connect(str(path))
        c.sendall((cmd + "\n").encode())
        return c.recv(256).decode().strip()


@pytest.mark.parametrize("err", [errno.EMFILE, errno.ECONNABORTED])
def test_listener_survives_a_transient_accept_error(
    err: int, tmp_path: Path, records: list[logging.LogRecord]
) -> None:
    from tui.control import ControlServer

    path = tmp_path / "control.sock"
    srv = ControlServer(lambda cmd: f"OK {cmd}", path=path)
    srv.start()
    try:
        assert srv._sock is not None
        flaky = _FlakyListener(srv._sock, err)
        srv._sock = flaky  # type: ignore[assignment]
        deadline = time.monotonic() + 3.0
        while not flaky.failed and time.monotonic() < deadline:
            time.sleep(0.05)
        assert flaky.failed
        assert _call(path, "STATUS") == "OK STATUS"
    finally:
        srv.stop()
    assert any("accept" in r.getMessage() for r in records)


def test_listener_logs_when_it_stops_on_an_unexpected_error(
    tmp_path: Path, records: list[logging.LogRecord]
) -> None:
    from tui.control import ControlServer

    path = tmp_path / "control.sock"
    srv = ControlServer(lambda cmd: f"OK {cmd}", path=path)
    srv.start()
    try:
        assert srv._sock is not None and srv._thread is not None
        srv._sock = _FlakyListener(srv._sock, errno.EBADF)  # type: ignore[assignment]
        srv._thread.join(timeout=3.0)
    finally:
        srv.stop()
    assert any(r.levelno >= logging.ERROR and "accept" in r.getMessage() for r in records)
