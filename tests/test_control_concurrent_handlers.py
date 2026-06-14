"""bug-class test.

Pre-fix `ControlServer._loop` handled each accepted connection
inline -- a single slow handler call stalled every subsequent
client at the socket layer. Concretely: a `MODEL` command that
took several seconds to apply blocked every `TOGGLE`, `PITCH`,
`STATUS`, etc. that arrived during the swap.

Post-fix the listener thread accepts connections and immediately
hands them off to a small ThreadPoolExecutor; up to 4 connections
are handled concurrently. The accept loop itself stays tight.

The test here drives a server with a deliberately-slow handler and
asserts that a fast follow-up command completes BEFORE the slow
one. Pre-fix this assertion fails: the fast command waits for the
slow handler to return first.
"""

from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

from tui import control


def _send(path: Path, cmd: str) -> tuple[float, str]:
    """Connect to `path`, send `cmd\\n`, read the reply, return
    (elapsed_seconds, reply)."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(10.0)
    t0 = time.monotonic()
    try:
        s.connect(str(path))
        s.sendall((cmd + "\n").encode("utf-8"))
        # Read until newline or EOF.
        buf = b""
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            buf += chunk
            if b"\n" in buf:
                break
    finally:
        s.close()
    return time.monotonic() - t0, buf.decode("utf-8", errors="replace").rstrip("\n")


def test_concurrent_handlers_unblock_fast_command_under_slow_one(
    tmp_path: Path,
) -> None:
    """Pre-fix this fails: the FAST handler's reply latency includes
    the SLOW handler's wall time because the accept loop ran serial.
    Post-fix the worker pool lets the fast command complete first."""

    slow_started = threading.Event()
    slow_release = threading.Event()

    def handler(cmd: str) -> str:
        if cmd == "SLOW":
            slow_started.set()
            # Hold the worker until the fast command has finished.
            slow_release.wait(timeout=5.0)
            return "OK slow"
        if cmd == "FAST":
            return "OK fast"
        return "ERR unknown"

    sock_path = tmp_path / "woys-test.sock"
    srv = control.ControlServer(handler=handler, path=sock_path)
    srv.start()
    try:
        # 1) Kick off the slow command in a background thread.
        slow_result: dict[str, tuple[float, str]] = {}

        def run_slow() -> None:
            slow_result["v"] = _send(sock_path, "SLOW")

        slow_thread = threading.Thread(target=run_slow, daemon=True)
        slow_thread.start()

        # 2) Wait until the slow handler has actually started
        # running in its worker. This avoids racing the accept loop.
        assert slow_started.wait(timeout=3.0), (
            "slow handler never started -- the listener thread "
            "didn't even reach the SLOW handler; the pool plumbing "
            "is broken"
        )

        # 3) Now send a fast command and time it. Pre-fix, this
        # blocks at accept() (or just after, inside the serial body)
        # until SLOW finishes. Post-fix the pool's second worker
        # handles it concurrently.
        fast_elapsed, fast_reply = _send(sock_path, "FAST")

        # 4) Let the slow handler return.
        slow_release.set()
        slow_thread.join(timeout=5.0)
        assert not slow_thread.is_alive(), "slow thread never joined"

        # The load-bearing assertions:
        assert fast_reply == "OK fast"
        # The fast command must NOT have been stalled behind the
        # slow handler. Pre-fix `fast_elapsed` >= ~the time SLOW was
        # blocked. Use a generous ceiling so the test isn't a
        # flake on a busy CI box: SLOW is held for as long as it
        # takes us to issue + complete the FAST round-trip, then
        # release it. Pre-fix the FAST round-trip wouldn't even
        # START until SLOW returned, so we'd see fast_elapsed >>
        # 2.0 seconds (we hold for as long as the assertion lets
        # us). Post-fix we expect well under 1 second.
        assert fast_elapsed < 1.0, (
            f"FAST command took {fast_elapsed:.3f}s -- accept loop "
            "appears to be serial (F-merged-025 regression)"
        )

        assert slow_result["v"][1] == "OK slow"

    finally:
        slow_release.set()  # safety in case the assert above tripped
        srv.stop()


def test_thread_pool_drains_on_stop(tmp_path: Path) -> None:
    """`stop()` must wait for in-flight workers to finish so the
    socket file is unlinked AFTER outstanding replies land. Without
    this, a `MODEL` reply could be lost mid-flight if the user hits
    Ctrl-C right after issuing the command."""
    in_flight = threading.Event()
    proceed = threading.Event()

    def handler(cmd: str) -> str:
        in_flight.set()
        proceed.wait(timeout=2.0)
        return "OK done"

    sock_path = tmp_path / "woys-test.sock"
    srv = control.ControlServer(handler=handler, path=sock_path)
    srv.start()
    try:
        result: dict[str, str] = {}

        def run_cmd() -> None:
            _elapsed, reply = _send(sock_path, "CMD")
            result["reply"] = reply

        t = threading.Thread(target=run_cmd, daemon=True)
        t.start()
        assert in_flight.wait(timeout=3.0)

        # Trigger shutdown WHILE the handler is still running.
        # stop() must wait for the worker to send its reply.
        def stop_server() -> None:
            srv.stop()

        stopper = threading.Thread(target=stop_server, daemon=True)
        stopper.start()

        # Let the handler finish.
        time.sleep(0.1)
        proceed.set()

        stopper.join(timeout=3.0)
        t.join(timeout=3.0)
        assert not stopper.is_alive()
        assert not t.is_alive()
        assert result.get("reply") == "OK done"
    finally:
        proceed.set()
        # stop() may have already run, idempotent re-call is fine.
        srv.stop()


def test_pool_size_constant_is_sane() -> None:
    """Document the pool ceiling as a checked constant rather than a
    magic number embedded in start()."""
    assert control.ControlServer._WORKER_POOL_SIZE >= 2
    assert control.ControlServer._WORKER_POOL_SIZE <= 16
