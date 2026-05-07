"""Unix-socket control channel.

Lets a running TUI (or headless engine) be poked from outside by a small CLI
client — `woys toggle`, `woys pitch +1`, etc. Wired here
instead of D-Bus to avoid running a GLib mainloop alongside Textual's asyncio
loop. KDE/GNOME WM shortcuts call the CLI; the CLI talks to this socket.

Protocol
--------
One newline-terminated command per connection. Server replies with a single
short status line and closes. Commands:

  TOGGLE        — start engine if stopped, stop if running
  PITCH +N      — pitch shift +N semitones (relative)
  PITCH -N      — pitch shift -N semitones (relative)
  PITCH 0       — reset to 0
  MODEL <slug>  — hot-swap the active RVC model (returns job id; v0.5.0 async)
  PROFILE <n>   — apply a saved profile by name (returns job id; v0.5.0 async)
  JOB <id>      — poll a previously-issued async job: pending/running/done/error
  STATUS        — print one-line status (instant, never blocks)
  QUIT          — instruct the TUI to exit

Async semantics (v0.5.0)
------------------------
Slow commands (MODEL, PROFILE) return immediately with `OK job=<id>` and
spawn the work on a background thread. Clients poll `JOB <id>` until the
state is `done` or `error <msg>`. Older clients that only spoke MODEL get
back the same OK reply but never poll — on a cache-cold swap they may see
the new voice up to ~600 ms later, but no error.

Path: $XDG_RUNTIME_DIR/woys/control.sock (falls back to /tmp).
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import os
import signal
import socket
import sys
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("woys.control")


def _runtime_dir() -> Path:
    """Resolve the user's runtime dir for woys ephemera (control socket,
    slow-chunk log, etc.). Prefers XDG_RUNTIME_DIR (mode 0700 by spec);
    falls back to /tmp/woys-<uid>/ for systems that don't set it."""
    base = os.environ.get("XDG_RUNTIME_DIR")
    if base:
        return Path(base) / "woys"
    return Path("/tmp") / f"woys-{os.getuid()}"


def control_socket_path() -> Path:
    out = _runtime_dir() / "control.sock"
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def runtime_path(name: str) -> Path:
    """Return `<runtime_dir>/<name>` and ensure the parent exists.

    B13 / corr-012 / sec-002: replaces predictable `/tmp/woys-*` paths
    that were symlink-attackable on multi-user systems. XDG_RUNTIME_DIR
    is mode 0700 by the systemd-logind contract, so the symlink TOCTOU
    surface closes.
    """
    rt = _runtime_dir()
    rt.mkdir(parents=True, exist_ok=True)
    return rt / name


HandlerFn = Callable[[str], str]


@dataclass
class Job:
    id: str
    state: str = "pending"  # pending → running → done | error
    message: str = ""
    started_at: float = field(default_factory=time.time)
    completed_at: float | None = None


class JobRegistry:
    """In-memory async-job table keyed by short UUID. Thread-safe.

    Drops jobs older than `ttl_seconds` on each `submit` to bound memory.
    Clients are expected to poll JOB <id> shortly after submission; stale
    poll results past TTL return "unknown".
    """

    def __init__(self, ttl_seconds: float = 300.0) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds

    def submit(self, fn: Callable[[], None]) -> str:
        """Run `fn()` on a background thread; return a job id immediately."""
        self._gc()
        jid = uuid.uuid4().hex[:12]
        job = Job(id=jid)
        with self._lock:
            self._jobs[jid] = job

        def runner() -> None:
            with self._lock:
                self._jobs[jid].state = "running"
            try:
                fn()
                with self._lock:
                    self._jobs[jid].state = "done"
                    self._jobs[jid].completed_at = time.time()
            except Exception as e:
                with self._lock:
                    self._jobs[jid].state = "error"
                    self._jobs[jid].message = f"{type(e).__name__}: {e}"
                    self._jobs[jid].completed_at = time.time()

        threading.Thread(target=runner, name=f"job-{jid}", daemon=True).start()
        return jid

    def status_line(self, jid: str) -> str:
        with self._lock:
            job = self._jobs.get(jid)
        if job is None:
            return "ERR unknown job"
        if job.state in ("done", "error"):
            elapsed = (job.completed_at or time.time()) - job.started_at
            base = f"OK state={job.state} elapsed_ms={int(elapsed * 1000)}"
        else:
            elapsed = time.time() - job.started_at
            base = f"OK state={job.state} elapsed_ms={int(elapsed * 1000)}"
        return base + (f" msg={job.message}" if job.message else "")

    def _gc(self) -> None:
        cutoff = time.time() - self._ttl
        with self._lock:
            stale = [
                jid
                for jid, job in self._jobs.items()
                if (job.completed_at or job.started_at) < cutoff
            ]
            for jid in stale:
                del self._jobs[jid]


class ControlServer:
    """Threaded Unix-domain socket listener.

    `handler(command)` returns a reply string. Server takes care of read/write
    framing. Server quits when `stop()` is called.
    """

    def __init__(self, handler: HandlerFn, path: Path | None = None) -> None:
        self.handler = handler
        self.path = path or control_socket_path()
        self._sock: socket.socket | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        # Remove a stale socket from a prior run.
        with contextlib.suppress(OSError):
            self.path.unlink(missing_ok=True)

        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.bind(str(self.path))
        self._sock.listen(4)
        self._sock.settimeout(0.5)
        os.chmod(self.path, 0o600)

        # v0.6.6 — guarantee the socket file gets unlinked even if stop()
        # never runs (e.g. `kill <tui-pid>`). Without this, the next
        # `woys status` / `woys toggle` finds a stale path that exists()
        # but refuses connect — the test_send_command_when_no_server case.
        atexit.register(self._unlink_path)
        if signal.getsignal(signal.SIGTERM) in (signal.SIG_DFL, signal.SIG_IGN):
            signal.signal(signal.SIGTERM, _exit_on_signal)

        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="vcclient-control", daemon=True)
        self._thread.start()

    def _unlink_path(self) -> None:
        """atexit-safe socket file unlink (idempotent)."""
        with contextlib.suppress(OSError):
            self.path.unlink(missing_ok=True)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.5)
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None
        with contextlib.suppress(OSError):
            self.path.unlink(missing_ok=True)

    def _loop(self) -> None:
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            with conn:
                try:
                    conn.settimeout(0.5)
                    data = conn.recv(256).decode("utf-8", errors="replace").strip()
                    reply = self.handler(data) if data else "ERR empty"
                    conn.sendall((reply + "\n").encode("utf-8"))
                except (TimeoutError, OSError) as e:
                    logger.warning("control conn error: %s", e)


def _exit_on_signal(signum: int, _frame: object) -> None:
    """SIGTERM → clean exit so atexit handlers (incl. socket unlink) fire."""
    sys.exit(128 + signum)


def send_command(cmd: str, timeout: float = 30.0) -> str:
    """Client side: connect once, send `cmd`, return reply.

    v0.5.0: default timeout bumped from 1 s to 30 s. Slow commands (MODEL,
    PROFILE) return a job id within milliseconds; the JOB poll happens
    inside the timeout window, so even cold-cache swaps fit comfortably.

    v0.6.6: handles stale socket files. If the TUI was killed without a
    clean shutdown (kill -9, crash, etc.), the socket path can exist as a
    file with no listener. We catch ConnectionRefusedError / FileNotFoundError
    and return a clear ERR string instead of letting the exception escape.
    """
    path = control_socket_path()
    if not path.exists():
        return "ERR control socket not found — TUI not running?"
    # B34 / corr-022: retry briefly on ConnectionRefusedError. The TUI's
    # bind → listen → settimeout sequence has a ~50 ms window where
    # `path.exists()` is True but `connect()` refuses. A client racing
    # TUI startup hit this consistently before; 3 attempts × 100 ms
    # absorbs the race without slowing down the truly-stale-socket path
    # noticeably.
    last_err: BaseException | None = None
    for _ in range(3):
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.settimeout(timeout)
                s.connect(str(path))
                s.sendall((cmd + "\n").encode("utf-8"))
                return s.recv(512).decode("utf-8", errors="replace").strip()
        except ConnectionRefusedError as e:
            last_err = e
            time.sleep(0.1)
            continue
        except FileNotFoundError as e:
            last_err = e
            break
    if isinstance(last_err, FileNotFoundError):
        return "ERR control socket stale — TUI not running?"
    return "ERR control socket refused — TUI not accepting connections?"


def submit_and_wait(
    cmd: str,
    *,
    poll_interval: float = 0.05,
    overall_timeout: float = 30.0,
) -> str:
    """Helper — issue a slow command, parse the `OK job=<id>` reply, poll
    `JOB <id>` until done/error/timeout. Returns the final JOB reply line.
    """
    submit_reply = send_command(cmd, timeout=5.0)
    if not submit_reply.startswith("OK"):
        return submit_reply
    if "job=" not in submit_reply:
        # Synchronous handler — no JOB protocol involved.
        return submit_reply
    jid = submit_reply.split("job=", 1)[1].split()[0]
    deadline = time.time() + overall_timeout
    last = ""
    while time.time() < deadline:
        last = send_command(f"JOB {jid}", timeout=2.0)
        if (" state=done" in last) or (" state=error" in last) or last.startswith("ERR"):
            return last
        time.sleep(poll_interval)
    return f"ERR job={jid} timed out after {overall_timeout}s; last={last!r}"
