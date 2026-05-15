"""Unix-socket control channel.

Lets a running TUI (or headless engine) be poked from outside by a small CLI
client - `woys toggle`, `woys pitch +1`, etc. Wired here
instead of D-Bus to avoid running a GLib mainloop alongside Textual's asyncio
loop. KDE/GNOME WM shortcuts call the CLI; the CLI talks to this socket.

Protocol
--------
One newline-terminated command per connection. Server replies with a single
short status line and closes. Commands:

  TOGGLE        - start engine if stopped, stop if running
  PITCH +N      - pitch shift +N semitones (relative)
  PITCH -N      - pitch shift -N semitones (relative)
  PITCH 0       - reset to 0
  MODEL <slug>  - hot-swap the active RVC model (returns job id; v0.5.0 async)
  PROFILE <n>   - apply a saved profile by name (returns job id; v0.5.0 async)
  JOB <id>      - poll a previously-issued async job: pending/running/done/error
  STATUS        - print one-line status (instant, never blocks)
  QUIT          - instruct the TUI to exit

Async semantics (v0.5.0)
------------------------
Slow commands (MODEL, PROFILE) return immediately with `OK job=<id>` and
spawn the work on a background thread. Clients poll `JOB <id>` until the
state is `done` or `error <msg>`. Older clients that only spoke MODEL get
back the same OK reply but never poll - on a cache-cold swap they may see
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
    slow-chunk log, etc.).

    review F-32-02 (commit-047, closes F-05-06, F-32-11,
    F-cx4-002): delegates to `woys.xdg.safe_runtime_dir`, which does:
      * `mode=0700, exist_ok=True` on the XDG branch (mode-belt-and-
        braces; systemd-logind already sets `$XDG_RUNTIME_DIR` to
        0700 per spec);
      * `mode=0o700, exist_ok=False` on the `/tmp` fallback first
        creation; if pre-existing, lstat-refused unless real-dir +
        own-UID + no group/other perms.

    Pre-fix this function did `mkdir(parents=True, exist_ok=True)`
    with no mode -- inherited the process umask (typically 0022,
    world-traversable 0755). A co-resident attacker could pre-create
    or symlink `/tmp/woys-{uid}` and position themselves around the
    control channel. The pre-fix code comment falsely claimed "the
    symlink TOCTOU surface closes" -- true only on the XDG branch.
    """
    from woys.xdg import safe_runtime_dir

    return safe_runtime_dir()


def control_socket_path() -> Path:
    # The runtime-dir is created by safe_runtime_dir() with mode 0700.
    return _runtime_dir() / "control.sock"


def runtime_path(name: str) -> Path:
    """Return `<runtime_dir>/<name>`. The directory is guaranteed
    to exist + safe-mode by `safe_runtime_dir()` (commit-047)."""
    return _runtime_dir() / name


HandlerFn = Callable[[str], str]


# review F-merged-020: framing limits. The docstring above promises
# "one newline-terminated command / single short reply", but the pre-fix
# code did a single fixed-size `recv` (256 server / 512 client) and
# silently truncated anything longer. The STATUS reply was already
# ~250 B and growing -- truncation was a *when*, not *if*. The caps
# below are intentionally generous (64 KiB) so the framing fix doesn't
# create a new bottleneck; the real protection is the read-until-`\n`
# logic in `_recv_line` / `_recv_reply` below.
MAX_COMMAND_BYTES = 64 * 1024
MAX_REPLY_BYTES = 64 * 1024

# review F-merged-020 part 2: protocol version. Bumps on
# incompatible wire changes (semantic shifts of OK / ERR / JOB, new
# required fields, framing renegotiation). The STATUS reply stamps
# `proto=<N>` so a client can read the server's version and degrade
# gracefully. Pre-fix the docstring reasoned about "older clients" but
# no version field existed (Hard Rule 1 violation).
#
# Wire history:
#   v1 - 2026-05-15 - newline framing (commit-037a) + JOB protocol +
#                     ERR-on-failure invariant + STATUS stamps
#                     `proto=1`. The first version with an explicit
#                     wire-version contract.
#
# When bumping, also update tests/test_control_protocol_version.py
# so the contract has a paper trail.
PROTOCOL_VERSION = 1


def _recv_line(conn: socket.socket, max_bytes: int = MAX_COMMAND_BYTES) -> str | None:
    """Read from `conn` until a `\\n` byte or `max_bytes` is reached.

    Returns the decoded line (`\\n` stripped) on success, `None` on
    immediate EOF, or raises `ValueError` when `max_bytes` is exceeded
    before a newline arrives. Used server-side to honor the docstring's
    "newline-terminated" framing promise.
    """
    chunks: list[bytes] = []
    total = 0
    while total < max_bytes:
        try:
            chunk = conn.recv(min(4096, max_bytes - total))
        except (TimeoutError, OSError):
            raise
        if not chunk:
            break  # peer closed
        chunks.append(chunk)
        total += len(chunk)
        if b"\n" in chunk:
            full = b"".join(chunks)
            line, _, _ = full.partition(b"\n")
            return line.decode("utf-8", errors="replace").strip()
    if total >= max_bytes:
        raise ValueError(f"command exceeded {max_bytes} bytes without newline")
    if not chunks:
        return None
    # Peer closed mid-line. Treat as best-effort decode (no terminator).
    return b"".join(chunks).decode("utf-8", errors="replace").strip()


def _recv_reply(sock: socket.socket, max_bytes: int = MAX_REPLY_BYTES) -> str:
    """Client-side counterpart to `_recv_line`. Reads until `\\n` or
    `max_bytes`. Returns the decoded reply (sans newline).

    Distinct from `_recv_line` in two ways:
    1. The client trusts the server to close the connection after the
       reply, so a partial-but-newline-less buffer at EOF is OK
       (returned as-is).
    2. On overflow (reply > max_bytes with no newline) we append a
       ` ... (truncated)` marker rather than raising -- the client
       still gives the user something usable.
    """
    chunks: list[bytes] = []
    total = 0
    while total < max_bytes:
        chunk = sock.recv(min(4096, max_bytes - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if b"\n" in chunk:
            full = b"".join(chunks)
            line, _, _ = full.partition(b"\n")
            return line.decode("utf-8", errors="replace").strip()
    # Either EOF (return what we have) or max_bytes hit without newline.
    if total >= max_bytes:
        return b"".join(chunks).decode("utf-8", errors="replace").rstrip() + " ... (truncated)"
    return b"".join(chunks).decode("utf-8", errors="replace").strip()


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

        # v0.14.0 (Lens 6 / Lens 12 / C211): set restrictive umask
        # AROUND the bind so the socket file is created with mode 0600
        # atomically. Pre-v0.14.0 the bind created the file with the
        # process's default umask (typically 0644 or 0664), then
        # `os.chmod(..., 0o600)` ran AFTER -- a race window between
        # bind and chmod where a co-resident attacker on the same UID's
        # XDG_RUNTIME_DIR could connect to the socket before the
        # restrictive mode landed. Setting umask 0077 makes the bind
        # create the file as 0600 directly; the explicit chmod stays
        # as a belt-and-braces guarantee on filesystems that don't
        # honor umask on AF_UNIX sockets.
        prior_umask = os.umask(0o077)
        try:
            self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._sock.bind(str(self.path))
        finally:
            os.umask(prior_umask)
        self._sock.listen(4)
        self._sock.settimeout(0.5)
        os.chmod(self.path, 0o600)

        # v0.6.6 - guarantee the socket file gets unlinked even if stop()
        # never runs (e.g. `kill <tui-pid>`). Without this, the next
        # `woys status` / `woys toggle` finds a stale path that exists()
        # but refuses connect - the test_send_command_when_no_server case.
        atexit.register(self._unlink_path)
        if signal.getsignal(signal.SIGTERM) in (signal.SIG_DFL, signal.SIG_IGN):
            signal.signal(signal.SIGTERM, _exit_on_signal)

        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="woys-control", daemon=True)
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
                    # review F-05-01: SO_PEERCRED UID check.
                    # The socket file is mode 0600 + lives under
                    # XDG_RUNTIME_DIR (which is mode 0700 by the
                    # systemd-logind contract), so cross-UID connect
                    # is already blocked at the filesystem layer.
                    # SAME-UID processes (a malicious pip dep in the
                    # same venv, a game mod, a misbehaving script)
                    # can still `connect()` and dispatch QUIT /
                    # MODEL / TOGGLE / PITCH / PROFILE without any
                    # auth gate -- the socket is reachable by
                    # design (the WM-shortcut control path) and the
                    # commands have real impact. We read SO_PEERCRED
                    # and reject any UID != ours so the same-UID
                    # trust boundary is at least made explicit (a
                    # regression guard for the 0600 + XDG_RUNTIME_
                    # DIR layer, not a substitute for it).
                    if not _check_peer_uid(conn):
                        conn.sendall(b"ERR unauthorized\n")
                        continue
                    # review F-merged-020: read until newline (or the
                    # MAX_COMMAND_BYTES cap), honoring the docstring's
                    # "newline-terminated" promise. Pre-fix this was a
                    # single recv(256) that silently truncated.
                    try:
                        data = _recv_line(conn) or ""
                    except ValueError:
                        reply = "ERR command too long"
                    else:
                        reply = self.handler(data) if data else "ERR empty"
                    conn.sendall((reply + "\n").encode("utf-8"))
                except (TimeoutError, OSError) as e:
                    logger.warning("control conn error: %s", e)


# review F-05-01: SO_PEERCRED UID check. Linux's `struct ucred`
# is 12 bytes (3 x int32: pid / uid / gid). `SO_PEERCRED` is the
# kernel API. The check is the regression-guard for the 0600 + XDG_
# RUNTIME_DIR layer; it does NOT replace those (filesystem perms still
# do the heavy lifting cross-UID).
_SO_PEERCRED_STRUCT = "iII"  # 1 x signed pid (i), 2 x unsigned uid/gid (I)


def _check_peer_uid(conn: socket.socket) -> bool:
    """Return True iff the connecting peer is the SAME UID as the
    server process. Used by the control-socket accept loop to gate
    every connection on a UID match.

    Wrapped in a try/except so a kernel without SO_PEERCRED (very
    old / non-Linux) does not crash the server -- we log + accept
    in that case. The filesystem permissions remain the load-bearing
    cross-UID guard.
    """
    import struct

    try:
        cred = conn.getsockopt(
            socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize(_SO_PEERCRED_STRUCT)
        )
        _pid, peer_uid, _gid = struct.unpack(_SO_PEERCRED_STRUCT, cred)
    except (OSError, AttributeError) as e:
        logger.warning(
            "SO_PEERCRED unavailable (%s: %s); accepting on filesystem-perm guarantees only",
            type(e).__name__,
            e,
        )
        return True
    if peer_uid != os.getuid():
        logger.warning(
            "control socket: rejecting connection from UID %d (server UID %d)",
            peer_uid,
            os.getuid(),
        )
        return False
    return True


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
        return "ERR control socket not found - TUI not running?"
    # B34 / corr-022: retry briefly on ConnectionRefusedError. The TUI's
    # bind → listen → settimeout sequence has a ~50 ms window where
    # `path.exists()` is True but `connect()` refuses. A client racing
    # TUI startup hit this consistently before; 3 attempts x 100 ms
    # absorbs the race without slowing down the truly-stale-socket path
    # noticeably.
    last_err: BaseException | None = None
    for _ in range(3):
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.settimeout(timeout)
                s.connect(str(path))
                s.sendall((cmd + "\n").encode("utf-8"))
                # review F-merged-020: read until newline (or
                # MAX_REPLY_BYTES). Pre-fix this was recv(512) which
                # silently truncated the STATUS reply (~250 B and
                # growing) when it eventually crossed the cap.
                return _recv_reply(s)
        except ConnectionRefusedError as e:
            last_err = e
            time.sleep(0.1)
            continue
        except FileNotFoundError as e:
            last_err = e
            break
    if isinstance(last_err, FileNotFoundError):
        return "ERR control socket stale - TUI not running?"
    return "ERR control socket refused - TUI not accepting connections?"


def submit_and_wait(
    cmd: str,
    *,
    poll_interval: float = 0.05,
    overall_timeout: float = 30.0,
) -> str:
    """Helper - issue a slow command, parse the `OK job=<id>` reply, poll
    `JOB <id>` until done/error/timeout. Returns the final JOB reply line.
    """
    submit_reply = send_command(cmd, timeout=5.0)
    if not submit_reply.startswith("OK"):
        return submit_reply
    if "job=" not in submit_reply:
        # Synchronous handler - no JOB protocol involved.
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
