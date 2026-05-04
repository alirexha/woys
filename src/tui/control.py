"""Unix-socket control channel.

Lets a running TUI (or headless engine) be poked from outside by a small CLI
client — `vcclient-cachy toggle`, `vcclient-cachy pitch +1`, etc. Wired here
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
  MODEL <slug>  — hot-swap the active RVC model (v0.4.1)
  PROFILE <n>   — apply a saved profile by name (v0.4.1)
  STATUS        — print one-line status (now includes model=...)
  QUIT          — instruct the TUI to exit

Path: $XDG_RUNTIME_DIR/vcclient-cachy/control.sock (falls back to /tmp).
"""

from __future__ import annotations

import contextlib
import logging
import os
import socket
import threading
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger("vcclient_cachy.control")


def control_socket_path() -> Path:
    base = os.environ.get("XDG_RUNTIME_DIR")
    root = Path(base) if base else Path("/tmp") / f"vcclient-cachy-{os.getuid()}"
    out = (Path(root) / "vcclient-cachy" / "control.sock") if base else (root / "control.sock")
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


HandlerFn = Callable[[str], str]


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

        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="vcclient-control", daemon=True)
        self._thread.start()

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


def send_command(cmd: str, timeout: float = 1.0) -> str:
    """Client side: connect once, send `cmd`, return reply."""
    path = control_socket_path()
    if not path.exists():
        return "ERR control socket not found — TUI not running?"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect(str(path))
        s.sendall((cmd + "\n").encode("utf-8"))
        return s.recv(256).decode("utf-8", errors="replace").strip()
