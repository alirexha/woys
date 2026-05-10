"""v0.14.0 (Lens 17 / C009): single-instance file lock.

Pre-v0.14.0 there was no instance-level lock. Two concurrent `woys run`
or `woys engine` invocations both unlinked each other's
`$XDG_RUNTIME_DIR/woys/control.sock` and bound their own; both engines
also wrote into WoysSink simultaneously, so listeners heard out-of-phase
double-converted audio. Phase 1 lens-17 F17.7 reproduced the corruption.

This module provides a context manager that takes an exclusive flock on
`$XDG_RUNTIME_DIR/woys/instance.lock` (or `/tmp/woys-{uid}/instance.lock`
when XDG_RUNTIME_DIR is unset). If another process already holds it,
acquire raises `InstanceLockBusy` with the holder PID for a friendly
error message.

The lock is released when the context exits. flock semantics tie the
lock to the file descriptor: a SIGKILL releases it automatically (the
kernel closes the fd), so a previous crashed instance never leaves the
lock stuck.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class InstanceLockBusy(RuntimeError):
    """Raised when another woys instance already holds the lock."""

    def __init__(self, holder_pid: str, lock_path: Path) -> None:
        super().__init__(
            f"another woys instance is running (pid={holder_pid}, lock={lock_path}). "
            f"Stop the other instance or remove the lock file if you're sure no "
            f"engine is running."
        )
        self.holder_pid = holder_pid
        self.lock_path = lock_path


def _runtime_dir() -> Path:
    """Pick the same dir that tui.control uses for the socket so the
    lock and the socket live next to each other.

    Priority:
      1. `$XDG_RUNTIME_DIR/woys/`  (preferred; user-private, tmpfs)
      2. `/tmp/woys-{uid}/`        (fallback; ownership/mode protected
                                    by the caller in tui/control.py)
    """
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if xdg:
        return Path(xdg) / "woys"
    return Path(f"/tmp/woys-{os.getuid()}")


@contextmanager
def acquire_instance_lock() -> Iterator[Path]:
    """Acquire an exclusive flock on the woys instance lock file.

    Raises InstanceLockBusy if another process holds the lock; reading
    the file's contents gives that process's PID (best-effort - the
    holder writes its pid right after acquiring).

    The flock is released automatically when the contextmanager exits
    OR when the holding process dies (kernel closes the fd).
    """
    runtime_dir = _runtime_dir()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    lock_path = runtime_dir / "instance.lock"

    # O_RDWR so we can write our PID after acquiring; O_CREAT so the
    # first instance bootstraps the file. Mode 0600 is hygiene - the
    # lock file holds only the holder's PID.
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            # Read the PID the holder wrote (best-effort).
            try:
                with open(lock_path) as fh:
                    holder = fh.read().strip() or "?"
            except OSError:
                holder = "?"
            raise InstanceLockBusy(holder, lock_path) from exc
        # Write our PID for diagnostic purposes (next instance reads this).
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        try:
            yield lock_path
        finally:
            # flock is released when fd closes; explicit unlock is a
            # belt-and-braces no-op.
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
