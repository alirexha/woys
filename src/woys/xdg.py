"""Shared XDG_RUNTIME_DIR + secure `/tmp` fallback helper.

review F-32-02 (closes F-05-06, F-32-11, F-cx4-002): pre-fix
`tui/control.py:_runtime_dir` and `woys/instance_lock.py:_runtime_dir`
both fell back to `/tmp/woys-{uid}` when `XDG_RUNTIME_DIR` was unset
and called bare `mkdir(parents=True, exist_ok=True)` -- which inherits
the process umask (typically 0022, i.e. world-traversable 0755).
A co-resident attacker could pre-create or symlink the predictable
`/tmp/woys-{uid}` path, positioning themselves around the control
channel and the instance lock.

Two code comments asserted the "symlink TOCTOU surface closes" --
true ONLY on the XDG branch, false on the `/tmp` fallback. The
`instance_lock.py` comment was worse: it claimed `tui/control.py`
"protected" the mode -- a Hard Rule 1 chain because `tui/control.py`
set no mode either.

This module provides ONE `safe_runtime_dir()` used by both, with:
  * `mode=0o700, exist_ok=False` on first creation of the `/tmp`
    fallback;
  * `lstat`-based refuse on a pre-existing fallback unless it is a
    real dir owned by `os.getuid()` with no group/other perms.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path


class UnsafeRuntimeDir(RuntimeError):
    """Raised when the `/tmp` runtime-dir fallback is pre-existing in
    an attacker-controllable state (wrong owner, world-perms, symlink).

    The caller may choose to surface the error (preferred -- a hard-
    fail is consistent with the project's hard-fail-on-missing-
    platform-feature stance), or to fall back to a different path.
    """


def safe_runtime_dir() -> Path:
    """Resolve the user's runtime dir for woys ephemera (control
    socket, slow-chunk log, instance lock).

    Priority:
      1. `$XDG_RUNTIME_DIR/woys/` (preferred; user-private tmpfs,
         mode 0700 by the systemd-logind contract).
      2. `/tmp/woys-{uid}/` (fallback; created with `mode=0700,
         exist_ok=False`; if pre-existing, lstat-refused unless real-
         dir + own-UID + no group/other perms).

    Raises `UnsafeRuntimeDir` if the `/tmp` fallback exists in an
    attacker-controllable state.

    Returns the resolved Path; the directory is guaranteed to exist
    on return (creating it if needed under the mode constraint).
    """
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if xdg:
        path = Path(xdg) / "woys"
        # The XDG_RUNTIME_DIR is per the systemd-logind contract mode
        # 0700 -- so `woys/` underneath inherits the parent's safety.
        # We still create it with mode 0700 explicitly for parity.
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        return path
    return _safe_tmp_fallback()


def _safe_tmp_fallback() -> Path:
    """Create / validate the `/tmp/woys-{uid}/` fallback. Refuses any
    pre-existing path that doesn't pass the lstat ownership + mode
    check."""
    path = Path(f"/tmp/woys-{os.getuid()}")
    try:
        os.mkdir(path, mode=0o700)
        return path
    except FileExistsError:
        pass  # validate below

    try:
        st = os.lstat(path)
    except OSError as e:
        raise UnsafeRuntimeDir(
            f"runtime-dir fallback {path}: cannot stat ({type(e).__name__}: {e})"
        ) from e

    # Order: not-a-dir first (catches symlinks), then mode (the most
    # likely real-world hit: an old umask-0022 dir from a pre-fix
    # woys install), then owner (last because a wrong-owner directory
    # is more likely an attacker than a user mistake).
    if not stat.S_ISDIR(st.st_mode):
        raise UnsafeRuntimeDir(
            f"runtime-dir fallback {path}: not a directory "
            f"(mode={oct(st.st_mode)}). Refusing -- a co-resident "
            f"attacker may have pre-created a symlink or non-dir at "
            f"this path. Remove it and re-run."
        )
    if st.st_mode & 0o077:
        raise UnsafeRuntimeDir(
            f"runtime-dir fallback {path}: world/group-accessible "
            f"(mode={oct(st.st_mode & 0o777)}, expected 0700 or "
            f"stricter). Refusing -- chmod 0700 it or remove it and "
            f"re-run."
        )
    if st.st_uid != os.getuid():
        raise UnsafeRuntimeDir(
            f"runtime-dir fallback {path}: owned by uid={st.st_uid}, "
            f"expected {os.getuid()}. Refusing -- a co-resident "
            f"attacker may have pre-created this path. Remove it and "
            f"re-run."
        )
    return path
