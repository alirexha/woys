"""Shared XDG_RUNTIME_DIR + secure `/tmp` fallback helper.

pre-fix
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
"protected" the mode -- a contract chain because `tui/control.py`
set no mode either.

This module provides ONE `safe_runtime_dir()` used by both, with:
  * `mode=0o700, exist_ok=False` on first creation of the `/tmp`
    fallback;
  * `lstat`-based refuse on a pre-existing fallback unless it is a
    real dir owned by `os.getuid()` with no group/other perms.

The same lstat check runs on a pre-existing `$XDG_RUNTIME_DIR/woys/`:
an inherited or misconfigured XDG_RUNTIME_DIR (`sudo -E`, `su` without
`-l`, a container env pointing into /tmp) can point at a directory
another user controls.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path


class UnsafeRuntimeDir(RuntimeError):
    """Raised when the runtime dir (XDG or the `/tmp` fallback) is pre-existing in
    an attacker-controllable state (wrong owner, world-perms, symlink).

    The caller may choose to surface the error (preferred -- a hard-
    fail is consistent with the project's hard-fail-on-missing-
    platform-feature stance), or to fall back to a different path.
    """


def safe_runtime_dir() -> Path:
    """Resolve the user's runtime dir for woys ephemera (control
    socket, slow-chunk log, instance lock).

    Priority:
      1. `$XDG_RUNTIME_DIR/woys/` when XDG_RUNTIME_DIR is absolute
         (preferred; user-private tmpfs, mode 0700 by the
         systemd-logind contract).
      2. `/tmp/woys-{uid}/` (fallback).

    Either dir is created with mode 0700; if it already exists it is
    lstat-refused unless it is a real dir owned by our uid with no
    group/other perms.

    Raises `UnsafeRuntimeDir` if the chosen dir exists in an
    attacker-controllable state.

    Returns the resolved Path; the directory is guaranteed to exist
    on return (creating it if needed under the mode constraint).
    """
    xdg = os.environ.get("XDG_RUNTIME_DIR", "")
    # A relative value would resolve against the CWD; the XDG spec says
    # to ignore it.
    if os.path.isabs(xdg):
        path = Path(xdg) / "woys"
        # logind makes XDG_RUNTIME_DIR 0700, but the variable can be
        # inherited from another user, so an existing `woys/` is checked
        # rather than trusted.
        try:
            path.mkdir(mode=0o700, parents=True)
            return path
        except FileExistsError:
            pass
        _check_private_dir(path, "runtime dir")
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
    _check_private_dir(path, "runtime-dir fallback")
    return path


def _check_private_dir(path: Path, label: str) -> None:
    """Raise UnsafeRuntimeDir unless `path` is a real directory (not a
    symlink) owned by our uid with no group/other permissions."""
    try:
        st = os.lstat(path)
    except OSError as e:
        raise UnsafeRuntimeDir(f"{label} {path}: cannot stat ({type(e).__name__}: {e})") from e

    # Order: not-a-dir first (catches symlinks), then mode (the most
    # likely real-world hit: an old umask-0022 dir from a pre-fix
    # woys install), then owner (last because a wrong-owner directory
    # is more likely an attacker than a user mistake).
    if not stat.S_ISDIR(st.st_mode):
        raise UnsafeRuntimeDir(
            f"{label} {path}: not a directory "
            f"(mode={oct(st.st_mode)}). Refusing -- a co-resident "
            f"attacker may have pre-created a symlink or non-dir at "
            f"this path. Remove it and re-run."
        )
    if st.st_mode & 0o077:
        raise UnsafeRuntimeDir(
            f"{label} {path}: world/group-accessible "
            f"(mode={oct(st.st_mode & 0o777)}, expected 0700 or "
            f"stricter). Refusing -- chmod 0700 it or remove it and "
            f"re-run."
        )
    if st.st_uid != os.getuid():
        raise UnsafeRuntimeDir(
            f"{label} {path}: owned by uid={st.st_uid}, "
            f"expected {os.getuid()}. Refusing -- a co-resident "
            f"attacker may have pre-created this path. Remove it and "
            f"re-run."
        )


def config_home() -> Path:
    """`$XDG_CONFIG_HOME`, or `~/.config` when it is unset, empty or
    relative (the XDG spec says to ignore a relative value; an empty one
    would otherwise resolve against the current directory)."""
    xdg = os.environ.get("XDG_CONFIG_HOME", "")
    return Path(xdg) if os.path.isabs(xdg) else Path.home() / ".config"


def config_dir() -> Path:
    """Resolve the woys config directory (holds `config.toml`).

    Priority:
      1. `$WOYS_CONFIG_DIR` (explicit override; the test suite points it
         at a temp dir so no test can reach the real config).
      2. `$XDG_CONFIG_HOME/woys/` when XDG_CONFIG_HOME is an absolute
         path (the XDG spec says a relative one must be ignored).
      3. `~/.config/woys/`.

    Before this helper existed the config always lived at
    `~/.config/woys/` regardless of XDG_CONFIG_HOME. If the XDG dir has
    no config yet but that legacy location does, keep using the legacy
    one so a user's saved settings and profiles don't vanish.
    """
    override = os.environ.get("WOYS_CONFIG_DIR")
    if override:
        return Path(override)
    legacy = Path.home() / ".config" / "woys"
    path = config_home() / "woys"
    if not (path / "config.toml").exists() and (legacy / "config.toml").exists():
        return legacy
    return path
