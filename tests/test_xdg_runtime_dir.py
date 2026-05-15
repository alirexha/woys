"""review F-32-02: shared safe runtime-dir helper with hardened
`/tmp/woys-{uid}/` fallback.

Pre-fix `_runtime_dir` in `tui/control.py` and `woys/instance_lock.py`
both fell back to `/tmp/woys-{uid}` via `mkdir(parents=True,
exist_ok=True)` -- inheriting the process umask (typically 0022,
world-traversable 0755). A co-resident attacker could pre-create or
symlink the predictable `/tmp/woys-{uid}` path. Two code comments
falsely asserted the symlink-TOCTOU surface "closes" -- true only on
the XDG branch.

Post-fix `woys.xdg.safe_runtime_dir` is the single source:
  * XDG branch creates `$XDG_RUNTIME_DIR/woys/` with mode 0700;
  * `/tmp` fallback first-create uses `mode=0o700, exist_ok=False`;
  * pre-existing fallback path is lstat-refused unless real-dir
    + own-UID + no group/other perms;
  * `UnsafeRuntimeDir` raised on any unsafe state.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "src" / "server") not in sys.path:
    sys.path.insert(0, str(REPO / "src" / "server"))


def test_xdg_branch_creates_woys_subdir_with_mode_0700(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When XDG_RUNTIME_DIR is set, safe_runtime_dir() creates a
    `woys/` subdir under it with mode 0700."""
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    from woys.xdg import safe_runtime_dir

    rt = safe_runtime_dir()
    assert rt == tmp_path / "woys"
    assert rt.exists()
    mode = stat.S_IMODE(rt.lstat().st_mode)
    assert mode == 0o700, f"woys/ subdir must be mode 0700; got {oct(mode)}"


def test_tmp_fallback_first_create_uses_mode_0700(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bug-class test: when XDG is unset and the fallback path does
    NOT exist, safe_runtime_dir creates it with `mode=0o700`. Pre-fix
    `mkdir(parents=True, exist_ok=True)` inherited the umask (0022)
    and yielded 0755."""
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    # Use a fake UID so the path is unique per test (and so it isn't
    # /tmp/woys-{real_uid}, which the dev machine may already have
    # in some pre-existing mode).
    fake_uid = 50_000 + os.getpid() % 1000
    monkeypatch.setattr("woys.xdg.os.getuid", lambda: fake_uid)
    fallback = Path(f"/tmp/woys-{fake_uid}")
    # Ensure clean state.
    if fallback.exists():
        fallback.rmdir()

    from woys.xdg import safe_runtime_dir

    try:
        rt = safe_runtime_dir()
        assert rt == fallback
        mode = stat.S_IMODE(rt.lstat().st_mode)
        assert mode == 0o700, (
            f"/tmp fallback first-create must be mode 0700; "
            f"got {oct(mode)} (pre-fix umask leaked 0755)"
        )
    finally:
        if fallback.exists():
            fallback.rmdir()


def test_tmp_fallback_refuses_world_perms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bug-class test: pre-existing `/tmp/woys-{uid}` with world-
    traversable mode (0755 -- the pre-fix default) must be REFUSED
    with UnsafeRuntimeDir, not silently used."""
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    fake_uid = 60_000 + os.getpid() % 1000
    monkeypatch.setattr("woys.xdg.os.getuid", lambda: fake_uid)
    fallback = Path(f"/tmp/woys-{fake_uid}")
    fallback.mkdir(mode=0o755, exist_ok=True)
    os.chmod(fallback, 0o755)  # explicit -- mkdir mode is also umasked

    from woys.xdg import UnsafeRuntimeDir, safe_runtime_dir

    try:
        with pytest.raises(UnsafeRuntimeDir, match=r"world/group-accessible"):
            safe_runtime_dir()
    finally:
        fallback.rmdir()


def test_tmp_fallback_refuses_symlink(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A symlink at the fallback path (the classic TOCTOU positioning
    attack) is refused."""
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    fake_uid = 70_000 + os.getpid() % 1000
    monkeypatch.setattr("woys.xdg.os.getuid", lambda: fake_uid)
    fallback = Path(f"/tmp/woys-{fake_uid}")
    target = tmp_path / "victim"
    target.mkdir(mode=0o700)
    os.symlink(str(target), str(fallback))

    from woys.xdg import UnsafeRuntimeDir, safe_runtime_dir

    try:
        with pytest.raises(UnsafeRuntimeDir, match=r"not a directory"):
            safe_runtime_dir()
    finally:
        os.unlink(fallback)


def test_tmp_fallback_refuses_wrong_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pre-existing fallback owned by ANOTHER UID is refused.

    We can't actually chown (would require root); we simulate by
    monkey-patching `os.lstat` to return a foreign uid for the
    fallback path. The helper's UID-check path is what's under test
    here, not the chown infrastructure."""
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    fake_uid = 80_000 + os.getpid() % 1000
    monkeypatch.setattr("woys.xdg.os.getuid", lambda: fake_uid)
    fallback = Path(f"/tmp/woys-{fake_uid}")
    fallback.mkdir(mode=0o700, exist_ok=True)
    os.chmod(fallback, 0o700)

    real_lstat = os.lstat

    def fake_lstat(path: object) -> os.stat_result:
        st = real_lstat(path)
        if str(path) == str(fallback):
            # Force the test-time real uid to differ from the fake one.
            class _FakeStat:
                st_mode = st.st_mode
                st_uid = fake_uid + 999  # mismatch
                st_gid = st.st_gid

            return _FakeStat()  # type: ignore[return-value]
        return st

    monkeypatch.setattr("woys.xdg.os.lstat", fake_lstat)

    from woys.xdg import UnsafeRuntimeDir, safe_runtime_dir

    try:
        with pytest.raises(UnsafeRuntimeDir, match=r"owned by uid="):
            safe_runtime_dir()
    finally:
        if fallback.exists():
            fallback.rmdir()


def test_both_control_and_instance_lock_call_safe_helper() -> None:
    """Structural pin: `tui.control._runtime_dir` and
    `woys.instance_lock._runtime_dir` MUST both go through
    `woys.xdg.safe_runtime_dir`. Pre-fix each had its own bare
    `mkdir(parents=True, exist_ok=True)` with no mode argument."""
    src_control = (
        Path(__file__).resolve().parent.parent / "src" / "tui" / "control.py"
    ).read_text()
    src_lock = (
        Path(__file__).resolve().parent.parent / "src" / "woys" / "instance_lock.py"
    ).read_text()

    for label, text in (("tui/control.py", src_control), ("woys/instance_lock.py", src_lock)):
        assert "safe_runtime_dir" in text, (
            f"{label}: must delegate to woys.xdg.safe_runtime_dir; "
            f"pre-fix had its own bare mkdir that inherited umask 0022"
        )
        # The pre-fix false comments must not appear as standalone
        # assertions. We allow them to appear in a `"""docstring..."""`
        # context that QUOTES the old wording while explaining the
        # fix (the natural shape of an review comment) -- the
        # forbidden form is the standalone phrase asserting the
        # surface "closes".
        bad_strings = [
            'symlink TOCTOU surface closes."""',
            '"protected by the caller in tui/control.py")',
        ]
        for bad in bad_strings:
            assert bad not in text, f"{label}: the false comment {bad!r} must be removed (F-32-02)"
