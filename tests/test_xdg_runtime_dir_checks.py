"""`$XDG_RUNTIME_DIR/woys` gets the same ownership / mode checks as the
`/tmp/woys-{uid}` fallback.

The XDG branch used to accept whatever was already at that path. With an
inherited or misconfigured XDG_RUNTIME_DIR (`sudo -E`, `su` without `-l`,
a container env pointing into /tmp) a `woys/` dir owned by another uid
with mode 0777, or a symlink, was used for the control socket and the
instance lock. A relative XDG_RUNTIME_DIR resolved against the CWD.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from woys import xdg
from woys.xdg import UnsafeRuntimeDir, safe_runtime_dir


def test_refuses_world_writable_existing_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    (tmp_path / "woys").mkdir()
    os.chmod(tmp_path / "woys", 0o777)
    with pytest.raises(UnsafeRuntimeDir, match=r"world/group-accessible"):
        safe_runtime_dir()


def test_refuses_symlinked_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "elsewhere"
    target.mkdir(mode=0o700)
    run = tmp_path / "run"
    run.mkdir()
    (run / "woys").symlink_to(target)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(run))
    with pytest.raises(UnsafeRuntimeDir, match=r"not a directory"):
        safe_runtime_dir()


def test_refuses_dir_owned_by_another_uid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    (tmp_path / "woys").mkdir(mode=0o700)
    real_uid = os.getuid()
    monkeypatch.setattr("woys.xdg.os.getuid", lambda: real_uid + 4242)
    with pytest.raises(UnsafeRuntimeDir, match=r"owned by uid="):
        safe_runtime_dir()


def test_accepts_existing_private_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    (tmp_path / "woys").mkdir(mode=0o700)
    assert safe_runtime_dir() == tmp_path / "woys"


def test_still_creates_missing_dir_0700(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    rt = safe_runtime_dir()
    assert rt == tmp_path / "woys"
    assert stat.S_IMODE(rt.lstat().st_mode) == 0o700


def test_relative_xdg_runtime_dir_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_RUNTIME_DIR", "run")
    sentinel = tmp_path / "fallback"
    monkeypatch.setattr(xdg, "_safe_tmp_fallback", lambda: sentinel)
    assert safe_runtime_dir() == sentinel
    assert not (tmp_path / "run").exists()
