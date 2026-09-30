"""pkg/.SRCINFO carries the version twice (pkgver and the source tag). It
drifted to 0.13.3 while PKGBUILD moved on, because neither the drift gate
nor release.py looked at it.

Both scripts run here against a throwaway copy of the files they touch.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_FILES = (
    "src/woys/__init__.py",
    "README.md",
    "pyproject.toml",
    "pkg/PKGBUILD",
    "pkg/.SRCINFO",
    "scripts/check_version_drift.sh",
    "scripts/release.py",
)


def _copy_repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    for rel in _FILES:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, root / rel)
    return root


def _version() -> str:
    for line in (REPO / "src" / "woys" / "__init__.py").read_text().splitlines():
        if line.startswith("__version__"):
            return line.split('"')[1]
    raise AssertionError("no __version__")


def _drift_check(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(root / "scripts" / "check_version_drift.sh")],
        capture_output=True,
        text=True,
    )


def test_srcinfo_matches_the_single_source() -> None:
    srcinfo = (REPO / "pkg" / ".SRCINFO").read_text()
    v = _version()
    assert f"\tpkgver = {v}\n" in srcinfo
    assert f"source = woys-{v}::" in srcinfo and f"#tag=v{v}\n" in srcinfo


def test_drift_check_catches_a_stale_srcinfo(tmp_path: Path) -> None:
    root = _copy_repo(tmp_path)
    assert _drift_check(root).returncode == 0
    srcinfo = root / "pkg" / ".SRCINFO"
    srcinfo.write_text(srcinfo.read_text().replace(_version(), "0.0.1"))
    proc = _drift_check(root)
    assert proc.returncode == 1 and ".SRCINFO" in proc.stderr, proc.stdout + proc.stderr


def test_release_updates_srcinfo(tmp_path: Path) -> None:
    root = _copy_repo(tmp_path)
    srcinfo = root / "pkg" / ".SRCINFO"
    srcinfo.write_text(srcinfo.read_text().replace(_version(), "0.0.1"))
    subprocess.run(
        [sys.executable, str(root / "scripts" / "release.py")], check=True, capture_output=True
    )
    assert srcinfo.read_text() == (REPO / "pkg" / ".SRCINFO").read_text()
    assert _drift_check(root).returncode == 0
