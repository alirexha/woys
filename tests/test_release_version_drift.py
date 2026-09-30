"""The README "## Status (vX.Y.Z)" header is the one documentation surface
that still carries the version. The drift gate must catch a stale header,
and release.py must bring it back in step.

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


def _make_readme_stale(root: Path) -> Path:
    readme = root / "README.md"
    header = f"## Status (v{_version()})"
    assert header in readme.read_text()
    readme.write_text(readme.read_text().replace(header, "## Status (v0.0.1)"))
    return readme


def test_readme_matches_the_single_source() -> None:
    assert f"\n## Status (v{_version()})\n" in (REPO / "README.md").read_text()


def test_drift_check_catches_a_stale_readme(tmp_path: Path) -> None:
    root = _copy_repo(tmp_path)
    assert _drift_check(root).returncode == 0
    _make_readme_stale(root)
    proc = _drift_check(root)
    assert proc.returncode == 1 and "README.md" in proc.stderr, proc.stdout + proc.stderr


def test_release_updates_readme(tmp_path: Path) -> None:
    root = _copy_repo(tmp_path)
    readme = _make_readme_stale(root)
    subprocess.run(
        [sys.executable, str(root / "scripts" / "release.py")], check=True, capture_output=True
    )
    assert readme.read_text() == (REPO / "README.md").read_text()
    assert _drift_check(root).returncode == 0
