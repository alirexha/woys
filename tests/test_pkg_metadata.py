"""pkg/PKGBUILD metadata checks that need no makepkg.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
_ARRAYS = ("depends", "makedepends", "optdepends")


def _pkgbuild_array(name: str) -> list[str]:
    out = subprocess.run(
        ["bash", "-c", f'source pkg/PKGBUILD && printf "%s\\n" "${{{name}[@]}}"'],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [ln for ln in out.splitlines() if ln]


def _srcinfo_values(key: str) -> list[str]:
    prefix = f"\t{key} = "
    return [
        ln[len(prefix) :]
        for ln in (REPO / "pkg" / ".SRCINFO").read_text().splitlines()
        if ln.startswith(prefix)
    ]


def test_srcinfo_mirrors_the_pkgbuild_arrays() -> None:
    for name in _ARRAYS:
        assert _srcinfo_values(name) == _pkgbuild_array(name), name


def test_optdepends_name_real_packages_for_real_features() -> None:
    """`evdev` is not an Arch package (python-evdev is), and the evdev
    global hotkey it advertised is never started by woys."""
    names = [entry.split(":", 1)[0] for entry in _pkgbuild_array("optdepends")]
    assert "evdev" not in names
    hotkey_wired = any(
        "EvdevHotkey(" in p.read_text()
        for p in (REPO / "src").rglob("*.py")
        if p.name != "hotkey.py"
    )
    if not hotkey_wired:
        assert not any("evdev" in n for n in names)


def test_makedepends_has_no_unused_pip() -> None:
    """The build uses python-build + python-installer; pip is never run."""
    assert "python-pip" not in _pkgbuild_array("makedepends")


def test_python_constraint_matches_pyproject() -> None:
    """requires-python is ">=3.11,<3.13"; a bare python>=3.11 let the
    package install onto an interpreter the pinned wheels do not support."""
    import tomllib

    with open(REPO / "pyproject.toml", "rb") as f:
        spec = tomllib.load(f)["project"]["requires-python"]
    wanted = sorted(f"python{part.strip()}" for part in spec.split(","))
    got = sorted(d for d in _pkgbuild_array("depends") if d.startswith("python"))
    assert got == wanted
