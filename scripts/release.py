#!/usr/bin/env python3
"""Propagate the version in src/woys/__init__.py to every surface that
carries it.

review F-merged-029: pre-commit-027 the version lived as a literal
in pyproject.toml, src/woys/__init__.py, README.md, PROGRESS.md and
pkg/PKGBUILD, and the four drifted from each other. Hatchling now reads
__version__ out of src/woys/__init__.py at build time (see
[tool.hatch.version] in pyproject.toml); this script handles the remaining
*documentation* surfaces.

Usage::

    # 1. bump src/woys/__init__.py::__version__ by hand
    # 2. run:
    python scripts/release.py
    # 3. then `uv pip compile pyproject.toml -o requirements.txt`
    #    (CI's deps-sync gate will reject if you skip it)
    # 4. commit + tag

The CI `docs-version-grep` gate runs `scripts/check_version_drift.sh` and
fails if any of these surfaces fall out of sync.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
INIT_PY = REPO / "src" / "woys" / "__init__.py"


def read_version() -> str:
    m = re.search(r'^__version__ = "([^"]+)"', INIT_PY.read_text(), re.MULTILINE)
    if not m:
        sys.exit(f"could not find __version__ in {INIT_PY}")
    return m.group(1)


def patch(path: Path, pattern: str, replacement: str) -> bool:
    """Substitute `pattern` -> `replacement` in `path`. Returns True iff the
    file actually changed (so the caller can report a no-op vs a write).
    """
    text = path.read_text()
    new = re.sub(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if new == text:
        return False
    path.write_text(new)
    return True


def main() -> int:
    version = read_version()
    print(f"propagating version {version} from {INIT_PY.relative_to(REPO)}")

    changes = [
        # README.md "## Status (vX.Y.Z)" header
        (
            REPO / "README.md",
            r"^## Status \(v[0-9]+\.[0-9]+\.[0-9]+\)$",
            f"## Status (v{version})",
        ),
        # PKGBUILD pkgver= line
        (
            REPO / "pkg" / "PKGBUILD",
            r"^pkgver=[0-9]+\.[0-9]+\.[0-9]+$",
            f"pkgver={version}",
        ),
    ]

    any_change = False
    for path, pattern, replacement in changes:
        if patch(path, pattern, replacement):
            print(f"  patched {path.relative_to(REPO)}")
            any_change = True
        else:
            print(f"  {path.relative_to(REPO)} already at v{version}")

    if not any_change:
        print("no changes needed")
    print("next: run `uv pip compile pyproject.toml -o requirements.txt`")
    return 0


if __name__ == "__main__":
    sys.exit(main())
