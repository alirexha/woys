#!/usr/bin/env bash
# CI gate that catches version-drift between the
# single source (src/woys/__init__.py::__version__) and the documentation
# surfaces that historically drifted (README.md, pkg/PKGBUILD).
#
# Historical references inside CHANGELOG.md, docs/, LESSONS.md, and inline
# code comments are *journal entries* (e.g. "v0.9.0: feature X was
# introduced") rather than current-status claims, and are not policed here.
#
# Run locally:   bash scripts/check_version_drift.sh
# Run in CI:     same; fails non-zero on drift.
set -euo pipefail

cd "$(dirname "$0")/.."

VERSION=$(awk -F'"' '/^__version__/ {print $2; exit}' src/woys/__init__.py)
if [[ -z "$VERSION" ]]; then
    echo "FAIL: could not parse __version__ from src/woys/__init__.py" >&2
    exit 1
fi
echo "single source: src/woys/__init__.py == v$VERSION"

fail=0

# README.md must carry a single "## Status (vX.Y.Z)" header matching the
# single source. Multiple headers, or a mismatching version, fails.
status_headers=$(grep -cE '^## Status \(v[0-9]+\.[0-9]+\.[0-9]+\)$' README.md || true)
if [[ "$status_headers" != "1" ]]; then
    echo "FAIL: README.md must have exactly one '## Status (vX.Y.Z)' header; got $status_headers" >&2
    fail=1
elif ! grep -qE "^## Status \(v${VERSION}\)$" README.md; then
    actual=$(grep -oE '^## Status \(v[0-9]+\.[0-9]+\.[0-9]+\)' README.md | head -n1)
    echo "FAIL: README.md '$actual' does not match __init__.py (v$VERSION)" >&2
    fail=1
fi

# pkg/PKGBUILD pkgver= must match
if ! grep -qE "^pkgver=${VERSION}$" pkg/PKGBUILD; then
    actual=$(grep -oE '^pkgver=[0-9]+\.[0-9]+\.[0-9]+' pkg/PKGBUILD || echo "(none)")
    echo "FAIL: pkg/PKGBUILD $actual does not match __init__.py (v$VERSION)" >&2
    fail=1
fi

# pyproject.toml must be dynamic (must NOT carry a static "version = ..." line
# in the [project] table). hatch reads from __init__.py.
if grep -qE '^version = "[0-9]+\.[0-9]+\.[0-9]+"' pyproject.toml; then
    echo "FAIL: pyproject.toml carries a static \`version = \"...\"\` line." >&2
    echo "  The version must be dynamic (\`dynamic = [\"version\"]\` + [tool.hatch.version])." >&2
    fail=1
fi

if (( fail )); then
    echo
    echo "version drift detected. To fix:" >&2
    echo "  1. edit src/woys/__init__.py to the desired version" >&2
    echo "  2. run \`python scripts/release.py\` to propagate" >&2
    echo "  3. run \`uv pip compile pyproject.toml -o requirements.txt\`" >&2
    exit 1
fi

echo "version drift check passed."
