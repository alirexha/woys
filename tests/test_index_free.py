"""review F-31-01 + F-CX6-03 (commit-051): woys is INDEX-FREE.

Pre-fix `woys models download <repo>` snapshot-fetched any `.index`
file present in the HF repo (RVC's optional faiss speaker-similarity
index). The engine never consumed it -- `index_rate` had no
implementation. Plus `faiss-cpu` was a runtime dependency paid by
every install for a feature that did not exist.

The F-31-01 product decision: drop the unused download AND the
`faiss-cpu` dependency rather than implement an unused feature.

This test pins:
  * `download_repo`'s entry filter excludes `.index` files.
  * `faiss-cpu` is not in `pyproject.toml` runtime deps.
  * `faiss-cpu` is not in the pinned `requirements.txt`.
  * No engine / convert source code imports `faiss`.
  * `docs/MODELS.md` documents the index-free contract.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "src" / "server") not in sys.path:
    sys.path.insert(0, str(REPO / "src" / "server"))


def test_download_repo_does_not_request_index_files() -> None:
    """Structural pin: the entry-filter in `download_repo` excludes
    `.index` files. Pre-fix it accepted both `.onnx` and `.index`."""
    src = REPO / "src" / "woys" / "models.py"
    text = src.read_text()
    # The relevant line is the entries-list comprehension.
    idx = text.find("entries = [s.rfilename for s in siblings")
    assert idx > 0, "the entries-list comprehension must exist"
    body = text[idx : idx + 200]
    assert ".onnx" in body
    assert ".index" not in body, (
        "F-31-01 + F-CX6-03: download_repo must NOT request `.index` "
        "files -- woys is index-free. Found `.index` in the entries "
        "filter, which means we're back to fetching the unused file."
    )


def test_faiss_cpu_not_in_pyproject() -> None:
    """`faiss-cpu` must NOT appear as a runtime dependency in
    `pyproject.toml`. Pre-fix it was pinned at `>=1.8`."""
    text = (REPO / "pyproject.toml").read_text()
    # The forbidden form is `"faiss-cpu>=...",` in a dependency list.
    # We allow the constant string to appear in a COMMENT explaining
    # the removal (the comment names `faiss-cpu` to make the
    # deprecation visible).
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue  # comments are allowed to mention the name
        assert "faiss-cpu" not in stripped, (
            f"F-CX6-03: faiss-cpu must NOT be in pyproject.toml as a runtime dep; found: {line!r}"
        )


def test_faiss_cpu_not_in_requirements_txt() -> None:
    """`faiss-cpu` must NOT appear as a pinned wheel in
    `requirements.txt`. Pre-fix `faiss-cpu==1.13.2` was pinned."""
    text = (REPO / "requirements.txt").read_text()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue  # comments are allowed
        if stripped.startswith("faiss-cpu") or stripped == "faiss-cpu":
            raise AssertionError(
                f"F-CX6-03: faiss-cpu must NOT be in requirements.txt; found pinned line: {line!r}"
            )


def test_no_faiss_imports_in_source() -> None:
    """No file under `src/woys` or `src/audio` imports `faiss`.
    Pre-fix nothing imported it either (the dep was paid for an
    unimplemented feature), but pin so a future refactor doesn't
    silently re-add it."""
    forbidden = ["import faiss", "from faiss "]
    for subdir in ("src/woys", "src/audio", "src/tui"):
        path = REPO / subdir
        for py in path.rglob("*.py"):
            text = py.read_text()
            for needle in forbidden:
                assert needle not in text, (
                    f"F-31-01: {py.relative_to(REPO)} contains `{needle}`; "
                    f"woys is index-free, faiss must not be imported"
                )


def test_models_md_documents_index_free_contract() -> None:
    """The user-facing doc `docs/MODELS.md` must state that woys
    is index-free, so users don't expect their `.index` files to
    be loaded."""
    text = (REPO / "docs" / "MODELS.md").read_text()
    assert "index-free" in text, "docs/MODELS.md must explicitly state woys is index-free"
    assert "F-31-01" in text or "F-CX6-03" in text, (
        "docs/MODELS.md must reference the verdict that closed this"
    )
