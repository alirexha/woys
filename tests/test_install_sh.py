"""review: structural guard rails for install.sh.

install.sh can't be exercised in CI (it builds a venv, downloads ~1 GiB of
weights, touches systemd) — but its *ordering* is load-bearing and easy to
regress. These tests read the script as text and pin the orderings the
audit fixed, the same way `test_engine_config_drift.py` AST-pins cli.py.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
INSTALL_SH = (REPO / "install.sh").read_text()


def test_prereqs_and_venv_build_run_before_destructive_migration() -> None:
    """review F-19-05: the destructive vcclient-cachy -> woys migration
    must run *after* the prereq checks and the venv + deps build.

    Pre-fix the migration ran first, so a `set -e` abort on a venv-build
    failure left the old install dismantled and the new one unbuilt.
    """
    migrate_pos = INSTALL_SH.index("migrate_to_woys.py")
    pactl_check = INSTALL_SH.index("command -v pactl")
    venv_build = INSTALL_SH.index("pip install --python")

    assert pactl_check < migrate_pos, (
        "the pactl/PipeWire prereq check must run before the migration"
    )
    assert venv_build < migrate_pos, (
        "the venv + deps build must complete before the destructive migration"
    )


def test_pinned_requirements_install_before_editable_no_deps_package() -> None:
    """review F-19-03: install the pinned dependency closure
    (requirements.txt) first, then the woys package with `--no-deps`.

    Pre-fix it was `pip install -e .` then `pip install -r
    requirements.txt` -- an order-dependent double-install where the second
    command silently re-resolved the first, and the slow torch + ORT-GPU
    step was paid twice.
    """
    req_install = INSTALL_SH.index("pip install --python")
    # The line that installs the editable package.
    editable_install = INSTALL_SH.index('-e "$REPO_DIR"')

    assert req_install < editable_install, (
        "requirements.txt (the pinned closure) must be installed before `-e .`"
    )
    # The editable install must be --no-deps so it doesn't re-resolve the
    # dependency set requirements.txt just pinned.
    editable_line = next(
        ln for ln in INSTALL_SH.splitlines() if '-e "$REPO_DIR"' in ln and "pip install" in ln
    )
    assert "--no-deps" in editable_line, (
        "the editable `-e .` install must pass --no-deps -- requirements.txt "
        "owns the dependency set"
    )
