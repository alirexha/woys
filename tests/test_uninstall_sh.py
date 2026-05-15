"""review F-merged-005: structural guard rails for uninstall.sh.

uninstall.sh can't be exercised in CI (it removes systemd units, touches
$HOME), but its *ordering* and *content* are load-bearing and easy to
regress. Mirrors test_install_sh.py's pattern: read the script as text
and pin the orderings the audit fixed.

Pre-fix uninstall.sh removed the venv before tearing down the chain, so
`woys chain disable` could not run (the binary it would invoke was
already gone) and an enabled woys-chain.service was left pointing at a
deleted binary — re-firing failed on every login until manual cleanup.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
UNINSTALL_SH = (REPO / "uninstall.sh").read_text()


def test_chain_disable_runs_before_app_dir_removal() -> None:
    """The `woys chain disable` call must run while the venv still exists.

    Otherwise the chain unit + loaded pactl modules survive the
    uninstall, leaving an enabled systemd unit pointing at a deleted
    binary that fails on every login.
    """
    chain_disable = UNINSTALL_SH.index("chain disable")
    # The destructive rmrf of $APP_HOME is the only `rm -rf "$HOME_DIR"`
    # in the script; it runs inside a for loop near the bottom.
    rmrf = UNINSTALL_SH.index('rm -rf "$HOME_DIR"')
    assert chain_disable < rmrf, (
        "`woys chain disable` must run before `rm -rf $HOME_DIR` — "
        "otherwise the binary is gone before it can disable the chain"
    )


def test_unit_cleanup_loop_includes_chain_service() -> None:
    """The systemd unit-name loop must contain `woys-chain.service`.

    This is the belt-and-suspenders pass for the case where `woys chain
    disable` could not run (binary missing, venv corrupted, etc.) and
    the unit file is orphaned.
    """
    # The loop body iterates over `unit in <names>`; the names list is
    # one line.
    loop_decl = "for unit in woys-mic.service woys-chain.service"
    assert loop_decl in UNINSTALL_SH, (
        f"expected the unit-name loop to start with `{loop_decl}` — "
        "the systemd cleanup pass must include woys-chain.service"
    )


def test_chain_service_documented_in_header() -> None:
    """The `# Removes:` block at the top of the script lists every
    surface the uninstaller deletes. The chain unit file must be there
    or the script's documented behavior diverges from what it does.
    """
    assert "woys-chain.service" in UNINSTALL_SH[: UNINSTALL_SH.index("set -euo")], (
        "the uninstall.sh header comment must mention woys-chain.service in its `# Removes:` block"
    )
