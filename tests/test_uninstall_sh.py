"""structural guard rails for uninstall.sh.

Text checks pin the orderings the audit fixed, and `_run_uninstall` runs
the real script in a sandbox HOME with stub systemctl / pactl on a PATH
that holds nothing else (the same pattern as test_install_sh.py).

Pre-fix uninstall.sh removed the venv before tearing down the chain, so
`woys chain disable` could not run (the binary it would invoke was
already gone) and an enabled woys-chain.service was left pointing at a
deleted binary — re-firing failed on every login until manual cleanup.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
UNINSTALL_SH = (REPO / "uninstall.sh").read_text()

_SYSTEM_TOOLS = ["cat", "find", "grep", "id", "rm", "rmdir", "sed", "sh", "stat"]


def _run_uninstall(tmp_path: Path, *args: str) -> tuple[int, str, Path]:
    """Run uninstall.sh against a sandbox HOME laid out like a real install
    (venv, foundation weights, one user voice, launcher, helper, units)."""
    home = tmp_path / "home dir"  # a space, to catch unquoted paths
    app = home / ".local" / "share" / "woys"
    (app / "venv" / "bin").mkdir(parents=True)
    (app / "models").mkdir()
    (app / "models" / "rmvpe_wrapped.onnx").write_bytes(b"w")
    (app / "models" / "my_voice.onnx").write_bytes(b"v")
    local_bin = home / ".local" / "bin"
    local_bin.mkdir(parents=True)
    for name in ("woys", "woys-pw-out"):
        (local_bin / name).write_text("#!/bin/sh\n")
    units = home / ".config" / "systemd" / "user"
    units.mkdir(parents=True)
    (units / "woys-mic.service").write_text("[Unit]\n")
    (home / ".config" / "woys").mkdir()
    (home / ".config" / "woys" / "config.toml").write_text("f0_up_key = 0\n")

    sysbin = tmp_path / "sysbin"
    sysbin.mkdir()
    for tool in _SYSTEM_TOOLS:
        found = shutil.which(tool)
        assert found, f"test host lacks {tool}"
        (sysbin / tool).symlink_to(found)
    for name in ("systemctl", "pactl"):
        (sysbin / name).write_text("#!/bin/sh\nexit 1\n")
        (sysbin / name).chmod(0o755)

    proc = subprocess.run(
        [shutil.which("bash") or "/bin/bash", str(REPO / "uninstall.sh"), *args],
        env={"HOME": str(home), "PATH": str(sysbin)},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return proc.returncode, proc.stdout + proc.stderr, home


def test_uninstall_keeps_voice_models_by_default(tmp_path: Path) -> None:
    """~/.local/share/woys/models is where users put their own voices
    (docs/MODELS.md). Pre-fix a plain ./uninstall.sh deleted it without
    asking, although TROUBLESHOOTING's reset recipe said it was kept."""
    rc, out, home = _run_uninstall(tmp_path)
    assert rc == 0, out
    app = home / ".local" / "share" / "woys"
    assert (app / "models" / "my_voice.onnx").is_file(), out
    assert not (app / "venv").exists()
    assert (home / ".config" / "woys" / "config.toml").is_file()


def test_uninstall_purge_models_removes_the_app_dir(tmp_path: Path) -> None:
    rc, out, home = _run_uninstall(tmp_path, "--purge-models")
    assert rc == 0, out
    assert not (home / ".local" / "share" / "woys").exists()


def test_uninstall_removes_the_native_helper(tmp_path: Path) -> None:
    """install.sh installs ~/.local/bin/woys-pw-out. Pre-fix uninstall left
    it on PATH, where the engine picks it up first on a later reinstall."""
    rc, out, home = _run_uninstall(tmp_path)
    assert rc == 0, out
    assert not (home / ".local" / "bin" / "woys-pw-out").exists()
    assert not (home / ".local" / "bin" / "woys").exists()
    # What it leaves behind on purpose is spelled out.
    assert ".cache/woys" in out and ".local/state/woys" in out


def test_uninstall_still_accepts_keep_models(tmp_path: Path) -> None:
    rc, out, home = _run_uninstall(tmp_path, "--keep-models")
    assert rc == 0, out
    assert (home / ".local" / "share" / "woys" / "models" / "my_voice.onnx").is_file()


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
