"""review: structural guard rails for install.sh.

install.sh can't be exercised in CI (it builds a venv, downloads ~1 GiB of
weights, touches systemd) — but its *ordering* is load-bearing and easy to
regress. These tests read the script as text and pin the orderings the
audit fixed, the same way `test_engine_config_drift.py` AST-pins cli.py.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
INSTALL_SH = (REPO / "install.sh").read_text()


def test_prereqs_and_venv_build_run_before_destructive_migration() -> None:
    """the destructive vcclient-cachy -> woys migration
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
    """install the pinned dependency closure
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


def test_install_sh_hard_fails_on_missing_nvidia_smi() -> None:
    """a missing NVIDIA GPU must hard-fail UNCONDITIONALLY -- not
    warn-and-continue, and with no opt-out flag that promises a CPU path
    the engine does not actually provide (woys is GPU-only)."""
    # The pre-fix warn-and-continue line must be gone.
    assert "engine will fall back to CPU" not in INSTALL_SH, (
        "the misleading 'fall back to CPU' warning must be removed"
    )
    # The nvidia-smi block must hard-fail with no conditional opt-out.
    start = INSTALL_SH.index("command -v nvidia-smi")
    block = INSTALL_SH[start : INSTALL_SH.index("\n\n", start)]
    assert "fail " in block, "a missing nvidia-smi must call fail()"
    assert "ALLOW_CPU" not in block, "the hard-fail must be unconditional"
    # The broken --allow-cpu opt-out (accepted, printed success, but never
    # threaded to the engine -> guaranteed CpuFallbackError at first run)
    # must be gone entirely.
    assert "--allow-cpu" not in INSTALL_SH, "the --allow-cpu flag must be removed"
    assert "ALLOW_CPU" not in INSTALL_SH, "the ALLOW_CPU variable must be removed"


def test_install_sh_verifies_all_three_foundation_weights() -> None:
    """the install must verify ALL three foundation
    weights, not just amitaro_v2_16k.onnx."""
    for weight in ("rmvpe_wrapped.onnx", "contentvec-f.onnx", "amitaro_v2_16k.onnx"):
        assert weight in INSTALL_SH, f"install.sh must verify the {weight} foundation weight"


def _run_install_until_uv(tmp_path: Path, uv_dir: Path | None, env_uv_bin: str | None) -> str:
    """Run install.sh in a sandboxed HOME with stub host tools until the first
    uv call. The stub uv exits 42 with a marker, so reaching it proves the
    lookup resolved; the real `$HOME/.local/bin/uv` is never on PATH."""
    home = tmp_path / "home"
    home.mkdir()
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    for name, body in (
        ("pactl", 'echo "Server Name: PulseAudio (on PipeWire 1.0.0)"'),
        ("nvidia-smi", "exit 0"),
    ):
        (stubs / name).write_text(f"#!/bin/sh\n{body}\n")
        (stubs / name).chmod(0o755)
    if uv_dir is not None:
        uv_dir.mkdir(parents=True, exist_ok=True)
        (uv_dir / "uv").write_text("#!/bin/sh\necho STUB-UV-CALLED >&2\nexit 42\n")
        (uv_dir / "uv").chmod(0o755)
    path_dirs = [str(stubs)]
    if uv_dir is not None and env_uv_bin is None:
        path_dirs.append(str(uv_dir))
    path_dirs += ["/usr/bin", "/bin"]
    env = {"HOME": str(home), "PATH": os.pathsep.join(path_dirs)}
    if env_uv_bin is not None:
        env["UV_BIN"] = env_uv_bin
    proc = subprocess.run(
        ["bash", str(REPO / "install.sh"), "--skip-models", "--no-systemd"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return f"rc={proc.returncode}\n{proc.stdout}\n{proc.stderr}"


def test_install_finds_uv_on_path(tmp_path: Path) -> None:
    """uv from the distro package manager (e.g. /usr/bin/uv) is not at
    ~/.local/bin/uv. Pre-fix install.sh only looked there and failed with
    'uv is required but not found' for those users."""
    out = _run_install_until_uv(tmp_path, tmp_path / "pkg-bin", None)
    assert "STUB-UV-CALLED" in out and "rc=42" in out, out


def test_install_honors_uv_bin_override(tmp_path: Path) -> None:
    """A UV_BIN the user exported wins over PATH lookup."""
    custom = tmp_path / "custom"
    out = _run_install_until_uv(tmp_path, custom, str(custom / "uv"))
    assert "STUB-UV-CALLED" in out and "rc=42" in out, out


def test_install_without_uv_fails_with_hint(tmp_path: Path) -> None:
    out = _run_install_until_uv(tmp_path, None, None)
    assert "rc=1" in out and "uv (Astral) is required" in out, out
