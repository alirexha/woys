"""review: structural guard rails for install.sh.

The real install builds a venv, downloads ~1 GiB of weights and touches
systemd, so it can't run in CI. Two kinds of tests cover it instead: text
checks that pin the orderings the audit fixed, and `_run_install` runs the
real script in a sandbox HOME with stub uv / pactl / nvidia-smi /
systemctl / gcc / make on a PATH that holds nothing else.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
INSTALL_SH = (REPO / "install.sh").read_text()

# Real tools install.sh needs besides the stubbed ones.
_SYSTEM_TOOLS = [
    "awk",
    "basename",
    "cat",
    "chmod",
    "cp",
    "dirname",
    "env",
    "find",
    "grep",
    "head",
    "id",
    "install",
    "ln",
    "ls",
    "mkdir",
    "mv",
    "readlink",
    "rm",
    "sed",
    "sh",
    "sort",
    "stat",
    "tail",
    "touch",
    "tr",
    "wc",
    "xargs",
]


def _write_stub(path: Path, body: str) -> None:
    path.write_text("#!/bin/sh\n" + body + "\n")
    path.chmod(0o755)


@dataclass
class InstallRun:
    rc: int
    out: str
    home: Path
    calls: str  # one line per stub call, "<tool> <args>"


def _run_install(
    tmp_path: Path,
    *args: str,
    stubs: dict[str, str | None] | None = None,
    setup_home: object = None,
) -> InstallRun:
    """Run install.sh from a throwaway copy of the repo layout.

    `stubs` overrides (body) or removes (None) a default stub. The venv
    python that the stub `uv venv` lays down forwards to this interpreter,
    except `-m pip`, which fails like it does in a real uv venv."""
    home = tmp_path / "home dir"  # a space, to catch unquoted paths
    home.mkdir()
    if callable(setup_home):
        setup_home(home)
    repo = tmp_path / "repo"
    repo.mkdir()
    shutil.copy(REPO / "install.sh", repo / "install.sh")
    (repo / "scripts").symlink_to(REPO / "scripts")
    (repo / "pkg").symlink_to(REPO / "pkg")
    (repo / "bin").mkdir()
    log = tmp_path / "calls.log"
    log.touch()

    sysbin = tmp_path / "sysbin"
    sysbin.mkdir()
    for tool in _SYSTEM_TOOLS:
        found = shutil.which(tool)
        assert found, f"test host lacks {tool}"
        (sysbin / tool).symlink_to(found)

    tmpl = tmp_path / "venv-template"
    tmpl.mkdir()
    _write_stub(
        tmpl / "python",
        'if [ "$1" = "-m" ] && [ "$2" = "pip" ]; then\n'
        '    echo "No module named pip" >&2; exit 1\nfi\n'
        f'exec "{sys.executable}" "$@"',
    )
    _write_stub(
        tmpl / "woys",
        f'echo "woys $*" >> "{log}"\n[ "$1" = "--version" ] && echo "woys 0.0.0-test"\nexit 0',
    )

    stub_dir = tmp_path / "stubs"
    stub_dir.mkdir()
    default_stubs: dict[str, str | None] = {
        "pactl": (
            'case "$1" in info) echo "Server Name: PulseAudio (on PipeWire 1.0.0)";; esac\nexit 0'
        ),
        "nvidia-smi": "exit 0",
        "uv": (
            f'echo "uv $*" >> "{log}"\n'
            'if [ "$1" = "venv" ]; then\n'
            '    for a; do v="$a"; done\n'
            '    mkdir -p "$v/bin"\n'
            f'    cp "{tmpl}/python" "{tmpl}/woys" "$v/bin/"\n'
            "fi\nexit 0"
        ),
        "gcc": "exit 0",
        "pkg-config": "exit 0",
        "make": (f'echo "make $*" >> "{log}"\n[ "$1" = "-C" ] && touch "$2/woys-pw-out"\nexit 0'),
        "systemctl": f'echo "systemctl $*" >> "{log}"\nexit 0',
    }
    default_stubs.update(stubs or {})
    for name, body in default_stubs.items():
        if body is not None:
            _write_stub(stub_dir / name, body)

    env = {"HOME": str(home), "PATH": f"{stub_dir}:{sysbin}"}
    proc = subprocess.run(
        [shutil.which("bash") or "/bin/bash", str(repo / "install.sh"), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return InstallRun(
        rc=proc.returncode,
        out=proc.stdout + proc.stderr,
        home=home,
        calls=log.read_text(),
    )


def test_sandbox_install_succeeds(tmp_path: Path) -> None:
    """The harness itself: a host with every prerequisite installs cleanly."""
    run = _run_install(tmp_path, "--skip-models")
    assert run.rc == 0, run.out
    assert "[install] done." in run.out
    assert (run.home / ".local" / "bin" / "woys-pw-out").is_file()


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
        ("gcc", "exit 0"),
        ("make", "exit 0"),
        ("pkg-config", "exit 0"),
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


def test_install_fails_without_gcc(tmp_path: Path) -> None:
    """woys-pw-out is the default playback backend (prefer_native_pw=True)
    and the engine refuses to start without it. Pre-fix a missing gcc was a
    warning and install.sh still printed "done" and exited 0."""
    run = _run_install(tmp_path, "--skip-models", "--no-systemd", stubs={"gcc": None})
    assert run.rc == 1, run.out
    assert "gcc" in run.out and "[install] done." not in run.out
    # Fails before the multi-GB venv build, not after it.
    assert "uv pip install" not in run.calls


def test_install_fails_without_pipewire_headers(tmp_path: Path) -> None:
    run = _run_install(tmp_path, "--skip-models", "--no-systemd", stubs={"pkg-config": "exit 1"})
    assert run.rc == 1, run.out
    assert "libpipewire-0.3" in run.out and "[install] done." not in run.out


def test_install_fails_when_the_helper_build_fails(tmp_path: Path) -> None:
    run = _run_install(tmp_path, "--skip-models", "--no-systemd", stubs={"make": "exit 2"})
    assert run.rc == 1, run.out
    assert "woys-pw-out" in run.out and "[install] done." not in run.out


def _legacy_install(home: Path) -> None:
    share = home / ".local" / "share" / "vcclient-cachy"
    (share / "models").mkdir(parents=True)
    (share / "venv" / "bin").mkdir(parents=True)
    (share / "models" / "myvoice.onnx").write_bytes(b"voice")
    cfg = home / ".config" / "vcclient-cachy"
    cfg.mkdir(parents=True)
    (cfg / "config.toml").write_text(f'rvc_model = "{share}/models/myvoice.onnx"\n')


def test_install_migrates_legacy_models_into_the_new_share_dir(tmp_path: Path) -> None:
    """install.sh builds the woys venv before migrating, so the migrator
    finds ~/.local/share/woys already there. The legacy voice must still
    end up where the rewritten config points, and the legacy dir must be
    gone so the next install does not migrate again."""
    run = _run_install(tmp_path, "--skip-models", "--no-systemd", setup_home=_legacy_install)
    assert run.rc == 0, run.out
    new_model = run.home / ".local" / "share" / "woys" / "models" / "myvoice.onnx"
    assert new_model.read_bytes() == b"voice"
    cfg = (run.home / ".config" / "woys" / "config.toml").read_text()
    assert str(new_model) in cfg
    assert not (run.home / ".local" / "share" / "vcclient-cachy").exists()


def test_installed_woys_mic_unit_runs_without_local_bin_on_path(tmp_path: Path) -> None:
    """systemd's user manager does not read shell rc files, so its PATH
    usually lacks ~/.local/bin. Pre-fix the unit ran `/usr/bin/env woys`,
    which exits 127 there, and woys-mic never came up at login."""
    run = _run_install(tmp_path, "--skip-models")
    assert run.rc == 0, run.out
    unit = (run.home / ".config" / "systemd" / "user" / "woys-mic.service").read_text()
    systemd_path = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin"
    execs = [ln for ln in unit.splitlines() if ln.startswith(("ExecStart", "ExecStop"))]
    assert len(execs) == 3, unit
    for line in execs:
        argv = shlex.split(line.split("=", 1)[1].replace("%%", "%"))
        proc = subprocess.run(argv, env={"HOME": str(run.home), "PATH": systemd_path})
        assert proc.returncode == 0, line
    assert "woys pw setup" in (tmp_path / "calls.log").read_text()


def test_install_help_prints_the_whole_header() -> None:
    out = subprocess.run(
        ["bash", str(REPO / "install.sh"), "--help"], capture_output=True, text=True, check=True
    ).stdout
    assert "libpipewire-0.3" in out and "--no-systemd" in out
    assert "set -euo" not in out
