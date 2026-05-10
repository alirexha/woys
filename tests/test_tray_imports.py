"""Smoke test: `woys tray` module imports cleanly without cli.py running first.

The arch-009 bug was tray.py doing `sys.path.insert(0, str(sys.path[0]))`
which is a no-op or shadow-clobber depending on cwd; the module only
worked because cli.py inserted the right path first. This test runs
`python -c "from woys.tray import cli_tray; assert callable(cli_tray)"`
in a FRESH subprocess (no cli.main side effects) to confirm the path
resolution stands on its own.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def test_tray_module_imports_in_fresh_python() -> None:
    """A subprocess invocation of `python -c "from woys.tray import ..."`
    should succeed without any cli.py preamble. arch-009 bug = fail."""
    env = os.environ.copy()
    # Mirror the dev install's path setup: src/ on PYTHONPATH so woys is importable.
    src = str(REPO / "src")
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = src + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = src

    code = "from woys.tray import cli_tray, _engine_status; assert callable(cli_tray); assert callable(_engine_status)"
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode != 0:
        pytest.fail(
            f"tray module import failed in fresh subprocess:\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )


def test_tray_engine_status_returns_safely_with_no_running_tui() -> None:
    """`_engine_status` should return (False, error_string) - never raise -
    when there is no TUI listening on the control socket."""
    # Run in a subprocess so we don't accidentally hit a TUI the dev started.
    env = os.environ.copy()
    src = str(REPO / "src")
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = src + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = src
    # Override XDG_RUNTIME_DIR to a tempdir so the test cannot connect to a
    # real running TUI.
    env["XDG_RUNTIME_DIR"] = "/tmp/woys-tray-test-no-such-runtime-dir"

    code = (
        "from woys.tray import _engine_status\n"
        "running, reply = _engine_status()\n"
        "assert running is False, repr(running)\n"
        "assert isinstance(reply, str)\n"
        "print('OK', reply[:80])\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, (
        f"_engine_status raised in fresh subprocess:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert result.stdout.startswith("OK"), result.stdout
