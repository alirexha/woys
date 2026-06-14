"""bug-class test.

Pre-fix `chain._pactl` was a thin `subprocess.run(["pactl", ...])`
with NO `shutil.which` existence check, while its twin
`audio.pipewire._run_pactl` already had one. On a system without
`pactl` installed, every chain entry-point (status / setup / teardown
/ enable / disable) raised `FileNotFoundError` from inside the
subprocess call -- with no typed error message and no graceful
exit path. Post-fix the chain wrapper returns
`CompletedProcess(returncode=127, stderr="pactl not found - ...")`,
which is the shape every chain caller already handles.

This test pins the post-fix contract:

  1. When `pactl` is on PATH: chain._pactl runs it (subprocess.run is
     reached, called with bare `"pactl"` as `argv[0]` so the test
     router can pattern-match).
  2. When `pactl` is NOT on PATH: chain._pactl returns a typed
     CompletedProcess (returncode=127, stderr names pactl) without
     raising and without reaching `subprocess.run`.

A pre-fix bisect would fail leg #2: the wrapper had no `shutil.which`
gate, so `subprocess.run` was always called, and on a no-pactl
system it raised `FileNotFoundError` before any returncode could be
returned.
"""

from __future__ import annotations

import subprocess
from unittest.mock import patch

from woys import chain


def test_pactl_wrapper_runs_pactl_when_on_path() -> None:
    """Sanity / post-fix leg 1: when shutil.which finds pactl, the
    wrapper actually runs the subprocess and uses the bare "pactl"
    argv[0] (NOT the absolute path) so chain tests' routers can
    pattern-match on it."""
    seen: dict[str, list[str]] = {}

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok\n", stderr="")

    with (
        patch.object(chain.shutil, "which", return_value="/usr/bin/pactl"),
        patch.object(chain.subprocess, "run", side_effect=fake_run),
    ):
        out = chain._pactl("get-default-sink")

    assert out.returncode == 0
    # Pre-fix would have passed this trivially because there was no
    # which() call. Post-fix the critical contract: the call uses
    # bare "pactl", not the absolute path, so test routers keying on
    # cmd[0] == "pactl" continue to match.
    assert seen["cmd"][0] == "pactl"
    assert seen["cmd"][1] == "get-default-sink"


def test_pactl_wrapper_returns_127_when_pactl_missing() -> None:
    """Post-fix leg 2 — the load-bearing one. Pre-fix had NO
    shutil.which gate, so `subprocess.run` was always called; on a
    no-pactl host that raised FileNotFoundError out of the wrapper.
    Post-fix the wrapper returns a typed CompletedProcess instead."""
    called = {"run": False}

    def fake_run(*a: object, **kw: object) -> subprocess.CompletedProcess[str]:
        called["run"] = True
        # Pre-fix this stand-in for subprocess.run would have been
        # reached and would have raised FileNotFoundError on the
        # real host. Post-fix we should never get here.
        raise FileNotFoundError("pactl")

    with (
        patch.object(chain.shutil, "which", return_value=None),
        patch.object(chain.subprocess, "run", side_effect=fake_run),
    ):
        out = chain._pactl("info")

    assert out.returncode == 127
    assert "pactl not found" in out.stderr
    assert out.args == ["pactl", "info"]
    # The critical post-fix assertion: shutil.which gated the call,
    # so subprocess.run never ran -- pre-fix this would be True.
    assert called["run"] is False


def test_pactl_wrapper_matches_pipewire_helper_missing_tool_message() -> None:
    """The two wrappers no longer diverge on missing-tool semantics.
    chain._pactl's stderr message mirrors the wording in
    audio.pipewire._run_pactl's PipeWireError so a future caller
    that switches helpers gets the same user-facing string."""
    from audio import pipewire as pw_mod

    with patch.object(chain.shutil, "which", return_value=None):
        out = chain._pactl("info")
    assert "pipewire-pulse" in out.stderr

    # Mirror in pipewire.py (it raises; we read the raised message).
    with patch.object(pw_mod.shutil, "which", return_value=None):
        try:
            pw_mod._run_pactl(["info"])
        except pw_mod.PipeWireError as e:
            assert "pipewire-pulse" in str(e)
        else:
            raise AssertionError("expected PipeWireError when pactl missing")
