"""`woys chain teardown` / `disable` / `setup` must not report success when
pactl could not unload the chain modules.

`_unload_chain_modules` used to ignore every `unload-module` return code
and return the number of modules it *tried* to unload, so a denied unload
printed `unloaded N module(s)` and exited 0 while the chain stayed loaded
(the systemd `ExecStop=woys chain teardown` logged success too). A failed
`pactl list short modules` was read as "no modules", so a dead daemon
printed `not loaded (nothing to tear down)`.
"""

from __future__ import annotations

import subprocess
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from woys import chain

CHAIN_MODULES = (
    f"10\tmodule-null-sink\tsink_name={chain.SINK_FINAL}\n"
    f"11\tmodule-ladspa-sink\tsink_name={chain.SINK_BRIDGE}\n"
)


def _cp(rc: int, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=rc, stdout=stdout, stderr=stderr)


class _Pactl:
    """subprocess.run stand-in: lists `modules`, fails the unloads in `deny`."""

    def __init__(
        self,
        modules: str = CHAIN_MODULES,
        deny: set[str] | None = None,
        list_rc: int = 0,
        unload_stderr: str = "Failure: Access denied",
    ) -> None:
        self.modules = modules
        self.deny = deny or set()
        self.list_rc = list_rc
        self.unload_stderr = unload_stderr
        self.calls: list[list[str]] = []

    def __call__(self, cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(cmd))
        if cmd[:4] == ["pactl", "list", "short", "modules"]:
            if self.list_rc != 0:
                return _cp(self.list_rc, stderr="Connection failure: Connection refused")
            return _cp(0, self.modules)
        if cmd[:4] == ["pactl", "list", "short", "sources"]:
            return _cp(0, "1\twoys-mic\tdrv\t1\tIDLE\n")
        if cmd[:2] == ["pactl", "unload-module"]:
            if cmd[2] in self.deny:
                return _cp(1, stderr=self.unload_stderr)
            return _cp(0)
        return _cp(0)


@pytest.fixture(autouse=True)
def _pactl_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chain.shutil, "which", lambda name: f"/usr/bin/{name}")


def test_teardown_fails_when_unload_is_denied(capsys: pytest.CaptureFixture[str]) -> None:
    router = _Pactl(deny={"10", "11"})
    with (
        patch.object(chain.subprocess, "run", side_effect=router),
        patch("audio.pipewire.relabel_source") as relabel,
    ):
        rc = chain.teardown()
    out = capsys.readouterr()
    assert rc != 0
    assert "unloaded 2 module(s)" not in out.out
    assert "Access denied" in out.err
    # The chain is still loaded, so woys-mic keeps its chain-active label.
    relabel.assert_not_called()


def test_teardown_fails_when_one_of_several_unloads_is_denied(
    capsys: pytest.CaptureFixture[str],
) -> None:
    router = _Pactl(deny={"11"})
    with (
        patch.object(chain.subprocess, "run", side_effect=router),
        patch("audio.pipewire.relabel_source"),
    ):
        rc = chain.teardown()
    assert rc != 0
    assert "11" in capsys.readouterr().err


def test_teardown_treats_already_gone_module_as_unloaded() -> None:
    # A module that vanished between the listing and the unload is the
    # state teardown wants; pactl reports it as "No such entity".
    router = _Pactl(deny={"10"}, unload_stderr="Failure: No such entity")
    with (
        patch.object(chain.subprocess, "run", side_effect=router),
        patch("audio.pipewire.relabel_source"),
    ):
        assert chain.teardown() == 0


def test_teardown_fails_when_module_list_fails(capsys: pytest.CaptureFixture[str]) -> None:
    router = _Pactl(list_rc=1)
    with (
        patch.object(chain.subprocess, "run", side_effect=router),
        patch("audio.pipewire.relabel_source"),
    ):
        rc = chain.teardown()
    out = capsys.readouterr()
    assert rc != 0
    assert "nothing to tear down" not in out.out


def test_disable_fails_when_unload_is_denied(capsys: pytest.CaptureFixture[str]) -> None:
    router = _Pactl(deny={"10", "11"})
    unit = MagicMock()
    unit.is_file.return_value = False
    with (
        patch.object(chain, "_systemd_unit_path", return_value=unit),
        patch.object(chain.subprocess, "run", side_effect=router),
    ):
        rc = chain.disable()
    assert rc != 0
    assert "unloaded 2 module(s)" not in capsys.readouterr().out


def test_disable_fails_when_module_list_fails() -> None:
    router = _Pactl(list_rc=1)
    unit = MagicMock()
    unit.is_file.return_value = False
    with (
        patch.object(chain, "_systemd_unit_path", return_value=unit),
        patch.object(chain.subprocess, "run", side_effect=router),
    ):
        assert chain.disable() != 0


def test_setup_aborts_when_stale_chain_cannot_be_cleared() -> None:
    """A failed stale clear used to fall through and load a second copy of
    every chain module next to the stuck one."""
    router = _Pactl(deny={"10"})
    with (
        patch.object(chain.Path, "is_file", lambda self: True),
        patch.object(chain.subprocess, "run", side_effect=router),
        patch("audio.pipewire.relabel_source"),
    ):
        rc = chain.setup()
    assert rc == 2
    assert not [c for c in router.calls if c[:2] == ["pactl", "load-module"]]


def test_status_survives_failed_module_list(capsys: pytest.CaptureFixture[str]) -> None:
    router = _Pactl(list_rc=1)
    unit = MagicMock()
    unit.is_file.return_value = False
    with (
        patch.object(chain, "_systemd_unit_path", return_value=unit),
        patch.object(chain.subprocess, "run", side_effect=router),
    ):
        chain.status()
    assert "Connection refused" in capsys.readouterr().err
