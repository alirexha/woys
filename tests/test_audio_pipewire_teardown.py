"""`woys pw teardown` must not print "woys-mic removed." while the modules
are still loaded.

VirtualMic.teardown ignored the `pactl unload-module` return codes, so a
denied unload let `cmd_pw_teardown` print success and exit 0. Pure unit
tests over a stateful fake pactl; no PipeWire daemon is touched.
"""

from __future__ import annotations

import subprocess

import pytest

from audio import pipewire
from woys import cli


class _FakePactl:
    """Keeps a module table; `unload-module` of an id in `deny` fails."""

    def __init__(self, deny: set[int], stderr: str = "Failure: Access denied") -> None:
        self.modules: dict[int, tuple[str, str]] = {
            7: ("module-null-sink", f"sink_name={pipewire.SINK_NAME}"),
            8: ("module-remap-source", f"source_name={pipewire.SOURCE_NAME}"),
        }
        self.deny = deny
        self.stderr = stderr

    def __call__(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        if args[:3] == ["list", "short", "modules"]:
            rows = "".join(f"{i}\t{n}\t{a}\n" for i, (n, a) in self.modules.items())
            return subprocess.CompletedProcess(args, 0, rows, "")
        if args[0] == "unload-module":
            mod_id = int(args[1])
            if mod_id in self.deny:
                return subprocess.CompletedProcess(args, 1, "", self.stderr)
            self.modules.pop(mod_id, None)
            return subprocess.CompletedProcess(args, 0, "", "")
        return subprocess.CompletedProcess(args, 0, "", "")


@pytest.fixture(autouse=True)
def _no_pw_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipewire, "_destroy_orphan_nodes", lambda: None)


def test_teardown_raises_when_unload_is_denied(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakePactl(deny={7, 8})
    monkeypatch.setattr(pipewire, "_run_pactl", fake)
    with pytest.raises(pipewire.PipeWireError, match="Access denied"):
        pipewire.VirtualMic().teardown()


def test_teardown_raises_when_only_the_sink_stays(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakePactl(deny={7})
    monkeypatch.setattr(pipewire, "_run_pactl", fake)
    with pytest.raises(pipewire.PipeWireError, match="7"):
        pipewire.VirtualMic().teardown()
    assert 8 not in fake.modules


def test_teardown_succeeds_when_modules_are_gone(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakePactl(deny=set())
    monkeypatch.setattr(pipewire, "_run_pactl", fake)
    pipewire.VirtualMic().teardown()
    assert fake.modules == {}


def test_cli_pw_teardown_exits_2_on_denied_unload(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(pipewire, "_run_pactl", _FakePactl(deny={7, 8}))
    assert cli.cmd_pw_teardown() == 2
    assert "woys-mic removed." not in capsys.readouterr().out
