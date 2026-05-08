"""v0.13.2 — guard rails for the RNNoise chain module.

These tests do NOT exercise pactl/pipewire-pulse — they would need a
real pipewire-pulse session and the LADSPA plugin installed, neither of
which we want CI to depend on. They DO lock in the topology decisions
that fix v0.13.0's leak so a future refactor can't quietly regress:

  * woys-mic-clean uses media.class=Audio/Sink (NOT Audio/Source/Virtual).
  * Both legs of the chain are mono (channels=1) end to end.
  * Setup unloads any stale chain modules before loading new ones.
  * Failed mid-load tears the chain back down.
"""

from __future__ import annotations

import subprocess
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from woys import chain


def _ok(stdout: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


def _err(stderr: str = "boom") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr=stderr)


class _PactlRouter:
    """Mock for subprocess.run that routes by the command list it receives.

    subprocess.run is called positionally with a single list (the argv),
    so the mock signature is `(cmd, **kwargs)` — NOT `*args`.
    """

    def __init__(
        self,
        sources: str = "",
        modules: str = "",
        load_results: list[subprocess.CompletedProcess[str]] | None = None,
        modules_after_load: dict[int, str] | None = None,
        pwlink_output: str = "",
    ) -> None:
        self.sources = sources
        self.modules = modules
        self.load_results = load_results or []
        self.modules_after_load = modules_after_load or {}
        self.pwlink_output = pwlink_output
        self.calls: list[list[str]] = []
        self._load_count = 0

    def __call__(self, cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(cmd))
        if cmd[0] == "pactl":
            if cmd[1:3] == ["list", "short"]:
                if cmd[3] == "sources":
                    return _ok(self.sources)
                if cmd[3] == "modules":
                    return _ok(self.modules)
            if cmd[1] == "load-module":
                idx = self._load_count
                self._load_count += 1
                # Fast-forward modules listing if the test wants the
                # 'list short modules' AFTER this load to differ from
                # what it returned before.
                if idx in self.modules_after_load:
                    self.modules = self.modules_after_load[idx]
                if idx < len(self.load_results):
                    return self.load_results[idx]
                return _ok()
            if cmd[1] == "unload-module":
                return _ok()
        if cmd[0].endswith("pw-link"):
            return _ok(self.pwlink_output)
        if cmd[0] == "systemctl":
            return _ok()
        return _ok()


def test_setup_loads_audio_sink_class_and_mono_chain() -> None:
    """The fix: media.class=Audio/Sink + channels=1 throughout. Regression guard."""
    router = _PactlRouter(
        sources="1\twoys-mic\tdummy-driver\t1\tIDLE\n",
        modules="",  # no stale chain
    )

    with (
        patch.object(chain.Path, "is_file", lambda self: True),
        patch.object(chain.subprocess, "run", side_effect=router),
    ):
        rc = chain.setup()

    assert rc == 0
    loads = [c for c in router.calls if len(c) >= 2 and c[1] == "load-module"]
    assert len(loads) == 3, f"expected 3 load-module calls, got {loads}"

    null_sink, ladspa_sink, loopback = loads

    # 1. Null-sink: must be Audio/Sink (NOT Audio/Source/Virtual — that
    #    was the v0.13.0 bug).
    assert "module-null-sink" in null_sink
    assert "media.class=Audio/Sink" in null_sink
    assert f"sink_name={chain.SINK_FINAL}" in null_sink
    assert "channels=1" in null_sink

    # 2. LADSPA-sink: mono, sink_master points at woys-mic-clean,
    #    plugin label is the mono variant.
    assert "module-ladspa-sink" in ladspa_sink
    assert f"sink_master={chain.SINK_FINAL}" in ladspa_sink
    assert f"sink_name={chain.SINK_BRIDGE}" in ladspa_sink
    assert "label=noise_suppressor_mono" in ladspa_sink
    assert "channels=1" in ladspa_sink

    # 3. Loopback: feeds woys-mic into the bridge, mono.
    assert "module-loopback" in loopback
    assert f"source={chain.SOURCE_RAW}" in loopback
    assert f"sink={chain.SINK_BRIDGE}" in loopback
    assert "channels=1" in loopback


def test_setup_refuses_when_plugin_missing() -> None:
    with patch.object(chain.Path, "is_file", lambda self: False):
        rc = chain.setup()
    assert rc == 2


def test_setup_refuses_when_woys_mic_absent() -> None:
    router = _PactlRouter(sources="99\tsome-other-source\tdrv\t1\tIDLE\n")
    with (
        patch.object(chain.Path, "is_file", lambda self: True),
        patch.object(chain.subprocess, "run", side_effect=router),
    ):
        rc = chain.setup()
    assert rc == 2


def test_setup_unloads_stale_chain_before_loading() -> None:
    """Idempotency: if an old (potentially v0.13.0-broken) chain is
    already loaded, setup must clear it first instead of stacking
    duplicates."""
    stale = (
        f"42\tmodule-null-sink\tmedia.class=Audio/Source/Virtual sink_name={chain.SINK_FINAL}\n"
        f"43\tmodule-ladspa-sink\tsink_name={chain.SINK_BRIDGE} sink_master={chain.SINK_FINAL}\n"
        f"44\tmodule-loopback\tsource={chain.SOURCE_RAW} sink={chain.SINK_BRIDGE}\n"
    )
    router = _PactlRouter(
        sources="1\twoys-mic\tdrv\t1\tIDLE\n",
        modules=stale,
        # After the 3 load-module calls, modules listing reverts to
        # empty so any leakage isn't double-counted as 'stale again'.
        modules_after_load={0: ""},
    )

    with (
        patch.object(chain.Path, "is_file", lambda self: True),
        patch.object(chain.subprocess, "run", side_effect=router),
    ):
        chain.setup()

    unload_ids = [c[2] for c in router.calls if c[:2] == ["pactl", "unload-module"]]
    assert {"42", "43", "44"} <= set(unload_ids), (
        f"expected stale 42/43/44 unloaded, got {unload_ids}"
    )


def test_setup_rolls_back_when_ladspa_load_fails() -> None:
    """If module-ladspa-sink fails (e.g. label mismatch), the null-sink
    we already loaded must be cleaned up — otherwise the user is left
    with an orphan woys-mic-clean sink that LOOKS routable but isn't."""
    null_sink_listing = (
        f"77\tmodule-null-sink\tmedia.class=Audio/Sink sink_name={chain.SINK_FINAL}\n"
    )
    router = _PactlRouter(
        sources="1\twoys-mic\tdrv\t1\tIDLE\n",
        modules="",
        # Loads in order: null-sink (ok), ladspa-sink (FAIL).
        load_results=[_ok(), _err("invalid label")],
        # After the null-sink load, the modules listing should show 77
        # so the rollback's _unload_chain_modules() can find it.
        modules_after_load={0: null_sink_listing},
    )

    with (
        patch.object(chain.Path, "is_file", lambda self: True),
        patch.object(chain.subprocess, "run", side_effect=router),
    ):
        rc = chain.setup()

    assert rc == 2
    unload_ids = [c[2] for c in router.calls if c[:2] == ["pactl", "unload-module"]]
    assert "77" in unload_ids, (
        f"expected null-sink (77) rollback after ladspa-sink failure, got {unload_ids}"
    )


def test_alsa_leak_links_flags_filter_chain_to_alsa() -> None:
    """Self-check used by status: if pw-link shows the LADSPA filter-chain
    output linked to alsa_output (= the v0.13.0 bug), report it."""
    pwlink_output = (
        "output.filter-chain-1803-15\n"
        "  |-> alsa_output.pci-0000_00_1f.3.analog-stereo:playback_FL\n"
        "  |-> alsa_output.pci-0000_00_1f.3.analog-stereo:playback_FR\n"
        "some-other-node\n"
        "  |-> some-non-alsa-input\n"
    )
    router = _PactlRouter(pwlink_output=pwlink_output)
    with (
        patch("shutil.which", return_value="/usr/bin/pw-link"),
        patch.object(chain.subprocess, "run", side_effect=router),
    ):
        leaks = chain._alsa_leak_links()
    assert len(leaks) == 2
    assert all("alsa_output" in row for row in leaks)
    assert all("filter-chain" in row for row in leaks)


def test_alsa_leak_links_returns_empty_when_pwlink_missing() -> None:
    with patch("shutil.which", return_value=None):
        assert chain._alsa_leak_links() == []


def test_systemd_unit_path_respects_xdg_config_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    p = chain._systemd_unit_path()
    assert p == tmp_path / "systemd" / "user" / "woys-chain.service"


def test_disable_no_unit_still_unloads_modules() -> None:
    """If the user runs 'woys chain disable' without ever having
    'enable'd, we should still unload any chain currently in memory
    rather than no-oping on the absence of a systemd file."""
    modules = (
        f"50\tmodule-null-sink\tsink_name={chain.SINK_FINAL}\n"
        f"51\tmodule-ladspa-sink\tsink_name={chain.SINK_BRIDGE}\n"
        f"52\tmodule-loopback\tsource={chain.SOURCE_RAW} sink={chain.SINK_BRIDGE}\n"
    )
    router = _PactlRouter(modules=modules)

    fake_unit_path = MagicMock()
    fake_unit_path.is_file.return_value = False

    with (
        patch.object(chain, "_systemd_unit_path", return_value=fake_unit_path),
        patch.object(chain.subprocess, "run", side_effect=router),
    ):
        rc = chain.disable()
    assert rc == 0
    fake_unit_path.unlink.assert_not_called()
    unload_ids = [c[2] for c in router.calls if c[:2] == ["pactl", "unload-module"]]
    assert {"50", "51", "52"} <= set(unload_ids)
