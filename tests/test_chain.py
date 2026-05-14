"""v0.13.2 - guard rails for the RNNoise chain module.

These tests do NOT exercise pactl/pipewire-pulse - they would need a
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
    so the mock signature is `(cmd, **kwargs)` - NOT `*args`.
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


def _patch_relabel_noop() -> Any:
    """v0.14.1 - chain.setup/teardown call audio.pipewire.relabel_source.
    Tests that don't care about the relabel itself stub it out so the
    pipewire module isn't required and pactl's source-state assumptions
    aren't tripped by an extra unload+reload pair."""
    return patch("audio.pipewire.relabel_source")


def test_setup_loads_audio_sink_class_and_mono_chain() -> None:
    """v0.13.2 routing fix + v0.13.3 friendly-naming topology. Regression guard
    for: media.class=Audio/Sink, channels=1 throughout, and a user-facing
    remap-source named `woys-by-alirexha` with matching device.description."""
    router = _PactlRouter(
        sources="1\twoys-mic\tdummy-driver\t1\tIDLE\n",
        modules="",  # no stale chain
    )

    with (
        patch.object(chain.Path, "is_file", lambda self: True),
        patch.object(chain.subprocess, "run", side_effect=router),
        _patch_relabel_noop(),
    ):
        rc = chain.setup()

    assert rc == 0
    loads = [c for c in router.calls if len(c) >= 2 and c[1] == "load-module"]
    assert len(loads) == 4, f"expected 4 load-module calls (v0.13.3), got {loads}"

    null_sink, ladspa_sink, loopback, user_remap = loads

    # 1. Null-sink: must be Audio/Sink (NOT Audio/Source/Virtual - that
    #    was the v0.13.0 bug). Description is the _internal- marker so
    #    users see "this isn't the source you want" in the dropdown.
    #    v0.14.1: also carries node.passive=true (PipeWire-native hint
    #    that this is plumbing) and session.suspend-timeout-seconds=0.
    null_sink_props = next((a for a in null_sink if a.startswith("sink_properties=")), "")
    assert "module-null-sink" in null_sink
    assert "media.class=Audio/Sink" in null_sink
    assert f"sink_name={chain.SINK_FINAL}" in null_sink
    assert "channels=1" in null_sink
    assert f"device.description={chain.DESC_FINAL_SINK}" in null_sink_props
    assert "node.passive=true" in null_sink_props, (
        f"v0.14.1: null-sink must have node.passive=true; got {null_sink_props!r}"
    )
    assert "session.suspend-timeout-seconds=0" in null_sink_props, (
        f"v0.14.1: null-sink must have session.suspend-timeout-seconds=0; got {null_sink_props!r}"
    )
    assert chain.DESC_FINAL_SINK.startswith("_internal-"), (
        "internal sink description must be marked clearly"
    )

    # 2. LADSPA-sink: mono, sink_master points at woys-mic-clean,
    #    plugin label is the mono variant, also tagged as internal.
    #    v0.14.1: also carries node.passive=true.
    ladspa_props = next((a for a in ladspa_sink if a.startswith("sink_properties=")), "")
    assert "module-ladspa-sink" in ladspa_sink
    assert f"sink_master={chain.SINK_FINAL}" in ladspa_sink
    assert f"sink_name={chain.SINK_BRIDGE}" in ladspa_sink
    assert "label=noise_suppressor_mono" in ladspa_sink
    assert "channels=1" in ladspa_sink
    assert f"device.description={chain.DESC_BRIDGE}" in ladspa_props
    assert "node.passive=true" in ladspa_props, (
        f"v0.14.1: ladspa-sink must have node.passive=true; got {ladspa_props!r}"
    )
    assert chain.DESC_BRIDGE.startswith("_internal-")

    # 3. Loopback: feeds woys-mic into the bridge, mono.
    assert "module-loopback" in loopback
    assert f"source={chain.SOURCE_RAW}" in loopback
    assert f"sink={chain.SINK_BRIDGE}" in loopback
    assert "channels=1" in loopback

    # 4. v0.13.3 user-facing remap-source. THIS is what apps pick.
    #    Both source_name and device.description are user-friendly,
    #    so Discord/Telegram/CS2/pavucontrol all show the same string.
    assert "module-remap-source" in user_remap
    assert f"master={chain.SINK_FINAL}.monitor" in user_remap
    assert f"source_name={chain.SOURCE_USER_FACING}" in user_remap
    assert f"source_properties=device.description={chain.DESC_USER_FACING}" in user_remap
    # A friendly name MUST NOT start with "_internal-" - otherwise users
    # looking for the daily-driver source won't recognize it.
    assert not chain.DESC_USER_FACING.startswith("_internal-")
    assert not chain.SOURCE_USER_FACING.startswith("_internal-")


def test_descriptions_have_no_spaces() -> None:
    """v0.13.3 lesson: pactl on pipewire-pulse splits sink/source-property
    values on whitespace before the proplist parser sees them. A description
    containing a space is silently truncated at the first space - apps see
    only the prefix. This test makes sure no future change reintroduces a
    space into any of the descriptions exposed to users."""
    for desc in (
        chain.DESC_USER_FACING,
        chain.DESC_BRIDGE,
        chain.DESC_FINAL_SINK,
    ):
        assert " " not in desc, (
            f"description {desc!r} contains a space - pactl will truncate it; use hyphens instead"
        )
        assert "\t" not in desc and "\n" not in desc, (
            f"description {desc!r} contains whitespace - same caveat"
        )


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
    """Idempotency: if an old chain is already loaded (including a v0.13.2
    chain or even a v0.13.0-broken one), setup must clear ALL four module
    types instead of stacking duplicates."""
    stale = (
        f"42\tmodule-null-sink\tmedia.class=Audio/Source/Virtual sink_name={chain.SINK_FINAL}\n"
        f"43\tmodule-ladspa-sink\tsink_name={chain.SINK_BRIDGE} sink_master={chain.SINK_FINAL}\n"
        f"44\tmodule-loopback\tsource={chain.SOURCE_RAW} sink={chain.SINK_BRIDGE}\n"
        f"45\tmodule-remap-source\tmaster={chain.SINK_FINAL}.monitor "
        f"source_name={chain.SOURCE_USER_FACING}\n"
    )
    router = _PactlRouter(
        sources="1\twoys-mic\tdrv\t1\tIDLE\n",
        modules=stale,
        # After the 4 load-module calls, modules listing reverts to
        # empty so any leakage isn't double-counted as 'stale again'.
        modules_after_load={0: ""},
    )

    with (
        patch.object(chain.Path, "is_file", lambda self: True),
        patch.object(chain.subprocess, "run", side_effect=router),
        _patch_relabel_noop(),
    ):
        chain.setup()

    unload_ids = [c[2] for c in router.calls if c[:2] == ["pactl", "unload-module"]]
    assert {"42", "43", "44", "45"} <= set(unload_ids), (
        f"expected stale 42/43/44/45 unloaded, got {unload_ids}"
    )


def test_setup_rolls_back_when_ladspa_load_fails() -> None:
    """If module-ladspa-sink fails (e.g. label mismatch), the null-sink
    we already loaded must be cleaned up - otherwise the user is left
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
        _patch_relabel_noop(),
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
        f"53\tmodule-remap-source\tmaster={chain.SINK_FINAL}.monitor "
        f"source_name={chain.SOURCE_USER_FACING}\n"
    )
    router = _PactlRouter(modules=modules)

    fake_unit_path = MagicMock()
    fake_unit_path.is_file.return_value = False

    with (
        patch.object(chain, "_systemd_unit_path", return_value=fake_unit_path),
        patch.object(chain.subprocess, "run", side_effect=router),
        _patch_relabel_noop(),
    ):
        rc = chain.disable()
    assert rc == 0
    fake_unit_path.unlink.assert_not_called()
    unload_ids = [c[2] for c in router.calls if c[:2] == ["pactl", "unload-module"]]
    assert {"50", "51", "52", "53"} <= set(unload_ids)


# v0.14.1 - single-default visibility ----------------------------------------


def test_setup_relabels_woys_mic_to_internal_raw_bypass() -> None:
    """v0.14.1 promise: when chain is active, woys-mic shows up to apps
    as `_internal-raw-bypass` (description), so users see only
    `woys-by-alirexha` as a non-internal woys option in their picker.
    The source NAME `woys-mic` is preserved for back-compat."""
    router = _PactlRouter(
        sources="1\twoys-mic\tdummy-driver\t1\tIDLE\n",
        modules="",
    )
    with (
        patch.object(chain.Path, "is_file", lambda self: True),
        patch.object(chain.subprocess, "run", side_effect=router),
        patch("audio.pipewire.relabel_source") as mock_relabel,
    ):
        rc = chain.setup()
    assert rc == 0
    mock_relabel.assert_called_once()
    args, kwargs = mock_relabel.call_args
    # Description is the chain-active constant; passive=True is set so
    # PipeWire-native consumers see the node as plumbing.
    from audio.pipewire import SOURCE_DESC_CHAIN_ACTIVE

    assert args == (SOURCE_DESC_CHAIN_ACTIVE,) or kwargs.get("description") == (
        SOURCE_DESC_CHAIN_ACTIVE
    )
    assert kwargs.get("passive") is True


def test_setup_fails_loudly_when_relabel_fails() -> None:
    """review F-merged-006: a relabel failure is a daemon-path step
    that did not complete -- `chain.setup()` must report it via exit 2,
    not swallow it and `return 0`.

    Pre-fix this test asserted `rc == 0` ("setup succeeds even when relabel
    fails"), which locked in the v0.14.2 bug class: `woys-chain.service`
    went `active` while a daemon-path mutation had silently not taken (and,
    before `relabel_source` became atomic, had destroyed woys-mic). The
    RNNoise chain modules are still loaded and woys-mic is intact
    (`relabel_source` rolls back atomically) -- but the exit code must make
    the unit show `failed` so the partial state is visible."""
    router = _PactlRouter(
        sources="1\twoys-mic\tdrv\t1\tIDLE\n",
        modules="",
    )
    with (
        patch.object(chain.Path, "is_file", lambda self: True),
        patch.object(chain.subprocess, "run", side_effect=router),
        patch("audio.pipewire.relabel_source", side_effect=RuntimeError("boom")),
    ):
        rc = chain.setup()
    assert rc == 2


def test_teardown_restores_woys_mic_default_description() -> None:
    """v0.14.1 promise: after teardown, woys-mic is back to
    `woys-no-cleanup` so users without the chain see a sensible
    daily-driver label."""
    router = _PactlRouter(modules="")
    with (
        patch.object(chain.subprocess, "run", side_effect=router),
        patch("audio.pipewire.relabel_source") as mock_relabel,
    ):
        rc = chain.teardown()
    assert rc == 0
    mock_relabel.assert_called_once()
    args, kwargs = mock_relabel.call_args
    from audio.pipewire import SOURCE_DESC

    assert args == (SOURCE_DESC,) or kwargs.get("description") == SOURCE_DESC
    assert kwargs.get("passive") is False


def test_user_facing_sources_filters_internal_descriptions() -> None:
    """v0.14.1 - parsing `pactl list sources` output. Sources with
    descriptions starting with `_internal-` are plumbing; everything
    else is user-facing. Used by `chain.status()` to render the
    'apps will display' section."""
    sample = """\
Source #95
\tState: RUNNING
\tName: woys-mic
\tDescription: _internal-raw-bypass
\tDriver: PipeWire
Source #337
\tState: RUNNING
\tName: woys-mic-clean.monitor
\tDescription: Monitor of _internal-clean-sink
\tDriver: PipeWire
Source #354
\tState: RUNNING
\tName: woys-by-alirexha
\tDescription: woys-by-alirexha
\tDriver: PipeWire
Source #58
\tState: RUNNING
\tName: alsa_input.usb-HyperX_QuadCast.analog-stereo
\tDescription: HyperX QuadCast 2 S
\tDriver: PipeWire
"""
    rows = chain._user_facing_sources(sample)
    names = [name for name, _ in rows]
    descs = dict(rows)

    # woys-mic is hidden (description starts with `_internal-`).
    assert "woys-mic" not in names
    # woys-by-alirexha is the user-facing chain endpoint.
    assert "woys-by-alirexha" in names
    assert descs["woys-by-alirexha"] == "woys-by-alirexha"
    # Real hardware mic comes through as user-facing too.
    assert "alsa_input.usb-HyperX_QuadCast.analog-stereo" in names

    # Crucial: only ONE woys* source should be user-facing when chain
    # is active. This is the test that fails if a regression makes
    # `_internal-` markers get dropped somewhere.
    woys_user_facing = [n for n in names if "woys" in n]
    assert woys_user_facing == ["woys-by-alirexha"], (
        f"expected exactly one user-facing woys source, got {woys_user_facing}"
    )


def test_user_facing_sources_filters_monitor_of_internal() -> None:
    """Pipewire-pulse auto-derives `.monitor` source descriptions as
    `Monitor of <sink description>`. When the sink is `_internal-...`,
    the monitor's description becomes `Monitor of _internal-...` and
    apps render it that way. The `_user_facing_sources` filter must
    catch this case too (matches by `contains _internal-`, not
    `startswith _internal-`)."""
    sample = """\
Source #1
\tName: foo.monitor
\tDescription: Monitor of _internal-clean-sink
Source #2
\tName: real-mic
\tDescription: Real Microphone
"""
    rows = chain._user_facing_sources(sample)
    names = [name for name, _ in rows]
    assert "foo.monitor" not in names, (
        "v0.14.1: Monitor of _internal-... must be filtered out as plumbing"
    )
    assert "real-mic" in names


def test_user_facing_sources_empty_input() -> None:
    assert chain._user_facing_sources("") == []


def test_is_user_facing_description() -> None:
    """The single-source-of-truth predicate. Used by status() and
    indirectly by `_user_facing_sources`. Pin the rules:
    - Descriptions starting with `_internal-` are plumbing.
    - Descriptions containing `_internal-` anywhere are plumbing
      (catches `Monitor of _internal-...`).
    - Real device names and `woys-by-alirexha` are user-facing.
    """
    assert chain._is_user_facing_description("woys-by-alirexha") is True
    assert chain._is_user_facing_description("HyperX QuadCast 2 S") is True
    assert chain._is_user_facing_description("woys-no-cleanup") is True
    assert chain._is_user_facing_description("_internal-raw-bypass") is False
    assert chain._is_user_facing_description("_internal-clean-sink") is False
    assert chain._is_user_facing_description("Monitor of _internal-clean-sink") is False
