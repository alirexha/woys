"""v0.14.2 - guard rails for the RNNoise filter-chain module.

These tests do NOT exercise pactl/pipewire-pulse/systemctl - they
would need a real PipeWire session and the LADSPA plugin installed,
neither of which we want CI to depend on. They DO lock in:

  * The filter-chain conf content (target.object=woys-mic, mono,
    media.class=Audio/Source on playback.props, _internal- markers
    on the capture/filter sides so libpulse can't render them as
    user-facing).
  * The conf file location (~/.config/pipewire/pipewire.conf.d/).
  * setup() writes the conf, restarts the pipewire stack, relabels
    woys-mic, and removes any legacy v0.14.1 systemd unit.
  * teardown() removes the conf, restarts pipewire, restores the
    woys-mic description, and unloads any leftover v0.14.1 modules.
  * Fallback path: if filter-chain is unavailable, setup() falls
    through to the v0.14.1 module-based chain.
  * The v0.14.1 visibility filter (`_user_facing_sources`,
    `_is_user_facing_description`) - reused unchanged in v0.14.2.
"""

from __future__ import annotations

import subprocess
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from woys import chain


def _ok(stdout: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


def _err(stderr: str = "boom", returncode: int = 1) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout="", stderr=stderr)


class _SubprocessRouter:
    """Mock for subprocess.run that routes by argv. v0.14.2 covers
    pactl + systemctl + pw-link calls."""

    def __init__(
        self,
        sources: str = "",
        modules: str = "",
        load_results: list[subprocess.CompletedProcess[str]] | None = None,
        modules_after_load: dict[int, str] | None = None,
        pwlink_output: str = "",
        systemctl_results: dict[str, subprocess.CompletedProcess[str]] | None = None,
        woys_mic_appears_after_systemctl_start: bool = True,
    ) -> None:
        self.sources = sources
        self.modules = modules
        self.load_results = load_results or []
        self.modules_after_load = modules_after_load or {}
        self.pwlink_output = pwlink_output
        # Map a leading systemctl subcommand to a CompletedProcess
        # (e.g. {"restart": _err("hosed")}). Defaults to OK.
        self.systemctl_results = systemctl_results or {}
        self.woys_mic_appears_after_systemctl_start = woys_mic_appears_after_systemctl_start
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
            if cmd[1] == "list" and cmd[2] == "sources":
                # Long-form 'pactl list sources' - empty by default;
                # specific tests override.
                return _ok(self.sources)
            if cmd[1] == "load-module":
                idx = self._load_count
                self._load_count += 1
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
            sub = cmd[2] if len(cmd) > 2 and cmd[1] == "--user" else cmd[1]
            if sub == "start" and "woys-mic.service" in cmd:
                # If the test wants woys-mic to appear after start,
                # also bump the sources fixture so _source_present sees it.
                if self.woys_mic_appears_after_systemctl_start and "woys-mic" not in self.sources:
                    self.sources = self.sources + "1\twoys-mic\tdrv\t1\tIDLE\n"
                return _ok()
            if sub in self.systemctl_results:
                return self.systemctl_results[sub]
            return _ok()
        return _ok()


def _patch_relabel_noop() -> Any:
    return patch("audio.pipewire.relabel_source")


# --- v0.14.2 filter-chain conf rendering ------------------------------------


def test_render_conf_pins_target_to_woys_mic_and_mono() -> None:
    """The conf MUST bind capture to `woys-mic` by name (target.object)
    and use mono throughout. The noise-suppressor_mono LADSPA plugin
    is single-channel; a stereo capture would either run two filter
    instances or refuse to bind."""
    conf = chain._render_conf()
    assert f'target.object       = "{chain.SOURCE_RAW}"' in conf
    assert "audio.position   = [ MONO ]" in conf
    assert "label  = noise_suppressor_mono" in conf
    assert "libpipewire-module-filter-chain" in conf


def test_render_conf_internal_markers_on_non_user_facing_nodes() -> None:
    """Capture node + filter wrapper must be marked `_internal-` in
    their descriptions/names so any future PipeWire-native consumer
    that filters by description does not render them as user picks.
    libpulse ignores these markers - capture.props's media.class
    (Stream/Input/Audio by default) is already pulse-hidden - but
    setting them here keeps the contract consistent with the v0.14.1
    visibility rule and protects against PipeWire someday exposing
    Stream/Input/Audio nodes to pulse."""
    conf = chain._render_conf()
    assert 'node.description = "_internal-rnnoise-filter"' in conf
    assert 'node.name           = "_internal-woys-chain-capture"' in conf
    # The user-facing playback node must NOT have an _internal- prefix
    # in its description, otherwise the visibility filter hides it
    # from apps and nobody can pick the daily driver.
    assert f'node.description = "{chain.DESC_USER_FACING}"' in conf
    assert not chain.DESC_USER_FACING.startswith("_internal-")


def test_render_conf_playback_is_audio_source() -> None:
    """The playback side has `media.class = Audio/Source` so libpulse
    enumerates it. This is the ONE woys-by-alirexha source apps see."""
    conf = chain._render_conf()
    assert "media.class      = Audio/Source" in conf
    assert f'node.name        = "{chain.SOURCE_USER_FACING}"' in conf


def test_render_conf_passive_capture() -> None:
    """`node.passive = true` on capture marks the input side as
    plumbing for PipeWire-native consumers that filter by it."""
    conf = chain._render_conf()
    assert "node.passive        = true" in conf


def test_render_conf_uses_provided_plugin_path() -> None:
    """The plugin path is parameterised so install scripts can override
    it (some distros put noise-suppression-for-voice in /usr/local/lib
    or /usr/lib64 instead of /usr/lib/ladspa/)."""
    conf = chain._render_conf(plugin_path="/opt/custom/librnnoise_ladspa.so")
    assert 'plugin = "/opt/custom/librnnoise_ladspa.so"' in conf


# --- filter_chain_supported() probe -----------------------------------------


def test_filter_chain_supported_when_so_exists(tmp_path: Any) -> None:
    fake_so = tmp_path / "libpipewire-module-filter-chain.so"
    fake_so.write_bytes(b"fake")
    with patch.object(chain, "FILTER_CHAIN_SO", str(fake_so)):
        assert chain.filter_chain_supported() is True


def test_filter_chain_supported_when_so_missing(tmp_path: Any) -> None:
    with patch.object(chain, "FILTER_CHAIN_SO", str(tmp_path / "nonexistent.so")):
        assert chain.filter_chain_supported() is False


# --- setup() (filter-chain primary path) ------------------------------------


def test_setup_writes_conf_and_restarts_pipewire(tmp_path: Any) -> None:
    """Happy path: plugin exists, filter-chain supported, woys-mic
    present after restart. setup() must:
      1. write the conf
      2. restart the pipewire stack
      3. wait for woys-by-alirexha to appear
      4. relabel woys-mic
      5. clean up legacy systemd unit (no-op if not present)
    """
    conf_path = tmp_path / "pipewire" / "pipewire.conf.d" / "99-woys-chain.conf"
    legacy_unit = tmp_path / "systemd" / "user" / "woys-chain.service"
    router = _SubprocessRouter(
        # Sources fixture grows as the test progresses: woys-mic comes
        # back after the simulated systemctl start, then woys-by-alirexha
        # gets added when the filter-chain wakes up.
        sources="1\twoys-mic\tdrv\t1\tIDLE\n",
        modules="",
    )

    def fake_source_present(name: str) -> bool:
        # Simulate woys-by-alirexha appearing 1 poll iteration later.
        if name == chain.SOURCE_USER_FACING:
            fake_source_present.calls += 1  # type: ignore[attr-defined]
            return fake_source_present.calls > 1  # type: ignore[attr-defined]
        return name in router.sources

    fake_source_present.calls = 0  # type: ignore[attr-defined]

    with (
        patch.object(chain.Path, "is_file", lambda self: str(self) == chain.PLUGIN_PATH),
        patch.object(chain, "filter_chain_supported", return_value=True),
        patch.object(chain, "_conf_file_path", return_value=conf_path),
        patch.object(chain, "_systemd_unit_path", return_value=legacy_unit),
        patch.object(chain, "_source_present", side_effect=fake_source_present),
        patch.object(chain, "time", MagicMock(monotonic=MagicMock(side_effect=range(100)))),
        patch.object(chain.subprocess, "run", side_effect=router),
        _patch_relabel_noop() as mock_relabel,
    ):
        rc = chain.setup()

    assert rc == 0
    assert conf_path.is_file()
    body = conf_path.read_text()
    assert "libpipewire-module-filter-chain" in body
    assert f'target.object       = "{chain.SOURCE_RAW}"' in body

    # systemctl restart was called for the pipewire stack.
    restart_calls = [c for c in router.calls if c[:3] == ["systemctl", "--user", "restart"]]
    assert len(restart_calls) >= 1, f"expected pipewire restart, got {router.calls}"
    assert all(unit in restart_calls[0] for unit in chain.PIPEWIRE_UNITS)

    # woys-mic relabel was called with the chain-active description.
    mock_relabel.assert_called()
    args, kwargs = mock_relabel.call_args
    from audio.pipewire import SOURCE_DESC_CHAIN_ACTIVE

    assert SOURCE_DESC_CHAIN_ACTIVE in args or kwargs.get("description") == SOURCE_DESC_CHAIN_ACTIVE
    assert kwargs.get("passive") is True


def test_setup_falls_back_when_filter_chain_unavailable(tmp_path: Any) -> None:
    """If the filter-chain .so is missing, setup() must fall through
    to the v0.14.1 module-based chain instead of writing a useless
    conf file that nothing will consume."""
    conf_path = tmp_path / "pipewire" / "pipewire.conf.d" / "99-woys-chain.conf"
    router = _SubprocessRouter(sources="1\twoys-mic\tdrv\t1\tIDLE\n", modules="")

    with (
        patch.object(chain.Path, "is_file", lambda self: str(self) == chain.PLUGIN_PATH),
        patch.object(chain, "filter_chain_supported", return_value=False),
        patch.object(chain, "_conf_file_path", return_value=conf_path),
        patch.object(chain.subprocess, "run", side_effect=router),
        _patch_relabel_noop(),
    ):
        rc = chain.setup()

    # Fallback ran successfully (4 module-load calls happened).
    assert rc == 0
    loads = [c for c in router.calls if len(c) >= 2 and c[1] == "load-module"]
    assert len(loads) == 4, f"v0.14.1 fallback must load 4 modules, got {len(loads)}: {loads}"

    # The filter-chain conf was NOT written (we couldn't load it anyway).
    assert not conf_path.is_file()


def test_setup_refuses_when_plugin_missing() -> None:
    """No LADSPA plugin -> hard fail before touching anything else."""
    with (
        patch.object(chain.Path, "is_file", lambda self: False),
    ):
        rc = chain.setup()
    assert rc == 2


def test_setup_rolls_back_conf_on_pipewire_restart_failure(tmp_path: Any) -> None:
    """If `systemctl --user restart` fails, the conf we just wrote
    would auto-load on the NEXT pipewire restart (e.g. login) and
    deliver a half-broken state. setup() must remove the conf on
    failure so the next manual retry starts from clean."""
    conf_path = tmp_path / "pipewire" / "pipewire.conf.d" / "99-woys-chain.conf"
    legacy_unit = tmp_path / "systemd" / "user" / "woys-chain.service"
    router = _SubprocessRouter(
        sources="",
        modules="",
        systemctl_results={"restart": _err("dbus down", returncode=1)},
    )

    with (
        patch.object(chain.Path, "is_file", lambda self: str(self) == chain.PLUGIN_PATH),
        patch.object(chain, "filter_chain_supported", return_value=True),
        patch.object(chain, "_conf_file_path", return_value=conf_path),
        patch.object(chain, "_systemd_unit_path", return_value=legacy_unit),
        patch.object(chain.subprocess, "run", side_effect=router),
        _patch_relabel_noop(),
    ):
        rc = chain.setup()

    assert rc == 2
    assert not conf_path.is_file(), "conf must be rolled back on restart failure"


# --- teardown() -------------------------------------------------------------


def test_teardown_removes_conf_and_restarts(tmp_path: Any) -> None:
    """teardown() removes the conf file, restarts pipewire, and
    restores woys-mic's daily-driver description."""
    conf_path = tmp_path / "pipewire" / "pipewire.conf.d" / "99-woys-chain.conf"
    conf_path.parent.mkdir(parents=True)
    conf_path.write_text("# stale conf to remove\n")
    legacy_unit = tmp_path / "systemd" / "user" / "woys-chain.service"
    router = _SubprocessRouter(modules="")

    with (
        patch.object(chain, "_conf_file_path", return_value=conf_path),
        patch.object(chain, "_systemd_unit_path", return_value=legacy_unit),
        patch.object(chain.subprocess, "run", side_effect=router),
        _patch_relabel_noop() as mock_relabel,
    ):
        rc = chain.teardown()

    assert rc == 0
    assert not conf_path.is_file()
    restart_calls = [c for c in router.calls if c[:3] == ["systemctl", "--user", "restart"]]
    assert len(restart_calls) >= 1

    mock_relabel.assert_called()
    from audio.pipewire import SOURCE_DESC

    args, kwargs = mock_relabel.call_args
    assert SOURCE_DESC in args or kwargs.get("description") == SOURCE_DESC
    assert kwargs.get("passive") is False


def test_teardown_when_no_conf_still_unloads_legacy_modules(tmp_path: Any) -> None:
    """If a user has v0.14.1 legacy modules loaded but no v0.14.2
    conf (e.g. mid-upgrade), teardown() must still cleanly unload
    the legacy modules and restore the description."""
    conf_path = tmp_path / "no-conf.conf"  # does not exist
    legacy_unit = tmp_path / "no-unit.service"  # does not exist
    legacy_modules = (
        f"50\tmodule-null-sink\tsink_name={chain.SINK_FINAL}\n"
        f"51\tmodule-ladspa-sink\tsink_name={chain.SINK_BRIDGE}\n"
        f"52\tmodule-loopback\tsource={chain.SOURCE_RAW} sink={chain.SINK_BRIDGE}\n"
        f"53\tmodule-remap-source\tmaster={chain.SINK_FINAL}.monitor "
        f"source_name={chain.SOURCE_USER_FACING}\n"
    )
    router = _SubprocessRouter(modules=legacy_modules)

    with (
        patch.object(chain, "_conf_file_path", return_value=conf_path),
        patch.object(chain, "_systemd_unit_path", return_value=legacy_unit),
        patch.object(chain.subprocess, "run", side_effect=router),
        _patch_relabel_noop(),
    ):
        rc = chain.teardown()

    assert rc == 0
    unload_ids = [c[2] for c in router.calls if c[:2] == ["pactl", "unload-module"]]
    assert {"50", "51", "52", "53"} <= set(unload_ids)


# --- legacy v0.14.1 fallback path -------------------------------------------


def test_legacy_setup_loads_audio_sink_class_and_mono_chain() -> None:
    """The v0.14.1 fallback chain must still produce media.class=
    Audio/Sink + node.passive=true on intermediates and a Audio/Source
    user-facing remap. Direct test of `_legacy_setup` so the fallback
    can't silently bitrot."""
    router = _SubprocessRouter(sources="1\twoys-mic\tdrv\t1\tIDLE\n", modules="")
    with (
        patch.object(chain.subprocess, "run", side_effect=router),
        _patch_relabel_noop(),
    ):
        rc = chain._legacy_setup()
    assert rc == 0
    loads = [c for c in router.calls if len(c) >= 2 and c[1] == "load-module"]
    assert len(loads) == 4
    null_sink, ladspa_sink, loopback, user_remap = loads

    null_sink_props = next((a for a in null_sink if a.startswith("sink_properties=")), "")
    assert "media.class=Audio/Sink" in null_sink
    assert "node.passive=true" in null_sink_props
    assert f"sink_name={chain.SINK_FINAL}" in null_sink

    ladspa_props = next((a for a in ladspa_sink if a.startswith("sink_properties=")), "")
    assert "node.passive=true" in ladspa_props
    assert "label=noise_suppressor_mono" in ladspa_sink

    assert f"source={chain.SOURCE_RAW}" in loopback
    assert f"source_name={chain.SOURCE_USER_FACING}" in user_remap


# --- v0.14.1 visibility filter (unchanged in v0.14.2) -----------------------


def test_is_user_facing_description() -> None:
    assert chain._is_user_facing_description("woys-by-alirexha") is True
    assert chain._is_user_facing_description("HyperX QuadCast 2 S") is True
    assert chain._is_user_facing_description("woys-no-cleanup") is True
    assert chain._is_user_facing_description("_internal-raw-bypass") is False
    assert chain._is_user_facing_description("_internal-clean-sink") is False
    assert chain._is_user_facing_description("Monitor of _internal-clean-sink") is False
    assert chain._is_user_facing_description("Monitor of _internal-rnnoise-filter") is False


def test_user_facing_sources_filters_internal_descriptions() -> None:
    sample = """\
Source #95
\tName: woys-mic
\tDescription: _internal-raw-bypass
Source #354
\tName: woys-by-alirexha
\tDescription: woys-by-alirexha
Source #58
\tName: alsa_input.usb-HyperX_QuadCast.analog-stereo
\tDescription: HyperX QuadCast 2 S
"""
    rows = chain._user_facing_sources(sample)
    names = [name for name, _ in rows]
    assert "woys-mic" not in names
    assert "woys-by-alirexha" in names
    assert "alsa_input.usb-HyperX_QuadCast.analog-stereo" in names
    woys_user_facing = [n for n in names if "woys" in n]
    assert woys_user_facing == ["woys-by-alirexha"]


def test_user_facing_sources_filters_monitor_of_internal() -> None:
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
    assert "foo.monitor" not in names
    assert "real-mic" in names


def test_user_facing_sources_empty_input() -> None:
    assert chain._user_facing_sources("") == []


# --- alsa-leak diagnostic + paths -------------------------------------------


def test_alsa_leak_links_returns_empty_when_pwlink_missing() -> None:
    with patch("shutil.which", return_value=None):
        assert chain._alsa_leak_links() == []


def test_alsa_leak_links_flags_filter_chain_to_alsa() -> None:
    pwlink_output = (
        "output.filter-chain-1803-15\n"
        "  |-> alsa_output.pci-0000_00_1f.3.analog-stereo:playback_FL\n"
        "some-other-node\n"
        "  |-> some-non-alsa-input\n"
    )
    router = _SubprocessRouter(pwlink_output=pwlink_output)
    with (
        patch("shutil.which", return_value="/usr/bin/pw-link"),
        patch.object(chain.subprocess, "run", side_effect=router),
    ):
        leaks = chain._alsa_leak_links()
    assert len(leaks) == 1
    assert "alsa_output" in leaks[0]


def test_systemd_unit_path_respects_xdg_config_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    p = chain._systemd_unit_path()
    assert p == tmp_path / "systemd" / "user" / "woys-chain.service"


def test_conf_file_path_respects_xdg_config_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    p = chain._conf_file_path()
    assert p == tmp_path / "pipewire" / "pipewire.conf.d" / "99-woys-chain.conf"


# --- description constants are space-free -----------------------------------


def test_descriptions_have_no_spaces() -> None:
    """pactl on pipewire-pulse splits property values on whitespace
    before the proplist parser sees them. A description containing a
    space is silently truncated. Hyphens substitute fine."""
    for desc in (chain.DESC_USER_FACING, chain.DESC_BRIDGE, chain.DESC_FINAL_SINK):
        assert " " not in desc, f"{desc!r} contains space; pactl will truncate"
        assert "\t" not in desc and "\n" not in desc
