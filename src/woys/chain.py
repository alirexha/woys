"""v0.14.2 - RNNoise filter-chain via PipeWire-native module-filter-chain.

Replaces the v0.13.x..v0.14.1 four-module pactl chain with a single
PipeWire-native filter-chain. Apps see only ONE woys input source
(`woys-by-alirexha`) plus the underlying `woys-mic` (relabelled to
`_internal-raw-bypass`). The intermediate sinks/monitors that v0.14.1
exposed (`woys-mic-clean.monitor`, `woys-mic-rnnoise-bridge.monitor`)
are eliminated entirely - filter-chain's internal Stream/Input/Audio
and Stream/Output/Audio nodes are not exposed to libpulse.

Verified empirically on PipeWire 1.6.4: `media.class` determines pulse
visibility - `Audio/Source` and `Audio/Sink` (with auto-monitor) are
exposed; `Stream/Input/Audio` and `Stream/Output/Audio` are not. The
v0.14.1 chain had three Audio/Sink modules (each adding a sink + a
monitor source) plus two remap-sources, so apps saw eight woys-related
rows; v0.14.2 keeps only the foundational woys-mic (Audio/Source) and
the new woys-by-alirexha (Audio/Source from filter-chain playback.props).

How activation works in v0.14.2:

  Conf file: ~/.config/pipewire/pipewire.conf.d/99-woys-chain.conf
  Persistence: PipeWire reads conf.d at every daemon startup, so the
    chain auto-loads on login - no separate systemd unit needed.
  Toggle cost: writing or removing the conf requires restarting the
    pipewire stack (`systemctl --user restart pipewire pipewire-pulse
    wireplumber`), which causes a 1-2 second audio glitch across the
    desktop. Acceptable because users keep the chain on permanently
    and rarely toggle.

Why no separate systemd woys-chain.service in v0.14.2:
  v0.13.2..v0.14.1's systemd unit existed because pactl modules don't
  persist across pipewire restarts; the unit re-loaded them on login.
  In v0.14.2, the conf file IS the persistence - PipeWire itself
  re-reads it at every startup, so the unit is redundant. setup()
  detects and removes any old v0.14.1 unit when it runs.

Fallback path:
  If `/usr/lib/pipewire-0.3/libpipewire-module-filter-chain.so` is
  missing (extremely old PipeWire, not seen on any modern distro),
  setup() prints a warning and falls back to the v0.14.1 module-based
  pactl chain. The v0.14.1 code lives intact in the `_legacy_*`
  functions below so the fallback is a real working code path, not a
  stub.

Hard limit (still): pulseaudio compat enumerates ALL Audio/Source and
Audio/Sink nodes regardless of `node.passive` / `node.virtual` /
`object.register=false`. v0.14.2 reduces the count of exposed nodes
to the minimum (two: woys-mic and woys-by-alirexha) but cannot make
woys-mic itself disappear without breaking the engine pipeline.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PLUGIN_PATH = "/usr/lib/ladspa/librnnoise_ladspa.so"
PLUGIN_LABEL = "noise_suppressor_mono"

# Filter-chain mode (v0.14.2 default) ----------------------------------------

# The PipeWire native module .so. We use its presence as the
# filter-chain availability probe - if the file exists, the daemon can
# load it. Older PipeWire releases (<0.3.x) lacked filter-chain;
# distros bundle PipeWire >= 0.3.50 across the board (released 2022),
# which is well past the `target.object` capture-binding feature we
# rely on. We check existence rather than parse a version because the
# .so absence is the only thing that matters and it's a hard bar.
FILTER_CHAIN_SO = "/usr/lib/pipewire-0.3/libpipewire-module-filter-chain.so"

# Conf file PipeWire reads on every daemon start. The `99-` prefix
# orders this file LAST in the conf.d directory's lexical sort so it
# overrides any earlier filter-chain configs if present.
CONF_FILENAME = "99-woys-chain.conf"

SOURCE_RAW = "woys-mic"  # the foundational engine output remap-source
SOURCE_USER_FACING = "woys-by-alirexha"  # the filter-chain Audio/Source
DESC_USER_FACING = "woys-by-alirexha"

# Pipewire-stack systemctl unit names. We restart all three because
# wireplumber holds device routing state that goes stale if pipewire-
# pulse restarts under it.
PIPEWIRE_UNITS = ("pipewire", "pipewire-pulse", "wireplumber")

# Legacy v0.14.1 module-based chain identifiers (used by _legacy_*) ----------

SINK_FINAL = "woys-mic-clean"
SINK_BRIDGE = "woys-mic-rnnoise-bridge"
DESC_BRIDGE = "_internal-rnnoise-stage"
DESC_FINAL_SINK = "_internal-clean-sink"
LOOPBACK_MARKER = f"source={SOURCE_RAW} sink={SINK_BRIDGE}"
USER_REMAP_MARKER = f"master={SINK_FINAL}.monitor source_name={SOURCE_USER_FACING}"

# v0.13.2..v0.14.1 systemd unit. v0.14.2 doesn't install it, but
# detects + cleans up an existing one during setup() so users
# upgrading from v0.14.1 don't end up with a stale unit firing the
# legacy `woys chain setup` flow at login alongside the new conf.d
# auto-load.
SYSTEMD_UNIT_NAME = "woys-chain.service"


# --- helpers ----------------------------------------------------------------


def _systemd_unit_path() -> Path:
    base = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
    return base / "systemd" / "user" / SYSTEMD_UNIT_NAME


def _conf_file_path() -> Path:
    base = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
    return base / "pipewire" / "pipewire.conf.d" / CONF_FILENAME


def _pactl(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["pactl", *args], capture_output=True, text=True, timeout=5, check=False)


def _systemctl(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["systemctl", "--user", *args],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def _list_modules() -> list[tuple[str, str, str]]:
    """Return [(id, type, args)] for currently loaded pipewire-pulse modules."""
    out = _pactl("list", "short", "modules").stdout
    rows: list[tuple[str, str, str]] = []
    for line in out.splitlines():
        parts = line.split("\t", 2)
        if len(parts) >= 2:
            mod_id = parts[0].strip()
            mod_type = parts[1].strip()
            mod_args = parts[2].strip() if len(parts) >= 3 else ""
            rows.append((mod_id, mod_type, mod_args))
    return rows


def _source_present(name: str) -> bool:
    out = _pactl("list", "short", "sources").stdout
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[1].strip() == name:
            return True
    return False


# --- pulse-visible source filter (shared by v0.14.1 + v0.14.2 status) -------


def _is_user_facing_description(description: str) -> bool:
    """A source description is user-facing iff it does NOT contain the
    `_internal-` marker. We check `contains` rather than `startswith` so
    the auto-derived "Monitor of _internal-..." description (pipewire-
    pulse adds the `Monitor of ` prefix to a sink's auto-monitor and
    offers no API to override it) is also filtered out.
    """
    return "_internal-" not in description


def _user_facing_sources(pactl_list_sources_output: str) -> list[tuple[str, str]]:
    """Parse `pactl list sources` (long form) and return [(name,
    description), ...] for sources whose description is user-facing.

    Pure function - the test suite feeds it canned output.
    """
    results: list[tuple[str, str]] = []
    cur_name: str | None = None
    cur_desc: str | None = None
    for raw in pactl_list_sources_output.splitlines():
        if raw.startswith("Source #"):
            if cur_name and cur_desc and _is_user_facing_description(cur_desc):
                results.append((cur_name, cur_desc))
            cur_name = None
            cur_desc = None
            continue
        line = raw.strip()
        if line.startswith("Name: "):
            cur_name = line[len("Name: ") :].strip()
        elif line.startswith("Description: "):
            cur_desc = line[len("Description: ") :].strip()
    if cur_name and cur_desc and _is_user_facing_description(cur_desc):
        results.append((cur_name, cur_desc))
    return results


# --- v0.14.2 filter-chain conf generation -----------------------------------


def filter_chain_supported() -> bool:
    """Probe: is `module-filter-chain` loadable on this PipeWire?

    We treat the .so file's presence as the gate. The capability we
    actually depend on (`target.object` on `capture.props`) has been
    in PipeWire since 0.3.50 (2022); any release new enough to ship
    the module ships the capability.
    """
    return Path(FILTER_CHAIN_SO).is_file()


def _render_conf(plugin_path: str = PLUGIN_PATH) -> str:
    """Build the conf.d content. Pure function so the test suite can
    pin the exact string we emit.

    Two non-obvious choices:
      * `target.object = "woys-mic"` binds the capture side to woys-mic
        BY NAME. PipeWire holds the binding lazily - if woys-mic isn't
        present at conf-load time, the filter-chain idles until it
        appears. This means startup ordering between the conf load and
        woys-mic.service doesn't have to be perfect.
      * `node.passive = true` on capture.props marks it as plumbing for
        any PipeWire-native consumer that filters by it. (libpulse
        ignores it; capture.props is Stream/Input/Audio so it isn't
        pulse-visible regardless.)
    """
    return f"""# Generated by `woys chain setup` (v0.14.2). Do not edit by hand;
# `woys chain teardown` removes this file.
#
# Single PipeWire-native filter-chain that pulls from `woys-mic`
# (the engine's raw output remap-source from src/audio/pipewire.py),
# applies RNNoise via the noise-suppressor_mono LADSPA plugin, and
# exposes the result as the `woys-by-alirexha` Audio/Source.
#
# All graph-internal nodes (the Stream/Input/Audio capture node, the
# LADSPA filter, the Stream/Output/Audio playback path) are NOT
# exposed to libpulse - they have media.class values pulse-protocol
# does not enumerate as sources/sinks. Apps therefore see exactly two
# woys-related entries: `woys-mic` (relabelled `_internal-raw-bypass`
# while this conf is in place) and `woys-by-alirexha`.
context.modules = [
    {{ name = libpipewire-module-filter-chain
        flags = [ nofail ]
        args = {{
            node.description = "_internal-rnnoise-filter"
            media.name       = "_internal-rnnoise-filter"
            audio.position   = [ MONO ]
            capture.props = {{
                node.name           = "_internal-woys-chain-capture"
                node.passive        = true
                target.object       = "{SOURCE_RAW}"
                stream.dont-remix   = true
            }}
            playback.props = {{
                node.name        = "{SOURCE_USER_FACING}"
                node.description = "{DESC_USER_FACING}"
                media.class      = Audio/Source
                audio.position   = [ MONO ]
            }}
            filter.graph = {{
                nodes = [
                    {{
                        type   = ladspa
                        name   = rnnoise
                        plugin = "{plugin_path}"
                        label  = {PLUGIN_LABEL}
                    }}
                ]
            }}
        }}
    }}
]
"""


def _restart_pipewire_stack() -> tuple[bool, str]:
    """Restart pipewire + pipewire-pulse + wireplumber. Returns
    (ok, message). Disruptive: ~1s desktop-wide audio glitch."""
    out = _systemctl("restart", *PIPEWIRE_UNITS)
    if out.returncode != 0:
        return (False, f"systemctl restart failed: {out.stderr.strip() or out.stdout.strip()}")
    return (True, "")


def _wait_for_user_facing_source(timeout_s: float = 8.0) -> bool:
    """Poll `pactl list short sources` until `woys-by-alirexha` shows up
    or we time out. After a pipewire restart, woys-mic.service has to
    re-load WoysSink + woys-mic before the filter-chain's target.object
    binding resolves - that takes a beat."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _source_present(SOURCE_USER_FACING):
            return True
        time.sleep(0.25)
    return False


def _ensure_woys_mic_loaded() -> bool:
    """After a pipewire restart the `woys-mic.service` user unit may
    not have re-armed (Requires=pipewire-pulse.socket triggers a stop,
    not a restart, when the socket cycles). Force-start it so woys-mic
    + WoysSink come back; if the unit isn't installed, skip."""
    state = _systemctl("status", "woys-mic.service")
    # `status` returns 4 for a non-existent unit; 3 for inactive;
    # 0 for active. We just need the service file to exist.
    if "could not be found" in state.stderr.lower() or state.returncode == 4:
        return False
    _systemctl("start", "woys-mic.service")
    # Give pactl a moment to register the freshly-loaded modules.
    deadline = time.monotonic() + 4.0
    while time.monotonic() < deadline:
        if _source_present(SOURCE_RAW):
            return True
        time.sleep(0.2)
    return False


def _cleanup_legacy_systemd_unit() -> None:
    """v0.14.2: remove the v0.13.2..v0.14.1 woys-chain.service unit if
    present. The conf-file approach makes it redundant, and leaving it
    enabled would have it fire the v0.14.1 setup flow at login,
    duplicating modules alongside the conf-loaded filter-chain."""
    unit_path = _systemd_unit_path()
    if not unit_path.is_file():
        return
    _systemctl("disable", "--now", SYSTEMD_UNIT_NAME)
    unit_path.unlink(missing_ok=True)
    _systemctl("daemon-reload")
    print(
        f"[woys chain] removed legacy systemd unit {unit_path}\n"
        "             (v0.14.2 uses pipewire conf.d instead - no unit needed)"
    )


# --- v0.14.1 legacy module-based chain (fallback path) ----------------------


def _legacy_unload_chain_modules() -> int:
    """Unload v0.14.1 chain modules in reverse load order. Returns count."""
    mods = _list_modules()
    targets: list[str] = []
    for mod_id, mod_type, mod_args in mods:
        if mod_type == "module-remap-source" and f"source_name={SOURCE_USER_FACING}" in mod_args:
            targets.append(mod_id)
    for mod_id, mod_type, mod_args in mods:
        if mod_type == "module-loopback" and LOOPBACK_MARKER in mod_args:
            targets.append(mod_id)
    for mod_id, mod_type, mod_args in mods:
        if mod_type == "module-ladspa-sink" and f"sink_name={SINK_BRIDGE}" in mod_args:
            targets.append(mod_id)
    for mod_id, mod_type, mod_args in mods:
        if mod_type == "module-null-sink" and f"sink_name={SINK_FINAL}" in mod_args:
            targets.append(mod_id)
    for mod_id in targets:
        _pactl("unload-module", mod_id)
    return len(targets)


def _legacy_setup() -> int:
    """v0.14.1 module-based chain. Used only when filter-chain is
    unavailable. Identical behaviour to the v0.14.1 setup() that
    shipped in commit b8cdee5."""
    if not _source_present(SOURCE_RAW):
        print(
            f"[woys chain] {SOURCE_RAW} not present - run 'woys pw setup' first",
            file=sys.stderr,
        )
        return 2
    _legacy_unload_chain_modules()
    r1 = _pactl(
        "load-module",
        "module-null-sink",
        "media.class=Audio/Sink",
        f"sink_name={SINK_FINAL}",
        f"sink_properties=device.description={DESC_FINAL_SINK} node.passive=true"
        " session.suspend-timeout-seconds=0",
        "rate=48000",
        "channels=1",
    )
    if r1.returncode != 0:
        print(f"[woys chain] failed to load null-sink: {r1.stderr.strip()}", file=sys.stderr)
        return 2
    r2 = _pactl(
        "load-module",
        "module-ladspa-sink",
        f"sink_name={SINK_BRIDGE}",
        f"sink_master={SINK_FINAL}",
        f"plugin={PLUGIN_PATH}",
        f"label={PLUGIN_LABEL}",
        f"sink_properties=device.description={DESC_BRIDGE} node.passive=true"
        " session.suspend-timeout-seconds=0",
        "rate=48000",
        "channels=1",
    )
    if r2.returncode != 0:
        _legacy_unload_chain_modules()
        print(f"[woys chain] failed to load ladspa-sink: {r2.stderr.strip()}", file=sys.stderr)
        return 2
    r3 = _pactl(
        "load-module",
        "module-loopback",
        f"source={SOURCE_RAW}",
        f"sink={SINK_BRIDGE}",
        "rate=48000",
        "channels=1",
        "latency_msec=30",
    )
    if r3.returncode != 0:
        _legacy_unload_chain_modules()
        print(f"[woys chain] failed to load loopback: {r3.stderr.strip()}", file=sys.stderr)
        return 2
    r4 = _pactl(
        "load-module",
        "module-remap-source",
        f"master={SINK_FINAL}.monitor",
        f"source_name={SOURCE_USER_FACING}",
        f"source_properties=device.description={DESC_USER_FACING}",
        "rate=48000",
        "channels=1",
    )
    if r4.returncode != 0:
        _legacy_unload_chain_modules()
        print(
            f"[woys chain] failed to load user-facing remap-source: {r4.stderr.strip()}",
            file=sys.stderr,
        )
        return 2
    try:
        from audio.pipewire import SOURCE_DESC_CHAIN_ACTIVE, relabel_source

        relabel_source(SOURCE_DESC_CHAIN_ACTIVE, passive=True)
    except Exception as exc:
        print(f"[woys chain] warning: woys-mic relabel failed ({exc}).", file=sys.stderr)
    return 0


def _legacy_teardown() -> int:
    n = _legacy_unload_chain_modules()
    try:
        from audio.pipewire import SOURCE_DESC, relabel_source

        relabel_source(SOURCE_DESC, passive=False)
    except Exception as exc:
        print(f"[woys chain] warning: woys-mic relabel failed ({exc}).", file=sys.stderr)
    return n


# --- pw-link diagnostic (used by status to flag v0.13.0 ALSA leak) ----------


def _alsa_leak_links() -> list[str]:
    """Return any pw-link rows that bridge the chain to a hardware ALSA
    sink. Exists as a regression sentinel for the v0.13.0 leak; should
    return [] in v0.14.2 since there's no LADSPA-sink monitor to misroute."""
    pwlink = shutil.which("pw-link")
    if not pwlink:
        return []
    out = subprocess.run(
        [pwlink, "-l"], capture_output=True, text=True, timeout=3, check=False
    ).stdout
    leaks: list[str] = []
    section = ""
    for line in out.splitlines():
        if not line.startswith(" "):
            section = line.strip()
            continue
        if "alsa_output" in line and (
            "filter-chain" in section
            or "loopback" in section
            or SOURCE_RAW in section
            or SINK_FINAL in section
            or SINK_BRIDGE in section
        ):
            leaks.append(f"{section} {line.strip()}")
    return leaks


# --- public commands --------------------------------------------------------


def setup() -> int:
    """v0.14.2: write the filter-chain conf, restart the pipewire stack,
    relabel woys-mic, clean up any legacy systemd unit. Falls back to
    the v0.14.1 module-based chain if filter-chain is unavailable."""
    if not Path(PLUGIN_PATH).is_file():
        print(
            f"[woys chain] {PLUGIN_PATH} not found.\n"
            "             Install with: sudo pacman -S noise-suppression-for-voice",
            file=sys.stderr,
        )
        return 2

    if not filter_chain_supported():
        print(
            f"[woys chain] {FILTER_CHAIN_SO} not found - this PipeWire build "
            "does not\n             ship module-filter-chain. Falling back to "
            "v0.14.1 module-based chain.",
            file=sys.stderr,
        )
        return _legacy_setup()

    # Clear any legacy v0.14.1 chain currently loaded so we don't end
    # up with both running side-by-side after the pipewire restart.
    _legacy_unload_chain_modules()

    conf_path = _conf_file_path()
    conf_path.parent.mkdir(parents=True, exist_ok=True)
    conf_path.write_text(_render_conf())
    print(f"[woys chain] wrote filter-chain conf: {conf_path}")

    print(
        f"[woys chain] restarting pipewire stack to load the conf ({', '.join(PIPEWIRE_UNITS)})..."
    )
    ok, err = _restart_pipewire_stack()
    if not ok:
        print(f"[woys chain] pipewire restart failed: {err}", file=sys.stderr)
        # Roll back the conf so the next manual restart doesn't load
        # a half-broken state.
        conf_path.unlink(missing_ok=True)
        return 2

    # woys-mic.service has to re-arm before the filter-chain's
    # `target.object = woys-mic` binding can resolve.
    if not _ensure_woys_mic_loaded():
        print(
            "[woys chain] warning: woys-mic.service did not re-arm after "
            "pipewire restart.\n             Run `systemctl --user start "
            "woys-mic.service` (or `woys pw setup`)\n             to bring "
            "woys-mic back; the filter-chain will bind once it does.",
            file=sys.stderr,
        )

    if not _wait_for_user_facing_source():
        print(
            f"[woys chain] warning: {SOURCE_USER_FACING} did not appear after "
            "8s.\n             The conf is in place; check `journalctl --user "
            "-u pipewire`\n             for filter-chain load errors.",
            file=sys.stderr,
        )

    # Relabel woys-mic to mark it as the bypass option.
    try:
        from audio.pipewire import SOURCE_DESC_CHAIN_ACTIVE, relabel_source

        relabel_source(SOURCE_DESC_CHAIN_ACTIVE, passive=True)
    except Exception as exc:
        print(
            f"[woys chain] warning: woys-mic relabel failed ({exc}). "
            "Description stays at the v0.14.1 default.",
            file=sys.stderr,
        )

    # Old v0.14.1 systemd unit is now redundant.
    _cleanup_legacy_systemd_unit()

    print(
        f"\n[woys chain] active. Apps see exactly two woys input sources:\n"
        f"             {SOURCE_USER_FACING:24}[daily driver]\n"
        f"             {SOURCE_RAW:24}[raw bypass; description "
        "_internal-raw-bypass]\n"
        f"             The conf at {conf_path}\n"
        "             auto-loads at every pipewire startup. Run "
        "`woys chain teardown`\n             to remove it (also restarts "
        "the pipewire stack)."
    )
    return 0


def teardown() -> int:
    """v0.14.2: remove the filter-chain conf, restart pipewire, and
    also unload any legacy v0.14.1 modules a previous setup may have
    loaded. Restores woys-mic's daily-driver description."""
    conf_path = _conf_file_path()
    legacy_n = _legacy_unload_chain_modules()

    # Restore woys-mic's user-facing description regardless of which
    # path was active.
    try:
        from audio.pipewire import SOURCE_DESC, relabel_source

        relabel_source(SOURCE_DESC, passive=False)
    except Exception as exc:
        print(
            f"[woys chain] warning: woys-mic relabel failed ({exc}). "
            "Run `woys pw teardown && woys pw setup` to restore the default.",
            file=sys.stderr,
        )

    if conf_path.is_file():
        conf_path.unlink()
        print(f"[woys chain] removed filter-chain conf: {conf_path}")
        print(
            "[woys chain] restarting pipewire stack to apply removal "
            f"({', '.join(PIPEWIRE_UNITS)})..."
        )
        ok, err = _restart_pipewire_stack()
        if not ok:
            print(f"[woys chain] pipewire restart failed: {err}", file=sys.stderr)
            return 2
        _ensure_woys_mic_loaded()  # bring woys-mic back

    # Also clean up any leftover legacy systemd unit.
    _cleanup_legacy_systemd_unit()

    if not conf_path.is_file() and legacy_n == 0 and not _systemd_unit_path().is_file():
        print("[woys chain] not loaded (nothing to tear down)")
    elif legacy_n > 0:
        print(f"[woys chain] also unloaded {legacy_n} legacy v0.14.1 module(s)")
    return 0


def status() -> int:
    """Show conf-file presence, filter-chain output node, legacy module
    state, and the user-facing visibility section."""
    conf_path = _conf_file_path()
    chain_active = False

    print("[woys chain] mode:")
    if conf_path.is_file():
        print(f"  v0.14.2 filter-chain  conf: {conf_path}")
        chain_active = True
    legacy_modules = [
        (mod_id, mod_type, mod_args)
        for mod_id, mod_type, mod_args in _list_modules()
        if (mod_type == "module-null-sink" and f"sink_name={SINK_FINAL}" in mod_args)
        or (mod_type == "module-ladspa-sink" and f"sink_name={SINK_BRIDGE}" in mod_args)
        or (mod_type == "module-loopback" and LOOPBACK_MARKER in mod_args)
        or (mod_type == "module-remap-source" and f"source_name={SOURCE_USER_FACING}" in mod_args)
    ]
    if legacy_modules:
        print(f"  v0.14.1 legacy modules ({len(legacy_modules)} loaded):")
        for mod_id, mod_type, mod_args in legacy_modules:
            print(f"    {mod_id}\t{mod_type}\t{mod_args}")
        chain_active = True
    if not chain_active:
        print("  not loaded - use 'woys chain setup' to load it")

    print("\n[woys chain] sources libpulse exposes to apps (woys-related rows):")
    out = _pactl("list", "short", "sources").stdout
    woys_rows = [line for line in out.splitlines() if "woys" in line or "rnnoise" in line]
    for line in woys_rows:
        print(f"  {line}")
    if not woys_rows:
        print("  (none)")

    print("\n[woys chain] user-facing input devices apps will display")
    print("             (sources whose description does NOT contain `_internal-`):")
    long_out = _pactl("list", "sources").stdout
    user_facing = _user_facing_sources(long_out)
    woys_user_facing = [(name, desc) for name, desc in user_facing if "woys" in name]
    if woys_user_facing:
        for name, desc in woys_user_facing:
            print(f"  {name}    (description: {desc})")
    else:
        print("  (none - chain not loaded or all sources are marked internal)")
    if len(woys_user_facing) > 1:
        print(
            "  WARNING: more than one user-facing woys source. The chain expects\n"
            f"  exactly one (`{SOURCE_USER_FACING}`) when active. Check chain state."
        )

    leaks = _alsa_leak_links()
    if leaks:
        print("\n[woys chain] WARNING - chain audio is reaching ALSA hardware:")
        for leak in leaks:
            print(f"  {leak}")
        print(
            "  This is the v0.13.0 regression. Run 'woys chain teardown' then"
            " 'woys chain setup' to reset; if it persists, file a bug."
        )

    legacy_unit = _systemd_unit_path()
    if legacy_unit.is_file():
        print(
            f"\n[woys chain] legacy v0.14.1 systemd unit still present at {legacy_unit}.\n"
            "             v0.14.2 doesn't need it - the conf file is its own\n"
            "             persistence. Run `woys chain setup` to clean it up."
        )
    return 0


def enable() -> int:
    """v0.14.2: thin wrapper that just calls setup(). The conf file IS
    the persistence mechanism in v0.14.2 (PipeWire reads it at every
    daemon startup), so there's no separate "enable for auto-load on
    login" step. Kept as a CLI command so existing muscle memory and
    docs keep working."""
    print(
        "[woys chain] v0.14.2: `enable` and `setup` are now equivalent.\n"
        "             The conf at ~/.config/pipewire/pipewire.conf.d/ "
        "auto-loads\n             at every pipewire startup; no systemd unit "
        "is needed."
    )
    return setup()


def disable() -> int:
    """v0.14.2: thin wrapper that calls teardown() and also cleans up
    any legacy v0.14.1 systemd unit."""
    rc = teardown()
    return rc
