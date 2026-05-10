"""v0.13.3 - RNNoise post-engine chain with friendly device descriptions.

Loads four pipewire-pulse modules. The only one apps should pick is the
last; the other three are internal plumbing labelled `_internal-...` in
their device descriptions:

    woys-mic                       (description: woys-no-cleanup)
                                   ← raw v0.12.4 engine output, low latency
    woys-mic-rnnoise-bridge        (description: _internal-rnnoise-stage)
                                   ← LADSPA filter; not a useful endpoint
    woys-mic-clean                 (description: _internal-clean-sink)
                                   ← null-sink that the LADSPA writes into;
                                     its auto-monitor would also be visible
                                     but is the wrong endpoint for apps
                                     because it cannot be renamed away from
                                     the auto-derived "Monitor of …" prefix
    woys-by-alirexha               (description: woys-by-alirexha)
                                   ← module-remap-source that exposes the
                                     cleaned audio under a friendly name
                                     with no "Monitor of" prefix

Why the extra remap-source: pipewire-pulse's pactl cannot pass a
description containing spaces (the value is split on whitespace before
the proplist parser sees it), and the auto-monitor of a sink takes its
description from the sink prefixed with "Monitor of ". Routing the
cleaned audio through a `module-remap-source` lets us set a clean
`device.description=woys-by-alirexha` directly on the user-facing node.

v0.13.0 history (now mostly relevant only as background): an earlier
version of this chain used `media.class=Audio/Source/Virtual` on the
final null-sink, which wireplumber refused as a playback target, so
the LADSPA filter-chain output became an orphan stream and was auto-
routed to the default ALSA sink (= laptop speakers). v0.13.2 fixed the
routing (Architecture B with `media.class=Audio/Sink` and the `.monitor`
consumption pattern); v0.13.3 polishes the naming on top of that.

Mirrored functionality is also in `scripts/v013_2_rnnoise_chain.sh`
for users who don't have `woys` on PATH; this module is the source of
truth used by `woys chain`. Keep the two in sync if you change the
chain topology.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

PLUGIN_PATH = "/usr/lib/ladspa/librnnoise_ladspa.so"
PLUGIN_LABEL = "noise_suppressor_mono"

# Internal plumbing - apps see these but their `_internal-` description
# prefix tells the user not to pick them.
SINK_FINAL = "woys-mic-clean"
SINK_BRIDGE = "woys-mic-rnnoise-bridge"
SOURCE_RAW = "woys-mic"

# v0.13.3 - the user-facing remap-source. Apps see this in their input
# device dropdown as the friendly name `woys-by-alirexha`, with no
# "Monitor of" prefix.
SOURCE_USER_FACING = "woys-by-alirexha"

# v0.13.3 device descriptions. Hyphens substitute for spaces (pactl on
# pipewire-pulse breaks values containing whitespace). The two leading
# underscores in the internals push them down in alphabetical sorts.
DESC_USER_FACING = "woys-by-alirexha"
DESC_BRIDGE = "_internal-rnnoise-stage"
DESC_FINAL_SINK = "_internal-clean-sink"

LOOPBACK_MARKER = f"source={SOURCE_RAW} sink={SINK_BRIDGE}"
USER_REMAP_MARKER = f"master={SINK_FINAL}.monitor source_name={SOURCE_USER_FACING}"

SYSTEMD_UNIT_NAME = "woys-chain.service"


def _systemd_unit_path() -> Path:
    base = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
    return base / "systemd" / "user" / SYSTEMD_UNIT_NAME


def _pactl(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["pactl", *args], capture_output=True, text=True, timeout=5, check=False)


def _systemctl(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["systemctl", "--user", *args],
        capture_output=True,
        text=True,
        timeout=5,
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


def _unload_chain_modules() -> int:
    """Unload chain modules in reverse load order. Idempotent. Returns count unloaded."""
    mods = _list_modules()
    targets: list[str] = []
    # Order matters: leaves first, root last. Reverse of load order so
    # we don't leave a feeder pointing at a vanished node mid-teardown.
    #   1. user-facing remap-source  (depends on the .monitor of SINK_FINAL)
    #   2. loopback                  (feeds woys-mic into SINK_BRIDGE)
    #   3. ladspa-sink (SINK_BRIDGE) (writes into SINK_FINAL via sink_master)
    #   4. null-sink   (SINK_FINAL)  (root of the chain)
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


def _alsa_leak_links() -> list[str]:
    """Return any pw-link rows that bridge the chain to a hardware ALSA sink.

    Used by `status` to flag a regression of the v0.13.0 leak in case
    something else on the system later breaks our routing assumption.
    """
    pwlink = shutil.which("pw-link")
    if not pwlink:
        return []
    out = subprocess.run(
        [pwlink, "-l"], capture_output=True, text=True, timeout=3, check=False
    ).stdout
    leaks: list[str] = []
    section = ""
    for line in out.splitlines():
        # pw-link -l groups by node; child rows start with whitespace + '|'
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


def setup() -> int:
    if not Path(PLUGIN_PATH).is_file():
        print(
            f"[woys chain] {PLUGIN_PATH} not found.\n"
            "             Install with: sudo pacman -S noise-suppression-for-voice",
            file=sys.stderr,
        )
        return 2
    if not _source_present(SOURCE_RAW):
        print(
            f"[woys chain] {SOURCE_RAW} not present - run 'woys pw setup' first",
            file=sys.stderr,
        )
        return 2

    _unload_chain_modules()  # clear stale chain so setup is idempotent

    # 1. Terminal null-sink. Audio/Sink class (NOT Audio/Source/Virtual)
    #    so wireplumber accepts it as a playback target for the LADSPA
    #    output. The `_internal-` description prefix tells users "do
    #    not pick this in your dropdown" - apps should pick the
    #    user-facing remap-source loaded as step 4.
    r1 = _pactl(
        "load-module",
        "module-null-sink",
        "media.class=Audio/Sink",
        f"sink_name={SINK_FINAL}",
        f"sink_properties=device.description={DESC_FINAL_SINK}",
        "rate=48000",
        "channels=1",
    )
    if r1.returncode != 0:
        print(f"[woys chain] failed to load null-sink: {r1.stderr.strip()}", file=sys.stderr)
        return 2

    # 2. LADSPA-sink - actual RNNoise. Mono throughout: the noise-
    #    suppressor_mono plugin processes one channel; a stereo sink
    #    would spawn two filter instances and the resulting stereo
    #    stream would not bind to the mono master.
    r2 = _pactl(
        "load-module",
        "module-ladspa-sink",
        f"sink_name={SINK_BRIDGE}",
        f"sink_master={SINK_FINAL}",
        f"plugin={PLUGIN_PATH}",
        f"label={PLUGIN_LABEL}",
        f"sink_properties=device.description={DESC_BRIDGE}",
        "rate=48000",
        "channels=1",
    )
    if r2.returncode != 0:
        _unload_chain_modules()
        print(f"[woys chain] failed to load ladspa-sink: {r2.stderr.strip()}", file=sys.stderr)
        return 2

    # 3. Loopback that feeds woys-mic into the LADSPA bridge.
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
        _unload_chain_modules()
        print(f"[woys chain] failed to load loopback: {r3.stderr.strip()}", file=sys.stderr)
        return 2

    # 4. v0.13.3 - user-facing remap-source. Without this, apps would
    #    have to pick `woys-mic-clean.monitor` whose description is
    #    auto-derived as "Monitor of <sink description>" - pipewire-
    #    pulse offers no API to override that. A remap-source gives us
    #    a brand-new node we name `woys-by-alirexha` with matching
    #    `device.description`, so apps see one obvious daily-driver
    #    option in their dropdown alongside `woys-no-cleanup`.
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
        _unload_chain_modules()
        print(
            f"[woys chain] failed to load user-facing remap-source: {r4.stderr.strip()}",
            file=sys.stderr,
        )
        return 2

    print(
        "[woys chain] active. Apps see two woys options in their input picker:\n"
        f"  {SOURCE_USER_FACING}      - RNNoise-cleaned (+40 ms, ~ -27% cuts/min) "
        "[daily driver]\n"
        f"  {SOURCE_RAW}             - raw engine output, low latency, "
        "no RNNoise [fallback]\n"
        "Anything else marked `_internal-...` is plumbing - don't pick it."
    )
    return 0


def teardown() -> int:
    n = _unload_chain_modules()
    if n == 0:
        print("[woys chain] not loaded (nothing to tear down)")
    else:
        print(f"[woys chain] unloaded {n} module(s)")
    return 0


def status() -> int:
    print("[woys chain] modules:")
    matched = False
    for mod_id, mod_type, mod_args in _list_modules():
        if (
            (mod_type == "module-null-sink" and f"sink_name={SINK_FINAL}" in mod_args)
            or (mod_type == "module-ladspa-sink" and f"sink_name={SINK_BRIDGE}" in mod_args)
            or (mod_type == "module-loopback" and LOOPBACK_MARKER in mod_args)
            or (
                mod_type == "module-remap-source"
                and f"source_name={SOURCE_USER_FACING}" in mod_args
            )
        ):
            print(f"  {mod_id}\t{mod_type}\t{mod_args}")
            matched = True
    if not matched:
        print("  (chain not loaded - use 'woys chain setup' to load it)")

    print("\n[woys chain] sources visible to apps:")
    out = _pactl("list", "short", "sources").stdout
    for line in out.splitlines():
        if "woys" in line or "rnnoise" in line:
            print(f"  {line}")

    leaks = _alsa_leak_links()
    if leaks:
        print("\n[woys chain] WARNING - chain audio is reaching ALSA hardware:")
        for leak in leaks:
            print(f"  {leak}")
        print(
            "  This is the v0.13.0 regression. Run 'woys chain teardown' then"
            " 'woys chain setup' to reset; if it persists, file a bug."
        )

    unit_path = _systemd_unit_path()
    print()
    if unit_path.is_file():
        is_enabled = _systemctl("is-enabled", SYSTEMD_UNIT_NAME).stdout.strip() or "?"
        is_active = _systemctl("is-active", SYSTEMD_UNIT_NAME).stdout.strip() or "?"
        print(
            f"[woys chain] systemd user unit installed at {unit_path}\n"
            f"             enabled: {is_enabled}    active: {is_active}"
        )
    else:
        print(
            "[woys chain] systemd user unit NOT installed"
            " (use 'woys chain enable' to auto-load on login)"
        )
    return 0


def enable() -> int:
    if not Path(PLUGIN_PATH).is_file():
        print(
            f"[woys chain] {PLUGIN_PATH} not found.\n"
            "             Install with: sudo pacman -S noise-suppression-for-voice",
            file=sys.stderr,
        )
        return 2

    woys_bin = shutil.which("woys") or sys.argv[0]
    unit_path = _systemd_unit_path()
    unit_path.parent.mkdir(parents=True, exist_ok=True)

    unit_text = f"""# v0.13.2 - woys RNNoise post-engine chain user unit. Auto-loads
# the chain on login so apps that select 'woys-mic-clean.monitor'
# get RNNoise-processed audio without manual setup. Generated by
# 'woys chain enable'; remove with 'woys chain disable'.
[Unit]
Description=woys RNNoise post-engine chain (v0.13.2)
After=pipewire.service pipewire-pulse.service
Requires=pipewire-pulse.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart={woys_bin} chain setup
ExecStop={woys_bin} chain teardown

[Install]
WantedBy=default.target
"""
    unit_path.write_text(unit_text)

    r1 = _systemctl("daemon-reload")
    if r1.returncode != 0:
        print(f"[woys chain] systemctl daemon-reload failed: {r1.stderr.strip()}", file=sys.stderr)
        return 2
    r2 = _systemctl("enable", "--now", SYSTEMD_UNIT_NAME)
    if r2.returncode != 0:
        print(
            f"[woys chain] systemctl enable --now failed:\n{r2.stderr.strip()}",
            file=sys.stderr,
        )
        return 2

    print(
        f"[woys chain] systemd user unit installed at {unit_path}\n"
        "             enabled + started - chain auto-loads on every login"
    )
    return 0


def disable() -> int:
    unit_path = _systemd_unit_path()
    if unit_path.is_file():
        _systemctl("disable", "--now", SYSTEMD_UNIT_NAME)
        unit_path.unlink(missing_ok=True)
        _systemctl("daemon-reload")
        print(f"[woys chain] systemd user unit disabled + removed: {unit_path}")
    else:
        print("[woys chain] no systemd user unit installed")
    n = _unload_chain_modules()
    print(f"[woys chain] unloaded {n} module(s)")
    return 0
