"""v0.13.2 — RNNoise post-engine chain (the "chain" subcommand).

Loads three pipewire-pulse modules to add an RNNoise-cleaned variant of
woys-mic that apps can select directly:

    woys-mic                       — raw v0.12.4 engine output
    woys-mic-clean.monitor         — RNNoise-cleaned (~ -27% cuts/min,
                                      +40 ms latency)

v0.13.0 had a routing bug: the LADSPA filter-chain output stream
auto-routed to the default ALSA sink (so audio leaked to speakers
regardless of woys's monitor toggle). Root cause: the destination
null-sink used `media.class=Audio/Source/Virtual`, which wireplumber
does not accept as a valid playback target — so `sink_master=` never
bound and the orphan stream fell through to default-sink policy
routing. Fix here: use `media.class=Audio/Sink` and have apps consume
from the auto-created `.monitor` source instead.

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
SINK_FINAL = "woys-mic-clean"
SINK_BRIDGE = "woys-mic-rnnoise-bridge"
SOURCE_RAW = "woys-mic"
LOOPBACK_MARKER = f"source={SOURCE_RAW} sink={SINK_BRIDGE}"

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
    # Order matters: loopback first (so we don't leave a feeder pointing
    # at a vanished sink), then ladspa-sink, then null-sink.
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
            f"[woys chain] {SOURCE_RAW} not present — run 'woys pw setup' first",
            file=sys.stderr,
        )
        return 2

    _unload_chain_modules()  # clear stale chain so setup is idempotent

    # 1. Terminal null-sink. Audio/Sink class (NOT Audio/Source/Virtual)
    #    so wireplumber accepts it as a playback target for the LADSPA
    #    output. Apps record from its auto-created .monitor source.
    r1 = _pactl(
        "load-module",
        "module-null-sink",
        "media.class=Audio/Sink",
        f"sink_name={SINK_FINAL}",
        "sink_properties=device.description=woys-mic-clean_rnnoise",
        "rate=48000",
        "channels=1",
    )
    if r1.returncode != 0:
        print(f"[woys chain] failed to load null-sink: {r1.stderr.strip()}", file=sys.stderr)
        return 2

    # 2. LADSPA-sink — actual RNNoise. Mono throughout: the noise-
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

    print(
        "[woys chain] active. Apps can now select:\n"
        "  woys-mic                — raw engine output\n"
        "  woys-mic-clean.monitor  — RNNoise-cleaned (+40 ms, ~ -27% cuts/min)\n"
        "Note the .monitor suffix — selecting woys-mic-clean directly is a"
        " sink, not a source."
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
        ):
            print(f"  {mod_id}\t{mod_type}\t{mod_args}")
            matched = True
    if not matched:
        print("  (chain not loaded — use 'woys chain setup' to load it)")

    print("\n[woys chain] sources visible to apps:")
    out = _pactl("list", "short", "sources").stdout
    for line in out.splitlines():
        if "woys" in line or "rnnoise" in line:
            print(f"  {line}")

    leaks = _alsa_leak_links()
    if leaks:
        print("\n[woys chain] WARNING — chain audio is reaching ALSA hardware:")
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

    unit_text = f"""# v0.13.2 — woys RNNoise post-engine chain user unit. Auto-loads
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
        "             enabled + started — chain auto-loads on every login"
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
