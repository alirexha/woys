"""v0.14.1 - RNNoise post-engine chain with single-default visibility.

Loads four pipewire-pulse modules and re-labels woys-mic so apps that
show device descriptions render exactly ONE non-internal option. State
when chain is active:

    woys-mic                       (description: _internal-raw-bypass)
                                   ← raw v0.12.4 engine output. v0.13.3
                                     and earlier showed `woys-no-cleanup`
                                     here as a "second daily driver"; v0.14.1
                                     marks it internal so users with the
                                     chain see only one option in app
                                     pickers. Apps that pin by exact name
                                     (`woys-mic`) still resolve - only the
                                     description changed. node.passive=true
                                     is also set as a hint to PipeWire-
                                     native consumers (libpulse ignores it).
    woys-mic-rnnoise-bridge        (description: _internal-rnnoise-stage)
                                   ← LADSPA filter; not a useful endpoint
    woys-mic-clean                 (description: _internal-clean-sink)
                                   ← null-sink that the LADSPA writes into;
                                     its auto-monitor would also be visible
                                     but is the wrong endpoint for apps
                                     because it cannot be renamed away from
                                     the auto-derived "Monitor of …" prefix
    woys-clean               (description: woys-clean)
                                   ← module-remap-source that exposes the
                                     cleaned audio under a friendly name
                                     with no "Monitor of" prefix

When the chain is torn down, woys-mic's description is restored to
`woys-no-cleanup` (the v0.13.3 daily-driver name) so users without the
chain still see a sensible label.

Hard limit on hiding: pulseaudio compat (libpulse, used by pavucontrol /
Telegram / Discord / KDE Volume Mixer) enumerates ALL sources
regardless of node.passive / node.virtual / object.register=false. We
verified empirically: the rnnoise-bridge monitor already has
`object.register=false` and still shows up in `pactl list short
sources`. There is no PipeWire property that hides a source from
libpulse-based apps. v0.14.1 therefore relies on the `_internal-`
description prefix as the only signal those apps render to a user, and
documents the limitation honestly. Apps still see four `woys*` sources;
they just see one with a non-`_internal-` description.

Why the extra remap-source: pipewire-pulse's pactl cannot pass a
description containing spaces (the value is split on whitespace before
the proplist parser sees it), and the auto-monitor of a sink takes its
description from the sink prefixed with "Monitor of ". Routing the
cleaned audio through a `module-remap-source` lets us set a clean
`device.description=woys-clean` directly on the user-facing node.

v0.13.0 history (now mostly relevant only as background): an earlier
version of this chain used `media.class=Audio/Source/Virtual` on the
final null-sink, which wireplumber refused as a playback target, so
the LADSPA filter-chain output became an orphan stream and was auto-
routed to the default ALSA sink (= laptop speakers). v0.13.2 fixed the
routing (Architecture B with `media.class=Audio/Sink` and the `.monitor`
consumption pattern); v0.13.3 polishes the naming on top of that.

v0.14.0 (Lens 1 / Lens 19 / C034 + C043): the parallel
`scripts/v013_*_rnnoise_chain.sh` shell implementations were deleted.
They duplicated this module's topology without `set -euo pipefail` or
return-code checks on `pactl load-module`, so partial chain failures
silently reported "active". This module is the single source of truth;
external scripts that want to reproduce the chain should `import
audio.woys.chain` rather than re-implement it in shell.
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
# device dropdown as the friendly name `woys-clean`, with no
# "Monitor of" prefix.
SOURCE_USER_FACING = "woys-clean"

# v0.13.3 device descriptions. Hyphens substitute for spaces (pactl on
# pipewire-pulse breaks values containing whitespace). The two leading
# underscores in the internals push them down in alphabetical sorts.
DESC_USER_FACING = "woys-clean"
DESC_BRIDGE = "_internal-rnnoise-stage"
DESC_FINAL_SINK = "_internal-clean-sink"

LOOPBACK_MARKER = f"source={SOURCE_RAW} sink={SINK_BRIDGE}"
USER_REMAP_MARKER = f"master={SINK_FINAL}.monitor source_name={SOURCE_USER_FACING}"

SYSTEMD_UNIT_NAME = "woys-chain.service"


def _systemd_unit_path() -> Path:
    base = Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config")))
    return base / "systemd" / "user" / SYSTEMD_UNIT_NAME


def _c_locale_env() -> dict[str, str]:
    """review F-15-05: env for parsing subprocesses. pactl / pw-link /
    systemctl output is localised; our parsers key off literal English
    tokens (`Description: `, `Source #`, ...), so on a non-English `$LANG`
    every parse silently misses and `woys chain status` shows zero devices
    with no error. Forcing `LC_ALL=C` keeps the output English."""
    return {**os.environ, "LC_ALL": "C"}


def _pactl(*args: str) -> subprocess.CompletedProcess[str]:
    """Run `pactl <args>` with the same hardening
    `audio.pipewire._run_pactl` provides.

    review F-merged-009 (commit-070): pre-fix chain.py and
    audio/pipewire.py had two divergent pactl shell-out wrappers.
    chain.py's pre-fix version used bare `"pactl"` (no `shutil.
    which` + missing-tool guard); audio/pipewire.py's had both.
    Inline the same behavior here -- a `subprocess.run` call from
    chain's module scope is what the chain tests already patch
    (`patch(chain.subprocess.run, ...)`), so delegating to
    `_run_pactl` would silently bypass those patches.
    """
    if shutil.which("pactl") is None:
        # F-merged-009: typed not-found, matching pipewire.py's
        # missing-tool error shape (but routed through the same
        # `CompletedProcess` surface chain callers already handle).
        # Returncode 127 is the conventional "command not found".
        return subprocess.CompletedProcess(
            args=["pactl", *args],
            returncode=127,
            stdout="",
            stderr="pactl not found - install `pipewire-pulse`",
        )
    # Use bare "pactl" (not the absolute path from shutil.which) so the
    # call shape stays stable for tests that pattern-match cmd[0].
    return subprocess.run(
        ["pactl", *args],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
        env=_c_locale_env(),
    )


def _systemctl(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["systemctl", "--user", *args],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
        env=_c_locale_env(),
    )


def _default_sink() -> str:
    """The system default sink's name (`pactl get-default-sink`), or '' on error."""
    out = _pactl("get-default-sink")
    return out.stdout.strip() if out.returncode == 0 else ""


def _list_modules() -> list[tuple[str, str, str]]:
    """Return [(id, type, args)] for currently loaded pipewire-pulse modules.

    review F-cx4-001 / F-05-14 P2a (commit-064): match
    `pipewire.py:_list_modules` by validating that the first column
    is a non-negative integer. Pre-fix any garbage in column 0
    propagated as a module ID to `_unload_chain_modules`'s
    `pactl unload-module <id>` call -- a daemon-corrupted output
    or a future pactl format change could feed an attacker-named
    token back as a command arg. `int()` validation rejects non-
    numeric tokens cleanly. The return type stays `str` because
    downstream callers use the value as a CLI argument verbatim;
    the validation is the load-bearing part.
    """
    out = _pactl("list", "short", "modules").stdout
    rows: list[tuple[str, str, str]] = []
    for line in out.splitlines():
        parts = line.split("\t", 2)
        if len(parts) < 2:
            continue
        try:
            int(parts[0])  # F-cx4-001: validation; result discarded.
        except ValueError:
            continue
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


def _is_user_facing_description(description: str) -> bool:
    """A source description is user-facing if it does NOT contain the
    `_internal-` marker. We check `contains` rather than `startswith`
    so the auto-derived "Monitor of _internal-..." description
    (pipewire-pulse adds the `Monitor of ` prefix to a sink's auto-
    monitor and offers no API to override it) is also filtered out.
    """
    return "_internal-" not in description


def _user_facing_sources(pactl_list_sources_output: str) -> list[tuple[str, str]]:
    """v0.14.1 - parse `pactl list sources` (long form) and return
    [(name, description), ...] for sources whose description is
    user-facing per `_is_user_facing_description`.

    Used by `status()` to show what apps showing device descriptions
    will render as user-facing options. Pure function so the test
    suite can feed it canned output.
    """
    results: list[tuple[str, str]] = []
    cur_name: str | None = None
    cur_desc: str | None = None
    for raw in pactl_list_sources_output.splitlines():
        # New "Source #N" block resets state.
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
    # Last block.
    if cur_name and cur_desc and _is_user_facing_description(cur_desc):
        results.append((cur_name, cur_desc))
    return results


def _alsa_leak_links() -> list[str]:
    """Return any pw-link rows that bridge the chain to a hardware ALSA sink.

    Used by `status` to flag a regression of the v0.13.0 leak in case
    something else on the system later breaks our routing assumption.
    """
    pwlink = shutil.which("pw-link")
    if not pwlink:
        return []
    out = subprocess.run(
        [pwlink, "-l"],
        capture_output=True,
        text=True,
        timeout=3,
        check=False,
        env=_c_locale_env(),  # review F-15-05: English-token parsing
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
    #
    #    v0.14.1 - sink_properties also carries node.passive=true and
    #    session.suspend-timeout-seconds=0. Pulseaudio compat ignores
    #    these (we tested), but PipeWire-native consumers respect
    #    node.passive as a "this is plumbing, not a user endpoint"
    #    hint, and the suspend override keeps the sink scheduled while
    #    the chain is running so the rnnoise filter doesn't go to
    #    sleep mid-conversation.
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
        f"sink_properties=device.description={DESC_BRIDGE} node.passive=true"
        " session.suspend-timeout-seconds=0",
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
    #    a brand-new node we name `woys-clean` with matching
    #    `device.description`, so apps see one obvious daily-driver
    #    option in their dropdown.
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

    # 5. v0.14.1 - relabel woys-mic as `_internal-raw-bypass` so apps
    #    that show descriptions render `woys-clean` as the only
    #    non-internal option. The source NAME stays `woys-mic` for
    #    back-compat with apps that pin by exact name. node.passive=
    #    true is also set on the relabel.
    #
    #    Done last so that if it fails, the rest of the chain (which
    #    actually carries audio) is already running. We log a warning
    #    and return success, because the chain is functionally fine
    #    even if the cosmetic relabel didn't take.
    # review F-merged-006: the pre-fix code caught a relabel failure,
    # printed a stderr "warning", and `return 0` -- so `woys-chain.service`
    # reported `active` even when (pre-F-merged-006) the relabel had
    # destroyed woys-mic. `relabel_source` is now atomic (it rolls back, so
    # woys-mic survives), but a relabel failure is still a daemon-path step
    # that did not complete: fail loudly with exit 2 so the unit shows
    # `failed` rather than reporting success on a partially-applied chain.
    try:
        from audio.pipewire import SOURCE_DESC_CHAIN_ACTIVE, relabel_source
    except ImportError as exc:
        print(
            f"[woys chain] internal: cannot import relabel_source ({exc}); "
            "woys-mic relabel did not run.",
            file=sys.stderr,
        )
        return 2
    try:
        relabel_source(SOURCE_DESC_CHAIN_ACTIVE, passive=True)
    except Exception as exc:
        print(
            f"[woys chain] woys-mic relabel failed ({exc}). The RNNoise chain "
            "modules are loaded and woys-mic is intact (relabel_source rolls "
            "back atomically), but this daemon-path step did not complete -- "
            "failing with exit 2 so the unit does not report success on a "
            "partially-applied chain.",
            file=sys.stderr,
        )
        return 2

    print(
        "[woys chain] active. Apps that show device descriptions will display\n"
        "             one non-internal woys option:\n"
        f"             {SOURCE_USER_FACING}      [daily driver]\n"
        "             Other woys* sources are still listed (libpulse limitation)\n"
        f"             but their `_internal-` descriptions mark them as plumbing.\n"
        f"             Use 'woys chain status' to see what apps will show."
    )
    return 0


def teardown() -> int:
    n = _unload_chain_modules()

    # v0.14.1 - restore woys-mic's daily-driver description so users
    # without the chain (or after teardown) see a sensible label.
    # review F-merged-006: a failed restore is reported via a non-zero
    # exit, not swallowed as a "warning" + return 0.
    relabel_failed = False
    try:
        from audio.pipewire import SOURCE_DESC, relabel_source
    except ImportError as exc:
        print(
            f"[woys chain] internal: cannot import relabel_source ({exc}); "
            "woys-mic restore did not run.",
            file=sys.stderr,
        )
        relabel_failed = True
    else:
        try:
            relabel_source(SOURCE_DESC, passive=False)
        except Exception as exc:
            print(
                f"[woys chain] woys-mic relabel-to-default failed ({exc}). The "
                "chain modules were unloaded and woys-mic is intact "
                "(relabel_source rolls back atomically) but keeps its "
                "chain-active description. Run `woys pw teardown && woys pw "
                "setup` to restore the default label.",
                file=sys.stderr,
            )
            relabel_failed = True

    if n == 0:
        print("[woys chain] not loaded (nothing to tear down)")
    else:
        print(f"[woys chain] unloaded {n} module(s)")
    return 1 if relabel_failed else 0


def _health_check() -> int:
    """review F-08-06: the assertions wired into woys-mic.service and
    woys-chain.service as `ExecStartPost`. Exits non-zero if the woys audio
    plumbing is in the broken state the v0.14.2 incident produced -- the
    only thing that would otherwise catch it is a human noticing dead audio,
    because the `Type=oneshot` units report `active (exited)` regardless.

    Asserts:
      (a) the base woys-mic source is present;
      (b) the system default sink is NOT a woys null-sink -- the v0.14.2
          hijack routed all desktop audio into woys plumbing; a leak-only
          check would miss it (CX2's mandatory scope pin);
      (c) no pw-link rows bridge the chain to ALSA hardware.
    """
    ok = True

    if _source_present(SOURCE_RAW):
        print(f"[woys chain] check: {SOURCE_RAW} source present  OK")
    else:
        print(f"[woys chain] check: {SOURCE_RAW} source MISSING  FAIL", file=sys.stderr)
        ok = False

    # (b) the load-bearing assertion. v0.14.2 left the system default sink
    # pointed at a woys null-sink, silently sending all desktop audio into
    # woys plumbing instead of the speakers.
    woys_sinks = {SINK_FINAL}
    try:
        from audio.pipewire import SINK_NAME as _PW_SINK

        woys_sinks.add(_PW_SINK)
    except ImportError:
        pass  # pipewire module unavailable; still check the chain's own sink
    default_sink = _default_sink()
    if default_sink and default_sink in woys_sinks:
        print(
            f"[woys chain] check: system default sink is {default_sink!r}, a woys "
            "null-sink -- desktop audio is being routed into woys plumbing  FAIL",
            file=sys.stderr,
        )
        ok = False
    else:
        print(
            f"[woys chain] check: default sink {default_sink or '(unknown)'!r} "
            "is not a woys null-sink  OK"
        )

    leaks = _alsa_leak_links()
    if leaks:
        print("[woys chain] check: chain audio is reaching ALSA hardware  FAIL", file=sys.stderr)
        for leak in leaks:
            print(f"  {leak}", file=sys.stderr)
        ok = False
    else:
        print("[woys chain] check: no ALSA leak links  OK")

    if ok:
        print("[woys chain] check: PASS")
        return 0
    print(
        "[woys chain] check: FAIL -- run `woys chain disable` and/or "
        "`woys pw teardown` to clear the broken state, then re-setup.",
        file=sys.stderr,
    )
    return 1


def status(check: bool = False) -> int:
    # review F-08-06: `--check` runs only the health assertions and
    # exits non-zero on failure, so the systemd units' ExecStartPost can
    # turn a broken-audio chain into a `failed` unit instead of a green
    # `active (exited)` that hides the v0.14.2 regression class.
    if check:
        return _health_check()
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

    print("\n[woys chain] sources libpulse exposes to apps (full list):")
    out = _pactl("list", "short", "sources").stdout
    woys_sources: list[str] = []
    for line in out.splitlines():
        if "woys" in line or "rnnoise" in line:
            print(f"  {line}")
            woys_sources.append(line)

    # v0.14.1 - "user sees" section: which of those sources have a
    # non-`_internal-` description, i.e. which ones an app showing
    # device descriptions will render as a sensible daily-driver pick.
    # Read descriptions via `pactl list sources` (long form) since
    # `list short` only has the name.
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
            "  exactly one (`woys-clean`) when active. Check your chain state."
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

    # review F-05-13 (commit-065): resolve + validate the
    # `woys` binary path before string-interpolating it into a
    # systemd unit. Pre-fix:
    #   woys_bin = shutil.which("woys") or sys.argv[0]
    # had two correctness + security gaps:
    #   1. A path containing a space (rare but possible -- a venv
    #      under `~/My Apps/woys/`) blew the systemd ExecStart line
    #      into a "command + extra arg" parse, breaking the unit on
    #      every login.
    #   2. `sys.argv[0]` fallback is caller-controlled (defense-in-
    #      depth: an attacker with code-exec could prime sys.argv[0]
    #      to point at a malicious binary that would later be
    #      executed by systemd on user login).
    # Fix: resolve to an absolute path, assert it's executable, and
    # refuse systemd-special characters (whitespace, percent,
    # newline) -- hard-fail with a clear message rather than write
    # a broken unit.
    woys_bin_raw = shutil.which("woys")
    if woys_bin_raw is None:
        woys_bin_raw = sys.argv[0]
    woys_bin_path = Path(woys_bin_raw).resolve()
    if not woys_bin_path.is_file():
        print(
            f"[chain] enable refused: woys binary not found at {woys_bin_path} "
            f"(resolved from {woys_bin_raw!r}). F-05-13: refusing to write a "
            f"systemd unit with a broken ExecStart.",
            file=sys.stderr,
        )
        return 2
    if not os.access(woys_bin_path, os.X_OK):
        print(
            f"[chain] enable refused: woys binary at {woys_bin_path} is not "
            f"executable. F-05-13: refusing to write a broken systemd unit.",
            file=sys.stderr,
        )
        return 2
    bad_chars = set(str(woys_bin_path)) & set(" \t\n%")
    if bad_chars:
        print(
            f"[chain] enable refused: woys binary path {woys_bin_path!r} "
            f"contains systemd-special characters ({bad_chars!r}). "
            f"F-05-13: refusing to write a unit that systemd would parse "
            f"as `cmd + extra-args`. Reinstall woys at a path without "
            f"whitespace / `%`.",
            file=sys.stderr,
        )
        return 2
    woys_bin = str(woys_bin_path)
    unit_path = _systemd_unit_path()
    unit_path.parent.mkdir(parents=True, exist_ok=True)

    unit_text = f"""# woys RNNoise post-engine chain user unit. Auto-loads
# the chain on login so apps that select 'woys-mic-clean.monitor'
# get RNNoise-processed audio without manual setup. Generated by
# 'woys chain enable'; remove with 'woys chain disable'.
[Unit]
Description=woys RNNoise post-engine chain
After=pipewire.service pipewire-pulse.service
Requires=pipewire-pulse.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart={woys_bin} chain setup
# review F-08-06: assert the chain is actually healthy after setup.
# A failed ExecStartPost makes the unit `failed` instead of a green
# `active (exited)` that hides a broken chain (the v0.14.2 class). Note:
# a failed check leaves the ExecStart modules loaded -- run
# `woys chain disable` to clear them.
ExecStartPost={woys_bin} chain status --check
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
