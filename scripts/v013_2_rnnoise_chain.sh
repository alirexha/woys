#!/bin/bash
# v0.13.3 — opt-in RNNoise chain after woys-mic, friendly device names.
#
# v0.13.3 adds `module-remap-source` on top of the v0.13.2 architecture
# so apps see one user-facing daily-driver source named
# `woys-by-alirexha` (with matching device.description), instead of
# the auto-derived `Monitor of <sink-description>` that pipewire-pulse
# offers no API to override. All intermediate sinks are tagged with
# `_internal-...` descriptions so they sort/communicate "do not pick".
#
# v0.13.2 — opt-in RNNoise chain after woys-mic, FIXED routing.
#
# v0.13.0 architecture had a leak: the LADSPA filter-chain output stream
# was auto-routed by wireplumber to the default ALSA sink AS WELL AS the
# intended `woys-mic-clean` source. Result: audio played out of speakers
# (regardless of woys's monitor toggle) AND the metric we thought was
# RNNoise-cleaned was actually post-RNNoise audio plus speaker echo.
#
# v0.13.2 root cause: `media.class=Audio/Source/Virtual` on the destination
# null-sink made wireplumber refuse to recognize it as a valid playback
# target for the filter-chain stream → orphan stream → policy routes to
# default ALSA. Fix: use `media.class=Audio/Sink` and have apps consume
# from the AUTO-CREATED `woys-mic-clean.monitor` source. The trade is
# one extra word (`.monitor`) in the app's input-device picker, in
# exchange for zero leak to speakers.
#
# Measured impact (TTS-driven engine, v0.12.4 defaults, mode=both, 30 s,
# both recordings via serial-ID pw-record):
#   woys-mic              : 75.4 cuts/min (woys-diag)
#   woys-mic-clean.monitor: 54.7 cuts/min  (-27 %)
#
# The 27 % is the real RNNoise contribution; v0.13.0's reported 13 %
# was contaminated by the alsa-leak feedback path.
#
# Latency cost on top of v0.12.4: ~40 ms (loopback + RNNoise frame).
#
# Requires: pacman -S noise-suppression-for-voice
#
# Usage:
#   ./scripts/v013_2_rnnoise_chain.sh setup     # load the chain (one-shot)
#   ./scripts/v013_2_rnnoise_chain.sh teardown  # unload everything
#   ./scripts/v013_2_rnnoise_chain.sh status    # show current state
#   ./scripts/v013_2_rnnoise_chain.sh enable    # install systemd user unit + start now + enable on login
#   ./scripts/v013_2_rnnoise_chain.sh disable   # stop, disable, remove unit
set -euo pipefail

PLUGIN_PATH="/usr/lib/ladspa/librnnoise_ladspa.so"
PLUGIN_LABEL="noise_suppressor_mono"
SINK_FINAL="woys-mic-clean"
SINK_BRIDGE="woys-mic-rnnoise-bridge"
SOURCE_USER_FACING="woys-by-alirexha"
DESC_USER_FACING="woys-by-alirexha"
DESC_BRIDGE="_internal-rnnoise-stage"
DESC_FINAL_SINK="_internal-clean-sink"
LOOPBACK_MARKER="source=woys-mic sink=$SINK_BRIDGE"
USER_REMAP_MARKER="master=$SINK_FINAL.monitor source_name=$SOURCE_USER_FACING"
SYSTEMD_UNIT_NAME="woys-chain.service"
SYSTEMD_UNIT_DIR="$HOME/.config/systemd/user"
SYSTEMD_UNIT_PATH="$SYSTEMD_UNIT_DIR/$SYSTEMD_UNIT_NAME"

# Resolve the absolute path of this script so the systemd unit can find it.
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

cmd="${1:-status}"

_require_plugin() {
    if [ ! -f "$PLUGIN_PATH" ]; then
        echo "[v0.13.2] $PLUGIN_PATH not found — install with:" >&2
        echo "          sudo pacman -S noise-suppression-for-voice" >&2
        exit 1
    fi
}

_require_woysmic() {
    if ! pactl list short sources 2>/dev/null | grep -qE "^[0-9]+\s+woys-mic\s"; then
        echo "[v0.13.2] woys-mic source not present — run 'woys pw setup' first" >&2
        exit 1
    fi
}

_unload_chain_modules() {
    # Order: leaves first, root last (reverse of load order).
    #   1. user-facing remap-source  (depends on the .monitor of SINK_FINAL)
    #   2. loopback                  (feeds woys-mic into SINK_BRIDGE)
    #   3. ladspa-sink (SINK_BRIDGE) (writes into SINK_FINAL via sink_master)
    #   4. null-sink   (SINK_FINAL)  (root)
    for spec in "module-remap-source.*source_name=$SOURCE_USER_FACING" \
                "module-loopback.*$LOOPBACK_MARKER" \
                "module-ladspa-sink.*sink_name=$SINK_BRIDGE" \
                "module-null-sink.*sink_name=$SINK_FINAL"; do
        for mod_id in $(pactl list short modules | awk -v p="$spec" '$0 ~ p {print $1}'); do
            pactl unload-module "$mod_id" 2>/dev/null || true
        done
    done
}

_load_chain() {
    _require_plugin
    _require_woysmic
    _unload_chain_modules  # idempotent — clear any stale chain first

    # 1. Terminal null-sink (Audio/Sink class — see v0.13.2 history).
    pactl load-module module-null-sink \
        media.class=Audio/Sink \
        sink_name="$SINK_FINAL" \
        sink_properties=device.description="$DESC_FINAL_SINK" \
        rate=48000 channels=1 >/dev/null

    # 2. LADSPA-sink with the rnnoise plugin. Output goes to its
    #    sink_master ($SINK_FINAL). Both legs MUST be mono (the
    #    `noise_suppressor_mono` plugin processes 1 channel; if the
    #    sink is stereo, PipeWire spawns two filter instances with
    #    a stereo output stream that won't bind to a mono master).
    pactl load-module module-ladspa-sink \
        sink_name="$SINK_BRIDGE" \
        sink_master="$SINK_FINAL" \
        plugin="$PLUGIN_PATH" \
        label="$PLUGIN_LABEL" \
        sink_properties=device.description="$DESC_BRIDGE" \
        rate=48000 channels=1 >/dev/null

    # 3. Loopback that feeds woys-mic into the LADSPA bridge.
    pactl load-module module-loopback \
        source=woys-mic sink="$SINK_BRIDGE" \
        rate=48000 channels=1 latency_msec=30 >/dev/null

    # 4. v0.13.3 — user-facing remap-source. Apps pick this one.
    pactl load-module module-remap-source \
        master="$SINK_FINAL.monitor" \
        source_name="$SOURCE_USER_FACING" \
        source_properties=device.description="$DESC_USER_FACING" \
        rate=48000 channels=1 >/dev/null
}

case "$cmd" in
setup)
    _load_chain
    echo "[v0.13.3] RNNoise chain active. Apps see two woys options in the input picker:"
    echo "          $SOURCE_USER_FACING      — RNNoise-cleaned (+40 ms, ~ -27% cuts/min) [daily driver]"
    echo "          woys-mic             — raw engine output, low latency, no RNNoise [fallback]"
    echo "          (anything tagged '_internal-...' is plumbing — don't pick it)"
    ;;

teardown)
    _unload_chain_modules
    echo "[v0.13.2] RNNoise chain unloaded (woys-mic itself remains unchanged)"
    ;;

status)
    echo "[v0.13.3] chain modules:"
    pactl list short modules 2>/dev/null \
        | grep -E "$SINK_BRIDGE|sink_name=$SINK_FINAL|$LOOPBACK_MARKER|source_name=$SOURCE_USER_FACING" \
        | head || echo "  (chain not loaded)"
    echo
    echo "[v0.13.3] sources visible to apps:"
    pactl list short sources 2>/dev/null | grep -E "woys|rnnoise" | head
    echo
    if [ -f "$SYSTEMD_UNIT_PATH" ]; then
        echo "[v0.13.3] systemd user unit installed at $SYSTEMD_UNIT_PATH:"
        systemctl --user is-enabled "$SYSTEMD_UNIT_NAME" 2>&1 | sed 's/^/  enabled: /'
        systemctl --user is-active  "$SYSTEMD_UNIT_NAME" 2>&1 | sed 's/^/  active:  /'
    else
        echo "[v0.13.3] systemd user unit NOT installed (use 'enable' to auto-load on login)"
    fi
    ;;

enable)
    _require_plugin
    _require_woysmic
    mkdir -p "$SYSTEMD_UNIT_DIR"
    cat > "$SYSTEMD_UNIT_PATH" <<EOF
# v0.13.2 — woys RNNoise chain user unit. Auto-loads the chain on
# login so apps that select 'woys-mic-clean.monitor' get the
# RNNoise-processed audio without manual setup.
#
# Generated by 'woys chain enable' / 'scripts/v013_2_rnnoise_chain.sh enable'.
# Path embedded below points at the script that was used to create
# this unit; if you move the woys repo, re-run 'woys chain disable'
# then 'woys chain enable' from the new location.
[Unit]
Description=woys RNNoise post-engine chain (v0.13.2)
After=pipewire.service pipewire-pulse.service
Requires=pipewire-pulse.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=$SCRIPT_PATH setup
ExecStop=$SCRIPT_PATH teardown

[Install]
WantedBy=default.target
EOF
    systemctl --user daemon-reload
    systemctl --user enable --now "$SYSTEMD_UNIT_NAME"
    echo "[v0.13.2] systemd user unit installed + enabled + started:"
    echo "          $SYSTEMD_UNIT_PATH"
    echo "          chain auto-loads on every login from now on"
    ;;

disable)
    if [ -f "$SYSTEMD_UNIT_PATH" ]; then
        systemctl --user disable --now "$SYSTEMD_UNIT_NAME" 2>&1 | head
        rm -f "$SYSTEMD_UNIT_PATH"
        systemctl --user daemon-reload
        echo "[v0.13.2] systemd user unit disabled + removed: $SYSTEMD_UNIT_PATH"
    else
        echo "[v0.13.2] no systemd user unit installed"
    fi
    _unload_chain_modules
    echo "[v0.13.2] chain unloaded"
    ;;

*)
    echo "Usage: $0 {setup|teardown|status|enable|disable}" >&2
    exit 1
    ;;
esac
