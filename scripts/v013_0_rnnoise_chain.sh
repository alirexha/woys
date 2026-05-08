#!/bin/bash
# v0.13.0 — opt-in RNNoise chain after woys-mic.
#
# Builds a parallel virtual source `woys-mic-clean` that takes the
# v0.12.4 `woys-mic` output, runs it through RNNoise (the rnnoise
# LADSPA plugin from `noise-suppression-for-voice`), and exposes
# the denoised audio as a new source apps can select. The original
# `woys-mic` stays available unchanged.
#
# Measured impact (TTS-driven engine, v0.12.4 defaults, 60 s):
#   woys-mic           : 86.5 cuts/min (woys-diag)
#   woys-mic-clean     : 75.2 cuts/min  (-13 %)
#   spectral autocorr peak at 150 ms: 0.111 → 0.079 (-29 %)
#
# Latency cost: +30 ms loopback + ~10 ms RNNoise frame ≈ +40 ms total
# on top of v0.12.4's ~640 ms. Apps consuming from `woys-mic-clean`
# experience ~680 ms total e2e.
#
# Requires: pacman -S noise-suppression-for-voice
#
# Usage:
#   ./scripts/v013_0_rnnoise_chain.sh setup     # load the chain
#   ./scripts/v013_0_rnnoise_chain.sh teardown  # unload everything
#   ./scripts/v013_0_rnnoise_chain.sh status    # show current state
set -euo pipefail

PLUGIN_PATH="/usr/lib/ladspa/librnnoise_ladspa.so"
PLUGIN_LABEL="noise_suppressor_mono"
SINK_FINAL="woys-mic-clean"
SINK_BRIDGE="woys-mic-rnnoise-bridge"

cmd="${1:-status}"

case "$cmd" in
setup)
    if [ ! -f "$PLUGIN_PATH" ]; then
        echo "[v0.13.0] $PLUGIN_PATH not found — install with:" >&2
        echo "          sudo pacman -S noise-suppression-for-voice" >&2
        exit 1
    fi
    if ! pactl list short sources 2>/dev/null | grep -qE "^[0-9]+\s+woys-mic\s"; then
        echo "[v0.13.0] woys-mic source not present — run 'woys pw setup' first" >&2
        exit 1
    fi

    # Idempotent: tear down any existing v0.13.0 chain before re-loading.
    "$0" teardown >/dev/null 2>&1 || true

    # Step 1: terminal sink that holds the denoised audio.
    pactl load-module module-null-sink \
        media.class=Audio/Source/Virtual \
        sink_name="$SINK_FINAL" \
        sink_properties=device.description=woys-mic-clean_rnnoise \
        rate=48000 channels=1 >/dev/null

    # Step 2: LADSPA sink that applies the rnnoise plugin; output goes to
    # its sink_master ($SINK_FINAL).
    pactl load-module module-ladspa-sink \
        sink_name="$SINK_BRIDGE" \
        sink_master="$SINK_FINAL" \
        plugin="$PLUGIN_PATH" \
        label="$PLUGIN_LABEL" >/dev/null

    # Step 3: Loopback that feeds woys-mic into the LADSPA bridge.
    pactl load-module module-loopback \
        source=woys-mic sink="$SINK_BRIDGE" \
        rate=48000 channels=1 latency_msec=30 >/dev/null

    echo "[v0.13.0] RNNoise chain active. Apps can now select:"
    echo "          woys-mic         (raw v0.12.4 engine output, 86.5 cuts/min)"
    echo "          woys-mic-clean   (RNNoise-processed, 75.2 cuts/min, +40 ms latency)"
    ;;

teardown)
    for label in "loopback.*sink=$SINK_BRIDGE" \
                 "module-ladspa-sink.*sink_name=$SINK_BRIDGE" \
                 "module-null-sink.*sink_name=$SINK_FINAL"; do
        for mod_id in $(pactl list short modules | awk -v p="$label" '$0 ~ p {print $1}'); do
            pactl unload-module "$mod_id" 2>/dev/null || true
        done
    done
    echo "[v0.13.0] RNNoise chain unloaded (woys-mic remains unchanged)"
    ;;

status)
    echo "[v0.13.0] RNNoise chain modules:"
    pactl list short modules | grep -E "$SINK_BRIDGE|$SINK_FINAL|loopback.*$SINK_BRIDGE" | head
    echo "[v0.13.0] woys-mic-clean source visibility (apps see this):"
    pactl list short sources | grep -E "$SINK_FINAL|woys-mic" | head
    ;;

*)
    echo "Usage: $0 {setup|teardown|status}" >&2
    exit 1
    ;;
esac
