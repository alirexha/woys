#!/bin/bash
# v0.12.0 Phase 1 — drive the synthetic harness with concurrent
# pw-record from WoysSink.monitor so we capture exactly what the
# native-pw helper writes. No human-in-the-loop.
#
# Usage: ./scripts/v012_run_and_record.sh [duration_s] [out_prefix]
set -euo pipefail
DURATION="${1:-30}"
OUT_PREFIX="${2:-/tmp/v012_synthetic}"
WAV="${OUT_PREFIX}.wav"
JSON="${OUT_PREFIX}.json"
LOG="${OUT_PREFIX}.log"
cd /home/alireza/ai/woys

# Make sure the WoysSink null-sink + woys-mic source exist (idempotent).
woys pw setup >/dev/null 2>&1 || true

# v0.12.2 — resolve WoysSink.monitor by SERIAL ID, not name. pw-record's
# name-based --target silently falls back to the default source if the
# named target isn't immediately recordable, which gave false-clean
# "WoysSink.monitor is silent" verdicts in v0.12.0/v0.12.1. Always use
# the numeric serial reported by `pactl list short sources` to pin the
# capture to the intended node.
WOYSSINK_SERIAL=$(pactl list short sources | awk '/WoysSink.monitor/ {print $1}')
if [ -z "${WOYSSINK_SERIAL:-}" ]; then
    echo "[record] WoysSink.monitor not present — `woys pw setup` first" >&2
    exit 1
fi
echo "[record] starting pw-record on WoysSink.monitor (serial ${WOYSSINK_SERIAL}) → ${WAV}"
pw-record --target=${WOYSSINK_SERIAL} --rate=48000 --channels=2 --format=f32 "${WAV}" &
RECPID=$!
# Give pw-record time to register with the graph before the engine starts streaming.
sleep 0.5

echo "[harness] starting v010_harness for ${DURATION}s with anti-jitter=both"
.venv/bin/python scripts/v010_harness.py \
    --duration "${DURATION}" \
    --anti-jitter-mode both \
    --out "${JSON}" 2>&1 | tee "${LOG}"

# Give pw-record a moment to flush the last chunks.
sleep 0.5
kill -INT "${RECPID}" 2>/dev/null || true
wait "${RECPID}" 2>/dev/null || true

echo "[record] saved: ${WAV} ($(stat -c '%s' "${WAV}") bytes)"
echo "[record] saved: ${JSON}"
