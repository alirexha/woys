#!/bin/bash
# v0.12.2 — re-run Phase 2 sweep with PROPER recording methodology
# (explicit pactl serial IDs to avoid pw-record name-fallback to default source).
#
# Same 4 conditions as v0.12.0's Phase 2 (baseline / chunk020 / sola_tuned / both),
# all under mode=both. Each runs ~30 s with TTS-driven input + concurrent
# recording from WoysSink.monitor by SERIAL ID (not name). Then runs both
# detectors on each recording.
set -euo pipefail
cd /home/alireza/ai/woys
woys pw setup >/dev/null 2>&1 || true
sleep 0.3

# Re-resolve serial IDs (they change after teardown/setup).
WOYSSINK_SERIAL=$(pactl list short sources | awk '/WoysSink.monitor/ {print $1}')
echo "[v012.2-sweep] WoysSink.monitor serial = $WOYSSINK_SERIAL"

run() {
    local label="$1"; shift
    local out="/tmp/v012_1/sweep2_${label}"
    echo
    echo "==== ${label} : $* ===="
    pw-record --target=$WOYSSINK_SERIAL --rate=48000 --channels=2 --format=f32 "${out}.wav" >/dev/null 2>&1 &
    local rec=$!
    sleep 0.5
    .venv/bin/python scripts/v012_1_tts_run.py \
        --duration 30 --anti-jitter-mode both \
        --out "${out}.json" "$@" 2>&1 | tail -3
    sleep 1
    kill -INT "${rec}" 2>/dev/null || true
    wait "${rec}" 2>/dev/null || true
    echo "  [woys-diag]"
    woys-diag analyze "${out}.wav" --duration 30 --source "${label}" 2>&1 | grep -E "Verdict|events" | head -2
    echo "  [autocorr at 150ms]"
    .venv/bin/python scripts/v012_spectral_flux.py "${out}.wav" --no-plot 2>&1 | grep -E "lag=\s*1[345]0\.0 ms|fraction.*chunk_seconds:" | head -3
}

run baseline
run chunk020 --chunk-seconds 0.20
run sola_tuned --sola-crossfade-ms 80 --sola-search-ms 12 --sola-context-ms 150 --sola-corr-threshold 0.30
run both --chunk-seconds 0.20 --sola-crossfade-ms 80 --sola-search-ms 12 --sola-context-ms 150 --sola-corr-threshold 0.30

# v0.12.0's chunk_seconds-0.20-aware variant: the autocorr peak should be at 200ms instead of 150ms
# if chunk-boundary periodic mechanism is confirmed.
echo
echo "==== summary ===="
for label in baseline chunk020 sola_tuned both; do
    out="/tmp/v012_1/sweep2_${label}"
    autocorr=$(.venv/bin/python scripts/v012_spectral_flux.py "${out}.wav" --no-plot --chunk-seconds 0.20 2>&1 | grep -E "lag=\s*150\.0 ms|lag=\s*200\.0 ms" | head -1)
    woysdiag=$(grep -E "events" /tmp/v012_1/sweep2_${label}*.md 2>/dev/null | head -1 || true)
    echo "  ${label}: ${autocorr} | ${woysdiag}"
done
