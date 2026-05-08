#!/bin/bash
# v0.12.0 Phase 2 — 4-condition 5-minute A/B sweep.
#
# Conditions (each runs with mode=both, the v0.11.0 winner):
#   1. p2_baseline    chunk=0.15, SOLA defaults (crossfade=50, search=6, context=100, corr=0.10)
#   2. p2_chunk020    chunk=0.20, SOLA defaults
#   3. p2_sola        chunk=0.15, SOLA tuned (crossfade=80, search=12, context=150, corr=0.30)
#   4. p2_both        chunk=0.20, SOLA tuned
#
# 5 min × 4 = ~20 min wall + warmup overhead.
# Captures JSON metrics + WoysSink.monitor recording per condition.
set -euo pipefail
cd /home/alireza/ai/woys
woys pw setup >/dev/null 2>&1 || true

run() {
    local label="$1"; shift
    local out="/tmp/v012_${label}"
    echo
    echo "==== ${label} : $* ===="
    pw-record --target=WoysSink.monitor --rate=48000 --channels=2 --format=f32 "${out}.wav" >/dev/null 2>&1 &
    local rec=$!
    sleep 0.5
    .venv/bin/python scripts/v010_harness.py \
        --duration 300 --anti-jitter-mode both --out "${out}.json" "$@" 2>&1 \
        | tee "${out}.log" \
        | grep -E '^\[harness\] t=|writer_jitter |inference  |.rvc  |underrun rate|late_chunks' \
        | tail -25
    sleep 0.5
    kill -INT "${rec}" 2>/dev/null || true
    wait "${rec}" 2>/dev/null || true
    echo "  saved ${out}.json + ${out}.wav"
}

run p2_baseline
run p2_chunk020 --chunk-seconds 0.20
run p2_sola     --sola-crossfade-ms 80 --sola-search-ms 12 --sola-context-ms 150 --sola-corr-threshold 0.30
run p2_both     --chunk-seconds 0.20 --sola-crossfade-ms 80 --sola-search-ms 12 --sola-context-ms 150 --sola-corr-threshold 0.30

echo
echo "==== sweep complete ===="
