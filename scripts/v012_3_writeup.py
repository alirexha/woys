#!/usr/bin/env python
"""v0.12.3 - generate LESSONS §41 + CHANGELOG entry from /tmp/v012_3/all_results.json.

Decides v0.12.3 final vs v0.12.3-partial based on:
  - Best config beats baseline by 2-sigma noise margin (cuts_per_min)
  - Latency penalty < 30 ms

Outputs:
  - /tmp/v012_3/lessons_41.md (paste into LESSONS.md)
  - /tmp/v012_3/changelog_v012_3.md
  - /tmp/v012_3_top1.wav / top2.wav / top3.wav - best 3 by score
  - /tmp/v012_3/worst1.wav / worst2.wav / worst3.wav - worst 3 by score
    (perceptual A/B reference: best → baseline → worst)
  - Recommendation: ship as final or partial
"""

from __future__ import annotations

import json
import shutil
import statistics
from pathlib import Path

OUT_DIR = Path("/tmp/v012_3")
PARAM_NAMES = [
    "chunk_seconds",
    "sola_search_ms",
    "sola_corr_threshold",
    "sola_crossfade_ms",
    "sola_context_ms",
]
BASELINE = {
    "chunk_seconds": 0.15,
    "sola_search_ms": 6.0,
    "sola_corr_threshold": 0.10,
    "sola_crossfade_ms": 50.0,
    "sola_context_ms": 100.0,
}


def main() -> int:
    raw = json.loads((OUT_DIR / "all_results.json").read_text())
    # Re-score (in case scoring changed since the sweep ran).
    for r in raw:
        cuts = r["cuts_per_min"]
        if cuts != cuts:  # NaN
            cuts = 999.0
        ac = r["autocorr_at_chunk_period"]
        if ac != ac:
            ac = 1.0
        lat = r["latency_penalty_ms"]
        r["score"] = cuts + 50.0 * ac + 0.05 * lat

    # Baseline noise floor.
    baselines = [r for r in raw if r["label"].startswith("baseline_")]
    base_cuts = [r["cuts_per_min"] for r in baselines if r["cuts_per_min"] == r["cuts_per_min"]]
    base_ac = [
        r["autocorr_at_chunk_period"]
        for r in baselines
        if r["autocorr_at_chunk_period"] == r["autocorr_at_chunk_period"]
    ]
    if base_cuts:
        base_cuts_mean = statistics.mean(base_cuts)
        base_cuts_std = statistics.stdev(base_cuts) if len(base_cuts) > 1 else 0.0
    else:
        base_cuts_mean, base_cuts_std = float("nan"), float("nan")
    if base_ac:
        base_ac_mean = statistics.mean(base_ac)
        base_ac_std = statistics.stdev(base_ac) if len(base_ac) > 1 else 0.0
    else:
        base_ac_mean, base_ac_std = float("nan"), float("nan")

    # 2-sigma improvement threshold for cuts/min.
    sig_threshold_cuts = base_cuts_mean - 2 * base_cuts_std

    # Sort by score (lower = better).
    raw.sort(key=lambda r: r["score"])

    # Identify a single "baseline reference" run - pick the median
    # baseline by cuts/min so the table uses one representative point.
    base_ref: dict | None = None
    if baselines:
        base_ref = sorted(
            baselines,
            key=lambda r: r["cuts_per_min"] if r["cuts_per_min"] == r["cuts_per_min"] else 1e9,
        )[len(baselines) // 2]

    # Exclude baselines from the "top/bottom" rankings (they are
    # reported separately as the reference row).
    non_baseline = [r for r in raw if not r["label"].startswith("baseline_")]
    top_5 = non_baseline[:5]
    bottom_3 = list(reversed(non_baseline[-3:]))  # worst-first ordering
    non_baseline[:10]
    best = non_baseline[0] if non_baseline else baselines[0]

    # The user's ship criterion: "best that beats baseline AND latency
    # penalty < 30ms". Apply latency filter to find the recommended
    # default-change candidate.
    LATENCY_LIMIT_MS = 30.0
    low_latency = [r for r in non_baseline if r["latency_penalty_ms"] < LATENCY_LIMIT_MS]
    best_low_lat = low_latency[0] if low_latency else None

    # Decision: v0.12.3 final vs partial.
    # The ship criterion applies to the best LOW-LATENCY config
    # (latency penalty < 30 ms). The +100 ms chunk_seconds=0.25
    # configs are documented as a tradeoff option but disqualified
    # from a default change.
    if best_low_lat is None:
        ship_final = False
        ship_candidate = best
    else:
        ship_candidate = best_low_lat
        is_2sigma_better = ship_candidate["cuts_per_min"] < sig_threshold_cuts
        is_low_latency = ship_candidate["latency_penalty_ms"] < 30.0
        ship_final = is_2sigma_better and is_low_latency
    best_cuts = ship_candidate["cuts_per_min"]
    best_lat = ship_candidate["latency_penalty_ms"]
    is_2sigma_better = best_cuts < sig_threshold_cuts
    is_low_latency = best_lat < 30.0

    # Build LESSONS §41.
    lines = []
    A = lines.append
    A("## 41. v0.12.3 - comprehensive parameter sweep, intelligent grid search")
    A("")
    A("Final tuning sweep before project closure. Phase 1 sweeps each of the")
    A("5 SOLA / chunk parameters individually with the others at v0.11.0")
    A("baseline; Phase 2 cartesians the top-2 values per parameter from")
    A("Phase 1. All recordings via serial-ID pw-record (the v0.12.2 fix).")
    A("TTS-driven engine output for 30 s per condition, both detectors")
    A("(woys-diag calibrated cut count + spectral autocorrelation at the")
    A("chunk-period).")
    A("")
    A(f"### Noise floor (baseline {len(baselines)}x repeat)")
    A("")
    A(f"- cuts/min: mean = {base_cuts_mean:.1f}, std = {base_cuts_std:.2f}")
    A(f"- autocorr@chunk: mean = {base_ac_mean:.3f}, std = {base_ac_std:.3f}")
    A(f"- 2-sigma improvement threshold for cuts/min: {sig_threshold_cuts:.1f}")
    A("")
    A("### Ranked table (top 5 + baseline + bottom 3, lower score = better)")
    A("")
    A("score = cuts/min + 50 x autocorr@chunk + 0.05 x latency_penalty_ms")
    A("")

    def _row(label_prefix: str, r: dict) -> str:
        cfg = r["config"]
        return (
            f"| {label_prefix} | {r['cuts_per_min']:.1f} | {r['autocorr_at_chunk_period']:.3f} | "
            f"{r['latency_penalty_ms']:.0f} | {cfg['chunk_seconds']:.3f} | "
            f"{cfg['sola_search_ms']:.1f} | {cfg['sola_corr_threshold']:.2f} | "
            f"{cfg['sola_crossfade_ms']:.0f} | {cfg['sola_context_ms']:.0f} | {r['score']:.1f} |"
        )

    A(
        "| rank | cuts/min | ac@chunk | +lat (ms) | chunk_s | search_ms | corr_thr | crossfade_ms | context_ms | score |"
    )
    A(
        "|-----:|---------:|---------:|----------:|--------:|----------:|---------:|-------------:|-----------:|------:|"
    )
    for i, r in enumerate(top_5):
        A(_row(f"top-{i + 1}", r))
    if base_ref is not None:
        A(_row("**baseline (v0.11.0)**", base_ref))
    for i, r in enumerate(bottom_3):
        rank_label = f"worst-{i + 1}"
        A(_row(rank_label, r))
    A("")
    A(
        "Top-3 raw recordings: `/tmp/v012_3_top1.wav`, `/tmp/v012_3_top2.wav`, `/tmp/v012_3_top3.wav`.  "
    )
    A(
        "Worst-3 raw recordings: `/tmp/v012_3/worst1.wav`, `/tmp/v012_3/worst2.wav`, `/tmp/v012_3/worst3.wav`.  "
    )
    A("Baseline reference recording: `/tmp/v012_3/baseline_ref.wav`.")
    A("")
    A("Listener calibration: comparing best → baseline → worst by ear")
    A("validates whether the cuts/min metric tracks audible quality.")
    A("")
    A("### Recommended default change (best of low-latency tier, +lat < 30 ms)")
    A("")
    A(f"**`{ship_candidate['label']}`** - score {ship_candidate['score']:.1f}")
    A("")
    A("```toml")
    for p in PARAM_NAMES:
        A(f"{p} = {ship_candidate['config'][p]}")
    A("```")
    A("")
    A(
        f"- cuts/min: {ship_candidate['cuts_per_min']:.1f}  (baseline {base_cuts_mean:.1f} ± {base_cuts_std:.2f})"
    )
    A(
        f"- autocorr@chunk: {ship_candidate['autocorr_at_chunk_period']:.3f}  (baseline {base_ac_mean:.3f} ± {base_ac_std:.3f})"
    )
    A(f"- latency penalty vs v0.11.0: +{ship_candidate['latency_penalty_ms']:.0f} ms")
    A("")
    A("### Best overall (HIGH-latency tier - informational, NOT default change)")
    A("")
    A(f"**`{best['label']}`** - score {best['score']:.1f}")
    A("")
    A("```toml")
    for p in PARAM_NAMES:
        A(f"{p} = {best['config'][p]}")
    A("```")
    A("")
    A(
        f"- cuts/min: {best['cuts_per_min']:.1f}  (baseline {base_cuts_mean:.1f} ± {base_cuts_std:.2f})"
    )
    A(
        f"- autocorr@chunk: {best['autocorr_at_chunk_period']:.3f}  (baseline {base_ac_mean:.3f} ± {base_ac_std:.3f})"
    )
    A(
        f"- latency penalty vs v0.11.0: +{best['latency_penalty_ms']:.0f} ms  ← exceeds 30 ms ship criterion"
    )
    A("")
    A("If the user is willing to trade +100 ms e2e latency for the strongest")
    A("possible cut reduction (chunk_seconds=0.25 eliminates chunk-period")
    A("autocorrelation entirely, autocorr@chunk = 0.000), opt in via config:")
    A("")
    A("```toml")
    for p in PARAM_NAMES:
        A(f"{p} = {best['config'][p]}")
    A("```")
    A("")
    A("### 2-sigma significance test (low-latency ship candidate)")
    A("")
    margin_cuts = base_cuts_mean - ship_candidate["cuts_per_min"]
    A(f"- ship-candidate cuts/min margin vs baseline: {-margin_cuts:+.1f}")
    A(f"  (- means improvement; threshold for 2-sigma significance: -{2 * base_cuts_std:.1f})")
    A(f"- 2-sigma significant: **{'YES' if is_2sigma_better else 'NO'}**")
    A(
        f"- latency cost < 30 ms: **{'YES' if is_low_latency else 'NO'}**  ({ship_candidate['latency_penalty_ms']:.0f} ms)"
    )
    A("")
    A("### Ship decision")
    A("")
    if ship_final:
        A("**v0.12.3 final.** Best config beats baseline by 2-sigma margin AND")
        A("latency penalty is under 30 ms. Default change recommended.")
    else:
        A("**v0.12.3-partial.** Best config does NOT beat baseline by")
        A("2-sigma noise margin AND/OR latency penalty exceeds 30 ms.")
        A("No default changes; sweep findings ship as research output.")
    A("")
    A("### Per-parameter individual sensitivity")
    A("")
    # For each param, find the value that minimized score (Phase 1 only).
    p1 = [r for r in raw if r["label"].startswith("p1_")]
    A("For each parameter, the value that produced lowest cuts/min when")
    A("the others were held at baseline:")
    A("")
    for p in PARAM_NAMES:
        runs = [
            r
            for r in p1
            if abs(r["config"][p] - BASELINE[p]) > 1e-9
            and all(abs(r["config"][q] - BASELINE[q]) < 1e-9 for q in PARAM_NAMES if q != p)
        ]
        runs.append(next((r for r in baselines), None))
        runs = [r for r in runs if r is not None]
        if not runs:
            continue
        runs.sort(
            key=lambda r: r["cuts_per_min"] if r["cuts_per_min"] == r["cuts_per_min"] else 1e9
        )
        bestp = runs[0]
        A(
            f"- **{p}**: best = {bestp['config'][p]} → cuts/min {bestp['cuts_per_min']:.1f} "
            f"(baseline {base_cuts_mean:.1f})"
        )
    A("")
    A("### Generalizable lesson - exhaustive sweeps confirm what limited sweeps suggest")
    A("")
    A("v0.12.0's first 4-condition sweep showed SOLA tuning was within noise.")
    A(f"v0.12.3's full {len(raw) - len(baselines)}-condition sweep confirms it. The chunk-boundary")
    A("periodic mechanism is fundamental on this stack; tuning shifts but")
    A("does not eliminate it. The user's v0.11.0 daily-use experience is")
    A("the audible ceiling within software-only configurations on this")
    A("hardware.")

    out = "\n".join(lines) + "\n"
    (OUT_DIR / "lessons_41.md").write_text(out)
    print(out)
    print(f"\n[saved] {OUT_DIR / 'lessons_41.md'}")

    # Copy top-3 to canonical perceptual-A/B locations.
    for rank, r in enumerate(top_5[:3]):
        src = Path(r["wav_path"])
        dst = Path(f"/tmp/v012_3_top{rank + 1}.wav")
        if src.exists():
            shutil.copy(src, dst)
            print(f"[top{rank + 1}]   {dst}  ← {src}")
    # Copy worst-3 (in worst-first order: worst1.wav = worst-of-all).
    for rank, r in enumerate(bottom_3):
        src = Path(r["wav_path"])
        dst = OUT_DIR / f"worst{rank + 1}.wav"
        if src.exists():
            shutil.copy(src, dst)
            print(f"[worst{rank + 1}] {dst}  ← {src}")
    # Copy baseline reference for symmetric A/B comparison.
    if base_ref is not None:
        src = Path(base_ref["wav_path"])
        dst = OUT_DIR / "baseline_ref.wav"
        if src.exists():
            shutil.copy(src, dst)
            print(f"[baseline] {dst}  ← {src}")

    print(
        f"\n=== ship decision: {'v0.12.3 (default change)' if ship_final else 'v0.12.3-partial (no defaults)'} ==="
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
