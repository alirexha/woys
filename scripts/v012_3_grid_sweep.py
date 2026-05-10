#!/usr/bin/env python
"""v0.12.3 - comprehensive parameter sweep, intelligent (Phase 1 individual,
Phase 2 cartesian top-2), serial-ID-based recording (the v0.12.2 fix).

Output: ranked table (cuts/min, autocorr at chunk-period, latency cost),
top-3 raw WAVs, best-config recommendation.

Single-process execution (engine + GPU + PipeWire route are shared
single-resource; parallel harnesses would mix audio in WoysSink and
contend for CUDA). Phase 1 ≈ 20 configs, Phase 2 ≈ 25 configs, plus 3
baseline repeats for noise floor → ~50 runs x ~45 s = ~40 min wall.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = Path("/tmp/v012_3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE: dict[str, float] = {
    "chunk_seconds": 0.15,
    "sola_search_ms": 6.0,
    "sola_corr_threshold": 0.10,
    "sola_crossfade_ms": 50.0,
    "sola_context_ms": 100.0,
}

# Phase 1 - sweep each parameter individually with the others at baseline.
PHASE1: dict[str, list[float]] = {
    "chunk_seconds": [0.10, 0.125, 0.15, 0.175, 0.20, 0.25],
    "sola_search_ms": [4.0, 6.0, 8.0, 12.0, 16.0],
    "sola_corr_threshold": [0.05, 0.10, 0.20, 0.30, 0.50],
    "sola_crossfade_ms": [30.0, 50.0, 70.0, 90.0, 120.0],
    "sola_context_ms": [50.0, 100.0, 150.0, 200.0],
}


# v0.11.0 baseline e2e latency: ~540 ms (chunk 150 + inference 80 +
# native-pw 170 + codec 30 ≈ 430 + 110 = 540). Latency penalty for a
# config = (chunk_seconds - 0.15) x 1000 ms. SOLA-context affects the
# input-history buffer and can add a few ms to inference but the
# chunk_seconds is the dominant lever.
def latency_penalty_ms(config: dict[str, float]) -> float:
    return max(0.0, (config["chunk_seconds"] - 0.15) * 1000.0)


def woys_sink_monitor_serial() -> int:
    """pactl serial id of WoysSink.monitor - re-resolved per run because
    a pw teardown/setup cycle can change it."""
    out = subprocess.run(
        ["pactl", "list", "short", "sources"],
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    )
    for line in out.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[1].strip() == "WoysSink.monitor":
            return int(parts[0])
    raise RuntimeError("WoysSink.monitor not present; run `woys pw setup` first")


def run_config(config: dict[str, float], label: str, duration_s: float = 30.0) -> dict[str, Any]:
    """Run engine with `config` for duration_s, capture WoysSink.monitor,
    analyze. Returns dict with metrics + path to wav."""
    wav_path = OUT_DIR / f"{label}.wav"
    json_path = OUT_DIR / f"{label}.json"
    log_path = OUT_DIR / f"{label}.log"

    # Resolve target by serial id (the v0.12.2 anti-fallback fix).
    serial = woys_sink_monitor_serial()

    # Spawn pw-record before harness so we don't miss the first chunks.
    rec_proc = subprocess.Popen(
        [
            "pw-record",
            f"--target={serial}",
            "--rate=48000",
            "--channels=2",
            "--format=f32",
            str(wav_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(0.5)

    cmd = [
        str(REPO / ".venv" / "bin" / "python"),
        str(REPO / "scripts" / "v012_1_tts_run.py"),
        "--duration",
        str(duration_s),
        "--anti-jitter-mode",
        "both",
        "--out",
        str(json_path),
        "--chunk-seconds",
        str(config["chunk_seconds"]),
        "--sola-crossfade-ms",
        str(config["sola_crossfade_ms"]),
        "--sola-search-ms",
        str(config["sola_search_ms"]),
        "--sola-context-ms",
        str(config["sola_context_ms"]),
        "--sola-corr-threshold",
        str(config["sola_corr_threshold"]),
    ]
    with log_path.open("wb") as logf:
        subprocess.run(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            check=True,
            cwd=str(REPO),
            timeout=duration_s + 90,
        )

    time.sleep(1.0)
    rec_proc.send_signal(signal.SIGINT)
    try:
        rec_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        rec_proc.kill()

    return analyze_wav(wav_path, config, label)


def analyze_wav(wav_path: Path, config: dict[str, float], label: str) -> dict[str, Any]:
    """Run woys-diag + spectral flux on wav, parse out cuts/min and
    autocorr peak at chunk_seconds period."""
    audio, sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    duration_s = len(audio) / sr
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))

    # woys-diag (calibrated cut detector).
    cuts_per_min: float = float("nan")
    diag_events: int = -1
    try:
        out = subprocess.run(
            [
                "woys-diag",
                "analyze",
                str(wav_path),
                "--duration",
                "30",
                "--source",
                label,
                "--no-spectrogram",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        # The verdict line OR the events line gives /min.
        # "150 events across 72s (124.3/min)" - pull "/min" pattern.
        m = re.search(r"(\d+)\s+events\s+across\s+\d+s\s*\((\d+\.\d+)/min\)", out.stdout)
        if m:
            diag_events = int(m.group(1))
            cuts_per_min = float(m.group(2))
        else:
            # Could be "Audio is clean" - explicit zero.
            if "Audio is clean" in out.stdout:
                diag_events = 0
                cuts_per_min = 0.0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Spectral autocorrelation at chunk_period.
    autocorr_at_chunk: float = float("nan")
    try:
        chunk_ms = round(config["chunk_seconds"] * 1000)
        out = subprocess.run(
            [
                str(REPO / ".venv" / "bin" / "python"),
                str(REPO / "scripts" / "v012_spectral_flux.py"),
                str(wav_path),
                "--no-plot",
                "--chunk-seconds",
                str(config["chunk_seconds"]),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        # Find the autocorrelation peak nearest chunk_period.
        # Output format includes "lag= 150.0 ms  autocorr=0.123" lines.
        peaks: list[tuple[float, float]] = []
        for line in out.stdout.splitlines():
            m = re.match(r"\s*lag=\s*([\d.]+)\s*ms\s+autocorr=([\d.-]+)", line)
            if m:
                peaks.append((float(m.group(1)), float(m.group(2))))
        # Pick the autocorr value at the lag closest to chunk_ms (within 10 ms).
        if peaks:
            close = [(lag, ac) for (lag, ac) in peaks if abs(lag - chunk_ms) <= 10.0]
            # If no peak near chunk-period in top-10, use 0 (effectively below threshold).
            autocorr_at_chunk = max(ac for _, ac in close) if close else 0.0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return {
        "label": label,
        "config": config,
        "cuts_per_min": cuts_per_min,
        "diag_events": diag_events,
        "autocorr_at_chunk_period": autocorr_at_chunk,
        "duration_s": duration_s,
        "rms": rms,
        "wav_path": str(wav_path),
        "latency_penalty_ms": latency_penalty_ms(config),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=30.0, help="seconds per condition")
    parser.add_argument("--phase", choices=["1", "2", "all"], default="all")
    parser.add_argument(
        "--baseline-repeats", type=int, default=3, help="number of baseline repeats for noise floor"
    )
    args = parser.parse_args()

    # Make sure the PipeWire setup is up.
    subprocess.run(["woys", "pw", "setup"], capture_output=True, timeout=10)
    time.sleep(0.5)

    results: list[dict[str, Any]] = []

    if args.phase in ("1", "all"):
        # Baseline noise-floor probes.
        print(f"\n=== noise-floor baseline ({args.baseline_repeats}x repeats) ===", flush=True)
        for i in range(args.baseline_repeats):
            label = f"baseline_{i}"
            print(f"  [{label}] running...", flush=True)
            try:
                r = run_config(BASELINE, label, args.duration)
                print(
                    f"    cuts/min={r['cuts_per_min']:.1f} autocorr@chunk={r['autocorr_at_chunk_period']:.3f}",
                    flush=True,
                )
                results.append(r)
            except Exception as e:
                print(f"    [error] {type(e).__name__}: {e}", flush=True)

        # Phase 1 - sweep each parameter individually.
        print("\n=== phase 1 - individual sweeps ===", flush=True)
        for param, values in PHASE1.items():
            for v in values:
                if abs(v - BASELINE[param]) < 1e-9:
                    continue  # skip duplicates of baseline
                cfg = dict(BASELINE)
                cfg[param] = v
                label = f"p1_{param}_{v}".replace(".", "p")
                print(f"  [{label}] {param}={v}", flush=True)
                try:
                    r = run_config(cfg, label, args.duration)
                    print(
                        f"    cuts/min={r['cuts_per_min']:.1f} autocorr@chunk={r['autocorr_at_chunk_period']:.3f}",
                        flush=True,
                    )
                    results.append(r)
                except Exception as e:
                    print(f"    [error] {type(e).__name__}: {e}", flush=True)

    # Save Phase 1 results so they survive a Phase 2 crash.
    p1_path = OUT_DIR / "phase1_results.json"
    p1_path.write_text(json.dumps(results, indent=2))
    print(f"\n[saved] {p1_path}", flush=True)

    if args.phase in ("2", "all"):
        # Determine top-2 per parameter from Phase 1.
        # Score: lower cuts_per_min is primary. Tie-break with lower
        # autocorr@chunk. (Latency cost is paid only by chunk_seconds.)
        # If Phase 1 was skipped (--phase 2), load from disk.
        if args.phase == "2":
            saved = (OUT_DIR / "phase1_results.json").read_text()
            results = json.loads(saved)

        # Group results by which-param-is-non-baseline.
        per_param_runs: dict[str, list[dict[str, Any]]] = {p: [] for p in PHASE1}
        for r in results:
            cfg = r["config"]
            differing = [p for p in PHASE1 if abs(cfg[p] - BASELINE[p]) > 1e-9]
            if len(differing) == 1:
                per_param_runs[differing[0]].append(r)
            elif not differing:
                # baseline - include in every group's reference.
                for p in PHASE1:
                    per_param_runs[p].append(r)

        # Pick top-2 values per param by cuts_per_min (lower better).
        top_values: dict[str, list[float]] = {}
        for param, runs in per_param_runs.items():
            sorted_runs = sorted(
                runs,
                key=lambda r: (
                    r["cuts_per_min"] if not np.isnan(r["cuts_per_min"]) else 1e9,
                    r["autocorr_at_chunk_period"]
                    if not np.isnan(r["autocorr_at_chunk_period"])
                    else 1e9,
                ),
            )
            picks: list[float] = []
            seen: set[float] = set()
            for r in sorted_runs:
                v = r["config"][param]
                if v not in seen:
                    picks.append(v)
                    seen.add(v)
                if len(picks) >= 2:
                    break
            top_values[param] = picks
            print(f"\n[top-2] {param} = {picks}", flush=True)

        # Phase 2 - cartesian over top-2 per param.
        from itertools import product as iproduct

        combos = list(iproduct(*[top_values[p] for p in PHASE1]))
        # Dedup against existing single-param + baseline runs.
        already_run: set[tuple[float, ...]] = set()
        for r in results:
            already_run.add(tuple(r["config"][p] for p in PHASE1))
        unique_combos = [c for c in combos if c not in already_run]
        print(f"\n=== phase 2 - {len(unique_combos)} combinations ===", flush=True)
        for combo in unique_combos:
            cfg = dict(zip(PHASE1.keys(), combo, strict=False))
            label = "p2_" + "_".join(
                f"{p[0]}{v}" for p, v in zip(PHASE1.keys(), combo, strict=False)
            ).replace(".", "p")
            print(f"  [{label}] {cfg}", flush=True)
            try:
                r = run_config(cfg, label, args.duration)
                print(
                    f"    cuts/min={r['cuts_per_min']:.1f} autocorr@chunk={r['autocorr_at_chunk_period']:.3f}",
                    flush=True,
                )
                results.append(r)
            except Exception as e:
                print(f"    [error] {type(e).__name__}: {e}", flush=True)

    # Save all results.
    all_path = OUT_DIR / "all_results.json"
    all_path.write_text(json.dumps(results, indent=2))
    print(f"\n[saved] {all_path}", flush=True)

    # Rank by combined score.
    # subjective_score = cuts_per_min + 50 x autocorr@chunk + 0.05 x latency_penalty
    # (rough weighting: 1 cut/min == 0.02 autocorr units == 20 ms latency)
    for r in results:
        c = r["cuts_per_min"] if not np.isnan(r["cuts_per_min"]) else 999.0
        a = r["autocorr_at_chunk_period"] if not np.isnan(r["autocorr_at_chunk_period"]) else 1.0
        lat = r["latency_penalty_ms"]
        r["score"] = c + 50.0 * a + 0.05 * lat

    results.sort(key=lambda r: r["score"])
    print("\n=== TOP 10 (lower score = better) ===", flush=True)
    print(
        f"{'rank':>4}  {'cuts/min':>8}  {'ac@chunk':>8}  {'+lat':>5}  {'score':>6}  {'label'}",
        flush=True,
    )
    for i, r in enumerate(results[:10]):
        print(
            f"  {i + 1:2d}  {r['cuts_per_min']:>8.1f}  {r['autocorr_at_chunk_period']:>8.3f}  "
            f"{r['latency_penalty_ms']:>4.0f}ms  {r['score']:>6.1f}  {r['label']}",
            flush=True,
        )

    # Save top-3 wavs to canonical locations.
    for rank, r in enumerate(results[:3]):
        src = Path(r["wav_path"])
        dst = Path(f"/tmp/v012_3_top{rank + 1}.wav")
        if src.exists():
            shutil.copy(src, dst)
            print(f"\n[top{rank + 1}] copied {src} → {dst}", flush=True)

    # Write summary JSON.
    summary = {
        "baseline_config": BASELINE,
        "n_configs_run": len(results),
        "top_10": [
            {
                "rank": i + 1,
                "label": r["label"],
                "config": r["config"],
                "cuts_per_min": r["cuts_per_min"],
                "autocorr_at_chunk_period": r["autocorr_at_chunk_period"],
                "latency_penalty_ms": r["latency_penalty_ms"],
                "score": r["score"],
            }
            for i, r in enumerate(results[:10])
        ],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[saved] {OUT_DIR / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
