#!/usr/bin/env python
"""v0.10.x harness post-run analysis.

Ingests one or more `v010_harness.py --out` JSON files and prints a
side-by-side table of the load-bearing percentiles. Optionally
correlates with an `nvidia-smi --format=csv -lms 100` clock log.

Usage:
  ./scripts/v010_analyze.py /tmp/v010_baseline.json
  ./scripts/v010_analyze.py /tmp/v010_baseline.json /tmp/v010_rc2.json
  ./scripts/v010_analyze.py /tmp/v010_baseline.json --gpu-csv /tmp/v010_gpu_baseline.csv

The two-file form is the rc-comparison surface: a fix landed in rc2
should move writer_jitter_p99 toward the acceptance gate (≤ 30 ms).
The GPU correlation reads clocks.gr and overlays p99 / max windows
to flag thermal / power-state contributions to the tail.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


GATE_WRITER_JITTER_P99 = 30.0  # ms
GATE_INFERENCE_AVG = 52.0  # ms (v0.9.0 baseline; brief target)
GATE_UNDERRUN_RATE = 0.5  # /sec


# Column width for the percentile table.
W = 12


def _format_row(label: str, values: list[Any], fmts: list[str]) -> str:
    pieces = [label.ljust(28)]
    for v, fmt in zip(values, fmts, strict=False):
        if isinstance(v, float):
            pieces.append(format(v, fmt).rjust(W))
        elif v is None:
            pieces.append("-".rjust(W))
        else:
            pieces.append(str(v).rjust(W))
    return "  ".join(pieces)


def _extract(stats: dict[str, Any], key: str, default: Any = None) -> Any:
    return stats.get(key, default)


def _percentile_pure(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    rank = (p / 100.0) * (len(s) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    frac = rank - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _gate_check(stats: dict[str, Any], duration_s: float) -> dict[str, tuple[bool, str]]:
    rate = _extract(stats, "player_underruns", 0) / max(duration_s, 1)
    inf_avg = _extract(stats, "inference_avg_ms", 0.0)
    wj_p99 = _extract(stats, "writer_jitter_p99_ms", float("inf"))
    return {
        "underrun_rate": (
            rate <= GATE_UNDERRUN_RATE,
            f"{rate:.2f} /sec  (gate ≤ {GATE_UNDERRUN_RATE})",
        ),
        "writer_jitter_p99": (
            wj_p99 <= GATE_WRITER_JITTER_P99,
            f"{wj_p99:.1f} ms  (gate ≤ {GATE_WRITER_JITTER_P99})",
        ),
        "inference_avg": (
            inf_avg <= GATE_INFERENCE_AVG,
            f"{inf_avg:.1f} ms  (gate ≤ {GATE_INFERENCE_AVG})",
        ),
    }


def _summarize_gpu_csv(path: Path) -> dict[str, float]:
    """Parse nvidia-smi clocks CSV.

    Format from `--query-gpu=timestamp,clocks.gr,clocks.mem,power.draw,
    utilization.gpu,temperature.gpu --format=csv,nounits`.
    Returns p50/p95/p99/max for clocks.gr (graphics) MHz.
    """
    if not path.exists():
        return {}
    clocks_gr: list[float] = []
    powers: list[float] = []
    utils: list[float] = []
    temps: list[float] = []
    with path.open() as f:
        reader = csv.reader(f)
        # Skip header (first non-empty line).
        rows = list(reader)
    if not rows:
        return {}
    # Detect header by checking if first row looks like text.
    start = 0
    if rows[0] and not rows[0][0].replace(":", "").replace(" ", "").replace(".", "").isdigit():
        start = 1
    for row in rows[start:]:
        try:
            # Columns: timestamp, clocks.gr, clocks.mem, power.draw,
            # utilization.gpu, temperature.gpu
            if len(row) < 6:
                continue
            clocks_gr.append(float(row[1]))
            powers.append(float(row[3]))
            utils.append(float(row[4]))
            temps.append(float(row[5]))
        except (ValueError, IndexError):
            continue
    if not clocks_gr:
        return {}

    def _pct(v: list[float], p: float) -> float:
        return _percentile_pure(v, p)

    return {
        "n_samples": float(len(clocks_gr)),
        "clocks_gr_min_mhz": min(clocks_gr),
        "clocks_gr_p50_mhz": _pct(clocks_gr, 50),
        "clocks_gr_p95_mhz": _pct(clocks_gr, 95),
        "clocks_gr_max_mhz": max(clocks_gr),
        "power_draw_p50_w": _pct(powers, 50),
        "power_draw_p99_w": _pct(powers, 99),
        "utilization_p50_pct": _pct(utils, 50),
        "utilization_p99_pct": _pct(utils, 99),
        "temp_max_c": max(temps),
        "temp_p99_c": _pct(temps, 99),
        "clock_dip_count": float(sum(1 for c in clocks_gr if c < _pct(clocks_gr, 50) - 100.0)),
    }


def _load_run(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="harness JSON paths")
    parser.add_argument("--gpu-csv", type=Path, default=None)
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="per-run column label (one per run; defaults to filename)",
    )
    args = parser.parse_args()

    if args.label is not None and len(args.label) != len(args.runs):
        print(
            f"[error] --label count ({len(args.label)}) must match runs count ({len(args.runs)})",
        )
        return 2

    runs = [_load_run(p) for p in args.runs]
    labels = args.label or [p.stem for p in args.runs]

    print()
    print("==== v0.10.x harness comparison ====")
    print()
    header = "metric".ljust(28) + "  " + "  ".join(label.rjust(W) for label in labels)
    print(header)
    print("-" * len(header))

    def _row(label: str, key: str, fmt: str = "8.2f") -> None:
        values = [_extract(r["stats"], key) for r in runs]
        fmts = [fmt] * len(runs)
        print(_format_row(label, values, fmts))

    print(_format_row("duration (s)", [r["duration_s"] for r in runs], ["8.0f"] * len(runs)))
    print(_format_row("chunk_seconds", [r["chunk_seconds"] for r in runs], ["8.3f"] * len(runs)))
    _row("chunks_processed", "chunks_processed", "8.0f")
    print()
    print("---- player ----")
    _row("player_underruns", "player_underruns", "8.0f")
    print(
        _format_row(
            "underrun_rate (/s)",
            [r["stats"]["player_underruns"] / max(r["duration_s"], 1) for r in runs],
            ["8.2f"] * len(runs),
        )
    )
    _row("player_restarts", "player_restarts", "8.0f")
    _row("late_chunks", "late_chunks", "8.0f")
    _row("dropped_chunks", "dropped_chunks", "8.0f")
    print()
    print("---- writer interval (ms) ----")
    _row("writer_interval p50", "writer_interval_p50_ms")
    _row("writer_interval p95", "writer_interval_p95_ms")
    _row("writer_interval p99", "writer_interval_p99_ms")
    _row("writer_interval max", "writer_interval_max_ms")
    _row("writer_jitter σ", "writer_jitter_ms_stddev")
    _row("**writer_jitter p99**", "writer_jitter_p99_ms")
    print()
    print("---- inference (ms) ----")
    _row("inference p50", "inference_p50_ms")
    _row("inference p95", "inference_p95_ms")
    _row("inference p99", "inference_p99_ms")
    _row("inference max", "inference_max_ms")
    _row("inference avg", "inference_avg_ms")
    print()
    print("---- per-stage (ms) ----")
    _row(".cv     p50", "cv_p50_ms")
    _row(".cv     p99", "cv_p99_ms")
    _row(".rmvpe  p50", "rmvpe_p50_ms")
    _row(".rmvpe  p99", "rmvpe_p99_ms")
    _row(".rvc    p50", "rvc_p50_ms")
    _row(".rvc    p99", "rvc_p99_ms")
    _row("  .rvc_pre  p50", "rvc_pre_p50_ms")
    _row("  .rvc_pre  p99", "rvc_pre_p99_ms")
    _row("  .rvc_run  p50", "rvc_run_p50_ms")
    _row("  .rvc_run  p99", "rvc_run_p99_ms")
    _row("  .rvc_post p50", "rvc_post_p50_ms")
    _row("  .rvc_post p99", "rvc_post_p99_ms")
    print()
    print("---- pre/post stages (ms) ----")
    _row("mic_read p50", "mic_read_p50_ms")
    _row("mic_read p99", "mic_read_p99_ms")
    _row("enqueue_lag p50", "enqueue_lag_p50_ms")
    _row("enqueue_lag p99", "enqueue_lag_p99_ms")
    print()
    print("---- gpu keep-alive ----")
    _row("keepalive_calls", "keepalive_calls", "8.0f")
    _row("keepalive_p50 ms", "keepalive_p50_ms")
    _row("keepalive_p99 ms", "keepalive_p99_ms")
    print()
    print("---- shapes ----")
    for label, r in zip(labels, runs, strict=False):
        s = r["stats"]
        warm = s.get("warmup_audio16_lens", [])
        runtime = s.get("runtime_audio16_lens", [])
        unwarmed = s.get("unwarmed_shapes", [])
        print(f"  [{label}] warmup={warm}")
        print(f"  [{label}] runtime={runtime}")
        if unwarmed:
            print(f"  [{label}] [!] unwarmed_shapes={unwarmed}")
    print()
    print("==== acceptance gates ====")
    for label, r in zip(labels, runs, strict=False):
        gates = _gate_check(r["stats"], r["duration_s"])
        status = "PASS" if all(g[0] for g in gates.values()) else "FAIL"
        print(f"  [{label}] overall: {status}")
        for gate_name, (passed, detail) in gates.items():
            mark = "✓" if passed else "✗"
            print(f"     {mark} {gate_name:25s} {detail}")
    print()

    if args.gpu_csv is not None:
        print("==== GPU clocks (concurrent with run) ====")
        gpu = _summarize_gpu_csv(args.gpu_csv)
        if not gpu:
            print(f"  [error] could not parse {args.gpu_csv}")
        else:
            for k, v in gpu.items():
                print(f"  {k:30s} {v:.1f}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
