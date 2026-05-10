"""Benchmark the realtime inference path at engine-realistic chunk sizes.

Uses the actual `audio.engine.RVCEngine` so per-stage timings (cv / rmvpe /
rvc) match what the realtime loop produces. Synthetic input - no audio I/O,
no pacat, no SOLA. The number we care about is `_infer()` cost as a
function of chunk_seconds + sola_context_ms.

Run:
    .venv/bin/python scripts/bench_inference.py [--chunk 0.25] [--passes 60]
        [--voice amitaro_v2_16k] [--warm 8]

Output: per-stage avg / p50 / p95 / p99 / max + total wall-clock.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import onnxruntime as ort

if hasattr(ort, "preload_dlls"):
    ort.preload_dlls()

from audio.engine import MODELS_DIR, EngineConfig, RealtimeEngine


def percentiles(samples: list[float], ps: tuple[float, ...]) -> list[float]:
    arr = np.array(samples, dtype=np.float64)
    return [float(np.percentile(arr, p)) for p in ps]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=float, default=0.25, help="chunk_seconds")
    ap.add_argument("--passes", type=int, default=60, help="timed passes after warmup")
    ap.add_argument("--warm", type=int, default=8, help="warmup passes (discarded)")
    ap.add_argument("--voice", type=str, default="amitaro_v2_16k", help="model slug (no .onnx)")
    ap.add_argument("--ctx-ms", type=float, default=100.0, help="sola_context_ms")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rvc_path = MODELS_DIR / f"{args.voice}.onnx"
    if not rvc_path.exists():
        print(f"ERROR: model not found: {rvc_path}", file=sys.stderr)
        sys.exit(2)

    cfg = EngineConfig(
        rvc_model=rvc_path,
        chunk_seconds=args.chunk,
        sola_enabled=False,
        sola_context_ms=args.ctx_ms,
    )
    engine = RealtimeEngine(cfg)
    engine._ensure_sessions()

    rng = np.random.default_rng(args.seed)
    n_samples_16k = int((args.chunk + args.ctx_ms / 1000.0) * 16_000)
    audio = (rng.standard_normal(n_samples_16k).astype(np.float32) * 0.05).clip(-1.0, 1.0)

    print(
        f"# bench_inference  voice={args.voice}  chunk={args.chunk}s  ctx={args.ctx_ms}ms  "
        f"n_samples_16k={n_samples_16k}  cuda={'CUDAExecutionProvider' in engine._rvc.get_providers()}"
    )

    for _ in range(args.warm):
        _ = engine._infer(audio)

    cv_ms: list[float] = []
    rmvpe_ms: list[float] = []
    rvc_ms: list[float] = []
    total_ms: list[float] = []

    for _ in range(args.passes):
        t0 = time.perf_counter()
        _ = engine._infer(audio)
        t1 = time.perf_counter()
        cv_ms.append(engine.stats.last_cv_ms)
        rmvpe_ms.append(engine.stats.last_rmvpe_ms)
        rvc_ms.append(engine.stats.last_rvc_ms)
        total_ms.append((t1 - t0) * 1000.0)

    def fmt(name: str, vals: list[float]) -> str:
        p50, p95, p99 = percentiles(vals, (50.0, 95.0, 99.0))
        return (
            f"  {name:<10}  avg={np.mean(vals):6.2f}  p50={p50:6.2f}  "
            f"p95={p95:6.2f}  p99={p99:6.2f}  max={max(vals):6.2f}  ms"
        )

    print(fmt("cv", cv_ms))
    print(fmt("rmvpe", rmvpe_ms))
    print(fmt("rvc", rvc_ms))
    print(fmt("TOTAL", total_ms))


if __name__ == "__main__":
    main()
