"""Standalone runtime stability sweep for v0.7.0 latency tuning.

Spins up the full RealtimeEngine (synthetic mic input, real pacat output,
real PipeWire null-sink) and reports xruns + queue_full + jitter +
inference distribution at a given (chunk_seconds, output_latency_ms)
pair. The existing slow test in `tests/test_pacat_health.py` is a single
hardcoded config; this lets us sweep the space.

Run:
    .venv/bin/python scripts/bench_engine_runtime.py \\
        --chunk 0.10 --latency 100 --voice catwoman --duration 30 --warm 3
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import sounddevice as sd

from audio.engine import MODELS_DIR, EngineConfig, RealtimeEngine


class _SyntheticInputStream:
    """Drop-in for `sd.InputStream` that yields paced 50 ms windows of
    band-limited noise. Same shape as the v0.5.2 slow-test patch.
    """

    def __init__(self, samplerate: int, channels: int, blocksize: int, **_kw: object) -> None:
        self.samplerate = samplerate
        self.channels = channels
        self.blocksize = blocksize
        self.rng = np.random.default_rng(2026)

    def __enter__(self) -> _SyntheticInputStream:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def close(self) -> None:
        return None

    def read(self, frames: int) -> tuple[np.ndarray, bool]:
        # Block for the equivalent wall-clock interval to mimic real mic timing.
        time.sleep(frames / self.samplerate)
        block = self.rng.standard_normal((frames, self.channels)).astype(np.float32) * 0.05
        return block.clip(-1.0, 1.0), False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=float, default=0.10)
    ap.add_argument("--latency", type=int, default=100)
    ap.add_argument("--voice", type=str, default="catwoman")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--warm", type=float, default=3.0)
    args = ap.parse_args()

    rvc_path = MODELS_DIR / f"{args.voice}.onnx"
    if not rvc_path.exists():
        print(f"ERROR: model not found: {rvc_path}", file=sys.stderr)
        sys.exit(2)

    sd.InputStream = _SyntheticInputStream  # type: ignore[assignment,misc]

    cfg = EngineConfig(
        rvc_model=rvc_path,
        chunk_seconds=args.chunk,
        output_latency_ms=args.latency,
    )
    eng = RealtimeEngine(cfg)
    eng.start()
    try:
        time.sleep(args.warm)
        # Reset health counters after warmup.
        eng.stats.xruns = 0
        eng.stats.queue_full_events = 0
        eng.stats.pacat_restarts = 0
        eng.stats._recent_inference.clear()
        eng.stats._recent_total.clear()
        eng.stats._writer_intervals_ms.clear()
        eng.stats.max_inference_ms = 0.0
        eng.stats.max_total_ms = 0.0
        eng.stats.late_chunks = 0

        time.sleep(args.duration)

        s = eng.stats
        intervals = list(s._writer_intervals_ms)
        infs = list(s._recent_inference)
        tots = list(s._recent_total)

        def stats(name: str, vals: list[float]) -> str:
            if not vals:
                return f"{name}: (no data)"
            arr = np.array(vals)
            return (
                f"  {name:<10} avg={arr.mean():6.2f}  p50={np.percentile(arr, 50):6.2f}  "
                f"p95={np.percentile(arr, 95):6.2f}  p99={np.percentile(arr, 99):6.2f}  "
                f"max={arr.max():6.2f}  ms (n={len(arr)})"
            )

        print(
            f"\n=== chunk={args.chunk}s  output_latency={args.latency}ms  "
            f"voice={args.voice}  duration={args.duration}s ==="
        )
        print(f"  chunks_processed = {s.chunks_processed}")
        print(f"  xruns            = {s.xruns}  (target = 0)")
        print(f"  queue_full       = {s.queue_full_events}  (target = 0)")
        print(
            f"  late_chunks      = {s.late_chunks}  (chunks where total_ms > {args.chunk * 1000:.0f})"
        )
        print(f"  pacat_restarts   = {s.pacat_restarts}")
        print(f"  dropped_chunks   = {s.dropped_chunks}")
        print(stats("inference", infs))
        print(stats("total", tots))
        if intervals:
            std = statistics.pstdev(intervals)
            mean = statistics.mean(intervals)
            print(f"  writer std = {std:.2f}ms, mean = {mean:.1f}ms (n={len(intervals)})")
    finally:
        eng.stop(timeout=3.0)


if __name__ == "__main__":
    main()
