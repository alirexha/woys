"""Like bench_inference.py but exercises the SOLA streaming wrapper +
soxr resamplers, mimicking the realtime engine path WITHOUT the audio
threads (no sounddevice, no pacat). Isolates streaming overhead from
GIL/thread contention.

If `_process_streaming_16k()` is fast here but slow inside the running
engine, the gap is threading. If it's slow here too, the gap is the
streaming wrapper itself.
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

from audio.engine import MODELS_DIR, EngineConfig, RealtimeEngine, _resample


def percentiles(samples: list[float], ps: tuple[float, ...]) -> list[float]:
    arr = np.array(samples, dtype=np.float64)
    return [float(np.percentile(arr, p)) for p in ps]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=float, default=0.10)
    ap.add_argument("--passes", type=int, default=60)
    ap.add_argument("--warm", type=int, default=12)
    ap.add_argument("--voice", type=str, default="catwoman")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rvc_path = MODELS_DIR / f"{args.voice}.onnx"
    cfg = EngineConfig(rvc_model=rvc_path, chunk_seconds=args.chunk)
    engine = RealtimeEngine(cfg)
    engine._ensure_sessions()
    # Build the SOLA + resampler state the same way `start()` would.
    engine.reset_streaming_state()

    rng = np.random.default_rng(args.seed)
    n_samples_mic = int(args.chunk * cfg.mic_rate)

    print(
        f"# bench_streaming  voice={args.voice}  chunk={args.chunk}s  "
        f"n_mic={n_samples_mic}  cuda={'CUDAExecutionProvider' in engine._rvc.get_providers()}"
    )

    def make_chunk() -> np.ndarray:
        return (rng.standard_normal(n_samples_mic).astype(np.float32) * 0.05).clip(-1.0, 1.0)

    # Warmup
    for _ in range(args.warm):
        audio_mic = make_chunk()
        audio16 = (
            engine._resampler_in.process(audio_mic)
            if engine._resampler_in is not None
            else _resample(audio_mic, cfg.mic_rate, 16_000)
        )
        if audio16.size:
            _ = engine._safe_process_streaming_16k(audio16)

    streaming_t: list[float] = []
    infer_t: list[float] = []
    resample_t: list[float] = []

    for _ in range(args.passes):
        audio_mic = make_chunk()
        t = time.perf_counter()
        audio16 = (
            engine._resampler_in.process(audio_mic)
            if engine._resampler_in is not None
            else _resample(audio_mic, cfg.mic_rate, 16_000)
        )
        resample_t.append((time.perf_counter() - t) * 1000.0)
        if not audio16.size:
            continue
        t = time.perf_counter()
        _ = engine._safe_process_streaming_16k(audio16)
        streaming_t.append((time.perf_counter() - t) * 1000.0)
        infer_t.append(
            engine.stats.last_cv_ms + engine.stats.last_rmvpe_ms + engine.stats.last_rvc_ms
        )

    def fmt(name: str, vals: list[float]) -> str:
        if not vals:
            return f"  {name:<12} (no data)"
        p50, p95, p99 = percentiles(vals, (50.0, 95.0, 99.0))
        return (
            f"  {name:<12} avg={np.mean(vals):6.2f}  p50={p50:6.2f}  p95={p95:6.2f}  "
            f"p99={p99:6.2f}  max={max(vals):6.2f}  ms"
        )

    print(fmt("resample_in", resample_t))
    print(fmt("infer_only", infer_t))
    print(fmt("streaming", streaming_t))


if __name__ == "__main__":
    main()
