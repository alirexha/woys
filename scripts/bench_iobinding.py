"""Compare baseline `.run()` vs IOBinding for the cv → rmvpe → rvc pipeline.

Goal: empirical answer to "is IOBinding worth implementing in the realtime
engine path?" The brief estimates -30 to -50 ms; LESSONS §6 estimates
-20 to -50 ms. With chunk_seconds=0.10-0.25, the inputs are small enough
that the .run()-side host↔device copy overhead might be negligible.

Run:
    .venv/bin/python scripts/bench_iobinding.py [--chunk 0.10] [--passes 60]
        [--voice amitaro_v2_16k]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

if hasattr(ort, "preload_dlls"):
    ort.preload_dlls()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from audio.engine import (
    MODELS_DIR,
    EngineConfig,
    RealtimeEngine,
    _interpolate_voiced_gaps_np,
    _to_pitch_coarse,
)


def percentiles(samples: list[float], ps: tuple[float, ...]) -> list[float]:
    arr = np.array(samples, dtype=np.float64)
    return [float(np.percentile(arr, p)) for p in ps]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=float, default=0.10)
    ap.add_argument("--passes", type=int, default=60)
    ap.add_argument("--warm", type=int, default=12)
    ap.add_argument("--voice", type=str, default="amitaro_v2_16k")
    ap.add_argument("--ctx-ms", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rvc_path = MODELS_DIR / f"{args.voice}.onnx"
    cfg = EngineConfig(
        rvc_model=rvc_path,
        chunk_seconds=args.chunk,
        sola_enabled=False,
        sola_context_ms=args.ctx_ms,
    )
    engine = RealtimeEngine(cfg)
    engine._ensure_sessions()
    cv = engine._cv
    rmvpe = engine._rmvpe
    rvc = engine._rvc
    is_half = engine._is_half
    cv_dtype = np.float16 if "float16" in engine._cv_input_dtype else np.float32
    rm_dtype = np.float16 if "float16" in engine._rmvpe_input_dtype else np.float32
    feats_dtype = np.float16 if is_half else np.float32

    rng = np.random.default_rng(args.seed)
    n_samples_16k = int((args.chunk + args.ctx_ms / 1000.0) * 16_000)
    audio = (rng.standard_normal(n_samples_16k).astype(np.float32) * 0.05).clip(-1.0, 1.0)

    print(
        f"# bench_iobinding  voice={args.voice}  chunk={args.chunk}s  "
        f"n={n_samples_16k}  is_half={is_half}  cuda={'CUDAExecutionProvider' in rvc.get_providers()}"
    )

    # ---- Baseline path (current engine code, .run()) ----
    def baseline_pass() -> np.ndarray:
        feats_raw = cv.run(["unit12"], {"audio": audio.reshape(1, -1).astype(cv_dtype)})[0]
        feats = feats_raw.astype(np.float32, copy=False)
        if np.isnan(feats).any():
            feats = np.nan_to_num(feats, nan=0.0)
        pitchf_raw = rmvpe.run(
            ["pitchf"],
            {
                "waveform": audio.reshape(1, -1).astype(rm_dtype),
                "threshold": np.array([cfg.threshold], dtype=rm_dtype),
            },
        )[0]
        pitchf = pitchf_raw.astype(np.float32).squeeze()
        pitchf = _interpolate_voiced_gaps_np(pitchf)
        feats_2x = np.repeat(feats, 2, axis=1)
        pitch_coarse, pitchf_aligned = _to_pitch_coarse(pitchf, target_len=feats_2x.shape[1])
        pitch_coarse = pitch_coarse[: feats_2x.shape[1]].reshape(1, -1)
        pitchf_aligned = pitchf_aligned[: feats_2x.shape[1]].reshape(1, -1).astype(np.float32)
        out = rvc.run(
            ["audio"],
            {
                "feats": feats_2x.astype(feats_dtype),
                "p_len": np.array([feats_2x.shape[1]], dtype=np.int64),
                "pitch": pitch_coarse,
                "pitchf": pitchf_aligned,
                "sid": np.array([cfg.sid], dtype=np.int64),
            },
        )[0]
        return np.array(out).astype(np.float32).squeeze()

    # ---- IOBinding path: pre-allocate the audio input on GPU once,
    # reuse across both cv and rmvpe (same data). RVC inputs vary per
    # call, but we still bind them as OrtValues to skip the .run()
    # host→device path.
    audio_in = audio.reshape(1, -1)
    cv_audio = ort.OrtValue.ortvalue_from_numpy(audio_in.astype(cv_dtype), "cuda", 0)
    rmvpe_audio = ort.OrtValue.ortvalue_from_numpy(audio_in.astype(rm_dtype), "cuda", 0)
    threshold_v = ort.OrtValue.ortvalue_from_numpy(
        np.array([cfg.threshold], dtype=rm_dtype), "cuda", 0
    )
    sid_v = ort.OrtValue.ortvalue_from_numpy(np.array([cfg.sid], dtype=np.int64), "cuda", 0)

    cv_io = cv.io_binding()
    rmvpe_io = rmvpe.io_binding()
    rvc_io = rvc.io_binding()

    def iobind_pass() -> np.ndarray:
        cv_io.clear_binding_inputs()
        cv_io.clear_binding_outputs()
        cv_io.bind_ortvalue_input("audio", cv_audio)
        cv_io.bind_output("unit12", "cuda", 0)
        cv.run_with_iobinding(cv_io)
        feats_raw = cv_io.get_outputs()[0].numpy()
        feats = feats_raw.astype(np.float32, copy=False)
        if np.isnan(feats).any():
            feats = np.nan_to_num(feats, nan=0.0)

        rmvpe_io.clear_binding_inputs()
        rmvpe_io.clear_binding_outputs()
        rmvpe_io.bind_ortvalue_input("waveform", rmvpe_audio)
        rmvpe_io.bind_ortvalue_input("threshold", threshold_v)
        rmvpe_io.bind_output("pitchf", "cuda", 0)
        rmvpe.run_with_iobinding(rmvpe_io)
        pitchf_raw = rmvpe_io.get_outputs()[0].numpy()
        pitchf = pitchf_raw.astype(np.float32).squeeze()
        pitchf = _interpolate_voiced_gaps_np(pitchf)
        feats_2x = np.repeat(feats, 2, axis=1)
        pitch_coarse, pitchf_aligned = _to_pitch_coarse(pitchf, target_len=feats_2x.shape[1])
        pitch_coarse = pitch_coarse[: feats_2x.shape[1]].reshape(1, -1)
        pitchf_aligned = pitchf_aligned[: feats_2x.shape[1]].reshape(1, -1).astype(np.float32)

        rvc_io.clear_binding_inputs()
        rvc_io.clear_binding_outputs()
        rvc_io.bind_cpu_input("feats", feats_2x.astype(feats_dtype))
        rvc_io.bind_cpu_input("p_len", np.array([feats_2x.shape[1]], dtype=np.int64))
        rvc_io.bind_cpu_input("pitch", pitch_coarse)
        rvc_io.bind_cpu_input("pitchf", pitchf_aligned)
        rvc_io.bind_ortvalue_input("sid", sid_v)
        rvc_io.bind_output("audio", "cuda", 0)
        rvc.run_with_iobinding(rvc_io)
        out = rvc_io.get_outputs()[0].numpy()
        return np.array(out).astype(np.float32).squeeze()

    # Warm both paths
    for _ in range(args.warm):
        baseline_pass()
        iobind_pass()

    base_t: list[float] = []
    iobind_t: list[float] = []
    for _ in range(args.passes):
        t = time.perf_counter()
        baseline_pass()
        base_t.append((time.perf_counter() - t) * 1000.0)
        t = time.perf_counter()
        iobind_pass()
        iobind_t.append((time.perf_counter() - t) * 1000.0)

    def fmt(name: str, vals: list[float]) -> str:
        p50, p95, p99 = percentiles(vals, (50.0, 95.0, 99.0))
        return (
            f"  {name:<12} avg={np.mean(vals):6.2f}  p50={p50:6.2f}  "
            f"p95={p95:6.2f}  p99={p99:6.2f}  max={max(vals):6.2f}  ms"
        )

    print(fmt("BASELINE", base_t))
    print(fmt("IOBINDING", iobind_t))
    delta = np.mean(base_t) - np.mean(iobind_t)
    pct = delta / np.mean(base_t) * 100
    print(f"  Δavg = {delta:+.2f} ms ({pct:+.1f}%)")


if __name__ == "__main__":
    main()
