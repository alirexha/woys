"""Phase 1 smoke test: run RVC ONNX inference end-to-end on a 1-second clip.

Bypasses upstream's pipeline classes — uses onnxruntime directly to verify:
  - ORT CUDA EP loads on this driver/CUDA combo
  - contentvec-f.onnx produces (1, T', 768) feats
  - rmvpe.onnx produces (T'',) pitch
  - amitaro_v2_16k.onnx accepts those and returns audio
  - end-to-end latency on the GPU

Run: `.venv/bin/python scripts/smoke_rvc_onnx.py`
"""

from __future__ import annotations

import time
import wave
from pathlib import Path

import numpy as np

# ORT-GPU 1.20+ on driver 595 needs explicit preload of the pip-shipped
# CUDA libs (nvidia-cublas-cu12, nvidia-cudnn-cu12) — they aren't on
# LD_LIBRARY_PATH by default.
import onnxruntime as ort

if hasattr(ort, "preload_dlls"):
    ort.preload_dlls()

ROOT = Path(__file__).resolve().parent.parent
MODELS = Path.home() / ".local" / "share" / "woys" / "models"
WAV = ROOT / "tests" / "fixtures" / "sine_voiced_1s.wav"


def make_session(path: Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3  # silence ORT chatter
    providers: list[tuple[str, dict] | str] = []
    avail = ort.get_available_providers()
    if "CUDAExecutionProvider" in avail:
        providers.append(
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                },
            )
        )
    providers.append("CPUExecutionProvider")
    sess = ort.InferenceSession(str(path), sess_options=so, providers=providers)
    return sess


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, sr


def run_contentvec(sess: ort.InferenceSession, audio16k: np.ndarray) -> np.ndarray:
    """contentvec-f: input audio (1, T) @ 16kHz → output unit12 (1, T', 768).

    contentvec-f.onnx exposes 3 outputs: units9 (256-dim), unit12 (768-dim),
    unit12s (768-dim, stop-grad). RVC v2 hidden_size=768, so we use unit12.
    """
    inputs = {"audio": audio16k.reshape(1, -1).astype(np.float32)}
    out = sess.run(["unit12"], inputs)[0]
    return out  # (1, T', 768)


def run_rmvpe(sess: ort.InferenceSession, audio16k: np.ndarray) -> np.ndarray:
    """rmvpe: input shape (1, T), threshold 0.3 → pitch (T'')."""
    inputs = {
        "waveform": audio16k.reshape(1, -1).astype(np.float32),
        "threshold": np.array([0.3], dtype=np.float32),
    }
    out = sess.run(["pitchf"], inputs)[0]
    return out.squeeze()  # (T'',)


def to_pitch_coarse(pitchf: np.ndarray, target_len: int) -> np.ndarray:
    f0_min, f0_max = 50.0, 1100.0
    f0_mel_min = 1127.0 * np.log(1 + f0_min / 700.0)
    f0_mel_max = 1127.0 * np.log(1 + f0_max / 700.0)
    pitch = np.zeros(target_len, dtype=np.float32)
    n = min(len(pitchf), target_len)
    pitch[-n:] = pitchf[:n]
    f0_mel = 1127.0 * np.log(1 + pitch / 700.0)
    mask = f0_mel > 0
    f0_mel[mask] = (f0_mel[mask] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
    f0_mel = np.clip(f0_mel, 1.0, 255.0)
    f0_coarse = np.rint(f0_mel).astype(np.int64)
    return f0_coarse, pitch.astype(np.float32)


def run_rvc(
    sess: ort.InferenceSession,
    feats: np.ndarray,
    pitch: np.ndarray,
    pitchf: np.ndarray,
    is_half: bool,
) -> np.ndarray:
    p_len = np.array([feats.shape[1]], dtype=np.int64)
    sid = np.array([0], dtype=np.int64)
    feats_dtype = np.float16 if is_half else np.float32
    inputs = {
        "feats": feats.astype(feats_dtype),
        "p_len": p_len,
        "pitch": pitch.reshape(1, -1).astype(np.int64),
        "pitchf": pitchf.reshape(1, -1).astype(np.float32),
        "sid": sid,
    }
    audio = sess.run(["audio"], inputs)[0]
    return np.array(audio).squeeze()


def main() -> int:
    print("smoke: RVC ONNX end-to-end")
    print(f"  ORT version: {ort.__version__}")
    print(f"  providers:   {ort.get_available_providers()}")
    cuda_ok = "CUDAExecutionProvider" in ort.get_available_providers()
    print(f"  CUDA EP:     {'YES' if cuda_ok else 'NO (CPU fallback)'}")

    rvc_path = MODELS / "amitaro_v2_16k.onnx"
    rmvpe_path = MODELS / "rmvpe_wrapped.onnx"
    contentvec_path = MODELS / "contentvec-f.onnx"
    for p in (rvc_path, rmvpe_path, contentvec_path, WAV):
        if not p.exists():
            print(f"  MISSING: {p}")
            return 2

    audio16k, sr = read_wav(WAV)
    assert sr == 16_000, f"expected 16kHz; got {sr}"
    print(f"  input wav:   {WAV.name} ({len(audio16k)} samples @ {sr} Hz)")

    t = time.perf_counter()
    rvc = make_session(rvc_path)
    rmvpe = make_session(rmvpe_path)
    cv = make_session(contentvec_path)
    print(f"  load (3 models): {(time.perf_counter() - t) * 1000:.1f} ms")

    print(f"  rvc providers: {rvc.get_providers()}")
    rvc_input_dtype = rvc.get_inputs()[0].type
    is_half = rvc_input_dtype != "tensor(float)"
    print(f"  rvc fp16:    {is_half}")

    # Warm-up — first GPU launch is always slow.
    for _ in range(2):
        feats = run_contentvec(cv, audio16k)
        pitchf = run_rmvpe(rmvpe, audio16k)
        pitch_coarse, pitchf_aligned = to_pitch_coarse(pitchf, target_len=feats.shape[1] * 2)
        # RVC expects feats interpolated up by 2x (50Hz -> 100Hz pitch).
        feats_2x = np.repeat(feats, 2, axis=1)
        pitch_coarse = pitch_coarse[: feats_2x.shape[1]].reshape(1, -1)
        pitchf_aligned = pitchf_aligned[: feats_2x.shape[1]].reshape(1, -1)
        _ = run_rvc(rvc, feats_2x, pitch_coarse, pitchf_aligned, is_half)

    runs = []
    n_iter = 10
    for _ in range(n_iter):
        t0 = time.perf_counter()
        feats = run_contentvec(cv, audio16k)
        t_cv = time.perf_counter()
        pitchf = run_rmvpe(rmvpe, audio16k)
        t_pitch = time.perf_counter()
        pitch_coarse, pitchf_aligned = to_pitch_coarse(pitchf, target_len=feats.shape[1] * 2)
        feats_2x = np.repeat(feats, 2, axis=1)
        pitch_coarse = pitch_coarse[: feats_2x.shape[1]].reshape(1, -1)
        pitchf_aligned = pitchf_aligned[: feats_2x.shape[1]].reshape(1, -1)
        out = run_rvc(rvc, feats_2x, pitch_coarse, pitchf_aligned, is_half)
        t_end = time.perf_counter()
        runs.append((t_cv - t0, t_pitch - t_cv, t_end - t_pitch, t_end - t0))

    cv_t = np.array([r[0] for r in runs]) * 1000
    pitch_t = np.array([r[1] for r in runs]) * 1000
    rvc_t = np.array([r[2] for r in runs]) * 1000
    total = np.array([r[3] for r in runs]) * 1000

    print(f"\n  per-run latency over {n_iter} iters (ms, mean ± stdev):")
    print(f"    contentvec : {cv_t.mean():6.2f} ± {cv_t.std():5.2f}")
    print(f"    rmvpe      : {pitch_t.mean():6.2f} ± {pitch_t.std():5.2f}")
    print(f"    rvc model  : {rvc_t.mean():6.2f} ± {rvc_t.std():5.2f}")
    print(
        f"    total e2e  : {total.mean():6.2f} ± {total.std():5.2f}   "
        f"min {total.min():.2f} / max {total.max():.2f}"
    )
    print(
        f"  output samples: {out.shape}, dtype {out.dtype}, min {out.min():.3f} max {out.max():.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
