"""Phase 1 smoke test: full ONNX RVC pipeline on a 1s WAV.

Measures real wall-clock latency on GPU. The brief's <80ms target applies to
end-to-end mic→output, but this test isolates the inference cost (no audio I/O,
no SOLA crossfade) and acts as the floor.
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

if hasattr(ort, "preload_dlls"):
    ort.preload_dlls()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS = Path.home() / ".local" / "share" / "woys" / "models"
WAV = PROJECT_ROOT / "tests" / "fixtures" / "sine_voiced_1s.wav"

LATENCY_FLOOR_MS = 80.0  # Phase 1 budget for inference-only.

# B24 / quality-020 / test-016: import the production f0-coarse function
# instead of re-implementing it. Pre-v0.8.0 this test had its own copy
# that drifted from the engine version (missed the early-exit added in
# B56 / perf-003, for instance). Single source of truth.
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT / "src" / "server") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src" / "server"))

from audio.engine import to_pitch_coarse as _to_pitch_coarse  # noqa: E402


def _have_models() -> bool:
    return all(
        (MODELS / n).exists()
        for n in ("amitaro_v2_16k.onnx", "rmvpe_wrapped.onnx", "contentvec-f.onnx")
    )


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0, sr


def _make_session(path: Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    providers: list = []
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.append(("CUDAExecutionProvider", {"device_id": 0}))
    providers.append("CPUExecutionProvider")
    return ort.InferenceSession(str(path), sess_options=so, providers=providers)


@pytest.mark.gpu
@pytest.mark.slow
def test_rvc_onnx_end_to_end_under_80ms() -> None:
    if not _have_models():
        pytest.skip(f"weights not in {MODELS} - run scripts/download_weights.py")

    audio, sr = _read_wav(WAV)
    assert sr == 16_000

    cv = _make_session(MODELS / "contentvec-f.onnx")
    rmvpe = _make_session(MODELS / "rmvpe_wrapped.onnx")
    rvc = _make_session(MODELS / "amitaro_v2_16k.onnx")

    # review F-merged-001 (P0): this was a `pytest.skip`, which silently
    # passed the run when CUDA EP failed to bind -- the exact silent-fallback
    # class the audit hard-fails. On a GPU box (this test is @pytest.mark.gpu)
    # a CPU-only binding is a real regression: assert, don't skip.
    assert "CUDAExecutionProvider" in rvc.get_providers(), (
        f"CUDA EP did not bind for the RVC session (providers="
        f"{rvc.get_providers()}); realtime RVC is unusable on CPU. Check the "
        f"onnxruntime-gpu wheel / NVIDIA driver / ort.preload_dlls()."
    )

    is_half = rvc.get_inputs()[0].type != "tensor(float)"

    def one_pass() -> np.ndarray:
        feats = cv.run(["unit12"], {"audio": audio.reshape(1, -1).astype(np.float32)})[0]
        pitchf = rmvpe.run(
            ["pitchf"],
            {
                "waveform": audio.reshape(1, -1).astype(np.float32),
                "threshold": np.array([0.3], dtype=np.float32),
            },
        )[0].squeeze()
        feats_2x = np.repeat(feats, 2, axis=1)
        pitch_coarse, pitchf_aligned = _to_pitch_coarse(pitchf, target_len=feats_2x.shape[1])
        pitch_coarse = pitch_coarse[: feats_2x.shape[1]].reshape(1, -1)
        pitchf_aligned = pitchf_aligned[: feats_2x.shape[1]].reshape(1, -1)
        out = rvc.run(
            ["audio"],
            {
                "feats": feats_2x.astype(np.float16 if is_half else np.float32),
                "p_len": np.array([feats_2x.shape[1]], dtype=np.int64),
                "pitch": pitch_coarse,
                "pitchf": pitchf_aligned.astype(np.float32),
                "sid": np.array([0], dtype=np.int64),
            },
        )[0]
        return np.array(out).squeeze()

    # Warm up; first GPU launch is always slow.
    for _ in range(2):
        one_pass()

    import time

    samples = []
    for _ in range(10):
        t = time.perf_counter()
        out = one_pass()
        samples.append((time.perf_counter() - t) * 1000)

    arr = np.array(samples)
    print(
        f"\n  e2e latency: mean {arr.mean():.2f} ± {arr.std():.2f} ms  "
        f"(min {arr.min():.2f}, max {arr.max():.2f})"
    )
    assert out.size > 0, "RVC produced empty output"
    assert np.isfinite(out).all(), "RVC output has NaN/Inf"
    assert arr.mean() < LATENCY_FLOOR_MS, (
        f"e2e mean {arr.mean():.2f}ms exceeds Phase 1 floor of {LATENCY_FLOOR_MS}ms"
    )
