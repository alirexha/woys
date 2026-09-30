"""Golden-output safety net for the realtime engine.

Drives the real `RealtimeEngine._run_loop` -- input gain, input gate, the
48k->16k stream resampler, contentvec -> rmvpe -> rvc inference, SOLA
stitching, the output resampler, the sink byte packing and the SOLA tail
flush -- on ONNX Runtime's CPUExecutionProvider with fixed input, and
compares every byte handed to the playback writer against a recorded hash.
Any refactor of `src/audio/engine.py` must keep this test passing.

Test-only seams, none reachable by a user:
  * `onnxruntime.get_available_providers` is patched to report CPU only, so
    `_make_session` builds CPU sessions, and `EngineConfig.allow_cpu_fallback`
    (a test-only field no config key or CLI flag reaches) lets them through
    the GPU-only check. (Creating a CUDA EP session on a box without an
    NVIDIA GPU segfaults inside onnxruntime-gpu.)
  * `onnxruntime.set_seed` is fixed: the RVC graph's NSF noise source uses
    unseeded RandomNormalLike / RandomUniformLike ops.
  * `time.perf_counter` advances by `chunk_seconds` per mic read, as it
    does in real time, so the input gate's hysteresis is not at the mercy
    of how fast this CPU runs inference.
  * `sounddevice` is replaced by a fake InputStream fed from the fixture,
    `_open_pacat` returns a fake player process, and `_enqueue_chunk` is
    wrapped to record each payload before it goes to the writer queue.

Stability: three back-to-back runs on the recording machine produced
identical bytes, so the check is an exact SHA-256 match. A different CPU
can pick different MLAS kernels; set WOYS_GOLDEN_TOLERANT=1 to compare the
10 ms RMS envelope within GOLDEN_ENVELOPE_TOL instead. Regenerate with
WOYS_GOLDEN_REGEN=1 only when an output change is intended.

Models: the three foundation weights `install.sh` downloads, looked up in
$WOYS_GOLDEN_MODELS (default ~/.local/share/woys/models). The test skips
when any is missing, and is `slow`-marked so CI never runs it.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
GOLDEN_FILE = REPO / "tests" / "fixtures" / "engine_golden.json"
SPEECH_FIXTURE = REPO / "tests" / "fixtures" / "auto_sweep_input.wav"
MODELS = Path(
    os.environ.get("WOYS_GOLDEN_MODELS", str(Path.home() / ".local" / "share" / "woys" / "models"))
)
MODEL_FILES = ("contentvec-f.onnx", "rmvpe_wrapped.onnx", "amitaro_v2_16k.onnx")
ORT_SEED = 1234
# Max abs difference allowed per 10 ms RMS frame in tolerant mode.
GOLDEN_ENVELOPE_TOL = 1e-3

# name -> EngineConfig overrides. "default" is the shipped config.
VARIANTS: dict[str, dict[str, Any]] = {
    "default": {},
    "pitch_gain": {"f0_up_key": 4, "input_gain_db": 6.0},
    "no_sola_44k": {"sola_enabled": False, "mic_rate": 44_100, "sink_rate": 44_100},
}

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not all((MODELS / m).exists() for m in MODEL_FILES),
        reason=f"foundation weights not found in {MODELS} (set WOYS_GOLDEN_MODELS)",
    ),
]


def _input_signal(sr: int) -> np.ndarray:
    """10 s: 1 s silence, 4 s of the speech-like block of the sweep
    fixture, 2 s of a 220 Hz tone, 2 s of four noise bursts, 1 s silence.
    Exercises the gate (both directions), voiced and unvoiced pitch, SOLA
    across transients and the flush of a silent tail."""
    import soundfile as sf

    sweep, fixture_sr = sf.read(str(SPEECH_FIXTURE), dtype="float32")
    assert fixture_sr == 48_000
    speech = sweep[43 * fixture_sr : 47 * fixture_sr]
    rng = np.random.default_rng(1234)
    t = np.arange(2 * sr) / sr
    tone = (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    bursts = np.zeros(2 * sr, dtype=np.float32)
    for k in range(4):
        start, n = int((0.1 + 0.5 * k) * sr), int(0.15 * sr)
        bursts[start : start + n] = rng.standard_normal(n).astype(np.float32) * 0.2
    silence = np.zeros(sr, dtype=np.float32)
    return np.concatenate([silence, speech, tone, bursts, silence]).astype(np.float32)


class _FakePlayer:
    """Stands in for the pw-cat / pacat / woys-pw-out child process."""

    def __init__(self) -> None:
        import io

        self.stdin = io.BytesIO()
        self.stderr = io.BytesIO(b"")
        self.pid = 0
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = 0
        return 0

    def terminate(self) -> None:
        self.returncode = 0

    def kill(self) -> None:
        self.returncode = 0


def _run_variant(overrides: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    import onnxruntime as ort

    import audio.engine as eng

    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    ort.set_seed(ORT_SEED)

    cfg = eng.EngineConfig(
        rvc_model=MODELS / "amitaro_v2_16k.onnx",
        contentvec_model=MODELS / "contentvec-f.onnx",
        rmvpe_model=MODELS / "rmvpe_wrapped.onnx",
        use_tensorrt=False,
        allow_cpu_fallback=True,
        **overrides,
    )
    engine = eng.RealtimeEngine(cfg)
    engine._ensure_sessions()

    audio = _input_signal(cfg.mic_rate)
    chunk = int(cfg.mic_rate * cfg.chunk_seconds)
    blocks = iter(audio[i : i + chunk] for i in range(0, len(audio) - chunk + 1, chunk))

    real_perf_counter = time.perf_counter
    clock_offset = [0.0]
    monkeypatch.setattr(time, "perf_counter", lambda: real_perf_counter() + clock_offset[0])

    class _FakeInputStream:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def __enter__(self) -> _FakeInputStream:
            return self

        def __exit__(self, *_exc: object) -> None:
            return None

        def read(self, n: int) -> tuple[np.ndarray, bool]:
            clock_offset[0] += cfg.chunk_seconds
            block = next(blocks, None)
            if block is None:
                # Input exhausted: this read's (silent) chunk is the last one
                # the loop processes, then the finally block flushes SOLA.
                engine._stop_event.set()
                block = np.zeros(n, dtype=np.float32)
            return block.reshape(-1, 1).copy(), False

    monkeypatch.setitem(
        sys.modules, "sounddevice", types.SimpleNamespace(InputStream=_FakeInputStream)
    )

    payloads: list[bytes] = []
    real_enqueue = engine._enqueue_chunk

    def _capture(payload: bytes) -> None:
        payloads.append(payload)
        real_enqueue(payload)

    monkeypatch.setattr(engine, "_enqueue_chunk", _capture)
    monkeypatch.setattr(engine, "_open_pacat", _FakePlayer)
    try:
        engine._run_loop()
    finally:
        engine._stop_event.set()

    out = np.frombuffer(b"".join(payloads), dtype=np.float32)
    channels = cfg.output_channels
    mono = out[0::channels]
    frame = cfg.sink_rate // 100
    n_frames = mono.shape[0] // frame
    envelope = np.sqrt((mono[: n_frames * frame].reshape(n_frames, frame) ** 2).mean(axis=1))
    return {
        "sha256": hashlib.sha256(out.tobytes()).hexdigest(),
        "payload_bytes": [len(p) for p in payloads],
        "gated_chunks": engine.stats.gated_chunks,
        "dropped_chunks": engine.stats.dropped_chunks,
        "crashed": engine.stats.crashed,
        "rms_envelope": [round(float(x), 6) for x in envelope],
    }


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_engine_output_matches_golden(variant: str, monkeypatch: pytest.MonkeyPatch) -> None:
    got = _run_variant(VARIANTS[variant], monkeypatch)
    assert not got["crashed"] and got["dropped_chunks"] == 0, got

    golden: dict[str, Any] = json.loads(GOLDEN_FILE.read_text()) if GOLDEN_FILE.exists() else {}
    if os.environ.get("WOYS_GOLDEN_REGEN") == "1":
        golden[variant] = got
        GOLDEN_FILE.write_text(json.dumps(golden, indent=1, sort_keys=True) + "\n")
        pytest.skip(f"regenerated golden output for {variant!r}")
    want = golden[variant]

    assert got["payload_bytes"] == want["payload_bytes"], "chunk framing changed"
    assert got["gated_chunks"] == want["gated_chunks"], "input gate decisions changed"
    if os.environ.get("WOYS_GOLDEN_TOLERANT") == "1":
        diff = np.abs(np.array(got["rms_envelope"]) - np.array(want["rms_envelope"]))
        worst = int(diff.argmax())
        assert diff[worst] <= GOLDEN_ENVELOPE_TOL, (
            f"RMS envelope off by {diff[worst]:.2e} at {worst * 10} ms"
        )
    else:
        assert got["sha256"] == want["sha256"], (
            "engine output bytes changed (set WOYS_GOLDEN_TOLERANT=1 on a different CPU)"
        )
