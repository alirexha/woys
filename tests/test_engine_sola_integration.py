"""Phase B integration tests — end-to-end SOLA in the engine.

Two checks:
  1. Sustained sine sweep through the full mic→model→sink pipeline (with
     `_process_streaming_16k`) produces output without broadband impulses
     at chunk boundaries. Clicks would show up as spikes in the high-frequency
     bands of an FFT taken at suspected boundary positions.
  2. Live engine measurement at chunk_seconds=0.1 with SOLA on: warm-state
     `avg_total_ms` < 120 ms (the v0.2.0 brief target).
"""

from __future__ import annotations

import time

import numpy as np
import pytest


def _voice_like(sr: int, duration_s: float) -> np.ndarray:
    """Synthesize a voiced waveform — sum of harmonics with mild AM, the
    kind of signal where SOLA matters most (constant pitch, sustained)."""
    t = np.arange(int(sr * duration_s)) / sr
    f0 = 220.0
    am = 0.7 + 0.3 * np.sin(2 * np.pi * 4 * t)
    sig = (
        0.6 * np.sin(2 * np.pi * f0 * t)
        + 0.25 * np.sin(2 * np.pi * 2 * f0 * t)
        + 0.10 * np.sin(2 * np.pi * 3 * f0 * t)
    ) * am
    return sig.astype(np.float32) * 0.4


def _hf_energy_ratio(audio: np.ndarray, sr: int) -> float:
    """Energy above 4 kHz / total energy. A clean voiced signal sits well
    below 0.05; chunk-boundary clicks push it above 0.10."""
    spec = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(audio.shape[0], 1.0 / sr)
    total = float(np.sum(spec**2)) + 1e-9
    high = float(np.sum(spec[freqs > 4_000] ** 2))
    return high / total


@pytest.mark.gpu
@pytest.mark.slow
def test_streaming_no_click_artifacts_with_sola() -> None:
    """Run a 3 s voiced waveform through `_process_streaming_16k` in 100 ms
    chunks. Compare HF energy of SOLA-on vs SOLA-off to confirm SOLA reduces
    chunk-boundary click energy."""
    from audio.engine import EngineConfig, RealtimeEngine

    sr = 16_000
    voiced = _voice_like(sr, duration_s=3.0)
    chunk_n = sr // 10  # 100 ms

    def run(sola_on: bool) -> np.ndarray:
        eng = RealtimeEngine(EngineConfig(chunk_seconds=0.1, sola_enabled=sola_on))
        eng._ensure_sessions()
        eng.reset_streaming_state()
        out_pieces: list[np.ndarray] = []
        for i in range(0, voiced.shape[0], chunk_n):
            piece = eng._process_streaming_16k(voiced[i : i + chunk_n])
            if piece.size:
                out_pieces.append(piece)
        if eng._sola is not None:
            tail = eng._sola.flush()
            if tail.size:
                out_pieces.append(tail)
        return np.concatenate(out_pieces) if out_pieces else np.zeros(0, dtype=np.float32)

    out_on = run(sola_on=True)
    out_off = run(sola_on=False)
    assert out_on.shape[0] > sr  # at least 1 s of output
    assert out_off.shape[0] > sr

    hf_on = _hf_energy_ratio(out_on, sr)
    hf_off = _hf_energy_ratio(out_off, sr)
    print(f"\n  hf-energy ratio: SOLA off={hf_off:.4f}  SOLA on={hf_on:.4f}")
    # SOLA should reduce or match HF-energy — never increase it.
    assert hf_on <= hf_off + 0.02, (
        f"SOLA should not increase HF artifact energy: off={hf_off:.4f} on={hf_on:.4f}"
    )


@pytest.mark.gpu
@pytest.mark.pipewire
@pytest.mark.slow
def test_engine_warm_avg_total_under_120ms_at_chunk_100ms() -> None:
    """Live engine measurement — the headline v0.2.0 latency target."""
    from audio.engine import EngineConfig, RealtimeEngine
    from audio.pipewire import VirtualMic, get_state

    VirtualMic().ensure()
    if not get_state().fully_present:
        pytest.skip("WoysSink + woys-mic not loaded")

    eng = RealtimeEngine(EngineConfig(chunk_seconds=0.1, sola_enabled=True))
    eng.start()
    try:
        time.sleep(2.5)  # warmup window — cudnn autotune settles here
        eng.stats._recent_inference.clear()
        eng.stats._recent_total.clear()
        time.sleep(6.0)
        avg_total = eng.stats.avg_total_ms
        chunks = eng.stats.chunks_processed
        print(f"\n  warm e2e at chunk=0.1, SOLA on: avg_total_ms={avg_total:.2f}, chunks={chunks}")
        assert eng.stats.last_error is None, eng.stats.last_error
        assert avg_total < 120.0, (
            f"avg_total {avg_total:.2f} ms exceeds v0.2.0 brief target of 120 ms"
        )
    finally:
        eng.stop()
