"""Generate the synthetic 60-second sweep fixture WAV.

Composition (matches v0.7.0 brief):

| Time     | Block            | Stress target                                    |
| -------- | ---------------- | ------------------------------------------------ |
| 0-3 s    | lead silence     | input gate, vocoder voicing-floor hallucination  |
| 3-13 s   | sustained vowel  | RMVPE pitch tracking + NSF source on stable f0   |
| 13-23 s  | counting bursts  | RVC attack/decay + pitch-step transitions        |
| 23-33 s  | plosive grid     | SOLA crossfade across sharp transients           |
| 33-43 s  | fricative noise  | embedder + vocoder under high-band noise         |
| 43-53 s  | speech-like mix  | combined vowel + plosive + voicing transitions   |
| 53-60 s  | tail silence     | quiescent end (matches woys-diag tail block)     |

The fixture is mono 48 kHz int16 PCM. The amplitude is normalized so the
loudest section peaks at -3 dBFS, mirroring a hot-but-not-clipped mic.
The RNG is seeded so re-running this script produces a byte-identical
file - important so that sweep results are reproducible and a future
re-run with the same engine code lands on the same numbers.

Run:
    .venv/bin/python scripts/gen_sweep_fixture.py
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

SR = 48_000
SEED = 42
OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "auto_sweep_input.wav"


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


def _vowel(seconds: float, f0: float = 200.0, n_harmonics: int = 5) -> np.ndarray:
    """Voiced periodic signal with vibrato. Sum of harmonics ~ a sustained 'aa'."""
    t = np.arange(int(seconds * SR)) / SR
    vib = 1.0 + 0.05 * np.sin(2.0 * np.pi * 5.0 * t)  # 5 Hz, ±5 % vibrato
    f0_t = f0 * vib
    phase = 2.0 * np.pi * np.cumsum(f0_t) / SR
    sig = np.zeros_like(t)
    for h in range(1, n_harmonics + 1):
        sig += (1.0 / h) * np.sin(h * phase)
    return (sig * 0.30).astype(np.float32)


def _counting(seconds: float, rng: np.random.Generator) -> np.ndarray:
    """Short voiced bursts at varying f0 - proxy for 'one, two, three…'."""
    out = np.zeros(int(seconds * SR), dtype=np.float32)
    pitches = (110, 130, 155, 180, 165, 200, 175, 220, 195, 240)
    burst = 0.30
    gap = 0.20
    cycle = burst + gap
    n_cycles = int(seconds // cycle)
    for i in range(n_cycles):
        start = int(i * cycle * SR)
        end = start + int(burst * SR)
        if end > out.size:
            break
        v = _vowel(burst, f0=float(pitches[i % len(pitches)]))
        # Triangular envelope to suppress chunk-boundary clicks the engine isn't
        # under test for.
        env = np.ones_like(v)
        att = int(0.020 * SR)
        rel = int(0.050 * SR)
        env[:att] = np.linspace(0.0, 1.0, att)
        env[-rel:] = np.linspace(1.0, 0.0, rel)
        out[start:end] = v * env
    return out


def _plosive(seconds: float, rng: np.random.Generator) -> np.ndarray:
    """Sharp impulsive bursts every 250 ms - stresses SOLA boundary phase."""
    out = np.zeros(int(seconds * SR), dtype=np.float32)
    interval = 0.25
    n = int(seconds // interval)
    for i in range(n):
        start = int(i * interval * SR)
        burst_len = int(0.030 * SR)
        if start + burst_len > out.size:
            break
        burst = rng.standard_normal(burst_len).astype(np.float32) * 0.50
        # Fast exponential decay, ~10 ms time constant.
        decay = np.exp(-np.arange(burst_len) / (0.30 * burst_len)).astype(np.float32)
        out[start : start + burst_len] = burst * decay
    return out


def _fricative(seconds: float, rng: np.random.Generator) -> np.ndarray:
    """Band-pass-filtered white noise 4-8 kHz - proxy for 'sss', 'fff'."""
    n = int(seconds * SR)
    noise = rng.standard_normal(n).astype(np.float32)
    spec = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    mask = (freqs >= 4_000.0) & (freqs <= 8_000.0)
    spec[~mask] = 0
    filt = np.fft.irfft(spec, n=n).astype(np.float32)
    # Normalize then scale.
    rms = float(np.sqrt(np.mean(filt * filt))) or 1.0
    return filt / rms * 0.20


def _speech_like(seconds: float, rng: np.random.Generator) -> np.ndarray:
    """Stochastic interleaving of vowel + plosive + brief gap."""
    out = np.zeros(int(seconds * SR), dtype=np.float32)
    pitches = (150.0, 200.0, 175.0, 230.0, 165.0, 250.0)
    pos = 0
    pi = 0
    while pos < out.size:
        v_dur = float(rng.uniform(0.20, 0.50))
        v_len = int(v_dur * SR)
        if pos + v_len > out.size:
            break
        v = _vowel(v_dur, f0=pitches[pi % len(pitches)])
        env = np.ones_like(v)
        att = int(0.020 * SR)
        rel = int(0.050 * SR)
        env[:att] = np.linspace(0.0, 1.0, att)
        env[-rel:] = np.linspace(1.0, 0.0, rel)
        out[pos : pos + v_len] = v * env
        pos += v_len
        pi += 1
        # Plosive
        p_len = int(0.030 * SR)
        if pos + p_len > out.size:
            break
        burst = rng.standard_normal(p_len).astype(np.float32) * 0.40
        decay = np.exp(-np.arange(p_len) / (0.30 * p_len)).astype(np.float32)
        out[pos : pos + p_len] = burst * decay
        pos += p_len
        # Brief gap 50-150 ms
        gap = int(rng.uniform(0.05, 0.15) * SR)
        pos += gap
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    parts = [
        _silence(3.0),
        _vowel(10.0, f0=200.0),
        _counting(10.0, rng),
        _plosive(10.0, rng),
        _fricative(10.0, rng),
        _speech_like(10.0, rng),
        _silence(7.0),
    ]
    sig = np.concatenate(parts)

    # Normalize peak to -3 dBFS so the loudest section is hot-but-not-clipped.
    peak = float(np.abs(sig).max())
    target = 10.0 ** (-3.0 / 20.0)
    if peak > 0:
        sig = (sig * (target / peak)).astype(np.float32)

    samples = np.clip(sig * 32_767.0, -32_768.0, 32_767.0).astype(np.int16)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(OUT), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(samples.tobytes())

    print(f"wrote {OUT}  ({samples.size} samples @ {SR} Hz = {samples.size / SR:.1f}s)")
    # Print per-block dBFS for sanity.
    block_sec = (3.0, 10.0, 10.0, 10.0, 10.0, 10.0, 7.0)
    block_names = ("silence", "vowel", "counting", "plosive", "fricative", "speech", "tail")
    pos = 0
    for name, sec in zip(block_names, block_sec, strict=False):
        n = int(sec * SR)
        chunk = sig[pos : pos + n]
        rms = float(np.sqrt(np.mean(chunk * chunk))) if chunk.size else 0.0
        peak = float(np.abs(chunk).max()) if chunk.size else 0.0
        rms_db = 20.0 * np.log10(rms) if rms > 0 else -240.0
        peak_db = 20.0 * np.log10(peak) if peak > 0 else -240.0
        print(f"  {name:<10} {sec:4.1f}s  rms={rms_db:6.1f} dBFS  peak={peak_db:6.1f} dBFS")
        pos += n


if __name__ == "__main__":
    main()
