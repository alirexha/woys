#!/usr/bin/env python
"""v0.12.1 — drive the engine with TTS-generated speech (real f0 contour,
real formants, real consonant/vowel transitions) for natural-speech-class
NSF-reset detection. Auto-resamples the TTS WAV to the harness's
mic_rate (48 kHz) and tiles to fill --duration.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
for p in (REPO / "src", REPO / "src" / "server", REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _load_and_resample(wav_path: Path, target_sr: int) -> np.ndarray:
    import soundfile as sf
    from scipy import signal

    audio, sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        n_out = int(round(len(audio) * target_sr / sr))
        audio = signal.resample(audio, n_out).astype(np.float32)
    return audio


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--tts-wav", type=Path, default=Path("/tmp/v012_1/tts_input.wav"))
    parser.add_argument("--out", type=Path, default=Path("/tmp/v012_1/tts_run.json"))
    parser.add_argument("--anti-jitter-mode", default="both")
    parser.add_argument("--chunk-seconds", type=float, default=None)
    parser.add_argument("--sola-crossfade-ms", type=float, default=None)
    parser.add_argument("--sola-search-ms", type=float, default=None)
    parser.add_argument("--sola-context-ms", type=float, default=None)
    parser.add_argument("--sola-corr-threshold", type=float, default=None)
    args = parser.parse_args()

    import v010_harness as harness

    print(f"[tts] loading {args.tts_wav}", file=sys.stderr)
    src = _load_and_resample(args.tts_wav, target_sr=48000)
    src_dur = len(src) / 48000.0
    print(f"[tts] source duration={src_dur:.2f}s @ 48kHz; tiling to fill {args.duration:.0f}s", file=sys.stderr)

    # Normalize to RMS=0.10 to match the synthetic harness's voiced-segment level
    rms = float(np.sqrt(np.mean(src.astype(np.float64) ** 2)))
    if rms > 0:
        src = src * np.float32(0.10 / rms)
    # Hard-clip to avoid out-of-range after gain.
    np.clip(src, -1.0, 1.0, out=src)

    def tts_builder(duration_s: float, sample_rate: int, *, seed: int = 42) -> np.ndarray:
        n_total = int(round(duration_s * sample_rate))
        n_loops = int(np.ceil(n_total / len(src))) + 1
        out = np.tile(src, n_loops)[:n_total]
        return out.astype(np.float32, copy=False)

    harness._build_signal = tts_builder
    print(f"[tts] driving harness for {args.duration:.0f}s, anti_jitter={args.anti_jitter_mode}",
          file=sys.stderr)

    out = harness._run_engine_synthetic(
        duration_s=args.duration,
        out_path=args.out,
        enable_sola=True,
        chunk_seconds=args.chunk_seconds,
        inference_subprocess=False,
        gpu_anti_jitter_mode=args.anti_jitter_mode,
        sola_crossfade_ms=args.sola_crossfade_ms,
        sola_search_ms=args.sola_search_ms,
        sola_context_ms=args.sola_context_ms,
        sola_corr_threshold=args.sola_corr_threshold,
    )
    harness._print_summary(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
