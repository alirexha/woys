#!/usr/bin/env python
"""SOLA A/B harness -- runs an input WAV through `SOLAStream` chunk by
chunk, optionally with the pre-F-07-03 loop reference monkey-patched
in, and writes the output WAVs side-by-side so Alireza can flip
between them in his audio player.

Usage:
    .venv/bin/python scripts/sola_ab_harness.py INPUT.wav OUT_DIR [options]

Outputs:
    OUT_DIR/A_vectorised.wav   -- production `_best_offset` (default)
    OUT_DIR/B_loop_reference.wav  -- pre-F-07-03 reference (--ab)
    OUT_DIR/diff_abs.wav       -- |A - B| amplified 100x (--ab + --diff)

Quick null-listener test:
    .venv/bin/python scripts/sola_ab_harness.py my_voice.wav /tmp/ab --ab --diff
    paplay /tmp/ab/A_vectorised.wav
    paplay /tmp/ab/B_loop_reference.wav   # should be perceptually identical
    paplay /tmp/ab/diff_abs.wav           # should be near-silent

This harness is SOLA-only: it does NOT run RVC inference, RMVPE, the
GPU clock-lock, PipeWire, or anything else. The input WAV is fed
DIRECTLY into SOLAStream as if it were already model-output. Use any
mono float-friendly WAV at any rate; the harness resamples to
SOLAConfig.rate (default 16 kHz) before running. The output is at
the same rate; resample to your sink rate to play.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
for p in (REPO / "src", REPO / "src" / "server"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _load_wav_mono(path: Path, target_rate: int) -> np.ndarray:
    """Load any WAV, mono-down-mix, resample to `target_rate`, return
    a contiguous fp32 array."""
    import soundfile as sf
    from scipy import signal

    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.ascontiguousarray(audio).astype(np.float32)
    if sr != target_rate:
        n_out = round(len(audio) * target_rate / sr)
        audio = signal.resample(audio, n_out).astype(np.float32)
    return audio


def _run_sola(
    audio: np.ndarray,
    *,
    rate: int,
    chunk_seconds: float,
    use_loop_reference: bool,
) -> tuple[np.ndarray, int]:
    """Stream `audio` through a fresh `SOLAStream`. Returns
    `(output, fallback_count)`."""
    from audio import sola

    cfg = sola.SOLAConfig(rate=rate)
    stream = sola.SOLAStream(cfg)
    cf = cfg.crossfade_samples
    search = cfg.search_samples
    chunk_n = round(rate * chunk_seconds)
    if chunk_n <= 0:
        raise ValueError("chunk_seconds too small")
    feed_len = chunk_n + cf + search

    # F-07-03 reference path: monkey-patch the production
    # `_best_offset` symbol back to the loop reference for this run
    # only. SOLAStream.process imports it from module scope, so this
    # is enough to flip the implementation.
    original = sola._best_offset
    if use_loop_reference:
        sola._best_offset = sola._best_offset_loop_reference  # type: ignore[assignment]

    try:
        outputs: list[np.ndarray] = []
        cursor = 0
        # Pad the tail so the last partial chunk has the full feed window.
        padded = np.concatenate([audio, np.zeros(feed_len, dtype=np.float32)])
        while cursor + feed_len <= padded.shape[0]:
            window = padded[cursor : cursor + feed_len]
            emit = stream.process(window)
            if emit.size > 0:
                outputs.append(emit)
            cursor += chunk_n
        tail = stream.flush()
        if tail.size > 0:
            outputs.append(tail)
        result = (
            np.concatenate(outputs).astype(np.float32) if outputs else np.zeros(0, dtype=np.float32)
        )
        return result, stream.fallback_count
    finally:
        sola._best_offset = original  # type: ignore[assignment]


def _write_wav(path: Path, audio: np.ndarray, rate: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, rate, subtype="FLOAT")


def _gen_synthetic_voice(rate: int, duration_s: float, *, seed: int = 42) -> np.ndarray:
    """Build a deterministic-but-voice-like signal: a 150 Hz triangle
    with an AM envelope and a small vibrato. Useful when the caller
    has no real model-output sample on hand."""
    rng = np.random.default_rng(seed)
    n = int(rate * duration_s)
    t = np.arange(n, dtype=np.float32) / rate
    f0 = 150.0 + 5.0 * np.sin(2 * np.pi * 5.0 * t)
    phase = np.cumsum(2 * np.pi * f0 / rate)
    base = 0.5 * np.sign(np.sin(phase)) * (1.0 - np.abs(np.sin(phase / 2)))
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 1.5 * t)
    sig = (base * env + 0.02 * rng.standard_normal(n)).astype(np.float32)
    return sig


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "input",
        type=Path,
        nargs="?",
        help="input WAV. If omitted, --synthetic must be passed.",
    )
    ap.add_argument(
        "out_dir",
        type=Path,
        help="output directory; written: A_vectorised.wav (+ B_loop_reference.wav + diff_abs.wav with --ab/--diff)",
    )
    ap.add_argument(
        "--rate",
        type=int,
        default=16000,
        help="SOLA stream rate (default 16000, matches engine model output rate)",
    )
    ap.add_argument(
        "--chunk-seconds",
        type=float,
        default=0.25,
        help="chunk size fed to SOLAStream (default 0.25 -- engine default since v0.12.4)",
    )
    ap.add_argument(
        "--ab",
        action="store_true",
        help="also run with `_best_offset_loop_reference` patched in and write B_loop_reference.wav",
    )
    ap.add_argument(
        "--diff",
        action="store_true",
        help="(implies --ab) also write `diff_abs.wav` = |A - B| x 100 for visual / aural verification of the null-listener claim",
    )
    ap.add_argument(
        "--synthetic",
        type=float,
        metavar="DURATION_S",
        help="generate a synthetic voice-like signal of this duration in seconds instead of reading a WAV",
    )
    args = ap.parse_args()

    if args.diff:
        args.ab = True

    if args.synthetic:
        signal = _gen_synthetic_voice(args.rate, args.synthetic)
        src_label = f"synthetic {args.synthetic:.1f} s @ {args.rate} Hz"
    else:
        if args.input is None:
            ap.error("either INPUT.wav or --synthetic DURATION must be passed")
        signal = _load_wav_mono(args.input, args.rate)
        src_label = f"{args.input} @ {args.rate} Hz"

    print(
        f"[harness] source: {src_label}  ({signal.shape[0] / args.rate:.2f} s, "
        f"{signal.shape[0]} samples)",
        file=sys.stderr,
    )

    out_a, fb_a = _run_sola(
        signal, rate=args.rate, chunk_seconds=args.chunk_seconds, use_loop_reference=False
    )
    a_path = args.out_dir / "A_vectorised.wav"
    _write_wav(a_path, out_a, args.rate)
    print(
        f"[A: vectorised   ] {a_path}  ({out_a.shape[0] / args.rate:.2f} s, fallback_count={fb_a})",
        file=sys.stderr,
    )

    if args.ab:
        out_b, fb_b = _run_sola(
            signal, rate=args.rate, chunk_seconds=args.chunk_seconds, use_loop_reference=True
        )
        b_path = args.out_dir / "B_loop_reference.wav"
        _write_wav(b_path, out_b, args.rate)
        print(
            f"[B: loop ref     ] {b_path}  ({out_b.shape[0] / args.rate:.2f} s, "
            f"fallback_count={fb_b})",
            file=sys.stderr,
        )

        # Stats on the A vs. B difference.
        common = min(out_a.shape[0], out_b.shape[0])
        a = out_a[:common]
        b = out_b[:common]
        diff = a - b
        diff_max = float(np.max(np.abs(diff))) if common > 0 else 0.0
        diff_rms = float(np.sqrt(np.mean(diff * diff))) if common > 0 else 0.0
        a_rms = float(np.sqrt(np.mean(a * a))) if common > 0 else 0.0
        snr_db = 20.0 * np.log10(a_rms / max(diff_rms, 1e-12)) if a_rms > 1e-12 else float("inf")
        print(
            f"[A vs B diff     ] max|Δ|={diff_max:.6f}  rms(Δ)={diff_rms:.6f}  "
            f"signal-to-diff SNR≈{snr_db:.1f} dB  "
            f"fallback_delta={fb_a - fb_b:+d}",
            file=sys.stderr,
        )

        if args.diff:
            d_path = args.out_dir / "diff_abs.wav"
            amplified = np.clip(np.abs(diff) * 100.0, -1.0, 1.0).astype(np.float32)
            _write_wav(d_path, amplified, args.rate)
            print(
                f"[diff (x100 amp) ] {d_path}  (audible noise here = behavioural divergence)",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
