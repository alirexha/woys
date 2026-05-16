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
    use_legacy_fade: bool = False,
) -> tuple[np.ndarray, int]:
    """Stream `audio` through a fresh `SOLAStream`. Returns
    `(output, fallback_count)`.

    `use_loop_reference=True` flips `_best_offset` to the pre-F-07-03
    loop reference (commit-077 A/B).
    `use_legacy_fade=True` flips `_USE_EQUAL_POWER_ON_FALLBACK` to
    `False` so the fall_back branch uses the pre-F-31-04 equal-gain
    Hann pair (commit-078 A/B). The two flags compose.
    """
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
    original_best = sola._best_offset
    if use_loop_reference:
        sola._best_offset = sola._best_offset_loop_reference  # type: ignore[assignment]
    # F-31-04 legacy-fade path: flip the module-level toggle so the
    # fall_back branch uses the pre-fix equal-gain pair.
    original_flag = sola._USE_EQUAL_POWER_ON_FALLBACK
    if use_legacy_fade:
        sola._USE_EQUAL_POWER_ON_FALLBACK = False

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
        sola._best_offset = original_best  # type: ignore[assignment]
        sola._USE_EQUAL_POWER_ON_FALLBACK = original_flag


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


def _run_sola_per_chunk_noise(
    *,
    rate: int,
    chunk_seconds: float,
    duration_s: float,
    use_legacy_fade: bool,
    seed: int = 42,
) -> tuple[np.ndarray, int]:
    """Feed `SOLAStream.process()` independent white-noise buffers
    every chunk. This bypasses the normal sliding-window feed --
    where prev_tail and head share input samples and correlate at 1
    by construction -- so the fall_back branch fires on EVERY chunk
    after the first. Used by the `--per-chunk-noise` mode for an
    aggressive F-31-04 stress test.

    The normal `_run_sola` path is correct for everything else
    (vectorisation null-listener, real-WAV throughput, the
    synthetic-fricatives realistic mode). Only F-31-04's equal-power
    branch needs per-chunk independent buffers to be exercised
    offline without running the actual model.
    """
    from audio import sola

    cfg = sola.SOLAConfig(rate=rate)
    stream = sola.SOLAStream(cfg)
    cf = cfg.crossfade_samples
    search = cfg.search_samples
    chunk_n = round(rate * chunk_seconds)
    feed_len = chunk_n + cf + search
    n_chunks = max(1, int(duration_s / chunk_seconds))
    rng = np.random.default_rng(seed)

    original_flag = sola._USE_EQUAL_POWER_ON_FALLBACK
    if use_legacy_fade:
        sola._USE_EQUAL_POWER_ON_FALLBACK = False
    try:
        outputs: list[np.ndarray] = []
        for _ in range(n_chunks):
            buf = (rng.standard_normal(feed_len) * 0.3).astype(np.float32)
            emit = stream.process(buf)
            if emit.size > 0:
                outputs.append(emit)
        tail = stream.flush()
        if tail.size > 0:
            outputs.append(tail)
        result = (
            np.concatenate(outputs).astype(np.float32) if outputs else np.zeros(0, dtype=np.float32)
        )
        return result, stream.fallback_count
    finally:
        sola._USE_EQUAL_POWER_ON_FALLBACK = original_flag


def _gen_synthetic_fricatives(
    rate: int, duration_s: float, *, chunk_seconds: float = 0.25, seed: int = 42
) -> np.ndarray:
    """Build a vowel-fricative-vowel-fricative-... pattern so SOLA's
    `fall_back` branch actually fires, exercising the F-31-04
    equal-power crossfade.

    The phoneme boundary must land *inside* SOLA's prev_tail /
    head correlation window for `_best_offset` to drop below the
    `corr_threshold` (default 0.25). Pure chunk-aligned segments
    miss this: prev_tail (last 50 ms of chunk N-1) and head (first
    50 ms of chunk N) both sit inside the SAME phoneme even when
    chunks alternate. The fix is a half-crossfade-width shift so a
    transition at sample `cf // 2` of every chunk falls in the
    middle of the prev_tail window.

    Fricatives are band-pass white noise centred ~4 kHz (a `/s/`-
    like profile); vowels are the same 150 Hz triangle as
    `_gen_synthetic_voice`. With the default chunk_seconds=0.25 +
    rate=16 kHz, this produces a fall_back every chunk past the
    first (an aggressive exercise of the F-31-04 fade branch).
    """
    from scipy import signal as sps

    rng = np.random.default_rng(seed)
    n = int(rate * duration_s)
    out = np.zeros(n, dtype=np.float32)
    seg_len = int(rate * chunk_seconds)
    # Half-crossfade shift so the phoneme transition lands inside
    # SOLA's prev_tail/head correlation window (where it drives
    # corr below the 0.25 threshold and triggers fall_back).
    cf_samples = int(rate * 0.050)  # matches SOLAConfig.crossfade_ms default
    shift = cf_samples // 2
    # Pre-build the two basis waveforms then tile.
    n_seg = max(1, (n - shift) // seg_len + 1)
    t_seg = np.arange(seg_len, dtype=np.float32) / rate
    f0 = 150.0 + 5.0 * np.sin(2 * np.pi * 5.0 * t_seg)
    phase = np.cumsum(2 * np.pi * f0 / rate)
    vowel = (0.5 * np.sign(np.sin(phase)) * (1.0 - np.abs(np.sin(phase / 2)))).astype(np.float32)
    # Band-pass filter design for the fricative; Butterworth 4-pole at
    # 3-6 kHz so we get noise concentrated in the /s/ band.
    sos = sps.butter(4, [3000.0 / (rate / 2), 6000.0 / (rate / 2)], btype="bandpass", output="sos")
    # Leading partial vowel segment to fill [0, shift) so the first
    # full segment starts at sample `shift` (and the phoneme
    # transitions fall at samples `shift`, `shift + seg_len`, etc. --
    # half a crossfade into the prev_tail window of the following chunk).
    if shift > 0:
        out[:shift] = vowel[:shift]
    for i in range(n_seg):
        if i % 2 == 0:
            noise = rng.standard_normal(seg_len).astype(np.float32)
            seg = sps.sosfilt(sos, noise).astype(np.float32) * 0.6
        else:
            seg = vowel.copy()
        start = shift + i * seg_len
        if start >= n:
            break
        end = min(start + seg_len, n)
        out[start:end] = seg[: end - start]
    return out


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
        help="also run with `_best_offset_loop_reference` patched in and write B_loop_reference.wav (commit-077 A/B)",
    )
    ap.add_argument(
        "--legacy-fade",
        action="store_true",
        help="run a B with the pre-F-31-04 equal-gain Hann fade on the fall_back branch and write B_legacy_fade.wav (commit-078 A/B); composes with --ab",
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
    ap.add_argument(
        "--synthetic-fricatives",
        type=float,
        metavar="DURATION_S",
        help="generate a vowel-fricative alternating signal so SOLA fall_back fires periodically (F-31-04 listener pass)",
    )
    ap.add_argument(
        "--per-chunk-noise",
        type=float,
        metavar="DURATION_S",
        help="feed SOLA independent white-noise buffers per chunk (bypasses the sliding-window feed) so fall_back fires on EVERY chunk -- aggressive F-31-04 stress test; ignores --ab",
    )
    args = ap.parse_args()

    if args.diff:
        args.ab = True

    if args.per_chunk_noise:
        # Special: bypass the sliding-window feed entirely, run SOLA
        # once with per-chunk independent buffers for A and once for
        # B (legacy fade). Skip the standard A run and the --ab loop-
        # reference branch.
        out_a, fb_a = _run_sola_per_chunk_noise(
            rate=args.rate,
            chunk_seconds=args.chunk_seconds,
            duration_s=args.per_chunk_noise,
            use_legacy_fade=False,
        )
        a_path = args.out_dir / "A_vectorised.wav"
        _write_wav(a_path, out_a, args.rate)
        print(
            f"[A: vectorised   ] {a_path}  ({out_a.shape[0] / args.rate:.2f} s, "
            f"fallback_count={fb_a}, per-chunk-noise mode)",
            file=sys.stderr,
        )
        out_c, fb_c = _run_sola_per_chunk_noise(
            rate=args.rate,
            chunk_seconds=args.chunk_seconds,
            duration_s=args.per_chunk_noise,
            use_legacy_fade=True,
        )
        c_path = args.out_dir / "B_legacy_fade.wav"
        _write_wav(c_path, out_c, args.rate)
        print(
            f"[B: legacy fade  ] {c_path}  ({out_c.shape[0] / args.rate:.2f} s, "
            f"fallback_count={fb_c}, per-chunk-noise mode)",
            file=sys.stderr,
        )
        # Quick stats: F-31-04 expectation is that the legacy fade
        # has a measurably LOWER overall RMS (the ~3 dB midpoint dip
        # integrates to a small but real attenuation).
        rms_a = float(np.sqrt(np.mean(out_a * out_a)))
        rms_c = float(np.sqrt(np.mean(out_c * out_c)))
        delta_db = 20.0 * np.log10(rms_c / max(rms_a, 1e-12)) if rms_a > 1e-12 else 0.0
        print(
            f"[F-31-04 power   ] A_rms={rms_a:.4f} B_rms={rms_c:.4f} "
            f"delta={delta_db:+.2f} dB  (legacy fade should be NEGATIVE -- power dip "
            "on fall_back; equal-power preserves)",
            file=sys.stderr,
        )
        return 0

    if args.synthetic_fricatives:
        signal = _gen_synthetic_fricatives(
            args.rate, args.synthetic_fricatives, chunk_seconds=args.chunk_seconds
        )
        src_label = f"synthetic-fricatives {args.synthetic_fricatives:.1f} s @ {args.rate} Hz"
    elif args.synthetic:
        signal = _gen_synthetic_voice(args.rate, args.synthetic)
        src_label = f"synthetic {args.synthetic:.1f} s @ {args.rate} Hz"
    else:
        if args.input is None:
            ap.error("INPUT.wav, --synthetic DURATION, or --synthetic-fricatives DURATION required")
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

    def _diff_stats(a: np.ndarray, b: np.ndarray, label: str, fb_a: int, fb_b: int) -> np.ndarray:
        common = min(a.shape[0], b.shape[0])
        a = a[:common]
        b = b[:common]
        diff = a - b
        diff_max = float(np.max(np.abs(diff))) if common > 0 else 0.0
        diff_rms = float(np.sqrt(np.mean(diff * diff))) if common > 0 else 0.0
        a_rms = float(np.sqrt(np.mean(a * a))) if common > 0 else 0.0
        snr_db = 20.0 * np.log10(a_rms / max(diff_rms, 1e-12)) if a_rms > 1e-12 else float("inf")
        print(
            f"[{label:16s}] max|d|={diff_max:.6f}  rms(d)={diff_rms:.6f}  "
            f"signal-to-diff SNR~{snr_db:.1f} dB  fallback_delta={fb_a - fb_b:+d}",
            file=sys.stderr,
        )
        return diff

    if args.ab:
        out_b, fb_b = _run_sola(
            signal,
            rate=args.rate,
            chunk_seconds=args.chunk_seconds,
            use_loop_reference=True,
        )
        b_path = args.out_dir / "B_loop_reference.wav"
        _write_wav(b_path, out_b, args.rate)
        print(
            f"[B: loop ref     ] {b_path}  ({out_b.shape[0] / args.rate:.2f} s, "
            f"fallback_count={fb_b})",
            file=sys.stderr,
        )
        diff_ab = _diff_stats(out_a, out_b, "A vs B (loop)", fb_a, fb_b)

        if args.diff:
            d_path = args.out_dir / "diff_abs.wav"
            amplified = np.clip(np.abs(diff_ab) * 100.0, -1.0, 1.0).astype(np.float32)
            _write_wav(d_path, amplified, args.rate)
            print(
                f"[diff (x100 amp) ] {d_path}  (audible noise here = behavioural divergence)",
                file=sys.stderr,
            )

    if args.legacy_fade:
        out_c, fb_c = _run_sola(
            signal,
            rate=args.rate,
            chunk_seconds=args.chunk_seconds,
            use_loop_reference=False,
            use_legacy_fade=True,
        )
        c_path = args.out_dir / "B_legacy_fade.wav"
        _write_wav(c_path, out_c, args.rate)
        print(
            f"[B: legacy fade  ] {c_path}  ({out_c.shape[0] / args.rate:.2f} s, "
            f"fallback_count={fb_c})  -- pre-F-31-04 equal-gain Hann on fall_back",
            file=sys.stderr,
        )
        _diff_stats(out_a, out_c, "A vs B (fade)", fb_a, fb_c)
        if fb_a == 0:
            print(
                "[note]              fall_back never fired on this input -- the two "
                "files will be bit-identical. Try fricative-rich audio "
                "(/s/, /sh/, /f/) or --synthetic with noise bursts to exercise "
                "the fall_back branch.",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
