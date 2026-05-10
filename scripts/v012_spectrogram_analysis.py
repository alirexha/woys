#!/usr/bin/env python
"""v0.12.0 Phase 1 - spectrogram analysis of sustained-vowel cuts.

Ingests a WAV recording of woys's WoysSink.monitor while the user
speaks a sustained vowel ("aaaa") and analyzes:

  1. Click detection - find sample positions where energy or
     phase changes abnormally
  2. Inter-click interval distribution - periodicity confirms or
     denies the chunk-boundary hypothesis (chunk_seconds=0.15 →
     150 ms period → 6.67 Hz click rate)
  3. Spectral signature of clicks vs steady state - broadband
     impulse vs f0-harmonic-aligned discontinuity tells us
     whether the cut is amplitude (writer-side) or phase (NSF-reset)
  4. Waveform-inspection windows around predicted chunk boundaries
     - saves PNG snippets for visual confirmation

Usage:
  ./scripts/v012_spectrogram_analysis.py /tmp/woys_aaaa.wav
  ./scripts/v012_spectrogram_analysis.py /tmp/woys_aaaa.wav --chunk-seconds 0.15

Output:
  Console: numeric findings + interpretation
  /tmp/v012_spectrogram.png - full spectrogram with chunk-boundary
                              vertical lines overlaid
  /tmp/v012_click_intervals.png - histogram of inter-click intervals
  /tmp/v012_click_spectrum.png - average spectrum at click vs steady
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np


def _load_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read a WAV file, return (mono_samples, sample_rate). Uses
    soundfile so float32 WAVs (`pw-record --format=f32`) are handled
    natively - Python's stdlib `wave` module rejects them."""
    import soundfile as sf

    samples, sr = sf.read(str(path), always_2d=False, dtype="float32")
    # Auto-merge multi-channel to mono.
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return samples.astype(np.float32, copy=False), int(sr)


def _detect_clicks(audio: np.ndarray, sr: int, *, window_ms: float = 4.0) -> np.ndarray:
    """Click detection via sample-to-sample energy derivative.

    Computes a short-window RMS, then looks at the derivative of that
    RMS. Clicks show up as positive spikes in the derivative (rapid
    energy increase) or alternating spike (energy dip then recover).

    Returns sample indices of detected clicks.
    """
    win = max(8, round(sr * window_ms / 1000.0))
    # Squared envelope.
    sq = audio.astype(np.float64) ** 2
    # Box-filter to get short-term energy.
    env = np.convolve(sq, np.ones(win) / win, mode="same")
    # Derivative.
    d_env = np.abs(np.diff(env, prepend=env[0]))
    # Threshold: 6 sigma above the median absolute deviation. Robust
    # to varying overall loudness within the recording.
    mad = np.median(np.abs(d_env - np.median(d_env)))
    threshold = np.median(d_env) + 6 * 1.4826 * mad
    is_peak = d_env > threshold
    # Group adjacent peaks within `win` samples into one event.
    peak_indices = np.where(is_peak)[0]
    if len(peak_indices) == 0:
        return np.array([], dtype=np.int64)
    grouped = [peak_indices[0]]
    for idx in peak_indices[1:]:
        if idx - grouped[-1] > win * 2:
            grouped.append(idx)
    return np.array(grouped, dtype=np.int64)


def _inter_click_stats(click_idx: np.ndarray, sr: int) -> dict[str, float]:
    if len(click_idx) < 2:
        return {"n": float(len(click_idx))}
    deltas_samples = np.diff(click_idx)
    deltas_ms = deltas_samples / sr * 1000.0
    return {
        "n": float(len(click_idx)),
        "mean_interval_ms": float(np.mean(deltas_ms)),
        "median_interval_ms": float(np.median(deltas_ms)),
        "std_interval_ms": float(np.std(deltas_ms)),
        "min_interval_ms": float(np.min(deltas_ms)),
        "max_interval_ms": float(np.max(deltas_ms)),
        # Coefficient of variation (std/mean) - low CV means very periodic.
        "cv": float(np.std(deltas_ms) / max(np.mean(deltas_ms), 1e-6)),
    }


def _click_rate_hypothesis_table(
    median_interval_ms: float,
    chunk_seconds: float,
) -> list[tuple[str, float, float]]:
    """For the candidate periodic mechanisms, compute predicted period
    and how close the observed interval is. Returns
    [(label, predicted_ms, abs_error_ms)] sorted ascending by error."""
    candidates = [
        ("chunk_seconds (engine cadence)", chunk_seconds * 1000.0),
        ("PipeWire quantum (1024/48000)", 1024.0 / 48000.0 * 1000.0),
        ("watchdog poll (50ms)", 50.0),
        ("torch keepalive (25ms)", 25.0),
        ("ORT keepalive default (25ms)", 25.0),
        ("2x chunk_seconds (subharmonic)", 2 * chunk_seconds * 1000.0),
        ("0.5x chunk_seconds (harmonic)", 0.5 * chunk_seconds * 1000.0),
    ]
    return sorted(
        ((label, pred, abs(median_interval_ms - pred)) for label, pred in candidates),
        key=lambda x: x[2],
    )


def _stft_for_spectrogram(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quick STFT via numpy. Returns (freqs, times, spectrogram_db)."""
    n_fft = 2048
    hop = n_fft // 4
    # Hann window.
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)
    n_frames = max(1, (len(audio) - n_fft) // hop + 1)
    spec = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.float32)
    for i in range(n_frames):
        frame = audio[i * hop : i * hop + n_fft]
        if len(frame) < n_fft:
            break
        windowed = frame * window
        f = np.fft.rfft(windowed)
        spec[:, i] = np.abs(f).astype(np.float32)
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    times = np.arange(n_frames) * hop / sr
    spec_db = 20.0 * np.log10(spec + 1e-12)
    return freqs, times, spec_db


def _click_spectrum(audio: np.ndarray, sr: int, click_indices: np.ndarray) -> np.ndarray:
    """Average FFT magnitude in a short window around each click,
    normalized to the FFT magnitude in equivalent windows from the
    no-click portion. Ratio > 1 at frequency f means the click adds
    energy at f. Returns (freqs, ratio_db)."""
    n_fft = 1024
    half = n_fft // 2
    if len(click_indices) > 0:
        usable = click_indices[(click_indices > half) & (click_indices < len(audio) - half)]
        click_specs = []
        for idx in usable:
            window = audio[idx - half : idx + half]
            click_specs.append(np.abs(np.fft.rfft(window)))
        click_avg = (
            np.mean(np.array(click_specs), axis=0) if click_specs else np.zeros(n_fft // 2 + 1)
        )
    else:
        click_avg = np.zeros(n_fft // 2 + 1)

    # Build "clean" spectrum from windows that are MID-INTERVAL between
    # consecutive clicks (presumed steady-state vowel).
    clean_specs = []
    if len(click_indices) >= 2:
        for prev, curr in itertools.pairwise(click_indices):
            mid = (prev + curr) // 2
            if mid > half and mid < len(audio) - half:
                window = audio[mid - half : mid + half]
                clean_specs.append(np.abs(np.fft.rfft(window)))
    clean_avg = np.mean(np.array(clean_specs), axis=0) if clean_specs else np.ones(n_fft // 2 + 1)

    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    ratio_db = 20 * np.log10((click_avg + 1e-12) / (clean_avg + 1e-12))
    return freqs, ratio_db


def _save_plots(
    audio: np.ndarray,
    sr: int,
    click_idx: np.ndarray,
    chunk_seconds: float,
    out_prefix: Path,
) -> list[Path]:
    """Save visualization PNGs if matplotlib is available; return list
    of generated file paths. Skips silently if matplotlib missing."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not available; skipping visualization", file=sys.stderr)
        return []

    out_files = []

    # 1. Spectrogram with chunk-boundary lines.
    freqs, times, spec_db = _stft_for_spectrogram(audio, sr)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.imshow(
        spec_db[:200, :],  # 0-~5 kHz
        aspect="auto",
        origin="lower",
        cmap="magma",
        extent=[times[0], times[-1], freqs[0], freqs[200]],
        vmin=spec_db[:200].max() - 70,
    )
    # Mark expected chunk boundaries (every chunk_seconds).
    n_boundaries = int(times[-1] / chunk_seconds)
    for k in range(n_boundaries + 1):
        t = k * chunk_seconds
        ax.axvline(t, color="cyan", alpha=0.3, linewidth=0.5)
    # Mark detected clicks.
    for idx in click_idx:
        t = idx / sr
        if t < times[-1]:
            ax.axvline(t, color="lime", alpha=0.5, linewidth=0.7, linestyle="--")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("freq (Hz)")
    ax.set_title(
        f"spectrogram (cyan = chunk_seconds={chunk_seconds}s boundaries; green = detected clicks)"
    )
    p = out_prefix.parent / "v012_spectrogram.png"
    fig.savefig(p, dpi=120, bbox_inches="tight")
    plt.close(fig)
    out_files.append(p)

    # 2. Inter-click interval histogram.
    if len(click_idx) >= 2:
        deltas_ms = np.diff(click_idx) / sr * 1000.0
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(deltas_ms, bins=80, color="steelblue", edgecolor="black", alpha=0.8)
        # Mark predicted boundaries.
        for label, pred_ms in [
            ("chunk", chunk_seconds * 1000.0),
            ("quantum", 1024.0 / 48000.0 * 1000.0),
            ("0.5x chunk", chunk_seconds * 500.0),
            ("2x chunk", chunk_seconds * 2000.0),
        ]:
            if deltas_ms.min() <= pred_ms <= deltas_ms.max():
                ax.axvline(pred_ms, color="red", alpha=0.6, label=f"{label} ({pred_ms:.1f}ms)")
        ax.set_xlabel("inter-click interval (ms)")
        ax.set_ylabel("count")
        ax.set_title(f"inter-click intervals (n={len(click_idx)})")
        ax.legend()
        p = out_prefix.parent / "v012_click_intervals.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        out_files.append(p)

    # 3. Click vs steady spectrum.
    if len(click_idx) >= 2:
        freqs, ratio_db = _click_spectrum(audio, sr, click_idx)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(freqs[: len(freqs) // 2], ratio_db[: len(freqs) // 2], color="darkred")
        ax.axhline(0, color="gray", alpha=0.5, linewidth=0.7)
        ax.set_xlabel("freq (Hz)")
        ax.set_ylabel("click excess over steady (dB)")
        ax.set_title(
            "click vs steady-state spectrum (positive = click adds energy at this frequency)"
        )
        ax.set_xscale("log")
        ax.set_xlim(20, sr / 2)
        ax.grid(True, alpha=0.3)
        p = out_prefix.parent / "v012_click_spectrum.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        out_files.append(p)

    return out_files


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("wav", type=Path, help="recorded WoysSink.monitor WAV")
    parser.add_argument("--chunk-seconds", type=float, default=0.15)
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="skip PNG generation (faster; numbers only)",
    )
    args = parser.parse_args()

    if not args.wav.exists():
        print(f"[error] {args.wav} not found", file=sys.stderr)
        return 1

    audio, sr = _load_wav(args.wav)
    duration_s = len(audio) / sr
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    peak = float(np.abs(audio).max())

    print()
    print("==== v0.12.0 Phase 1 - sustained-vowel spectrogram analysis ====")
    print(f"  file        {args.wav}")
    print(f"  duration    {duration_s:.2f} s")
    print(f"  sample rate {sr} Hz")
    print(f"  channels    {1}  (mono after merge)")
    print(f"  rms         {rms:.4f}")
    print(f"  peak        {peak:.4f}")

    if peak < 0.001:
        print(
            "  [error] recording is essentially silent (peak < 0.001). "
            "Did the engine produce output? Check WoysSink wiring."
        )
        return 2

    click_idx = _detect_clicks(audio, sr)
    stats = _inter_click_stats(click_idx, sr)

    print()
    print("  ---- click detection ----")
    print(f"  n_clicks    {len(click_idx)}")
    if len(click_idx) > 0:
        print(f"  click rate  {len(click_idx) / duration_s:.2f} /sec")
    if len(click_idx) >= 2:
        print(
            f"  interval p50 {stats['median_interval_ms']:.2f} ms  ({1000.0 / stats['median_interval_ms']:.2f} Hz)"
        )
        print(f"  interval p_mean {stats['mean_interval_ms']:.2f} ms")
        print(f"  interval std  {stats['std_interval_ms']:.2f} ms  CV={stats['cv']:.3f}")
        print(
            f"  interval min/max {stats['min_interval_ms']:.2f} / {stats['max_interval_ms']:.2f} ms"
        )

    if len(click_idx) >= 2:
        print()
        print("  ---- mechanism candidates (predicted period vs observed median) ----")
        ranked = _click_rate_hypothesis_table(stats["median_interval_ms"], args.chunk_seconds)
        for label, predicted_ms, error_ms in ranked[:5]:
            marker = "← BEST" if error_ms < 5.0 else ""
            print(
                f"  {label:40s} predicted={predicted_ms:7.2f} ms  error={error_ms:6.2f} ms  {marker}"
            )

    if not args.no_plot and len(click_idx) >= 2:
        out_files = _save_plots(audio, sr, click_idx, args.chunk_seconds, args.wav)
        if out_files:
            print()
            print("  ---- plots ----")
            for p in out_files:
                print(f"  {p}")

    print()
    print("  ---- interpretation ----")
    if len(click_idx) < 5:
        print(f"  too few clicks detected ({len(click_idx)}); detector threshold may be high")
        print("  for this recording's noise floor. Consider repeating with a louder vowel.")
    else:
        median_ms = stats["median_interval_ms"]
        cv = stats["cv"]
        chunk_ms = args.chunk_seconds * 1000.0
        if cv < 0.30:
            periodicity = "STRONGLY PERIODIC"
        elif cv < 0.60:
            periodicity = "moderately periodic"
        else:
            periodicity = "non-periodic / aperiodic"
        print(f"  periodicity: {periodicity} (coefficient of variation = {cv:.2f})")
        if abs(median_ms - chunk_ms) < 5.0:
            print(f"  matched mechanism: chunk_seconds={args.chunk_seconds}s - engine boundary")
            print(
                f"    interval median {median_ms:.1f} ms within 5 ms of chunk_seconds {chunk_ms:.0f} ms"
            )
            print("  → NSF reset at chunk boundaries IS the mechanism, OR another chunk-aligned")
            print("    process (writer thread cadence, SOLA crossfade boundary). Check the")
            print("    spectrum plot: f0-aligned excess = NSF phase reset; broadband excess =")
            print("    amplitude-discontinuity (writer/ring underrun pattern).")
        elif abs(median_ms - chunk_ms / 2) < 5.0:
            print("  matched mechanism: chunk_seconds/2 - half-chunk subharmonic, possibly two")
            print("    distinct artifacts at chunk_seconds spaced offset")
        elif abs(median_ms - chunk_ms * 2) < 5.0:
            print("  matched mechanism: 2x chunk_seconds - every-other-chunk pattern, possibly")
            print("    chunks alternate between 'good' and 'bad' alignment")
        elif abs(median_ms - 1024.0 / 48000.0 * 1000.0) < 1.5:
            print("  matched mechanism: PipeWire quantum boundary - ring-side issue, not engine")
        else:
            print("  no clean match to any predicted mechanism period")
            print(f"  median interval {median_ms:.1f} ms doesn't align to chunk / quantum / etc.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
