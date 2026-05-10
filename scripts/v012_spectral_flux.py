#!/usr/bin/env python
"""v0.12.0 Phase 1 - spectral-flux click detection.

The simple energy-derivative detector over-fires on glottal pulses
during voiced content (one peak per f0 cycle ≈ every ~8 ms at typical
male voice). Chunk-boundary clicks at chunk_seconds=0.15 → 6.67 Hz =
150 ms cadence get drowned out.

Spectral flux is the right discriminator:

  flux[i] = Σ_k max(0, |X[i, k]| - |X[i-1, k]|)

It measures positive-going spectral change frame-to-frame. Glottal
pulses on voiced content produce SMOOTH spectral evolution (low
flux); chunk-boundary phase discontinuities produce ABRUPT flux
spikes that survive the per-frame normalization.

Usage:
  ./scripts/v012_spectral_flux.py /tmp/v012_synth.wav
  ./scripts/v012_spectral_flux.py /tmp/v012_synth.wav --chunk-seconds 0.15
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _load_wav(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    samples, sr = sf.read(str(path), always_2d=False, dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return samples.astype(np.float32, copy=False), int(sr)


def _spectral_flux(
    audio: np.ndarray, sr: int, *, hop_ms: float = 5.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return (flux_values, flux_times_seconds) for the audio."""
    n_fft = 1024
    hop = max(8, round(sr * hop_ms / 1000.0))
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)
    n_frames = max(1, (len(audio) - n_fft) // hop + 1)
    if n_frames < 2:
        return np.zeros(0), np.zeros(0)
    spec = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.float32)
    for i in range(n_frames):
        frame = audio[i * hop : i * hop + n_fft]
        if len(frame) < n_fft:
            break
        windowed = frame * window
        spec[:, i] = np.abs(np.fft.rfft(windowed)).astype(np.float32)
    diff = spec[:, 1:] - spec[:, :-1]
    flux = np.sum(np.maximum(diff, 0.0), axis=0)
    times = (np.arange(len(flux)) + 1) * hop / sr
    return flux, times


def _peak_pick(
    flux: np.ndarray, times: np.ndarray, *, n_sigma: float = 4.0, min_gap_ms: float = 50.0
) -> np.ndarray:
    """Return time positions (seconds) of flux peaks above
    median + n_sigma * MAD, with a minimum gap suppressor so a
    single broad event doesn't count as multiple peaks."""
    if len(flux) == 0:
        return np.zeros(0)
    median = np.median(flux)
    mad = np.median(np.abs(flux - median))
    threshold = median + n_sigma * 1.4826 * mad
    is_peak = (flux > threshold) & (flux > np.maximum(np.roll(flux, 1), np.roll(flux, -1)))
    candidates = times[is_peak]
    if len(candidates) == 0:
        return candidates
    # Enforce min_gap_ms suppressor.
    accepted = [candidates[0]]
    min_gap_s = min_gap_ms / 1000.0
    for t in candidates[1:]:
        if t - accepted[-1] >= min_gap_s:
            accepted.append(t)
    return np.array(accepted)


def _periodicity_test(
    intervals_ms: np.ndarray, target_ms: float, tolerance_ms: float = 8.0
) -> dict[str, float]:
    """Quantitative test: what fraction of intervals fall within
    `tolerance_ms` of `target_ms` (or its harmonics 2x, 3x)?"""
    if len(intervals_ms) == 0:
        return {"frac_at_target": 0.0, "frac_at_2x": 0.0, "frac_at_3x": 0.0}
    n = len(intervals_ms)
    return {
        "frac_at_target": float(np.sum(np.abs(intervals_ms - target_ms) < tolerance_ms) / n),
        "frac_at_2x": float(np.sum(np.abs(intervals_ms - 2 * target_ms) < tolerance_ms) / n),
        "frac_at_3x": float(np.sum(np.abs(intervals_ms - 3 * target_ms) < tolerance_ms) / n),
    }


def _autocorrelation_test(
    peak_times: np.ndarray, max_lag_ms: float = 1000.0, sr: int = 48000
) -> tuple[np.ndarray, np.ndarray]:
    """Compute autocorrelation of the click-impulse train. If clicks
    are periodic at chunk_seconds, autocorrelation peaks at
    chunk_seconds, 2*chunk_seconds, ... A flat autocorrelation means
    aperiodic.

    Returns (lag_ms_axis, autocorr) for plotting / inspection.
    """
    if len(peak_times) < 4:
        return np.zeros(0), np.zeros(0)
    duration_s = peak_times[-1] + 0.5
    bin_ms = 1.0
    n_bins = int(duration_s * 1000.0 / bin_ms)
    impulse = np.zeros(n_bins, dtype=np.float32)
    for t in peak_times:
        bin_idx = int(t * 1000.0 / bin_ms)
        if 0 <= bin_idx < n_bins:
            impulse[bin_idx] = 1.0
    # Autocorrelation, lag 0 → max_lag_ms.
    max_lag = int(max_lag_ms / bin_ms)
    ac = np.correlate(impulse, impulse, mode="full")
    center = len(ac) // 2
    ac = ac[center : center + max_lag + 1]
    lags = np.arange(len(ac)) * bin_ms
    # Normalize.
    if ac[0] > 0:
        ac = ac / ac[0]
    return lags, ac


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("wav", type=Path)
    parser.add_argument("--chunk-seconds", type=float, default=0.15)
    parser.add_argument("--hop-ms", type=float, default=5.0)
    parser.add_argument("--n-sigma", type=float, default=4.0)
    parser.add_argument("--min-gap-ms", type=float, default=50.0)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    audio, sr = _load_wav(args.wav)
    duration_s = len(audio) / sr
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    print("==== v0.12.0 Phase 1 - spectral-flux click detection ====")
    print(f"  file        {args.wav}")
    print(f"  duration    {duration_s:.2f} s   sr={sr}   rms={rms:.4f}")

    flux, times = _spectral_flux(audio, sr, hop_ms=args.hop_ms)
    if len(flux) == 0:
        print("  [error] flux empty, file too short")
        return 1
    print("\n  ---- spectral flux ----")
    print(f"  frames={len(flux)}  hop_ms={args.hop_ms}")
    print(
        f"  flux median={np.median(flux):.4f}  p95={np.percentile(flux, 95):.4f}  max={flux.max():.4f}"
    )

    peaks = _peak_pick(flux, times, n_sigma=args.n_sigma, min_gap_ms=args.min_gap_ms)
    print(f"  detected peaks: {len(peaks)}  rate={len(peaks) / duration_s:.2f}/sec")

    if len(peaks) >= 4:
        intervals = np.diff(peaks) * 1000.0
        print(f"  interval median={np.median(intervals):.2f} ms")
        print(f"  interval mean={np.mean(intervals):.2f} ms  std={np.std(intervals):.2f} ms")
        print(f"  interval CV={np.std(intervals) / max(np.mean(intervals), 1e-6):.3f}")
        print(
            f"  interval p25/p50/p75={np.percentile(intervals, 25):.1f} / "
            f"{np.percentile(intervals, 50):.1f} / "
            f"{np.percentile(intervals, 75):.1f} ms"
        )

        chunk_ms = args.chunk_seconds * 1000.0
        per_test = _periodicity_test(intervals, chunk_ms)
        print(f"\n  ---- periodicity test (chunk={chunk_ms:.0f}ms ± 8ms tolerance) ----")
        print(f"  fraction of intervals at chunk_seconds:    {per_test['frac_at_target']:.1%}")
        print(f"  fraction of intervals at 2x chunk_seconds: {per_test['frac_at_2x']:.1%}")
        print(f"  fraction of intervals at 3x chunk_seconds: {per_test['frac_at_3x']:.1%}")

    if len(peaks) >= 4:
        lags, ac = _autocorrelation_test(peaks, max_lag_ms=1500.0, sr=sr)
        print("\n  ---- autocorrelation peaks (top 10 lags excluding 0) ----")
        # Find peaks in autocorrelation (exclude small lags <= 30ms which
        # can be self-similar from broad peak detection).
        searchable = ac.copy()
        searchable[:30] = 0  # mask lag 0-30ms
        top_idx = np.argsort(searchable)[::-1]
        for i in top_idx[:10]:
            print(f"  lag={lags[i]:6.1f} ms  autocorr={ac[i]:.3f}")

    if not args.no_plot:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[plot] matplotlib not available")
        else:
            # Plot flux over time with chunk-boundary lines + detected peaks
            fig, ax = plt.subplots(figsize=(14, 4))
            ax.plot(times, flux, color="steelblue", linewidth=0.5, alpha=0.7)
            n_boundaries = int(times[-1] / args.chunk_seconds)
            for k in range(n_boundaries + 1):
                t = k * args.chunk_seconds
                ax.axvline(t, color="cyan", alpha=0.2, linewidth=0.4)
            for p in peaks:
                ax.axvline(p, color="lime", alpha=0.5, linewidth=0.7, linestyle="--")
            ax.set_xlabel("time (s)")
            ax.set_ylabel("spectral flux")
            ax.set_title(
                f"spectral flux (n={len(peaks)} peaks; chunk_seconds={args.chunk_seconds}s = cyan grid)"
            )
            p = args.wav.parent / f"{args.wav.stem}_flux.png"
            fig.savefig(p, dpi=120, bbox_inches="tight")
            plt.close(fig)
            print(f"\n  flux plot: {p}")

            # Autocorrelation plot
            if len(peaks) >= 4:
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(lags, ac, color="darkred", linewidth=1)
                ax.axhline(0, color="gray", alpha=0.4, linewidth=0.5)
                # Mark expected chunk_seconds and harmonics
                for k in range(1, 6):
                    t = k * args.chunk_seconds * 1000.0
                    if t < lags[-1]:
                        ax.axvline(
                            t,
                            color="cyan",
                            alpha=0.5,
                            linestyle="--",
                            label=f"{k}xchunk={t:.0f}ms" if k <= 3 else None,
                        )
                ax.set_xlabel("lag (ms)")
                ax.set_ylabel("normalized autocorrelation")
                ax.set_title(
                    "impulse-train autocorrelation (peak at chunk_seconds = positive periodicity test)"
                )
                ax.legend()
                p = args.wav.parent / f"{args.wav.stem}_autocorr.png"
                fig.savefig(p, dpi=120, bbox_inches="tight")
                plt.close(fig)
                print(f"  autocorr plot: {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
