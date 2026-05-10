#!/usr/bin/env python
"""v0.12.0 Phase 1 - drive the engine with a STATIONARY voiced tone
(constant 220 Hz + tiny noise) for `--duration` seconds, while
recording WoysSink.monitor concurrently. The constant input has no
internal transitions, so any flux events in the recording must be
ENGINE-side artifacts (chunk boundaries, NSF resets, ring underruns).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "src" / "server") not in sys.path:
    sys.path.insert(0, str(REPO / "src" / "server"))
# Allow `import v010_harness` (same-dir module) without packaging.
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))


def _build_stationary_signal(
    duration_s: float, sr: int, freq_hz: float = 220.0, rms: float = 0.10
) -> np.ndarray:
    """Constant-pitch sine + low pink-ish noise. No transitions, no
    silences. Same RMS the harness uses for its voiced sections."""
    n = round(duration_s * sr)
    t = np.arange(n, dtype=np.float32) / sr
    sine = np.sin(2.0 * math.pi * freq_hz * t).astype(np.float32)
    rng = np.random.default_rng(0xC0FFEE)
    noise = rng.standard_normal(n).astype(np.float32) * 0.5
    accum = 0.0
    for i in range(n):
        accum = 0.97 * accum + 0.03 * noise[i]
        noise[i] = accum
    mix = 0.7 * sine + 0.3 * (noise / max(noise.std(), 1e-6))
    cur = float(np.sqrt(np.mean(mix.astype(np.float64) ** 2)))
    if cur > 0:
        mix = mix * np.float32(rms / cur)
    return mix.astype(np.float32, copy=False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--freq", type=float, default=220.0, help="stationary tone Hz")
    parser.add_argument("--out", type=Path, default=Path("/tmp/v012_stationary.json"))
    parser.add_argument("--anti-jitter-mode", default="both")
    args = parser.parse_args()

    # Patch the harness to use stationary signal instead of the loop.
    import v010_harness as harness

    original_builder = harness._build_signal

    def stationary_builder(duration_s: float, sample_rate: int, *, seed: int = 42) -> np.ndarray:
        return _build_stationary_signal(duration_s, sample_rate, freq_hz=args.freq, rms=0.10)

    harness._build_signal = stationary_builder
    print(
        f"[stationary] driving harness with constant {args.freq} Hz tone for {args.duration:.0f}s",
        file=sys.stderr,
    )

    out = harness._run_engine_synthetic(
        duration_s=args.duration,
        out_path=args.out,
        enable_sola=True,
        chunk_seconds=None,
        inference_subprocess=False,
        gpu_anti_jitter_mode=args.anti_jitter_mode,
    )
    harness._print_summary(out)

    # Restore (not strictly necessary for one-shot script).
    harness._build_signal = original_builder
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
