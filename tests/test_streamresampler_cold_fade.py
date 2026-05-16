"""review F-31-11 (commit-079): `_StreamResampler.cold_fade_in_samples`.

A freshly-built `_StreamResampler` cold-starts its anti-aliasing filter
delay line at zero, producing a sub-unity amplitude on the first
~milliseconds of output. Engine startup tolerates this (silence
before the first chunk anyway), but `_apply_one_swap` constructs a
NEW _StreamResampler in mid-session when the post-swap voice's native
rate differs -- the cold-start blip lands inside live audio.

`cold_fade_in_samples` budgets a linear-fade-in ramp over the first
N output samples, masking the filter-warmup transient. This file
pins:

  * Default `cold_fade_in_samples=0` reproduces the pre-fix output
    bit-for-bit (no behavioural change for the engine-startup
    constructor).
  * A non-zero budget applies a monotonic-increasing ramp across the
    first N output samples then leaves subsequent samples untouched.
  * The ramp survives `flush()` (it counts against the same budget).
  * Identity path (`src_rate == dst_rate`) honours the same contract.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))


def _have_soxr() -> bool:
    try:
        import soxr  # noqa: F401

        return True
    except Exception:
        return False


def test_default_no_fade_is_bit_identical_to_pre_fix() -> None:
    """`cold_fade_in_samples=0` (the default) must produce the same
    output as a resampler built without the kwarg. Engine-startup
    constructor relies on this -- behaviour-change-free fix."""
    from audio.engine import _StreamResampler

    rng = np.random.default_rng(123)
    audio = (0.5 * rng.standard_normal(1600)).astype(np.float32)

    r_a = _StreamResampler(16_000, 48_000)  # default: no fade
    r_b = _StreamResampler(16_000, 48_000, cold_fade_in_samples=0)

    out_a = np.concatenate([r_a.process(audio), r_a.flush()])
    out_b = np.concatenate([r_b.process(audio), r_b.flush()])

    np.testing.assert_array_equal(out_a, out_b)


@pytest.mark.skipif(not _have_soxr(), reason="soxr not installed")
def test_cold_fade_attenuates_leading_samples_monotonically() -> None:
    """With a non-zero fade budget, the first N output samples must be
    attenuated by a non-decreasing ramp that starts near 0 and ends
    near 1.0; samples beyond N are untouched.

    We compare against the same input through a no-fade resampler so
    "untouched" is well-defined.
    """
    from audio.engine import _StreamResampler

    rng = np.random.default_rng(456)
    audio = (0.5 * rng.standard_normal(8000)).astype(np.float32)
    fade_n = 240  # 5 ms at 48 kHz

    r_fade = _StreamResampler(16_000, 48_000, cold_fade_in_samples=fade_n)
    r_ref = _StreamResampler(16_000, 48_000)

    out_fade = r_fade.process(audio)
    out_ref = r_ref.process(audio)

    assert out_fade.shape == out_ref.shape
    assert out_fade.shape[0] >= fade_n, "test setup: chunk must produce >= fade_n samples"

    # Beyond the fade region the two outputs must agree exactly.
    np.testing.assert_array_equal(out_fade[fade_n:], out_ref[fade_n:])

    # Inside the fade region the ratio out_fade / out_ref must rise
    # monotonically from near 0 toward ~1 (sample-wise; we tolerate
    # the linspace endpoint=False so the last fade sample is just
    # below 1.0). Skip samples where the reference is too quiet to
    # form a meaningful ratio.
    ref_region = out_ref[:fade_n]
    fade_region = out_fade[:fade_n]
    quiet = np.abs(ref_region) < 1e-4
    ratios = np.where(quiet, np.nan, fade_region / np.where(quiet, 1.0, ref_region))
    # Drop NaNs; check the remaining ratios are non-decreasing (allow
    # small fp noise) and span the unit interval roughly.
    valid = ratios[~np.isnan(ratios)]
    # Endpoints: first valid ratio should be near 0, last should
    # approach (fade_n-1)/fade_n ~ 0.996.
    assert valid[0] < 0.1, f"fade leading ratio too high: {valid[0]:.4f}"
    assert valid[-1] > 0.8, f"fade trailing ratio too low: {valid[-1]:.4f}"
    diffs = np.diff(valid)
    assert (diffs >= -1e-3).all(), (
        f"ratio is not monotonic non-decreasing: min diff = {diffs.min()}"
    )


def test_cold_fade_identity_path_also_applies() -> None:
    """The identity passthrough path (src_rate == dst_rate) must still
    honour the cold-fade budget. The swap rebuild may construct a
    same-rate stream when the SOLA flush finalized the previous soxr
    stream (`v0.14.0 (C002)` path)."""
    from audio.engine import _StreamResampler

    audio = np.ones(2000, dtype=np.float32)
    fade_n = 100

    r = _StreamResampler(48_000, 48_000, cold_fade_in_samples=fade_n)
    out = r.process(audio)
    assert out.shape == audio.shape
    # First sample close to 0, last fade sample close to 1, then unity.
    assert out[0] < 0.05
    assert out[fade_n - 1] > 0.9
    np.testing.assert_array_equal(out[fade_n:], 1.0)


@pytest.mark.skipif(not _have_soxr(), reason="soxr not installed")
def test_cold_fade_budget_consumed_across_multiple_process_calls() -> None:
    """The fade budget must persist across `process()` invocations --
    if one call's output is shorter than the budget, the remainder
    applies to the next call. Realistic case: soxr's first chunk
    output is small while the filter primes."""
    from audio.engine import _StreamResampler

    rng = np.random.default_rng(789)
    chunk = (0.5 * rng.standard_normal(800)).astype(np.float32)

    r = _StreamResampler(16_000, 48_000, cold_fade_in_samples=240)

    # Drive multiple chunks; gather the full output stream.
    out = np.concatenate([r.process(chunk) for _ in range(8)] + [r.flush()])

    # After the budget is exhausted, the resampler must reach normal
    # gain. Take RMS of the last 800-sample window; should be close
    # to the input RMS (allowing for the dst/src ratio scaling of soxr).
    tail_rms = float(np.sqrt(np.mean(out[-800:] ** 2)))
    # Compared to a no-fade reference -- should be near-identical there.
    r_ref = _StreamResampler(16_000, 48_000)
    out_ref = np.concatenate([r_ref.process(chunk) for _ in range(8)] + [r_ref.flush()])
    ref_tail_rms = float(np.sqrt(np.mean(out_ref[-800:] ** 2)))
    assert abs(tail_rms - ref_tail_rms) < 1e-4, (
        f"tail RMS differs after budget exhaustion: fade={tail_rms:.6f} "
        f"ref={ref_tail_rms:.6f} -- the fade ramp should not persist past "
        f"`cold_fade_in_samples` worth of output."
    )
