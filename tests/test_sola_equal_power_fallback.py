"""equal-power crossfade on the
SOLA fall_back branch only; equal-gain (Hann²) on the aligned branch.

The aligned branch sees prev_tail and head sharing phase (the
alignment search picked an in-phase overlap), so they add coherently
in amplitude -- amplitude-summing Hann² is the correct power model
and preserves perceived loudness.

The fall_back branch sees prev_tail and head effectively
uncorrelated (the alignment search gave up below threshold), so
power adds incoherently -- equal-gain Hann² produces a `cos**4 +
sin**4` power envelope that dips ~3 dB at midpoint; equal-power
(`cos`/`sin` weights with `cos**2 + sin**2 == 1`) keeps the power
flat across the fade. Audible on fricatives / sibilants / phoneme
transitions where fall_back fires most.

The review-required test material (review line 896): "RMS of the
crossfade region on two uncorrelated noise chunks + a fricative
listener pass". The fricative pass is the maintainer's
ears-verify task; the unit tests here cover the RMS math.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "src" / "server") not in sys.path:
    sys.path.insert(0, str(REPO / "src" / "server"))


# ---------------------------------------------------------------------------
# Fade pairs: mathematical properties
# ---------------------------------------------------------------------------


def test_hann_fade_sums_to_one_pointwise() -> None:
    """The equal-gain pair: `fade_out + fade_in == 1` pointwise.
    This is the amplitude-preserving Hann² crossfade — preserves
    amplitude when the two crossfaded signals share phase."""
    from audio.sola import _hann_fade

    for n in (1, 16, 64, 800, 1024):
        fo, fi = _hann_fade(n)
        s = fo + fi
        # fp32 tolerance: cos^2 + sin^2 should be 1 within ~1e-7.
        assert np.allclose(s, 1.0, atol=1e-6), (
            f"_hann_fade({n}): fade_out + fade_in deviates from 1; max diff "
            f"{float(np.max(np.abs(s - 1.0))):.3e}"
        )


def test_equal_power_fade_squared_sums_to_one_pointwise() -> None:
    """The equal-power pair: `fade_out**2 + fade_in**2 == 1`
    pointwise. This is the energy-preserving cos/sin crossfade —
    preserves total power when the two crossfaded signals are
    uncorrelated (powers add)."""
    from audio.sola import _equal_power_fade

    for n in (1, 16, 64, 800, 1024):
        fo, fi = _equal_power_fade(n)
        ss = fo * fo + fi * fi
        assert np.allclose(ss, 1.0, atol=1e-6), (
            f"_equal_power_fade({n}): fade_out**2 + fade_in**2 deviates from 1; "
            f"max diff {float(np.max(np.abs(ss - 1.0))):.3e}"
        )


def test_equal_power_fade_zero_n_returns_empty_pair() -> None:
    from audio.sola import _equal_power_fade

    fo, fi = _equal_power_fade(0)
    assert fo.shape == (0,) and fi.shape == (0,)
    assert fo.dtype == np.float32 and fi.dtype == np.float32


def test_hann_and_equal_power_are_distinct_at_midpoint() -> None:
    """Sanity: at the midpoint (k = n // 2) the two pairs disagree.
    Specifically, `_hann_fade` gives ~0.5/0.5 (equal-gain) while
    `_equal_power_fade` gives ~sqrt(0.5)/sqrt(0.5) ≈ 0.707/0.707.
    If they were the same we'd have nothing to fix."""
    from audio.sola import _equal_power_fade, _hann_fade

    n = 800
    eg_out, eg_in = _hann_fade(n)
    ep_out, ep_in = _equal_power_fade(n)
    mid = n // 2
    # Equal-gain midpoint amplitudes: ~0.5 each.
    assert 0.45 <= eg_out[mid] <= 0.55
    assert 0.45 <= eg_in[mid] <= 0.55
    # Equal-power midpoint amplitudes: ~0.707 each.
    assert 0.65 <= ep_out[mid] <= 0.75
    assert 0.65 <= ep_in[mid] <= 0.75


# ---------------------------------------------------------------------------
# SOLAStream: branch selection on fell_back
# ---------------------------------------------------------------------------


def _run_independent_chunks(*, n_chunks: int, use_legacy_fade: bool, seed: int = 0) -> tuple:
    """Feed `SOLAStream.process` independent random buffers per
    chunk. This bypasses the normal sliding-window feed where
    prev_tail and head share input samples (and correlate at 1 by
    construction), so fall_back fires every chunk after the first.

    Returns `(output, fallback_count, sola_stream)`.
    """
    from audio import sola

    cfg = sola.SOLAConfig()
    stream = sola.SOLAStream(cfg)
    cf = cfg.crossfade_samples
    search = cfg.search_samples
    chunk_n = round(cfg.rate * 0.25)
    feed_len = chunk_n + cf + search
    rng = np.random.default_rng(seed)

    original = sola._USE_EQUAL_POWER_ON_FALLBACK
    if use_legacy_fade:
        sola._USE_EQUAL_POWER_ON_FALLBACK = False
    try:
        outputs: list[np.ndarray] = []
        for _ in range(n_chunks):
            buf = rng.standard_normal(feed_len).astype(np.float32) * 0.3
            emit = stream.process(buf)
            if emit.size > 0:
                outputs.append(emit)
        out = (
            np.concatenate(outputs).astype(np.float32) if outputs else np.zeros(0, dtype=np.float32)
        )
        return out, stream.fallback_count, stream
    finally:
        sola._USE_EQUAL_POWER_ON_FALLBACK = original


def test_fallback_fires_when_chunks_are_independent() -> None:
    """Sanity setup test: with independent per-chunk noise buffers,
    SOLA's alignment search drops below threshold and falls back on
    every chunk after the first. If this ever returns 0 the F-31-04
    branch-selection tests below are silently never exercising the
    equal-power path."""
    _, fb, _ = _run_independent_chunks(n_chunks=10, use_legacy_fade=False)
    assert fb >= 9, f"expected fall_back on ~every chunk past the first; got {fb}/9"


def test_equal_power_preserves_power_better_than_equal_gain() -> None:
    """F-31-04 review 'RMS of the crossfade region on two
    uncorrelated noise chunks' (line 896). Compare integrated RMS
    of the SOLA output between equal-power (production) and
    equal-gain (legacy). The legacy path should be measurably
    quieter -- the ~3 dB midpoint dip integrates to ~0.2 dB across a
    chunk_seconds=0.25 / crossfade_ms=50 SOLA window."""
    out_ep, fb_ep, _ = _run_independent_chunks(n_chunks=16, use_legacy_fade=False)
    out_eg, fb_eg, _ = _run_independent_chunks(n_chunks=16, use_legacy_fade=True)
    assert fb_ep == fb_eg, "fall_back count must be identical between A and B runs"
    assert fb_ep >= 15

    rms_ep = float(np.sqrt(np.mean(out_ep * out_ep)))
    rms_eg = float(np.sqrt(np.mean(out_eg * out_eg)))
    delta_db = 20.0 * np.log10(rms_eg / max(rms_ep, 1e-12))
    # Equal-power must be at least 0.1 dB louder than equal-gain on
    # the same independent-noise stream (the legacy fade dips, the
    # equal-power fade preserves). The exact figure floats with the
    # noise seed; require a clear sign and a measurable magnitude.
    assert delta_db < -0.1, (
        f"legacy equal-gain fade should be at least 0.1 dB quieter than "
        f"equal-power on uncorrelated chunks; got delta={delta_db:+.3f} dB "
        f"(rms_ep={rms_ep:.4f} rms_eg={rms_eg:.4f})"
    )


def test_aligned_branch_unchanged_by_commit_078() -> None:
    """The aligned branch (when `fell_back == False`) still uses the
    equal-GAIN Hann pair. Commit-078 must NOT touch correlated-content
    behaviour. We feed SOLA the same input twice -- once with the
    production flag on, once with legacy on -- and assert byte-
    identical output, since on aligned content the legacy branch
    path is taken in both cases."""
    from audio import sola

    cfg = sola.SOLAConfig()
    cf = cfg.crossfade_samples
    search = cfg.search_samples
    chunk_n = round(cfg.rate * 0.25)
    feed_len = chunk_n + cf + search

    # Build a long sliding-window signal so adjacent chunks share
    # their overlap region (the SOLA-correct setup) -- alignment
    # search will succeed, fall_back stays at 0.
    rng = np.random.default_rng(7)
    total = chunk_n * 12 + cf + search
    signal = rng.standard_normal(total).astype(np.float32) * 0.3

    def _run(flag: bool) -> tuple[np.ndarray, int]:
        original = sola._USE_EQUAL_POWER_ON_FALLBACK
        sola._USE_EQUAL_POWER_ON_FALLBACK = flag
        try:
            stream = sola.SOLAStream(cfg)
            outs: list[np.ndarray] = []
            cursor = 0
            while cursor + feed_len <= signal.shape[0]:
                emit = stream.process(signal[cursor : cursor + feed_len])
                if emit.size > 0:
                    outs.append(emit)
                cursor += chunk_n
            return np.concatenate(outs).astype(np.float32), stream.fallback_count
        finally:
            sola._USE_EQUAL_POWER_ON_FALLBACK = original

    out_prod, fb_prod = _run(True)
    out_legacy, fb_legacy = _run(False)

    # On sliding-window signal the alignment search succeeds -- so the
    # fall_back branch should NOT fire, and the two outputs must be
    # bit-identical (both took the equal-gain branch).
    assert fb_prod == fb_legacy == 0, (
        f"this test requires no fall_back on the aligned branch; got "
        f"prod fb={fb_prod}, legacy fb={fb_legacy}"
    )
    assert np.array_equal(out_prod, out_legacy), (
        "aligned-branch output must be bit-identical between flag states; "
        f"max diff = {float(np.max(np.abs(out_prod - out_legacy))):.6f}"
    )
