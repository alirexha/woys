"""SOLA crossfade unit tests.

These verify the building blocks (Hann windows, cross-correlation alignment,
state machine). The end-to-end "no audible artifact at chunk boundary" test
requires the real engine and lives in `test_engine_sola_*` (Phase B integ).
"""

from __future__ import annotations

import numpy as np

from audio.sola import SOLAConfig, SOLAStream, _best_offset, _hann_fade


def test_hann_fade_sums_to_one() -> None:
    fo, fi = _hann_fade(64)
    assert fo.shape == (64,)
    assert fi.shape == (64,)
    # Hann pair must sum to 1 everywhere - that's the whole point.
    assert np.allclose(fo + fi, 1.0, atol=1e-6)
    # Endpoints: starts at (1, 0), ends near (0, 1).
    assert fo[0] > 0.99 and fi[0] < 0.01


def test_hann_fade_zero_size() -> None:
    fo, fi = _hann_fade(0)
    assert fo.shape == (0,) and fi.shape == (0,)


def test_best_offset_finds_aligned_shift() -> None:
    """Synthesize a tail, place a copy of it at a known shift inside a
    longer 'head' buffer, recover the shift via _best_offset.

    v0.7.0-rc5: search range is now `[0, search]` (one-sided) to match
    upstream w-okada's contract. true_shift values are in that range.
    """
    rng = np.random.default_rng(seed=42)
    overlap = 128
    search = 16

    tail = rng.standard_normal(overlap).astype(np.float32)
    head_len = overlap + search
    head = np.zeros(head_len, dtype=np.float32)

    for true_shift in (0, 3, 7, 11, 16):
        head[:] = rng.standard_normal(head_len).astype(np.float32) * 0.05  # background noise
        head[true_shift : true_shift + overlap] = tail
        recovered, fell_back = _best_offset(tail, head, search=search, threshold=0.5)
        assert recovered == true_shift, f"expected {true_shift}, got {recovered}"
        assert not fell_back, f"unexpected threshold-fallback at shift {true_shift}"


def test_best_offset_below_threshold_returns_zero() -> None:
    """Pure noise shouldn't pick a non-zero offset; should signal fallback."""
    rng = np.random.default_rng(seed=42)
    overlap = 128
    search = 16
    tail = rng.standard_normal(overlap).astype(np.float32)
    head = rng.standard_normal(overlap + search).astype(np.float32)
    # Threshold high enough that random noise won't beat it.
    offset, fell_back = _best_offset(tail, head, search=search, threshold=0.9)
    assert offset == 0
    assert fell_back


def test_best_offset_silent_tail() -> None:
    """A near-silent prev_tail can't drive correlation; expect 0 + fallback."""
    overlap = 64
    search = 8
    tail = np.zeros(overlap, dtype=np.float32) + 1e-9
    head = np.random.default_rng(0).standard_normal(overlap + search).astype(np.float32)
    offset, fell_back = _best_offset(tail, head, search=search, threshold=0.1)
    assert offset == 0
    assert fell_back


def test_sola_first_chunk_emits_chunk_n() -> None:
    """v0.7.0-rc5 contract: input is sized chunk_n + cf + search; output
    on every call is exactly chunk_n samples. First chunk emits the
    leading chunk_n samples; the trailing cf samples become prev_tail.
    The trailing `search` slack is unused on first chunk."""
    cfg = SOLAConfig(rate=16_000, crossfade_ms=20.0, search_ms=4.0)
    sola = SOLAStream(cfg)
    cf = cfg.crossfade_samples
    search = cfg.search_samples
    chunk_n = 1600
    inp = np.ones(chunk_n + cf + search, dtype=np.float32)

    out = sola.process(inp)

    assert out.shape == (chunk_n,), f"first-chunk emit {out.shape[0]} != chunk_n {chunk_n}"
    # prev_tail saved from the END of input (last cf samples).
    assert sola._prev_tail is not None
    assert sola._prev_tail.shape == (cf,)


def test_sola_no_nan_or_clip_on_random_stream() -> None:
    """Process a random-noise stream - output must stay finite and bounded
    by the input's amplitude (Hann fades sum to 1, so no gain explosion)."""
    rng = np.random.default_rng(seed=1)
    cfg = SOLAConfig(rate=16_000, crossfade_ms=30.0, search_ms=3.0)
    sola = SOLAStream(cfg)
    cf = cfg.crossfade_samples
    search = cfg.search_samples

    chunk_n = 1600 - cf - search  # adjust so chunk_n stays within sensible range
    chunk_sz = chunk_n + cf + search
    pieces = []
    for _ in range(20):
        chunk = (0.5 * rng.standard_normal(chunk_sz)).astype(np.float32)
        out = sola.process(chunk)
        pieces.append(out)
    pieces.append(sola.flush())
    full = np.concatenate(pieces)
    assert np.isfinite(full).all()
    # Hann pair sums to 1 → max output amplitude can't exceed max input amplitude.
    # We use 0.5 input with stdev 0.5 so max input ≈ 2-3; output should sit in that range.
    assert np.abs(full).max() <= 5.0


def test_sola_emit_length_constant_across_offsets() -> None:
    """v0.7.0-rc5 invariant: SOLA emits chunk_n samples regardless of
    which alignment offset wins. Pre-rc5, output length depended on
    offset (variable, draining the downstream buffer); rc5's upstream-
    style contract makes output length fixed by sizing the input to
    chunk_n + cf + search and sliding the emit window inside the slack.
    This test pins the invariant against threshold-fallback (worst case)
    and against forced-correlation-success (best case)."""
    cfg_fb = SOLAConfig(rate=16_000, crossfade_ms=20.0, search_ms=4.0, corr_threshold=0.99)
    cfg_ok = SOLAConfig(rate=16_000, crossfade_ms=20.0, search_ms=4.0, corr_threshold=0.5)
    cf = cfg_fb.crossfade_samples
    search = cfg_fb.search_samples
    chunk_n = 800
    chunk_sz = chunk_n + cf + search

    rng = np.random.default_rng(seed=11)

    # Threshold-fallback case: every subsequent chunk has corr below 0.99,
    # so _best_offset returns offset=0 + fell_back=True.
    sola_fb = SOLAStream(cfg_fb)
    sola_fb.process(rng.standard_normal(chunk_sz).astype(np.float32))  # warmup
    out_fb = sola_fb.process(rng.standard_normal(chunk_sz).astype(np.float32))
    assert out_fb.shape == (chunk_n,), (
        f"fallback emit {out_fb.shape[0]} != chunk_n {chunk_n}; "
        "the rc4 zero-pad regression must not return."
    )
    assert sola_fb.fallback_count == 1

    # Forced-success case: prev_tail seeded with a known buffer; chunk
    # built so offset=0 has correlation 1.0 (uniquely beats threshold).
    sola_ok = SOLAStream(cfg_ok)
    seeded_tail = rng.standard_normal(cf).astype(np.float32)
    sola_ok._prev_tail = seeded_tail.copy()
    chunk = np.empty(chunk_sz, dtype=np.float32)
    chunk[:cf] = seeded_tail  # offset=0 match
    chunk[cf:] = rng.standard_normal(chunk_sz - cf).astype(np.float32)
    out_ok = sola_ok.process(chunk)
    assert out_ok.shape == (chunk_n,), f"success emit {out_ok.shape[0]} != chunk_n {chunk_n}"
    assert sola_ok.fallback_count == 0


def test_sola_emit_is_signal_not_zeros() -> None:
    """v0.7.0-rc5 contract guarantees output length without padding silence.
    A non-silent input must produce a non-silent emit - there must be NO
    trailing zero-pad region. This is the test that would have caught the
    rc4 zero-pad regression: rc4 emitted chunk_n samples but with a known-
    silence suffix, producing audible cuts. rc5 must not."""
    cfg = SOLAConfig(rate=16_000, crossfade_ms=20.0, search_ms=4.0, corr_threshold=0.99)
    sola = SOLAStream(cfg)
    cf = cfg.crossfade_samples
    search = cfg.search_samples
    chunk_n = 800
    chunk_sz = chunk_n + cf + search

    rng = np.random.default_rng(seed=13)
    # Warmup with random noise (creates prev_tail).
    sola.process(rng.standard_normal(chunk_sz).astype(np.float32))

    # Non-silent chunk - every sample is loud noise, no zeros.
    loud_chunk = (0.5 * rng.standard_normal(chunk_sz)).astype(np.float32)
    out = sola.process(loud_chunk)

    assert out.shape == (chunk_n,)
    # Tail of emit (the last `search` samples) must not be all-zeros.
    # Pre-rc5's pad-on-fallback path would have produced exactly that.
    tail_rms = float(np.sqrt(np.mean(out[-search:].astype(np.float64) ** 2)))
    assert tail_rms > 1e-3, (
        f"emit tail RMS={tail_rms:.6f} suggests zero-pad regression; "
        "rc5 must emit real signal across the full chunk_n window."
    )
    # Whole-emit RMS should be in a reasonable range for unit-scale noise input.
    full_rms = float(np.sqrt(np.mean(out.astype(np.float64) ** 2)))
    assert full_rms > 1e-3


def test_sola_reset_clears_fallback_counter() -> None:
    """reset() must wipe the fallback counter alongside prev_tail so
    post-restart sessions don't carry stale numbers into woys diag."""
    cfg = SOLAConfig(rate=16_000, crossfade_ms=20.0, search_ms=4.0, corr_threshold=0.99)
    sola = SOLAStream(cfg)
    cf = cfg.crossfade_samples
    search = cfg.search_samples
    chunk_sz = 800 + cf + search
    rng = np.random.default_rng(seed=3)
    sola.process(rng.standard_normal(chunk_sz).astype(np.float32))
    sola.process(rng.standard_normal(chunk_sz).astype(np.float32))
    assert sola.fallback_count > 0
    sola.reset()
    assert sola.fallback_count == 0
    assert sola._prev_tail is None


def test_sola_flush_emits_held_tail() -> None:
    cfg = SOLAConfig(rate=16_000, crossfade_ms=10.0, search_ms=1.0)
    sola = SOLAStream(cfg)
    cf = cfg.crossfade_samples
    search = cfg.search_samples
    chunk_sz = 4 * cf + search  # well above the warmup threshold
    chunk = np.ones(chunk_sz, dtype=np.float32)
    sola.process(chunk)
    flushed = sola.flush()
    assert flushed.shape == (cf,)
    # After flush, the buffer is cleared.
    assert sola._prev_tail is None
    assert sola.flush().shape == (0,)
