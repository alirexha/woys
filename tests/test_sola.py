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
    # Hann pair must sum to 1 everywhere — that's the whole point.
    assert np.allclose(fo + fi, 1.0, atol=1e-6)
    # Endpoints: starts at (1, 0), ends near (0, 1).
    assert fo[0] > 0.99 and fi[0] < 0.01


def test_hann_fade_zero_size() -> None:
    fo, fi = _hann_fade(0)
    assert fo.shape == (0,) and fi.shape == (0,)


def test_best_offset_finds_aligned_shift() -> None:
    """Synthesize a signal, take its tail, and place a shifted copy of that
    tail inside a longer 'head' buffer. _best_offset should recover the shift
    and not flag the result as a threshold-fallback."""
    rng = np.random.default_rng(seed=42)
    overlap = 128
    search = 16

    tail = rng.standard_normal(overlap).astype(np.float32)
    head_len = overlap + 2 * search
    head = np.zeros(head_len, dtype=np.float32)

    for true_shift in (-12, -5, 0, 3, 11):
        # Place the tail at index (search + true_shift) inside head.
        head[:] = rng.standard_normal(head_len).astype(np.float32) * 0.05  # background noise
        head[search + true_shift : search + true_shift + overlap] = tail
        recovered, fell_back = _best_offset(tail, head, search=search, threshold=0.5)
        assert recovered == true_shift, f"expected {true_shift}, got {recovered}"
        assert not fell_back, f"unexpected threshold-fallback at shift {true_shift}"


def test_best_offset_below_threshold_returns_zero() -> None:
    """Pure noise shouldn't pick a non-zero offset; should signal fallback."""
    rng = np.random.default_rng(seed=42)
    overlap = 128
    search = 16
    tail = rng.standard_normal(overlap).astype(np.float32)
    head = rng.standard_normal(overlap + 2 * search).astype(np.float32)
    # Threshold high enough that random noise won't beat it.
    offset, fell_back = _best_offset(tail, head, search=search, threshold=0.9)
    assert offset == 0
    assert fell_back


def test_best_offset_silent_tail() -> None:
    """A near-silent prev_tail can't drive correlation; expect 0 + fallback."""
    overlap = 64
    search = 8
    tail = np.zeros(overlap, dtype=np.float32) + 1e-9
    head = np.random.default_rng(0).standard_normal(overlap + 2 * search).astype(np.float32)
    offset, fell_back = _best_offset(tail, head, search=search, threshold=0.1)
    assert offset == 0
    assert fell_back


def test_sola_first_chunk_holds_back_tail() -> None:
    """First chunk: the trailing crossfade region is held back, not emitted."""
    cfg = SOLAConfig(rate=16_000, crossfade_ms=20.0, search_ms=2.0)
    sola = SOLAStream(cfg)
    cf = cfg.crossfade_samples
    chunk = np.ones(2 * cf, dtype=np.float32)
    out = sola.process(chunk)
    assert out.shape == (cf,)
    # Tail saved internally.
    assert sola._prev_tail is not None
    assert sola._prev_tail.shape == (cf,)


def test_sola_no_nan_or_clip_on_random_stream() -> None:
    """Process a random-noise stream — output must stay finite and bounded
    by the input's amplitude (Hann fades sum to 1, so no gain explosion)."""
    rng = np.random.default_rng(seed=1)
    cfg = SOLAConfig(rate=16_000, crossfade_ms=30.0, search_ms=3.0)
    sola = SOLAStream(cfg)

    chunk_sz = 1600
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


def test_sola_pads_fallback_shortfall_to_input_length() -> None:
    """v0.7.0-rc4 — the per-call output must be (input - cf) samples
    regardless of which offset _best_offset picks. Pre-rc4, fallback
    chunks emitted (input - cf - search) samples, draining the
    downstream output buffer at ~7 ms/sec at chunk=0.15 with 18 %
    fallback rate (audit lens 03). rc4 zero-pads the shortfall so the
    output stream stays length-stable; this test pins that behavior."""
    rng = np.random.default_rng(seed=7)
    # Use threshold=0.99 so every chunk falls back; that's the worst case.
    cfg = SOLAConfig(rate=16_000, crossfade_ms=20.0, search_ms=4.0, corr_threshold=0.99)
    sola = SOLAStream(cfg)
    cf = cfg.crossfade_samples

    chunk_sz = 800  # 50 ms at 16 kHz; well above 2*cf + 2*search
    # First chunk: held back, no fallback path exercised.
    sola.process(rng.standard_normal(chunk_sz).astype(np.float32))
    assert sola.fallback_count == 0

    # Subsequent chunks: every one should fall back (decorrelated noise).
    out = sola.process(rng.standard_normal(chunk_sz).astype(np.float32))
    assert out.shape == (chunk_sz - cf,), (
        f"fallback chunk emitted {out.shape[0]} samples; expected {chunk_sz - cf}"
    )
    # Fallback was signalled and drain was tracked.
    assert sola.fallback_count == 1
    assert sola.cumulative_drain_samples == cfg.search_samples


def test_sola_no_drain_on_clean_alignment() -> None:
    """When the alignment search uniquely picks offset = -search
    (correlation 1.0, no equal-correlation peak elsewhere in the
    search range), the natural output is (input - cf) samples and
    no padding fires. We construct that case explicitly: seed
    `_prev_tail` with a known random buffer, then build a chunk
    whose first `cf` samples MATCH that buffer exactly and whose
    remainder is decorrelated noise. The unique correlation peak
    is at offset=-search by construction; all other offsets
    correlate against random noise (corr ≈ 0)."""
    cfg = SOLAConfig(rate=16_000, crossfade_ms=20.0, search_ms=4.0, corr_threshold=0.5)
    sola = SOLAStream(cfg)
    cf = cfg.crossfade_samples
    search = cfg.search_samples

    rng = np.random.default_rng(seed=29)
    seeded_tail = rng.standard_normal(cf).astype(np.float32)
    sola._prev_tail = seeded_tail.copy()  # bypass the first-chunk path

    # Chunk: exact copy of prev_tail, then random noise to fill it out.
    # offset = -search → head[0:cf] = chunk[0:cf] = seeded_tail. Match.
    # offset = 0 → head[search:search+cf] = chunk[search:search+cf] which
    # is mostly noise → corr ≈ 0.
    chunk_sz = 800
    chunk = np.empty(chunk_sz, dtype=np.float32)
    chunk[:cf] = seeded_tail
    chunk[cf:] = rng.standard_normal(chunk_sz - cf).astype(np.float32)

    out = sola.process(chunk)

    # Output length should be exactly chunk_sz - cf (no padding fired).
    assert out.shape == (chunk_sz - cf,), f"output shape {out.shape} != expected ({chunk_sz - cf},)"
    # No fallback (correlation 1.0 beats threshold) and zero drain
    # (offset=-search produces full natural output).
    assert sola.fallback_count == 0
    assert sola.cumulative_drain_samples == 0, (
        f"unexpected drain on uniquely-aligned input: {sola.cumulative_drain_samples}"
    )
    # Sanity.
    assert search > 0 and cf > 0


def test_sola_reset_clears_fallback_counters() -> None:
    """reset() must wipe the fallback/drain accounting alongside prev_tail
    so post-restart sessions don't carry stale numbers into woys-diag."""
    cfg = SOLAConfig(rate=16_000, crossfade_ms=20.0, search_ms=4.0, corr_threshold=0.99)
    sola = SOLAStream(cfg)
    rng = np.random.default_rng(seed=3)
    chunk = rng.standard_normal(800).astype(np.float32)
    sola.process(chunk)
    sola.process(chunk)
    assert sola.fallback_count > 0 or sola.cumulative_drain_samples > 0
    sola.reset()
    assert sola.fallback_count == 0
    assert sola.cumulative_drain_samples == 0
    assert sola._prev_tail is None


def test_sola_flush_emits_held_tail() -> None:
    cfg = SOLAConfig(rate=16_000, crossfade_ms=10.0, search_ms=1.0)
    sola = SOLAStream(cfg)
    cf = cfg.crossfade_samples
    chunk = np.ones(4 * cf, dtype=np.float32)
    sola.process(chunk)
    flushed = sola.flush()
    assert flushed.shape == (cf,)
    # After flush, the buffer is cleared.
    assert sola._prev_tail is None
    assert sola.flush().shape == (0,)
