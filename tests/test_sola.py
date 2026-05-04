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
    tail inside a longer 'head' buffer. _best_offset should recover the shift."""
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
        recovered = _best_offset(tail, head, search=search, threshold=0.5)
        assert recovered == true_shift, f"expected {true_shift}, got {recovered}"


def test_best_offset_below_threshold_returns_zero() -> None:
    """Pure noise shouldn't pick a non-zero offset."""
    rng = np.random.default_rng(seed=42)
    overlap = 128
    search = 16
    tail = rng.standard_normal(overlap).astype(np.float32)
    head = rng.standard_normal(overlap + 2 * search).astype(np.float32)
    # Threshold high enough that random noise won't beat it.
    assert _best_offset(tail, head, search=search, threshold=0.9) == 0


def test_best_offset_silent_tail() -> None:
    """A near-silent prev_tail can't drive correlation; expect 0."""
    overlap = 64
    search = 8
    tail = np.zeros(overlap, dtype=np.float32) + 1e-9
    head = np.random.default_rng(0).standard_normal(overlap + 2 * search).astype(np.float32)
    assert _best_offset(tail, head, search=search, threshold=0.1) == 0


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
