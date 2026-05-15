"""review F-07-03 (commit-077): the vectorised `_best_offset`
must match the pre-fix loop reference exactly (or close enough that
the only differences are fp-tie reshuffles, not real divergence) over
a battery of structured and random inputs.

If this test ever fails on real-world-shaped data (overlap=800,
search=64), the SOLA crossfade window has shifted in a way the user
will hear -- it's a behavioural regression, not just a perf
regression. The parity test is the null-listener guarantee.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "src" / "server") not in sys.path:
    sys.path.insert(0, str(REPO / "src" / "server"))


def _params() -> list[tuple[int, int, float]]:
    """(overlap, search, threshold) tuples covering:
    - real defaults at 16 kHz (cf=800, search=64)
    - smaller / faster variant (cf=200, search=16)
    - degenerate-edge variant (cf=64, search=4)
    """
    return [(800, 64, 0.25), (200, 16, 0.25), (64, 4, 0.25), (800, 64, 0.5), (800, 64, 0.0)]


# ---------------------------------------------------------------------------
# Parity: structured inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("overlap", "search", "threshold"), _params())
def test_exact_shift_recovers_same_offset(overlap: int, search: int, threshold: float) -> None:
    """If `head` is literally `tail` shifted by `k`, both impls must
    return `k` (the canonical "found the alignment" case)."""
    from audio.sola import _best_offset, _best_offset_loop_reference

    rng = np.random.default_rng(0xC0FFEE)
    for true_shift in (0, 1, search // 2, search):
        # Synthesise: tail is the LAST `overlap` of a longer signal;
        # head starts at offset `-true_shift` so head[true_shift:
        # true_shift+overlap] == tail.
        base = rng.standard_normal(overlap + search + 8).astype(np.float32)
        tail = base[-overlap:].copy()
        # Build head so head[true_shift : true_shift + overlap] == tail.
        head = np.zeros(overlap + search, dtype=np.float32)
        head[true_shift : true_shift + overlap] = tail
        # Pad the leading + trailing slack with noise so the alignment
        # is the unique global max (a flat zero would let the search
        # tie-break against any zero-only slice).
        if true_shift > 0:
            head[:true_shift] = rng.standard_normal(true_shift).astype(np.float32) * 0.01
        post_start = true_shift + overlap
        if head.shape[0] - post_start > 0:
            head[post_start:] = (
                rng.standard_normal(head.shape[0] - post_start).astype(np.float32) * 0.01
            )

        off_loop, fb_loop = _best_offset_loop_reference(tail, head, search, threshold)
        off_vec, fb_vec = _best_offset(tail, head, search, threshold)
        assert off_loop == off_vec, (
            f"offset mismatch at overlap={overlap}, search={search}, "
            f"shift={true_shift}: loop={off_loop} vec={off_vec}"
        )
        assert fb_loop == fb_vec


@pytest.mark.parametrize(("overlap", "search", "threshold"), _params())
def test_silence_tail_falls_back(overlap: int, search: int, threshold: float) -> None:
    """A near-silent `tail` triggers the `tail_norm < 1e-6` early-out
    in both impls -- (0, True)."""
    from audio.sola import _best_offset, _best_offset_loop_reference

    tail = np.zeros(overlap, dtype=np.float32)
    head = np.random.default_rng(0).standard_normal(overlap + search).astype(np.float32)

    assert _best_offset_loop_reference(tail, head, search, threshold) == (0, True)
    assert _best_offset(tail, head, search, threshold) == (0, True)


@pytest.mark.parametrize(("overlap", "search", "threshold"), _params())
def test_uncorrelated_falls_back_under_threshold(
    overlap: int, search: int, threshold: float
) -> None:
    """Two unrelated random buffers should produce a peak correlation
    below typical thresholds -- both impls return `(0, True)`."""
    from audio.sola import _best_offset, _best_offset_loop_reference

    rng = np.random.default_rng(0xBADBEEF)
    tail = rng.standard_normal(overlap).astype(np.float32)
    head = rng.standard_normal(overlap + search).astype(np.float32)

    loop = _best_offset_loop_reference(tail, head, search, threshold)
    vec = _best_offset(tail, head, search, threshold)
    assert loop == vec


@pytest.mark.parametrize(("overlap", "search", "threshold"), _params())
def test_random_battery_parity(overlap: int, search: int, threshold: float) -> None:
    """50 random adversarial inputs -- both impls must agree on offset
    AND fell_back. Drives both correlated and uncorrelated cases by
    seeding the head's relevant window with a noisy copy of tail."""
    from audio.sola import _best_offset, _best_offset_loop_reference

    rng = np.random.default_rng(0xFEEDFACE)
    for trial in range(50):
        tail = rng.standard_normal(overlap).astype(np.float32)
        head = rng.standard_normal(overlap + search).astype(np.float32)
        # 50% chance: bury a noisy copy of tail at a random offset so the
        # peak is real.
        if trial % 2 == 0:
            shift = int(rng.integers(0, search + 1))
            noise_db = float(rng.uniform(-30.0, -3.0))
            scale = 10 ** (noise_db / 20.0)
            head[shift : shift + overlap] = (
                tail + rng.standard_normal(overlap).astype(np.float32) * scale
            )

        off_loop, fb_loop = _best_offset_loop_reference(tail, head, search, threshold)
        off_vec, fb_vec = _best_offset(tail, head, search, threshold)
        assert (off_loop, fb_loop) == (off_vec, fb_vec), (
            f"trial {trial}: loop=({off_loop},{fb_loop}) vec=({off_vec},{fb_vec})"
        )


def test_too_short_head_falls_back_in_both() -> None:
    """The early-out for `len(head) < overlap + search` must fire in
    both impls."""
    from audio.sola import _best_offset, _best_offset_loop_reference

    tail = np.ones(64, dtype=np.float32)
    head = np.ones(64, dtype=np.float32)  # exactly overlap, no room for search
    assert _best_offset_loop_reference(tail, head, search=4, threshold=0.25) == (0, True)
    assert _best_offset(tail, head, search=4, threshold=0.25) == (0, True)


def test_empty_tail_falls_back_in_both() -> None:
    from audio.sola import _best_offset, _best_offset_loop_reference

    tail = np.zeros(0, dtype=np.float32)
    head = np.ones(100, dtype=np.float32)
    assert _best_offset_loop_reference(tail, head, search=10, threshold=0.25) == (0, True)
    assert _best_offset(tail, head, search=10, threshold=0.25) == (0, True)


# ---------------------------------------------------------------------------
# Perf: vectorised version must be measurably faster on the real shape
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_vectorised_is_faster_at_realistic_shape() -> None:
    """At the default SOLA shape (overlap=800 samples at cf=50 ms,
    search=64 samples at search=4 ms, both @ 16 kHz) the vectorised
    impl should be at least 4x faster than the loop. The actual delta
    is typically 10-30x on a warm CPU.

    Gated on `slow` because micro-benchmarks are inherently noisy --
    we just want a measurable signal, not a tight bound.
    """
    from audio.sola import _best_offset, _best_offset_loop_reference

    rng = np.random.default_rng(7)
    overlap, search, threshold = 800, 64, 0.25
    n_calls = 500
    inputs = [
        (
            rng.standard_normal(overlap).astype(np.float32),
            rng.standard_normal(overlap + search).astype(np.float32),
        )
        for _ in range(n_calls)
    ]

    # Warm-up (page cache, BLAS init).
    for tail, head in inputs[:10]:
        _best_offset(tail, head, search, threshold)
        _best_offset_loop_reference(tail, head, search, threshold)

    t0 = time.perf_counter()
    for tail, head in inputs:
        _best_offset_loop_reference(tail, head, search, threshold)
    loop_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for tail, head in inputs:
        _best_offset(tail, head, search, threshold)
    vec_s = time.perf_counter() - t0

    ratio = loop_s / vec_s if vec_s > 0 else float("inf")
    print(
        f"\n[sola] {n_calls} calls @ overlap={overlap} search={search}: "
        f"loop={loop_s * 1000:.1f}ms vec={vec_s * 1000:.1f}ms ratio={ratio:.1f}x",
        file=sys.stderr,
    )
    assert ratio >= 4.0, (
        f"vectorised impl should be ≥4x faster than the loop reference; "
        f"got ratio={ratio:.2f} (loop={loop_s * 1000:.1f}ms "
        f"vec={vec_s * 1000:.1f}ms)"
    )
