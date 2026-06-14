"""SOLAStream.search_window_clipped.

The one-sided alignment search `[0, search]` can only delay the emit
window. If real-world audio ever has its true alignment beyond `search`,
`_best_offset` silently returns `best_idx == search` with a high
correlation -- previously indistinguishable from a healthy near-edge
alignment. F-31-05's counter splits those cases:

  * `fallback_count`        -- peak corr below threshold (silence /
                               de-correlated content); we used offset=0.
  * `search_window_clipped` -- peak corr cleared threshold but landed
                               at the FAR edge of the search window;
                               we trusted the offset, but the true
                               alignment may lie beyond.

Both counters are surfaced as `EngineStats.sola_fallback_count` /
`sola_search_clipped` so `woys diag` can refute (or confirm) the
"RVC bias is purely toward late emission" assumption that motivates
the one-sided contract.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))


def test_clipped_fires_when_peak_at_far_edge() -> None:
    """Construct an input where the unique-correlation peak lands at
    `best_idx == search`. SOLA must increment `search_window_clipped`
    on that chunk while leaving `fallback_count` at 0."""
    from audio.sola import SOLAConfig, SOLAStream

    cfg = SOLAConfig(rate=16_000, crossfade_ms=20.0, search_ms=4.0, corr_threshold=0.5)
    sola = SOLAStream(cfg)
    cf = cfg.crossfade_samples
    search = cfg.search_samples
    chunk_n = 800
    chunk_sz = chunk_n + cf + search

    rng = np.random.default_rng(seed=7)

    # Seed prev_tail so we control the correlation target. The "head"
    # the next call's _best_offset searches across is
    # new_audio[: cf + search] -- we place the prev_tail copy at the
    # FAR edge of that range (index = search) so the peak is uniquely
    # at the far edge.
    seeded_tail = rng.standard_normal(cf).astype(np.float32)
    sola._prev_tail = seeded_tail.copy()
    chunk = np.zeros(chunk_sz, dtype=np.float32)
    chunk[search : search + cf] = seeded_tail
    chunk[:search] = rng.standard_normal(search).astype(np.float32) * 0.01
    chunk[search + cf :] = rng.standard_normal(chunk_sz - search - cf).astype(np.float32) * 0.5

    sola.process(chunk)

    assert sola.fallback_count == 0, "corr cleared threshold; must not bump fallback"
    assert sola.search_window_clipped == 1, (
        f"expected search_window_clipped == 1, got {sola.search_window_clipped}"
    )


def test_clipped_does_not_fire_when_peak_in_middle() -> None:
    """Peak at a mid-window offset: trusted, but NOT clipped."""
    from audio.sola import SOLAConfig, SOLAStream

    cfg = SOLAConfig(rate=16_000, crossfade_ms=20.0, search_ms=4.0, corr_threshold=0.5)
    sola = SOLAStream(cfg)
    cf = cfg.crossfade_samples
    search = cfg.search_samples
    chunk_n = 800
    chunk_sz = chunk_n + cf + search

    rng = np.random.default_rng(seed=9)
    seeded_tail = rng.standard_normal(cf).astype(np.float32)
    sola._prev_tail = seeded_tail.copy()

    # Place the peak at search // 2 -- well inside the window.
    mid = search // 2
    chunk = (rng.standard_normal(chunk_sz) * 0.01).astype(np.float32)
    chunk[mid : mid + cf] = seeded_tail

    sola.process(chunk)
    assert sola.fallback_count == 0
    assert sola.search_window_clipped == 0


def test_clipped_does_not_fire_when_we_fall_back() -> None:
    """A fall_back chunk forces offset=0 unconditionally. Even though
    `best_idx == search` is structurally possible during the search
    walk, the clipped flag must remain False once we've decided to
    fall back -- the offset returned is 0, not `search`."""
    from audio.sola import SOLAConfig, SOLAStream

    cfg = SOLAConfig(rate=16_000, crossfade_ms=20.0, search_ms=4.0, corr_threshold=0.99)
    sola = SOLAStream(cfg)
    cf = cfg.crossfade_samples
    search = cfg.search_samples
    chunk_n = 800
    chunk_sz = chunk_n + cf + search
    rng = np.random.default_rng(seed=11)

    sola.process(rng.standard_normal(chunk_sz).astype(np.float32))  # warmup
    sola.process(rng.standard_normal(chunk_sz).astype(np.float32))

    assert sola.fallback_count == 1, "high threshold should force fall_back"
    assert sola.search_window_clipped == 0, "clipped must not fire on fall_back"


def test_reset_clears_clipped_counter() -> None:
    """`reset()` must wipe both SOLA counters together."""
    from audio.sola import SOLAConfig, SOLAStream

    cfg = SOLAConfig(rate=16_000, crossfade_ms=20.0, search_ms=4.0, corr_threshold=0.5)
    sola = SOLAStream(cfg)
    cf = cfg.crossfade_samples
    search = cfg.search_samples
    chunk_n = 800
    chunk_sz = chunk_n + cf + search

    rng = np.random.default_rng(seed=13)
    seeded_tail = rng.standard_normal(cf).astype(np.float32)
    sola._prev_tail = seeded_tail.copy()
    chunk = (rng.standard_normal(chunk_sz) * 0.01).astype(np.float32)
    chunk[search : search + cf] = seeded_tail
    sola.process(chunk)
    assert sola.search_window_clipped == 1

    sola.reset()
    assert sola.search_window_clipped == 0
    assert sola.fallback_count == 0
