"""review F-31-12 (commit-079): cross-chunk pitch carry for
`interpolate_voiced_gaps_np`.

Pre-fix the bridge required a voiced anchor on BOTH sides of an
unvoiced run within the SAME pitchf vector. A voiced→unvoiced→voiced
transition that straddled a chunk boundary was therefore split:
chunk N saw `..voiced, unvoiced[, unvoiced...]` (no trailing anchor;
`j == n` → not bridged); chunk N+1 saw `[..unvoiced,] unvoiced,
voiced..` (no leading anchor; `last_valid == -1` → not bridged).
docs/12-vad-misfire-investigation.md attributed 6 of 14 residual
dropouts on a real voice to this signature.

The fix carries the last-voiced f0 and its age in frames across calls.
The next call uses the carry as a synthetic anchor when the in-window
`last_valid` is -1, bridging the leading-edge unvoiced run.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))


def test_default_kwargs_reproduce_pre_fix_behaviour() -> None:
    """No prior pitch carry → identical output to the pre-F-31-12
    no-kwargs call. Backwards-compat guard."""
    from audio.engine import interpolate_voiced_gaps_np

    rng = np.random.default_rng(0xC0DE)
    # Synthesise a pitchf with a mix of voiced and unvoiced.
    pitchf = rng.uniform(80.0, 400.0, 64).astype(np.float32)
    pitchf[10:15] = 0.0  # short unvoiced gap
    pitchf[:3] = 0.0  # leading unvoiced run, no in-window anchor before

    a = interpolate_voiced_gaps_np(pitchf.copy())
    b = interpolate_voiced_gaps_np(pitchf.copy(), prior_voiced_f0=0.0, prior_voiced_age_frames=-1)
    np.testing.assert_array_equal(a, b)


def test_leading_unvoiced_bridged_with_prior_anchor() -> None:
    """A pitchf starting with a short unvoiced run AND a voiced frame
    in range gets the leading run bridged using the prior anchor.

    F-31-03 (commit-080): bridge is log-linear; expected values reflect
    that domain change.
    """
    from audio.engine import interpolate_voiced_gaps_np

    # pitchf = [0, 0, 0, 220, voiced...] -- leading 3-frame unvoiced run.
    n = 20
    pitchf = np.full(n, 220.0, dtype=np.float32)
    pitchf[:3] = 0.0
    # Prior chunk's last voiced was 1 frame before this chunk: f0=180.
    out = interpolate_voiced_gaps_np(pitchf, prior_voiced_f0=180.0, prior_voiced_age_frames=0)

    # The leading 3 frames must now be > 0 (bridged), interpolated
    # log-linearly from 180 (at virtual index -1) to 220 (at index 3).
    # Span = 3 - (-1) = 4 frames.
    log_lo = np.log(180.0)
    log_hi = np.log(220.0)
    expected = []
    for i in range(3):
        alpha = (i - (-1)) / 4.0
        expected.append(np.exp(log_lo * (1.0 - alpha) + log_hi * alpha))
    np.testing.assert_allclose(out[:3], expected, atol=1e-3)


def test_leading_bridge_rejected_when_prior_too_old() -> None:
    """If the carry's age plus the leading run length exceeds
    `_VOICED_GAP_MAX_FRAMES` the bridge is rejected (the carry is
    stale)."""
    from audio.engine import _VOICED_GAP_MAX_FRAMES, interpolate_voiced_gaps_np

    n = 20
    pitchf = np.full(n, 220.0, dtype=np.float32)
    pitchf[:3] = 0.0
    # Age 6 frames + 3-frame run = 9 > MAX (8). Bridge rejected.
    assert _VOICED_GAP_MAX_FRAMES == 8
    out = interpolate_voiced_gaps_np(pitchf, prior_voiced_f0=180.0, prior_voiced_age_frames=6)
    np.testing.assert_array_equal(out[:3], np.zeros(3, dtype=np.float32))


def test_leading_bridge_only_fires_when_no_in_window_anchor() -> None:
    """If `last_valid >= 0` (voiced frame exists before the gap in
    this same pitchf), the in-window bridge takes priority and the
    carry is ignored. Carried priors must NOT override the legitimate
    in-window anchor on chunks where one exists."""
    from audio.engine import interpolate_voiced_gaps_np

    n = 20
    pitchf = np.full(n, 220.0, dtype=np.float32)
    pitchf[0] = 100.0  # in-window leading voiced anchor
    pitchf[1:4] = 0.0  # 3-frame gap
    out = interpolate_voiced_gaps_np(pitchf, prior_voiced_f0=999.0, prior_voiced_age_frames=0)
    # The 999.0 prior must NOT show up; the bridge interpolates 100 → 220
    # log-linearly (F-31-03). Span = 4 frames; the in-window anchor at
    # idx 0 (=100) is used, NOT the 999.0 prior.
    log_lo = np.log(100.0)
    log_hi = np.log(220.0)
    expected = [
        np.exp(log_lo * (1 - 0.25) + log_hi * 0.25),
        np.exp(log_lo * (1 - 0.5) + log_hi * 0.5),
        np.exp(log_lo * (1 - 0.75) + log_hi * 0.75),
    ]
    np.testing.assert_allclose(out[1:4], expected, atol=1e-3)
    # Also assert the values are nowhere near the 999.0 prior -- the
    # in-window anchor wins.
    assert (out[1:4] < 250.0).all()


def test_all_voiced_fast_path_unaffected_by_prior() -> None:
    """When pitchf has no invalid frames, the function returns early
    regardless of the prior kwargs -- this is the steady-state hot
    path, must not pay the carry-tax."""
    from audio.engine import interpolate_voiced_gaps_np

    pitchf = np.full(40, 220.0, dtype=np.float32)
    out = interpolate_voiced_gaps_np(pitchf, prior_voiced_f0=180.0, prior_voiced_age_frames=2)
    # All-voiced fast path returns the input array unmodified.
    assert out is pitchf or np.array_equal(out, pitchf)


def test_all_unvoiced_fast_path_unaffected_by_prior() -> None:
    """All-unvoiced pitchf returns the all-zero version regardless of
    prior. A pure-silence chunk must decode to silence; the carry
    cannot retroactively fabricate pitch where there is none."""
    from audio.engine import interpolate_voiced_gaps_np

    pitchf = np.zeros(40, dtype=np.float32)
    out = interpolate_voiced_gaps_np(pitchf, prior_voiced_f0=180.0, prior_voiced_age_frames=0)
    np.testing.assert_array_equal(out, np.zeros(40, dtype=np.float32))


def test_trailing_unvoiced_unchanged_by_prior_kwargs() -> None:
    """The carry only applies to LEADING unvoiced runs (where
    `last_valid == -1`). A trailing unvoiced run that hits `j == n`
    is still left as zeros -- the fix doesn't pretend to know future
    frames."""
    from audio.engine import interpolate_voiced_gaps_np

    n = 20
    pitchf = np.full(n, 220.0, dtype=np.float32)
    pitchf[-3:] = 0.0  # trailing 3-frame unvoiced run, no future anchor
    out = interpolate_voiced_gaps_np(pitchf, prior_voiced_f0=180.0, prior_voiced_age_frames=0)
    np.testing.assert_array_equal(out[-3:], np.zeros(3, dtype=np.float32))
