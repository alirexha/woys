"""Unit tests for engine inference helpers.

B17 / B53 / B16 / test-015: pin behavior of `interpolate_voiced_gaps_np`
before perf-002's vectorization changes the implementation; covers
synthetic inputs (silence, all-voiced, brief gap, long gap exceeding
`_VOICED_GAP_MAX_FRAMES`).

B56 / B24 / B17: pin `to_pitch_coarse` behavior - early-exit on all-zero
input + boundary cases that match the upstream RVC f0-coarse contract.

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
if str(REPO / "src" / "server") not in sys.path:
    sys.path.insert(0, str(REPO / "src" / "server"))

from audio.engine import (  # noqa: E402
    _VOICED_GAP_MAX_FRAMES,
    interpolate_voiced_gaps_np,
    to_pitch_coarse,
)

# ---- interpolate_voiced_gaps_np ---------------------------------------------


def test_interpolate_empty_returns_empty() -> None:
    out = interpolate_voiced_gaps_np(np.zeros(0, dtype=np.float32))
    assert out.size == 0


def test_interpolate_all_zero_passes_through_as_zeros() -> None:
    pitchf = np.zeros(50, dtype=np.float32)
    out = interpolate_voiced_gaps_np(pitchf)
    np.testing.assert_array_equal(out, np.zeros(50, dtype=np.float32))


def test_interpolate_all_voiced_unchanged() -> None:
    pitchf = np.full(20, 220.0, dtype=np.float32)
    out = interpolate_voiced_gaps_np(pitchf)
    np.testing.assert_array_equal(out, pitchf)


def test_interpolate_short_gap_bridges_log_linearly() -> None:
    """the gap-bridge is log-linear
    in f0, not linear-in-Hz. Pre-fix this test asserted the Hz-linear
    contour; post-fix the values are the geometric interpolant. Pin
    log-linear by checking each interior frame equals
    `exp(log(lo)*(1-alpha) + log(hi)*alpha)`."""
    pitchf = np.array(
        [100.0, 100.0, 0.0, 0.0, 0.0, 0.0, 200.0, 200.0],
        dtype=np.float32,
    )
    out = interpolate_voiced_gaps_np(pitchf)
    # Frames 2..5 are the gap; bridge from 100 (idx 1) to 200 (idx 6).
    # alpha = (k - 1) / (6 - 1) = (k - 1) / 5
    log_lo = np.log(100.0)
    log_hi = np.log(200.0)
    expected = np.array(
        [
            100.0,
            100.0,
            np.exp(log_lo * (1 - 1 / 5) + log_hi * (1 / 5)),
            np.exp(log_lo * (1 - 2 / 5) + log_hi * (2 / 5)),
            np.exp(log_lo * (1 - 3 / 5) + log_hi * (3 / 5)),
            np.exp(log_lo * (1 - 4 / 5) + log_hi * (4 / 5)),
            200.0,
            200.0,
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(out, expected, rtol=1e-5)


def test_interpolate_log_geometric_midpoint() -> None:
    """review-required test: bridge
    100 Hz -> 400 Hz across a 5-frame gap. Midpoint must be ~200 Hz
    (geometric mean, log-linear) and decidedly NOT 250 Hz (arithmetic
    mean, Hz-linear). The 25% delta at the midpoint is the size of
    the audible "sag" the fix removes from voiced->unvoiced->voiced
    transitions."""
    # 5-frame gap: indices 1..5; anchors at idx 0 (100) and idx 6 (400).
    pitchf = np.array(
        [100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 400.0],
        dtype=np.float32,
    )
    out = interpolate_voiced_gaps_np(pitchf)
    # Midpoint frame is idx 3. alpha = (3 - 0) / (6 - 0) = 0.5.
    # log-linear midpoint = exp(0.5*log(100) + 0.5*log(400))
    #                     = exp(log(sqrt(100*400))) = sqrt(40_000) = 200.
    np.testing.assert_allclose(out[3], 200.0, rtol=1e-5)
    # Refute the Hz-linear midpoint (== 250). Should miss by ~50.
    assert abs(out[3] - 250.0) > 40.0, (
        f"Midpoint at {out[3]:.2f} is suspiciously close to the Hz-linear "
        "value (250). F-31-03 must put it at the geometric mean (200)."
    )


def test_interpolate_long_gap_left_as_zeros() -> None:
    """Gap longer than `_VOICED_GAP_MAX_FRAMES` is genuine silence, not
    bridged - RVC vocoder should produce silence rather than fabricated
    pitch."""
    gap_len = _VOICED_GAP_MAX_FRAMES + 2
    pitchf = np.concatenate(
        [
            np.full(2, 220.0, dtype=np.float32),
            np.zeros(gap_len, dtype=np.float32),
            np.full(2, 220.0, dtype=np.float32),
        ]
    )
    out = interpolate_voiced_gaps_np(pitchf)
    # The gap should still be zeros (not bridged).
    assert (out[2 : 2 + gap_len] == 0.0).all()


def test_interpolate_handles_nan() -> None:
    pitchf = np.array([100.0, np.nan, np.nan, 200.0], dtype=np.float32)
    out = interpolate_voiced_gaps_np(pitchf)
    assert not np.isnan(out).any()
    # Bridge between idx 0 (100) and idx 3 (200): alphas at 1/3 and 2/3.
    # F-31-03: log-linear interpolation.
    log_lo = np.log(100.0)
    log_hi = np.log(200.0)
    np.testing.assert_allclose(out[1], np.exp(log_lo * (2 / 3) + log_hi * (1 / 3)), rtol=1e-5)
    np.testing.assert_allclose(out[2], np.exp(log_lo * (1 / 3) + log_hi * (2 / 3)), rtol=1e-5)


# ---- to_pitch_coarse --------------------------------------------------------


def test_pitch_coarse_all_zero_short_circuits() -> None:
    """B56 / perf-003: early-exit on all-zero input. Returns int64 zeros
    + float32 zeros without running the full mel transform."""
    coarse, pitch = to_pitch_coarse(np.zeros(50, dtype=np.float32), target_len=50)
    np.testing.assert_array_equal(coarse, np.zeros(50, dtype=np.int64))
    np.testing.assert_array_equal(pitch, np.zeros(50, dtype=np.float32))
    assert coarse.dtype == np.int64
    assert pitch.dtype == np.float32


def test_pitch_coarse_clips_to_valid_bin_range() -> None:
    """Output bins are in [1, 255] for any voiced frame; the leading
    zero-pad gets clipped to bin 1 (matches upstream RVC f0_coarse contract
    in `upstream/.../DioPitchExtractor.py:45-46`)."""
    pitchf = np.array([220.0, 440.0, 880.0], dtype=np.float32)
    coarse, _ = to_pitch_coarse(pitchf, target_len=10)
    assert coarse.min() >= 1
    assert coarse.max() <= 255
    assert coarse.dtype == np.int64


def test_pitch_coarse_short_input_right_aligned() -> None:
    """Pitchf shorter than target_len is right-aligned; leading frames
    are zero-padded (and clip to bin 1 by design)."""
    pitchf = np.array([220.0, 440.0], dtype=np.float32)
    coarse, pitch = to_pitch_coarse(pitchf, target_len=5)
    assert coarse.shape == (5,)
    assert pitch.shape == (5,)
    # Trailing elements are the input.
    assert pitch[-2:].tolist() == [220.0, 440.0]
    # Leading elements are zero-padded -> mapped to bin 1.
    assert (coarse[:3] == 1).all()


def test_pitch_coarse_overlength_keeps_trailing_frames() -> None:
    """an over-length pitchf must keep its *last*
    target_len frames, matching upstream `Pipeline.py:288`
    (`pitch[:, -feats_len:]`). Pre-fix it kept the *first* target_len
    frames (`pitchf[:n]`), temporally scrambling the F0 contour against
    the content features."""
    target_len = 10
    # 25 distinct frames so leading vs trailing slices are unambiguous.
    pitchf = np.arange(1, 26, dtype=np.float32) * 20.0  # 20, 40, ... 500
    _coarse, pitch = to_pitch_coarse(pitchf, target_len)
    # `pitch` (the float32 return) must hold the TRAILING target_len frames.
    np.testing.assert_array_equal(pitch, pitchf[-target_len:])
    assert not np.array_equal(pitch, pitchf[:target_len]), (
        "keeping the leading frames is the F-31-02 bug"
    )


@pytest.mark.parametrize("voiced_count", [1, 2, 5, 50])
def test_pitch_coarse_returns_correct_shapes(voiced_count: int) -> None:
    pitchf = np.full(voiced_count, 220.0, dtype=np.float32)
    coarse, pitch = to_pitch_coarse(pitchf, target_len=100)
    assert coarse.shape == (100,)
    assert pitch.shape == (100,)


def test_pitch_coarse_negative_input_clamped_not_int64_min() -> None:
    """v0.14.0 (area 7 / C093): negative pitchf must NOT propagate
    NaN through log -> mask -> clip -> int64 to produce INT64_MIN.
    RMVPE in practice emits non-negative Hz, but transient / NaN-replaced
    regions can leak negatives. Clamping at function entry preserves
    the invariant `1 <= coarse <= 255` for every cell.
    """
    pitchf = np.array([-200.0, -50.0, 220.0, 440.0], dtype=np.float32)
    coarse, _ = to_pitch_coarse(pitchf, target_len=10)
    assert coarse.dtype == np.int64
    # No bin should be INT64_MIN. The cheapest expressive check is the
    # documented invariant from existing tests: every bin is in [0, 255].
    # Pad cells are 0 (or 1 after the mel transform); voiced cells are
    # >= 1; nothing escapes that range.
    assert int(coarse.min()) >= 0
    assert int(coarse.max()) <= 255


def test_pitch_coarse_all_negative_returns_zeros() -> None:
    """v0.14.0 (area 7 / C093): all-negative pitchf clamps to all-zero
    and short-circuits (same path as the all-zero input case).
    """
    pitchf = np.array([-100.0, -200.0, -700.0, -2000.0], dtype=np.float32)
    coarse, pitch = to_pitch_coarse(pitchf, target_len=8)
    np.testing.assert_array_equal(coarse, np.zeros(8, dtype=np.int64))
    np.testing.assert_array_equal(pitch, np.zeros(8, dtype=np.float32))


def test_pitch_shift_modifies_pitchf_and_pitch_coarse_consistently() -> None:
    """v0.14.0 (area 4 / area 7 / C001): pitch shift in semitones must be
    applied to the f0 vector BEFORE deriving pitch_coarse. Otherwise
    pitch_coarse points at the unshifted f0 bin while pitchf is the
    shifted Hz vector -> RVC sees mismatched harmonic-source vs pitch-
    class-embedding pairs.

    This test mimics the engine's _infer pitch path: take a sine-tone
    pitchf, apply f0_up_key=+12 (octave up), and verify the resulting
    coarse bin moves up by ~17 mel bins (one octave on the
    1127*log(1+f/700) curve at 220 Hz -> 440 Hz).
    """
    pitchf = np.full(20, 220.0, dtype=np.float32)
    coarse_unshifted, _ = to_pitch_coarse(pitchf, target_len=20)
    # Apply pitch shift the way the engine does in v0.14.0.
    pitchf_shifted = pitchf * (2.0 ** (12 / 12.0))  # +12 semitones = octave
    coarse_shifted, pitch_shifted = to_pitch_coarse(pitchf_shifted, target_len=20)
    # The coarse bin must move (mismatch was the bug).
    assert int(coarse_shifted[-1]) > int(coarse_unshifted[-1])
    # The pitch vector must reflect the shift.
    assert pitch_shifted[-1] > pitchf[-1] * 1.5  # at least ~1.5x shift visible
