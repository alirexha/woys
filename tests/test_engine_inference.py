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


def test_interpolate_short_gap_bridges_linearly() -> None:
    """Gap of 4 frames between two voiced frames at 100 Hz / 200 Hz -
    output should linearly interpolate."""
    pitchf = np.array(
        [100.0, 100.0, 0.0, 0.0, 0.0, 0.0, 200.0, 200.0],
        dtype=np.float32,
    )
    out = interpolate_voiced_gaps_np(pitchf)
    # Frames 2..5 are the gap; bridge from 100 (idx 1) to 200 (idx 6).
    # alpha = (k - 1) / (6 - 1) = (k - 1) / 5
    expected = np.array(
        [
            100.0,
            100.0,
            100.0 * (1 - 1 / 5) + 200.0 * (1 / 5),
            100.0 * (1 - 2 / 5) + 200.0 * (2 / 5),
            100.0 * (1 - 3 / 5) + 200.0 * (3 / 5),
            100.0 * (1 - 4 / 5) + 200.0 * (4 / 5),
            200.0,
            200.0,
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(out, expected, rtol=1e-5)


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
    np.testing.assert_allclose(out[1], 100.0 * (2 / 3) + 200.0 * (1 / 3), rtol=1e-5)
    np.testing.assert_allclose(out[2], 100.0 * (1 / 3) + 200.0 * (2 / 3), rtol=1e-5)


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


@pytest.mark.parametrize("voiced_count", [1, 2, 5, 50])
def test_pitch_coarse_returns_correct_shapes(voiced_count: int) -> None:
    pitchf = np.full(voiced_count, 220.0, dtype=np.float32)
    coarse, pitch = to_pitch_coarse(pitchf, target_len=100)
    assert coarse.shape == (100,)
    assert pitch.shape == (100,)
