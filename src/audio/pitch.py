"""Pitch-track helpers for the realtime inference path: bridging short
unvoiced gaps in RMVPE's f0 and mapping f0 to RVC's coarse pitch bins.

Split out of `audio.engine` (which re-exports every name here).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

NDArrayF32 = npt.NDArray[np.float32]
NDArrayI64 = npt.NDArray[np.int64]


# v0.6.9 - pitchf sanitization for the realtime inference path.
# Frames with NaN or f0 <= 0 are treated as "unvoiced" by the RVC vocoder's
# NSF source module; a single such frame mid-utterance zeros the harmonic
# source and produces an audible dropout. We replace NaN with 0 first
# (defensive against extractor bugs), then linearly interpolate runs of
# unvoiced frames up to `_VOICED_GAP_MAX_FRAMES` long between two voiced
# frames. Long unvoiced runs are left as zeros so true silence still
# decodes as silence. See `docs/12-vad-misfire-investigation.md`.
_VOICED_GAP_MAX_FRAMES = 8  # ~80 ms at the RMVPE 100 fps frame rate


def interpolate_voiced_gaps_np(
    pitchf: NDArrayF32,
    *,
    prior_voiced_f0: float = 0.0,
    prior_voiced_age_frames: int = -1,
) -> NDArrayF32:
    """B16 / perf-002: vectorized version. The pre-v0.8.0 implementation
    walked an inner `for k in range(i, j)` Python loop that ran ~50-200
    iterations per chunk under typical RMVPE pitch tracks. numpy slicing
    replaces the loop with a single broadcast multiply per gap.

    Also keeps the dtype path in float32 throughout (pre-v0.8.0 cast to
    float64 for the linspace arithmetic, then back to float32) - minor
    alloc churn reduction for B16's perf-001 partial.

    optional ``prior_voiced_f0`` +
    ``prior_voiced_age_frames`` carry the last-voiced anchor from the
    PREVIOUS chunk so a chunk-leading unvoiced run can still be bridged
    when its in-window `last_valid` is -1. The semantics match the
    in-window path: if the leading run length plus the prior age stays
    within ``_VOICED_GAP_MAX_FRAMES`` and a trailing voiced frame exists
    within this chunk, we synthesize a virtual anchor at index
    ``-prior_voiced_age_frames`` (negative, conceptually "outside the
    chunk to the left") and interpolate from `prior_voiced_f0` through
    the run to the trailing in-window anchor. Defaults ``(0.0, -1)``
    reproduce the pre-F-31-12 fall-back behaviour (no leading-edge
    bridge) -- no caller is forced to thread state through. Engine
    streaming path passes carry state from
    `EngineWorker._pitch_carry_*`.

    bridging interpolates in
    **log-f0** (geometric mean), not in Hz. A glide that is
    perceptually linear (constant semitones / second) is linear in
    log-frequency, not in Hz; bridging 100 Hz → 400 Hz in Hz puts
    the midpoint at 250 Hz, while a perceptually-straight glide
    crosses 200 Hz at midpoint (= sqrt(100 * 400)). Bridging over
    ≤ ``_VOICED_GAP_MAX_FRAMES`` (8 frames ≈ 80 ms) frames in Hz
    produces a sub-perceptual sag in the contour audible on
    voiced->unvoiced->voiced syllable boundaries. One ``log/exp``
    pair per gap; cost is negligible (≤ 8 frame-samples per gap).
    Pitch shift downstream of this function is a constant
    multiplicative factor in Hz, i.e. a constant additive offset
    in log-f0, so the shape of the log-linear bridge is preserved
    by the shift (verified in
    ``test_pitch_shift_modifies_pitchf_and_pitch_coarse_consistently``).
    """
    if pitchf.size == 0:
        return pitchf
    invalid = np.isnan(pitchf) | (pitchf <= 0.0)
    if not invalid.any():
        return pitchf
    if (~invalid).sum() == 0:
        # Whole chunk is unvoiced - preserve so vocoder produces silence.
        return np.nan_to_num(pitchf, nan=0.0).astype(np.float32, copy=False)
    out = np.nan_to_num(pitchf, nan=0.0).astype(np.float32, copy=True)
    n = len(invalid)
    have_prior = (
        prior_voiced_f0 > 0.0
        and prior_voiced_age_frames >= 0
        and prior_voiced_age_frames < _VOICED_GAP_MAX_FRAMES
    )
    # Walk the runs of invalid; bridge each ≤ _VOICED_GAP_MAX_FRAMES gap
    # via vectorized log-linear interpolation between the bracketing
    # voiced frames (F-31-03). Pre-F-31-03 this was linear-in-Hz.
    last_valid = -1
    i = 0
    while i < n:
        if not invalid[i]:
            last_valid = i
            i += 1
            continue
        j = i
        while j < n and invalid[j]:
            j += 1
        run_len = j - i
        if (
            run_len <= _VOICED_GAP_MAX_FRAMES
            and last_valid >= 0
            and j < n
            and out[last_valid] > 0.0
            and out[j] > 0.0
        ):
            # F-31-03: log-linear bridge. alpha vector over the gap,
            # interpolate in log space then exp back to Hz.
            alphas = (np.arange(i, j, dtype=np.float32) - last_valid) / (j - last_valid)
            log_lo = float(np.log(out[last_valid]))
            log_hi = float(np.log(out[j]))
            out[i:j] = np.exp(log_lo * (1.0 - alphas) + log_hi * alphas).astype(np.float32)
        elif (
            # F-31-12 leading-edge bridge: no in-window prior anchor, but
            # the previous chunk left a recent voiced f0 we can use.
            run_len <= _VOICED_GAP_MAX_FRAMES
            and last_valid < 0
            and j < n
            and have_prior
            and out[j] > 0.0
            and prior_voiced_age_frames + run_len <= _VOICED_GAP_MAX_FRAMES
        ):
            # Synthetic anchor at index -prior_voiced_age_frames - 1
            # (one full frame "older" than i==0). Total span from the
            # virtual anchor to j is (prior_voiced_age_frames + 1 + j).
            # F-31-03: log-linear here too.
            virt_anchor_offset = -(prior_voiced_age_frames + 1)
            span = float(j - virt_anchor_offset)
            alphas = (np.arange(i, j, dtype=np.float32) - virt_anchor_offset) / span
            log_lo = float(np.log(prior_voiced_f0))
            log_hi = float(np.log(out[j]))
            out[i:j] = np.exp(log_lo * (1.0 - alphas) + log_hi * alphas).astype(np.float32)
        i = j
    return out


def to_pitch_coarse(pitchf: NDArrayF32, target_len: int) -> tuple[NDArrayI64, NDArrayF32]:
    """B24 / quality-020: now a public name (drop leading underscore) so the
    smoke test can `from audio.engine import to_pitch_coarse` instead of
    re-implementing the algorithm. Single source of truth.

    B56 / perf-003: early-exit on all-zero pitchf - the engine's input gate
    fully zeroes audio during sub-hysteresis transitions (engine.py:2184)
    and the resulting RMVPE output is all-zero. Skipping the four numpy
    passes (log, mask multiply, clip, rint) saves ~8 µs per such chunk.

    v0.14.0 (area 7 / C093): clamp negative pitchf at entry. RMVPE in
    practice emits non-negative Hz, but transients / NaN-replaced regions
    can leak negatives. log(1 + pitch/700) at pitch < -700 produces NaN;
    NaN survives the `mask > 0` filter (NaN > 0 is False so the cell is
    untouched), then `clip(NaN, 1, 255)` returns NaN, then
    `rint().astype(int64)` becomes INT64_MIN, which RVC's harmonic-source
    table reads as out-of-bounds garbage. Clamping at entry makes the
    contract explicit and prevents the silent failure mode.
    """
    if pitchf.size == 0:
        return (
            np.zeros(target_len, dtype=np.int64),
            np.zeros(target_len, dtype=np.float32),
        )
    if pitchf.min() < 0.0:
        pitchf = np.clip(pitchf, 0.0, None)
    if float(pitchf.max()) == 0.0:
        return (
            np.zeros(target_len, dtype=np.int64),
            np.zeros(target_len, dtype=np.float32),
        )
    f0_min, f0_max = 50.0, 1100.0
    f0_mel_min = 1127.0 * np.log(1 + f0_min / 700.0)
    f0_mel_max = 1127.0 * np.log(1 + f0_max / 700.0)
    pitch = np.zeros(target_len, dtype=np.float32)
    n = min(len(pitchf), target_len)
    # when pitchf is over-length keep the *last*
    # n frames, not the first. RMVPE and contentvec emit unequal frame
    # counts, so `pitchf[:n]` temporally scrambled the F0 contour against
    # the content features -- frame i of the harmonic source corresponded
    # to a different point in time than frame i of the content. Upstream
    # keeps the trailing frames (Pipeline.py:288, `pitch[:, -feats_len:]`).
    # (When pitchf is not over-length, pitchf[-n:] == pitchf[:n].)
    pitch[-n:] = pitchf[-n:]
    f0_mel = 1127.0 * np.log(1 + pitch / 700.0)
    mask = f0_mel > 0
    f0_mel[mask] = (f0_mel[mask] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
    f0_mel = np.clip(f0_mel, 1.0, 255.0)
    return np.rint(f0_mel).astype(np.int64), pitch
