"""Synchronized Overlap-Add (SOLA) crossfade for low-latency RVC streaming.

The streaming inference loop produces a short audio block per chunk. Smaller
blocks → lower latency, but raw concatenation of independent model outputs
introduces phase discontinuities at chunk boundaries (audible clicks /
"buzz" on sustained vowels).

SOLA splits the work in two:

  1. **Sliding context** — each model call sees the new mic chunk *plus* a
     fixed history window. Model output gets trimmed back to just the new
     content. This keeps the embedder + vocoder seeing enough context to
     produce stable features.

  2. **Cross-correlated crossfade** — the *start* of the new output is
     correlated against the *tail* of the previous output across a small
     search window (≈2-3 ms). Pick the offset where the two waveforms align
     in phase, then linearly crossfade over the overlap region. If the peak
     correlation is too weak (e.g. silence, noise), fall back to a centered
     overlap.

This module is purely numpy; no torch, no ORT. All operations work on float32
mono buffers at the engine's chosen rate (typically 16 kHz model output).

Parameters
----------
``crossfade_samples``  — overlap region width in samples.
``search_samples``     — how far we shift the new chunk to look for an
                         alignment that beats centered overlap. Half a pitch
                         period at the lowest typical voice f0 (~80 Hz =
                         200 samples @ 16 kHz) is plenty; we use ~32 by
                         default which covers ~250 Hz pitches well.
``corr_threshold``     — minimum normalized correlation peak to trust.
                         Below this, fall back to offset=0.

Why these defaults: 50 ms crossfade @ 16 kHz = 800 samples. That's 5 pitch
periods at 100 Hz (low male voice) — enough to mask seams without smearing
phonemes.

Original work — Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

NDArrayF32 = npt.NDArray[np.float32]


@dataclass
class SOLAConfig:
    rate: int = 16_000
    crossfade_ms: float = 50.0
    search_ms: float = 4.0  # ±4 ms search window
    context_ms: float = 100.0  # extra history fed to the model each call
    corr_threshold: float = 0.25  # normalized correlation must beat this to use it

    @property
    def crossfade_samples(self) -> int:
        return round(self.rate * self.crossfade_ms / 1000.0)

    @property
    def search_samples(self) -> int:
        return round(self.rate * self.search_ms / 1000.0)

    @property
    def context_samples(self) -> int:
        return round(self.rate * self.context_ms / 1000.0)


def _hann_fade(n: int) -> tuple[NDArrayF32, NDArrayF32]:
    """Return (fade_out, fade_in) windows of length n. Hann shape, sums to 1."""
    if n <= 0:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty
    t = np.linspace(0.0, np.pi, n, endpoint=False, dtype=np.float32)
    fade_out: NDArrayF32 = (np.cos(t / 2.0) ** 2).astype(np.float32)
    fade_in: NDArrayF32 = (np.sin(t / 2.0) ** 2).astype(np.float32)
    return fade_out, fade_in


def _best_offset(
    tail: NDArrayF32, head: NDArrayF32, search: int, threshold: float
) -> tuple[int, bool]:
    """Find the integer shift `k` (within ±search) that maximizes the
    normalized correlation between `tail[-overlap:]` and `head[k : k + overlap]`.

    Returns `(offset, fell_back)`. `fell_back=True` means the peak
    correlation was below `threshold` (silence / de-correlated content)
    and `offset` is `0` regardless of which shift won.
    """
    overlap = len(tail)
    if overlap == 0 or len(head) < overlap + 2 * search:
        return 0, True

    # Normalize tail once.
    tail_norm = float(np.linalg.norm(tail))
    if tail_norm < 1e-6:
        return 0, True

    best_offset = 0
    best_corr = -np.inf
    for k in range(-search, search + 1):
        start = search + k  # head's first sample we read at offset k
        slice_ = head[start : start + overlap]
        if slice_.shape[0] != overlap:
            continue
        s_norm = float(np.linalg.norm(slice_))
        if s_norm < 1e-6:
            continue
        corr = float(np.dot(tail, slice_)) / (tail_norm * s_norm)
        if corr > best_corr:
            best_corr = corr
            best_offset = k

    if best_corr >= threshold:
        return best_offset, False
    return 0, True


class SOLAStream:
    """Stateful SOLA crossfader.

    Feed `process(new_audio)` one chunk at a time. The first call passes
    its full input straight through (no history yet). Subsequent calls
    crossfade against the saved tail from the previous output.

    The optional `context_samples` is *not* applied here — that's the model's
    business. SOLAStream only knows about the *output* stream. The engine
    is expected to:
      - feed `context + new_chunk` into the model
      - trim the leading context samples from the model output
      - hand the trimmed output to `process()`

    This keeps SOLA model-agnostic.
    """

    def __init__(self, cfg: SOLAConfig | None = None) -> None:
        self.cfg = cfg or SOLAConfig()
        self._fade_out, self._fade_in = _hann_fade(self.cfg.crossfade_samples)
        # The "kept tail" carries the *unfaded* end of the previous emit so we
        # can correlate against it next time. We store length=crossfade_samples.
        self._prev_tail: NDArrayF32 | None = None
        # v0.7.0-rc4 — counters for the per-call output drain. When
        # `_best_offset` picks any offset other than `-search`, the
        # algorithm's natural output is short by `search + offset`
        # samples vs the optimum. Untracked, this drains the downstream
        # output buffer at a buffer-size-independent rate (~7 ms/sec at
        # chunk=0.15 with 18 % fallback rate per audit lens 03). We now
        # zero-pad the shortfall to keep output ≈ input length and
        # surface the totals so woys-diag / woys status can attribute
        # cuts to SOLA fallback drain rather than to the output buffer.
        self.fallback_count: int = 0
        self.cumulative_drain_samples: int = 0

    def reset(self) -> None:
        self._prev_tail = None
        self.fallback_count = 0
        self.cumulative_drain_samples = 0

    @property
    def context_samples(self) -> int:
        return self.cfg.context_samples

    def process(self, new_audio: NDArrayF32) -> NDArrayF32:
        """Consume the next chunk's model output; return the part to emit.

        Streaming guarantee: the bytes returned are *causal* — they only
        depend on the current and prior chunks, never on a chunk we haven't
        seen yet. Latency floor is `crossfade_samples` (the trailing region
        we hold back to crossfade with the next chunk).
        """
        cfg = self.cfg
        cf = cfg.crossfade_samples
        new_audio = np.ascontiguousarray(new_audio.astype(np.float32, copy=False))

        if self._prev_tail is None:
            # First chunk: emit everything except the trailing crossfade
            # region (held back for next call).
            if new_audio.shape[0] <= cf:
                # New chunk too small to hold back a tail — emit nothing,
                # save what we have, await the next chunk.
                self._prev_tail = new_audio.copy()
                return np.zeros(0, dtype=np.float32)
            self._prev_tail = new_audio[-cf:].copy()
            return new_audio[:-cf].astype(np.float32, copy=False)

        # Need at least crossfade + 2*search samples in the new chunk to do
        # a meaningful correlation; otherwise fall back to centered overlap.
        head_needed = cf + 2 * cfg.search_samples
        fell_back = False
        if new_audio.shape[0] >= head_needed:
            offset, fell_back = _best_offset(
                self._prev_tail,
                new_audio[:head_needed],
                cfg.search_samples,
                cfg.corr_threshold,
            )
        else:
            offset = 0
            fell_back = True

        # Aligned head: the part of new_audio that overlaps with prev_tail.
        head_start = cfg.search_samples + offset if new_audio.shape[0] >= head_needed else 0
        head_end = head_start + cf
        if head_end > new_audio.shape[0]:
            # Not enough new samples to fill an overlap region; can't crossfade.
            # Emit prev_tail directly, save current as the new prev_tail.
            emit = self._prev_tail.copy()
            self._prev_tail = new_audio.copy()
            return emit

        head = new_audio[head_start:head_end]
        crossfaded = self._prev_tail * self._fade_out + head * self._fade_in

        # Emit the crossfaded region followed by the bulk of new_audio after
        # the overlap, holding back the new trailing crossfade for next call.
        body_start = head_end
        body_end = max(body_start, new_audio.shape[0] - cf)
        body = new_audio[body_start:body_end]
        new_tail = new_audio[max(body_start, new_audio.shape[0] - cf) :]
        # Pad new_tail to cf samples (rare: short chunks).
        if new_tail.shape[0] < cf:
            pad = np.zeros(cf - new_tail.shape[0], dtype=np.float32)
            new_tail = np.concatenate([pad, new_tail])
        self._prev_tail = new_tail[-cf:].copy()

        out: NDArrayF32 = np.concatenate([crossfaded, body]).astype(np.float32, copy=False)

        # v0.7.0-rc4 — pad fallback shortfall. When `_best_offset` picks
        # `offset != -search` (either fallback or just a non-optimal
        # alignment), the natural output is `search + offset` samples
        # short of the optimum. Untracked, this drains the downstream
        # output buffer at chunk_seconds-search/chunk_seconds rate during
        # any sustained voicing burst that doesn't beat corr_threshold.
        # Zero-padding the shortfall keeps output ≈ input length; the
        # alternative (let the buffer drain) was hypothesised by the
        # audit synthesis as an independent contributor to the persistent
        # cuts. The pad is a small click at the body→pad boundary, but
        # the next chunk's crossfade smooths it; cumulative drain on the
        # other hand caused audible mid-utterance dropouts and was the
        # P0 we're closing.
        expected_len = new_audio.shape[0] - cf
        if expected_len > 0 and out.shape[0] < expected_len:
            shortfall = expected_len - out.shape[0]
            self.cumulative_drain_samples += shortfall
            if fell_back:
                self.fallback_count += 1
            pad = np.zeros(shortfall, dtype=np.float32)
            out = np.concatenate([out, pad]).astype(np.float32, copy=False)

        return out

    def flush(self) -> NDArrayF32:
        """Emit and clear any held-back tail. Call once on engine shutdown."""
        if self._prev_tail is None:
            return np.zeros(0, dtype=np.float32)
        out = self._prev_tail
        self._prev_tail = None
        return out
