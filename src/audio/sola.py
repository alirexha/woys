"""Synchronized Overlap-Add (SOLA) crossfade for low-latency RVC streaming.

The streaming inference loop produces a short audio block per chunk. Smaller
blocks → lower latency, but raw concatenation of independent model outputs
introduces phase discontinuities at chunk boundaries (audible clicks /
"buzz" on sustained vowels).

SOLA splits the work in two:

  1. **Sliding context** - each model call sees the new mic chunk *plus* a
     fixed history window. Model output gets trimmed back to just the new
     content. This keeps the embedder + vocoder seeing enough context to
     produce stable features.

  2. **Cross-correlated crossfade** - the *start* of the new output is
     correlated against the *tail* of the previous output across a small
     search window (≈2-3 ms). Pick the offset where the two waveforms align
     in phase, then linearly crossfade over the overlap region. If the peak
     correlation is too weak (e.g. silence, noise), fall back to a centered
     overlap.

This module is purely numpy; no torch, no ORT. All operations work on float32
mono buffers at the engine's chosen rate (typically 16 kHz model output).

Parameters
----------
``crossfade_samples``  - overlap region width in samples.
``search_samples``     - how far we shift the new chunk to look for an
                         alignment that beats centered overlap. Half a pitch
                         period at the lowest typical voice f0 (~80 Hz =
                         200 samples @ 16 kHz) is plenty; we use ~32 by
                         default which covers ~250 Hz pitches well.
``corr_threshold``     - minimum normalized correlation peak to trust.
                         Below this, fall back to offset=0.

Why these defaults: 50 ms crossfade @ 16 kHz = 800 samples. That's 5 pitch
periods at 100 Hz (low male voice) - enough to mask seams without smearing
phonemes.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
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
    """Find the integer shift `k` in `[0, search]` that maximizes the
    normalized correlation between `tail[-overlap:]` and
    `head[k : k + overlap]`.

    Returns `(offset, fell_back)`. `fell_back=True` means the peak
    correlation was below `threshold` (silence / de-correlated content)
    and `offset` is `0` in that case.

    v0.7.0-rc5: switched from bidirectional `[-search, +search]` to
    one-sided `[0, search]` to match upstream w-okada's SOLA contract
    (`upstream/server/voice_changer/VoiceChangerV2.py:248-266`). The
    practical effect: emit window is `head[offset : offset + chunk_n]`
    instead of `head[search + offset : search + offset + chunk_n]`,
    so the algorithm can produce a constant-size emit per call
    regardless of which offset wins. Bidirectional search would let
    us recover from "model emitted slightly early" cases too, but
    real RVC's bias is purely toward late emission, so one-sided is
    sufficient and matches the reference implementation. Hard Rule 1
    note: the "RVC bias is purely late" claim is an assumed property
    of the reference implementation, not an in-tree measurement; the
    one-sided contract follows upstream rather than independent
    evidence. F-31-05 cluster (commit-079).

    review F-07-03 (commit-077): vectorised. Pre-fix this ran a
    Python loop over `range(search + 1)` (default 65 iterations
    per call), each computing `np.linalg.norm(slice_)` + `np.dot(tail,
    slice_)`. At `chunk_seconds=0.25` and a 100% voiced session the
    loop fires 4x per second and lights up `_best_offset` in py-spy
    flame graphs as the dominant pure-Python hot-spot in the SOLA
    streaming path. The vectorised form uses one `np.correlate` for
    the numerator (cross-correlation at every offset in one BLAS
    call) and one cumulative-sum-of-squares for the rolling norm,
    so the per-call cost is two O(overlap + search) array ops
    instead of `search+1` Python iterations each doing two BLAS
    calls. The reference implementation
    (`_best_offset_loop_reference` below) stays in-module so the
    parity test in `tests/test_sola_best_offset_vectorised.py` can
    pin the two paths against each other.
    """
    overlap = len(tail)
    if overlap == 0 or len(head) < overlap + search:
        return 0, True

    # Normalise tail once (fp32 -> Python float; matches the reference).
    tail_norm = float(np.linalg.norm(tail))
    if tail_norm < 1e-6:
        return 0, True

    # Numerator: cross-correlation `sum(head[k : k + overlap] * tail)`
    # for k in [0, search]. `np.correlate(head, tail, mode='valid')`
    # returns exactly `len(head) - overlap + 1 = search + 1` samples.
    corr_num = np.correlate(head, tail, mode="valid")

    # Denominator: rolling L2 norm of `head[k : k + overlap]` across
    # all valid k. cumsum-of-squares yields the per-window sum-of-
    # squares in O(len(head)) with no temporaries beyond `head**2`.
    # fp64 accumulator keeps the rolling sum honest for long overlaps
    # (default 800 samples) where fp32 partial sums accumulate ~6
    # ULPs of error -- below the reference's own `np.linalg.norm`
    # promotion behaviour so the parity test passes within fp32.
    head_sq = head.astype(np.float64) * head.astype(np.float64)
    cs = np.empty(head_sq.shape[0] + 1, dtype=np.float64)
    cs[0] = 0.0
    np.cumsum(head_sq, out=cs[1:])
    window_ss = cs[overlap:] - cs[:-overlap]
    window_norm = np.sqrt(window_ss)

    # Skip windows whose L2 norm is below the same `1e-6` epsilon the
    # reference uses; without this the division would produce `inf`
    # in those rows and `argmax` could land on a noise window.
    safe = window_norm > 1e-6
    corr = np.full(corr_num.shape, -np.inf, dtype=np.float64)
    np.divide(corr_num, tail_norm * window_norm, out=corr, where=safe)

    # `argmax` matches the reference's `>`-based tie-break: on a tie
    # both pick the LOWEST-INDEX winner.
    best_idx = int(np.argmax(corr))
    best_corr = float(corr[best_idx])

    if best_corr >= threshold:
        return best_idx, False
    return 0, True


def _best_offset_loop_reference(
    tail: NDArrayF32, head: NDArrayF32, search: int, threshold: float
) -> tuple[int, bool]:
    """Pre-F-07-03 reference implementation, kept in-module exclusively
    for the parity test in `tests/test_sola_best_offset_vectorised.py`.
    Do NOT call from the streaming path -- the vectorised
    `_best_offset` above is the production hot-path.

    The behavioural contract pinned by the parity test:
      * Same `(offset, fell_back)` return on identical inputs.
      * Same tie-break (lowest-index winner on equal correlations).
      * Same fallback predicate (`peak_corr >= threshold`).
    """
    overlap = len(tail)
    if overlap == 0 or len(head) < overlap + search:
        return 0, True
    tail_norm = float(np.linalg.norm(tail))
    if tail_norm < 1e-6:
        return 0, True

    best_offset = 0
    best_corr = -np.inf
    for k in range(search + 1):
        slice_ = head[k : k + overlap]
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

    Feed `process(new_audio)` one chunk at a time. Each call returns
    exactly `chunk_n` samples - the constant emit length. The
    algorithm's alignment search slides the emit window inside the
    leading `crossfade + search` samples of the input, so picking a
    non-zero offset never shrinks the emit; the search slack lives
    in the input, not the output.

    The contract:
      - Engine feeds `new_audio` of length `chunk_n + crossfade + search`.
      - First chunk: emit `new_audio[:chunk_n]`, hold the last
        `crossfade` samples as `prev_tail`, discard the trailing
        `search` slack (no prev_tail to align against on first call).
      - Subsequent chunks: search `new_audio[:crossfade + search]` for
        the offset `k ∈ [0, search]` whose `crossfade`-sample window
        best correlates with `prev_tail`. Emit
        `new_audio[k : k + chunk_n]`. Crossfade the leading
        `crossfade` samples of emit with `prev_tail`. Save the new
        `prev_tail` from the last `crossfade` samples of `new_audio`
        (a fixed temporal position regardless of which offset won).

    Streaming guarantee: output is causal. Latency floor is
    `crossfade + search` (the trailing region the caller must keep
    available for the next call's search).

    Matches the upstream w-okada SOLA contract at
    `upstream/server/voice_changer/VoiceChangerV2.py:248-285`. The
    pre-rc5 implementation emitted `len(new_audio) - cf - search -
    offset` samples (variable length); the rc4 zero-pad covered the
    symptom by injecting silence and was audibly worse than letting
    the buffer drain. rc5 fixes the math instead of padding over it.
    See `docs/16-audit/11-rc4-postmortem.md`.
    """

    def __init__(self, cfg: SOLAConfig | None = None) -> None:
        self.cfg = cfg or SOLAConfig()
        self._fade_out, self._fade_in = _hann_fade(self.cfg.crossfade_samples)
        # The "kept tail" carries the *unfaded* tail of the previous
        # input so we can correlate against it next time. Length =
        # crossfade_samples. Sourced from `new_audio[-cf:]` of the
        # previous call (a fixed temporal position; matches upstream).
        self._prev_tail: NDArrayF32 | None = None
        # Threshold-fallback events: the alignment search's peak
        # correlation fell below `corr_threshold`, so the algorithm
        # used `offset = 0` (centered, no shift). Voice-correlated
        # spikes in this counter are evidence the corr_threshold or
        # search_ms tuning is off; they no longer indicate output drain
        # (rc5 emits constant `chunk_n` samples per call regardless).
        self.fallback_count: int = 0

    def reset(self) -> None:
        self._prev_tail = None
        self.fallback_count = 0

    @property
    def context_samples(self) -> int:
        return self.cfg.context_samples

    def process(self, new_audio: NDArrayF32) -> NDArrayF32:
        """Consume the next chunk's model output; return `chunk_n`
        samples to emit.

        `chunk_n` is implied by the input length:
          chunk_n = len(new_audio) - crossfade_samples - search_samples

        That is, the caller is expected to feed `chunk_n + cf + search`
        samples per call. If the input is shorter than `cf + search`
        we fall back to "hold everything as prev_tail, emit nothing"
        (warmup case). If `chunk_n == 0` the call is also a warmup.
        """
        cfg = self.cfg
        cf = cfg.crossfade_samples
        search = cfg.search_samples
        new_audio = np.ascontiguousarray(new_audio.astype(np.float32, copy=False))

        # Implied chunk length given the upstream contract.
        chunk_n = new_audio.shape[0] - cf - search

        if chunk_n <= 0:
            # Input too small to extract a chunk_n emit. Hold the last cf
            # samples as the next `_prev_tail` if available; otherwise
            # reset state so the next normal chunk takes the first-chunk
            # path (no crossfade) instead of trying to crossfade against
            # an under-sized tail. (B22 / audio-002 - see review.)
            if new_audio.shape[0] >= cf:
                self._prev_tail = new_audio[-cf:].copy()
            else:
                self._prev_tail = None
            return np.zeros(0, dtype=np.float32)

        if self._prev_tail is None:
            # First chunk: no prev_tail to align against. Emit the
            # leading chunk_n samples; save the next cf as prev_tail.
            # The trailing `search` slack is unused on first call.
            emit = new_audio[:chunk_n].astype(np.float32, copy=True)
            self._prev_tail = new_audio[-cf:].copy()
            return emit

        # Subsequent chunks: search for the offset in [0, search] that
        # best aligns prev_tail with new_audio[k : k + cf].
        offset, fell_back = _best_offset(
            self._prev_tail, new_audio[: cf + search], search, cfg.corr_threshold
        )
        if fell_back:
            self.fallback_count += 1

        # Emit window: chunk_n samples starting at `offset`.
        emit_start = offset
        emit_end = offset + chunk_n
        emit = new_audio[emit_start:emit_end].astype(np.float32, copy=True)

        # Crossfade the leading cf samples of emit with prev_tail. The
        # Hann pair sums to 1, so amplitude is preserved.
        if emit.shape[0] >= cf:
            emit[:cf] = self._prev_tail * self._fade_out + emit[:cf] * self._fade_in

        # Save new prev_tail from the END of new_audio - a fixed
        # temporal position regardless of which offset won. The next
        # chunk's input will overlap this region, so the next search
        # correlates against a known location in the audio stream.
        self._prev_tail = new_audio[-cf:].copy()

        return emit

    def flush(self) -> NDArrayF32:
        """Emit and clear any held-back tail. Call once on engine shutdown.

        B22 / audio-009: apply a linear fade-out so the tail doesn't end
        with a hard cutoff at a non-zero amplitude - the click at session
        end was a small but audible UX nit on every shutdown.
        """
        if self._prev_tail is None:
            return np.zeros(0, dtype=np.float32)
        n = self._prev_tail.shape[0]
        if n > 1:
            fade = np.linspace(1.0, 0.0, n, dtype=np.float32)
            out = (self._prev_tail * fade).astype(np.float32, copy=False)
        else:
            out = self._prev_tail
        self._prev_tail = None
        return out
