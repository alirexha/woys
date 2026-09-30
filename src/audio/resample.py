"""Sample-rate conversion for the realtime path (soxr).

`_StreamResampler` keeps soxr's filter state across chunks so consecutive
chunks splice cleanly; `_resample` is the one-shot variant. Split out of
`audio.engine` (which re-exports both).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

NDArrayF32 = npt.NDArray[np.float32]


def _resample(audio: NDArrayF32, src_rate: int, dst_rate: int) -> NDArrayF32:
    """High-quality stateless resampler - used for one-shot tests + tail flushes.

    The realtime engine path uses `_StreamResampler` instead so the
    anti-aliasing filter state survives across chunks. See
    `docs/11-microcuts-bug.md` for why per-chunk stateless resampling
    leaks a 4 Hz envelope artifact.

    Cost: ~0.5 ms for a 100 ms chunk on this CPU.
    """
    if src_rate == dst_rate:
        return audio.astype(np.float32, copy=False)
    if audio.size == 0:
        return audio.astype(np.float32, copy=False)
    import soxr  # type: ignore[import-untyped]

    out = soxr.resample(audio, src_rate, dst_rate, quality="HQ")
    return np.asarray(out, dtype=np.float32)


class _StreamResampler:
    """Stateful soxr resampler - preserves filter state across chunks.

    Per-call `soxr.resample(...)` resets the anti-aliasing filter every
    invocation; concatenating the resampled chunks introduces a brief
    filter-transient amplitude dip at every chunk boundary, audible as
    a 4 Hz flutter on sustained content (`docs/11-microcuts-bug.md`).
    `soxr.ResampleStream` carries the filter buffer across calls and
    eliminates the per-chunk warm-up.

    Identity case (`src_rate == dst_rate`) is a passthrough - no soxr
    object created.

    -- ``cold_fade_in_samples``. The
    *steady-state* per-chunk warmup is eliminated by carrying filter
    state, but a freshly-constructed `_StreamResampler` still cold-
    starts on its first emit: the soxr filter delay line begins zero-
    filled, so the first ~few-ms of output has a sub-unity gain.
    Engine.run_loop tolerates this on engine startup (silence before
    the first chunk anyway) but the model-swap path in
    `_apply_one_swap` builds a NEW _StreamResampler when the post-swap
    voice's native rate differs from the previous one -- the cold-
    start blip lands inside live, voiced audio. We mask the transient
    by applying a linear fade-in across the first `cold_fade_in_samples`
    of *output*; the swap path passes ~5 ms of dst_rate samples so the
    listener hears a brief amplitude ramp instead of a hard step.
    Default 0 means "no fade-in" -- the engine-startup constructor
    leaves it zero because the engine emits silence on startup anyway.
    """

    def __init__(
        self,
        src_rate: int,
        dst_rate: int,
        *,
        quality: str = "HQ",
        cold_fade_in_samples: int = 0,
    ) -> None:
        self.src_rate = src_rate
        self.dst_rate = dst_rate
        # F-31-11: remaining cold-fade-in budget. Decremented per emitted
        # sample; once exhausted the resampler is fully primed and
        # subsequent calls bypass the fade-in path entirely.
        self._cold_fade_remaining = int(max(0, cold_fade_in_samples))
        self._cold_fade_total = self._cold_fade_remaining
        if src_rate == dst_rate:
            self._stream = None
            return
        import soxr

        self._stream = soxr.ResampleStream(src_rate, dst_rate, num_channels=1, quality=quality)

    def _apply_cold_fade(self, out: NDArrayF32) -> NDArrayF32:
        """F-31-11: scale the leading samples of `out` by a linear ramp
        that completes the budgeted fade-in. Mutates `out` in place
        (a fresh array each call from soxr) and decrements the
        remaining budget. Cheap path when remaining is 0.
        """
        if self._cold_fade_remaining <= 0 or out.size == 0:
            return out
        total = self._cold_fade_total
        consumed = total - self._cold_fade_remaining
        n_fade_this_chunk = min(self._cold_fade_remaining, out.size)
        if n_fade_this_chunk > 0:
            # Ramp from consumed/total → (consumed + n_fade_this_chunk)/total
            start = consumed / total
            end = (consumed + n_fade_this_chunk) / total
            ramp = np.linspace(start, end, n_fade_this_chunk, endpoint=False, dtype=np.float32)
            out[:n_fade_this_chunk] *= ramp
        self._cold_fade_remaining -= n_fade_this_chunk
        return out

    def process(self, audio: NDArrayF32) -> NDArrayF32:
        """Consume `audio` (1-D float32 mono); return whatever soxr emits
        for this chunk. Output length will lag input length slightly while
        the internal buffer fills - flush() drains the rest."""
        if self._stream is None:
            # Identity path: still honour the cold-fade-in budget so the
            # F-31-11 contract holds even when no rate change is needed.
            out = audio.astype(np.float32, copy=True)
            return self._apply_cold_fade(out)
        if audio.size == 0:
            return np.zeros(0, dtype=np.float32)
        out = self._stream.resample_chunk(audio, last=False)
        out = np.asarray(out, dtype=np.float32).reshape(-1)
        return self._apply_cold_fade(out)

    def flush(self) -> NDArrayF32:
        """Drain any audio held in soxr's internal buffer. Call once before
        discarding (engine stop / model swap when output rate changes)."""
        if self._stream is None:
            return np.zeros(0, dtype=np.float32)
        out = self._stream.resample_chunk(np.zeros(0, dtype=np.float32), last=True)
        out = np.asarray(out, dtype=np.float32).reshape(-1)
        return self._apply_cold_fade(out)
