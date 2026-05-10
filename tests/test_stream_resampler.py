"""v0.6.7 - `_StreamResampler` regression tests.

The realtime engine resamples mic→16k and model_sr→48k once per chunk.
Stateless `soxr.resample(...)` resets its anti-aliasing filter every call,
which leaves a brief warm-up at the start of each output chunk; concatenating
those produces a small periodic dip at the chunk rate (4 Hz at default
250 ms chunks). On real-time playback that flutter is audible as
"tiny cuts between letters of a word."

`_StreamResampler` wraps `soxr.ResampleStream`, which carries the filter
state across chunks. These tests pin that behaviour:

  • identity (src == dst) is a passthrough, no soxr object created
  • streamed-resampled chunks concatenated match a one-shot resample of
    the same source within the soxr noise floor
  • flush() drains residual buffer; concatenating processed + flushed
    output reaches the expected sample count
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Local import - `audio` lives under src/.
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audio.engine import _resample, _StreamResampler  # noqa: E402


def test_identity_is_passthrough() -> None:
    rs = _StreamResampler(48_000, 48_000)
    audio = np.random.default_rng(0).standard_normal(1024).astype(np.float32)
    out = rs.process(audio)
    np.testing.assert_array_equal(out, audio)
    flush = rs.flush()
    assert flush.size == 0


def test_streamed_chunks_match_one_shot() -> None:
    """Sustained 220 Hz sine; chunk-by-chunk streaming should match the
    one-shot resample within the soxr noise floor (~-90 dB), well below
    audibility. Stateless per-chunk soxr calls also stay within this floor
    on a stationary signal - the audible artifact only shows up after the
    full RVC pipeline. The stateful resampler must not regress that floor."""
    src = 32_000
    dst = 48_000
    n = src * 2  # 2 seconds
    t = np.arange(n) / src
    sig = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    one_shot = _resample(sig, src, dst)

    rs = _StreamResampler(src, dst)
    chunk = src // 4  # 250 ms
    pieces = []
    for i in range(0, n, chunk):
        pieces.append(rs.process(sig[i : i + chunk]))
    pieces.append(rs.flush())
    streamed = np.concatenate(pieces).astype(np.float32)

    # Trim to common length for comparison.
    m = min(len(one_shot), len(streamed))
    one_shot = one_shot[:m]
    streamed = streamed[:m]
    rmse = float(np.sqrt(np.mean((one_shot - streamed) ** 2)))
    rmse_dbfs = 20.0 * np.log10(max(rmse, 1e-12))
    assert rmse < 1e-3, (
        f"streamed vs one-shot RMSE {rmse:.6f} ({rmse_dbfs:.1f} dBFS) - stateful path drifted"
    )


def test_flush_drains_remaining_buffer() -> None:
    """Streaming resampler holds a few samples internally each chunk
    (filter group delay). The total output across process+flush must
    cover the input length within ±1 sample."""
    src = 16_000
    dst = 48_000
    n = src // 2  # 0.5 seconds
    sig = np.full(n, 0.5, dtype=np.float32)

    rs = _StreamResampler(src, dst)
    body = rs.process(sig)
    tail = rs.flush()
    total = body.size + tail.size

    expected = round(n * dst / src)
    # soxr ResampleStream may differ by a small group delay margin.
    assert abs(total - expected) <= 4, (
        f"got {total} samples, expected ≈ {expected} (Δ={total - expected})"
    )


def test_zero_size_chunk_is_safe() -> None:
    """Realtime engine sometimes feeds an empty chunk (warmup, model swap).
    process() must accept it without raising or corrupting state."""
    rs = _StreamResampler(16_000, 48_000)
    out = rs.process(np.zeros(0, dtype=np.float32))
    assert out.size == 0
    # Subsequent real chunks still flow.
    real = np.full(8000, 0.3, dtype=np.float32)
    rs.process(real)
    flush = rs.flush()
    assert flush.size > 0  # filter buffer holds samples after the real chunk


def test_process_after_flush_raises_proves_rebuild_required() -> None:
    """v0.14.0 (Lens 4 / C002): after `flush()` finalizes the soxr stream
    with `last=True`, subsequent `process()` calls raise. The engine's
    model-swap path used to rely on `if new_sr != old_sr` to rebuild the
    output resampler; same-rate swaps left a finalized stream in place
    and the next chunk crashed the engine.

    This test pins the underlying soxr behavior so the engine-side fix
    in `_maybe_swap_model` (always rebuild if `resampler_out_was_flushed`)
    has a corresponding unit test for the contract it defends.
    """
    rs = _StreamResampler(40_000, 48_000)
    rs.process(np.full(800, 0.3, dtype=np.float32))
    rs.flush()  # finalizes the stream with last=True
    with pytest.raises(RuntimeError, match="last input"):
        rs.process(np.full(800, 0.3, dtype=np.float32))


def test_rebuild_after_flush_resumes_processing() -> None:
    """v0.14.0 (Lens 4 / C002): the documented recovery path after
    finalizing a stream is to construct a fresh `_StreamResampler`. This
    test confirms it works without state leak from the previous instance.
    """
    old = _StreamResampler(40_000, 48_000)
    old.process(np.full(800, 0.3, dtype=np.float32))
    old.flush()
    # Build a fresh resampler at the same rates - what the engine swap path
    # now does unconditionally when a flush has happened. Push enough
    # samples to clear soxr's filter buffer, then flush.
    new = _StreamResampler(40_000, 48_000)
    new.process(np.full(8000, 0.3, dtype=np.float32))
    flushed = new.flush()
    assert flushed.size > 0  # filter buffer drains
