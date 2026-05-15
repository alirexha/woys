"""review F-03-02 + F-13-12 (commit-042-043): the model-swap
queue + per-call completion events.

Pre-fix:
- F-03-02: `request_model_swap` overwrote a single `_pending_model_
  swap: Path` slot, and `_swap_done` was a SHARED `threading.Event`
  that all waiters watched. Two rapid swap requests collapsed:
  voiceB overwrote voiceA in the slot; the worker applied voiceB
  only; the broadcast `_swap_done.set()` released ALL waiters --
  Job A reported "done" even though voiceA was NEVER loaded. Hard
  Rule 2 silent-failure.
- F-13-12: `_swap_done.set()` had three setters, all inside
  `_maybe_swap_model` (engine thread). `stop()` never set it -- so
  the "queue a swap, toggle off" sequence left a JobRegistry daemon
  thread parked for the full 10 s timeout.

Post-fix `request_model_swap` returns a per-call
`threading.Event`; the worker drains the queue in order and sets
each event when ITS swap completes; `stop()` resolves every
outstanding event in teardown so callers never park.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "src" / "server") not in sys.path:
    sys.path.insert(0, str(REPO / "src" / "server"))


def test_request_model_swap_returns_per_call_request() -> None:
    """Pin the API contract: F-23-17 (commit-076) widened the return
    type from `threading.Event` to `_SwapRequest` so callers can
    read `.error` after `.completion.wait()`. Two requests carry
    DIFFERENT events (no shared broadcast); the original F-03-02
    invariant still holds at the event level."""
    from audio import engine

    eng = engine.RealtimeEngine(engine.EngineConfig())
    r1 = eng.request_model_swap(Path("/tmp/a.onnx"))
    r2 = eng.request_model_swap(Path("/tmp/b.onnx"))
    assert isinstance(r1, engine._SwapRequest)
    assert isinstance(r2, engine._SwapRequest)
    assert isinstance(r1.completion, threading.Event)
    assert isinstance(r2.completion, threading.Event)
    assert r1 is not r2 and r1.completion is not r2.completion, (
        "F-03-02: each request must return its OWN event; pre-fix "
        "the shared `_swap_done` broadcast released all waiters when "
        "the first swap completed"
    )
    assert r1.error is None and r2.error is None


def test_two_rapid_swaps_both_queue_and_apply_in_order() -> None:
    """Bug-class test for F-03-02. Two distinct rapid swaps must
    BOTH land in the queue and BOTH apply (in order). Pre-fix the
    second overwrote the first in the single-slot and only one
    actually loaded; voice A was silently dropped."""
    from audio import engine

    eng = engine.RealtimeEngine(engine.EngineConfig())
    applied: list[Path] = []

    # Capture the actual targets applied by the worker. Stub
    # _apply_one_swap to record + resolve immediately (no real
    # session loading needed for this test).
    def fake_apply(req: engine._SwapRequest) -> None:
        applied.append(req.target)
        eng._resolve_swap(req)

    eng._apply_one_swap = fake_apply  # type: ignore[method-assign]

    a = Path("/tmp/voiceA.onnx")
    b = Path("/tmp/voiceB.onnx")
    r1 = eng.request_model_swap(a)
    r2 = eng.request_model_swap(b)
    eng._maybe_swap_model()

    assert applied == [a, b], f"both queued swaps must apply, in order; got {applied}"
    assert r1.completion.is_set()
    assert r2.completion.is_set()


def test_each_completion_event_fires_only_when_its_own_swap_completes() -> None:
    """The fine-grained guarantee. Apply swap A; assert e1 is set
    but e2 is NOT (since B hasn't been drained yet)."""
    from audio import engine

    eng = engine.RealtimeEngine(engine.EngineConfig())

    apply_order: list[Path] = []

    def fake_apply(req: engine._SwapRequest) -> None:
        apply_order.append(req.target)
        eng._resolve_swap(req)

    eng._apply_one_swap = fake_apply  # type: ignore[method-assign]

    r1 = eng.request_model_swap(Path("/tmp/a.onnx"))
    r2 = eng.request_model_swap(Path("/tmp/b.onnx"))
    # Drain ONE at a time by injecting into _apply_one_swap with a
    # gate. The simplest way: pop the first off the queue manually
    # and apply it.
    req_a = eng._swap_queue.get_nowait()
    fake_apply(req_a)
    assert r1.completion.is_set(), "voice A's event must be set after voice A applies"
    assert not r2.completion.is_set(), (
        "voice B's event must NOT be set yet -- pre-fix the shared "
        "broadcast would falsely release it here"
    )
    # Now apply B.
    req_b = eng._swap_queue.get_nowait()
    fake_apply(req_b)
    assert r2.completion.is_set()


def test_stop_resolves_outstanding_swap_waiters_promptly() -> None:
    """Bug-class test for F-13-12. Queue a swap, then call stop().
    The per-call event MUST be set during teardown -- pre-fix `stop()`
    never touched `_swap_done`, so a JobRegistry waiter parked for
    the full 10 s timeout on the "queue a swap, toggle off"
    sequence."""
    from audio import engine

    eng = engine.RealtimeEngine(engine.EngineConfig())
    # Don't actually start the engine; just verify the teardown
    # path resolves outstanding swaps.
    req = eng.request_model_swap(Path("/tmp/whatever.onnx"))
    assert not req.completion.is_set()

    t0 = time.monotonic()
    eng.stop(timeout=0.5)
    elapsed = time.monotonic() - t0

    assert req.completion.is_set(), (
        "F-13-12: stop() must resolve outstanding swap events so "
        "callers don't park for the full 10s JobRegistry timeout"
    )
    # F-23-17 (commit-076): a swap resolved by `stop()` must carry an
    # error so callers know the swap was NOT applied.
    assert req.error is not None
    assert "stopped" in str(req.error).lower()
    assert elapsed < 2.0, (
        f"stop() must complete promptly even with pending swaps; got {elapsed:.2f}s"
    )


def test_request_model_swap_after_stop_resolves_immediately() -> None:
    """F-13-12 sister case: a caller that races stop() and submits
    a swap AFTER stop() completed gets an immediately-resolved
    event (with `error` set on the _SwapRequest)."""
    from audio import engine

    eng = engine.RealtimeEngine(engine.EngineConfig())
    eng.stop(timeout=0.1)
    # Engine is now stopped (`_stopped == True`).

    req = eng.request_model_swap(Path("/tmp/late.onnx"))
    assert req.completion.is_set(), "swap submitted to a stopped engine must resolve immediately"
    # F-23-17 (commit-076): the stopped-engine fast-fail must surface
    # via `req.error` so the TUI can route to record_error.
    assert req.error is not None
    assert "stopped" in str(req.error).lower()


def test_swap_queue_uses_queue_Queue_not_single_slot() -> None:
    """Structural pin: `_swap_queue` exists and is a `queue.Queue`,
    not a single-slot field. A future refactor that reintroduces the
    single-slot pattern would silently re-enable the F-03-02
    collapse-on-rapid-swap bug."""
    import queue as _queue

    from audio import engine

    eng = engine.RealtimeEngine(engine.EngineConfig())
    assert isinstance(eng._swap_queue, _queue.Queue)
    # The single-slot `_pending_model_swap` should be GONE.
    assert not hasattr(eng, "_pending_model_swap"), (
        "the pre-fix `_pending_model_swap` single-slot field must be "
        "removed -- its presence would mean someone re-introduced the "
        "F-03-02 collapse-on-rapid-swap bug class"
    )
    # The shared broadcast Event should be GONE too.
    assert not hasattr(eng, "_swap_done"), (
        "the pre-fix `_swap_done` shared broadcast Event must be "
        "removed -- callers must wait on per-call events instead"
    )
