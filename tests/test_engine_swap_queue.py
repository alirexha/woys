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


def test_request_model_swap_returns_per_call_event() -> None:
    """Pin the new API contract: `request_model_swap` returns a
    `threading.Event`, not None. Two requests return DIFFERENT
    events (no shared broadcast)."""
    from audio import engine

    eng = engine.RealtimeEngine(engine.EngineConfig())
    e1 = eng.request_model_swap(Path("/tmp/a.onnx"))
    e2 = eng.request_model_swap(Path("/tmp/b.onnx"))
    assert isinstance(e1, threading.Event)
    assert isinstance(e2, threading.Event)
    assert e1 is not e2, (
        "F-03-02: each request must return its OWN event; pre-fix "
        "the shared `_swap_done` broadcast released all waiters when "
        "the first swap completed"
    )


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
    e1 = eng.request_model_swap(a)
    e2 = eng.request_model_swap(b)
    eng._maybe_swap_model()

    assert applied == [a, b], f"both queued swaps must apply, in order; got {applied}"
    assert e1.is_set()
    assert e2.is_set()


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

    e1 = eng.request_model_swap(Path("/tmp/a.onnx"))
    e2 = eng.request_model_swap(Path("/tmp/b.onnx"))
    # Drain ONE at a time by injecting into _apply_one_swap with a
    # gate. The simplest way: pop the first off the queue manually
    # and apply it.
    req_a = eng._swap_queue.get_nowait()
    fake_apply(req_a)
    assert e1.is_set(), "voice A's event must be set after voice A applies"
    assert not e2.is_set(), (
        "voice B's event must NOT be set yet -- pre-fix the shared "
        "broadcast would falsely release it here"
    )
    # Now apply B.
    req_b = eng._swap_queue.get_nowait()
    fake_apply(req_b)
    assert e2.is_set()


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
    completion = eng.request_model_swap(Path("/tmp/whatever.onnx"))
    assert not completion.is_set()

    t0 = time.monotonic()
    eng.stop(timeout=0.5)
    elapsed = time.monotonic() - t0

    assert completion.is_set(), (
        "F-13-12: stop() must resolve outstanding swap events so "
        "callers don't park for the full 10s JobRegistry timeout"
    )
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

    completion = eng.request_model_swap(Path("/tmp/late.onnx"))
    assert completion.is_set(), "swap submitted to a stopped engine must resolve immediately"


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
