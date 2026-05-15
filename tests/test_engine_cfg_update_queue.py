"""review F-merged-017 commit-040b: multi-field cfg apply
consistency.

Pre-fix `_apply_profile_named` wrote four `engine.cfg.X = ...`
assignments one at a time:
  engine.cfg.f0_up_key   = ...
  engine.cfg.sid         = ...
  engine.cfg.monitor     = ...
  engine.cfg.input_gain_db = ...

The engine reads those same fields at scattered points within a single
chunk (`cfg.monitor` at engine.py:3843 and again at :4021; `cfg.input_
gain_db` at :3908; `cfg.f0_up_key` / `cfg.sid` inside `_infer`). An
apply interleaved with a chunk left the engine reading a half-applied
composite: new monitor flag with old pitch, or new pitch with old
input_gain. The bug class is multi-field *consistency*, distinct from
the lost-update class of commit-040a.

Post-fix callers stage a dict of updates via
`engine.request_cfg_update({...})`; the engine drains the dict
atomically at the top of each chunk loop iteration via
`_maybe_apply_pending_cfg()`. Within a chunk the engine sees a
consistent snapshot.

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


def test_cfg_lock_and_pending_updates_exist() -> None:
    """Pin the new infrastructure surface."""
    from audio import engine

    eng = engine.RealtimeEngine(engine.EngineConfig())
    assert hasattr(eng, "_cfg_lock"), "engine must expose _cfg_lock for tests / introspection"
    assert hasattr(eng, "_pending_cfg_updates")
    assert eng._pending_cfg_updates == {}


def test_request_cfg_update_stages_dict_does_not_apply_immediately() -> None:
    """request_cfg_update is just the staging step. The apply happens
    only when _maybe_apply_pending_cfg runs (chunk boundary)."""
    from audio import engine

    eng = engine.RealtimeEngine(engine.EngineConfig())
    original_f0 = eng.cfg.f0_up_key

    eng.request_cfg_update({"f0_up_key": 17, "sid": 99})
    # Staged, not applied:
    assert eng.cfg.f0_up_key == original_f0, (
        "request_cfg_update must NOT mutate engine.cfg directly -- "
        "the apply is deferred to the chunk boundary"
    )
    assert eng._pending_cfg_updates == {"f0_up_key": 17, "sid": 99}


def test_maybe_apply_pending_cfg_drains_atomically() -> None:
    """The drain step applies all queued fields at once and clears
    the queue. Calling again with an empty queue is a no-op."""
    from audio import engine

    eng = engine.RealtimeEngine(engine.EngineConfig())
    eng.request_cfg_update({"f0_up_key": 7, "sid": 42, "input_gain_db": -3.0})
    eng._maybe_apply_pending_cfg()

    assert eng.cfg.f0_up_key == 7
    assert eng.cfg.sid == 42
    assert eng.cfg.input_gain_db == -3.0
    assert eng._pending_cfg_updates == {}, "drain must clear the queue"

    # Second drain is a no-op (cheap path -- one bool check inside a lock).
    eng._maybe_apply_pending_cfg()
    assert eng.cfg.f0_up_key == 7  # unchanged


def test_unknown_field_in_update_is_ignored_not_raises() -> None:
    """A caller staging an update for a non-existent cfg field (e.g.,
    because a future EngineConfig renamed it) must NOT raise -- the
    drain step skips unknown fields. Defensive against version skew."""
    from audio import engine

    eng = engine.RealtimeEngine(engine.EngineConfig())
    eng.request_cfg_update({"f0_up_key": 9, "nonexistent_future_field": "x"})
    eng._maybe_apply_pending_cfg()
    assert eng.cfg.f0_up_key == 9
    assert not hasattr(eng.cfg, "nonexistent_future_field")


def test_multi_field_profile_apply_is_atomic_no_half_applied_state() -> None:
    """Bug-class test (the verdict's stated case).

    Simulate the engine's per-chunk read pattern. Worker thread reads
    the four runtime-tunable fields one at a time at scattered points
    (just like the real `_run_loop`). A separate TUI thread issues a
    profile-apply via `request_cfg_update`. After the worker calls
    `_maybe_apply_pending_cfg` at its chunk boundary, every
    subsequent read in the same chunk MUST see the SAME profile --
    no mix of old + new fields.

    Pre-fix the TUI thread would mutate the four fields one at a time
    AFTER the worker had already started reading them; the worker
    would observe e.g. new f0_up_key with old monitor.
    """
    from audio import engine

    eng = engine.RealtimeEngine(engine.EngineConfig())
    # Initial state (Profile A).
    eng.cfg.f0_up_key = 5
    eng.cfg.sid = 1
    eng.cfg.monitor = True
    eng.cfg.input_gain_db = 0.0

    # Profile B values we'll apply atomically.
    profile_b = {
        "f0_up_key": -7,
        "sid": 2,
        "monitor": False,
        "input_gain_db": -10.0,
    }

    observed_snapshots: list[dict[str, object]] = []
    chunk_start = threading.Event()
    apply_done = threading.Event()

    def worker_chunk() -> None:
        """Imitates a single _run_loop iteration. Reads each field
        at a scattered point so a non-atomic apply WOULD be caught."""
        # Chunk boundary: drain pending cfg.
        eng._maybe_apply_pending_cfg()
        # Take a 4-stage snapshot like the real engine does.
        snap = {}
        snap["f0_up_key"] = eng.cfg.f0_up_key
        chunk_start.set()
        # Let the TUI thread try (and fail) to inject mid-chunk.
        apply_done.wait(timeout=0.5)
        time.sleep(0.01)  # 10 ms of "chunk processing"
        snap["sid"] = eng.cfg.sid
        snap["monitor"] = eng.cfg.monitor
        snap["input_gain_db"] = eng.cfg.input_gain_db
        observed_snapshots.append(snap)

    def tui_apply() -> None:
        """Imitates the TUI thread issuing a profile-apply."""
        chunk_start.wait(timeout=1.0)
        # Stage the update -- this is the only call. The actual
        # apply waits until _maybe_apply_pending_cfg fires in the
        # NEXT chunk.
        eng.request_cfg_update(profile_b)
        apply_done.set()

    t_worker = threading.Thread(target=worker_chunk, name="chunk-A")
    t_tui = threading.Thread(target=tui_apply, name="tui-apply")
    t_worker.start()
    t_tui.start()
    t_worker.join(timeout=2.0)
    t_tui.join(timeout=2.0)

    # The worker chunk started with profile A; its in-chunk snapshot
    # must be consistent (all profile A) since the apply was staged
    # but never drained inside this chunk.
    snap = observed_snapshots[0]
    assert snap == {
        "f0_up_key": 5,
        "sid": 1,
        "monitor": True,
        "input_gain_db": 0.0,
    }, (
        f"the in-chunk snapshot must reflect Profile A wholesale; "
        f"a half-applied composite (Profile A + Profile B mixed) is "
        f"the F-merged-017 commit-040b bug class. Got: {snap}"
    )

    # The apply is now pending; the NEXT chunk's drain picks it up.
    eng._maybe_apply_pending_cfg()
    assert eng.cfg.f0_up_key == -7
    assert eng.cfg.sid == 2
    assert eng.cfg.monitor is False
    assert eng.cfg.input_gain_db == -10.0


def test_concurrent_request_cfg_update_merges_correctly() -> None:
    """Two threads queue different sets of fields. Both staged updates
    must survive until the drain. The drain applies a coherent merge
    (later writer wins on collisions; non-colliding fields are
    preserved)."""
    from audio import engine

    eng = engine.RealtimeEngine(engine.EngineConfig())
    barrier = threading.Barrier(2)

    def tui_a() -> None:
        barrier.wait()
        eng.request_cfg_update({"f0_up_key": 11, "sid": 1})

    def tui_b() -> None:
        barrier.wait()
        eng.request_cfg_update({"monitor": True, "input_gain_db": -5.0})

    threads = [threading.Thread(target=t) for t in (tui_a, tui_b)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=2.0)

    # Both threads' fields landed in the queue (no overlap, so
    # ordering doesn't matter).
    eng._maybe_apply_pending_cfg()
    assert eng.cfg.f0_up_key == 11
    assert eng.cfg.sid == 1
    assert eng.cfg.monitor is True
    assert eng.cfg.input_gain_db == -5.0


def test_run_loop_calls_maybe_apply_pending_cfg_at_top_of_chunk() -> None:
    """Structural pin: the engine's `_run_loop` MUST call
    `_maybe_apply_pending_cfg()` at the top of each chunk iteration
    (next to `_maybe_swap_model`). A future refactor that drops the
    call would silently reintroduce the consistency bug; this pin
    surfaces it as a test failure."""
    src = Path(__file__).resolve().parent.parent / "src" / "audio" / "engine.py"
    text = src.read_text()
    assert "self._maybe_apply_pending_cfg()" in text, (
        "_run_loop must drain pending cfg updates at the chunk boundary; see commit-040b doc"
    )
    # The call must occur near `_maybe_swap_model` so they share the
    # same chunk-boundary semantics.
    idx_swap = text.find("self._maybe_swap_model()")
    idx_apply = text.find("self._maybe_apply_pending_cfg()")
    assert idx_swap > 0 and idx_apply > 0
    # Within ~500 chars of each other (i.e., adjacent in the run loop).
    assert abs(idx_apply - idx_swap) < 1000, (
        "_maybe_apply_pending_cfg must be co-located with "
        "_maybe_swap_model at the chunk-boundary hook"
    )
