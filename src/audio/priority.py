"""Shared thread-priority helper.

B47 / quality-013: pre-v0.8.0, `engine._apply_thread_priority` and
`inference_worker._try_set_rt_priority` were near-duplicates with
slightly different semantics (the engine wrote to `stats.last_error`,
the child returned a string). Both implemented the same SCHED_FIFO →
nice(-10) → warning fallback ladder.

This module is the single canonical implementation. Engine + child
both call `try_set_realtime_priority`. Engine reports failures via
`stats.priority_warnings` (B28); child reports via the child→parent
RESP_ERROR channel.

Original work — Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import os


def try_set_realtime_priority(label: str, *, priority: int = 60) -> str | None:
    """Request SCHED_FIFO at the given priority for the calling thread.

    Returns None on success. Returns a human-readable error string if
    the request was denied — the caller decides how to surface it
    (stats append, log, RESP_ERROR, etc.).

    Falls back through:
      1. SCHED_FIFO @ priority — preempts SCHED_OTHER, but needs
         CAP_SYS_NICE or RLIMIT_RTPRIO >= priority.
      2. nice(-10) — same scheduler class as before, just higher
         relative priority. Doesn't prevent same-class preemption
         (which is what we wanted), but at least doesn't crash.
      3. Returns a warning string if both fail.
    """
    try:
        param = os.sched_param(priority)
        os.sched_setscheduler(0, os.SCHED_FIFO, param)
        return None
    except (OSError, PermissionError, AttributeError) as rt_err:
        try:
            os.nice(-10)
            return None
        except (OSError, PermissionError) as nice_err:
            return (
                f"realtime_priority[{label}] denied "
                f"(SCHED_FIFO: {type(rt_err).__name__}: {rt_err}; "
                f"nice -10: {type(nice_err).__name__}: {nice_err}); "
                f"needs CAP_SYS_NICE or RLIMIT_RTPRIO >= {priority}"
            )


def try_set_affinity(core: int | None, label: str) -> str | None:
    """Pin the calling thread to `core`. None = no-op.

    Returns None on success or a warning string on failure. Same
    surface contract as `try_set_realtime_priority`.
    """
    if core is None:
        return None
    try:
        os.sched_setaffinity(0, {core})
        return None
    except (OSError, AttributeError) as e:
        return f"affinity[{label}] failed ({type(e).__name__}: {e})"
