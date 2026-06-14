"""the SIGTERM/SIGINT handler must be
async-signal-safe -- fast, fork-free, and re-entrant.

Pre-fix `_signal_handler_revert_lock` called `_revert_gpu_clock_lock()`
inline, which forks `sudo nvidia-smi` and can block the main thread up to
its 4 s timeout on *every* signal. A Ctrl-C could hang the process and, if
it landed mid-`subprocess.run`, deadlock.

These tests invoke the handler directly with `os.kill` stubbed (so the test
process is never actually signalled) and assert it does only the safe work:
flip `_stop_event`, record the signal, restore prior handlers, re-raise --
and crucially does NOT fork / run a subprocess.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import signal
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "src" / "server") not in sys.path:
    sys.path.insert(0, str(REPO / "src" / "server"))

from audio import engine  # noqa: E402


def _mk_engine() -> object:
    return engine.RealtimeEngine(engine.EngineConfig())


def test_signal_handler_does_no_fork_or_subprocess_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bug-class test. The handler must NOT call `_revert_gpu_clock_lock`
    or `_run_nvidia_smi` -- those fork `sudo nvidia-smi` and block up to 4 s.

    Pre-fix the handler calls `_revert_gpu_clock_lock()` inline, so the
    `unsafe` tripwire list is non-empty and `assert unsafe == []` fails.
    """
    eng = _mk_engine()
    # Real `_restore_prior_signal_handlers` runs on an empty dict (no-op);
    # only the genuinely-unsafe calls are tripwired.
    eng._prior_signal_handlers = {}

    unsafe: list[str] = []
    monkeypatch.setattr(
        eng, "_revert_gpu_clock_lock", lambda: unsafe.append("_revert_gpu_clock_lock")
    )
    monkeypatch.setattr(
        eng,
        "_run_nvidia_smi",
        lambda *a, **k: (unsafe.append("_run_nvidia_smi"), (False, ""))[1],
    )
    # Stub os.kill so the handler's re-raise never actually signals the
    # test process -- just record it.
    reraised: list[int] = []
    monkeypatch.setattr(engine.os, "kill", lambda _pid, sig: reraised.append(sig))

    eng._signal_handler_revert_lock(signal.SIGTERM, None)

    assert unsafe == [], f"signal handler ran unsafe fork/subprocess work: {unsafe}"
    # ...and it DID do the fast, safe work:
    assert eng._stop_event.is_set()
    assert eng._signal_received == signal.SIGTERM
    assert reraised == [signal.SIGTERM], "handler must re-raise to the prior handler"


def test_signal_handler_reentrancy_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """A repeated SIGTERM/SIGINT must not redo the handler's work -- the
    second invocation only re-raises (the review's "repeated SIGTERM"
    requirement). Pre-fix there is no guard, so the restore runs twice."""
    eng = _mk_engine()

    reraised: list[int] = []
    monkeypatch.setattr(engine.os, "kill", lambda _pid, sig: reraised.append(sig))
    restore_calls: list[bool] = []
    monkeypatch.setattr(
        eng,
        "_restore_prior_signal_handlers",
        lambda: restore_calls.append(True),
        raising=False,
    )

    eng._signal_handler_revert_lock(signal.SIGINT, None)
    eng._signal_handler_revert_lock(signal.SIGINT, None)  # repeated signal

    assert restore_calls == [True], "prior handlers restored exactly once, not per-signal"
    assert reraised == [signal.SIGINT, signal.SIGINT], "both invocations re-raise"


def test_revert_gpu_clock_lock_still_restores_handlers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The clean-stop path (`stop()` -> `_revert_gpu_clock_lock`) must still
    restore prior signal handlers -- the restore was split into
    `_restore_prior_signal_handlers` but the clean path still runs it."""
    eng = _mk_engine()
    # No clock lock active -> the nvidia-smi branch is skipped entirely.
    eng.stats.gpu_clock_lock_active = False
    restored: list[bool] = []
    monkeypatch.setattr(
        eng,
        "_restore_prior_signal_handlers",
        lambda: restored.append(True),
        raising=False,
    )

    eng._revert_gpu_clock_lock()

    assert restored == [True]
