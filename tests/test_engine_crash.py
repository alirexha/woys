"""review F-17-06 (P1): the headless path must not be blind to a
worker-thread crash.

If `_run_loop` exits via an unhandled exception, `EngineStats.crashed` must
be set (distinct from `running == False`, which a clean `stop()` also
produces). `cmd_engine`'s loop reads `eng.stats.crashed` to break early and
exit non-zero -- pre-fix it checked only `stop["now"]` / `deadline` and
printed frozen `chunks=0` stats for the full `--seconds`.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import contextlib
import sys
import threading
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "src" / "server") not in sys.path:
    sys.path.insert(0, str(REPO / "src" / "server"))

from audio import engine  # noqa: E402


def test_engine_stats_crashed_defaults_false() -> None:
    """A fresh engine has not crashed."""
    eng = engine.RealtimeEngine(engine.EngineConfig())
    assert eng.stats.crashed is False


def test_run_loop_sets_crashed_on_unhandled_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bug-class test. An unhandled exception inside `_run_loop` must
    leave `EngineStats.crashed` True (and not propagate -- the except
    handler catches it). Pre-fix `crashed` did not exist, so accessing it
    raises AttributeError; the headless loop had no crash signal to read.
    """
    eng = engine.RealtimeEngine(engine.EngineConfig())

    def _boom() -> object:
        raise RuntimeError("pacat blew up")

    # `_open_pacat` is the first call inside `_run_loop`'s guarded block;
    # making it raise drives the engine straight into the except handler.
    monkeypatch.setattr(eng, "_open_pacat", _boom)
    assert eng.stats.crashed is False

    eng._run_loop()  # must NOT propagate -- the except handler catches it

    assert eng.stats.crashed is True, "_run_loop crash must set stats.crashed"
    assert eng.stats.running is False
    assert eng.stats.last_error and "RuntimeError" in eng.stats.last_error


def test_start_resets_crashed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A prior crash must not stick across a fresh `start()`. `start()`
    clears `crashed` before spawning the worker (verified without actually
    running the engine: stub the heavy steps)."""
    eng = engine.RealtimeEngine(engine.EngineConfig())
    eng.stats.crashed = True  # simulate a prior crashed run

    # Stub everything start() does after the `crashed = False` reset so the
    # test needs no GPU / PipeWire and never spawns the real worker thread.
    monkeypatch.setattr(eng, "_install_signal_handlers", lambda: None)
    monkeypatch.setattr(eng, "_warn_if_default_sink_hijacked", lambda: None)
    monkeypatch.setattr(eng, "_resolve_anti_jitter_flags", lambda: (False, False))
    monkeypatch.setattr(eng, "_ensure_sessions", lambda: None)
    monkeypatch.setattr(eng, "_warmup_realtime_pipeline", lambda: None)

    class _DummyThread:
        def start(self) -> None:
            pass

        def is_alive(self) -> bool:
            return False

    monkeypatch.setattr(engine.threading, "Thread", lambda *a, **k: _DummyThread())

    eng.start()
    assert eng.stats.crashed is False, "start() must clear a stale crashed flag"


# ---- review F-17-10: playback-helper liveness check + respawn cap -----


def test_spawn_checked_raises_when_helper_exits_immediately() -> None:
    """A playback helper that exits immediately on spawn must raise -- not
    be returned as a dead Popen the watchdog respawns forever."""
    eng = engine.RealtimeEngine(engine.EngineConfig())
    with pytest.raises(RuntimeError, match="exited immediately"):
        eng._spawn_checked(["false"])  # coreutils `false` exits 1 instantly


def test_spawn_checked_returns_a_live_proc() -> None:
    """A helper that stays alive past the poll window is returned normally."""
    eng = engine.RealtimeEngine(engine.EngineConfig())
    proc = eng._spawn_checked(["sleep", "10"])
    try:
        assert proc.poll() is None  # still alive
    finally:
        proc.terminate()
        proc.wait(timeout=2)


def test_watchdog_loop_caps_respawn_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """A playback helper that can never respawn must stop the engine after
    `_PLAYER_RESPAWN_CAP` attempts. Pre-fix the watchdog retried forever
    (`running=True`, zero audio); pre-fix this test HANGS, and the
    thread-join timeout turns that hang into a clean failure."""
    eng = engine.RealtimeEngine(engine.EngineConfig())

    class _DeadProc:
        returncode = 1

        def poll(self) -> int:
            return 1  # always dead -> every tick triggers a respawn

    eng._pacat_proc = _DeadProc()  # type: ignore[assignment]

    def _always_fails() -> object:
        raise RuntimeError("helper permanently broken")

    monkeypatch.setattr(eng, "_open_pacat", _always_fails)
    # Spin the loop fast: no real waiting.
    monkeypatch.setattr(eng._pacat_dead_event, "wait", lambda timeout=None: None)
    monkeypatch.setattr(engine.time, "sleep", lambda *_a: None)

    t = threading.Thread(target=eng._watchdog_loop, daemon=True)
    t.start()
    t.join(timeout=5.0)
    alive = t.is_alive()
    # Stop the spinner regardless -- matters on pre-fix code where the loop
    # is uncapped and would otherwise spin hot for the rest of the session.
    eng._stop_event.set()
    t.join(timeout=2.0)

    assert not alive, "watchdog did not terminate -- the respawn loop is uncapped"
    assert eng._stop_event.is_set(), "hitting the cap must set _stop_event"
    assert eng.stats.last_error and "respawned" in eng.stats.last_error


# ---- review F-14-02: parent-death signal on playback-helper spawns ----


def test_spawn_checked_arms_parent_death_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    """`_spawn_checked` must pass `preexec_fn=_set_pdeathsig` so a `kill -9`
    of the engine doesn't orphan the playback subprocess. Pre-fix no
    `preexec_fn` was passed (and `_set_pdeathsig` did not exist)."""

    class _StubProc:
        def poll(self) -> None:
            return None  # alive past the liveness window

    captured: dict[str, object] = {}

    def fake_popen(_cmd: object, **kwargs: object) -> _StubProc:
        captured.update(kwargs)
        return _StubProc()

    eng = engine.RealtimeEngine(engine.EngineConfig())
    monkeypatch.setattr(engine.subprocess, "Popen", fake_popen)

    eng._spawn_checked(["true"])

    assert captured.get("preexec_fn") is engine._set_pdeathsig, (
        "_spawn_checked must arm PR_SET_PDEATHSIG via preexec_fn"
    )


def test_set_pdeathsig_kills_child_when_parent_dies() -> None:
    """Real mechanism test: a process spawned with
    `preexec_fn=_set_pdeathsig` is SIGTERM'd when its parent exits.

    Uses a grandchild so the test process is never the dying parent: an
    intermediate Python process spawns `sleep 30` with the preexec_fn,
    prints the grandchild PID, then exits -- PR_SET_PDEATHSIG should then
    take the grandchild down.
    """
    import os
    import signal
    import subprocess
    import time

    code = (
        f"import subprocess, sys\n"
        f"sys.path.insert(0, {str(REPO / 'src')!r})\n"
        f"sys.path.insert(0, {str(REPO / 'src' / 'server')!r})\n"
        f"from audio.engine import _set_pdeathsig\n"
        f"p = subprocess.Popen(['sleep', '30'], preexec_fn=_set_pdeathsig)\n"
        f"print(p.pid); sys.stdout.flush()\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=15)
    assert out.stdout.strip(), f"intermediate process produced no PID (stderr: {out.stderr})"
    grandchild_pid = int(out.stdout.strip())

    # The intermediate has exited; PR_SET_PDEATHSIG should SIGTERM the
    # grandchild. Poll for it to disappear.
    deadline = time.time() + 5.0
    alive = True
    while time.time() < deadline:
        try:
            os.kill(grandchild_pid, 0)
        except ProcessLookupError:
            alive = False
            break
        time.sleep(0.05)

    if alive:
        with contextlib.suppress(ProcessLookupError):
            os.kill(grandchild_pid, signal.SIGKILL)  # cleanup before failing
        pytest.fail("grandchild survived parent death -- PR_SET_PDEATHSIG not armed")
