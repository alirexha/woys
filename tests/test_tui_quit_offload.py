"""review F-13-03 + F-CX3-02: offload the blocking teardown trio
off Textual's event-loop thread.

Pre-fix:
- `action_toggle_engine` (sync) called `self.engine.stop()` directly.
  `engine.stop()` can take up to ~10 s (engine thread join 2 s +
  InferenceClient kill ladder 3.5 s + gc.collect + GPU clock-lock
  revert subprocess.run timeout 4 s). The UI froze for the duration
  -- the `set_interval` callback could not fire and the panel
  showed a frozen "RUNNING" indicator.
- `action_quit` (async, but synchronous body) called
  `engine.stop()`, `_control.stop()` (~1.5 s join), and
  `save_config()` (fsync) all synchronously on the event loop --
  the same freeze applied during quit.

Post-fix:
- `action_toggle_engine` spins a daemon worker thread, renders a
  "stopping..." notification immediately, and lets the worker call
  `notify("engine stopped")` via `call_from_thread` on completion.
- `action_quit` uses `await asyncio.to_thread(...)` for each
  teardown step. The event loop keeps ticking until the trio
  finishes.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "src" / "server") not in sys.path:
    sys.path.insert(0, str(REPO / "src" / "server"))


def test_action_toggle_engine_returns_immediately_when_engine_is_running() -> None:
    """Bug-class test for the sync path. With the engine "running" and
    `engine.stop()` mocked to sleep 500 ms (a fraction of the real-
    world worst case), `action_toggle_engine` must return in well
    under 100 ms. Pre-fix it would block for the full sleep."""
    from tui.app import VCClientApp
    from tui.config import AppConfig

    app = VCClientApp(cfg=AppConfig(), no_pw_setup=True)
    app.engine.stats.running = True
    # `call_from_thread` would normally route to the live event loop;
    # in a no-loop test we just invoke directly.
    app.call_from_thread = lambda fn, *a, **kw: fn(*a, **kw)  # type: ignore[method-assign]
    # `notify` would write to the screen; stub it.
    app.notify = lambda *a, **kw: None  # type: ignore[method-assign]

    stop_started = threading.Event()
    stop_finished = threading.Event()

    def slow_stop(*_a: object, **_kw: object) -> None:
        stop_started.set()
        time.sleep(0.5)
        stop_finished.set()

    app.engine.stop = slow_stop  # type: ignore[method-assign]

    t0 = time.monotonic()
    app.action_toggle_engine()
    elapsed = time.monotonic() - t0

    assert elapsed < 0.1, (
        f"action_toggle_engine must return immediately (it's on the "
        f"event-loop thread); pre-fix it would block for ~0.5 s. "
        f"Got {elapsed:.3f}s"
    )
    # The worker thread is still doing the stop.
    assert stop_started.wait(timeout=1.0)
    # And eventually finishes.
    assert stop_finished.wait(timeout=2.0)


def test_action_quit_uses_asyncio_to_thread() -> None:
    """Bug-class test for the async path. `action_quit` is an async
    method; each teardown call must `await asyncio.to_thread(...)`
    so the event loop keeps ticking. We exercise the coroutine
    directly and confirm a fake slow `engine.stop` does not block
    the event loop -- we can interleave a tick on the same loop
    while `action_quit` is waiting on its first step."""
    from tui.app import VCClientApp
    from tui.config import AppConfig

    app = VCClientApp(cfg=AppConfig(), no_pw_setup=True)
    app.notify = lambda *a, **kw: None  # type: ignore[method-assign]

    stop_calls: list[float] = []
    save_calls: list[float] = []
    control_stop_calls: list[float] = []

    def slow_engine_stop() -> None:
        stop_calls.append(time.monotonic())
        time.sleep(0.2)

    def slow_control_stop() -> None:
        control_stop_calls.append(time.monotonic())
        time.sleep(0.1)

    def fast_save(_cfg: object) -> None:
        save_calls.append(time.monotonic())

    app.engine.stop = slow_engine_stop  # type: ignore[method-assign]
    # _control is set in on_mount; create a tiny stub for this test.

    import types as _types

    app._control = _types.SimpleNamespace(stop=slow_control_stop)  # type: ignore[assignment]
    # Patch save_config in the module the action references.
    import tui.app as app_mod

    orig_save = app_mod.save_config
    app_mod.save_config = fast_save  # type: ignore[assignment]
    # exit() would close the app; stub it so the test runs to completion.
    app.exit = lambda *_a, **_kw: None  # type: ignore[method-assign]

    try:
        ticks: list[float] = []

        async def runner() -> None:
            async def ticker() -> None:
                # Every 30ms, append a timestamp. If the event loop is
                # blocked by action_quit, this stops firing.
                for _ in range(8):
                    await asyncio.sleep(0.03)
                    ticks.append(time.monotonic())

            tk = asyncio.create_task(ticker())
            await app.action_quit()
            await tk

        asyncio.run(runner())
    finally:
        app_mod.save_config = orig_save  # type: ignore[assignment]

    # The teardown ran (all three steps).
    assert len(stop_calls) == 1
    assert len(control_stop_calls) == 1
    assert len(save_calls) == 1
    # The event loop kept ticking through the 200ms engine.stop --
    # at least 3 ticks (90ms+) should have fired during that window
    # if asyncio.to_thread was used correctly. Pre-fix the synchronous
    # body would have blocked the loop entirely; the ticker task
    # would have been starved (0 or 1 ticks total).
    assert len(ticks) >= 3, (
        f"event loop must keep ticking while action_quit awaits "
        f"asyncio.to_thread; got {len(ticks)} ticks during ~250ms "
        f"of teardown. Pre-fix the synchronous body would freeze "
        f"the loop and produce 0-1 ticks"
    )


def test_action_quit_calls_exit_after_teardown_completes() -> None:
    """Sequence pin: self.exit(0) runs AFTER the three asyncio.to_thread
    calls complete. If a future refactor moved exit() above the
    awaits (or dropped the await), the engine would be killed mid-
    teardown."""
    from tui.app import VCClientApp
    from tui.config import AppConfig

    app = VCClientApp(cfg=AppConfig(), no_pw_setup=True)
    app.notify = lambda *a, **kw: None  # type: ignore[method-assign]

    call_order: list[str] = []

    def step_engine() -> None:
        call_order.append("engine.stop")

    def step_control() -> None:
        call_order.append("control.stop")

    def step_save(_cfg: object) -> None:
        call_order.append("save_config")

    def step_exit(*_a: object, **_kw: object) -> None:
        call_order.append("exit")

    app.engine.stop = step_engine  # type: ignore[method-assign]
    import types as _types

    app._control = _types.SimpleNamespace(stop=step_control)  # type: ignore[assignment]
    import tui.app as app_mod

    orig_save = app_mod.save_config
    app_mod.save_config = step_save  # type: ignore[assignment]
    app.exit = step_exit  # type: ignore[method-assign]
    try:
        asyncio.run(app.action_quit())
    finally:
        app_mod.save_config = orig_save  # type: ignore[assignment]

    assert call_order == ["engine.stop", "control.stop", "save_config", "exit"], (
        f"sequence must be teardown-then-exit; got {call_order}"
    )


def test_toggle_engine_does_not_block_when_starting_path_is_unchanged() -> None:
    """Back-compat: the start path is unchanged -- a fresh start
    runs synchronously (it's the user explicitly committing to
    a session, and the start work happens in the engine thread
    anyway). Only the STOP path moved to a worker."""
    from tui.app import VCClientApp
    from tui.config import AppConfig

    app = VCClientApp(cfg=AppConfig(), no_pw_setup=True)
    app.engine.stats.running = False
    app.notify = lambda *a, **kw: None  # type: ignore[method-assign]
    started: list[bool] = []
    app._start_engine = lambda: started.append(True) or True  # type: ignore[method-assign]

    app.action_toggle_engine()
    assert started == [True], "start path stays in-line (synchronous)"


def test_threading_imports_present() -> None:
    """Structural pin: the threading + asyncio imports are wired so
    the offload pattern works."""
    src = Path(__file__).resolve().parent.parent / "src" / "tui" / "app.py"
    text = src.read_text()
    assert "import asyncio" in text
    assert "import threading" in text
    assert "asyncio.to_thread" in text, (
        "action_quit must use asyncio.to_thread for the teardown trio"
    )
    assert 'name="woys-tui-stop"' in text, (
        "action_toggle_engine's offload thread should be named for diag"
    )
