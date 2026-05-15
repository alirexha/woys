"""review F-merged-027: evdev hotkey fd/thread leak fix.

Pre-fix `EvdevHotkey._loop` had three concrete defects:

1. **FD leak**: opened an `InputDevice` for EVERY entry in
   `evdev.list_devices()` and kept only the one exposing `EV_KEY`.
   The non-keyboard `InputDevice` objects were left open and leaked
   for the lifetime of the engine process. F-14-04's P1 class.
2. **Thread leak on stop**: `for ev in kbd.read_loop()` is a
   blocking generator. `stop()` set `_stop`, then `join(timeout=1.5)`
   always timed out because the loop only re-evaluates `_stop` on
   the next key event -- which never arrives during an idle
   shutdown.
3. **start() overwrite**: `start()` unconditionally cleared `_stop`
   and overwrote `_thread`. A second call while the first thread
   was still alive leaked the first thread.

Post-fix:
1. The non-kbd devices are closed in a `try/finally` block after the
   keyboard is picked.
2. `select.select([kbd.fileno()], [], [], 0.2)` wraps the read loop
   so the loop wakes every 200 ms to check `_stop`. `stop()` returns
   within ~200 ms even on an idle keyboard.
3. `start()` has an `if self._thread and self._thread.is_alive():
   return` guard, AND allocates a FRESH `_stop` Event per call.

These tests mock out `evdev` entirely so they run on systems where
the real module is not installed (CI / dev boxes without
`uv pip install -e .[evdev]`).

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import sys
import threading
import time
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "src" / "server") not in sys.path:
    sys.path.insert(0, str(REPO / "src" / "server"))


class _FakeInputDevice:
    """Stand-in for `evdev.InputDevice`. Tracks open/close state and
    capabilities so we can verify the fd-cleanup path."""

    def __init__(self, path: str, has_kbd: bool, all_devices: list[_FakeInputDevice]) -> None:
        self.path = path
        self._has_kbd = has_kbd
        self.closed = False
        self._all = all_devices
        self._all.append(self)

    def capabilities(self) -> dict[int, list[int]]:
        # Returning EV_KEY=1 in the dict signals "this is a keyboard".
        if self._has_kbd:
            return {1: [16, 17, 18]}  # EV_KEY mapped to fake key codes
        return {}

    def fileno(self) -> int:
        # Anything non-zero is fine for select.select() in our stubbed
        # loop -- our fake select() never actually reads from this.
        return 100

    def read(self) -> list[object]:
        return []  # no real events

    def close(self) -> None:
        self.closed = True


def _install_fake_evdev(monkeypatch: pytest.MonkeyPatch) -> list[_FakeInputDevice]:
    """Install a `evdev` stub into sys.modules and return the list
    of opened devices so the test can introspect them after."""
    opened: list[_FakeInputDevice] = []

    fake_module = types.SimpleNamespace()
    fake_module.list_devices = lambda: [
        "/dev/input/event0",
        "/dev/input/event1",
        "/dev/input/event2",
    ]
    fake_module.InputDevice = lambda path: _FakeInputDevice(
        path,
        has_kbd=(path == "/dev/input/event1"),  # only event1 is a keyboard
        all_devices=opened,
    )
    fake_module.ecodes = types.SimpleNamespace(EV_KEY=1)
    fake_module.categorize = lambda _ev: None
    fake_module.KeyEvent = type("KeyEvent", (), {"key_down": 1, "key_up": 0})

    monkeypatch.setitem(sys.modules, "evdev", fake_module)
    return opened


def _install_fake_select(monkeypatch: pytest.MonkeyPatch, never_ready: bool = True) -> None:
    """Install a fake `select.select` so we never block. `never_ready
    = True` returns no-ready (the loop spins on the timeout); a real
    test would feed events but this isolates the fd/lifecycle paths."""
    import select as real_select

    def fake_select(
        rs: list[int],
        ws: list[int],
        xs: list[int],
        timeout: float | None = None,
    ) -> tuple[list[int], list[int], list[int]]:
        if never_ready:
            # Honor the timeout so the loop wakes on _stop.
            time.sleep(timeout if timeout is not None else 0.0)
            return ([], [], [])
        return real_select.select(rs, ws, xs, timeout)

    monkeypatch.setattr("tui.hotkey.select.select", fake_select)


def test_stop_returns_promptly_on_idle_keyboard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bug-class test for the thread leak. With no key events
    arriving (the idle case), stop() must return within ~400 ms
    -- the select() timeout is 200 ms so the loop wakes within
    that window and observes `_stop`. Pre-fix `kbd.read_loop()`
    blocked indefinitely and join(timeout=1.5) always timed out."""
    from tui.hotkey import EvdevHotkey

    _install_fake_evdev(monkeypatch)
    _install_fake_select(monkeypatch, never_ready=True)

    hotkey = EvdevHotkey(on_press=lambda: None)
    hotkey.start()
    # Let the loop reach select().
    time.sleep(0.05)

    t0 = time.monotonic()
    hotkey.stop()
    elapsed = time.monotonic() - t0

    assert elapsed < 0.5, (
        f"F-merged-027: stop() must return within ~select-timeout "
        f"(0.2s) + epsilon; got {elapsed:.3f}s. Pre-fix the blocking "
        f"read_loop() forced a 1.5s join timeout and leaked the thread"
    )
    assert hotkey._thread is None or not hotkey._thread.is_alive(), (
        "hotkey thread must have actually exited (not just timed out)"
    )


def test_non_keyboard_devices_get_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bug-class test for the fd leak. `_loop` opens an InputDevice
    for every list_devices() entry, then picks the keyboard. The
    others MUST be closed so file descriptors don't leak."""
    from tui.hotkey import EvdevHotkey

    opened = _install_fake_evdev(monkeypatch)
    _install_fake_select(monkeypatch, never_ready=True)

    hotkey = EvdevHotkey(on_press=lambda: None)
    hotkey.start()
    # Let the loop run long enough to open + close the non-kbds.
    time.sleep(0.1)
    hotkey.stop()

    # 3 InputDevice instances were created (event0, event1, event2);
    # event1 is the keyboard -- the others (event0, event2) must be
    # closed by the loop. The keyboard ALSO gets closed by the
    # finally block on exit.
    assert len(opened) == 3
    non_kbd_closed = [d.closed for d in opened if not d._has_kbd]
    assert all(non_kbd_closed), (
        f"non-keyboard InputDevices must be closed; got close-state {non_kbd_closed}"
    )
    # The keyboard is closed in the finally block on loop exit.
    kbd = next(d for d in opened if d._has_kbd)
    assert kbd.closed, (
        "keyboard device must also be closed on loop exit so a re-start doesn't double-open"
    )


def test_double_start_is_idempotent_no_thread_leak(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bug-class test for fix #3. Calling start() twice while the first
    thread is alive must NOT spawn a second thread. Pre-fix the second
    call overwrote `_thread` and the first thread was orphaned."""
    from tui.hotkey import EvdevHotkey

    _install_fake_evdev(monkeypatch)
    _install_fake_select(monkeypatch, never_ready=True)

    hotkey = EvdevHotkey(on_press=lambda: None)
    hotkey.start()
    first_thread = hotkey._thread
    time.sleep(0.05)

    hotkey.start()  # second call
    assert hotkey._thread is first_thread, (
        "F-merged-027 fix #3: a second start() while the first thread "
        "is alive must be a no-op (no second thread spawned)"
    )

    hotkey_threads_alive = [
        t for t in threading.enumerate() if t.name == "woys-hotkey" and t.is_alive()
    ]
    assert len(hotkey_threads_alive) == 1

    hotkey.stop()


def test_fresh_stop_event_per_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """start() after stop() must allocate a FRESH `_stop` Event so the
    new loop doesn't inherit a `set()` from the prior teardown."""
    from tui.hotkey import EvdevHotkey

    _install_fake_evdev(monkeypatch)
    _install_fake_select(monkeypatch, never_ready=True)

    hotkey = EvdevHotkey(on_press=lambda: None)
    hotkey.start()
    first_stop = hotkey._stop
    time.sleep(0.05)
    hotkey.stop()
    assert first_stop.is_set()

    hotkey.start()
    # New Event instance, not the one we just set().
    assert hotkey._stop is not first_stop
    assert not hotkey._stop.is_set()
    hotkey.stop()


def test_no_keyboard_found_returns_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sanity: if no input device exposes EV_KEY, the loop logs a
    warning and exits cleanly -- no thread leak even on the no-
    keyboard path."""
    opened: list[_FakeInputDevice] = []

    fake_module = types.SimpleNamespace()
    fake_module.list_devices = lambda: ["/dev/input/event0"]
    fake_module.InputDevice = lambda path: _FakeInputDevice(path, has_kbd=False, all_devices=opened)
    fake_module.ecodes = types.SimpleNamespace(EV_KEY=1)
    monkeypatch.setitem(sys.modules, "evdev", fake_module)
    _install_fake_select(monkeypatch, never_ready=True)

    from tui.hotkey import EvdevHotkey

    hotkey = EvdevHotkey(on_press=lambda: None)
    hotkey.start()
    time.sleep(0.1)
    # Thread should have exited on its own (no kbd found).
    assert hotkey._thread is not None
    assert not hotkey._thread.is_alive()
    # And the one opened device should still get closed.
    assert opened and opened[0].closed
    hotkey.stop()
