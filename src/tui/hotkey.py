"""Opt-in evdev global hotkey.

Per Q7 / project brief: evdev raw-grab tripping VAC heuristics is a real risk
for CS2 users. The default control path is the Unix socket plus the WM-level
shortcut wrapper - this module is only loaded when the user explicitly opts
in via `enable_evdev_hotkey = true` in config.toml.

Setup
-----
- Install the optional dep: `uv pip install -e .[evdev]`
- Add the user to the `input` group:  `sudo usermod -aG input $USER && reboot`
- (Or: ship a `udev` rule under `pkg/udev/` granting r/w on `/dev/input/event*`
  to the `woys` group; see `docs/TROUBLESHOOTING.md`.)
"""

from __future__ import annotations

import logging
import select
import threading
from collections.abc import Callable

logger = logging.getLogger("woys.hotkey")

DEFAULT_COMBO = ("KEY_LEFTCTRL", "KEY_LEFTALT", "KEY_V")

# poll interval for the select() wake-up loop.
# `kbd.read_loop()` is a blocking generator that only yields on key events;
# pre-fix `stop()` then `.join(timeout=1.5)` always timed out and leaked
# the thread (no key event ever arrived during teardown). Polling with a
# 200 ms select() timeout means `stop()` returns in at most ~200 ms even
# on an idle keyboard. The cost is one wake-up every 200 ms when idle,
# which is negligible (the cost of being polite to /dev/input/event*).
_SELECT_TIMEOUT_S = 0.2


class EvdevHotkey:
    """Background thread that fires `on_press` whenever the combo lights up.

    Imports `evdev` lazily so the rest of the app remains usable without it.

    pre-fix bugs:
    - `_loop` opened an `InputDevice` for EVERY entry in `evdev.list_devices()`
      and kept only the one that exposed `EV_KEY` -- the rest were left
      open (file descriptor leak; F-14-04's P1 class).
    - `read_loop()` is a blocking generator that only yields on key events,
      so `stop()` + `.join(timeout=1.5)` always timed out and the thread
      leaked. The user had to send a key event to unstick teardown.
    - `start()` unconditionally cleared `_stop` and overwrote `_thread`,
      so a second call leaked the first thread.
    """

    def __init__(
        self, on_press: Callable[[], None], combo: tuple[str, ...] = DEFAULT_COMBO
    ) -> None:
        self.on_press = on_press
        self.combo = combo
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        # F-merged-027 fix #3: idempotent. A second start() while the first
        # thread is alive is a no-op; pre-fix it overwrote the field and
        # leaked the thread.
        if self._thread is not None and self._thread.is_alive():
            return
        try:
            import evdev  # noqa: F401  (just probe the import)
        except ImportError as e:
            logger.warning(
                "evdev not installed; skipping global hotkey. "
                "Install with: uv pip install -e '.[evdev]' (%s)",
                e,
            )
            return
        # Fresh `_stop` per start so a re-start cannot inherit a stale
        # set() from a prior teardown.
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="woys-hotkey", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.5)

    def _loop(self) -> None:
        import evdev

        # F-merged-027 fix #1: open all devices in a list, pick the keyboard,
        # then close every device we didn't pick. Pre-fix the non-kbd
        # devices' file descriptors leaked for the lifetime of the engine
        # process.
        try:
            devices = [evdev.InputDevice(p) for p in evdev.list_devices()]
        except PermissionError:
            logger.error(
                "evdev: permission denied opening /dev/input/event*. "
                "Add user to the `input` group or use the udev rule (docs/TROUBLESHOOTING.md)."
            )
            return
        kbd = next(
            (d for d in devices if evdev.ecodes.EV_KEY in d.capabilities()),
            None,
        )
        for d in devices:
            if d is kbd:
                continue
            try:
                d.close()
            except OSError:
                logger.debug("evdev: close on non-kbd device failed", exc_info=True)
        if kbd is None:
            logger.warning("evdev: no keyboard device found")
            return

        held: set[str] = set()
        target = set(self.combo)
        try:
            kbd_fd = kbd.fileno()
            while not self._stop.is_set():
                # F-merged-027 fix #2: poll select() with a timeout so the
                # loop wakes every _SELECT_TIMEOUT_S to check `_stop` even
                # on an idle keyboard. Pre-fix `for ev in kbd.read_loop()`
                # blocked indefinitely waiting for a key; stop() then
                # join(timeout=1.5) always timed out and the thread leaked.
                ready, _, _ = select.select([kbd_fd], [], [], _SELECT_TIMEOUT_S)
                if not ready:
                    continue
                for ev in kbd.read():
                    if self._stop.is_set():
                        break
                    if ev.type != evdev.ecodes.EV_KEY:
                        continue
                    key_event = evdev.categorize(ev)
                    if not isinstance(key_event, evdev.KeyEvent):
                        continue
                    name = (
                        key_event.keycode
                        if isinstance(key_event.keycode, str)
                        else key_event.keycode[0]
                    )
                    if key_event.keystate == evdev.KeyEvent.key_down:
                        held.add(name)
                        if target.issubset(held):
                            try:
                                self.on_press()
                            except Exception:
                                logger.exception("hotkey on_press handler failed")
                    elif key_event.keystate == evdev.KeyEvent.key_up:
                        held.discard(name)
        except OSError:
            return
        finally:
            # Close the keyboard fd too on exit so a re-start does not
            # double-open the device.
            try:
                kbd.close()
            except OSError:
                logger.debug("evdev: close on kbd device failed", exc_info=True)
