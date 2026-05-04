"""Opt-in evdev global hotkey.

Per Q7 / project brief: evdev raw-grab tripping VAC heuristics is a real risk
for CS2 users. The default control path is the Unix socket plus the WM-level
shortcut wrapper — this module is only loaded when the user explicitly opts
in via `enable_evdev_hotkey = true` in config.toml.

Setup
-----
- Install the optional dep: `uv pip install -e .[evdev]`
- Add the user to the `input` group:  `sudo usermod -aG input $USER && reboot`
- (Or: ship a `udev` rule under `pkg/udev/` granting r/w on `/dev/input/event*`
  to the `vcclient-cachy` group; see `docs/TROUBLESHOOTING.md`.)
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

logger = logging.getLogger("vcclient_cachy.hotkey")

DEFAULT_COMBO = ("KEY_LEFTCTRL", "KEY_LEFTALT", "KEY_V")


class EvdevHotkey:
    """Background thread that fires `on_press` whenever the combo lights up.

    Imports `evdev` lazily so the rest of the app remains usable without it.
    """

    def __init__(
        self, on_press: Callable[[], None], combo: tuple[str, ...] = DEFAULT_COMBO
    ) -> None:
        self.on_press = on_press
        self.combo = combo
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        try:
            import evdev  # noqa: F401  (just probe the import)
        except ImportError as e:
            logger.warning(
                "evdev not installed; skipping global hotkey. "
                "Install with: uv pip install -e '.[evdev]' (%s)",
                e,
            )
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="vcclient-hotkey", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.5)

    def _loop(self) -> None:
        import evdev

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
        if kbd is None:
            logger.warning("evdev: no keyboard device found")
            return

        held: set[str] = set()
        target = set(self.combo)
        try:
            for ev in kbd.read_loop():
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
