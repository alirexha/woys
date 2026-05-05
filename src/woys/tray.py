"""Optional system-tray icon (KDE Plasma 6 / GNOME via libappindicator).

Talks to a *running* TUI through its Unix-domain control socket — same
protocol as `woys toggle`. The tray therefore *requires* a TUI
to already be running; if not, it offers to launch one.

Install path
------------
The tray uses `pystray` + `Pillow`, which aren't part of the default
runtime install. Pull them via the optional extra:

    uv pip install -e ".[tray]"

Then:

    woys tray   # spawns the icon

Stays in the background until the user clicks "Quit" in the menu.

Original work — Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import sys
import threading
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


_ICON_SIZE = 64


def _make_icon_image(running: bool) -> Any:
    """Tiny circular icon — green when the engine is running, dim otherwise."""
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (_ICON_SIZE, _ICON_SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    fill = (60, 200, 90, 255) if running else (120, 120, 120, 200)
    edge = (255, 255, 255, 220) if running else (180, 180, 180, 200)
    pad = 4
    draw.ellipse((pad, pad, _ICON_SIZE - pad, _ICON_SIZE - pad), fill=fill, outline=edge, width=3)
    # Mic-shape glyph in the middle (simplified).
    cx = _ICON_SIZE // 2
    draw.rectangle((cx - 6, cx - 14, cx + 6, cx + 8), fill=(255, 255, 255, 230))
    draw.line((cx, cx + 8, cx, cx + 18), fill=(255, 255, 255, 230), width=3)
    return img


def _engine_status() -> tuple[bool, str]:
    """Ping the TUI control socket. Returns (running, raw_reply)."""
    try:
        repo_root = sys.path[0]
        sys.path.insert(0, str(repo_root))
        from tui.control import send_command

        reply = send_command("STATUS", timeout=0.5)
    except Exception as e:
        return False, f"ERR {type(e).__name__}: {e}"
    return ("running=True" in reply), reply


def _on_toggle(_icon: Any, _item: Any) -> None:
    from tui.control import send_command

    send_command("TOGGLE")


def _on_status(_icon: Any, _item: Any) -> None:
    print(_engine_status()[1])


def _on_quit(icon: Any, _item: Any) -> None:
    icon.stop()


def run_tray() -> int:
    """Block and run the tray loop. Returns when the user picks Quit."""
    try:
        import pystray
    except ImportError as e:  # pragma: no cover — optional extra
        print(
            f"[tray] pystray not installed ({e}). Install with: uv pip install -e '.[tray]'",
            file=sys.stderr,
        )
        return 2

    # Initial icon state from the engine.
    running, _ = _engine_status()
    icon = pystray.Icon(
        "woys",
        icon=_make_icon_image(running),
        title="woys",
        menu=pystray.Menu(
            pystray.MenuItem("Toggle engine", _on_toggle, default=True),
            pystray.MenuItem("Print status", _on_status),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit tray", _on_quit),
        ),
    )

    stop_event = threading.Event()

    def refresh_loop() -> None:
        last_running = running
        while not stop_event.is_set():
            time.sleep(1.0)
            new_running, _ = _engine_status()
            if new_running != last_running:
                icon.icon = _make_icon_image(new_running)
                last_running = new_running

    refresher = threading.Thread(target=refresh_loop, name="tray-refresh", daemon=True)
    refresher.start()

    try:
        icon.run()
    finally:
        stop_event.set()
        refresher.join(timeout=2.0)
    return 0


def cli_tray() -> int:
    return run_tray()
