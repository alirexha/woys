"""Socket PITCH commands: no lost updates, and the value stays in range.

The control server runs up to four handlers at once (a WM shortcut on
key repeat sends a burst of `woys pitch +1`). Each handler must read
and write the pitch on the event loop, or two of them read the same
base and one step is lost.
"""

from __future__ import annotations

import threading
import time
from typing import Any


def _app() -> Any:
    from tui.app import WoysApp
    from tui.config import AppConfig

    app = WoysApp(cfg=AppConfig(), no_pw_setup=True)
    app.notify = lambda *_a, **_k: None  # type: ignore[method-assign]
    loop_lock = threading.Lock()

    def call_from_thread(fn: Any, *a: Any, **k: Any) -> Any:
        # One "event loop": callbacks run one at a time, a little later
        # than they were posted.
        time.sleep(0.01)
        with loop_lock:
            return fn(*a, **k)

    app.call_from_thread = call_from_thread  # type: ignore[method-assign]
    return app


def test_concurrent_pitch_steps_are_not_lost() -> None:
    app = _app()
    replies: list[str] = []

    def send() -> None:
        replies.append(app._handle_control("PITCH +1"))

    threads = [threading.Thread(target=send) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert int(app.pitch) == 8
    assert app.engine.cfg.f0_up_key == 8
    assert app.cfg.f0_up_key == 8
    assert sorted(replies) == sorted(f"OK pitch={n}" for n in range(1, 9))
