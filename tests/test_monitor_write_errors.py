"""Monitor write failures must be reported, not swallowed.

`_monitor_writer_loop` wrapped every `stream.write` in
`contextlib.suppress(Exception)`, although its docstring promised the
failures went through `record_error`. When the monitor device went away
(a Bluetooth headset disconnecting) every write failed silently and the
self-monitor just went quiet with nothing in the error log.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import sys
import threading
import time
import types
from typing import Any

import numpy as np
import pytest

from audio import engine


class _DeadDeviceStream:
    def __init__(self, **_kw: Any) -> None:
        self.writes = 0

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass

    def write(self, _chunk: object) -> None:
        self.writes += 1
        raise OSError("PortAudioError: Stream is stopped (device unplugged)")


def test_monitor_write_errors_are_recorded_and_rate_limited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_sd = types.SimpleNamespace(OutputStream=_DeadDeviceStream)
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    eng = engine.RealtimeEngine(engine.EngineConfig(monitor=True))
    for _ in range(8):
        eng._monitor_queue.put_nowait(np.zeros(480, dtype=np.float32))

    t = threading.Thread(target=eng._monitor_writer_loop, daemon=True)
    t.start()
    deadline = time.monotonic() + 3.0
    while not eng._monitor_queue.empty() and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.1)
    eng._stop_event.set()
    t.join(timeout=2.0)

    monitor_errors = [m for _ts, _th, m in eng.recent_errors(20) if "monitor write" in m]
    assert monitor_errors, "a failing monitor write must reach record_error"
    assert "device unplugged" in monitor_errors[0]
    # 8 failures, but only the first few are logged (same scheme as dropped
    # inference chunks) so a dead device cannot flood the error ring.
    assert len(monitor_errors) == 3
