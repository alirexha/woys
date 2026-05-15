"""review F-07-17 (commit-049): monitor writes decoupled from
the engine main thread via a bounded queue + dedicated writer thread.

Pre-fix `_run_loop` opened `sd.OutputStream` lazily on `cfg.monitor`
going True and called `monitor_stream.write(out48.reshape(-1, 1))`
SYNCHRONOUSLY on the engine main thread. A slow host-default sink
(Bluetooth glitch, ALSA underrun on a busy system) blocked the
write -- the engine couldn't service the next mic read -- audio
drops.

Post-fix the chunk loop does `self._monitor_queue.put_nowait(out48)`
(non-blocking) and a dedicated `_monitor_writer_loop` thread owns
the `sd.OutputStream` lifecycle + drains the queue. Queue overflow
counts as a monitor-drop; the engine main thread is never blocked.

These tests mock out `sounddevice` so they run on systems without
real audio output (CI / headless).

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import contextlib
import queue
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "src" / "server") not in sys.path:
    sys.path.insert(0, str(REPO / "src" / "server"))


class _FakeOutputStream:
    def __init__(self, samplerate: int, channels: int, dtype: str) -> None:
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self.started = False
        self.closed = False
        self.writes: list[np.ndarray] = []
        self._slow_seconds = 0.0

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def close(self) -> None:
        self.closed = True

    def write(self, chunk: np.ndarray) -> None:
        if self._slow_seconds > 0:
            time.sleep(self._slow_seconds)
        self.writes.append(chunk)


def _install_fake_sounddevice(monkeypatch: pytest.MonkeyPatch) -> list[_FakeOutputStream]:
    """Install a fake `sounddevice` so the writer thread doesn't try
    to open real audio. Returns the list that captures every
    OutputStream instance created."""
    created: list[_FakeOutputStream] = []
    fake_sd = types.SimpleNamespace()

    def factory(*, samplerate: int, channels: int, dtype: str) -> _FakeOutputStream:
        stream = _FakeOutputStream(samplerate, channels, dtype)
        created.append(stream)
        return stream

    fake_sd.OutputStream = factory
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    return created


def test_monitor_queue_and_writer_thread_attributes_exist() -> None:
    """Pin the new infrastructure surface."""
    from audio import engine

    eng = engine.RealtimeEngine(engine.EngineConfig())
    assert isinstance(eng._monitor_queue, queue.Queue)
    assert eng._monitor_queue.maxsize == 8
    assert eng._monitor_thread is None  # spawned only in _worker_preamble
    assert hasattr(eng.stats, "monitor_drops")
    assert eng.stats.monitor_drops == 0


def test_writer_thread_drains_queue_and_writes_to_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end. cfg.monitor=True; push 5 chunks onto the queue;
    the writer thread opens an OutputStream + writes each chunk +
    closes the stream on stop()."""
    from audio import engine

    streams = _install_fake_sounddevice(monkeypatch)

    eng = engine.RealtimeEngine(engine.EngineConfig())
    eng.cfg.monitor = True

    # Spawn the writer thread directly (we're not running the full
    # engine; we just exercise the writer loop).
    t = threading.Thread(target=eng._monitor_writer_loop, name="test-monitor")
    t.start()

    try:
        for i in range(5):
            eng._monitor_queue.put(np.full(2400, float(i), dtype=np.float32))
        # Wait for writes to land.
        time.sleep(0.2)
        assert len(streams) == 1, "exactly one OutputStream should have been opened"
        s = streams[0]
        assert s.started is True
        assert len(s.writes) == 5
    finally:
        eng._stop_event.set()
        t.join(timeout=2.0)

    assert streams[0].closed is True, "stream must be closed on stop"


def test_engine_chunk_loop_never_blocks_on_slow_monitor_sink(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bug-class test. A slow monitor sink (each write sleeps 100 ms)
    must NOT block the engine main thread. We simulate the chunk
    loop's monitor-push by calling put_nowait 16 times back-to-back;
    each call returns in microseconds. Queue overflow lands on
    stats.monitor_drops.

    Pre-fix the chunk loop did `monitor_stream.write(out48)`
    synchronously -- 16 chunks x 100ms = 1.6s of engine stall."""
    from audio import engine

    streams = _install_fake_sounddevice(monkeypatch)

    eng = engine.RealtimeEngine(engine.EngineConfig())
    eng.cfg.monitor = True

    t = threading.Thread(target=eng._monitor_writer_loop, name="test-slow-monitor")
    t.start()

    try:
        # Wait for the writer to open the stream.
        for _ in range(20):
            if streams:
                streams[0]._slow_seconds = 0.1  # 100ms per write
                break
            time.sleep(0.01)
        assert streams, "writer thread should have opened a stream"

        # Simulate the chunk loop pushing 16 chunks in rapid
        # succession. The queue is 8 slots; the rest must drop.
        t0 = time.monotonic()
        for i in range(16):
            try:
                eng._monitor_queue.put_nowait(np.full(2400, float(i), dtype=np.float32))
            except queue.Full:
                with eng._stats_lock:
                    eng.stats.monitor_drops += 1
        push_elapsed = time.monotonic() - t0

        assert push_elapsed < 0.05, (
            f"F-07-17: the engine main thread's 16 put_nowait calls must "
            f"complete in microseconds total (not 1.6s like pre-fix's "
            f"synchronous writes). Got {push_elapsed:.3f}s"
        )
        # The queue is 8 slots; 16 pushes -> 8 land, 8 drop (some may
        # land while the writer is mid-drain; bound the drop count).
        assert eng.stats.monitor_drops >= 1, (
            "rapid pushes against a slow sink must produce some drops; "
            f"got {eng.stats.monitor_drops}"
        )
    finally:
        eng._stop_event.set()
        t.join(timeout=3.0)


def test_writer_thread_opens_stream_on_monitor_toggle_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`cfg.monitor` going False->True causes the writer to OPEN a
    stream the next time it wakes from its 50ms get-timeout."""
    from audio import engine

    streams = _install_fake_sounddevice(monkeypatch)

    eng = engine.RealtimeEngine(engine.EngineConfig())
    eng.cfg.monitor = False  # start off

    t = threading.Thread(target=eng._monitor_writer_loop, name="test-toggle-on")
    t.start()
    try:
        # Wait one tick; no stream yet.
        time.sleep(0.1)
        assert streams == [], "no stream opened while cfg.monitor=False"
        # Toggle on.
        eng.cfg.monitor = True
        # Push a chunk so the writer has something to drain (otherwise
        # it just spins on the get-timeout).
        eng._monitor_queue.put(np.zeros(2400, dtype=np.float32))
        # Wait for the writer to open + write.
        for _ in range(30):
            if streams and streams[0].writes:
                break
            time.sleep(0.05)
        assert streams and streams[0].started is True
    finally:
        eng._stop_event.set()
        t.join(timeout=2.0)


def test_writer_thread_closes_stream_on_monitor_toggle_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`cfg.monitor` going True->False causes the writer to CLOSE
    the stream the next time it wakes."""
    from audio import engine

    streams = _install_fake_sounddevice(monkeypatch)

    eng = engine.RealtimeEngine(engine.EngineConfig())
    eng.cfg.monitor = True

    t = threading.Thread(target=eng._monitor_writer_loop, name="test-toggle-off")
    t.start()
    try:
        # Wait for the writer to open.
        for _ in range(30):
            if streams:
                break
            time.sleep(0.02)
        assert streams
        # Toggle off.
        eng.cfg.monitor = False
        # Push something so the writer wakes promptly.
        with contextlib.suppress(queue.Full):
            for _ in range(8):
                eng._monitor_queue.put_nowait(np.zeros(2400, dtype=np.float32))
        # Wait for the writer to close.
        for _ in range(30):
            if streams[0].closed:
                break
            time.sleep(0.02)
        assert streams[0].closed is True
    finally:
        eng._stop_event.set()
        t.join(timeout=2.0)


def test_engine_chunk_loop_uses_queue_not_direct_write() -> None:
    """Structural pin: the engine's chunk-loop monitor block must
    push to `_monitor_queue`, not call `monitor_stream.write(...)`
    directly."""
    src = Path(__file__).resolve().parent.parent / "src" / "audio" / "engine.py"
    text = src.read_text()

    # Find the run loop's "if self.cfg.monitor:" block.
    idx = text.find("if self.cfg.monitor:")
    while idx > 0:
        # Skip the writer-loop's "want_monitor" branch and the comment lines.
        body = text[idx : idx + 400]
        if "put_nowait" in body and "_monitor_queue" in body:
            return  # found the post-fix put-to-queue site
        idx = text.find("if self.cfg.monitor:", idx + 1)
    pytest.fail(
        "engine's chunk loop must push to _monitor_queue instead of "
        "calling monitor_stream.write() directly (F-07-17 / commit-049)"
    )
