"""`woys diag` / `woys engine` must print the error history, not just
`last_error`.

`stats.last_error` clears itself one chunk after it is written, so a run
that hit dropped chunks, a playback-helper respawn or a config fallback
and then recovered ended with an empty `last_error` and no trace of what
went wrong. The bounded ring behind `recent_errors()` keeps them.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import contextlib
import signal
from collections.abc import Iterator

import pytest

import audio.engine as eng_mod
import audio.pipewire as pw
import woys.cli as cli
import woys.instance_lock as instance_lock


class _RecoveredEngine:
    """Hit two errors mid-run, then recovered (last_error cleared)."""

    def __init__(self, cfg: eng_mod.EngineConfig) -> None:
        self.cfg = cfg
        self.player_backend = "pw-cat"
        self.stats = eng_mod.EngineStats()
        self._rvc_output_sr = 40_000
        self.active_embedder = "onnx"
        self._inf_client = None

    def start(self) -> None:
        self.stats.running = True
        self.stats.chunks_processed = 40
        self.stats.warmup_stage = "ready"
        self.stats.last_error = None

    def stop(self, timeout: float = 2.0) -> None:
        self.stats.running = False

    def recent_errors(self, n: int = 5) -> list[tuple[float, str, str]]:
        ring = [
            (10.0, "woys-engine", "unknown embedder 'hubert'. Falling back to onnx"),
            (12.5, "woys-pacat-writer", "pw-cat write failed (BrokenPipeError); respawning"),
        ]
        return ring[-n:]


@pytest.fixture
def stubs(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    monkeypatch.setattr(cli, "cmd_info", lambda: 0)
    monkeypatch.setattr(pw.VirtualMic, "ensure", lambda self: None)
    monkeypatch.setattr(pw, "get_state", lambda: pw.VirtualMicState(True, True, 1, 2))
    monkeypatch.setattr(instance_lock, "acquire_instance_lock", contextlib.nullcontext)
    monkeypatch.setattr(eng_mod, "RealtimeEngine", _RecoveredEngine)
    yield None
    for sig, handler in saved.items():
        signal.signal(sig, handler)


def test_diag_prints_recent_errors(stubs: None, capsys: pytest.CaptureFixture[str]) -> None:
    cli.cmd_diag(0.0, no_engine=False)
    out = capsys.readouterr().out
    assert "Falling back to onnx" in out
    assert "respawning" in out


def test_engine_prints_recent_errors(stubs: None, capsys: pytest.CaptureFixture[str]) -> None:
    cli._cmd_engine_locked(0.001, quiet=True)
    out = capsys.readouterr().out
    assert "Falling back to onnx" in out
    assert "respawning" in out
