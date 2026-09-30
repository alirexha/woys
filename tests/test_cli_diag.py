"""`woys diag` reporting and exit status.

`cmd_diag` is driven end to end with a stand-in engine (no GPU, no
PipeWire, no models): the fake's `start()` sets whatever stats the case
needs, and `stop()` clears what the real stop() clears.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from typing import Any

import pytest

import audio.engine as eng_mod
import audio.pipewire as pw
import woys.cli as cli
import woys.instance_lock as instance_lock


class _FakeEngine:
    on_start: Callable[[_FakeEngine], None] = staticmethod(lambda _e: None)

    def __init__(self, cfg: eng_mod.EngineConfig) -> None:
        self.cfg = cfg
        self.player_backend = "pw-cat"
        self.stats = eng_mod.EngineStats()
        self._errors: list[tuple[float, str, str]] = []

    def start(self) -> None:
        self.stats.running = True
        type(self).on_start(self)

    def stop(self, timeout: float = 2.0) -> None:
        self.stats.running = False
        self.stats.child_pid = None
        self.stats.warmup_stage = ""

    def record(self, msg: str) -> None:
        self._errors.append((0.0, "woys-engine", msg))
        self.stats.last_error = msg

    def recent_errors(self, n: int = 5) -> list[tuple[float, str, str]]:
        return self._errors[-n:]


def _healthy(e: _FakeEngine) -> None:
    e.stats.chunks_processed = 40
    e.stats.warmup_stage = "ready"


@pytest.fixture
def diag(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[..., int]]:
    monkeypatch.setattr(cli, "cmd_info", lambda: 0)
    monkeypatch.setattr(pw.VirtualMic, "ensure", lambda self: None)
    monkeypatch.setattr(pw, "get_state", lambda: pw.VirtualMicState(True, True, 1, 2))
    monkeypatch.setattr(instance_lock, "acquire_instance_lock", contextlib.nullcontext)
    monkeypatch.setattr(eng_mod, "RealtimeEngine", _FakeEngine)
    monkeypatch.setattr(_FakeEngine, "on_start", staticmethod(_healthy))

    def run(on_start: Callable[[_FakeEngine], None] | None = None, **_: Any) -> int:
        if on_start is not None:
            monkeypatch.setattr(_FakeEngine, "on_start", staticmethod(on_start))
        return cli.cmd_diag(0.0, no_engine=False)

    yield run


def test_diag_reports_cpu_bound_sessions(
    diag: Callable[..., int], capsys: pytest.CaptureFixture[str]
) -> None:
    def cpu_bound(e: _FakeEngine) -> None:
        _healthy(e)
        e.stats.cpu_fallback_active = True

    diag(cpu_bound)
    out = capsys.readouterr().out
    assert "cpu fallback" in out
    assert "CPU-only" in out


def test_diag_exits_zero_on_a_healthy_run(diag: Callable[..., int]) -> None:
    assert diag() == 0


def test_diag_exits_nonzero_when_engine_crashed(diag: Callable[..., int]) -> None:
    def warmup_failed(e: _FakeEngine) -> None:
        e.stats.crashed = True
        e.stats.running = False
        e.record("engine warmup: FileNotFoundError: contentvec model not found")

    assert diag(warmup_failed) == 1


def test_diag_exits_nonzero_when_no_chunk_was_processed(
    diag: Callable[..., int], capsys: pytest.CaptureFixture[str]
) -> None:
    def still_warming(e: _FakeEngine) -> None:
        e.stats.warmup_stage = "warming pipeline"

    assert diag(still_warming) == 1
    assert "--seconds" in capsys.readouterr().out


def test_diag_exits_nonzero_on_dropped_chunks(diag: Callable[..., int]) -> None:
    def drops(e: _FakeEngine) -> None:
        _healthy(e)
        e.stats.dropped_chunks = 3

    assert diag(drops) == 1


def test_diag_reads_the_child_pid_before_stop_clears_it(
    diag: Callable[..., int],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import tui.config as tcfg

    real = tcfg.app_config_to_engine_config

    def subprocess_mode(*a: Any, **k: Any) -> eng_mod.EngineConfig:
        cfg = real(*a, **k)
        cfg.inference_subprocess = True
        return cfg

    monkeypatch.setattr(tcfg, "app_config_to_engine_config", subprocess_mode)

    def child_up(e: _FakeEngine) -> None:
        _healthy(e)
        e.stats.child_pid = 4242

    assert diag(child_up) == 0
    assert "SUBPROCESS (child pid=4242)" in capsys.readouterr().out


def test_diag_exits_nonzero_when_the_child_never_came_up(
    diag: Callable[..., int], monkeypatch: pytest.MonkeyPatch
) -> None:
    import tui.config as tcfg

    real = tcfg.app_config_to_engine_config

    def subprocess_mode(*a: Any, **k: Any) -> eng_mod.EngineConfig:
        cfg = real(*a, **k)
        cfg.inference_subprocess = True
        return cfg

    monkeypatch.setattr(tcfg, "app_config_to_engine_config", subprocess_mode)
    assert diag() == 1


def test_diag_no_engine_still_reports_sink_state(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """--no-engine promises "static info (CUDA, PipeWire, sink state)" but
    returned before the PipeWire block. It must report the state without
    loading anything (no ensure())."""
    monkeypatch.setattr(cli, "cmd_info", lambda: 0)

    def no_side_effects(_self: object) -> None:
        raise AssertionError("--no-engine must not load the virtual devices")

    monkeypatch.setattr(pw.VirtualMic, "ensure", no_side_effects)
    monkeypatch.setattr(pw, "get_state", lambda: pw.VirtualMicState(True, False, 1, None))
    assert cli.cmd_diag(0.0, no_engine=True) == 0
    assert "sink=True source=False" in capsys.readouterr().out
