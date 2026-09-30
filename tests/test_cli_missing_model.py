"""A configured rvc_model that does not exist must fail loudly.

`woys diag` and `woys engine` replaced a missing configured model with
the default voice without a word (`rvc_path = ... if exists() else
None`), so a stale config.toml or an unmounted drive silently turned the
user's voice into the stock one. An empty rvc_model still means "use the
default".

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import contextlib
import signal
from collections.abc import Iterator
from pathlib import Path

import pytest

import audio.engine as eng_mod
import audio.pipewire as pw
import tui.config as tcfg
import woys.cli as cli
import woys.instance_lock as instance_lock


class _NoEngine:
    def __init__(self, _cfg: object) -> None:
        raise AssertionError("the engine must not be built with a substituted model")


@pytest.fixture
def stubs(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    saved = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    monkeypatch.setattr(cli, "cmd_info", lambda: 0)
    monkeypatch.setattr(pw.VirtualMic, "ensure", lambda self: None)
    monkeypatch.setattr(pw, "get_state", lambda: pw.VirtualMicState(True, True, 1, 2))
    monkeypatch.setattr(instance_lock, "acquire_instance_lock", contextlib.nullcontext)
    monkeypatch.setattr(eng_mod, "RealtimeEngine", _NoEngine)
    yield None
    for sig, handler in saved.items():
        signal.signal(sig, handler)


def _configure(monkeypatch: pytest.MonkeyPatch, rvc_model: str) -> None:
    monkeypatch.setattr(tcfg, "load_config", lambda *a, **k: tcfg.AppConfig(rvc_model=rvc_model))


def test_diag_refuses_a_missing_configured_model(
    stubs: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _configure(monkeypatch, str(tmp_path / "my_voice.onnx"))
    with pytest.raises(FileNotFoundError, match=r"my_voice\.onnx"):
        cli.cmd_diag(0.0, no_engine=False)


def test_engine_refuses_a_missing_configured_model(
    stubs: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _configure(monkeypatch, str(tmp_path / "my_voice.onnx"))
    with pytest.raises(FileNotFoundError, match=r"my_voice\.onnx"):
        cli._cmd_engine_locked(0.0, quiet=True)


def test_missing_model_exits_1_with_the_path(
    stubs: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _configure(monkeypatch, str(tmp_path / "my_voice.onnx"))
    monkeypatch.setattr("woys.logsetup.setup_logging", lambda: tmp_path / "woys.log")
    assert cli.main(["diag", "--seconds", "0"]) == 1
    assert "my_voice.onnx" in capsys.readouterr().err


def test_empty_rvc_model_still_means_the_default_voice(tmp_path: Path) -> None:
    assert cli._configured_rvc_model(tcfg.AppConfig(rvc_model="")) is None
    present = tmp_path / "voice.onnx"
    present.write_bytes(b"x")
    assert cli._configured_rvc_model(tcfg.AppConfig(rvc_model=str(present))) == present
