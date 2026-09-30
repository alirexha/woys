"""MODEL / PROFILE jobs must end in state=error when nothing was applied.

JobRegistry reports state=error only when the job body raises. The job
bodies used to swallow a failed or timed-out swap (and an unknown
profile), so `JOB <id>` said done and `woys models use` / `woys profile
use` printed success and exited 0.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest


def _app() -> Any:
    from tui.app import WoysApp
    from tui.config import AppConfig

    app = WoysApp(cfg=AppConfig(), no_pw_setup=True)
    app.call_from_thread = lambda fn, *a, **k: fn(*a, **k)  # type: ignore[method-assign]
    app.notify = lambda *_a, **_k: None  # type: ignore[method-assign]
    return app


def _final_job_line(app: Any, reply: str) -> str:
    assert reply.startswith("OK job="), reply
    jid = reply.split("job=", 1)[1].split()[0]
    deadline = time.monotonic() + 5.0
    while True:
        line: str = app._jobs.status_line(jid)
        if "state=done" in line or "state=error" in line:
            return line
        assert time.monotonic() < deadline, line
        time.sleep(0.02)


@pytest.fixture
def voice(tmp_path: Path) -> Path:
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"not a real model")
    return model


def test_model_job_errors_when_the_swap_fails(voice: Path) -> None:
    app = _app()
    app.engine.stop(timeout=0.1)  # a stopped engine rejects swaps at once

    line = _final_job_line(app, app._handle_control(f"MODEL {voice}"))

    assert "state=error" in line
    assert "engine stopped" in line
    assert app._swap_in_flight is None


def test_model_job_errors_when_the_swap_times_out(
    voice: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from audio.engine import _SwapRequest

    app = _app()
    monkeypatch.setattr("tui.app._SWAP_WAIT_S", 0.1)
    app.engine.request_model_swap = lambda p: _SwapRequest(target=p)  # type: ignore[method-assign]

    line = _final_job_line(app, app._handle_control(f"MODEL {voice}"))

    assert "state=error" in line
    assert "not applied" in line
    assert "not applied" in (app.engine.stats.last_error or "")


def test_profile_job_errors_for_an_unknown_profile() -> None:
    app = _app()

    line = _final_job_line(app, app._handle_control("PROFILE nosuch"))

    assert "state=error" in line
    assert "nosuch" in line


def test_profile_job_errors_when_the_swap_fails(tmp_path: Path) -> None:
    model = tmp_path / "other.onnx"
    model.write_bytes(b"x")
    app = _app()
    app.cfg._extras["profiles"] = {"p": {"f0_up_key": 0, "rvc_model": str(model)}}
    app.engine.stop(timeout=0.1)

    line = _final_job_line(app, app._handle_control("PROFILE p"))

    assert "state=error" in line
