"""v0.11.0 — unit tests for the GPU anti-jitter features.

Tests cover:
  * `_resolve_anti_jitter_flags`: mode knob → (clock_lock, torch_keepalive)
  * `_query_max_graphics_clock_mhz`: nvidia-smi parsing happy path + failures
  * `_resolve_clock_lock_range`: auto-detect, explicit, sanity-check failures,
    over-stock-spec refusal
  * `_run_nvidia_smi`: success / nonzero exit / "error" in output / missing binary
  * `_apply_gpu_clock_lock` / `_revert_gpu_clock_lock`: subprocess mocking,
    state transitions, idempotent revert
  * Torch keepalive falls back gracefully when torch / CUDA unavailable

The engine isn't started in any of these tests — the methods are exercised
on a constructed-but-not-started `RealtimeEngine` instance with
config.gpu_clock_lock_enabled toggled per test.

Original work — Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "src" / "server") not in sys.path:
    sys.path.insert(0, str(REPO / "src" / "server"))


# ---- Fixtures: build engines without loading ORT sessions -------------------


def _mk_engine(**cfg_kwargs: object):
    """Construct a `RealtimeEngine` with a barely-valid config. Heavy
    session loads (`_ensure_sessions`) are NOT called — we only test the
    pure logic methods."""
    from audio.engine import EngineConfig, RealtimeEngine

    cfg = EngineConfig(**cfg_kwargs)  # type: ignore[arg-type]
    return RealtimeEngine(cfg)


# ---- _resolve_anti_jitter_flags ---------------------------------------------


def test_resolve_anti_jitter_mode_off_uses_underlying_booleans() -> None:
    eng = _mk_engine(gpu_anti_jitter_mode="off")
    assert eng._resolve_anti_jitter_flags() == (False, False)

    eng = _mk_engine(
        gpu_anti_jitter_mode="off",
        gpu_clock_lock_enabled=True,
        gpu_keepalive_torch_stream=True,
    )
    assert eng._resolve_anti_jitter_flags() == (True, True)


def test_resolve_anti_jitter_mode_keepalive() -> None:
    eng = _mk_engine(gpu_anti_jitter_mode="keepalive")
    assert eng._resolve_anti_jitter_flags() == (False, True)


def test_resolve_anti_jitter_mode_clock_lock() -> None:
    eng = _mk_engine(gpu_anti_jitter_mode="clock_lock")
    assert eng._resolve_anti_jitter_flags() == (True, False)


def test_resolve_anti_jitter_mode_both() -> None:
    eng = _mk_engine(gpu_anti_jitter_mode="both")
    assert eng._resolve_anti_jitter_flags() == (True, True)


def test_resolve_anti_jitter_mode_unknown_falls_back_to_off() -> None:
    eng = _mk_engine(gpu_anti_jitter_mode="garbage")
    assert eng._resolve_anti_jitter_flags() == (False, False)
    assert "unknown gpu_anti_jitter_mode" in (eng.stats.last_error or "")


def test_resolve_anti_jitter_mode_case_and_whitespace_insensitive() -> None:
    eng = _mk_engine(gpu_anti_jitter_mode="  BOTH  ")
    assert eng._resolve_anti_jitter_flags() == (True, True)


# ---- _query_max_graphics_clock_mhz ------------------------------------------


def test_query_max_graphics_clock_happy_path() -> None:
    from audio.engine import RealtimeEngine

    completed = subprocess.CompletedProcess(
        args=["nvidia-smi"], returncode=0, stdout="2100\n", stderr=""
    )
    with patch("audio.engine.subprocess.run", return_value=completed):
        assert RealtimeEngine._query_max_graphics_clock_mhz() == 2100


def test_query_max_graphics_clock_handles_unit_suffix() -> None:
    from audio.engine import RealtimeEngine

    # nvidia-smi sometimes appends a unit (depends on flags); the parser
    # must tolerate either.
    completed = subprocess.CompletedProcess(
        args=["nvidia-smi"], returncode=0, stdout="1845.0\n", stderr=""
    )
    with patch("audio.engine.subprocess.run", return_value=completed):
        assert RealtimeEngine._query_max_graphics_clock_mhz() == 1845


def test_query_max_graphics_clock_zero_on_nvidia_smi_missing() -> None:
    from audio.engine import RealtimeEngine

    with patch("audio.engine.subprocess.run", side_effect=FileNotFoundError("nvidia-smi")):
        assert RealtimeEngine._query_max_graphics_clock_mhz() == 0


def test_query_max_graphics_clock_zero_on_garbage_output() -> None:
    from audio.engine import RealtimeEngine

    completed = subprocess.CompletedProcess(
        args=["nvidia-smi"], returncode=0, stdout="not a number\n", stderr=""
    )
    with patch("audio.engine.subprocess.run", return_value=completed):
        assert RealtimeEngine._query_max_graphics_clock_mhz() == 0


def test_query_max_graphics_clock_zero_on_out_of_range() -> None:
    from audio.engine import RealtimeEngine

    # 100 MHz is suspiciously low; treat as malformed.
    completed = subprocess.CompletedProcess(
        args=["nvidia-smi"], returncode=0, stdout="100\n", stderr=""
    )
    with patch("audio.engine.subprocess.run", return_value=completed):
        assert RealtimeEngine._query_max_graphics_clock_mhz() == 0


# ---- _resolve_clock_lock_range ----------------------------------------------


def test_resolve_clock_lock_range_auto_detects_floor() -> None:
    eng = _mk_engine(
        gpu_clock_lock_floor_mhz=0,
        gpu_clock_lock_ceiling_mhz=0,
        gpu_clock_lock_floor_offset_mhz=255,
    )
    with patch.object(type(eng), "_query_max_graphics_clock_mhz", return_value=2100):
        floor, ceiling = eng._resolve_clock_lock_range()
    assert floor == 1845
    assert ceiling == 2100


def test_resolve_clock_lock_range_explicit_overrides_auto() -> None:
    eng = _mk_engine(
        gpu_clock_lock_floor_mhz=1665,
        gpu_clock_lock_ceiling_mhz=1845,
    )
    with patch.object(type(eng), "_query_max_graphics_clock_mhz", return_value=2100):
        floor, ceiling = eng._resolve_clock_lock_range()
    assert floor == 1665
    assert ceiling == 1845


def test_resolve_clock_lock_range_refuses_over_stock_ceiling() -> None:
    eng = _mk_engine(
        gpu_clock_lock_floor_mhz=1845,
        gpu_clock_lock_ceiling_mhz=2400,  # over stock max
    )
    with patch.object(type(eng), "_query_max_graphics_clock_mhz", return_value=2100):
        with pytest.raises(RuntimeError, match="exceeds.*max.graphics"):
            eng._resolve_clock_lock_range()


def test_resolve_clock_lock_range_rejects_floor_above_ceiling() -> None:
    eng = _mk_engine(
        gpu_clock_lock_floor_mhz=2000,
        gpu_clock_lock_ceiling_mhz=1500,  # < floor
    )
    with patch.object(type(eng), "_query_max_graphics_clock_mhz", return_value=2100):
        with pytest.raises(RuntimeError, match="floor>ceiling"):
            eng._resolve_clock_lock_range()


def test_resolve_clock_lock_range_fails_when_auto_detect_unavailable() -> None:
    eng = _mk_engine(
        gpu_clock_lock_floor_mhz=0,
        gpu_clock_lock_ceiling_mhz=0,
    )
    with patch.object(type(eng), "_query_max_graphics_clock_mhz", return_value=0):
        with pytest.raises(RuntimeError, match="auto-detect.*failed"):
            eng._resolve_clock_lock_range()


# ---- _run_nvidia_smi --------------------------------------------------------


def test_run_nvidia_smi_returns_false_when_binary_missing() -> None:
    eng = _mk_engine()
    with patch("audio.engine.shutil.which", return_value=None):
        ok, msg = eng._run_nvidia_smi(["-lgc", "1845,1845"])
    assert ok is False
    assert "not on PATH" in msg


def test_run_nvidia_smi_happy_path() -> None:
    eng = _mk_engine()
    completed = subprocess.CompletedProcess(
        args=["sudo", "nvidia-smi"],
        returncode=0,
        stdout="GPU clocks set to ...\nAll done.\n",
        stderr="",
    )
    with (
        patch("audio.engine.shutil.which", return_value="/usr/bin/nvidia-smi"),
        patch("audio.engine.subprocess.run", return_value=completed),
    ):
        ok, msg = eng._run_nvidia_smi(["-lgc", "1845,1845"])
    assert ok is True
    assert "All done." in msg


def test_run_nvidia_smi_nonzero_exit_is_failure() -> None:
    eng = _mk_engine()
    completed = subprocess.CompletedProcess(
        args=["sudo", "nvidia-smi"], returncode=1, stdout="", stderr="permission denied"
    )
    with (
        patch("audio.engine.shutil.which", return_value="/usr/bin/nvidia-smi"),
        patch("audio.engine.subprocess.run", return_value=completed),
    ):
        ok, msg = eng._run_nvidia_smi(["-lgc", "1845,1845"])
    assert ok is False
    assert "exit=1" in msg
    assert "permission denied" in msg


def test_run_nvidia_smi_error_in_output_is_failure() -> None:
    eng = _mk_engine()
    # Some nvidia-smi paths return 0 but emit "Error: ..." for malformed
    # arguments. Treat string "error" in output as failure.
    completed = subprocess.CompletedProcess(
        args=["sudo", "nvidia-smi"],
        returncode=0,
        stdout="Error: invalid clock value\n",
        stderr="",
    )
    with (
        patch("audio.engine.shutil.which", return_value="/usr/bin/nvidia-smi"),
        patch("audio.engine.subprocess.run", return_value=completed),
    ):
        ok, msg = eng._run_nvidia_smi(["-lgc", "1845,1845"])
    assert ok is False
    assert "error" in msg.lower()


def test_run_nvidia_smi_timeout_is_failure() -> None:
    eng = _mk_engine()
    with (
        patch("audio.engine.shutil.which", return_value="/usr/bin/nvidia-smi"),
        patch(
            "audio.engine.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=4.0),
        ),
    ):
        ok, msg = eng._run_nvidia_smi(["-lgc", "1845,1845"])
    assert ok is False
    assert "TimeoutExpired" in msg


# ---- _apply_gpu_clock_lock / _revert_gpu_clock_lock -------------------------


def test_apply_gpu_clock_lock_happy_path() -> None:
    eng = _mk_engine(
        gpu_clock_lock_enabled=True,
        gpu_clock_lock_floor_mhz=1845,
        gpu_clock_lock_ceiling_mhz=1845,
    )
    with patch.object(type(eng), "_run_nvidia_smi", return_value=(True, "All done.")):
        eng._apply_gpu_clock_lock()
    assert eng.stats.gpu_clock_lock_active is True
    assert eng.stats.gpu_clock_lock_floor_mhz == 1845
    assert eng.stats.gpu_clock_lock_ceiling_mhz == 1845
    assert "All done" in eng.stats.gpu_clock_lock_last_message


def test_apply_gpu_clock_lock_hard_fails_on_subprocess_error() -> None:
    eng = _mk_engine(
        gpu_clock_lock_enabled=True,
        gpu_clock_lock_floor_mhz=1845,
        gpu_clock_lock_ceiling_mhz=1845,
    )
    with patch.object(
        type(eng),
        "_run_nvidia_smi",
        return_value=(False, "exit=1: sudo: a password is required"),
    ):
        with pytest.raises(RuntimeError, match="nvidia-smi -lgc.*failed"):
            eng._apply_gpu_clock_lock()
    assert eng.stats.gpu_clock_lock_active is False


def test_revert_gpu_clock_lock_idempotent_when_inactive() -> None:
    eng = _mk_engine()
    # Not active, so revert should be a no-op (no nvidia-smi call).
    with patch.object(type(eng), "_run_nvidia_smi") as run:
        eng._revert_gpu_clock_lock()
    run.assert_not_called()


def test_revert_gpu_clock_lock_calls_rgc_when_active() -> None:
    eng = _mk_engine(
        gpu_clock_lock_enabled=True,
        gpu_clock_lock_floor_mhz=1845,
        gpu_clock_lock_ceiling_mhz=1845,
    )
    with patch.object(type(eng), "_run_nvidia_smi", return_value=(True, "All done.")):
        eng._apply_gpu_clock_lock()
    with patch.object(type(eng), "_run_nvidia_smi", return_value=(True, "All done.")) as run:
        eng._revert_gpu_clock_lock()
    run.assert_called_once_with(["-rgc"], timeout=4.0)
    assert eng.stats.gpu_clock_lock_active is False


def test_revert_gpu_clock_lock_logs_failure_but_marks_inactive() -> None:
    """If -rgc fails for any reason, we still flip active=False so a
    subsequent engine restart doesn't try to "re-revert". The error is
    surfaced via last_error so the user can run -rgc manually."""
    eng = _mk_engine(
        gpu_clock_lock_enabled=True,
        gpu_clock_lock_floor_mhz=1845,
        gpu_clock_lock_ceiling_mhz=1845,
    )
    with patch.object(type(eng), "_run_nvidia_smi", return_value=(True, "All done.")):
        eng._apply_gpu_clock_lock()
    with patch.object(type(eng), "_run_nvidia_smi", return_value=(False, "exit=1: oh no")):
        eng._revert_gpu_clock_lock()
    assert eng.stats.gpu_clock_lock_active is False
    assert "nvidia-smi -rgc failed" in (eng.stats.last_error or "")


# ---- Torch keepalive fallback paths -----------------------------------------


def test_torch_keepalive_loop_logs_when_torch_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """When `import torch` fails, the keepalive thread exits cleanly with
    a `last_error` set. The engine continues running without keepalive."""
    eng = _mk_engine(gpu_anti_jitter_mode="keepalive")
    # Make `import torch` raise ImportError when the loop tries to import.
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__  # type: ignore[attr-defined]

    def _fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "torch":
            raise ImportError("torch is not installed")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_fake_import):
        eng._torch_keepalive_loop()  # returns cleanly
    assert "torch import failed" in (eng.stats.last_error or "")


def test_torch_keepalive_loop_logs_when_cuda_unavailable() -> None:
    eng = _mk_engine(gpu_anti_jitter_mode="keepalive")
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = False
    with patch.dict(sys.modules, {"torch": fake_torch}):
        eng._torch_keepalive_loop()
    assert "cuda.is_available" in (eng.stats.last_error or "")


def test_torch_keepalive_loop_disables_on_stream_alloc_failure() -> None:
    """If torch.cuda.Stream() raises (e.g. driver hiccup), the keepalive
    thread retires cleanly without bringing down the engine."""
    eng = _mk_engine(gpu_anti_jitter_mode="keepalive")
    fake_torch = MagicMock()
    fake_torch.cuda.is_available.return_value = True
    fake_torch.cuda.Stream.side_effect = RuntimeError("CUDA out of memory")
    fake_torch.empty.return_value = MagicMock()
    with patch.dict(sys.modules, {"torch": fake_torch}):
        eng._torch_keepalive_loop()
    assert "torch keepalive setup failed" in (eng.stats.last_error or "")
