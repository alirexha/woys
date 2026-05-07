"""v0.6.8 polish-release regression tests.

Three discrete behaviours pinned here:

1. **`AppConfig` defaults forward from `EngineConfig`.** Without this,
   shared fields drift over releases and fresh installs reproduce bugs we
   thought were fixed. v0.6.7 shipped with `AppConfig.output_latency_ms = 100`
   while `EngineConfig.output_latency_ms = 300` — fresh users hit the
   micro-cut bug v0.6.7 was meant to escape. See `LESSONS.md §17`.

2. **Malformed `config.toml` no longer crashes the app.** TOML decode errors
   and read errors fall back to in-memory defaults with a clear stderr
   message; the bad file is left in place for the user to inspect.

3. **Engine drops a single bad-inference chunk instead of dying.** A
   transient ORT / CUDA / numerical exception during inference bumps
   `stats.dropped_chunks`, updates `stats.last_error`, and lets the
   audio loop continue.
"""

from __future__ import annotations

import sys
from dataclasses import fields
from pathlib import Path

# Make src/ importable.
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from audio.engine import EngineConfig, EngineStats, RealtimeEngine  # noqa: E402
from tui.config import AppConfig, load_config  # noqa: E402

# ---- 1. AppConfig / EngineConfig drift ------------------------------------


def test_app_config_forwards_engine_config_defaults() -> None:
    """Every field that exists in BOTH dataclasses must have the same default
    value. Catches drift like the v0.6.7 `output_latency_ms` mismatch."""
    eng_defaults = {f.name: f.default for f in fields(EngineConfig)}
    app_defaults = {f.name: f.default for f in fields(AppConfig)}

    # `rvc_model` is intentionally a different type in each class:
    # AppConfig stores it as `str` ("" sentinel = use engine default);
    # EngineConfig stores it as a `Path`. Excluding from the check.
    shared = (set(eng_defaults) & set(app_defaults)) - {"rvc_model"}
    # Sanity — there ARE shared fields (otherwise the test would silently pass).
    assert {"chunk_seconds", "output_latency_ms", "sink_name"} <= shared

    mismatches = {
        name: (eng_defaults[name], app_defaults[name])
        for name in shared
        if eng_defaults[name] != app_defaults[name]
    }
    assert not mismatches, (
        f"AppConfig/EngineConfig defaults drifted: {mismatches}. "
        f"AppConfig must forward from EngineConfig (see tui/config.py:_E)."
    )


def test_app_config_output_latency_ms_is_engine_default() -> None:
    """Pin the current expected value explicitly so a future re-bump in
    EngineConfig is visible at this level too. v0.6.8 pinned 300; v0.7.0-rc1
    dropped to 80 (user-rejected); v0.7.0-rc2 bumped to 220 (also
    user-rejected in real Telegram VoIP); v0.7.0-rc3 bumped to 280 — the
    last rung before the structural floor. The test pins 280 so a future
    drift in either direction is caught."""
    assert AppConfig().output_latency_ms == 280


# ---- 2. TOML decode safety ------------------------------------------------


def test_load_config_handles_malformed_toml(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A user with a broken `config.toml` (typo, fat-fingered edit) used to
    crash the app with `TOMLDecodeError`. v0.6.8 — print a clear message,
    fall back to defaults, leave the bad file in place."""
    bad = tmp_path / "config.toml"
    bad.write_text("this is not = valid TOML [[\n")

    cfg = load_config(bad)

    # We got a working AppConfig back, not a crash.
    assert isinstance(cfg, AppConfig)
    # And the values are EngineConfig-forwarded defaults, not garbage.
    # v0.7.0-rc3 — output_latency 220 → 280 (rc2's 220 was user-rejected
    # in Telegram VoIP), chunk_seconds 0.25 → 0.15.
    assert cfg.output_latency_ms == 280
    assert cfg.chunk_seconds == 0.15

    err = capsys.readouterr().err
    assert "malformed TOML" in err
    assert str(bad) in err

    # The bad file is left in place — user can fix and re-launch.
    assert bad.exists()
    assert "this is not = valid TOML" in bad.read_text()


def test_load_config_handles_unreadable_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Permission-denied / EIO on read also falls back instead of crashing."""
    target = tmp_path / "config.toml"
    target.write_text("# placeholder so .exists() is True")

    real_open = open

    def fake_open(path: object, *args: object, **kwargs: object) -> object:
        if str(path) == str(target):
            raise PermissionError("simulated EACCES")
        return real_open(path, *args, **kwargs)  # type: ignore[arg-type,no-any-return]

    monkeypatch.setattr("builtins.open", fake_open)

    cfg = load_config(target)
    assert isinstance(cfg, AppConfig)
    err = capsys.readouterr().err
    assert "cannot read" in err
    assert "PermissionError" in err


# ---- 3. Engine resilience (chunk-skip) ------------------------------------


def _make_engine_with_stub_inference(*, raise_exc: Exception | None) -> RealtimeEngine:
    """Build a RealtimeEngine without touching ORT, with
    `_process_streaming_16k` either raising the given exception or
    passing the input through. We DON'T call `_ensure_sessions()` —
    `_safe_process_streaming_16k` is fully testable without it because
    the real `_process_streaming_16k` is the only thing it dispatches to,
    and we replace that here."""
    eng = RealtimeEngine(EngineConfig())
    if raise_exc is not None:

        def _raise(audio16: np.ndarray) -> np.ndarray:  # type: ignore[type-arg]
            raise raise_exc

        eng._process_streaming_16k = _raise  # type: ignore[method-assign]
    else:

        def _passthrough(audio16: np.ndarray) -> np.ndarray:  # type: ignore[type-arg]
            return audio16.astype(np.float32, copy=False)

        eng._process_streaming_16k = _passthrough  # type: ignore[method-assign]
    return eng


def test_safe_process_returns_none_and_bumps_counter_on_failure() -> None:
    eng = _make_engine_with_stub_inference(raise_exc=RuntimeError("simulated CUDA OOM"))
    audio = np.zeros(160, dtype=np.float32)

    assert eng.stats.dropped_chunks == 0
    out = eng._safe_process_streaming_16k(audio)
    assert out is None
    assert eng.stats.dropped_chunks == 1
    assert eng.stats.last_error is not None
    assert "RuntimeError" in eng.stats.last_error
    assert "simulated CUDA OOM" in eng.stats.last_error
    assert "#1" in eng.stats.last_error


def test_safe_process_logs_first_three_then_circuit_breaker_fires() -> None:
    """First three failures log in detail; #4-#49 increment silently;
    at #50 the B14 circuit breaker fires `_stop_event` so the engine
    exits cleanly rather than serving silence indefinitely."""
    eng = _make_engine_with_stub_inference(raise_exc=RuntimeError("boom"))
    audio = np.zeros(160, dtype=np.float32)

    # Failures 1-3: each updates last_error with the new count.
    for expected_n in (1, 2, 3):
        eng.stats.last_error = None
        eng._safe_process_streaming_16k(audio)
        assert eng.stats.dropped_chunks == expected_n
        assert eng.stats.last_error is not None
        assert f"#{expected_n}" in eng.stats.last_error

    # Failures 4-49 increment silently (last_error left from previous tick).
    eng.stats.last_error = "marker"
    for _ in range(46):  # bring dropped_chunks to 49
        eng._safe_process_streaming_16k(audio)
    assert eng.stats.dropped_chunks == 49
    assert eng.stats.last_error == "marker"
    assert not eng._stop_event.is_set()

    # Failure #50 trips the circuit breaker.
    eng._safe_process_streaming_16k(audio)
    assert eng.stats.dropped_chunks == 50
    assert eng._stop_event.is_set()
    assert eng.stats.last_error is not None
    assert "stopping" in eng.stats.last_error.lower()
    assert "50 consecutive" in eng.stats.last_error


def test_safe_process_consecutive_drops_resets_on_success() -> None:
    """The circuit breaker only fires on truly *consecutive* failures —
    a single successful chunk between failures resets the counter."""
    # Stub that fails for the first 3 calls, then succeeds.
    state = {"calls": 0}

    eng = _make_engine_with_stub_inference(raise_exc=None)
    real_proc = eng._process_streaming_16k

    def flaky(audio: np.ndarray) -> np.ndarray:
        state["calls"] += 1
        if state["calls"] <= 3:
            raise RuntimeError("transient")
        return real_proc(audio)

    eng._process_streaming_16k = flaky  # type: ignore[method-assign]
    audio = np.full(160, 0.5, dtype=np.float32)

    for _ in range(3):
        eng._safe_process_streaming_16k(audio)
    assert eng._consecutive_drops == 3
    eng._safe_process_streaming_16k(audio)  # success
    assert eng._consecutive_drops == 0
    assert not eng._stop_event.is_set()


def test_safe_process_passes_through_on_success() -> None:
    eng = _make_engine_with_stub_inference(raise_exc=None)
    audio = np.full(160, 0.5, dtype=np.float32)
    out = eng._safe_process_streaming_16k(audio)
    assert out is not None
    assert out.size == 160
    np.testing.assert_array_equal(out, audio)
    assert eng.stats.dropped_chunks == 0
    assert eng.stats.last_error is None


# ---- 4. Bonus: EngineStats has the new field ------------------------------


def test_engine_stats_dropped_chunks_starts_zero() -> None:
    s = EngineStats()
    assert s.dropped_chunks == 0
    # Field exists in `fields()` so anyone serialising stats picks it up.
    names = {f.name for f in fields(EngineStats)}
    assert "dropped_chunks" in names
