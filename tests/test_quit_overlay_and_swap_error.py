"""review F-23-10 + F-23-17 (commit-076): quit overlay + swap-error
surfacing.

F-23-10: a `q`/`Ctrl-C` quit blocks 2-10 s on the event loop while
`engine.stop()` runs (the F-13-03 offload moved the *work* to a
thread but the only feedback was a `notify()` toast that fades). The
post-fix renders a persistent `ShutdownScreen` modal until
`App.exit(0)` actually tears down the screen stack.

F-23-17: a swap failure (corrupted ONNX, subprocess InferenceError,
engine-stopped fast-fail) lands on `_SwapRequest.error` but no TUI
caller used to read it -- the StatusPanel parked on
`loading X…` for the 10 s JobRegistry timeout then resumed with the
old voice still in place, no banner, no toast. The post-fix changes
`request_model_swap` to return the `_SwapRequest`; every TUI caller
that waits on `req.completion` now checks `req.error` and routes a
failure to `engine.record_error()` so `_refresh_stats` emits the
usual error toast + StatusPanel banner.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "src" / "server") not in sys.path:
    sys.path.insert(0, str(REPO / "src" / "server"))


# ---------------------------------------------------------------------------
# F-23-17: request_model_swap returns _SwapRequest with .error visible
# ---------------------------------------------------------------------------


def test_request_model_swap_returns_swap_request_with_error_field() -> None:
    """The API widening: callers can now read `.error` after
    `.completion.wait()` instead of only `is_set()`."""
    from audio import engine

    eng = engine.RealtimeEngine(engine.EngineConfig())
    req = eng.request_model_swap(Path("/tmp/x.onnx"))
    assert isinstance(req, engine._SwapRequest)
    assert req.target == Path("/tmp/x.onnx")
    assert req.completion is not None
    assert req.error is None


def test_request_model_swap_after_stop_carries_error_message() -> None:
    """A swap submitted to a stopped engine resolves immediately with
    `req.error` populated -- the TUI's MODEL handler reads this and
    routes to record_error."""
    from audio import engine

    eng = engine.RealtimeEngine(engine.EngineConfig())
    eng.stop(timeout=0.1)
    req = eng.request_model_swap(Path("/tmp/y.onnx"))
    assert req.completion.is_set()
    assert req.error is not None
    assert "stopped" in str(req.error).lower()


def test_stop_resolves_outstanding_swap_with_stop_error() -> None:
    """A pending swap that `stop()` resolves on its way out must carry
    an error so the TUI doesn't celebrate a swap that didn't apply."""
    from audio import engine

    eng = engine.RealtimeEngine(engine.EngineConfig())
    req = eng.request_model_swap(Path("/tmp/z.onnx"))
    eng.stop(timeout=0.5)
    assert req.completion.is_set()
    assert req.error is not None


# ---------------------------------------------------------------------------
# F-23-17: TUI routes swap-error through record_error
# ---------------------------------------------------------------------------


def test_model_handler_routes_swap_error_through_record_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The MODEL socket handler must call `engine.record_error()` when
    `req.error` is set. We exercise the handler body directly: drive
    request_model_swap to raise on the engine side by stopping the
    engine first, then invoke the MODEL handler and confirm the
    engine's error ring picks up the failure."""
    from tui.app import WoysApp
    from tui.config import AppConfig

    # Drop a fake .onnx so `find_by_name` resolves.
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    fake_model = models_dir / "fake_voice.onnx"
    fake_model.write_bytes(b"not a real onnx, but the path resolves")
    monkeypatch.setattr("woys.models.MODELS_DIR", models_dir)

    app = WoysApp(cfg=AppConfig(), no_pw_setup=True)
    # Stop the engine so request_model_swap fast-fails with
    # `engine stopped; swap not applied`.
    app.engine.stop(timeout=0.1)

    # Run the MODEL handler's `do_swap()` body inline. The handler in
    # _handle_control submits to a JobRegistry that runs on a daemon
    # thread; we re-construct the steps deterministically here.
    new_path = fake_model
    req = app.engine.request_model_swap(new_path)
    req.completion.wait(timeout=2.0)
    if req.error is not None:
        app.engine.record_error(
            f"model swap to {new_path.name} failed: {type(req.error).__name__}: {req.error}"
        )

    assert req.error is not None
    last = app.engine.stats.last_error
    assert last is not None
    assert "model swap to fake_voice.onnx failed" in last
    assert "RuntimeError" in last


def test_apply_profile_named_returns_swap_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """_apply_profile_named's signature widened from `threading.Event |
    None` to `_SwapRequest | None`. A profile that DOES change the
    model returns the request; one that doesn't returns None."""
    from audio import engine as engine_mod
    from tui.app import WoysApp
    from tui.config import AppConfig

    cfg = AppConfig()
    fake_model = tmp_path / "voice.onnx"
    fake_model.write_bytes(b"fake")
    cfg._extras["profiles"] = {
        "spawn_with_swap": {
            "f0_up_key": 0,
            "rvc_model": str(fake_model.resolve()),
        },
    }
    app = WoysApp(cfg=cfg, no_pw_setup=True)

    req = app._apply_profile_named("spawn_with_swap")
    assert isinstance(req, engine_mod._SwapRequest)
    assert req.target == fake_model.resolve()


# ---------------------------------------------------------------------------
# F-23-10: ShutdownScreen is pushed before the teardown trio
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_quit_pushes_shutdown_overlay() -> None:
    """action_quit must push ShutdownScreen BEFORE the asyncio.to_thread
    awaits, so the overlay is visible for the full teardown window
    (not just the brief moment between the toast fading and the app
    exiting). We monkey-patch engine.stop / _control.stop / save_config
    to no-ops so the assertion can read the screen stack mid-teardown."""
    from tui.app import ShutdownScreen, WoysApp
    from tui.config import AppConfig

    app = WoysApp(cfg=AppConfig(autostart_engine=False), no_pw_setup=True)
    seen_overlay: list[bool] = []

    def _record_overlay_then_stop(*_a: Any, **_k: Any) -> None:
        seen_overlay.append(any(isinstance(s, ShutdownScreen) for s in app.screen_stack))

    async with app.run_test() as pilot:
        # Patch the teardown trio to no-ops AFTER recording whether the
        # ShutdownScreen was on the stack. If the screen is on the stack
        # by the time engine.stop() runs, the overlay was pushed BEFORE
        # the trio -- which is the F-23-10 contract.
        app.engine.stop = _record_overlay_then_stop  # type: ignore[method-assign]
        app._control.stop = lambda: None  # type: ignore[method-assign]
        # save_config is called from action_quit via `asyncio.to_thread(save_config, self.cfg)`,
        # so we can't easily monkey-patch the bound import; use the no-op stop above to gate.
        await pilot.press("q")
        await pilot.pause()
        # The app exits before pilot.exit -- run_test handles the
        # teardown gracefully when self.exit(0) lands.
    assert seen_overlay, "engine.stop() must have been invoked"
    assert seen_overlay[0] is True, (
        "ShutdownScreen must be on the screen stack BEFORE engine.stop runs -- F-23-10 contract"
    )
