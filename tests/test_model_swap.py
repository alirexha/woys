"""v0.4.1 — model-switching regression tests.

Covers the three failures reported in `V0_4_1_BUGFIX_BRIEF.md`:
  1. TUI startup ignored `cfg.rvc_model` (used hardcoded Amitaro default).
  2. TUI `p`-key cycle changed the displayed profile name but not the loaded
     RVC model.
  3. CLI `status` didn't include the active model.

Plus the new hot-swap mechanism (`request_model_swap` → worker `_maybe_swap_model`).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

MODELS_DIR = Path.home() / ".local" / "share" / "woys" / "models"
DEFAULT = MODELS_DIR / "amitaro_v2_16k.onnx"


def _have_two_models() -> tuple[Path, Path] | None:
    """Find two distinct RVC ONNX files in the user's library so we have
    a real source / target pair to swap between."""
    if not MODELS_DIR.exists():
        return None
    voices = sorted(
        p
        for p in MODELS_DIR.glob("*.onnx")
        if p.name
        not in {
            "contentvec-f.onnx",
            "contentvec-f-fp16.onnx",
            "rmvpe.onnx",
            "rmvpe_wrapped.onnx",
            "rmvpe_wrapped-fp16.onnx",
            "hubert_base.onnx",
        }
    )
    return (voices[0], voices[1]) if len(voices) >= 2 else None


def test_engine_honors_cfg_rvc_model_on_init() -> None:
    """v0.4.1 #1: the TUI used to drop `cfg.rvc_model` on construct.
    Verify it's now plumbed through to EngineConfig."""
    from audio.engine import DEFAULT_RVC_MODEL
    from tui.app import VCClientApp
    from tui.config import AppConfig

    pair = _have_two_models()
    if pair is None:
        pytest.skip("need ≥ 2 ONNX voice models in the library")
    target, _ = pair

    cfg = AppConfig()
    cfg.rvc_model = str(target.resolve())
    app = VCClientApp(cfg=cfg, no_pw_setup=True)
    assert app.engine.cfg.rvc_model == target.resolve()
    # Sanity: shouldn't have fallen back to Amitaro.
    assert app.engine.cfg.rvc_model != DEFAULT_RVC_MODEL


def test_engine_falls_back_to_default_when_cfg_path_invalid(tmp_path: Path) -> None:
    """A stale config.toml pointing at a deleted .onnx must not brick the TUI —
    we fall back to the engine's hardcoded default and let the user re-pick."""
    from audio.engine import DEFAULT_RVC_MODEL
    from tui.app import VCClientApp
    from tui.config import AppConfig

    cfg = AppConfig()
    cfg.rvc_model = str(tmp_path / "does-not-exist.onnx")
    app = VCClientApp(cfg=cfg, no_pw_setup=True)
    assert app.engine.cfg.rvc_model == DEFAULT_RVC_MODEL


@pytest.mark.gpu
def test_request_model_swap_replaces_rvc_session() -> None:
    """`request_model_swap` queues; `_maybe_swap_model` picks it up + replaces
    the ORT session. We call _maybe_swap_model directly here to avoid
    spinning up the audio thread."""
    from audio.engine import EngineConfig, RealtimeEngine

    pair = _have_two_models()
    if pair is None:
        pytest.skip("need ≥ 2 ONNX voice models")
    a, b = pair

    eng = RealtimeEngine(EngineConfig(chunk_seconds=0.1, rvc_model=a))
    eng._ensure_sessions()
    first_session = eng._rvc
    assert first_session is not None
    assert eng.cfg.rvc_model == a

    eng.request_model_swap(b)
    assert eng._pending_model_swap == b
    eng._maybe_swap_model()
    assert eng._pending_model_swap is None
    assert eng.cfg.rvc_model == b
    # New ORT session, not the same Python object as before.
    assert eng._rvc is not first_session


def test_request_model_swap_is_idempotent_replace() -> None:
    """Two queue-and-replace calls leave the most-recent target pending."""
    from audio.engine import EngineConfig, RealtimeEngine

    eng = RealtimeEngine(EngineConfig(chunk_seconds=0.1))
    eng.request_model_swap(Path("/tmp/first.onnx"))
    eng.request_model_swap(Path("/tmp/second.onnx"))
    assert eng._pending_model_swap == Path("/tmp/second.onnx")


def test_status_handler_includes_model_name() -> None:
    """v0.4.1 #3: STATUS reply must contain `model=<basename>`."""
    from tui.app import VCClientApp
    from tui.config import AppConfig

    pair = _have_two_models()
    if pair is None:
        pytest.skip("need ≥ 2 ONNX voice models")
    target, _ = pair

    cfg = AppConfig()
    cfg.rvc_model = str(target.resolve())
    app = VCClientApp(cfg=cfg, no_pw_setup=True)
    reply = app._handle_control("STATUS")
    assert reply.startswith("OK ")
    assert f"model={target.name}" in reply, reply


def test_model_command_unknown_slug_returns_error() -> None:
    """The MODEL handler should reject unknown slugs with a clear error,
    not silently no-op."""
    from tui.app import VCClientApp
    from tui.config import AppConfig

    app = VCClientApp(cfg=AppConfig(), no_pw_setup=True)
    reply = app._handle_control("MODEL definitely-not-a-real-voice")
    assert reply.startswith("ERR")
    assert "definitely-not-a-real-voice" in reply


def test_cli_models_use_falls_back_to_config_when_no_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When no engine is running, `models use` must still update config —
    just without the obnoxious 'restart the engine' message.

    Patches load_config / save_config inside the models module so they read
    and write a tmp config rather than the user's real ~/.config file.
    `tui.config.CONFIG_FILE` is bound as a default arg at function definition,
    so we have to inject the path through the wrappers themselves.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from tui.config import AppConfig
    from tui.config import load_config as real_load
    from tui.config import save_config as real_save

    cfg_path = tmp_path / "config.toml"
    real_save(AppConfig(), cfg_path)

    pair = _have_two_models()
    if pair is None:
        pytest.skip("need ≥ 2 ONNX voice models")
    target, _ = pair

    def fake_load(*_args: object, **_kwargs: object) -> AppConfig:
        return real_load(cfg_path)

    def fake_save(cfg: AppConfig, *_args: object, **_kwargs: object) -> None:
        real_save(cfg, cfg_path)

    monkeypatch.setattr("woys.models.load_config", fake_load, raising=False)
    monkeypatch.setattr("woys.models.save_config", fake_save, raising=False)
    # Models module imports tui.config inside the function, so patch the
    # imported names there too (after the import happens once). The CLI
    # path imports lazily — easier to patch the source modules.
    monkeypatch.setattr("tui.config.load_config", fake_load)
    monkeypatch.setattr("tui.config.save_config", fake_save)

    from woys.models import cli_models_use

    with patch("tui.control.send_command", return_value="ERR control socket not found"):
        rc = cli_models_use(target.stem)
    assert rc == 0
    cfg2 = real_load(cfg_path)
    assert cfg2.rvc_model == str(target.resolve())
