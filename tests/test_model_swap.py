"""v0.4.1 - model-switching regression tests.

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
    a real source / target pair to swap between. Excludes amitaro (the
    engine's hardcoded default) so the "didn't fall back to Amitaro"
    assertion in `test_engine_honors_cfg_rvc_model_on_init` is meaningful."""
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
            "amitaro_v2_16k.onnx",  # the engine's default - picking it as
            # `target` would make the `target != DEFAULT` assertion meaningless.
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
    """A stale config.toml pointing at a deleted .onnx must not brick the TUI -
    we fall back to the engine's hardcoded default and let the user re-pick."""
    from audio.engine import DEFAULT_RVC_MODEL
    from tui.app import VCClientApp
    from tui.config import AppConfig

    cfg = AppConfig()
    cfg.rvc_model = str(tmp_path / "does-not-exist.onnx")
    app = VCClientApp(cfg=cfg, no_pw_setup=True)
    assert app.engine.cfg.rvc_model == DEFAULT_RVC_MODEL


@pytest.mark.gpu
def test_resamplers_initialized_in_constructor() -> None:
    """review F-merged-011: `_resampler_in` / `_resampler_out` must be
    initialized in `__init__`.

    They used to sit as dead code after a `return` inside the
    `inference_subprocess_pid` property, so the attributes did not exist
    until `_run_loop` ran -- any swap-path access (`_maybe_swap_model`,
    `reload_rvc`) before the run loop raised AttributeError. Pre-fix this
    test raises AttributeError on the first access; no models needed.
    """
    from audio.engine import EngineConfig, RealtimeEngine

    eng = RealtimeEngine(EngineConfig())
    assert eng._resampler_in is None
    assert eng._resampler_out is None


def test_ensure_sessions_raises_clean_filenotfound_for_missing_model(
    tmp_path: Path,
) -> None:
    """review F-CX2-04: a missing model file must raise a typed
    `FileNotFoundError` naming the path.

    Pre-fix `_ensure_sessions` handed the path straight to ONNX Runtime,
    which raised an opaque ORT-internal exception far from the cause --
    so `pytest.raises(FileNotFoundError)` does not catch it and this test
    errors. Post-fix the existence pre-check raises before any session is
    built (fast, no GPU)."""
    from audio.engine import EngineConfig, RealtimeEngine

    missing = tmp_path / "no_such_voice.onnx"
    eng = RealtimeEngine(
        EngineConfig(
            rvc_model=missing,
            contentvec_model=missing,
            rmvpe_model=missing,
        )
    )
    with pytest.raises(FileNotFoundError) as exc:
        eng._ensure_sessions()
    assert str(missing) in str(exc.value), "the error must name the missing path"


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
    """When no engine is running, `models use` must still update config -
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
    # path imports lazily - easier to patch the source modules.
    monkeypatch.setattr("tui.config.load_config", fake_load)
    monkeypatch.setattr("tui.config.save_config", fake_save)

    from woys.models import cli_models_use

    with patch("tui.control.send_command", return_value="ERR control socket not found"):
        rc = cli_models_use(target.stem)
    assert rc == 0
    cfg2 = real_load(cfg_path)
    assert cfg2.rvc_model == str(target.resolve())


# --- review F-16-07 / F-23-05 ----------------------------------------
# Pre-fix `cli_models_use` only matched `startswith("ERR control socket
# not found")` -- so when the TUI was kill -9'd (stale socket) or
# starting up (connect refused), the user's `woys models use X` was
# silently dropped. The fix broadens the persist branch to cover all
# three transport-failure ERR strings + the state=error path + the
# legacy synchronous-OK path. State=done remains the only branch that
# does NOT persist (the TUI's own MODEL handler at `tui/app.py:306-328`
# already writes config in that branch).
#
# These tests are self-contained: they synthesize a fake model file in
# tmp_path so they run on CI without a real ~/.local/share/woys/models
# library. The existing `test_cli_models_use_falls_back_to_config_when_
# no_socket` above keeps the integration-style real-model test.

_SOCKET_ERR_STRINGS = [
    "ERR control socket not found - TUI not running?",  # pre-fix: already matched
    "ERR control socket stale - TUI not running?",  # pre-fix: dropped on the floor
    "ERR control socket refused - TUI not accepting connections?",  # pre-fix: same
]


def _stub_models_use(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    reply: str,
) -> tuple[Path, Path]:
    """Wire `cli_models_use` so it operates in a hermetic tmp_path:
    - `find_by_name` returns a synthetic onnx Path under tmp_path,
    - `load_config` / `save_config` read+write a tmp config.toml,
    - `submit_and_wait` returns `reply` verbatim.

    Returns `(model_path, config_path)`.
    """
    from tui.config import AppConfig
    from tui.config import load_config as real_load
    from tui.config import save_config as real_save

    fake_model = tmp_path / "fake_voice.onnx"
    fake_model.write_bytes(b"fake-onnx")
    cfg_path = tmp_path / "config.toml"
    real_save(AppConfig(), cfg_path)

    def fake_find(name: str, models_dir: Path = tmp_path) -> Path | None:
        return fake_model if name == fake_model.stem else None

    def fake_load(*_a: object, **_kw: object) -> AppConfig:
        return real_load(cfg_path)

    def fake_save(cfg: AppConfig, *_a: object, **_kw: object) -> None:
        real_save(cfg, cfg_path)

    monkeypatch.setattr("woys.models.find_by_name", fake_find)
    monkeypatch.setattr("tui.config.load_config", fake_load)
    monkeypatch.setattr("tui.config.save_config", fake_save)
    monkeypatch.setattr("woys.models.load_config", fake_load, raising=False)
    monkeypatch.setattr("woys.models.save_config", fake_save, raising=False)
    monkeypatch.setattr("tui.control.submit_and_wait", lambda *a, **kw: reply)
    return fake_model, cfg_path


@pytest.mark.parametrize("reply", _SOCKET_ERR_STRINGS)
def test_cli_models_use_persists_on_every_socket_err_string(
    reply: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bug-class test. For each of the three ERR strings
    `tui.control.send_command` can emit, `models use` must persist
    rvc_model to `config.toml` and exit 0. Pre-fix the `stale` and
    `refused` cases silently dropped the write."""
    from tui.config import AppConfig
    from tui.config import load_config as real_load
    from woys.models import cli_models_use

    model_path, cfg_path = _stub_models_use(monkeypatch, tmp_path, reply)
    rc = cli_models_use(model_path.stem)

    assert rc == 0, f"socket ERR reply {reply!r} must still exit 0; got {rc}"
    cfg2: AppConfig = real_load(cfg_path)
    assert cfg2.rvc_model == str(model_path.resolve()), (
        f"socket ERR reply {reply!r} must persist rvc_model to config "
        f"(F-16-07 bug-class); got {cfg2.rvc_model!r}"
    )


def test_cli_models_use_persists_on_engine_state_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """state=error: the engine actively rejected the swap. Persist the
    user's intent + exit 1 + name the failure on stderr."""
    from tui.config import AppConfig
    from tui.config import load_config as real_load
    from woys.models import cli_models_use

    reply = "OK job=42 state=error msg=onnx load failed"
    model_path, cfg_path = _stub_models_use(monkeypatch, tmp_path, reply)
    rc = cli_models_use(model_path.stem)
    out = capsys.readouterr()

    assert rc == 1
    assert "swap failed" in out.err
    cfg2: AppConfig = real_load(cfg_path)
    assert cfg2.rvc_model == str(model_path.resolve()), (
        "state=error must still persist rvc_model (F-16-07)"
    )


def test_cli_models_use_persists_on_legacy_ok_without_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Old synchronous OK reply (no `job=` field) - shouldn't occur
    post-v0.5.0 but the back-compat branch must persist defensively."""
    from tui.config import AppConfig
    from tui.config import load_config as real_load
    from woys.models import cli_models_use

    reply = "OK swapped"
    model_path, cfg_path = _stub_models_use(monkeypatch, tmp_path, reply)
    rc = cli_models_use(model_path.stem)

    assert rc == 0
    cfg2: AppConfig = real_load(cfg_path)
    assert cfg2.rvc_model == str(model_path.resolve())


def test_cli_models_use_does_not_persist_on_state_done(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """state=done: the TUI's MODEL handler already wrote config under
    its own save_config. We must NOT re-write here (double-write opens
    a TOCTOU window against the TUI for unrelated fields). The CLI's
    in-memory tmp config stays empty (its `_stub_models_use` set
    rvc_model='')."""
    from tui.config import AppConfig
    from tui.config import load_config as real_load
    from woys.models import cli_models_use

    reply = "OK job=1 state=done elapsed_ms=120"
    model_path, cfg_path = _stub_models_use(monkeypatch, tmp_path, reply)
    rc = cli_models_use(model_path.stem)

    assert rc == 0
    cfg2: AppConfig = real_load(cfg_path)
    assert cfg2.rvc_model == "", (
        "state=done is the TUI-handles-persistence branch; the CLI "
        "must NOT also write (F-16-07 design note)"
    )


def test_cli_models_use_does_not_persist_on_unknown_reply(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Unclassified reply: don't persist, surface it. A future engine
    error class might break the contract; persisting on every reply
    masks the real failure."""
    from tui.config import AppConfig
    from tui.config import load_config as real_load
    from woys.models import cli_models_use

    reply = "WAT some-future-thing-we-do-not-know-about"
    model_path, cfg_path = _stub_models_use(monkeypatch, tmp_path, reply)
    rc = cli_models_use(model_path.stem)
    out = capsys.readouterr()

    assert rc == 1
    assert "WAT some-future-thing" in out.err
    cfg2: AppConfig = real_load(cfg_path)
    assert cfg2.rvc_model == ""
