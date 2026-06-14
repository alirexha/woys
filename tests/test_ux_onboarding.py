"""/
F-23-15 / F-23-16 / F-23-19: the UX onboarding batch.

Each behaviour the batch is supposed to deliver gets a focused test here.
The TUI itself is exercised against `WoysApp` instances constructed with
`no_pw_setup=True` so the tests do not need a real PipeWire daemon; the
HelpScreen modal is covered by the Textual `run_test()` async harness.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "src" / "server") not in sys.path:
    sys.path.insert(0, str(REPO / "src" / "server"))


# ---------------------------------------------------------------------------
# F-23-14: EngineStats.last_error_ts + chunk-success auto-clear
# ---------------------------------------------------------------------------


def test_record_error_stamps_last_error_ts() -> None:
    """record_error must populate `last_error_ts` together with
    `last_error`, so the TUI can render a relative age."""
    from audio import engine

    eng = engine.RealtimeEngine(engine.EngineConfig())
    assert eng.stats.last_error_ts is None
    before = time.monotonic()
    eng.record_error("boom")
    after = time.monotonic()
    ts = eng.stats.last_error_ts
    assert ts is not None
    assert before <= ts <= after


def test_last_error_ts_advances_on_subsequent_record_error() -> None:
    from audio import engine

    eng = engine.RealtimeEngine(engine.EngineConfig())
    eng.record_error("first")
    first_ts = eng.stats.last_error_ts
    time.sleep(0.005)
    eng.record_error("second")
    second_ts = eng.stats.last_error_ts
    assert first_ts is not None and second_ts is not None
    assert second_ts > first_ts


# ---------------------------------------------------------------------------
# F-23-04 / F-23-14 / F-23-19: StatusPanel render variations
# ---------------------------------------------------------------------------


def test_status_panel_idle_renders_press_t_hint() -> None:
    from tui.app import StatusPanel

    panel = StatusPanel()
    out = panel.render_status(
        running=False,
        model=Path("model.onnx"),
        pitch=0,
        profile=None,
        cold_start=False,
        swapping=None,
        error=None,
    )
    assert "stopped" in out
    assert "press [bold]t[/bold] to start" in out
    assert "[bold]?[/bold] for help" in out


def test_status_panel_running_does_not_render_idle_hint() -> None:
    from tui.app import StatusPanel

    panel = StatusPanel()
    out = panel.render_status(
        running=True,
        model=Path("model.onnx"),
        pitch=0,
        profile=None,
        cold_start=False,
        swapping=None,
        error=None,
    )
    assert "press [bold]t[/bold] to start" not in out
    assert "RUNNING" in out


def test_status_panel_error_with_age_is_rendered() -> None:
    from tui.app import StatusPanel

    panel = StatusPanel()
    out = panel.render_status(
        running=True,
        model=Path("model.onnx"),
        pitch=0,
        profile=None,
        cold_start=False,
        swapping=None,
        error="cuda boom",
        error_age_s=3.0,
    )
    assert "cuda boom" in out
    assert "3 s ago" in out


def test_status_panel_mic_silent_only_renders_while_running() -> None:
    from tui.app import StatusPanel

    panel = StatusPanel()
    running_silent = panel.render_status(
        running=True,
        model=Path("model.onnx"),
        pitch=0,
        profile=None,
        cold_start=False,
        swapping=None,
        error=None,
        mic_silent=True,
    )
    assert "no mic signal" in running_silent

    stopped_silent = panel.render_status(
        running=False,
        model=Path("model.onnx"),
        pitch=0,
        profile=None,
        cold_start=False,
        swapping=None,
        error=None,
        mic_silent=True,
    )
    # When the engine is stopped, the mic-silence banner is suppressed --
    # a stopped engine isn't a "broken mic" signal.
    assert "no mic signal" not in stopped_silent


def test_fmt_age_buckets() -> None:
    from tui.app import _fmt_age

    assert _fmt_age(0.4) == "now"
    assert _fmt_age(3.0) == "3 s ago"
    assert _fmt_age(59.0) == "59 s ago"
    assert _fmt_age(60.0) == "1 m ago"
    assert _fmt_age(3599.0) == "59 m ago"
    assert _fmt_age(3600.0) == "1 h ago"


# ---------------------------------------------------------------------------
# F-23-11 / F-23-15: pitch toast on every change + warn past ±24
# ---------------------------------------------------------------------------


def _captured_notifier(app: object) -> list[tuple[Any, ...]]:
    notes: list[tuple[Any, ...]] = []

    def _n(*a: Any, **k: Any) -> None:
        notes.append((a, k))

    # type: ignore[attr-defined]
    app.notify = _n  # type: ignore[assignment]
    return notes


def test_pitch_actions_emit_one_toast_each() -> None:
    from tui.app import WoysApp
    from tui.config import AppConfig

    app = WoysApp(cfg=AppConfig(), no_pw_setup=True)
    notes = _captured_notifier(app)
    app.action_pitch_up()
    app.action_pitch_up()
    app.action_pitch_down()
    app.action_pitch_reset()
    # 4 actions → at least 4 pitch toasts (warnings on top are also possible
    # but with starting pitch 0 we stay well within ±24).
    pitch_msgs = [str(n[0][0]) for n in notes if n[0] and isinstance(n[0][0], str)]
    assert sum(1 for m in pitch_msgs if "pitch " in m and "st" in m) >= 4


def test_pitch_warn_fires_once_per_crossing() -> None:
    from tui.app import _PITCH_WARN_ST, WoysApp
    from tui.config import AppConfig

    app = WoysApp(cfg=AppConfig(), no_pw_setup=True)
    notes = _captured_notifier(app)
    # Walk pitch up past +_PITCH_WARN_ST. Only the *first* tap past the
    # threshold should warn; subsequent taps in the same direction must
    # stay silent (the threshold flag is sticky until pitch comes back
    # below).
    app._apply_pitch(_PITCH_WARN_ST + 1)
    app._apply_pitch(_PITCH_WARN_ST + 2)
    app._apply_pitch(_PITCH_WARN_ST + 3)
    warn_msgs = [
        str(n[0][0])
        for n in notes
        if n[0] and isinstance(n[0][0], str) and "decouple" in str(n[0][0])
    ]
    assert len(warn_msgs) == 1
    # Cross back below and out the other side -- the LOW warn must fire
    # exactly once.
    app._apply_pitch(0)
    app._apply_pitch(-(_PITCH_WARN_ST + 1))
    app._apply_pitch(-(_PITCH_WARN_ST + 2))
    warn_msgs = [
        str(n[0][0])
        for n in notes
        if n[0] and isinstance(n[0][0], str) and "decouple" in str(n[0][0])
    ]
    assert len(warn_msgs) == 2


# ---------------------------------------------------------------------------
# F-23-13: ? opens HelpScreen modal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_help_screen_opens_on_question_mark() -> None:
    from tui.app import HelpScreen, WoysApp
    from tui.config import AppConfig

    app = WoysApp(cfg=AppConfig(autostart_engine=False), no_pw_setup=True)
    async with app.run_test() as pilot:
        await pilot.press("question_mark")
        await pilot.pause()
        # The ModalScreen subclass must now be on the screen stack.
        assert any(isinstance(s, HelpScreen) for s in app.screen_stack)
        # Any key dismisses the modal.
        await pilot.press("space")
        await pilot.pause()
        assert not any(isinstance(s, HelpScreen) for s in app.screen_stack)


# ---------------------------------------------------------------------------
# F-23-09: CLI prompts on profile delete + models download
# ---------------------------------------------------------------------------


def test_cli_profile_delete_aborts_on_non_tty_without_yes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A pipe / here-doc stdin with no `--yes` must NOT delete."""
    import tui.config as tui_config

    # Re-route the config file into tmp_path so the test never touches
    # the user's real ~/.config/woys/config.toml.
    cfg_path = tmp_path / "config.toml"
    monkeypatch.setattr(tui_config, "CONFIG_FILE", cfg_path)
    cfg = tui_config.AppConfig()
    cfg._extras["profiles"] = {"victim": {"f0_up_key": 0, "rvc_model": ""}}
    tui_config.save_config(cfg)
    # Force stdin to look non-interactive.
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    from woys.profiles import cli_profile_delete

    rc = cli_profile_delete("victim", assume_yes=False)
    assert rc == 1
    # Profile must still exist.
    reloaded = tui_config.load_config()
    assert "victim" in reloaded._extras.get("profiles", {})


def test_cli_profile_delete_assume_yes_proceeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import tui.config as tui_config

    cfg_path = tmp_path / "config.toml"
    monkeypatch.setattr(tui_config, "CONFIG_FILE", cfg_path)
    cfg = tui_config.AppConfig()
    cfg._extras["profiles"] = {"victim": {"f0_up_key": 0, "rvc_model": ""}}
    tui_config.save_config(cfg)

    from woys.profiles import cli_profile_delete

    rc = cli_profile_delete("victim", assume_yes=True)
    assert rc == 0
    reloaded = tui_config.load_config()
    assert "victim" not in reloaded._extras.get("profiles", {})


def test_cli_profile_delete_typo_returns_1_without_prompt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A non-existent profile name must error out BEFORE prompting -- the
    prompt would ask the user to confirm a non-action."""
    import tui.config as tui_config

    cfg_path = tmp_path / "config.toml"
    monkeypatch.setattr(tui_config, "CONFIG_FILE", cfg_path)
    tui_config.save_config(tui_config.AppConfig())

    # If the prompt fires we'd EOF here (stdin is whatever pytest gave us)
    # and the test would be flaky. Force isatty to True so the prompt path
    # would be taken if reached, but the early-exit on missing-name must
    # short-circuit before any input() call.
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    called = {"n": 0}

    def _fail_input(_prompt: str) -> str:
        called["n"] += 1
        return "n"

    monkeypatch.setattr("builtins.input", _fail_input)

    from woys.profiles import cli_profile_delete

    rc = cli_profile_delete("does_not_exist", assume_yes=False)
    assert rc == 1
    assert called["n"] == 0
    err = capsys.readouterr().err
    assert "no such profile" in err


class _FakeLfs:
    def __init__(self, size: int) -> None:
        self.size = size
        self.sha256 = "0" * 64


class _FakeSibling:
    def __init__(self, name: str, lfs_size: int | None = None) -> None:
        self.rfilename = name
        self.lfs = _FakeLfs(lfs_size) if lfs_size is not None else None
        self.size = None


class _FakeRepoInfo:
    def __init__(self, siblings: list[_FakeSibling]) -> None:
        self.siblings = siblings
        self.sha = "deadbeef" * 5


class _FakeHfApi:
    def __init__(self, info: _FakeRepoInfo) -> None:
        self._info = info

    def repo_info(self, _repo: str) -> _FakeRepoInfo:
        return self._info


def _patch_hf_api(monkeypatch: pytest.MonkeyPatch, info: _FakeRepoInfo) -> None:
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: _FakeHfApi(info))


def test_cli_models_download_previews_size_and_aborts_on_non_tty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without --yes and without a tty, the preview must print but the
    download must NOT proceed -- not a single byte ships."""
    info = _FakeRepoInfo(
        [
            _FakeSibling("voice_a.onnx", lfs_size=120 * 1024 * 1024),
            _FakeSibling("voice_b.onnx", lfs_size=80 * 1024 * 1024),
            _FakeSibling("README.md"),  # excluded -- not .onnx
        ]
    )
    _patch_hf_api(monkeypatch, info)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)

    from woys import models as wm

    called = {"download_repo": 0}

    def _no_download(*_a: Any, **_k: Any) -> list[Path]:
        called["download_repo"] += 1
        return []

    monkeypatch.setattr(wm, "download_repo", _no_download)

    rc = wm.cli_models_download("dummy/repo", models_dir=tmp_path, assume_yes=False)
    assert rc == 1
    assert called["download_repo"] == 0
    out = capsys.readouterr().out
    assert "2 .onnx file(s)" in out
    assert "200.0 MiB total" in out
    assert "voice_a.onnx" in out
    assert "voice_b.onnx" in out


def test_cli_models_download_yes_skips_prompt_and_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    info = _FakeRepoInfo(
        [
            _FakeSibling("voice_a.onnx", lfs_size=50 * 1024 * 1024),
        ]
    )
    _patch_hf_api(monkeypatch, info)

    from woys import models as wm

    sentinel: list[Path] = [tmp_path / "voice_a.onnx"]

    def _stub_download(*_a: Any, **_k: Any) -> list[Path]:
        return sentinel

    monkeypatch.setattr(wm, "download_repo", _stub_download)

    rc = wm.cli_models_download("dummy/repo", models_dir=tmp_path, assume_yes=True)
    assert rc == 0


def test_cli_models_download_prompt_n_cancels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An interactive 'N' (or bare Enter -- the default) must cancel
    cleanly without invoking download_repo."""
    info = _FakeRepoInfo(
        [
            _FakeSibling("voice.onnx", lfs_size=10 * 1024 * 1024),
        ]
    )
    _patch_hf_api(monkeypatch, info)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys, "stdin", io.StringIO("n\n"))
    monkeypatch.setattr("builtins.input", lambda _p: "n")

    from woys import models as wm

    called = {"download_repo": 0}

    def _no_download(*_a: Any, **_k: Any) -> list[Path]:
        called["download_repo"] += 1
        return []

    monkeypatch.setattr(wm, "download_repo", _no_download)

    rc = wm.cli_models_download("dummy/repo", models_dir=tmp_path, assume_yes=False)
    assert rc == 1
    assert called["download_repo"] == 0
