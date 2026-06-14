"""protocol_version constant
on the wire + cli_profile_use wired to the orphaned PROFILE socket
handler.

Pre-fix:
- The control-protocol docstring at the top of `tui/control.py` reasoned
  about "older clients" but no version field existed. A future
  incompatible wire change would silently fail on older clients.
- The PROFILE socket handler at `tui/app.py:342-357` was functional but
  unreachable from the CLI: `woys profile use` only wrote `config.toml`
  and told the user to "restart the engine" -- so even with the TUI
  running, the live-apply path was orphaned.

Post-fix:
- `PROTOCOL_VERSION = 1` constant in `tui/control.py`. The STATUS reply
  stamps `proto=1` so a client can read the server version.
- `cli_profile_use` now tries the socket first (live-apply via
  `submit_and_wait("PROFILE name")`), with `save_config` as the
  fallback on transport ERR / state=error / unknown.

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


# --- protocol_version constant + STATUS stamp -----------------------------


def test_protocol_version_constant_exists() -> None:
    """`PROTOCOL_VERSION` is the wire-version contract anchor; future
    bumps land here + the wire-history comment block above it."""
    from tui.control import PROTOCOL_VERSION

    assert isinstance(PROTOCOL_VERSION, int)
    assert PROTOCOL_VERSION >= 1


def test_status_reply_stamps_protocol_version() -> None:
    """The STATUS handler must include `proto=<N>` in its reply so a
    client can detect the server's wire version. Pre-fix the STATUS
    reply had no version field at all."""
    from tui.app import VCClientApp
    from tui.config import AppConfig
    from tui.control import PROTOCOL_VERSION

    app = VCClientApp(cfg=AppConfig(), no_pw_setup=True)
    # The STATUS handler reads from self.engine.stats which exists
    # after __init__; it does NOT require start(). Same shortcut the
    # existing `test_status_handler_includes_model_name` uses.
    reply = app._handle_control("STATUS")

    assert reply.startswith("OK")
    assert f"proto={PROTOCOL_VERSION}" in reply, (
        f"STATUS reply must include `proto={PROTOCOL_VERSION}` so older/"
        f"newer clients can detect the server version; got: {reply!r}"
    )


# --- cli_profile_use wired to the PROFILE socket handler ------------------


def _stub_cli_profile_use(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    profile_name: str = "loud",
    reply: str,
) -> tuple[Path, list[Any]]:
    """Set up a hermetic environment for cli_profile_use: a tmp
    config.toml with the named profile present, monkeypatched
    save_config / load_config / submit_and_wait. Returns
    `(cfg_path, save_calls)` so tests can both read the resulting
    config and assert on how many times save_config was called."""
    from tui.config import AppConfig
    from tui.config import load_config as real_load
    from tui.config import save_config as real_save
    from woys.profiles import save_profile

    cfg_path = tmp_path / "config.toml"
    cfg = AppConfig()
    cfg.rvc_model = "/fake/voice.onnx"
    cfg.f0_up_key = 5
    save_profile(cfg, profile_name)
    real_save(cfg, cfg_path)

    def fake_load(*_a: object, **_kw: object) -> AppConfig:
        return real_load(cfg_path)

    save_calls: list[Any] = []

    def fake_save(c: AppConfig, *_a: object, **_kw: object) -> None:
        save_calls.append(c)
        real_save(c, cfg_path)

    monkeypatch.setattr("tui.config.load_config", fake_load)
    monkeypatch.setattr("tui.config.save_config", fake_save)
    monkeypatch.setattr("woys.profiles.load_config", fake_load, raising=False)
    monkeypatch.setattr("woys.profiles.save_config", fake_save, raising=False)
    monkeypatch.setattr("tui.control.submit_and_wait", lambda *_a, **_kw: reply)
    return cfg_path, save_calls


def test_cli_profile_use_sends_profile_socket_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bug-class half-A: cli_profile_use must actually send PROFILE on
    the socket. Pre-fix the PROFILE handler in the TUI was orphaned --
    no caller reached it from the CLI side."""
    sent: list[str] = []

    def _capturing_submit(cmd: str, **_kw: Any) -> str:
        sent.append(cmd)
        return "OK job=42 state=done elapsed_ms=15"

    from tui.config import AppConfig
    from tui.config import load_config as real_load
    from tui.config import save_config as real_save
    from woys.profiles import save_profile

    cfg_path = tmp_path / "config.toml"
    cfg = AppConfig()
    save_profile(cfg, "loud")
    real_save(cfg, cfg_path)

    def fake_load(*_a: object, **_kw: object) -> AppConfig:
        return real_load(cfg_path)

    monkeypatch.setattr("tui.config.load_config", fake_load)
    monkeypatch.setattr("tui.config.save_config", lambda *a, **kw: None)
    monkeypatch.setattr("tui.control.submit_and_wait", _capturing_submit)

    from woys.profiles import cli_profile_use

    rc = cli_profile_use("loud")

    assert rc == 0
    assert sent == ["PROFILE loud"], f"cli_profile_use must send PROFILE on the socket; got {sent}"


def test_cli_profile_use_does_not_persist_on_state_done(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """state=done: the TUI's _apply_profile_named (app.py:519) already
    saved config. The CLI must NOT also write -- double-writing opens
    a TOCTOU window against the TUI for unrelated fields. Mirrors the
    F-16-07 `models use` design note."""
    _cfg_path, save_calls = _stub_cli_profile_use(
        monkeypatch,
        tmp_path,
        reply="OK job=1 state=done elapsed_ms=20",
    )
    from woys.profiles import cli_profile_use

    rc = cli_profile_use("loud")
    out = capsys.readouterr()

    assert rc == 0
    assert "applied live" in out.out
    assert save_calls == [], (
        f"state=done must NOT trigger a CLI-side save_config; got {len(save_calls)} call(s)"
    )


def test_cli_profile_use_persists_and_fails_on_state_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """state=error: TUI rejected the profile (e.g., referenced model
    file missing). Persist config (user's intent) + return 1."""
    _cfg_path, save_calls = _stub_cli_profile_use(
        monkeypatch,
        tmp_path,
        reply="OK job=2 state=error msg=onnx not found",
    )
    from woys.profiles import cli_profile_use

    rc = cli_profile_use("loud")
    out = capsys.readouterr()

    assert rc == 1
    assert "live-apply failed" in out.err
    assert len(save_calls) == 1, (
        "state=error must persist config to preserve user intent; "
        f"got {len(save_calls)} save_config calls"
    )


@pytest.mark.parametrize(
    "reply",
    [
        "ERR control socket not found - TUI not running?",
        "ERR control socket stale - TUI not running?",
        "ERR control socket refused - TUI not accepting connections?",
    ],
)
def test_cli_profile_use_persists_on_every_transport_err(
    reply: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Transport-layer ERR (any of the 3 strings): engine not running.
    Persist config so the next launch picks up the profile.
    Mirrors F-16-07's "every non-state=done branch persists" design."""
    _cfg_path, save_calls = _stub_cli_profile_use(monkeypatch, tmp_path, reply=reply)
    from woys.profiles import cli_profile_use

    rc = cli_profile_use("loud")
    out = capsys.readouterr()

    assert rc == 0
    assert "active profile -> loud" in out.out
    assert "engine not running" in out.out
    assert len(save_calls) == 1, (
        f"transport ERR ({reply!r}) must persist config for the next "
        f"launch; got {len(save_calls)} save_config calls"
    )


def test_cli_profile_use_returns_1_when_profile_unknown_locally(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Local-only failure: if the requested profile isn't in
    config.toml at all, fail BEFORE touching the socket. The socket
    handler would also reject it, but the local check is faster and
    more informative ('available: a, b, c')."""
    from tui.config import AppConfig, save_config
    from tui.config import load_config as real_load

    cfg_path = tmp_path / "config.toml"
    save_config(AppConfig(), cfg_path)

    def fake_load(*_a: object, **_kw: object) -> AppConfig:
        return real_load(cfg_path)

    sent: list[str] = []
    monkeypatch.setattr("tui.config.load_config", fake_load)
    monkeypatch.setattr("woys.profiles.load_config", fake_load, raising=False)
    monkeypatch.setattr("tui.control.submit_and_wait", lambda *a, **kw: sent.append(a[0]) or "OK")

    from woys.profiles import cli_profile_use

    rc = cli_profile_use("nonexistent")
    err = capsys.readouterr().err

    assert rc == 1
    assert "no such profile" in err
    assert sent == [], (
        "unknown-profile path must NOT touch the socket; the local "
        "check is faster and more informative"
    )
