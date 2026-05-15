"""review F-16-01: config-schema migration honors `_user_overrides`.

Pre-fix all migration legs unconditionally rewrote `field == old_default`
to the new default -- so a user who deliberately pinned
`output_latency_ms = 300` (the v0.6.x default) lost that value on
upgrade. The comment claimed "explicit user overrides are preserved",
which was false. The fix:

  * adds an `_user_overrides` list in the TOML / AppConfig._extras
    naming fields the user has explicitly touched (TUI pitch keys,
    monitor toggle, ...);
  * gates every migration leg behind `name not in user_overrides`;
  * replaces the silent `contextlib.suppress(OSError)` around the
    migration-rewrite with a stderr warning so a read-only home does
    not eat the user's data on subsequent loads.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "src" / "server") not in sys.path:
    sys.path.insert(0, str(REPO / "src" / "server"))


def _write_toml(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


def test_pinned_old_default_survives_migration_when_in_user_overrides(
    tmp_path: Path,
) -> None:
    """The bug-class test. A user who explicitly set
    `output_latency_ms = 300` (which happens to match the v0.6.x default)
    and listed the field in `_user_overrides` keeps that value after a
    schema 0 → 10 migration. Pre-fix the field was clobbered."""
    from tui.config import load_config

    cfg_path = tmp_path / "config.toml"
    _write_toml(
        cfg_path,
        """
config_schema_version = 0
_user_overrides = ["output_latency_ms"]
output_latency_ms = 300
""",
    )
    cfg = load_config(cfg_path)

    assert cfg.output_latency_ms == 300, (
        "F-16-01: a field listed in _user_overrides must survive a schema "
        f"migration even when its value matches an old default; got {cfg.output_latency_ms}"
    )
    assert cfg._extras["config_schema_version"] == 10, (
        "schema_version must still bump even when no fields were migrated"
    )
    # round-trip the marker
    assert "output_latency_ms" in cfg._extras.get("_user_overrides", [])


def test_legacy_config_without_override_marker_still_migrates(
    tmp_path: Path,
) -> None:
    """Back-compat. A pre-v0.14.x config with no `_user_overrides` key
    behaves exactly as before -- `output_latency_ms = 300` (matching the
    v0.6.x default) gets bumped to the current EngineConfig default."""
    from audio.engine import EngineConfig
    from tui.config import load_config

    cfg_path = tmp_path / "config.toml"
    _write_toml(
        cfg_path,
        """
config_schema_version = 0
output_latency_ms = 300
""",
    )
    cfg = load_config(cfg_path)

    assert cfg.output_latency_ms == EngineConfig().output_latency_ms
    assert cfg.output_latency_ms != 300


def test_override_marker_pins_sola_search_ms(tmp_path: Path) -> None:
    """Coverage for a second field (sola_search_ms 4.0 -> 16.0). Without
    the marker the field bumps to the current EngineConfig default; with
    the marker it survives. The verdict's test names output_latency_ms,
    but the bug class covers every migrated default, so we test at
    least two field paths to make sure the gating is not output-latency-
    specific. (Chunk_seconds happens to round-trip 0.25 -> 0.25 today
    -- its leg is a no-op -- which is why this test uses sola_search_ms
    instead.)"""
    from audio.engine import EngineConfig

    assert EngineConfig().sola_search_ms != 4.0, (
        "this test depends on the schema 0 -> 7 sola_search_ms leg "
        "actually changing the value; if the engine default is reset "
        "to 4.0 in the future, pick another field"
    )

    from tui.config import load_config

    bumped = tmp_path / "bumped.toml"
    _write_toml(
        bumped,
        """
config_schema_version = 0
sola_search_ms = 4.0
""",
    )
    cfg_bumped = load_config(bumped)
    assert cfg_bumped.sola_search_ms == EngineConfig().sola_search_ms
    assert cfg_bumped.sola_search_ms != 4.0  # the leg fired

    pinned = tmp_path / "pinned.toml"
    _write_toml(
        pinned,
        """
config_schema_version = 0
_user_overrides = ["sola_search_ms"]
sola_search_ms = 4.0
""",
    )
    cfg_pinned = load_config(pinned)
    assert cfg_pinned.sola_search_ms == 4.0


def test_mark_override_helper_is_idempotent_and_round_trips(
    tmp_path: Path,
) -> None:
    from tui.config import AppConfig, load_config, mark_override, save_config

    cfg = AppConfig()
    mark_override(cfg, "f0_up_key")
    mark_override(cfg, "f0_up_key")  # idempotent
    mark_override(cfg, "monitor")
    assert sorted(cfg._extras["_user_overrides"]) == ["f0_up_key", "monitor"]

    cfg_path = tmp_path / "config.toml"
    save_config(cfg, cfg_path)
    reloaded = load_config(cfg_path)
    assert sorted(reloaded._extras["_user_overrides"]) == ["f0_up_key", "monitor"]


def test_migration_emits_stderr_notice(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A user whose config gets migrated should see a notice on stderr
    that names the old schema -> new schema and where the file lives.
    Pre-fix the migration was silent (the suppress(OSError) wrapper
    hid both successful migrations and write failures)."""
    from tui.config import load_config

    cfg_path = tmp_path / "config.toml"
    _write_toml(
        cfg_path,
        """
config_schema_version = 0
chunk_seconds = 0.25
""",
    )
    load_config(cfg_path)
    out = capsys.readouterr()
    assert "migrated config schema 0" in out.err
    assert "10" in out.err
    assert str(cfg_path) in out.err
    # The notice points users at the escape hatch:
    assert "_user_overrides" in out.err


def test_migration_oserror_logs_to_stderr_instead_of_silent_suppress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Pre-fix the migration-rewrite was wrapped in
    `contextlib.suppress(OSError)` -- so a read-only home, a full disk,
    or a chmod-locked config file silently failed and the next launch
    re-migrated. Post-fix the failure prints a stderr warning naming
    the OSError class so the user knows their config is stale."""
    from tui import config as cfg_mod

    cfg_path = tmp_path / "config.toml"
    _write_toml(
        cfg_path,
        """
config_schema_version = 0
chunk_seconds = 0.25
""",
    )

    real_save = cfg_mod.save_config

    def _boom(cfg: cfg_mod.AppConfig, path: Path = cfg_mod.CONFIG_FILE) -> None:
        if path == cfg_path:
            raise PermissionError(13, "simulated read-only config dir")
        real_save(cfg, path)

    monkeypatch.setattr(cfg_mod, "save_config", _boom)

    cfg = cfg_mod.load_config(cfg_path)
    out = capsys.readouterr()

    # Memory state is still migrated (the user's session works):
    from audio.engine import EngineConfig

    assert cfg.chunk_seconds == EngineConfig().chunk_seconds
    # ...but stderr now tells them the disk write failed:
    assert "failed to persist migrated config" in out.err
    assert "PermissionError" in out.err


def test_first_run_save_oserror_logs_to_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The companion to the migration case: the first-run
    `save_config` (when the config file does not exist) used to be
    wrapped in the same silent `contextlib.suppress(OSError)`.
    A user on a read-only home should see a stderr warning explaining
    why settings will not persist, not silence."""
    from tui import config as cfg_mod

    cfg_path = tmp_path / "missing.toml"
    assert not cfg_path.exists()

    def _boom(cfg: cfg_mod.AppConfig, path: Path = cfg_mod.CONFIG_FILE) -> None:
        raise OSError(30, "Read-only file system")

    monkeypatch.setattr(cfg_mod, "save_config", _boom)

    cfg_mod.load_config(cfg_path)
    out = capsys.readouterr()

    assert "cannot write" in out.err
    assert str(cfg_path) in out.err
    assert "in-memory" in out.err
