"""TUI config round-trip with extras pass-through."""

from __future__ import annotations

import tomllib
from pathlib import Path

from tui.config import AppConfig, load_config, save_config


def test_default_round_trip(tmp_path: Path) -> None:
    cfg = AppConfig()
    out = tmp_path / "c.toml"
    save_config(cfg, out)
    back = load_config(out)
    assert back == cfg


def test_modified_round_trip(tmp_path: Path) -> None:
    cfg = AppConfig(f0_up_key=4, autostart_engine=True, sid=2)
    out = tmp_path / "c.toml"
    save_config(cfg, out)
    back = load_config(out)
    assert back.f0_up_key == 4
    assert back.autostart_engine is True
    assert back.sid == 2


def test_unknown_keys_pass_through(tmp_path: Path) -> None:
    out = tmp_path / "c.toml"
    out.write_text('f0_up_key = 5\nautostart_engine = false\nfuture_user_field = "abc"\n')
    cfg = load_config(out)
    assert cfg.f0_up_key == 5
    # v0.7.0 — load() stamps `config_schema_version` for migration tracking,
    # so it appears alongside genuine extras. The user's unknown key still
    # passes through.
    assert cfg._extras.get("future_user_field") == "abc"
    save_config(cfg, out)
    raw = tomllib.loads(out.read_text())
    assert raw["future_user_field"] == "abc"
    assert raw["f0_up_key"] == 5


def test_load_missing_returns_defaults(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "doesnotexist.toml")
    assert cfg == AppConfig()
