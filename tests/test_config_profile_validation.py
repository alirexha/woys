"""Values inside `[profiles.*]` are validated on load, like top-level ones.

Pre-fix only the top-level fields went through validate_field. A profile
with `chunk_seconds = -1.0` or `f0_up_key = "high"` loaded silently and
apply_profile copied it straight into the running config, where the
engine and the TUI's `{pitch:+d}` formatting crashed on it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_BAD_PROFILES = """config_schema_version = 10
f0_up_key = 2

[profiles.bad]
chunk_seconds = -1.0
f0_up_key = "high"
embedder = "fairseq"
sola_context_ms = nan
monitor = true
my_note = "kept"

[profiles.good]
f0_up_key = 5
chunk_seconds = 0.3
"""


def test_bad_profile_values_reset_to_defaults_on_load(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from tui.config import AppConfig, load_config
    from woys.profiles import _profiles_bag

    p = tmp_path / "config.toml"
    p.write_text(_BAD_PROFILES)
    cfg = load_config(p)
    defaults = AppConfig()
    bad = _profiles_bag(cfg)["bad"]
    assert bad["chunk_seconds"] == defaults.chunk_seconds
    assert bad["f0_up_key"] == defaults.f0_up_key
    assert bad["embedder"] == defaults.embedder
    assert bad["sola_context_ms"] == defaults.sola_context_ms
    # Valid and unknown keys are left alone.
    assert bad["monitor"] is True
    assert bad["my_note"] == "kept"
    assert _profiles_bag(cfg)["good"] == {"f0_up_key": 5, "chunk_seconds": 0.3}
    err = capsys.readouterr().err
    assert "profiles.bad" in err
    assert "chunk_seconds" in err
    assert "profiles.good" not in err


def test_apply_profile_after_load_yields_valid_config(tmp_path: Path) -> None:
    from tui.config import load_config, validate_field
    from woys.profiles import apply_profile

    p = tmp_path / "config.toml"
    p.write_text(_BAD_PROFILES)
    cfg = load_config(p)
    assert apply_profile(cfg, "bad")
    for name in ("chunk_seconds", "f0_up_key", "embedder", "sola_context_ms"):
        assert validate_field(name, getattr(cfg, name)) is None
    # The TUI's profile toast formats the pitch with `:+d`.
    assert f"{cfg.f0_up_key:+d}"


def test_non_table_profile_entry_is_dropped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from tui.config import load_config
    from woys.profiles import _profiles_bag, apply_profile

    p = tmp_path / "config.toml"
    p.write_text("config_schema_version = 10\n[profiles]\nbroken = 5\n[profiles.ok]\nsid = 1\n")
    cfg = load_config(p)
    assert "broken" not in _profiles_bag(cfg)
    assert apply_profile(cfg, "ok")
    assert "profiles.broken" in capsys.readouterr().err
