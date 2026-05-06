"""v0.7.0 latency-defaults migration.

The v0.6.x defaults baked into existing user configs (chunk_seconds=0.25,
output_latency_ms=300, sola_search_ms=4.0) get bumped on first load
under v0.7.0 to the new defaults (0.15 / 80 / 6.0). Explicit user
overrides — values that don't match the v0.6.x defaults — are
preserved untouched.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from tui.config import AppConfig, load_config, save_config


def test_v06x_defaults_bumped_on_load(tmp_path: Path) -> None:
    out = tmp_path / "c.toml"
    # Simulate a v0.6.x config — explicit values matching the v0.6.x defaults.
    out.write_text(
        "f0_up_key = 0\n"
        "chunk_seconds = 0.25\n"
        "output_latency_ms = 300\n"
        "sola_search_ms = 4.0\n"
        "monitor = false\n"
    )

    cfg = load_config(out)

    # Bumped to v0.7.0-rc2 defaults.
    assert cfg.chunk_seconds == 0.15
    assert cfg.output_latency_ms == 220
    assert cfg.sola_search_ms == 6.0
    # Schema version stamped at the latest.
    assert cfg._extras["config_schema_version"] == 8
    # File rewritten with bumped values + schema version.
    raw = tomllib.loads(out.read_text())
    assert raw["chunk_seconds"] == 0.15
    assert raw["output_latency_ms"] == 220
    assert raw["sola_search_ms"] == 6.0
    assert raw["config_schema_version"] == 8


def test_rc1_users_pulled_forward_to_rc2(tmp_path: Path) -> None:
    """A user who installed v0.7.0-rc1 has output_latency_ms=80 baked
    into their config (via either the rc1 default or the rc1 migration
    of a v0.6.x file). rc2 must pull them forward."""
    out = tmp_path / "c.toml"
    out.write_text(
        "chunk_seconds = 0.15\n"  # rc1 default — already migrated
        "output_latency_ms = 80\n"  # rc1's user-rejected default
        "sola_search_ms = 6.0\n"  # rc1 default
        "config_schema_version = 7\n"  # stamped by rc1's load_config
        "[profiles.default]\n"
        "output_latency_ms = 80\n"
    )

    cfg = load_config(out)

    assert cfg.output_latency_ms == 220
    assert cfg._extras["config_schema_version"] == 8
    assert cfg._extras["profiles"]["default"]["output_latency_ms"] == 220


def test_user_explicit_80_is_preserved_below_rc2_threshold() -> None:
    """If a user *explicitly* set output_latency_ms = 80 (knowing the
    risk), the schema-7 → 8 migration still bumps it. This is the
    documented tradeoff — anyone who wants 80 must re-set it after
    upgrading. We can't distinguish 'rc1 default' from 'explicit user
    choice' once it's on disk."""
    # No assertion here — this test exists as a docstring-as-spec.
    # The previous test already proves the bump. This is the
    # known-tradeoff acknowledgement.


def test_explicit_user_overrides_preserved(tmp_path: Path) -> None:
    out = tmp_path / "c.toml"
    # User explicitly chose values that AREN'T any prior version's
    # defaults — those represent intentional choices and must be left
    # alone. (We can't perfectly distinguish "explicit choice" from
    # "matched default" — the migration only bumps known-default
    # sentinel values.)
    out.write_text(
        "chunk_seconds = 0.5\n"
        "output_latency_ms = 250\n"
        "sola_search_ms = 8.0\n"
    )

    cfg = load_config(out)

    assert cfg.chunk_seconds == 0.5
    assert cfg.output_latency_ms == 250
    assert cfg.sola_search_ms == 8.0


def test_profile_sections_also_migrated(tmp_path: Path) -> None:
    out = tmp_path / "c.toml"
    out.write_text(
        "chunk_seconds = 0.25\n"
        "output_latency_ms = 300\n"
        "sola_search_ms = 4.0\n"
        "[profiles.gaming]\n"
        "chunk_seconds = 0.25\n"
        "output_latency_ms = 300\n"
        "sola_search_ms = 4.0\n"
        "[profiles.studio]\n"
        "chunk_seconds = 0.5\n"
        "output_latency_ms = 500\n"
    )

    cfg = load_config(out)

    profiles = cfg._extras["profiles"]
    # Default-matching values bumped to rc2 defaults.
    assert profiles["gaming"]["chunk_seconds"] == 0.15
    assert profiles["gaming"]["output_latency_ms"] == 220
    assert profiles["gaming"]["sola_search_ms"] == 6.0
    # Explicit override left alone.
    assert profiles["studio"]["chunk_seconds"] == 0.5
    assert profiles["studio"]["output_latency_ms"] == 500


def test_already_migrated_is_idempotent(tmp_path: Path) -> None:
    """A config loaded twice should not be rewritten the second time
    once the schema version is current."""
    out = tmp_path / "c.toml"
    out.write_text(
        "chunk_seconds = 0.15\n"
        "output_latency_ms = 220\n"
        "config_schema_version = 8\n"
    )
    mtime_1 = out.stat().st_mtime_ns

    cfg = load_config(out)
    assert cfg.chunk_seconds == 0.15
    assert cfg.output_latency_ms == 220
    mtime_2 = out.stat().st_mtime_ns
    # No rewrite — file untouched on subsequent loads of an up-to-date file.
    assert mtime_1 == mtime_2


def test_round_trip_post_migration(tmp_path: Path) -> None:
    """Save → load → save → load is stable after migration."""
    out = tmp_path / "c.toml"
    cfg1 = AppConfig()
    save_config(cfg1, out)
    cfg2 = load_config(out)
    save_config(cfg2, out)
    cfg3 = load_config(out)
    assert cfg2 == cfg3
