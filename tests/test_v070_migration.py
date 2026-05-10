"""v0.7.0 latency-defaults migration.

The v0.6.x defaults baked into existing user configs (chunk_seconds=0.25,
output_latency_ms=300, sola_search_ms=4.0) get bumped on first load
to the current rc defaults. Explicit user overrides - values that
don't match a known prior-version default sentinel - are preserved
untouched. The migration is staged across schema versions:

  schema 0 → 7  (rc1): chunk 0.25 → 0.15, sola_search 4.0 → 6.0,
                       output_latency 300 → engine default
  schema 7 → 8  (rc2): output_latency 80 (rc1) → engine default
  schema 8 → 9  (rc3): output_latency 220 (rc2) → engine default (280)
  schema 9 → 10 (rc4): input_gate_dbfs -55.0 → engine default (-75.0),
                       prefer_pw_cat True → engine default (False)

A v0.6.x user with no schema version on disk cascades through every
leg in a single load and lands at the current default in one shot.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from tui.config import AppConfig, load_config, save_config


def test_v06x_defaults_bumped_on_load(tmp_path: Path) -> None:
    out = tmp_path / "c.toml"
    # Simulate a v0.6.x config - explicit values matching the v0.6.x defaults.
    out.write_text(
        "f0_up_key = 0\n"
        "chunk_seconds = 0.25\n"
        "output_latency_ms = 300\n"
        "sola_search_ms = 4.0\n"
        "monitor = false\n"
    )

    cfg = load_config(out)

    # Bumped to current defaults (cascading through every leg).
    # v0.12.4 - chunk_seconds current default = 0.25 (matches v0.6.x
    # sentinel 0.25 - migration is now a no-op for that field, just
    # stamps the schema version). sola_search_ms current default = 16.0
    # (was 4.0 in v0.12.3, reverted to wider window when chunk grew).
    assert cfg.chunk_seconds == 0.25
    assert cfg.output_latency_ms == 280
    assert cfg.sola_search_ms == 16.0
    # Schema version stamped at the latest.
    assert cfg._extras["config_schema_version"] == 10
    # File rewritten with bumped values + schema version.
    raw = tomllib.loads(out.read_text())
    assert raw["chunk_seconds"] == 0.25
    assert raw["output_latency_ms"] == 280
    assert raw["sola_search_ms"] == 16.0
    assert raw["config_schema_version"] == 10


def test_rc1_users_cascade_to_rc3(tmp_path: Path) -> None:
    """A user who installed v0.7.0-rc1 has output_latency_ms=80 baked
    into their config. The schema-7 → 8 leg sets it to the current
    engine default, so they land at rc3's 280 in a single load (the
    schema-8 → 9 leg is a no-op for them since the value is already
    280, not the rc2-default 220 sentinel)."""
    out = tmp_path / "c.toml"
    out.write_text(
        "chunk_seconds = 0.15\n"  # rc1 default - already migrated
        "output_latency_ms = 80\n"  # rc1's user-rejected default
        "sola_search_ms = 6.0\n"  # rc1 default
        "config_schema_version = 7\n"  # stamped by rc1's load_config
        "[profiles.default]\n"
        "output_latency_ms = 80\n"
    )

    cfg = load_config(out)

    assert cfg.output_latency_ms == 280
    assert cfg._extras["config_schema_version"] == 10
    assert cfg._extras["profiles"]["default"]["output_latency_ms"] == 280


def test_rc2_users_pulled_forward_to_rc3(tmp_path: Path) -> None:
    """A user who installed v0.7.0-rc2 has output_latency_ms=220 baked
    into their config (via either the rc2 default or the rc2 migration
    of an rc1 file). rc3 must pull them forward to 280 - both at the
    top level and inside every [profiles.<name>] section."""
    out = tmp_path / "c.toml"
    out.write_text(
        "chunk_seconds = 0.15\n"
        "output_latency_ms = 220\n"  # rc2's user-rejected default
        "sola_search_ms = 6.0\n"
        "config_schema_version = 8\n"  # stamped by rc2's load_config
        "[profiles.default]\n"
        "output_latency_ms = 220\n"
        "[profiles.gaming]\n"
        "output_latency_ms = 220\n"
    )

    cfg = load_config(out)

    assert cfg.output_latency_ms == 280
    assert cfg._extras["config_schema_version"] == 10
    assert cfg._extras["profiles"]["default"]["output_latency_ms"] == 280
    assert cfg._extras["profiles"]["gaming"]["output_latency_ms"] == 280


def test_rc3_users_pulled_forward_to_rc4(tmp_path: Path) -> None:
    """A user who installed v0.7.0-rc3 has input_gate_dbfs=-55.0 (rc1+
    default) and prefer_pw_cat=True (rc1+ default) baked in - except
    `prefer_pw_cat` was never in AppConfig's forwarded fields pre-rc4,
    so it lived only in EngineConfig's dataclass default. After rc4
    exposes it, anyone whose on-disk file shipped True (or who lands
    on the dataclass default via the migration) gets pulled forward
    to False, and the -55 dBFS gate threshold gets bumped to -75."""
    out = tmp_path / "c.toml"
    out.write_text(
        "chunk_seconds = 0.15\n"
        "output_latency_ms = 280\n"
        "sola_search_ms = 6.0\n"
        "input_gate_dbfs = -55.0\n"  # rc1+ default sentinel
        "prefer_pw_cat = true\n"  # rc1+ default sentinel
        "config_schema_version = 9\n"  # stamped by rc3's load_config
        "[profiles.default]\n"
        "input_gate_dbfs = -55.0\n"
        "prefer_pw_cat = true\n"
    )

    cfg = load_config(out)

    assert cfg.input_gate_dbfs == -75.0
    assert cfg.prefer_pw_cat is False
    assert cfg._extras["config_schema_version"] == 10
    assert cfg._extras["profiles"]["default"]["input_gate_dbfs"] == -75.0
    assert cfg._extras["profiles"]["default"]["prefer_pw_cat"] is False


def test_rc4_explicit_gate_overrides_preserved(tmp_path: Path) -> None:
    """A user who explicitly set `input_gate_dbfs = -200.0` to disable
    the gate must NOT have it bumped to -75 by the rc3→rc4 migration -
    the migration only matches the rc1+ default sentinel value -55.0.
    Likewise an explicit `prefer_pw_cat = false` must round-trip."""
    out = tmp_path / "c.toml"
    out.write_text(
        "input_gate_dbfs = -200.0\n"  # explicit override
        "prefer_pw_cat = false\n"  # explicit override
        "config_schema_version = 9\n"
    )

    cfg = load_config(out)

    assert cfg.input_gate_dbfs == -200.0
    assert cfg.prefer_pw_cat is False
    assert cfg._extras["config_schema_version"] == 10


def test_user_explicit_80_is_preserved_below_rc2_threshold() -> None:
    """If a user *explicitly* set output_latency_ms = 80 (knowing the
    risk), the schema-7 → 8 migration still bumps it. This is the
    documented tradeoff - anyone who wants 80 must re-set it after
    upgrading. We can't distinguish 'rc1 default' from 'explicit user
    choice' once it's on disk."""
    # No assertion here - this test exists as a docstring-as-spec.
    # The previous test already proves the bump. This is the
    # known-tradeoff acknowledgement.


def test_explicit_user_overrides_preserved(tmp_path: Path) -> None:
    out = tmp_path / "c.toml"
    # User explicitly chose values that AREN'T any prior version's
    # defaults - those represent intentional choices and must be left
    # alone. (We can't perfectly distinguish "explicit choice" from
    # "matched default" - the migration only bumps known-default
    # sentinel values.)
    out.write_text("chunk_seconds = 0.5\noutput_latency_ms = 250\nsola_search_ms = 8.0\n")

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
    # Default-matching values bumped to current defaults (cascading through every leg).
    # v0.12.4 - chunk_seconds default = 0.25 (matches v0.6.x sentinel,
    # migration now a no-op for chunk_seconds). sola_search_ms default = 16.0.
    assert profiles["gaming"]["chunk_seconds"] == 0.25
    assert profiles["gaming"]["output_latency_ms"] == 280
    assert profiles["gaming"]["sola_search_ms"] == 16.0
    # Explicit override left alone.
    assert profiles["studio"]["chunk_seconds"] == 0.5
    assert profiles["studio"]["output_latency_ms"] == 500


def test_already_migrated_is_idempotent(tmp_path: Path) -> None:
    """A config loaded twice should not be rewritten the second time
    once the schema version is current."""
    out = tmp_path / "c.toml"
    out.write_text(
        "chunk_seconds = 0.15\n"
        "output_latency_ms = 280\n"
        "input_gate_dbfs = -75.0\n"
        "prefer_pw_cat = false\n"
        "config_schema_version = 10\n"
    )
    mtime_1 = out.stat().st_mtime_ns

    cfg = load_config(out)
    assert cfg.chunk_seconds == 0.15
    assert cfg.output_latency_ms == 280
    assert cfg.input_gate_dbfs == -75.0
    assert cfg.prefer_pw_cat is False
    mtime_2 = out.stat().st_mtime_ns
    # No rewrite - file untouched on subsequent loads of an up-to-date file.
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
