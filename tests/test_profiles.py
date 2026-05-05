"""Tests for the v0.3.0 profiles system."""

from __future__ import annotations

from pathlib import Path

from tui.config import AppConfig, load_config, save_config
from woys.profiles import (
    apply_profile,
    cycle_profile,
    delete_profile,
    list_profiles,
    save_profile,
)


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    """A profile saved → reload → apply → reload should give back the
    snapshotted values."""
    p = tmp_path / "config.toml"
    cfg = AppConfig()
    save_profile(cfg, "default")
    save_config(cfg, p)

    cfg2 = load_config(p)
    cfg2.f0_up_key = 12
    save_profile(cfg2, "high")
    save_config(cfg2, p)

    # New session: should see both profiles + the modified top-level value.
    cfg3 = load_config(p)
    assert list_profiles(cfg3) == ["default", "high"]
    assert cfg3.f0_up_key == 12

    # Applying default rolls back the pitch.
    assert apply_profile(cfg3, "default") is True
    assert cfg3.f0_up_key == 0
    save_config(cfg3, p)

    cfg4 = load_config(p)
    assert cfg4.f0_up_key == 0


def test_apply_unknown_profile_returns_false(tmp_path: Path) -> None:
    cfg = AppConfig()
    assert apply_profile(cfg, "nonexistent") is False


def test_delete_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "config.toml"
    cfg = AppConfig()
    save_profile(cfg, "alpha")
    save_profile(cfg, "beta")
    save_config(cfg, p)

    cfg2 = load_config(p)
    assert list_profiles(cfg2) == ["alpha", "beta"]
    assert delete_profile(cfg2, "alpha") is True
    save_config(cfg2, p)

    cfg3 = load_config(p)
    assert list_profiles(cfg3) == ["beta"]


def test_delete_unknown_profile_returns_false(tmp_path: Path) -> None:
    cfg = AppConfig()
    assert delete_profile(cfg, "ghost") is False


def test_cycle_profile() -> None:
    cfg = AppConfig()
    save_profile(cfg, "a")
    save_profile(cfg, "b")
    save_profile(cfg, "c")
    assert cycle_profile(cfg, None) == "a"
    assert cycle_profile(cfg, "a") == "b"
    assert cycle_profile(cfg, "b") == "c"
    assert cycle_profile(cfg, "c") == "a"  # wraps
    # Unknown current → first.
    assert cycle_profile(cfg, "ghost") == "a"


def test_cycle_with_no_profiles_returns_none() -> None:
    cfg = AppConfig()
    assert cycle_profile(cfg, None) is None


def test_save_overwrites_existing_profile(tmp_path: Path) -> None:
    """Saving a profile twice updates the snapshot, doesn't error."""
    p = tmp_path / "config.toml"
    cfg = AppConfig()
    cfg.f0_up_key = 4
    save_profile(cfg, "p")
    save_config(cfg, p)

    cfg2 = load_config(p)
    cfg2.f0_up_key = -3
    save_profile(cfg2, "p")
    save_config(cfg2, p)

    cfg3 = load_config(p)
    assert apply_profile(cfg3, "p") is True
    assert cfg3.f0_up_key == -3
