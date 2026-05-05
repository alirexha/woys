"""v0.6.0 — unit tests for the rename migration script.

Drive `scripts/migrate_to_woys.py::migrate` against a synthetic $HOME in
tmp_path, verify everything moves to the new layout and `config.toml` paths
get rewritten without losing data.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest

# Make scripts/ importable.
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _build_old_install(home: Path) -> Path:
    """Lay down a fake `vcclient-cachy` install under a fake $HOME."""
    share = home / ".local" / "share" / "vcclient-cachy"
    config = home / ".config" / "vcclient-cachy"
    cache = home / ".cache" / "vcclient-cachy"
    (share / "models").mkdir(parents=True)
    (share / "venv").mkdir(parents=True)
    config.mkdir(parents=True)
    cache.mkdir(parents=True)
    (share / "models" / "amitaro_v2_16k.onnx").write_bytes(b"\x00" * 16)
    (share / "models" / "donald_trump.onnx").write_bytes(b"\x00" * 16)

    config_text = (
        f'rvc_model = "{share}/models/donald_trump.onnx"\n'
        "f0_up_key = 0\n"
        'sink_name = "VCClientCachySink"\n'
        "\n"
        "[profiles.default]\n"
        f'rvc_model = "{share}/models/amitaro_v2_16k.onnx"\n'
        '_display = "Amitaro"\n'
    )
    (config / "config.toml").write_text(config_text)
    return config / "config.toml"


def test_migrate_fresh_install_is_noop(tmp_path: Path) -> None:
    from migrate_to_woys import migrate

    changed, log = migrate(home=tmp_path)
    assert changed is False
    assert "fresh install path" in "\n".join(log)
    # No woys/ dirs created on a fresh box either.
    assert not (tmp_path / ".local" / "share" / "woys").exists()


def test_migrate_moves_all_dirs(tmp_path: Path) -> None:
    from migrate_to_woys import migrate

    _build_old_install(tmp_path)
    changed, _log = migrate(home=tmp_path)
    assert changed is True

    # Old dirs gone.
    assert not (tmp_path / ".local" / "share" / "vcclient-cachy").exists()
    assert not (tmp_path / ".config" / "vcclient-cachy").exists()
    assert not (tmp_path / ".cache" / "vcclient-cachy").exists()

    # New dirs present with their contents.
    new_share = tmp_path / ".local" / "share" / "woys"
    new_config = tmp_path / ".config" / "woys"
    assert (new_share / "models" / "amitaro_v2_16k.onnx").is_file()
    assert (new_share / "models" / "donald_trump.onnx").is_file()
    assert (new_share / "venv").is_dir()
    assert (new_config / "config.toml").is_file()
    assert (tmp_path / ".cache" / "woys").is_dir()


def test_migrate_rewrites_model_paths_in_config(tmp_path: Path) -> None:
    from migrate_to_woys import migrate

    _build_old_install(tmp_path)
    migrate(home=tmp_path)

    cfg_path = tmp_path / ".config" / "woys" / "config.toml"
    with open(cfg_path, "rb") as f:
        data = tomllib.load(f)

    # Top-level rvc_model and the nested profile both point to the new layout.
    assert "/woys/models/donald_trump.onnx" in data["rvc_model"]
    assert "vcclient-cachy" not in data["rvc_model"]
    assert "/woys/models/amitaro_v2_16k.onnx" in data["profiles"]["default"]["rvc_model"]
    assert "vcclient-cachy" not in data["profiles"]["default"]["rvc_model"]

    # Other fields preserved verbatim.
    assert data["f0_up_key"] == 0
    assert data["sink_name"] == "VCClientCachySink"
    assert data["profiles"]["default"]["_display"] == "Amitaro"


def test_migrate_idempotent_when_target_exists(tmp_path: Path) -> None:
    """Running twice must not corrupt state. Second run is a near-no-op."""
    from migrate_to_woys import migrate

    _build_old_install(tmp_path)
    migrate(home=tmp_path)
    # Second pass: the OLD dirs are gone, NEW dirs exist → migrator should
    # detect 'fresh install path' and return changed=False.
    changed, _log = migrate(home=tmp_path)
    assert changed is False


def test_migrate_skips_target_if_already_migrated_partial(tmp_path: Path) -> None:
    """If the new dir already exists from a half-finished previous run, the
    migrator should not overwrite it (the old dir would still be present and
    would be left alone — operator intervention required)."""
    from migrate_to_woys import migrate

    _build_old_install(tmp_path)
    # Pre-create the destination (simulating a half-completed prior migrate).
    (tmp_path / ".local" / "share" / "woys").mkdir(parents=True)
    (tmp_path / ".local" / "share" / "woys" / "marker").write_text("preexisting")

    migrate(home=tmp_path)
    # The marker should still be there (we didn't trample the new dir).
    assert (tmp_path / ".local" / "share" / "woys" / "marker").read_text() == "preexisting"
    # The old dir should still exist (we didn't move on top of the existing target).
    assert (tmp_path / ".local" / "share" / "vcclient-cachy").exists()


def test_migrate_dry_run_reports_but_changes_nothing(tmp_path: Path) -> None:
    from migrate_to_woys import migrate

    _build_old_install(tmp_path)
    changed, log = migrate(home=tmp_path, dry_run=True)
    assert changed is True
    assert any("dry_run=True" in line for line in log)
    # All old paths still in place.
    assert (tmp_path / ".config" / "vcclient-cachy" / "config.toml").exists()
    assert (tmp_path / ".local" / "share" / "vcclient-cachy" / "models").exists()
    assert not (tmp_path / ".local" / "share" / "woys").exists()


@pytest.mark.parametrize("missing", ["share", "config", "cache"])
def test_migrate_partial_install(tmp_path: Path, missing: str) -> None:
    """User might have $HOME/.config/vcclient-cachy but not the share dir
    (or vice versa). Migrator must move whatever's there without erroring on
    the missing one."""
    from migrate_to_woys import migrate

    _build_old_install(tmp_path)
    if missing == "share":
        import shutil

        shutil.rmtree(tmp_path / ".local" / "share" / "vcclient-cachy")
    elif missing == "config":
        import shutil

        shutil.rmtree(tmp_path / ".config" / "vcclient-cachy")
    elif missing == "cache":
        import shutil

        shutil.rmtree(tmp_path / ".cache" / "vcclient-cachy")

    changed, _log = migrate(home=tmp_path)
    assert changed is True
