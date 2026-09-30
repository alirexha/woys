"""v0.6.0 - unit tests for the rename migration script.

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
    # v0.6.4: legacy sink name is rewritten to the v0.6.0+ name.
    assert data["sink_name"] == "WoysSink"
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


def test_migrate_merges_into_already_existing_target(tmp_path: Path) -> None:
    """If the new dir already exists (install.sh pre-creates it, or a
    half-finished previous run), the migrator merges into it without
    trampling anything already there."""
    from migrate_to_woys import migrate

    _build_old_install(tmp_path)
    # Pre-create the destination (simulating a half-completed prior migrate).
    (tmp_path / ".local" / "share" / "woys").mkdir(parents=True)
    (tmp_path / ".local" / "share" / "woys" / "marker").write_text("preexisting")

    migrate(home=tmp_path)
    # The marker should still be there (we didn't trample the new dir).
    assert (tmp_path / ".local" / "share" / "woys" / "marker").read_text() == "preexisting"
    # The legacy content moved in, and the emptied old dir is gone.
    assert (tmp_path / ".local" / "share" / "woys" / "models" / "donald_trump.onnx").is_file()
    assert not (tmp_path / ".local" / "share" / "vcclient-cachy").exists()


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


def test_migrate_rewrites_legacy_sink_name(tmp_path: Path) -> None:
    """v0.6.4 - a v0.5.x config carries `sink_name = "VCClientCachySink"`
    which v0.6.0+ doesn't load (the sink is named `WoysSink` now). Without
    this rewrite, pw-cat falls back to the default sink and audio leaks
    to laptop speakers. See docs/10-monitor-leak-diag.md."""
    from migrate_to_woys import migrate

    _build_old_install(tmp_path)
    migrate(home=tmp_path)

    cfg_path = tmp_path / ".config" / "woys" / "config.toml"
    with open(cfg_path, "rb") as f:
        data = tomllib.load(f)
    assert data["sink_name"] == "WoysSink"


def test_migrate_idempotent_on_already_correct_sink_name(tmp_path: Path) -> None:
    """If a config already has `sink_name = "WoysSink"`, re-running the
    migrator must not corrupt it."""
    from migrate_to_woys import migrate

    _build_old_install(tmp_path)
    migrate(home=tmp_path)
    cfg_path = tmp_path / ".config" / "woys" / "config.toml"
    with open(cfg_path, "rb") as f:
        first = tomllib.load(f)
    assert first["sink_name"] == "WoysSink"

    changed, _log = migrate(home=tmp_path)
    assert changed is False
    with open(cfg_path, "rb") as f:
        second = tomllib.load(f)
    assert second["sink_name"] == "WoysSink"


def test_migrate_bumps_stale_output_latency_ms(tmp_path: Path) -> None:
    """v0.6.7 - `output_latency_ms` below 300 makes the playback backend's
    ring buffer too small to absorb the engine's bursty 250 ms-chunk
    writes. v0.6.7 retro on v0.5.2: pw-cat at 100 ms still drops one
    PipeWire quantum every chunk; pacat at 300 ms is 40x cleaner.
    Migrator bumps any stored value < 300 to 300. See
    `docs/11-microcuts-bug.md` part 2."""
    from migrate_to_woys import migrate

    _build_old_install(tmp_path)
    cfg_path = tmp_path / ".config" / "vcclient-cachy" / "config.toml"
    text = cfg_path.read_text()
    head, _, tail = text.partition("[profiles.default]")
    cfg_path.write_text(head + "output_latency_ms = 30\n[profiles.default]" + tail)

    migrate(home=tmp_path)

    new_cfg = tmp_path / ".config" / "woys" / "config.toml"
    with open(new_cfg, "rb") as f:
        data = tomllib.load(f)
    assert data["output_latency_ms"] == 300


def test_migrate_bumps_intermediate_latency_to_300(tmp_path: Path) -> None:
    """v0.6.7 - also bumps the v0.5.2 default of 100 (which v0.6.7 found
    insufficient) to the new 300 ms baseline."""
    from migrate_to_woys import migrate

    _build_old_install(tmp_path)
    cfg_path = tmp_path / ".config" / "vcclient-cachy" / "config.toml"
    text = cfg_path.read_text()
    head, _, tail = text.partition("[profiles.default]")
    cfg_path.write_text(head + "output_latency_ms = 100\n[profiles.default]" + tail)

    migrate(home=tmp_path)

    new_cfg = tmp_path / ".config" / "woys" / "config.toml"
    with open(new_cfg, "rb") as f:
        data = tomllib.load(f)
    assert data["output_latency_ms"] == 300


def test_migrate_leaves_above_threshold_output_latency_alone(tmp_path: Path) -> None:
    """If the user already has output_latency_ms >= 300, don't touch it
    (preserves any explicit power-user override on the high side)."""
    from migrate_to_woys import migrate

    _build_old_install(tmp_path)
    cfg_path = tmp_path / ".config" / "vcclient-cachy" / "config.toml"
    text = cfg_path.read_text()
    head, _, tail = text.partition("[profiles.default]")
    cfg_path.write_text(head + "output_latency_ms = 500\n[profiles.default]" + tail)

    migrate(home=tmp_path)

    new_cfg = tmp_path / ".config" / "woys" / "config.toml"
    with open(new_cfg, "rb") as f:
        data = tomllib.load(f)
    assert data["output_latency_ms"] == 500


def test_migrate_does_not_rewrite_unrelated_strings_containing_sink_word(
    tmp_path: Path,
) -> None:
    """Rewrite must be exact-string match on `VCClientCachySink`. A free-text
    value that merely *contains* the legacy name as a substring should not
    be modified."""
    from migrate_to_woys import migrate

    _build_old_install(tmp_path)
    cfg_path = tmp_path / ".config" / "vcclient-cachy" / "config.toml"
    # Inject a top-level free-text key BEFORE the [profiles.default] section so
    # it parses at top level, not under the profile.
    text = cfg_path.read_text()
    head, _, tail = text.partition("[profiles.default]")
    cfg_path.write_text(
        head + '_note = "fwd from old VCClientCachySink era"\n' + "[profiles.default]" + tail
    )

    migrate(home=tmp_path)

    new_cfg = tmp_path / ".config" / "woys" / "config.toml"
    with open(new_cfg, "rb") as f:
        data = tomllib.load(f)
    assert data["_note"] == "fwd from old VCClientCachySink era"
    assert data["sink_name"] == "WoysSink"


def test_migrate_merges_legacy_share_into_existing_woys_dir(tmp_path: Path) -> None:
    """install.sh creates ~/.local/share/woys (and its venv) before it runs
    the migrator. The legacy models must still land in woys/models/, where
    the rewritten config points; skipping the move stranded them."""
    from migrate_to_woys import migrate

    _build_old_install(tmp_path)
    new_share = tmp_path / ".local" / "share" / "woys"
    (new_share / "venv" / "bin").mkdir(parents=True)
    (new_share / "venv" / "bin" / "python").write_text("fresh venv")

    migrate(home=tmp_path)

    assert (new_share / "models" / "donald_trump.onnx").is_file()
    assert (new_share / "models" / "amitaro_v2_16k.onnx").is_file()
    # The venv install.sh just built is kept; the legacy one is dropped.
    assert (new_share / "venv" / "bin" / "python").read_text() == "fresh venv"
    # Nothing is left behind, so install.sh's legacy-dir guard goes false.
    assert not (tmp_path / ".local" / "share" / "vcclient-cachy").exists()
    with open(tmp_path / ".config" / "woys" / "config.toml", "rb") as f:
        data = tomllib.load(f)
    assert Path(data["rvc_model"]).is_file()


def test_migrate_merge_never_overwrites_existing_files(tmp_path: Path) -> None:
    """A file already present in the woys dir wins; the legacy copy stays
    where it was so nothing is lost."""
    from migrate_to_woys import migrate

    _build_old_install(tmp_path)
    new_models = tmp_path / ".local" / "share" / "woys" / "models"
    new_models.mkdir(parents=True)
    (new_models / "amitaro_v2_16k.onnx").write_bytes(b"new")

    migrate(home=tmp_path)

    assert (new_models / "amitaro_v2_16k.onnx").read_bytes() == b"new"
    assert (new_models / "donald_trump.onnx").is_file()
    old_copy = tmp_path / ".local" / "share" / "vcclient-cachy" / "models" / "amitaro_v2_16k.onnx"
    assert old_copy.read_bytes() == b"\x00" * 16


def test_migrate_rerun_leaves_the_current_woys_config_alone(tmp_path: Path) -> None:
    """A later install.sh run that still sees a legacy dir must not rewrite
    a config.toml the migrator did not move in this run. Pre-fix every rerun
    bumped the user's output_latency_ms (top level and profiles) to 300."""
    from migrate_to_woys import migrate

    # Only a leftover legacy share dir remains; the config is already woys'.
    (tmp_path / ".local" / "share" / "vcclient-cachy" / "models").mkdir(parents=True)
    cfg = tmp_path / ".config" / "woys" / "config.toml"
    cfg.parent.mkdir(parents=True)
    text = (
        "output_latency_ms = 150\n"
        "config_schema_version = 10\n"
        '_user_overrides = ["output_latency_ms"]\n'
        "\n[profiles.q]\noutput_latency_ms = 120\n"
    )
    cfg.write_text(text)

    migrate(home=tmp_path)

    assert cfg.read_text() == text


def test_migrate_does_not_rewrite_an_existing_config_it_did_not_move(tmp_path: Path) -> None:
    """When both config dirs hold a config.toml, the woys one wins and is
    left untouched; the legacy one stays where it was."""
    from migrate_to_woys import migrate

    _build_old_install(tmp_path)
    cfg = tmp_path / ".config" / "woys" / "config.toml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("output_latency_ms = 150\n")

    migrate(home=tmp_path)

    assert cfg.read_text() == "output_latency_ms = 150\n"
    assert (tmp_path / ".config" / "vcclient-cachy" / "config.toml").is_file()
