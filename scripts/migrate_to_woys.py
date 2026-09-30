"""v0.6.0 - migrate an existing vcclient-cachy install to woys.

Run by `install.sh` with the new venv's python, after the venv and its
dependencies are built, so the user's models + config + systemd unit move
to the new layout. Safe to re-run (idempotent) and safe to invoke on a
fresh install (no-op).

What moves:
    ~/.config/vcclient-cachy/         →  ~/.config/woys/
    ~/.local/share/vcclient-cachy/    →  ~/.local/share/woys/
    ~/.cache/vcclient-cachy/          →  ~/.cache/woys/  (if present)

What gets rewritten:
    config.toml: any string containing 'vcclient-cachy/models/' becomes
    'woys/models/' (covers `rvc_model` + per-profile model paths). We use a
    real TOML parse + rewrite; no sed.

Systemd:
    Old unit `vcclient-cachy-mic.service` is stopped + disabled + removed.
    Install of the new unit (`woys-mic.service`) is left to install.sh -
    that's where the new file lives.

PipeWire:
    v0.6.0 to v0.6.4: the user-facing SOURCE name (`vcclient-mic`) was
    deliberately preserved across the rename so Discord / CS2 / Telegram
    didn't need re-configuration.
    v0.6.5: that compromise was retired - the source is now `woys-mic`
    too. Apps need to re-select their input device once. The remap-source
    rename is handled by `pipewire.VirtualMic.ensure()` (it unloads any
    legacy `vcclient-mic` module before loading the new one), not by this
    migrator.

    The internal SINK name changed in v0.6.0 (`VCClientCachySink` →
    `WoysSink`). This migrator rewrites the `sink_name` key in
    `config.toml` accordingly so the engine targets the sink that
    v0.6.0+ actually loads. v0.6.4 fix - without this rewrite,
    `pw-cat --target=VCClientCachySink` silently falls back to the
    default sink (laptop speakers) since the legacy sink no longer
    exists. See `docs/10-monitor-leak-diag.md`.

Usage:
    ~/.local/share/woys/venv/bin/python scripts/migrate_to_woys.py [--dry-run]

    Writing the rewritten config.toml needs tomli_w (in the woys venv).

Exits 0 on success or no-op, non-zero on hard error.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tomllib
from pathlib import Path
from typing import Any

OLD_NAME = "vcclient-cachy"
NEW_NAME = "woys"

# v0.6.4 - the v0.6.0 rename also changed the internal PipeWire sink
# name. Configs from v0.5.x carry the legacy string and must be rewritten
# or the engine routes playback to the default sink (laptop speakers).
LEGACY_SINK_NAME = "VCClientCachySink"
NEW_SINK_NAME = "WoysSink"

# Anchor points relative to $HOME - overridable for tests.
DEFAULT_HOME = Path.home()


def _move_path(old: Path, new: Path) -> None:
    new.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.rename(old, new)  # atomic when on the same filesystem
    except OSError:
        # Cross-FS fallback. On a personal dev box this should never
        # happen ($HOME is one mount), but install.sh shouldn't crash if
        # the user's $HOME spans two mounts.
        if old.is_dir():
            shutil.copytree(old, new, symlinks=True)
            shutil.rmtree(old)
        else:
            shutil.copy2(old, new, follow_symlinks=False)
            old.unlink()


def _move_dir(
    old: Path,
    new: Path,
    *,
    dry_run: bool,
    log: list[str],
    drop_on_conflict: frozenset[str] = frozenset(),
) -> None:
    """Move `old` to `new`: a plain rename when `new` does not exist yet,
    otherwise a merge.

    install.sh builds ~/.local/share/woys/venv before it runs us, so the
    share target always exists by then. Merging moves every entry that is
    missing in `new`, recurses into directories present on both sides, and
    never overwrites: a conflicting file stays in `old`. Names in
    `drop_on_conflict` (the legacy venv) are deleted instead when `new`
    already has its own copy. `old` is removed once it is empty.
    """
    if not old.exists():
        return
    if not new.exists():
        log.append(f"  move: {old}  →  {new}")
        if not dry_run:
            _move_path(old, new)
        return
    if not (old.is_dir() and new.is_dir()) or old.is_symlink() or new.is_symlink():
        log.append(f"  skip (target exists): {new}")
        return
    log.append(f"  merge: {old}  →  {new}")
    for child in sorted(old.iterdir()):
        dst = new / child.name
        if not dst.exists() and not dst.is_symlink():
            log.append(f"  move: {child}  →  {dst}")
            if not dry_run:
                _move_path(child, dst)
        elif child.name in drop_on_conflict:
            log.append(f"  remove stale legacy {child.name}: {child}")
            if not dry_run:
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        elif child.is_dir() and dst.is_dir() and not child.is_symlink():
            _move_dir(child, dst, dry_run=dry_run, log=log)
        else:
            log.append(f"  keep (target exists, left in place): {child}")
    if not dry_run:
        try:
            old.rmdir()
        except OSError:
            log.append(f"  {old} still holds files that conflict with {new}; left in place")


def _rewrite_paths_in_value(value: Any, *, key: str | None = None) -> Any:
    """Recursively rewrite legacy strings + numeric defaults in any TOML value.

    String substitutions:
      • '<OLD_NAME>/models/' → '<NEW_NAME>/models/'   (path migration)
      • exact string LEGACY_SINK_NAME → NEW_SINK_NAME (sink rename, v0.6.4)

    Numeric bumps (only fire when `key` matches a known stale-default name):
      • output_latency_ms < 300 → 300   (v0.6.7 - needed to absorb the
        engine's 250 ms-chunk writer cadence without ring-buffer
        underruns at the playback backend. Original v0.5.2 bump was
        30 → 100; v0.6.7 bumps further to 300 because the playback
        backend changed from pw-cat to pacat (see EngineConfig
        commentary), and pacat needs more headroom.)

    Tuples/lists/dicts walked. Other types passed through.
    """
    path_needle = f"{OLD_NAME}/models/"
    path_replacement = f"{NEW_NAME}/models/"
    if isinstance(value, str):
        out = value
        if path_needle in out:
            out = out.replace(path_needle, path_replacement)
        if out == LEGACY_SINK_NAME:
            out = NEW_SINK_NAME
        return out
    if isinstance(value, bool):
        # bool is a subclass of int - must short-circuit before the int branch.
        return value
    if isinstance(value, int) and key == "output_latency_ms" and value < 300:
        return 300
    if isinstance(value, list):
        return [_rewrite_paths_in_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _rewrite_paths_in_value(v, key=k) for k, v in value.items()}
    return value


def _toml_dump(data: dict[str, Any], path: Path) -> None:
    """Write `data` to a new file at `path`, created 0600 from the start.

    install.sh runs us with the venv python after the dependencies are
    installed, so tomli_w is available; it quotes keys such as profile
    names with spaces or dots, which a hand-rolled emitter got wrong.
    """
    import tomli_w

    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        tomli_w.dump(data, f)
        f.flush()
        os.fsync(f.fileno())


def _rewrite_config_toml(config_path: Path, *, dry_run: bool, log: list[str]) -> None:
    if not config_path.exists():
        return
    with open(config_path, "rb") as f:
        data = tomllib.load(f)
    rewritten = _rewrite_paths_in_value(data)
    if rewritten == data:
        log.append("  config.toml: no path rewrites needed")
        return
    log.append(
        f"  config.toml: rewrote legacy values "
        f"({OLD_NAME}/models/ → {NEW_NAME}/models/, "
        f"{LEGACY_SINK_NAME} → {NEW_SINK_NAME}, "
        "output_latency_ms < 300 → 300)"
    )
    if dry_run:
        return
    # Atomic write via .tmp + rename so a crash mid-write can't corrupt config.
    tmp = config_path.with_suffix(config_path.suffix + ".tmp")
    # B65 / sec-003: the file is created 0600 (never world-readable, even
    # briefly). A stale .tmp from a crashed run is removed first so the
    # exclusive create can't pick up its mode or contents.
    tmp.unlink(missing_ok=True)
    _toml_dump(rewritten, tmp)
    os.replace(tmp, config_path)


def _systemctl(args: list[str], *, dry_run: bool, log: list[str]) -> int:
    log.append(f"  systemctl --user {' '.join(args)}")
    if dry_run:
        return 0
    res = subprocess.run(
        ["systemctl", "--user", *args],
        capture_output=True,
        text=True,
        timeout=10,
    )
    return res.returncode


def _stop_old_systemd_unit(home: Path, *, dry_run: bool, log: list[str]) -> None:
    """Stop, disable, remove the old unit if present. Always best-effort -
    a failure shouldn't abort the migration."""
    unit_name = f"{OLD_NAME}-mic.service"
    unit_path = home / ".config" / "systemd" / "user" / unit_name
    if not unit_path.exists():
        log.append(f"  no old systemd unit at {unit_path} - skip")
        return
    _systemctl(["stop", unit_name], dry_run=dry_run, log=log)
    _systemctl(["disable", unit_name], dry_run=dry_run, log=log)
    log.append(f"  remove: {unit_path}")
    if not dry_run:
        unit_path.unlink(missing_ok=True)
        _systemctl(["daemon-reload"], dry_run=dry_run, log=log)


def migrate(home: Path | None = None, *, dry_run: bool = False) -> tuple[bool, list[str]]:
    """Run the migration. Returns (changed, log_lines).

    `changed = True` if anything moved. Used by install.sh to decide whether
    to print a migration summary.
    """
    h = home or DEFAULT_HOME
    log: list[str] = []
    log.append(f"[migrate] {OLD_NAME} → {NEW_NAME}  (dry_run={dry_run})")

    old_share = h / ".local" / "share" / OLD_NAME
    new_share = h / ".local" / "share" / NEW_NAME
    old_config = h / ".config" / OLD_NAME
    new_config = h / ".config" / NEW_NAME
    old_cache = h / ".cache" / OLD_NAME
    new_cache = h / ".cache" / NEW_NAME

    fresh_install = not (old_share.exists() or old_config.exists() or old_cache.exists())
    if fresh_install:
        log.append("  no old install detected - fresh install path, nothing to do")
        return False, log

    # Only the legacy config.toml gets the rewrite below. A config.toml that
    # already lives in the woys dir is woys' own, and rewriting it on every
    # rerun would force its output_latency_ms back up to 300.
    legacy_config_moves = (old_config / "config.toml").is_file() and not (
        new_config / "config.toml"
    ).exists()

    # 1) Stop the old systemd unit BEFORE moving anything (so the running
    #    service can't race a half-renamed dir).
    _stop_old_systemd_unit(h, dry_run=dry_run, log=log)

    # 2) Move the three user-data dirs. install.sh has already built a
    #    fresh venv in the share dir, so the legacy one is dropped.
    _move_dir(
        old_share,
        new_share,
        dry_run=dry_run,
        log=log,
        drop_on_conflict=frozenset({"venv"}),
    )
    _move_dir(old_config, new_config, dry_run=dry_run, log=log)
    _move_dir(old_cache, new_cache, dry_run=dry_run, log=log)

    # 3) Rewrite model paths in the (now relocated) config.toml so they
    #    point at .../woys/models/ instead of .../vcclient-cachy/models/.
    if legacy_config_moves:
        _rewrite_config_toml(
            (old_config if dry_run else new_config) / "config.toml", dry_run=dry_run, log=log
        )
    else:
        log.append("  config.toml: none moved in this run, left as is")

    log.append("[migrate] complete")
    return True, log


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Migrate vcclient-cachy install to woys (v0.6.0).",
    )
    parser.add_argument("--dry-run", action="store_true", help="report only, change nothing")
    args = parser.parse_args()
    changed, log = migrate(dry_run=args.dry_run)
    for line in log:
        print(line)
    # No-op on a fresh install is success - a fresh box with no
    # vcclient-cachy state is the expected case for new users.
    _ = changed
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
