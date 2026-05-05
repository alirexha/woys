"""v0.6.0 — migrate an existing vcclient-cachy install to woys.

Run by `install.sh` before installing the new code, so the user's models +
config + systemd unit move to the new layout in one atomic step. Safe to
re-run (idempotent) and safe to invoke on a fresh install (no-op).

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
    Install of the new unit (`woys-mic.service`) is left to install.sh —
    that's where the new file lives.

PipeWire:
    v0.6.0 to v0.6.4: the user-facing SOURCE name (`vcclient-mic`) was
    deliberately preserved across the rename so Discord / CS2 / Telegram
    didn't need re-configuration.
    v0.6.5: that compromise was retired — the source is now `woys-mic`
    too. Apps need to re-select their input device once. The remap-source
    rename is handled by `pipewire.VirtualMic.ensure()` (it unloads any
    legacy `vcclient-mic` module before loading the new one), not by this
    migrator.

    The internal SINK name changed in v0.6.0 (`VCClientCachySink` →
    `WoysSink`). This migrator rewrites the `sink_name` key in
    `config.toml` accordingly so the engine targets the sink that
    v0.6.0+ actually loads. v0.6.4 fix — without this rewrite,
    `pw-cat --target=VCClientCachySink` silently falls back to the
    default sink (laptop speakers) since the legacy sink no longer
    exists. See `docs/10-monitor-leak-diag.md`.

Usage:
    python3 scripts/migrate_to_woys.py [--dry-run]

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

# v0.6.4 — the v0.6.0 rename also changed the internal PipeWire sink
# name. Configs from v0.5.x carry the legacy string and must be rewritten
# or the engine routes playback to the default sink (laptop speakers).
LEGACY_SINK_NAME = "VCClientCachySink"
NEW_SINK_NAME = "WoysSink"

# Anchor points relative to $HOME — overridable for tests.
DEFAULT_HOME = Path.home()


def _move_dir(old: Path, new: Path, *, dry_run: bool, log: list[str]) -> None:
    """Atomic rename if same filesystem; tree-copy fallback otherwise. No-op
    if `new` already exists (treated as 'already migrated' on idempotent
    re-run).
    """
    if not old.exists():
        return
    if new.exists():
        log.append(f"  skip move (target exists): {new}")
        return
    log.append(f"  move: {old}  →  {new}")
    if dry_run:
        return
    new.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.rename(old, new)  # atomic when on the same filesystem
    except OSError:
        # Cross-FS fallback. On a personal dev box this should never
        # happen ($HOME is one mount), but install.sh shouldn't crash if
        # the user's $HOME spans two mounts.
        shutil.copytree(old, new)
        shutil.rmtree(old)


def _rewrite_paths_in_value(value: Any) -> Any:
    """Recursively rewrite legacy strings in any TOML value.

    Two substitutions:
      • '<OLD_NAME>/models/' → '<NEW_NAME>/models/'   (path migration)
      • exact string LEGACY_SINK_NAME → NEW_SINK_NAME (sink rename, v0.6.4)

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
    if isinstance(value, list):
        return [_rewrite_paths_in_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _rewrite_paths_in_value(v) for k, v in value.items()}
    return value


def _toml_dump(data: dict[str, Any], path: Path) -> None:
    """Hand-rolled minimal TOML emitter — sufficient for config.toml's flat
    [section]-based layout. We avoid pulling in `tomli_w` here so the
    migrator runs on a stock Python 3.11 (the install.sh runs us BEFORE
    creating the venv).
    """
    lines: list[str] = []
    top_level = {k: v for k, v in data.items() if not isinstance(v, dict)}
    sections = {k: v for k, v in data.items() if isinstance(v, dict)}

    def _fmt(v: Any) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return repr(v)
        if isinstance(v, str):
            escaped = v.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        if isinstance(v, list):
            return "[" + ", ".join(_fmt(item) for item in v) + "]"
        raise TypeError(f"unsupported TOML value type: {type(v).__name__}")

    for k, v in top_level.items():
        lines.append(f"{k} = {_fmt(v)}")

    for section_name, section in sections.items():
        lines.append("")
        lines.append(f"[{section_name}]")
        for k, v in section.items():
            if isinstance(v, dict):
                # Nested section, e.g. [profiles.default]
                lines.append(f"\n[{section_name}.{k}]")
                for sub_k, sub_v in v.items():
                    lines.append(f"{sub_k} = {_fmt(sub_v)}")
            else:
                lines.append(f"{k} = {_fmt(v)}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        f"  config.toml: rewrote legacy strings "
        f"({OLD_NAME}/models/ → {NEW_NAME}/models/, "
        f"{LEGACY_SINK_NAME} → {NEW_SINK_NAME})"
    )
    if dry_run:
        return
    # Atomic write via .tmp + rename so a crash mid-write can't corrupt config.
    tmp = config_path.with_suffix(config_path.suffix + ".tmp")
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
    """Stop, disable, remove the old unit if present. Always best-effort —
    a failure shouldn't abort the migration."""
    unit_name = f"{OLD_NAME}-mic.service"
    unit_path = home / ".config" / "systemd" / "user" / unit_name
    if not unit_path.exists():
        log.append(f"  no old systemd unit at {unit_path} — skip")
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
        log.append("  no old install detected — fresh install path, nothing to do")
        return False, log

    # 1) Stop the old systemd unit BEFORE moving anything (so the running
    #    service can't race a half-renamed dir).
    _stop_old_systemd_unit(h, dry_run=dry_run, log=log)

    # 2) Move the three user-data dirs.
    _move_dir(old_share, new_share, dry_run=dry_run, log=log)
    _move_dir(old_config, new_config, dry_run=dry_run, log=log)
    _move_dir(old_cache, new_cache, dry_run=dry_run, log=log)

    # 3) Rewrite model paths in the (now relocated) config.toml so they
    #    point at .../woys/models/ instead of .../vcclient-cachy/models/.
    _rewrite_config_toml(new_config / "config.toml", dry_run=dry_run, log=log)

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
    # No-op on a fresh install is success — a fresh box with no
    # vcclient-cachy state is the expected case for new users.
    _ = changed
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
