"""Named profiles - full-state snapshots of pitch / model / chunk / monitor.

Stored in `~/.config/woys/config.toml` under a `[profiles.<name>]`
section. The top-level keys mirror `AppConfig`; profile sections have the
same key namespace, so applying a profile == copy fields into the top
level + write.

CLI:
  woys profile save <name>
  woys profile use <name>
  woys profile list
  woys profile delete <name>

The TUI bindings (Phase 4 polish) cycle through the saved profiles.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


def _ensure_audio_path() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


# B9 / arch-005 - derive _PROFILE_FIELDS from the single source of truth
# in `audio.engine.USER_VISIBLE_ENGINE_FIELDS`. Adding a user-visible
# EngineConfig field there now automatically makes it survive a profile
# save/use cycle. Pre-v0.8.0, this was a hand-maintained tuple that lost
# `input_gate_dbfs`, `prefer_pw_cat`, etc. - exactly the rc4 drift class.
_ensure_audio_path()
from audio.engine import USER_VISIBLE_ENGINE_FIELDS as _ENGINE_FIELDS  # noqa: E402

# `rvc_model` is a profile field too, but it's stored as a string at this
# layer (Path on EngineConfig). Prepend explicitly.
_PROFILE_FIELDS: tuple[str, ...] = ("rvc_model", *_ENGINE_FIELDS)


def _ensure_tui_path() -> None:
    """Alias kept for backwards compat - same as `_ensure_audio_path`."""
    _ensure_audio_path()


def _profiles_bag(cfg: Any) -> dict[str, dict[str, Any]]:
    """The on-disk profiles live under `cfg._extras["profiles"]` because
    AppConfig itself doesn't have a `profiles` field - `_extras` round-trips
    unknown keys (see `tui.config.load_config`)."""
    bag = cfg._extras.get("profiles", {})
    if not isinstance(bag, dict):
        return {}
    return bag


def list_profiles(cfg: Any) -> list[str]:
    return sorted(_profiles_bag(cfg).keys())


def save_profile(cfg: Any, name: str) -> None:
    """Snapshot the current `cfg`'s profile fields under the given name."""
    snapshot: dict[str, Any] = {}
    cfg_dict = asdict(cfg)
    for field_name in _PROFILE_FIELDS:
        if field_name in cfg_dict:
            snapshot[field_name] = cfg_dict[field_name]
    bag = dict(_profiles_bag(cfg))
    bag[name] = snapshot
    cfg._extras["profiles"] = bag


def apply_profile(cfg: Any, name: str) -> bool:
    """Copy the named profile's fields into the top-level config. Returns
    False if no such profile exists."""
    bag = _profiles_bag(cfg)
    if name not in bag:
        return False
    snap = bag[name]
    for field_name in _PROFILE_FIELDS:
        if field_name in snap:
            setattr(cfg, field_name, snap[field_name])
    return True


def delete_profile(cfg: Any, name: str) -> bool:
    bag = dict(_profiles_bag(cfg))
    if name not in bag:
        return False
    del bag[name]
    cfg._extras["profiles"] = bag
    return True


def cycle_profile(cfg: Any, current: str | None) -> str | None:
    """Pick the next profile in the saved order; wraps around. Returns the
    new profile name (or None if no profiles exist)."""
    names = list_profiles(cfg)
    if not names:
        return None
    if current is None or current not in names:
        return names[0]
    idx = (names.index(current) + 1) % len(names)
    return names[idx]


# CLI handlers ------------------------------------------------------------------


def cli_profile_save(name: str) -> int:
    _ensure_tui_path()
    from tui.config import load_config, save_config

    cfg = load_config()
    save_profile(cfg, name)
    save_config(cfg)
    print(f"[profile] saved snapshot: {name!r}")
    return 0


def cli_profile_use(name: str) -> int:
    _ensure_tui_path()
    from tui.config import load_config, save_config

    cfg = load_config()
    if not apply_profile(cfg, name):
        print(f"[profile] no such profile: {name!r}", file=sys.stderr)
        print("  available: " + ", ".join(list_profiles(cfg)))
        return 1
    save_config(cfg)
    print(f"[profile] active profile -> {name}")
    print("  restart the engine for the change to take effect")
    return 0


def cli_profile_list() -> int:
    _ensure_tui_path()
    from tui.config import load_config

    cfg = load_config()
    names = list_profiles(cfg)
    if not names:
        print("no saved profiles. Use `woys profile save <name>` to create one.")
        return 0
    print(f"{'name':24s}  {'pitch':>6s}  {'chunk_s':>8s}  rvc_model")
    bag = _profiles_bag(cfg)
    for n in names:
        snap = bag[n]
        pitch = snap.get("f0_up_key", 0)
        chunk = snap.get("chunk_seconds", "?")
        model = snap.get("rvc_model", "")
        model_short = Path(model).name if model else "(default)"
        print(f"{n:24s}  {pitch:>+6d}  {chunk:>8}  {model_short}")
    return 0


def cli_profile_delete(name: str) -> int:
    _ensure_tui_path()
    from tui.config import load_config, save_config

    cfg = load_config()
    if not delete_profile(cfg, name):
        print(f"[profile] no such profile: {name!r}", file=sys.stderr)
        return 1
    save_config(cfg)
    print(f"[profile] deleted {name!r}")
    return 0
