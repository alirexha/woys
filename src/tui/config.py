"""User config persisted at `~/.config/woys/config.toml`.

Round-trips (load → save → load) are stable: any unknown keys present in the
on-disk file pass through untouched.

v0.6.8 — `AppConfig` defaults forward from `EngineConfig`. EngineConfig is
the canonical source of truth for runtime parameters; AppConfig is the
user-facing config-file shape. Without forwarding, the two drift over
releases (LESSONS §17 — v0.6.7 shipped with `output_latency_ms = 100`
in `AppConfig` while `EngineConfig` had been bumped to 300, so fresh
installs reproduced the v0.6.7 micro-cut bug we'd just fixed).
"""

from __future__ import annotations

import contextlib
import os
import sys
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import tomli_w

from audio.engine import EngineConfig as _EngineConfig

CONFIG_DIR = Path.home() / ".config" / "woys"
CONFIG_FILE = CONFIG_DIR / "config.toml"

# Single shared instance — evaluated at module import. AppConfig's
# field defaults reference attributes of this instance so a future
# default-bump in `EngineConfig` propagates here automatically.
_E = _EngineConfig()


@dataclass
class AppConfig:
    rvc_model: str = ""  # absolute path, "" = use engine default
    # Runtime parameters — defaults forwarded from EngineConfig.
    f0_up_key: int = _E.f0_up_key
    sid: int = _E.sid
    chunk_seconds: float = _E.chunk_seconds
    mic_rate: int = _E.mic_rate
    sink_rate: int = _E.sink_rate
    sink_name: str = _E.sink_name
    monitor: bool = _E.monitor
    output_latency_ms: int = _E.output_latency_ms
    output_process_time_ms: int = _E.output_process_time_ms
    embedder: str = _E.embedder
    sola_enabled: bool = _E.sola_enabled
    sola_crossfade_ms: float = _E.sola_crossfade_ms
    sola_search_ms: float = _E.sola_search_ms
    sola_context_ms: float = _E.sola_context_ms
    input_gain_db: float = _E.input_gain_db
    # TUI / app-only settings (not in EngineConfig).
    autostart_engine: bool = False
    enable_dbus: bool = True  # reserved for future D-Bus wiring (currently unused)
    enable_evdev_hotkey: bool = False
    evdev_hotkey: str = "ctrl+alt+v"  # only meaningful when enable_evdev_hotkey=True

    # Pass-through bag for unknown keys; kept on save so user-added fields survive.
    _extras: dict[str, Any] = field(default_factory=dict, repr=False)


def load_config(path: Path = CONFIG_FILE) -> AppConfig:
    """Load config from disk; on first run, write defaults to $path before returning.

    Writing defaults on first run gives the user a discoverable place to twiddle
    options (sink_name, monitor, chunk_seconds, etc.) without having to read the
    source code first.

    v0.6.8 — malformed TOML or unreadable file no longer crashes the app.
    A clear message is printed to stderr and an in-memory `AppConfig()`
    with EngineConfig-forwarded defaults is returned instead. The bad
    file is left in place for the user to inspect / fix.
    """
    if not path.exists():
        cfg = AppConfig()
        # Read-only home / unwritable XDG dir → fall back to in-memory defaults.
        with contextlib.suppress(OSError):
            save_config(cfg, path)
        return cfg
    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        print(
            f"[woys] {path} is malformed TOML — using in-memory defaults instead.\n"
            f"       parse error: {e}\n"
            f"       (the file was NOT touched; fix the syntax and re-launch)",
            file=sys.stderr,
        )
        return AppConfig()
    except OSError as e:
        print(
            f"[woys] cannot read {path} ({type(e).__name__}: {e}) — "
            f"using in-memory defaults instead.",
            file=sys.stderr,
        )
        return AppConfig()
    known = {f.name for f in AppConfig.__dataclass_fields__.values()} - {"_extras"}
    fields_in: dict[str, Any] = {k: raw[k] for k in known if k in raw}
    extras = {k: v for k, v in raw.items() if k not in known}
    return AppConfig(**fields_in, _extras=extras)


def save_config(cfg: AppConfig, path: Path = CONFIG_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {k: v for k, v in asdict(cfg).items() if not k.startswith("_")}
    data.update(cfg._extras)
    # Write atomically via .tmp + rename so a crash mid-write can't corrupt
    # config. v0.6.8 — chmod 0600 (was inheriting umask 0644). Config can
    # contain user paths and tuning that other local users have no business
    # reading.
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        tomli_w.dump(data, f)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
