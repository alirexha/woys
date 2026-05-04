"""User config persisted at `~/.config/vcclient-cachy/config.toml`.

Round-trips (load → save → load) are stable: any unknown keys present in the
on-disk file pass through untouched.
"""

from __future__ import annotations

import contextlib
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import tomli_w

CONFIG_DIR = Path.home() / ".config" / "vcclient-cachy"
CONFIG_FILE = CONFIG_DIR / "config.toml"


@dataclass
class AppConfig:
    rvc_model: str = ""  # absolute path, "" = use engine default
    f0_up_key: int = 0
    sid: int = 0
    # v0.2.0 default dropped from 0.5 → 0.1 thanks to SOLA crossfade. Existing
    # config.toml files keep their saved value (TOML round-trip preserves it).
    chunk_seconds: float = 0.1
    mic_rate: int = 48_000
    sink_rate: int = 48_000
    sink_name: str = "VCClientCachySink"  # explicit target — must match systemd unit
    monitor: bool = False  # play transformed audio to default output too (self-monitor)
    output_latency_ms: int = 30  # pacat playback latency request
    embedder: str = "onnx"  # "onnx" (default, no torch) or "fairseq" (heavy fallback)
    # SOLA streaming params (Phase B — see src/audio/sola.py).
    sola_enabled: bool = True
    sola_crossfade_ms: float = 50.0
    sola_search_ms: float = 4.0
    sola_context_ms: float = 100.0
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
    """
    if not path.exists():
        cfg = AppConfig()
        # Read-only home / unwritable XDG dir → fall back to in-memory defaults.
        with contextlib.suppress(OSError):
            save_config(cfg, path)
        return cfg
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    known = {f.name for f in AppConfig.__dataclass_fields__.values()} - {"_extras"}
    fields_in: dict[str, Any] = {k: raw[k] for k in known if k in raw}
    extras = {k: v for k, v in raw.items() if k not in known}
    return AppConfig(**fields_in, _extras=extras)


def save_config(cfg: AppConfig, path: Path = CONFIG_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {k: v for k, v in asdict(cfg).items() if not k.startswith("_")}
    data.update(cfg._extras)
    with open(path, "wb") as f:
        tomli_w.dump(data, f)
