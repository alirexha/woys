"""User config persisted at `~/.config/vcclient-cachy/config.toml`.

Round-trips (load → save → load) are stable: any unknown keys present in the
on-disk file pass through untouched.
"""

from __future__ import annotations

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
    chunk_seconds: float = 0.5
    mic_rate: int = 48_000
    sink_rate: int = 48_000
    autostart_engine: bool = False
    enable_dbus: bool = True
    enable_evdev_hotkey: bool = False
    evdev_hotkey: str = "ctrl+alt+v"  # only meaningful when enable_evdev_hotkey=True

    # Pass-through bag for unknown keys; kept on save so user-added fields survive.
    _extras: dict[str, Any] = field(default_factory=dict, repr=False)


def load_config(path: Path = CONFIG_FILE) -> AppConfig:
    if not path.exists():
        return AppConfig()
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
