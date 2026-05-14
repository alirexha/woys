"""User config persisted at `~/.config/woys/config.toml`.

Round-trips (load → save → load) are stable: any unknown keys present in the
on-disk file pass through untouched.

v0.6.8 - `AppConfig` defaults forward from `EngineConfig`. EngineConfig is
the canonical source of truth for runtime parameters; AppConfig is the
user-facing config-file shape. Without forwarding, the two drift over
releases (LESSONS §17 - v0.6.7 shipped with `output_latency_ms = 100`
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

from audio.engine import USER_VISIBLE_ENGINE_FIELDS as _USER_VISIBLE_ENGINE_FIELDS
from audio.engine import EngineConfig as _EngineConfig

CONFIG_DIR = Path.home() / ".config" / "woys"
CONFIG_FILE = CONFIG_DIR / "config.toml"

# Single shared instance - evaluated at module import. AppConfig's
# field defaults reference attributes of this instance so a future
# default-bump in `EngineConfig` propagates here automatically.
_E = _EngineConfig()


@dataclass
class AppConfig:
    rvc_model: str = ""  # absolute path, "" = use engine default
    # Runtime parameters - defaults forwarded from EngineConfig.
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
    # v0.7.0-rc4 - added to AppConfig's forwarded set. Pre-rc4 these
    # lived only on EngineConfig, so user overrides in `config.toml`
    # were silently ignored (the audit `docs/16-audit/synthesis.md`
    # confirmed alireza's `input_gate_dbfs = -200.0` never made it
    # to the engine - every prior rc ran the dataclass default).
    # `prefer_pw_cat` had the same drift since rc1.
    input_gate_dbfs: float = _E.input_gate_dbfs
    input_gate_hysteresis_ms: float = _E.input_gate_hysteresis_ms
    prefer_pw_cat: bool = _E.prefer_pw_cat
    prefer_native_pw: bool = _E.prefer_native_pw
    prefer_native_pw_buffer_ms: int = _E.prefer_native_pw_buffer_ms
    # v0.10.0-rc3 - GPU keep-alive. Default off (A/B test surface).
    gpu_keepalive_enabled: bool = _E.gpu_keepalive_enabled
    gpu_keepalive_interval_ms: int = _E.gpu_keepalive_interval_ms
    gpu_keepalive_input_len: int = _E.gpu_keepalive_input_len
    # v0.11.0 - GPU clock lock + torch separate-stream keepalive.
    gpu_anti_jitter_mode: str = _E.gpu_anti_jitter_mode
    gpu_clock_lock_enabled: bool = _E.gpu_clock_lock_enabled
    gpu_clock_lock_floor_mhz: int = _E.gpu_clock_lock_floor_mhz
    gpu_clock_lock_ceiling_mhz: int = _E.gpu_clock_lock_ceiling_mhz
    gpu_clock_lock_floor_offset_mhz: int = _E.gpu_clock_lock_floor_offset_mhz
    gpu_keepalive_torch_stream: bool = _E.gpu_keepalive_torch_stream
    gpu_keepalive_torch_interval_ms: int = _E.gpu_keepalive_torch_interval_ms
    # TUI / app-only settings (not in EngineConfig).
    autostart_engine: bool = False
    enable_dbus: bool = True  # reserved for future D-Bus wiring (currently unused)
    enable_evdev_hotkey: bool = False
    evdev_hotkey: str = "ctrl+alt+v"  # only meaningful when enable_evdev_hotkey=True

    # Pass-through bag for unknown keys; kept on save so user-added fields survive.
    _extras: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        # Stamp the schema version on every fresh AppConfig so round-trips
        # match. The migration in load_config() bumps it on legacy files.
        self._extras.setdefault("config_schema_version", 10)


def app_config_to_engine_config(cfg: AppConfig, *, rvc_model: Path | None = None) -> _EngineConfig:
    """The single AppConfig -> EngineConfig forwarding path.

    review F-merged-008 / F-01-04: this replaces three hand-written,
    byte-drifting `EngineConfig(...)` blocks (`woys run`, `woys diag`,
    `woys engine`). It iterates `USER_VISIBLE_ENGINE_FIELDS`, so a new
    user-tunable field reaches every entry point by being added to that
    one tuple -- nothing else to edit.

    Before this, the two `cli.py` blocks silently omitted `mic_rate` /
    `sink_rate` while the TUI forwarded them, so `woys diag` / `woys engine`
    ran 48 kHz defaults on non-48k hardware (F-01-04).

    `rvc_model` is passed separately: it is Path-typed and resolved by the
    caller (an empty / missing path falls back to the engine's default).
    """
    engine_cfg = _EngineConfig()
    for name in _USER_VISIBLE_ENGINE_FIELDS:
        setattr(engine_cfg, name, getattr(cfg, name))
    if rvc_model is not None:
        engine_cfg.rvc_model = rvc_model
    return engine_cfg


def load_config(path: Path = CONFIG_FILE) -> AppConfig:
    """Load config from disk; on first run, write defaults to $path before returning.

    Writing defaults on first run gives the user a discoverable place to twiddle
    options (sink_name, monitor, chunk_seconds, etc.) without having to read the
    source code first.

    v0.6.8 - malformed TOML or unreadable file no longer crashes the app.
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
            f"[woys] {path} is malformed TOML - using in-memory defaults instead.\n"
            f"       parse error: {e}\n"
            f"       (the file was NOT touched; fix the syntax and re-launch)",
            file=sys.stderr,
        )
        return AppConfig()
    except OSError as e:
        print(
            f"[woys] cannot read {path} ({type(e).__name__}: {e}) - "
            f"using in-memory defaults instead.",
            file=sys.stderr,
        )
        return AppConfig()
    known = {f.name for f in AppConfig.__dataclass_fields__.values()} - {"_extras"}
    fields_in: dict[str, Any] = {k: raw[k] for k in known if k in raw}
    extras = {k: v for k, v in raw.items() if k not in known}
    # v0.7.0 - bump stale v0.6.x defaults so existing users get the latency
    # win. Keyed off `config_schema_version`: absent / < 7 means the file
    # was last written by v0.6.x or earlier. We only touch fields whose
    # value matches the previous version's *default* - explicit user
    # overrides are preserved. The bumped fields are then written back.
    schema = int(extras.pop("config_schema_version", 0) or 0)
    migrated = False
    # ---- schema 0 → 7 - the original v0.7.0-rc1 latency-defaults bump.
    if schema < 7:
        # chunk_seconds 0.25 → engine default (mic-input wait, biggest lever).
        if fields_in.get("chunk_seconds") == 0.25:
            fields_in["chunk_seconds"] = _E.chunk_seconds
            migrated = True
        # sola_search_ms 4.0 → 6.0 (v0.6.9 SOLA tuning that never propagated
        # into existing configs because of the v0.6.8 forwarding gap).
        if fields_in.get("sola_search_ms") == 4.0:
            fields_in["sola_search_ms"] = _E.sola_search_ms  # 6.0
            migrated = True
        # output_latency_ms 300 → engine default. The actual value is
        # bumped again under schema 7→8 below; this is just the first
        # leg of the migration so a user coming straight from v0.6.x
        # with schema=0 on disk lands at the current default after the
        # combined run, not on rc1's now-deprecated 80.
        if fields_in.get("output_latency_ms") == 300:
            fields_in["output_latency_ms"] = _E.output_latency_ms
            migrated = True
        profiles = extras.get("profiles")
        if isinstance(profiles, dict):
            for pdata in profiles.values():
                if not isinstance(pdata, dict):
                    continue
                if pdata.get("chunk_seconds") == 0.25:
                    pdata["chunk_seconds"] = _E.chunk_seconds
                    migrated = True
                if pdata.get("output_latency_ms") == 300:
                    pdata["output_latency_ms"] = _E.output_latency_ms
                    migrated = True
                if pdata.get("sola_search_ms") == 4.0:
                    pdata["sola_search_ms"] = _E.sola_search_ms
                    migrated = True
    # ---- schema 7 → 8 - v0.7.0-rc1's 80 ms output_latency was empirically
    # too aggressive (user reported audible cut increase). rc2 bumped the
    # default to 220 ms; users who landed on 80 via the rc1 migration
    # get pulled forward. Note: the rc3 leg below cascades them on to 280.
    if schema < 8:
        if fields_in.get("output_latency_ms") == 80:
            fields_in["output_latency_ms"] = _E.output_latency_ms  # rc3: 280
            migrated = True
        profiles = extras.get("profiles")
        if isinstance(profiles, dict):
            for pdata in profiles.values():
                if not isinstance(pdata, dict):
                    continue
                if pdata.get("output_latency_ms") == 80:
                    pdata["output_latency_ms"] = _E.output_latency_ms
                    migrated = True
    # ---- schema 8 → 9 - v0.7.0-rc2's 220 ms was still audibly cutting in
    # real-world Telegram VoIP testing. rc3 bumps to 280 ms (the last rung -
    # 20 ms under the known-clean v0.6.x 300 ms default). Users who landed
    # on 220 via the rc2 default get pulled forward; explicit non-default
    # values (e.g. 250) are left alone.
    if schema < 9:
        if fields_in.get("output_latency_ms") == 220:
            fields_in["output_latency_ms"] = _E.output_latency_ms  # 280
            migrated = True
        profiles = extras.get("profiles")
        if isinstance(profiles, dict):
            for pdata in profiles.values():
                if not isinstance(pdata, dict):
                    continue
                if pdata.get("output_latency_ms") == 220:
                    pdata["output_latency_ms"] = _E.output_latency_ms
                    migrated = True
    # ---- schema 9 → 10 - v0.7.0-rc4 audit (`docs/16-audit/synthesis.md`)
    # found rc3's persistent cuts came from sources other than
    # `output_latency_ms`. Two defaults bump and two new fields land.
    #
    # `input_gate_dbfs` -55 → -75: the rc1+ default fired on intra-
    # speech RMS dips, emitting full chunks of zeros that bypassed
    # every downstream buffer. rc4 lowers the threshold to well
    # below typical room ambient.
    #
    # `prefer_pw_cat` True → False: v0.6.7's documented per-quantum
    # zero-gap pattern matches lens 08's waveform evidence (sample-
    # exact zeros, ~40 ms quantized) better than pacat's underrun
    # pattern. rc1's "smaller chunks dodge the race" reasoning was
    # not empirically backed.
    #
    # `input_gate_hysteresis_ms` and `input_gate_dbfs` were also added
    # to AppConfig's forwarded field set this release; the migration
    # that follows is the first time `prefer_pw_cat` lands in user
    # configs at all (pre-rc4 it had no on-disk surface).
    if schema < 10:
        if fields_in.get("input_gate_dbfs") == -55.0:
            fields_in["input_gate_dbfs"] = _E.input_gate_dbfs  # -75.0
            migrated = True
        if fields_in.get("prefer_pw_cat") is True:
            fields_in["prefer_pw_cat"] = _E.prefer_pw_cat  # False
            migrated = True
        profiles = extras.get("profiles")
        if isinstance(profiles, dict):
            for pdata in profiles.values():
                if not isinstance(pdata, dict):
                    continue
                if pdata.get("input_gate_dbfs") == -55.0:
                    pdata["input_gate_dbfs"] = _E.input_gate_dbfs
                    migrated = True
                if pdata.get("prefer_pw_cat") is True:
                    pdata["prefer_pw_cat"] = _E.prefer_pw_cat
                    migrated = True
    cfg = AppConfig(**fields_in, _extras=extras)
    cfg._extras["config_schema_version"] = 10
    if migrated:
        with contextlib.suppress(OSError):
            save_config(cfg, path)
    return cfg


def save_config(cfg: AppConfig, path: Path = CONFIG_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {k: v for k, v in asdict(cfg).items() if not k.startswith("_")}
    data.update(cfg._extras)
    # Write atomically via .tmp + rename so a crash mid-write can't corrupt
    # config. v0.6.8 - chmod 0600 (was inheriting umask 0644). Config can
    # contain user paths and tuning that other local users have no business
    # reading.
    # v0.14.0 (Lens 6 / Lens 17 / C123 + C268): create file with mode 0600
    # atomically via os.open(O_WRONLY|O_CREAT|O_EXCL, 0o600). Pre-v0.14.0
    # `open(tmp, "wb")` created the file with default umask (typically
    # 0644) and then `os.chmod(0o600)` ran AFTER -- a race window where
    # another local user could read tunings + model paths before the
    # restrictive mode landed. O_EXCL also rejects pre-existing tmp files
    # (a stale .tmp from a crashed write would be an attack vector for
    # symlink-replace; O_EXCL refuses to follow). Plus fsync before
    # replace per C268 so power-loss-during-write doesn't corrupt config.
    tmp = path.with_suffix(path.suffix + ".tmp")
    # Clean up any stale tmp file from a crashed prior write (own UID
    # only, since the parent dir is XDG_CONFIG_HOME).
    if tmp.exists():
        tmp.unlink()
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as f:
            tomli_w.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise
    os.replace(tmp, path)
