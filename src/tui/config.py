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
    # were silently ignored (the audit internal notes
    # confirmed the maintainer's `input_gate_dbfs = -200.0` never made it
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
    # the `enable_dbus` field
    # is dropped. It was reserved for a D-Bus control path that was
    # rejected in favor of the Unix-domain control socket (see
    # the project notes "D-Bus is replaced by Unix-domain sockets"). A
    # dead config field carries the implication that the feature
    # might land -- it won't.
    enable_evdev_hotkey: bool = False
    evdev_hotkey: str = "ctrl+alt+v"  # only meaningful when enable_evdev_hotkey=True

    # Pass-through bag for unknown keys; kept on save so user-added fields survive.
    _extras: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        # Stamp the schema version on every fresh AppConfig so round-trips
        # match. The migration in load_config() bumps it on legacy files.
        self._extras.setdefault("config_schema_version", 10)
        # names of fields the user has explicitly
        # touched (TUI pitch keys, `woys pitch +2`, monitor toggle, ...).
        # Migration legs that match `value == old_default` skip fields
        # listed here -- so a user who deliberately set
        # `output_latency_ms = 300` (the v0.6.x default) keeps that value
        # after a schema bump. See `mark_override()` below.
        self._extras.setdefault("_user_overrides", [])


# --- ----------------------------------------------
# Per-field sane-range / type validator table. Pre-fix `AppConfig(**fields_
# in)` and `.vcprofile`'s `setattr` loop accepted any TOML value: a
# `chunk_seconds = "fast"` crashed deep in `_run_loop`; a shared
# `.vcprofile` with `chunk_seconds = -1` was a DoS. The contrast: the
# engine hard-fails a bad `embedder` (engine.py:1567-1569). This table
# extends that pattern to every user-tunable field with a numeric range.
#
# - `load_config` (TOML on disk, user-owned): on validation failure,
#   warn to stderr and reset the field to the AppConfig default. The
#   user keeps a working config; only the offending field is replaced.
# - `vcprofile.import_profile` (untrusted artifact, can be downloaded):
#   on validation failure, raise ValueError naming the field. Refuse
#   the import entirely -- this is the project's primary threat class.


@dataclass(frozen=True)
class _FieldSpec:
    """Per-field validation spec for AppConfig / vcprofile values."""

    py_type: type | tuple[type, ...]
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[Any, ...] | None = None


# `bool` <: `int` in Python's type hierarchy, so `isinstance(True, int)`
# is True. Every numeric spec below pairs `(int, ...)` or `(float, ...)`
# with an explicit `bool` reject in `validate_field()` so a hand-edited
# `f0_up_key = true` is refused, not silently coerced.
_FIELD_VALIDATORS: dict[str, _FieldSpec] = {
    "rvc_model": _FieldSpec(str),
    "sink_name": _FieldSpec(str),
    "embedder": _FieldSpec(str, choices=("onnx",)),
    "evdev_hotkey": _FieldSpec(str),
    "gpu_anti_jitter_mode": _FieldSpec(str, choices=("off", "clock_only", "stream_only", "both")),
    # Numeric -- pitch / speaker / rates.
    "f0_up_key": _FieldSpec(int, minimum=-24, maximum=24),
    "sid": _FieldSpec(int, minimum=0, maximum=1000),
    "mic_rate": _FieldSpec(
        int, choices=(8000, 16000, 22050, 24000, 32000, 44100, 48000, 88200, 96000)
    ),
    "sink_rate": _FieldSpec(
        int, choices=(8000, 16000, 22050, 24000, 32000, 44100, 48000, 88200, 96000)
    ),
    # Numeric -- timing / chunking.
    "chunk_seconds": _FieldSpec(float, minimum=0.01, maximum=2.0),
    "output_latency_ms": _FieldSpec(int, minimum=10, maximum=5000),
    "output_process_time_ms": _FieldSpec(int, minimum=1, maximum=5000),
    # Numeric -- SOLA.
    "sola_crossfade_ms": _FieldSpec(float, minimum=0.0, maximum=500.0),
    "sola_search_ms": _FieldSpec(float, minimum=0.0, maximum=500.0),
    "sola_context_ms": _FieldSpec(float, minimum=0.0, maximum=2000.0),
    # Numeric -- input gain / gate.
    "input_gain_db": _FieldSpec(float, minimum=-60.0, maximum=30.0),
    "input_gate_dbfs": _FieldSpec(float, minimum=-200.0, maximum=0.0),
    "input_gate_hysteresis_ms": _FieldSpec(float, minimum=0.0, maximum=5000.0),
    # Numeric -- PipeWire path.
    "prefer_native_pw_buffer_ms": _FieldSpec(int, minimum=0, maximum=5000),
    # Numeric -- GPU keepalive / clock lock.
    "gpu_keepalive_interval_ms": _FieldSpec(int, minimum=1, maximum=10_000),
    "gpu_keepalive_input_len": _FieldSpec(int, minimum=1, maximum=100_000),
    "gpu_keepalive_torch_interval_ms": _FieldSpec(int, minimum=1, maximum=10_000),
    "gpu_clock_lock_floor_mhz": _FieldSpec(int, minimum=0, maximum=5000),
    "gpu_clock_lock_ceiling_mhz": _FieldSpec(int, minimum=0, maximum=5000),
    "gpu_clock_lock_floor_offset_mhz": _FieldSpec(int, minimum=-5000, maximum=5000),
    # Booleans (the dataclass annotation says `bool`; reject everything else).
    "monitor": _FieldSpec(bool),
    "sola_enabled": _FieldSpec(bool),
    "prefer_pw_cat": _FieldSpec(bool),
    "prefer_native_pw": _FieldSpec(bool),
    "autostart_engine": _FieldSpec(bool),
    "enable_evdev_hotkey": _FieldSpec(bool),
    "gpu_keepalive_enabled": _FieldSpec(bool),
    "gpu_clock_lock_enabled": _FieldSpec(bool),
    "gpu_keepalive_torch_stream": _FieldSpec(bool),
}


def validate_field(name: str, value: Any) -> str | None:
    """Per-field validator. Returns `None` if `value` is acceptable,
    otherwise a human error message naming `name` and explaining what
    is wrong.

    Unknown fields (not in `_FIELD_VALIDATORS`) return `None` so the
    `_extras` pass-through bag is not gated -- a user adding their
    own keys to config.toml is allowed.
    """
    spec = _FIELD_VALIDATORS.get(name)
    if spec is None:
        return None  # unknown field: not gated (pass-through)

    py_type = spec.py_type
    types_tuple = py_type if isinstance(py_type, tuple) else (py_type,)

    # Python quirk: bool is a subclass of int. Reject bool wherever a
    # numeric is wanted (or specifically reject non-bool where bool is
    # wanted), so `f0_up_key = true` is not silently coerced to 1.
    if bool in types_tuple:
        if not isinstance(value, bool):
            return f"{name}: expected bool, got {type(value).__name__} ({value!r})"
    else:
        if isinstance(value, bool):
            return f"{name}: bool used where {types_tuple[0].__name__} expected (got {value!r})"
        if not isinstance(value, types_tuple):
            # Accept int where float is wanted (lossless widening); reject
            # everything else.
            if py_type is float and isinstance(value, int):
                value = float(value)
            else:
                want = "/".join(t.__name__ for t in types_tuple)
                return f"{name}: expected {want}, got {type(value).__name__} ({value!r})"

    if spec.choices is not None and value not in spec.choices:
        choices_display = list(spec.choices)
        return f"{name}: must be one of {choices_display}, got {value!r}"
    if spec.minimum is not None and value < spec.minimum:
        return f"{name}: must be >= {spec.minimum}, got {value!r}"
    if spec.maximum is not None and value > spec.maximum:
        return f"{name}: must be <= {spec.maximum}, got {value!r}"
    return None


def _validate_appconfig(cfg: AppConfig, *, source: str = "config") -> None:
    """Validate every gated field on `cfg`. On a violation, print a
    stderr warning naming the field + the source (config.toml or
    .vcprofile or test) and reset the field to the AppConfig default.

    This is the *user-owned* path: a bad value should not lose the
    whole config -- only the offending field is replaced.
    """
    defaults = AppConfig()
    for name in _FIELD_VALIDATORS:
        try:
            value = getattr(cfg, name)
        except AttributeError:
            continue
        err = validate_field(name, value)
        if err is None:
            continue
        default = getattr(defaults, name)
        print(
            f"[woys] {source}: invalid {err}; resetting to default {default!r}",
            file=sys.stderr,
        )
        setattr(cfg, name, default)


def mark_override(cfg: AppConfig, *keys: str) -> None:
    """Record that the user has explicitly touched these AppConfig fields.

    fields listed here are pinned across schema
    migrations -- the migration logic in `load_config()` will not bump
    them even if their current value matches an old default. Call this
    from every user-input mutation site (TUI keypress, CLI command,
    settings panel) before `save_config()`.

    Idempotent: a key already in the list is not duplicated. Unknown
    field names are accepted (no validation) so a future field rename
    doesn't crash old config files; the migration legs only consult
    the list by name, so a stale entry is a harmless no-op.
    """
    overrides = cfg._extras.setdefault("_user_overrides", [])
    if not isinstance(overrides, list):
        overrides = []
        cfg._extras["_user_overrides"] = overrides
    for key in keys:
        if key not in overrides:
            overrides.append(key)


def app_config_to_engine_config(cfg: AppConfig, *, rvc_model: Path | None = None) -> _EngineConfig:
    """The single AppConfig -> EngineConfig forwarding path.

    this replaces three hand-written,
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
        # surface the failure to stderr instead of
        # silently suppressing it. The user still gets a working in-memory
        # config; they just also get told why settings won't persist.
        try:
            save_config(cfg, path)
        except OSError as e:
            print(
                f"[woys] cannot write {path} ({type(e).__name__}: {e}) - "
                f"running with in-memory defaults; settings will not persist.",
                file=sys.stderr,
            )
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
    # was last written by v0.6.x or earlier. Each leg matches `value ==
    # old_default` and overwrites with the new default.
    #
    # the pre-fix legs unconditionally
    # clobbered every field whose value matched the old default -- so a
    # user who *deliberately* pinned `output_latency_ms = 300` lost that
    # value on upgrade. Each leg now consults `_user_overrides` (an
    # opt-in list of field names persisted in the TOML) and skips fields
    # the user has marked explicit. Legacy configs (pre-v0.14.x) ship
    # without the list, so the empty-default behavior is identical to
    # the pre-fix migration -- back-compat is intact.
    schema = int(extras.pop("config_schema_version", 0) or 0)
    user_overrides_raw = extras.pop("_user_overrides", []) or []
    user_overrides: set[str] = {str(k) for k in user_overrides_raw if isinstance(k, str)}
    migrated = False

    def _maybe_bump(name: str, old_default: Any, new_value: Any) -> None:
        nonlocal migrated
        if name in user_overrides:
            return
        if fields_in.get(name) == old_default:
            fields_in[name] = new_value
            migrated = True

    # ---- schema 0 → 7 - the original v0.7.0-rc1 latency-defaults bump.
    if schema < 7:
        # chunk_seconds 0.25 → engine default (mic-input wait, biggest lever).
        _maybe_bump("chunk_seconds", 0.25, _E.chunk_seconds)
        # sola_search_ms 4.0 → 6.0 (v0.6.9 SOLA tuning that never propagated
        # into existing configs because of the v0.6.8 forwarding gap).
        _maybe_bump("sola_search_ms", 4.0, _E.sola_search_ms)
        # output_latency_ms 300 → engine default. The actual value is
        # bumped again under schema 7→8 below; this is just the first
        # leg of the migration so a user coming straight from v0.6.x
        # with schema=0 on disk lands at the current default after the
        # combined run, not on rc1's now-deprecated 80.
        _maybe_bump("output_latency_ms", 300, _E.output_latency_ms)
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
        _maybe_bump("output_latency_ms", 80, _E.output_latency_ms)  # rc3: 280
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
        _maybe_bump("output_latency_ms", 220, _E.output_latency_ms)  # 280
        profiles = extras.get("profiles")
        if isinstance(profiles, dict):
            for pdata in profiles.values():
                if not isinstance(pdata, dict):
                    continue
                if pdata.get("output_latency_ms") == 220:
                    pdata["output_latency_ms"] = _E.output_latency_ms
                    migrated = True
    # ---- schema 9 → 10 - v0.7.0-rc4 audit
    # found rc3's persistent cuts came from sources other than
    # `output_latency_ms`. Two defaults bump and two new fields land.
    #
    # `input_gate_dbfs` -55 → -75: the rc1+ default fired on intra-
    # speech RMS dips, emitting full chunks of zeros that bypassed
    # every downstream buffer. rc4 lowers the threshold to well
    # below typical room ambient.
    #
    # `prefer_pw_cat` True → False: v0.6.7's documented per-quantum
    # zero-gap pattern matches area 08's waveform evidence (sample-
    # exact zeros, ~40 ms quantized) better than pacat's underrun
    # pattern. rc1's "smaller chunks dodge the race" reasoning was
    # not empirically backed.
    #
    # `input_gate_hysteresis_ms` and `input_gate_dbfs` were also added
    # to AppConfig's forwarded field set this release; the migration
    # that follows is the first time `prefer_pw_cat` lands in user
    # configs at all (pre-rc4 it had no on-disk surface).
    if schema < 10:
        _maybe_bump("input_gate_dbfs", -55.0, _E.input_gate_dbfs)  # -75.0
        # `prefer_pw_cat` `is True` was the pre-fix check; _maybe_bump
        # uses `==` which behaves identically for `bool` values
        # (True == True, but also 1 == True). Tighten by an explicit
        # type-and-value match so a hand-edited `prefer_pw_cat = 1`
        # is not silently bumped.
        if (
            "prefer_pw_cat" not in user_overrides
            and isinstance(fields_in.get("prefer_pw_cat"), bool)
            and fields_in.get("prefer_pw_cat") is True
        ):
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
    # re-stamp the override list so a round-trip
    # (load → save → load) preserves it. The `extras.pop` above removed
    # it from `extras` to keep migration logic clear; putting it back
    # here is the canonical write surface.
    cfg._extras["_user_overrides"] = sorted(user_overrides)
    # validate every gated field. Dataclasses
    # don't enforce annotations at runtime, so a hand-edited
    # `chunk_seconds = "fast"` lands here as `cfg.chunk_seconds = "fast"`
    # and would later crash deep in `_run_loop`. The validator names
    # the field, warns to stderr, and resets to the AppConfig default.
    _validate_appconfig(cfg, source=str(path))
    if migrated:
        # announce the migration so a user who set
        # a value and then sees it change has a paper trail. The notice
        # also explains how to opt out of future bumps for a given key.
        print(
            f"[woys] migrated config schema {schema} → 10 at {path}; "
            f"pin a value across future schema bumps by adding its "
            f"field name to `_user_overrides` in config.toml.",
            file=sys.stderr,
        )
        try:
            save_config(cfg, path)
        except OSError as e:
            print(
                f"[woys] failed to persist migrated config: "
                f"{type(e).__name__}: {e}\n"
                f"       (running with migrated values in memory; the "
                f"file on disk is unchanged and will migrate again on "
                f"the next launch).",
                file=sys.stderr,
            )
    return cfg


_CONFIG_HEADER = b"""# woys config.toml -- managed by `woys` (the engine, TUI, CLI). Edit
# any knob below; the next launch picks it up.
# Earlier this file was written with zero comments,
# so a user opening it for the first time had to grep the source
# to figure out which fields existed and what they did.
#
# MANAGED keys -- do NOT hand-edit:
#   config_schema_version    F-16-01 migration anchor.
#                            The schema-migration code in load_config
#                            relies on this value to decide whether
#                            to bump stale defaults. If you change
#                            it by hand to a higher value, the next
#                            load thinks your file is already
#                            current and skips migrations that
#                            should have run.
#   _user_overrides          F-16-01 explicit-override list. The
#                            migration leaves fields named here
#                            alone even when they match an old
#                            default. Append a field name yourself
#                            to pin its current value across future
#                            schema bumps; otherwise the file
#                            manages this list automatically when
#                            you touch a knob via TUI keys / CLI.
#
# Knob reference: see `docs/MODELS.md` (foundation weights),
# `docs/INSTALL.md` (rates, sinks), and the inline comments in
# `src/audio/engine.py` `EngineConfig` (the canonical source of
# truth for runtime defaults). `woys info` prints the active
# values alongside the runtime state.
#
# Path: this file lives at $XDG_CONFIG_HOME/woys/config.toml
# (typically ~/.config/woys/config.toml). The directory is mode
# 0700; the file is mode 0600 -- woys' tuning + model paths are
# user-private by design (F-32-02 / commit-047).

"""


def save_config(cfg: AppConfig, path: Path = CONFIG_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {k: v for k, v in asdict(cfg).items() if not k.startswith("_")}
    data.update(cfg._extras)
    # Write atomically via .tmp + rename so a crash mid-write can't corrupt
    # config. v0.6.8 - chmod 0600 (was inheriting umask 0644). Config can
    # contain user paths and tuning that other local users have no business
    # reading.
    # v0.14.0 (area 6 / area 17 / C123 + C268): create file with mode 0600
    # atomically via os.open(O_WRONLY|O_CREAT|O_EXCL, 0o600). Pre-v0.14.0
    # `open(tmp, "wb")` created the file with default umask (typically
    # 0644) and then `os.chmod(0o600)` ran AFTER -- a race window where
    # another local user could read tunings + model paths before the
    # restrictive mode landed. O_EXCL also rejects pre-existing tmp files
    # (a stale .tmp from a crashed write would be an attack vector for
    # symlink-replace; O_EXCL refuses to follow). Plus fsync before
    # replace per C268 so power-loss-during-write doesn't corrupt config.
    tmp = path.with_suffix(path.suffix + ".tmp")
    # drop the
    # `if tmp.exists(): tmp.unlink()` pre-step that pre-fix lived
    # here. That was a classic TOCTOU dance: an attacker (or a
    # crashed prior write that happened to be in flight) could
    # land a symlink at `tmp` between the `.exists()` and the
    # `.unlink()`. Just open with O_CREAT|O_EXCL and handle
    # FileExistsError -- if a stale tmp is there, unlink under
    # `O_NOFOLLOW`-safe semantics by retrying once.
    try:
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        # Stale tmp from a crashed prior write. Unlink + retry.
        # The parent dir is XDG_CONFIG_HOME (~/.config/woys),
        # mode 0700 -- a cross-UID symlink here would have had to
        # cross the parent dir's perm boundary.
        with contextlib.suppress(OSError):
            tmp.unlink()
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as f:
            # write a discoverability
            # header before the TOML body. tomli_w doesn't emit
            # comments natively; we prepend the comment block as raw
            # bytes and let tomli_w handle the data section.
            f.write(_CONFIG_HEADER)
            tomli_w.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise
    os.replace(tmp, path)
