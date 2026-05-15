"""review F-merged-012: per-field type + sane-range validation
for AppConfig and `.vcprofile` import.

Pre-fix:
- `AppConfig(**fields_in)` in `tui/config.py:489` accepted any TOML
  value (dataclasses do not enforce annotations). `chunk_seconds =
  "fast"` crashed deep in `_run_loop`.
- `.vcprofile`'s `setattr(tmp_cfg, k, v)` loop accepted any value.
  A shared `.vcprofile` with `chunk_seconds = -1` was a DoS class --
  the project's primary threat surface (cross-user share artifacts
  are by definition untrusted).
- The contrast: the engine hard-fails a bad `embedder` (engine.py:
  1567-1569). The pattern was just not applied to numeric / bool
  fields.

Post-fix the same `_FIELD_VALIDATORS` table gates both surfaces:
- `load_config` (user-owned TOML): warn-to-stderr + reset to default.
- `vcprofile.import_profile` (untrusted artifact): raise ValueError.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "src" / "server") not in sys.path:
    sys.path.insert(0, str(REPO / "src" / "server"))


# --- validate_field direct unit tests -------------------------------------


@pytest.mark.parametrize(
    "name, value, error_fragment",
    [
        ("chunk_seconds", "fast", "expected float"),
        ("chunk_seconds", -1.0, "must be >= 0.01"),
        ("chunk_seconds", 5.0, "must be <= 2.0"),
        ("f0_up_key", 999, "must be <= 24"),
        ("f0_up_key", True, "bool used where int expected"),
        ("f0_up_key", "12", "expected int"),
        ("mic_rate", 99_000, "must be one of"),
        ("sid", -3, "must be >= 0"),
        ("embedder", "torch", "must be one of ['onnx']"),
        ("gpu_anti_jitter_mode", "max", "must be one of"),
        ("monitor", 1, "expected bool"),
        ("monitor", "yes", "expected bool"),
        ("input_gate_dbfs", 10.0, "must be <= 0"),
        ("output_latency_ms", -50, "must be >= 10"),
    ],
)
def test_validate_field_rejects_invalid(name: str, value: object, error_fragment: str) -> None:
    from tui.config import validate_field

    err = validate_field(name, value)
    assert err is not None, f"{name}={value!r} should be rejected"
    assert error_fragment in err, (
        f"error for {name}={value!r} should contain {error_fragment!r}; got {err!r}"
    )


@pytest.mark.parametrize(
    "name, value",
    [
        ("chunk_seconds", 0.25),
        ("chunk_seconds", 0.5),
        ("f0_up_key", 0),
        ("f0_up_key", -12),
        ("f0_up_key", 24),
        ("mic_rate", 48_000),
        ("mic_rate", 16_000),
        ("embedder", "onnx"),
        ("gpu_anti_jitter_mode", "off"),
        ("gpu_anti_jitter_mode", "both"),
        ("monitor", True),
        ("monitor", False),
        ("rvc_model", ""),
        ("rvc_model", "/home/user/voice.onnx"),
    ],
)
def test_validate_field_accepts_valid(name: str, value: object) -> None:
    from tui.config import validate_field

    err = validate_field(name, value)
    assert err is None, f"{name}={value!r} should be accepted; got {err!r}"


def test_validate_field_unknown_passes_through() -> None:
    """An unknown field (e.g., a user-added key in _extras) must NOT
    be gated -- the pass-through bag is intentional. validate_field
    returns None for any name not in the table."""
    from tui.config import validate_field

    assert validate_field("user_custom_field", "anything") is None
    assert validate_field("future_engine_knob", 42) is None


def test_validate_field_accepts_int_where_float_is_wanted() -> None:
    """TOML happily parses `0.0` as a float and `0` as an int. AppConfig
    declares the SOLA fields as float; the validator should accept `0`
    (lossless widening) without rejecting it as a type mismatch."""
    from tui.config import validate_field

    assert validate_field("sola_search_ms", 0) is None
    assert validate_field("sola_search_ms", 16) is None


# --- load_config sanitization (user-owned config.toml) --------------------


def _write_toml(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


def test_load_config_warns_and_defaults_on_string_for_float(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The verdict's test: a `chunk_seconds = "fast"` in config gives a
    clear per-field error on stderr, not a deep traceback. The user
    keeps a working config; only the bad field is reset."""
    from audio.engine import EngineConfig
    from tui.config import load_config

    cfg_path = tmp_path / "config.toml"
    _write_toml(
        cfg_path,
        """
config_schema_version = 10
chunk_seconds = "fast"
f0_up_key = 5
""",
    )
    cfg = load_config(cfg_path)
    out = capsys.readouterr()

    assert "chunk_seconds: expected float" in out.err, (
        f"stderr should explain the bad field; got: {out.err!r}"
    )
    assert "resetting to default" in out.err
    # Bad field reset, OTHER fields preserved:
    assert cfg.chunk_seconds == EngineConfig().chunk_seconds
    assert cfg.f0_up_key == 5


def test_load_config_warns_and_defaults_on_out_of_range_float(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from audio.engine import EngineConfig
    from tui.config import load_config

    cfg_path = tmp_path / "config.toml"
    _write_toml(
        cfg_path,
        """
config_schema_version = 10
chunk_seconds = -1.0
""",
    )
    cfg = load_config(cfg_path)
    out = capsys.readouterr()

    assert "chunk_seconds: must be >= 0.01" in out.err
    assert cfg.chunk_seconds == EngineConfig().chunk_seconds


def test_load_config_warns_and_defaults_on_bad_enum(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from tui.config import AppConfig, load_config

    cfg_path = tmp_path / "config.toml"
    _write_toml(
        cfg_path,
        """
config_schema_version = 10
embedder = "torch"
""",
    )
    cfg = load_config(cfg_path)
    out = capsys.readouterr()

    assert "embedder: must be one of ['onnx']" in out.err
    assert cfg.embedder == AppConfig().embedder  # "onnx"


def test_load_config_clean_config_emits_no_validator_warnings(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Back-compat: a clean config with all-valid values must not
    emit any [woys] config: invalid ... warning."""
    from tui.config import AppConfig, load_config, save_config

    cfg_path = tmp_path / "config.toml"
    save_config(AppConfig(), cfg_path)
    # Drain anything emitted by save_config (e.g., a fresh write
    # notice from another fix doesn't apply on the SECOND load).
    capsys.readouterr()

    load_config(cfg_path)
    out = capsys.readouterr()
    assert "invalid" not in out.err
    assert "resetting to default" not in out.err


# --- vcprofile refusal (untrusted artifact) -------------------------------


def _write_vcprofile_with_bad_field(path: Path, body: str) -> None:
    path.write_text(
        '[meta]\nformat_version = 1\nprofile_name = "x"\n'
        f'[profile]\n{body}\n[model]\nfilename = ""\nsha256 = ""\n',
        encoding="utf-8",
    )


def test_vcprofile_import_refuses_chunk_seconds_negative(tmp_path: Path) -> None:
    """The verdict's bug-class test: a `.vcprofile` with
    `chunk_seconds = -1` is a DoS class -- the importer must REFUSE,
    not warn-and-default. (`.vcprofile` is the untrusted-artifact
    surface; `load_config` is the user-owned surface.)"""
    from woys.vcprofile import import_profile

    bad = tmp_path / "evil.vcprofile"
    _write_vcprofile_with_bad_field(bad, "chunk_seconds = -1.0")

    with pytest.raises(ValueError, match=r"refusing import.*chunk_seconds.*must be >= 0\.01"):
        import_profile(bad, "x")


def test_vcprofile_import_refuses_string_where_float_expected(tmp_path: Path) -> None:
    from woys.vcprofile import import_profile

    bad = tmp_path / "evil.vcprofile"
    _write_vcprofile_with_bad_field(bad, 'chunk_seconds = "fast"')

    with pytest.raises(ValueError, match=r"refusing import.*chunk_seconds.*expected float"):
        import_profile(bad, "x")


def test_vcprofile_import_refuses_bool_where_int_expected(tmp_path: Path) -> None:
    """The Python `bool <: int` quirk could let an attacker sneak a
    `True` past an `isinstance(value, int)` check. The validator
    must reject it explicitly."""
    from woys.vcprofile import import_profile

    bad = tmp_path / "evil.vcprofile"
    _write_vcprofile_with_bad_field(bad, "f0_up_key = true")

    with pytest.raises(ValueError, match=r"refusing import.*f0_up_key.*bool used"):
        import_profile(bad, "x")


def test_vcprofile_import_refuses_unknown_embedder(tmp_path: Path) -> None:
    from woys.vcprofile import import_profile

    bad = tmp_path / "evil.vcprofile"
    _write_vcprofile_with_bad_field(bad, 'embedder = "torch"')

    with pytest.raises(ValueError, match=r"refusing import.*embedder.*one of \['onnx'\]"):
        import_profile(bad, "x")
