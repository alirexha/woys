"""Non-finite floats are refused by the field validator.

TOML accepts `nan` and `inf`. Every comparison with NaN is False, so a
`chunk_seconds = nan` slipped past both range checks, loaded without a
warning, and later crashed engine start (`round(nan * rate)`). A shared
`.vcprofile` could carry the same value past the untrusted-import gate.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

_FLOAT_FIELDS = [
    "chunk_seconds",
    "sola_crossfade_ms",
    "sola_search_ms",
    "sola_context_ms",
    "input_gain_db",
    "input_gate_dbfs",
    "input_gate_hysteresis_ms",
]


@pytest.mark.parametrize("field", _FLOAT_FIELDS)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_validate_field_rejects_non_finite(field: str, value: float) -> None:
    from tui.config import validate_field

    err = validate_field(field, value)
    assert err is not None
    assert field in err


def test_load_config_resets_nan_to_default(tmp_path: Path) -> None:
    from tui.config import AppConfig, load_config

    p = tmp_path / "config.toml"
    p.write_text("config_schema_version = 10\nchunk_seconds = nan\ninput_gain_db = nan\n")
    cfg = load_config(p)
    assert math.isfinite(cfg.chunk_seconds)
    assert cfg.chunk_seconds == AppConfig().chunk_seconds
    assert cfg.input_gain_db == AppConfig().input_gain_db


def test_vcprofile_import_refuses_nan(tmp_path: Path) -> None:
    from woys.vcprofile import import_profile

    vp = tmp_path / "e.vcprofile"
    vp.write_text(
        '[meta]\nformat_version = 1\nprofile_name = "e"\n'
        "[profile]\nsola_context_ms = nan\n"
        '[model]\nsha256 = ""\n'
    )
    cfg_path = tmp_path / "config.toml"
    with pytest.raises(ValueError, match="refusing import"):
        import_profile(vp, config_path=cfg_path, models_dir=tmp_path)
