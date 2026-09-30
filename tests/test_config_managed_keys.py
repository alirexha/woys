"""Hand-edited managed keys (`config_schema_version`, `_user_overrides`)
with the wrong TOML type must not crash load_config or corrupt the file.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.parametrize("raw", ['"abc"', "[1]", "{ a = 1 }", "9.9", "true"])
def test_non_int_schema_version_is_treated_as_zero(
    tmp_path: Path, raw: str, capsys: pytest.CaptureFixture[str]
) -> None:
    from tui.config import load_config

    p = tmp_path / "config.toml"
    # 4.0 is the schema<7 old sola_search_ms default: it only migrates when
    # the bad version is treated as 0 (a legacy file).
    p.write_text(f"config_schema_version = {raw}\nsola_search_ms = 4.0\n")
    cfg = load_config(p)
    assert cfg._extras["config_schema_version"] == 10
    assert cfg.sola_search_ms != 4.0
    assert "config_schema_version" in capsys.readouterr().err


def test_user_overrides_as_a_string_pins_that_one_field(tmp_path: Path) -> None:
    import tomllib

    from tui.config import load_config

    p = tmp_path / "config.toml"
    # 220 is the schema<9 old output_latency_ms default; the pin must hold it.
    # sola_search_ms = 4.0 migrates (schema<7), so the file is re-saved.
    p.write_text(
        'config_schema_version = 6\n_user_overrides = "output_latency_ms"\n'
        "output_latency_ms = 220\nsola_search_ms = 4.0\n"
    )
    cfg = load_config(p)
    assert cfg.output_latency_ms == 220
    assert cfg._extras["_user_overrides"] == ["output_latency_ms"]
    # The migration saved the file: the list must round-trip intact, not
    # as the string's characters.
    with open(p, "rb") as f:
        assert tomllib.load(f)["_user_overrides"] == ["output_latency_ms"]


@pytest.mark.parametrize("raw", ["{ output_latency_ms = 1 }", "7", "true"])
def test_user_overrides_of_wrong_type_is_dropped_with_a_warning(
    tmp_path: Path, raw: str, capsys: pytest.CaptureFixture[str]
) -> None:
    from tui.config import load_config

    p = tmp_path / "config.toml"
    p.write_text(f"config_schema_version = 8\n_user_overrides = {raw}\noutput_latency_ms = 220\n")
    cfg = load_config(p)
    assert cfg._extras["_user_overrides"] == []
    assert cfg.output_latency_ms != 220
    assert "_user_overrides" in capsys.readouterr().err
