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
