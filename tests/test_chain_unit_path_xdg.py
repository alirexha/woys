"""`woys chain enable` must not write its unit relative to the CWD.

`XDG_CONFIG_HOME=""` (set but empty) gave `Path("")`, so the unit landed
at ./systemd/user/woys-chain.service. The XDG spec says an empty or
relative XDG_CONFIG_HOME is to be ignored.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from woys import chain


@pytest.mark.parametrize("value", ["", "relative/cfg", "."])
def test_unit_path_ignores_empty_or_relative_xdg_config_home(
    value: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", value)
    path = chain._systemd_unit_path()
    assert path.is_absolute()
    assert path == tmp_path / ".config" / "systemd" / "user" / chain.SYSTEMD_UNIT_NAME


def test_unit_path_uses_absolute_xdg_config_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    assert chain._systemd_unit_path() == (
        tmp_path / "cfg" / "systemd" / "user" / chain.SYSTEMD_UNIT_NAME
    )
