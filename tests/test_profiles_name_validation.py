"""`woys profile save` only accepts plain profile names.

Names end up in TOML keys, TUI toasts and shell completions. The rule
matches the .vcprofile import: 1-64 chars of ASCII letters, digits,
space, `.`, `_`, `-`, starting with a letter or digit, no trailing space.
"""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    "name",
    ["", "x[/bold]", "[@click=app.quit]x[/]", "-rf", ".hidden", "a/b", "trailing ", "é", "a" * 65],
)
def test_cli_profile_save_rejects_unsafe_names(
    name: str, capsys: pytest.CaptureFixture[str]
) -> None:
    import tui.config as tc
    from woys.profiles import cli_profile_save

    rc = cli_profile_save(name)

    assert rc == 1
    assert "invalid profile name" in capsys.readouterr().err
    assert name not in tc.load_config()._extras.get("profiles", {})


@pytest.mark.parametrize("name", ["main", "Deep voice 2", "a.b_c-d", "0", "a" * 64])
def test_cli_profile_save_accepts_plain_names(name: str) -> None:
    import tui.config as tc
    from woys.profiles import cli_profile_save

    assert cli_profile_save(name) == 0
    assert name in tc.load_config()._extras["profiles"]
