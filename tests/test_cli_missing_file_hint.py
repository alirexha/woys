"""The missing-file hint names the config file woys actually reads.

Since the config dir honors XDG_CONFIG_HOME / WOYS_CONFIG_DIR, the
hardcoded "~/.config/woys/config.toml" in the hint pointed some users at a
file woys does not use.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from woys import cli


def test_missing_file_hint_uses_the_resolved_config_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg_dir = tmp_path / "custom-config"
    monkeypatch.setenv("WOYS_CONFIG_DIR", str(cfg_dir))

    def _boom(_p: argparse.ArgumentParser, _a: argparse.Namespace) -> int:
        raise FileNotFoundError("rvc model not found at /nope.onnx")

    monkeypatch.setattr(cli, "_dispatch", _boom)
    assert cli.main(["info"]) == 1
    assert str(cfg_dir / "config.toml") in capsys.readouterr().err
