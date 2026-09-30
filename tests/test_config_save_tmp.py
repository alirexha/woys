"""save_config's temp-file handling.

save_config writes `config.toml.tmp` with O_EXCL and renames it over
config.toml. Two paths had no test: a stale .tmp left by a crashed write
must be removed and the save retried (otherwise every later save fails
and nothing persists again), and a write that fails midway must remove
its .tmp and leave config.toml untouched.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pytest


def test_stale_tmp_is_replaced_and_the_save_succeeds(tmp_path: Path) -> None:
    from tui.config import AppConfig, load_config, save_config

    path = tmp_path / "config.toml"
    tmp = path.with_suffix(".toml.tmp")
    tmp.write_text("half-written by a crashed save")
    cfg = AppConfig()
    cfg.f0_up_key = 4
    save_config(cfg, path)
    assert not tmp.exists()
    assert load_config(path).f0_up_key == 4


def test_failed_write_removes_tmp_and_keeps_the_old_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tomli_w

    from tui.config import AppConfig, save_config

    path = tmp_path / "config.toml"
    cfg = AppConfig()
    cfg.f0_up_key = 3
    save_config(cfg, path)
    before = path.read_bytes()

    def boom(*_a: Any, **_k: Any) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(tomli_w, "dump", boom)
    cfg.f0_up_key = 9
    with pytest.raises(RuntimeError, match="disk full"):
        save_config(cfg, path)
    assert path.read_bytes() == before
    assert not path.with_suffix(".toml.tmp").exists()
    assert tomllib.loads(path.read_text())["f0_up_key"] == 3
