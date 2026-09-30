"""The suite must never write the user's real woys log.

`cli.main()` calls `setup_logging()`, which resolves `$XDG_STATE_HOME` (or
`~/.local/state`) and attaches a file handler for the rest of the process.
Without isolation every CLI test appended fake startup lines and fake
"operational error" records to the developer's real
`~/.local/state/woys/woys.log` -- the file they read for post-mortems.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest

REAL_STATE_DIR = Path.home() / ".local" / "state" / "woys"


@pytest.fixture
def clean_woys_logger() -> Iterator[logging.Logger]:
    log = logging.getLogger("woys")
    saved = list(log.handlers)
    for h in saved:
        log.removeHandler(h)
    yield log
    for h in list(log.handlers):
        h.close()
        log.removeHandler(h)
    for h in saved:
        log.addHandler(h)


def test_log_path_is_per_test_tmp(tmp_path: Path) -> None:
    from woys import logsetup

    assert logsetup.log_path().is_relative_to(tmp_path)


def test_cli_main_logs_outside_the_real_state_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clean_woys_logger: logging.Logger,
) -> None:
    from woys import cli, logsetup

    # Check before main() runs so a broken isolation fails here instead of
    # appending to the real log.
    assert logsetup.log_path().is_relative_to(tmp_path)

    monkeypatch.setattr(cli, "cmd_info", lambda: 0)
    assert cli.main(["info"]) == 0

    files = [
        Path(h.baseFilename)
        for h in clean_woys_logger.handlers
        if isinstance(h, RotatingFileHandler)
    ]
    assert files, "cli.main() must attach the file handler"
    for f in files:
        assert f.is_relative_to(tmp_path)
        assert not f.is_relative_to(REAL_STATE_DIR)
