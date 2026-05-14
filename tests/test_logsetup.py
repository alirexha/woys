"""review F-merged-014 (P1): woys must configure a persistent log
file.

Pre-fix `logging.getLogger("woys.*")` calls in `tui/hotkey.py` /
`tui/control.py` had no handler attached anywhere, so their records went
to Python's `lastResort` stderr -- which Textual hijacks -- and a
non-developer's post-mortem evidence vanished on quit. `setup_logging()`
attaches a `RotatingFileHandler` to the `woys` logger namespace.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))


@pytest.fixture
def clean_woys_logger() -> Any:
    """Detach handlers on the `woys` logger around the test, so
    `setup_logging()` starts from a known state and the test doesn't leak
    a file handler into the rest of the suite."""
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


def test_log_dir_respects_xdg_state_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Logs are *state* data -- they belong under XDG_STATE_HOME."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    from woys import logsetup

    assert logsetup.log_dir() == tmp_path / "state" / "woys"


def test_setup_logging_writes_to_a_persistent_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_woys_logger: Any
) -> None:
    """The bug-class test: after `setup_logging()`, a `woys.*` logger's
    records must land in a persistent file. Pre-fix `woys.logsetup` does
    not exist, so the import fails."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    from woys import logsetup

    path = logsetup.setup_logging()
    assert path == tmp_path / "woys" / "woys.log"

    logging.getLogger("woys.test").error("canary-error-ABC123")
    for h in clean_woys_logger.handlers:
        h.flush()

    assert path.exists(), "setup_logging() must create the log file"
    assert "canary-error-ABC123" in path.read_text(), "woys.* records must reach the file"


def test_setup_logging_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_woys_logger: Any
) -> None:
    """The CLI, TUI, and inference child all call `setup_logging()` -- it
    must not stack a fresh handler each time."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    from woys import logsetup

    logsetup.setup_logging()
    logsetup.setup_logging()
    logsetup.setup_logging()

    file_handlers = [h for h in clean_woys_logger.handlers if isinstance(h, RotatingFileHandler)]
    assert len(file_handlers) == 1, "repeated setup_logging() must not stack handlers"
