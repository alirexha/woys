"""woys must configure a persistent log
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


@pytest.fixture
def umask_022() -> Any:
    import os

    old = os.umask(0o022)
    try:
        yield
    finally:
        os.umask(old)


def test_log_dir_and_file_are_private(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clean_woys_logger: Any,
    umask_022: Any,
) -> None:
    """The log records model paths and control commands, the same data
    config.toml keeps 0600 in a 0700 dir. Pre-fix the log dir and file came
    out 0755 / 0644 under umask 022, and so did every rotated file."""
    import stat

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    from woys import logsetup

    path = logsetup.setup_logging()
    logging.getLogger("woys.test").error("x")
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600

    handler = next(h for h in clean_woys_logger.handlers if isinstance(h, RotatingFileHandler))
    handler.doRollover()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_existing_world_readable_log_is_tightened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_woys_logger: Any
) -> None:
    """A woys.log left 0644 by an older release is made 0600 on open."""
    import stat

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    from woys import logsetup

    old = logsetup.log_path()
    old.parent.mkdir(parents=True)
    old.write_text("earlier run\n")
    old.chmod(0o644)
    logsetup.setup_logging()
    assert stat.S_IMODE(old.stat().st_mode) == 0o600
    assert old.read_text().startswith("earlier run\n")
