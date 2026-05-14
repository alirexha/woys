"""review F-merged-014 (P1): the single place woys configures logging.

Pre-fix `logging.getLogger("woys.*")` was called in `tui/hotkey.py` and
`tui/control.py`, but no handler / `basicConfig` was ever configured
anywhere -- so those `logger.warning` / `logger.error` / `logger.exception`
records went to Python's `lastResort` stderr handler, which Textual
hijacks. A non-developer's post-mortem evidence was terminal scrollback
that vanished on quit.

`setup_logging()` attaches a `RotatingFileHandler` to the `woys` logger
namespace, so every `getLogger("woys.*")` call across the CLI, the TUI,
and the inference child lands in one persistent file. It is idempotent --
all three entry points may call it freely.

This module is the keystone for the observability findings that "log it"
(F-08-04 / F-08-07 / F-08-09 / F-08-12): they have a real destination now.
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOGGER_NAME = "woys"
_MAX_BYTES = 2 * 1024 * 1024  # 2 MiB per file
_BACKUP_COUNT = 3


def log_dir() -> Path:
    """`$XDG_STATE_HOME/woys/` (or `~/.local/state/woys/`).

    Per the XDG Base Directory spec, logs are *state* data -- not config
    (`XDG_CONFIG_HOME`, where `config.toml` lives) and not runtime
    (`XDG_RUNTIME_DIR`, where the control socket lives).
    """
    base = os.environ.get("XDG_STATE_HOME") or str(Path.home() / ".local" / "state")
    return Path(base) / "woys"


def log_path() -> Path:
    """The rotating log file: `<log_dir()>/woys.log`."""
    return log_dir() / "woys.log"


def setup_logging(*, level: int = logging.INFO) -> Path:
    """Attach a `RotatingFileHandler` to the `woys` logger namespace and
    return the log file path.

    Idempotent: if a `RotatingFileHandler` is already attached (a prior
    call in this process, or a test that pre-configured it), this is a
    no-op apart from re-asserting the level. Safe to call from the CLI,
    the TUI, and the inference child -- they share the one log file.
    """
    path = log_path()
    logger = logging.getLogger(_LOGGER_NAME)
    if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            path, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8"
        )
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    # Keep woys records in the woys namespace -- don't propagate to the
    # root logger / whatever the host process configured.
    logger.propagate = False
    return path
