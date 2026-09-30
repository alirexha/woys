"""An unusable log directory must not break every command.

main() set up the rotating log file before its operational-error guard,
so a read-only home or a file sitting where `$XDG_STATE_HOME/woys` should
be made every command -- `woys info` included -- die with a raw
PermissionError / FileExistsError traceback.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import woys.cli as cli
import woys.logsetup as logsetup


@pytest.mark.parametrize("exc", [PermissionError, FileExistsError])
def test_commands_run_without_a_log_file(
    exc: type[OSError],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def broken(*_a: object, **_k: object) -> Path:
        raise exc(13, "cannot create log dir", "/state/woys")

    monkeypatch.setattr(logsetup, "setup_logging", broken)
    monkeypatch.setattr(cli, "cmd_info", lambda: 0)
    assert cli.main(["info"]) == 0
    err = capsys.readouterr().err
    assert "log file" in err
    assert "/state/woys" in err
