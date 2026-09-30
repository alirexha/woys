"""`woys profile use` must not report success on an ERR reply.

A running TUI that times out the PROFILE job, or answers with an
internal error, used to fall through to the "unrecognized reply" branch,
which printed "active profile -> X" and exited 0.
"""

from __future__ import annotations

import pytest


def test_cli_profile_use_exits_nonzero_on_err_reply(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import tui.config as tc
    from woys.profiles import cli_profile_use

    cfg = tc.AppConfig()
    cfg._extras["profiles"] = {"p": {"f0_up_key": 3}}
    tc.save_config(cfg)
    monkeypatch.setattr(
        "tui.control.submit_and_wait",
        lambda *_a, **_k: "ERR job=abc timed out after 10.0s; last='OK state=running'",
    )

    rc = cli_profile_use("p")

    assert rc == 1
    assert "timed out" in capsys.readouterr().err
