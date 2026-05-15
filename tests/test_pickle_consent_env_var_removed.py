"""review F-05-04: remove process-wide env-var pickle consent.

Pre-fix `_user_trusts_pickle` honored `WOYS_YES_I_TRUST_THE_PICKLE`
from the environment. Set once in a shell rc, it auto-trusted
EVERY `woys convert` -- including any future batch flow that
iterates third-party `.pth` files. `weights_only=False` is genuine
arbitrary-code-execution, so the silent expansion of "trust this
file" to "trust every file forever" was an unacceptable consent
regression.

Post-fix the only consent path is the per-invocation `--yes-i-trust-
the-pickle` CLI flag. The env var is NOT honored. Every unsafe load
emits `[security] UNSAFE pickle load of <path>` to stderr so an
audit can grep for it.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "src" / "server") not in sys.path:
    sys.path.insert(0, str(REPO / "src" / "server"))


def test_env_var_alone_does_not_grant_consent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bug-class test. With WOYS_YES_I_TRUST_THE_PICKLE=1 in the
    environment AND `flag=False`, `_user_trusts_pickle` must return
    False. Pre-fix the env var was honored alone -- a single shell-
    rc export collapsed all per-file consent."""
    from woys.convert import _user_trusts_pickle

    monkeypatch.setenv("WOYS_YES_I_TRUST_THE_PICKLE", "1")
    assert _user_trusts_pickle(False) is False, (
        "F-05-04: the env var must no longer expand 'trust this call' to "
        "'trust every call'; only the per-invocation flag counts"
    )
    monkeypatch.setenv("WOYS_YES_I_TRUST_THE_PICKLE", "yes")
    assert _user_trusts_pickle(False) is False
    monkeypatch.setenv("WOYS_YES_I_TRUST_THE_PICKLE", "true")
    assert _user_trusts_pickle(False) is False


def test_flag_grants_per_call_consent() -> None:
    """Sanity: `flag=True` (set by --yes-i-trust-the-pickle) grants
    consent for this call only. The CLI passes the flag explicitly
    per `woys convert` invocation."""
    from woys.convert import _user_trusts_pickle

    assert _user_trusts_pickle(True) is True


def test_safe_torch_load_error_message_names_removed_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When safe-load fails and the flag is NOT granted, the error
    message must name the removed env var so a user with the old
    config sees explicit deprecation rather than a silently-ignored
    setting."""
    from woys import convert

    # Force `torch.load(weights_only=True)` to fail.
    class _Boom:
        @staticmethod
        def load(*_a: object, **_kw: object) -> object:
            raise RuntimeError("simulated v1-pickle that weights_only rejects")

    monkeypatch.setattr("torch.load", _Boom.load)
    monkeypatch.setenv("WOYS_YES_I_TRUST_THE_PICKLE", "1")

    fake_pth = tmp_path / "voice.pth"
    fake_pth.write_bytes(b"fake-pickle")

    with pytest.raises(RuntimeError, match=r"--yes-i-trust-the-pickle"):
        convert._safe_torch_load(fake_pth, trust_pickle=False)

    # And the env var is explicitly named as REMOVED so the user
    # sees the deprecation.
    try:
        convert._safe_torch_load(fake_pth, trust_pickle=False)
    except RuntimeError as e:
        assert "WOYS_YES_I_TRUST_THE_PICKLE" in str(e)
        assert "no longer honored" in str(e)


def test_safe_torch_load_logs_security_line_on_unsafe_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Forensic pin. Every unsafe load (consent granted) must emit
    `[security] UNSAFE pickle load of <path>` to stderr so an audit
    can grep for it."""
    from woys import convert

    call_log: list[dict[str, object]] = []

    def fake_load(*args: object, **kwargs: object) -> str:
        call_log.append({"args": args, "kwargs": dict(kwargs)})
        if kwargs.get("weights_only") is True:
            raise RuntimeError("simulated v1-pickle")
        # weights_only=False path
        return "loaded-ok"

    monkeypatch.setattr("torch.load", fake_load)
    fake_pth = tmp_path / "voice.pth"
    fake_pth.write_bytes(b"fake")

    result = convert._safe_torch_load(fake_pth, trust_pickle=True)
    err = capsys.readouterr().err

    assert result == "loaded-ok"
    assert "[security] UNSAFE pickle load of" in err
    assert str(fake_pth) in err
    # The second torch.load call must explicitly use weights_only=False.
    unsafe_calls = [c for c in call_log if c["kwargs"].get("weights_only") is False]
    assert len(unsafe_calls) == 1


def test_env_var_name_constant_marked_removed_in_source() -> None:
    """Structural pin: the env-var constant is renamed to
    `_TRUST_PICKLE_ENV_NAME_REMOVED` so a future maintainer who
    grep-and-restores the old name surfaces the rename via this test.
    The OLD name `_TRUST_PICKLE_ENV` (without the suffix) must NOT be
    used as a `if env.get(...)` consent gate again."""
    src = Path(__file__).resolve().parent.parent / "src" / "woys" / "convert.py"
    text = src.read_text()
    # The new name exists.
    assert "_TRUST_PICKLE_ENV_NAME_REMOVED" in text
    # No `os.environ.get(_TRUST_PICKLE_ENV` consent read.
    assert "os.environ.get(_TRUST_PICKLE_ENV" not in text, (
        "the env var must NOT be consulted as a consent input; F-05-04"
    )
    assert 'os.environ.get("WOYS_YES_I_TRUST_THE_PICKLE"' not in text
