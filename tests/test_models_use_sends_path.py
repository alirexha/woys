"""`woys models use` sends the resolved model path to the running TUI.

It used to send only the file stem, so `models use /ext/bar.onnx` failed
("no such model: 'bar'") and `models use /ext/foo.onnx` silently swapped
to the library's foo.onnx instead, persisted it, and exited 0.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.parametrize("stem", ["foo", "bar"])
def test_models_use_external_path_reaches_the_tui_intact(
    stem: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tui.control
    from woys.models import cli_models_use, find_by_name

    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "foo.onnx").write_bytes(b"library voice")
    ext = tmp_path / "ext"
    ext.mkdir()
    external = ext / f"{stem}.onnx"
    external.write_bytes(b"external voice")
    sent: list[str] = []

    def fake_submit(cmd: str, **_k: object) -> str:
        sent.append(cmd)
        return "OK state=done elapsed_ms=1"

    monkeypatch.setattr(tui.control, "submit_and_wait", fake_submit)

    assert cli_models_use(str(external), models_dir=lib) == 0

    assert sent[0].startswith("MODEL ")
    # What the TUI's MODEL handler resolves the argument to.
    assert find_by_name(sent[0][len("MODEL ") :], models_dir=lib) == external.resolve()


def test_models_use_library_name_still_resolves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tui.control
    from woys.models import cli_models_use, find_by_name

    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "foo.onnx").write_bytes(b"library voice")
    sent: list[str] = []
    monkeypatch.setattr(
        tui.control,
        "submit_and_wait",
        lambda cmd, **_k: sent.append(cmd) or "OK state=done elapsed_ms=1",
    )

    assert cli_models_use("foo", models_dir=lib) == 0
    assert find_by_name(sent[0][len("MODEL ") :], models_dir=lib) == (lib / "foo.onnx").resolve()
