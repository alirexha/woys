"""`woys fp16-convert` must never leave a truncated `*-fp16.onnx` behind.

onnx.save wrote straight into `rmvpe_wrapped-fp16.onnx`, which the engine
prefers whenever it exists. A failed write (ENOSPC, Ctrl-C) left a
truncated file; every later run printed "[skip] already present" and the
engine kept loading the broken model. A missing source model also exited 0.
"""

from __future__ import annotations

import errno
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from woys import fp16_convert


def _tiny_model(path: Path) -> None:
    w = numpy_helper.from_array(np.ones((8, 8), dtype=np.float32), "W")
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "W"], ["y"])],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8])],
        [w],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, str(path))


def test_failed_write_leaves_no_fp16_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _tiny_model(tmp_path / "rmvpe_wrapped.onnx")
    real_save = onnx.save

    def failing_save(model: Any, path: str, *args: Any, **kwargs: Any) -> None:
        Path(path).write_bytes(b"\x08\x07partial")
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(onnx, "save", failing_save)
    with pytest.raises(OSError):
        fp16_convert.convert_rmvpe(tmp_path)
    assert not (tmp_path / "rmvpe_wrapped-fp16.onnx").exists()
    assert not (tmp_path / "rmvpe_wrapped-fp16.onnx.part").exists()

    monkeypatch.setattr(onnx, "save", real_save)
    dst = fp16_convert.convert_rmvpe(tmp_path)
    assert dst is not None
    onnx.checker.check_model(str(dst))


def test_truncated_existing_file_is_regenerated(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _tiny_model(tmp_path / "rmvpe_wrapped.onnx")
    dst = tmp_path / "rmvpe_wrapped-fp16.onnx"
    fp16_convert.convert_rmvpe(tmp_path)
    good = dst.read_bytes()
    dst.write_bytes(good[: len(good) // 2])

    assert fp16_convert.convert_rmvpe(tmp_path) == dst
    assert "already present" not in capsys.readouterr().out
    onnx.checker.check_model(str(dst))


def test_valid_existing_file_is_still_skipped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _tiny_model(tmp_path / "rmvpe_wrapped.onnx")
    fp16_convert.convert_rmvpe(tmp_path)
    capsys.readouterr()
    fp16_convert.convert_rmvpe(tmp_path)
    assert "already present" in capsys.readouterr().out


def test_missing_source_exits_nonzero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fp16_convert, "MODELS_DIR", tmp_path)
    assert fp16_convert.cli_fp16_convert(["rmvpe"]) != 0


def test_cli_converts_present_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fp16_convert, "MODELS_DIR", tmp_path)
    _tiny_model(tmp_path / "rmvpe_wrapped.onnx")
    assert fp16_convert.cli_fp16_convert(["rmvpe"]) == 0
    assert (tmp_path / "rmvpe_wrapped-fp16.onnx").is_file()
