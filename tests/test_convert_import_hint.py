"""`woys convert`'s import-failure hint must name the real missing module.

The hint told users to install the `[convert]` extra (fastapi, uvicorn,
socketio...), but the convert path imports none of those: the only
things that can be missing there are torch, onnx and onnxsim, which are
core dependencies. Installing the extra never fixed the error.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from woys import convert

EXPORTER = "voice_changer.RVC.onnxExporter.export2onnx"


def test_import_hint_names_missing_module_not_convert_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pth = tmp_path / "voice.pth"
    pth.write_bytes(b"")
    meta = convert._RVCMeta(
        modelType="pyTorchRVCv2",
        samplingRate=40000,
        f0=True,
        embChannels=768,
        embedder="hubert_base",
        embOutputLayer=12,
        useFinalProj=False,
    )
    monkeypatch.setattr(convert, "_probe_pth_metadata", lambda *a, **k: meta)
    # A None entry makes `import` raise ImportError naming the module.
    monkeypatch.setitem(sys.modules, EXPORTER, None)

    with pytest.raises(RuntimeError) as exc_info:
        convert.convert_pth_to_onnx(pth)
    msg = str(exc_info.value)
    assert "[convert]" not in msg
    assert EXPORTER in msg
