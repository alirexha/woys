"""`models download` must not replace the SHA-pinned foundation weights.

Voices are saved by basename into the directory that also holds
contentvec / rmvpe. A repo file named e.g. `sub/contentvec-f.onnx`
used to unlink the pinned weight and put the repo's file in its place.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest


class _Lfs:
    def __init__(self, data: bytes) -> None:
        self.sha256 = hashlib.sha256(data).hexdigest()
        self.size = len(data)


class _Sibling:
    def __init__(self, name: str, data: bytes) -> None:
        self.rfilename = name
        self.lfs = _Lfs(data)


class _Info:
    def __init__(self, siblings: list[_Sibling]) -> None:
        self.siblings = siblings
        self.sha = "c0ffee" * 6


@pytest.fixture
def fake_hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    import huggingface_hub

    cache = tmp_path / "hf-cache"
    cache.mkdir()
    files = {
        "voice.onnx": b"a voice",
        "sub/contentvec-f.onnx": b"not the real contentvec",
        "rmvpe_wrapped.onnx": b"not the real rmvpe",
    }
    info = _Info([_Sibling(n, d) for n, d in files.items()])
    requested: list[str] = []

    class _Api:
        def repo_info(self, _repo: str) -> _Info:
            return info

    def hf_hub_download(*, repo_id: str, filename: str, revision: str) -> str:
        requested.append(filename)
        out = cache / filename.replace("/", "__")
        out.write_bytes(files[filename])
        return str(out)

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)
    return {"requested": requested}


def test_download_refuses_foundation_weight_names(
    fake_hub: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from woys.models import download_repo

    models = tmp_path / "models"
    models.mkdir()
    pinned = models / "contentvec-f.onnx"
    pinned.write_bytes(b"the pinned contentvec weights")

    landed = download_repo("someone/voices", models)

    assert [p.name for p in landed] == ["voice.onnx"]
    assert pinned.read_bytes() == b"the pinned contentvec weights"
    assert not (models / "rmvpe_wrapped.onnx").exists()
    assert fake_hub["requested"] == ["voice.onnx"]
    err = capsys.readouterr().err
    assert "contentvec-f.onnx" in err and "rmvpe_wrapped.onnx" in err
