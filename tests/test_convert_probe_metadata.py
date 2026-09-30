"""`_probe_pth_metadata` classifies a checkpoint from its dict alone.

The probe is a CPU-only torch.load plus a decision tree over `config`,
`version` and `f0`, so it runs on small synthetic checkpoints instead of the
real amitaro .pth (which CI never has).

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

# Standard RVC checkpoints carry an 18-entry `config`; the last one is the
# sampling rate.
_CONFIG_18: list[Any] = [0] * 17 + [40_000]


def _write_ckpt(path: Path, cpt: dict[str, Any]) -> Path:
    torch = pytest.importorskip("torch")
    torch.save(cpt, str(path))
    return path


def test_probe_v2_f0(tmp_path: Path) -> None:
    from woys.convert import _probe_pth_metadata

    pth = _write_ckpt(tmp_path / "v2.pth", {"config": _CONFIG_18, "version": "v2", "f0": 1})
    meta = _probe_pth_metadata(pth)
    assert meta.f0 is True
    assert meta.samplingRate == 40_000
    assert meta.embChannels == 768
    assert meta.embOutputLayer == 12
    assert meta.useFinalProj is False


def test_probe_v1_nono(tmp_path: Path) -> None:
    from woys.convert import _probe_pth_metadata

    pth = _write_ckpt(tmp_path / "v1.pth", {"config": _CONFIG_18, "version": "v1", "f0": 0})
    meta = _probe_pth_metadata(pth)
    assert meta.f0 is False
    assert meta.embChannels == 256
    assert meta.embOutputLayer == 9
    assert meta.useFinalProj is True


def test_probe_rejects_checkpoint_without_config(tmp_path: Path) -> None:
    from woys.convert import _probe_pth_metadata

    pth = _write_ckpt(tmp_path / "bad.pth", {"weight": {}})
    with pytest.raises(ValueError, match="missing 'config'"):
        _probe_pth_metadata(pth)
