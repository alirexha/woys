"""Tests for the v0.3.0 model-library CLI surface."""

from __future__ import annotations

import shutil
from pathlib import Path

from woys.models import (
    FOUNDATION_NAMES,
    ModelEntry,
    discover_models,
    find_by_name,
)


def _write_dummy_onnx(path: Path) -> None:
    path.write_bytes(b"\x08\x07ir_garbage")  # not a valid ONNX, but the prober tolerates that


def test_discover_filters_out_foundation_files(tmp_path: Path) -> None:
    """rmvpe/contentvec/hubert files must not show up as voice models."""
    models = tmp_path
    for fn in ("rmvpe_wrapped.onnx", "contentvec-f.onnx", "amitaro_v2_16k.onnx"):
        _write_dummy_onnx(models / fn)
    discovered = {e.name for e in discover_models(models)}
    assert "amitaro_v2_16k" in discovered
    assert "rmvpe_wrapped" not in discovered
    assert "contentvec-f" not in discovered


def test_discover_empty_dir(tmp_path: Path) -> None:
    assert discover_models(tmp_path) == []


def test_find_by_name_resolves_stem_and_filename(tmp_path: Path) -> None:
    f = tmp_path / "foo.onnx"
    _write_dummy_onnx(f)
    assert find_by_name("foo", tmp_path) == f
    assert find_by_name("foo.onnx", tmp_path) == f
    assert find_by_name("does-not-exist", tmp_path) is None


def test_find_by_name_accepts_absolute_path(tmp_path: Path) -> None:
    f = tmp_path / "bar.onnx"
    _write_dummy_onnx(f)
    # Absolute-path passthrough — useful when the user has a model outside the cache.
    assert find_by_name(str(f), tmp_path) == f


def test_foundation_names_set_is_complete() -> None:
    """Sanity: any new foundation file must be added here so it's filtered out."""
    expected = {
        "rmvpe.onnx",
        "rmvpe-fp16.onnx",
        "rmvpe_wrapped.onnx",
        "rmvpe_wrapped-fp16.onnx",
        "contentvec-f.onnx",
        "contentvec-f-fp16.onnx",
        "hubert_base.onnx",
    }
    assert expected == FOUNDATION_NAMES


def test_model_entry_dataclass_round_trip(tmp_path: Path) -> None:
    f = tmp_path / "small.onnx"
    _write_dummy_onnx(f)
    f.write_bytes(f.read_bytes() + b"\x00" * 1024)
    entries = discover_models(tmp_path)
    assert len(entries) == 1
    e = entries[0]
    assert isinstance(e, ModelEntry)
    assert e.name == "small"
    assert e.size_mib > 0
    # Probe will fail on garbage bytes — that's allowed (sr/is_v2/f0 = None).
    assert e.sample_rate is None
    assert e.is_v2 is None
    assert e.f0 is None


def test_cleanup() -> None:  # pragma: no cover — keeps pytest happy if temp leaks
    leftover = Path("/tmp/woys_models_test")
    if leftover.exists():
        shutil.rmtree(leftover, ignore_errors=True)
