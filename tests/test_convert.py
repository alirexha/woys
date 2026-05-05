"""Phase C tests — `woys convert <pth>`.

Heavy: actually exports a small RVC checkpoint to ONNX. Skipped when the
fixture .pth is missing (no model download in CI / dry-runs).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PTH_FIXTURE = Path.home() / ".local" / "share" / "woys" / "models" / "amitaro_v2.pth"


@pytest.mark.gpu
@pytest.mark.slow
def test_convert_amitaro_pth_to_onnx(tmp_path: Path) -> None:
    """Convert a real .pth, then load the result through ORT and verify the
    I/O names match what the engine consumes."""
    if not PTH_FIXTURE.exists():
        pytest.skip(
            f"{PTH_FIXTURE} missing — fetch via "
            "`curl -L -o $models/amitaro_v2.pth "
            "https://huggingface.co/wok000/vcclient_model/resolve/main/"
            "rvc_v2_alpha/amitaro/amitaro_v2_40k_e100.pth`"
        )

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from woys.convert import convert_pth_to_onnx

    out = tmp_path / "amitaro_test.onnx"
    result = convert_pth_to_onnx(PTH_FIXTURE, out)
    assert result == out
    assert out.exists()
    assert out.stat().st_size > 10 * 1024 * 1024  # > 10 MB

    # Engine must be able to load the converted ONNX.
    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    in_names = {i.name for i in sess.get_inputs()}
    out_names = {o.name for o in sess.get_outputs()}
    # Same I/O signature the engine's _infer() reads. The f0 path adds
    # pitch/pitchf; the nono path drops them.
    assert "feats" in in_names and "p_len" in in_names and "sid" in in_names
    assert "audio" in out_names


@pytest.mark.gpu
def test_convert_metadata_probe_v2() -> None:
    """The metadata probe alone (no torch.onnx.export) — fast unit check."""
    if not PTH_FIXTURE.exists():
        pytest.skip(f"{PTH_FIXTURE} missing")

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from woys.convert import _probe_pth_metadata

    meta = _probe_pth_metadata(PTH_FIXTURE)
    # The amitaro fixture is a v2 40 kHz f0 model.
    assert meta.f0 is True
    assert meta.samplingRate in {40_000, 48_000, 32_000}
    assert meta.embChannels == 768
    assert meta.embOutputLayer == 12
    assert meta.useFinalProj is False


def test_cli_convert_handles_missing_input(tmp_path: Path) -> None:
    """A clear error, exit code 1 — never silent failure."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from woys.convert import cli_convert

    rc = cli_convert(str(tmp_path / "does-not-exist.pth"))
    assert rc == 1
