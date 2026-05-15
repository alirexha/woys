"""review F-31-09: post-export fp16 numerical quality gate.

Pre-fix `convert_pth_to_onnx` with `--fp16` ran only `_validate_onnx_
loads` (load + I/O names) -- the docstring admitted "v1 models often
degrade" but nothing in the pipeline MEASURED the degradation. Users
discovered fp16 quality regressions by ear, possibly mid-call.

Post-fix the export:
- runs a SECOND fp32 reference export to a tmp path;
- calls `_fp16_quality_gate` to compute fp16-vs-fp32 SNR on a seeded
  reference input;
- prints SNR + a loud stderr warning when SNR < threshold;
- emits a v1+fp16 advisory line (embChannels=256 is the v1 signature);
- deletes the tmp fp32 reference whether the gate ran or failed.

These tests stay fast: they mock ORT InferenceSession with stub
sessions that return prepared output arrays. The slow / GPU
integration test (real .pth -> real .onnx) lives in test_convert.py
and stays gated behind `gpu+slow` markers.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "src" / "server") not in sys.path:
    sys.path.insert(0, str(REPO / "src" / "server"))


class _StubInputMeta:
    def __init__(self, name: str, type_str: str) -> None:
        self.name = name
        self.type = type_str


class _StubSession:
    """In-place stand-in for `onnxruntime.InferenceSession` that returns
    a pre-baked output array. `inputs` controls the input type strings
    so we can exercise the fp16-cast path."""

    def __init__(
        self,
        output: np.ndarray[Any, Any],
        *,
        is_f0: bool = True,
        feats_type: str = "tensor(float)",
    ) -> None:
        self._output = output
        self._inputs = [
            _StubInputMeta("feats", feats_type),
            _StubInputMeta("p_len", "tensor(int64)"),
            _StubInputMeta("sid", "tensor(int64)"),
        ]
        if is_f0:
            self._inputs.append(_StubInputMeta("pitch", "tensor(int64)"))
            self._inputs.append(_StubInputMeta("pitchf", "tensor(float)"))

    def get_inputs(self) -> list[_StubInputMeta]:
        return self._inputs

    def run(self, _outs: object, _feed: object) -> list[np.ndarray[Any, Any]]:
        return [self._output]


def _patch_sessions(
    monkeypatch: pytest.MonkeyPatch,
    fp16_out: np.ndarray[Any, Any],
    fp32_out: np.ndarray[Any, Any],
    *,
    is_f0: bool = True,
    fp16_feats_type: str = "tensor(float16)",
) -> None:
    """Replace `ort.InferenceSession` with a factory that returns the
    right stub based on the path argument. The first call (fp16_path)
    returns the fp16_out session; the second (fp32_path) returns the
    fp32_out session."""

    seen: list[str] = []

    def _factory(path: str, providers: list[str]) -> _StubSession:
        seen.append(path)
        if "_fp32" in path or "fp32" in path:
            return _StubSession(fp32_out, is_f0=is_f0, feats_type="tensor(float)")
        return _StubSession(fp16_out, is_f0=is_f0, feats_type=fp16_feats_type)

    monkeypatch.setattr("onnxruntime.InferenceSession", _factory)


def test_fp16_quality_gate_returns_infinite_snr_when_outputs_identical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Identical outputs -> noise is zero -> SNR is +inf. The gate
    should print the SNR (the inf is unusual but valid) and NOT emit
    the loud warning."""
    from woys.convert import _fp16_quality_gate

    out = np.ones((1, 4800), dtype=np.float32)
    _patch_sessions(monkeypatch, out, out)

    fp16 = tmp_path / "test.onnx"
    fp32 = tmp_path / "test_fp32_ref.onnx"
    fp16.write_bytes(b"stub-fp16")
    fp32.write_bytes(b"stub-fp32")

    snr = _fp16_quality_gate(fp16, fp32, is_f0=True, emb_channels=768)
    out_err = capsys.readouterr()

    assert snr == float("inf"), f"identical outputs must yield +inf SNR; got {snr}"
    assert "SNR = inf" in out_err.out, (
        f"the gate must announce the SNR on stdout; got: {out_err.out!r}"
    )
    assert "WARNING: fp16 export degraded" not in out_err.err


def test_fp16_quality_gate_emits_loud_warning_when_below_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Bug-class test: when fp16 differs from fp32 by enough that SNR
    falls below the threshold, the gate emits a LOUD stderr warning
    (containing the word 'WARNING' and the SNR value). Pre-fix nothing
    measured the degradation at all."""
    from woys.convert import _fp16_quality_gate

    # Construct signals with a known SNR. Use a uniform-amplitude
    # signal and add uniform-amplitude noise to make the SNR math
    # easy: SNR = 10 * log10(sig_power / noise_power).
    sig = np.ones((1, 4800), dtype=np.float32)
    # Noise amplitude 1.0 -> noise power 1.0; signal power 1.0 -> SNR = 0 dB.
    noise = np.ones_like(sig)
    fp32 = sig
    fp16 = sig + noise
    _patch_sessions(monkeypatch, fp16, fp32)

    fp16_path = tmp_path / "test.onnx"
    fp32_path = tmp_path / "test_fp32_ref.onnx"
    fp16_path.write_bytes(b"stub")
    fp32_path.write_bytes(b"stub")

    snr = _fp16_quality_gate(fp16_path, fp32_path, is_f0=True, emb_channels=768)
    out = capsys.readouterr()

    assert snr == pytest.approx(0.0, abs=0.01), f"SNR for sig=1, noise=1 should be 0 dB; got {snr}"
    assert "WARNING: fp16 export degraded" in out.err, (
        f"sub-threshold SNR must emit the loud stderr warning; got stderr: {out.err!r}"
    )
    assert f"SNR {snr:.1f}" in out.err


def test_fp16_quality_gate_silent_when_above_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A healthy fp16 export (SNR > threshold) prints the SNR line on
    stdout but emits NO stderr warning -- the user sees the quality
    readout but nothing alarming."""
    from woys.convert import _FP16_SNR_THRESHOLD_DB, _fp16_quality_gate

    sig = np.full((1, 4800), 100.0, dtype=np.float32)  # large signal power
    noise = np.full_like(sig, 0.01)  # tiny noise
    fp32 = sig
    fp16 = sig + noise
    _patch_sessions(monkeypatch, fp16, fp32)

    fp16_path = tmp_path / "test.onnx"
    fp32_path = tmp_path / "test_fp32_ref.onnx"
    fp16_path.write_bytes(b"stub")
    fp32_path.write_bytes(b"stub")

    snr = _fp16_quality_gate(fp16_path, fp32_path, is_f0=True, emb_channels=768)
    out = capsys.readouterr()

    assert snr > _FP16_SNR_THRESHOLD_DB, f"expected SNR above {_FP16_SNR_THRESHOLD_DB}; got {snr}"
    assert "WARNING: fp16 export degraded" not in out.err


def test_fp16_quality_gate_raises_on_shape_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the fp16 and fp32 models produce different output shapes,
    that's a structural bug in the export, not a quality issue.
    The gate raises so the caller surfaces it."""
    from woys.convert import _fp16_quality_gate

    fp16_out = np.zeros((1, 4800), dtype=np.float32)
    fp32_out = np.zeros((1, 4801), dtype=np.float32)
    _patch_sessions(monkeypatch, fp16_out, fp32_out)

    fp16_path = tmp_path / "test.onnx"
    fp32_path = tmp_path / "test_fp32_ref.onnx"
    fp16_path.write_bytes(b"stub")
    fp32_path.write_bytes(b"stub")

    with pytest.raises(RuntimeError, match=r"shape mismatch"):
        _fp16_quality_gate(fp16_path, fp32_path, is_f0=True, emb_channels=768)


def test_fp16_quality_gate_handles_nono_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 'nono' (no-f0) RVC variants drop the pitch/pitchf inputs.
    The gate must build the right input dict for is_f0=False -- a
    f0-feeding gate would fail at session.run() with 'unexpected
    input'."""
    from woys.convert import _fp16_quality_gate

    sig = np.ones((1, 4800), dtype=np.float32)
    _patch_sessions(monkeypatch, sig, sig, is_f0=False)

    fp16_path = tmp_path / "test.onnx"
    fp32_path = tmp_path / "test_fp32_ref.onnx"
    fp16_path.write_bytes(b"stub")
    fp32_path.write_bytes(b"stub")

    snr = _fp16_quality_gate(fp16_path, fp32_path, is_f0=False, emb_channels=768)
    assert snr == float("inf")


def test_fp16_quality_gate_seed_is_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two runs of the gate against the same models produce the SAME
    SNR. The seeded reference input is the reproducibility guarantee."""
    from woys.convert import _fp16_quality_gate

    # Make the stubs RECORD the inputs they receive so we can compare.
    seen_feats: list[np.ndarray[Any, Any]] = []

    class _RecordSession(_StubSession):
        def run(
            self,
            _outs: object,
            feed: dict[str, np.ndarray[Any, Any]],
        ) -> list[np.ndarray[Any, Any]]:
            seen_feats.append(np.asarray(feed["feats"]))
            return [self._output]

    out = np.zeros((1, 4800), dtype=np.float32)

    def _factory(path: str, providers: list[str]) -> _RecordSession:
        ft = "tensor(float16)" if "fp32" not in path else "tensor(float)"
        return _RecordSession(out, is_f0=True, feats_type=ft)

    monkeypatch.setattr("onnxruntime.InferenceSession", _factory)

    fp16 = tmp_path / "test.onnx"
    fp32 = tmp_path / "test_fp32_ref.onnx"
    fp16.write_bytes(b"stub")
    fp32.write_bytes(b"stub")

    _fp16_quality_gate(fp16, fp32, is_f0=True, emb_channels=768)
    _fp16_quality_gate(fp16, fp32, is_f0=True, emb_channels=768)

    # The fp32 session sees fp32 feats both runs; the fp16 session
    # sees fp16-cast feats both runs. Each pair of runs must produce
    # IDENTICAL arrays (the seed produces the same RNG draw).
    assert len(seen_feats) == 4
    # seen_feats indexes: [run1_fp16_call, run1_fp32_call, run2_fp16_call, run2_fp32_call]
    # but the order is determined by the factory's call order in the
    # gate (fp16 session created first). Compare run1 vs run2 by index
    # alignment regardless of which session it hit:
    np.testing.assert_array_equal(seen_feats[0], seen_feats[2])
    np.testing.assert_array_equal(seen_feats[1], seen_feats[3])


# --- structural-pin tests for the convert_pth_to_onnx wiring --------------
# These exercise the call SHAPE without executing the full PyTorch export,
# by mocking _export2onnx + the metadata probe + the gate itself. The
# real-export path is covered by tests/test_convert.py::
# test_convert_amitaro_pth_to_onnx (slow + gpu).


def _make_fake_meta(emb_channels: int, f0: bool, sr: int = 48_000) -> Any:
    """Build a fake `_RVCMeta` minimal enough for the wiring."""
    from woys.convert import _RVCMeta

    return _RVCMeta(
        modelType="pyTorchRVCv2" if emb_channels == 768 else "pyTorchRVC",
        samplingRate=sr,
        f0=f0,
        embChannels=emb_channels,
        embedder="hubert_base",
        embOutputLayer=12 if emb_channels == 768 else 9,
        useFinalProj=emb_channels == 256,
    )


def test_convert_pth_to_onnx_calls_gate_when_fp16_true(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Structural pin: when fp16=True, convert_pth_to_onnx must run a
    second fp32 reference export and call _fp16_quality_gate. Pre-fix
    the gate did not exist."""
    from woys import convert as conv_mod

    pth = tmp_path / "voice.pth"
    pth.write_bytes(b"fake-pth")
    out = tmp_path / "voice.onnx"

    monkeypatch.setattr(conv_mod, "_user_trusts_pickle", lambda _flag: True)
    monkeypatch.setattr(
        conv_mod, "_probe_pth_metadata", lambda *a, **kw: _make_fake_meta(768, True)
    )
    monkeypatch.setattr(conv_mod, "_validate_onnx_loads", lambda *_a, **_kw: None)

    export_calls: list[tuple[str, bool]] = []

    def _fake_export2onnx(
        inp: str, output: str, simple: str, fp16_arg: bool, _meta: dict[str, Any]
    ) -> None:
        # Touch the output path so the existence guard inside
        # convert_pth_to_onnx passes for both calls.
        Path(output).write_bytes(b"fake-onnx")
        Path(simple).write_bytes(b"fake-simple")
        export_calls.append((output, fp16_arg))

    # Patch the late-import location.
    import voice_changer.RVC.onnxExporter.export2onnx as ex_mod  # type: ignore[import-not-found]

    monkeypatch.setattr(ex_mod, "_export2onnx", _fake_export2onnx)

    gate_calls: list[dict[str, Any]] = []

    def _fake_gate(
        fp16_path: Path, fp32_path: Path, *, is_f0: bool, emb_channels: int, **_kw: Any
    ) -> float:
        gate_calls.append(
            {
                "fp16": fp16_path,
                "fp32": fp32_path,
                "is_f0": is_f0,
                "emb": emb_channels,
            }
        )
        return 45.0

    monkeypatch.setattr(conv_mod, "_fp16_quality_gate", _fake_gate)

    conv_mod.convert_pth_to_onnx(pth, out, fp16=True, trust_pickle=True)

    assert len(export_calls) == 2, (
        f"fp16=True must trigger two _export2onnx calls (main + fp32 ref); got {export_calls}"
    )
    assert export_calls[0][1] is True  # the user's fp16 export
    assert export_calls[1][1] is False  # the fp32 reference
    assert len(gate_calls) == 1
    assert gate_calls[0]["is_f0"] is True
    assert gate_calls[0]["emb"] == 768
    assert gate_calls[0]["fp32"].name.endswith("_fp32_ref.onnx")
    # The fp32 reference must be deleted after the gate.
    assert not gate_calls[0]["fp32"].exists(), (
        "fp32 reference tmp must be cleaned up after the gate runs"
    )


def test_convert_pth_to_onnx_skips_gate_when_fp16_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Back-compat: a fp16=False export does NOT do a second export
    and does NOT call the gate. The gate is opt-in via --fp16."""
    from woys import convert as conv_mod

    pth = tmp_path / "voice.pth"
    pth.write_bytes(b"fake")
    out = tmp_path / "voice.onnx"

    monkeypatch.setattr(conv_mod, "_user_trusts_pickle", lambda _flag: True)
    monkeypatch.setattr(
        conv_mod, "_probe_pth_metadata", lambda *a, **kw: _make_fake_meta(768, True)
    )
    monkeypatch.setattr(conv_mod, "_validate_onnx_loads", lambda *_a, **_kw: None)

    export_calls: list[tuple[str, bool]] = []

    def _fake(inp: str, output: str, simple: str, fp16_arg: bool, _meta: dict[str, Any]) -> None:
        Path(output).write_bytes(b"fake-onnx")
        Path(simple).write_bytes(b"fake-simple")
        export_calls.append((output, fp16_arg))

    import voice_changer.RVC.onnxExporter.export2onnx as ex_mod  # type: ignore[import-not-found]

    monkeypatch.setattr(ex_mod, "_export2onnx", _fake)

    gate_calls: list[Any] = []
    monkeypatch.setattr(
        conv_mod, "_fp16_quality_gate", lambda *a, **kw: gate_calls.append(1) or 99.0
    )

    conv_mod.convert_pth_to_onnx(pth, out, fp16=False, trust_pickle=True)

    assert len(export_calls) == 1, "fp16=False must do exactly one export"
    assert export_calls[0][1] is False
    assert gate_calls == [], "fp16=False must not call the gate"


def test_convert_pth_to_onnx_emits_v1_advisory_for_v1_plus_fp16(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """v1 models (embChannels=256) historically degrade under fp16
    more than v2 -- per the convert_pth_to_onnx docstring. The wiring
    emits an advisory line to stderr naming v1 BEFORE the gate runs,
    so the user knows what to look at if the SNR is poor."""
    from woys import convert as conv_mod

    pth = tmp_path / "v1_voice.pth"
    pth.write_bytes(b"fake")
    out = tmp_path / "v1_voice.onnx"

    monkeypatch.setattr(conv_mod, "_user_trusts_pickle", lambda _flag: True)
    monkeypatch.setattr(
        conv_mod,
        "_probe_pth_metadata",
        lambda *a, **kw: _make_fake_meta(256, True),  # embChannels=256 = v1
    )
    monkeypatch.setattr(conv_mod, "_validate_onnx_loads", lambda *_a, **_kw: None)

    def _fake(inp: str, output: str, simple: str, fp16_arg: bool, _meta: dict[str, Any]) -> None:
        Path(output).write_bytes(b"fake-onnx")
        Path(simple).write_bytes(b"fake-simple")

    import voice_changer.RVC.onnxExporter.export2onnx as ex_mod  # type: ignore[import-not-found]

    monkeypatch.setattr(ex_mod, "_export2onnx", _fake)
    monkeypatch.setattr(conv_mod, "_fp16_quality_gate", lambda *a, **kw: 45.0)

    conv_mod.convert_pth_to_onnx(pth, out, fp16=True, trust_pickle=True)
    err = capsys.readouterr().err

    assert "RVC v1 checkpoint" in err
    assert "embChannels=256" in err


def test_convert_pth_to_onnx_cleans_up_fp32_ref_even_if_gate_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """If the gate itself raises (e.g., ORT can't load one of the
    models, or shape mismatch), the user's fp16 ONNX must NOT be
    lost and the tmp fp32 reference must be cleaned up. The wiring
    catches the exception, warns, and proceeds."""
    from woys import convert as conv_mod

    pth = tmp_path / "voice.pth"
    pth.write_bytes(b"fake")
    out = tmp_path / "voice.onnx"

    monkeypatch.setattr(conv_mod, "_user_trusts_pickle", lambda _flag: True)
    monkeypatch.setattr(
        conv_mod, "_probe_pth_metadata", lambda *a, **kw: _make_fake_meta(768, True)
    )
    monkeypatch.setattr(conv_mod, "_validate_onnx_loads", lambda *_a, **_kw: None)

    def _fake(inp: str, output: str, simple: str, fp16_arg: bool, _meta: dict[str, Any]) -> None:
        Path(output).write_bytes(b"fake-onnx")
        Path(simple).write_bytes(b"fake-simple")

    import voice_changer.RVC.onnxExporter.export2onnx as ex_mod  # type: ignore[import-not-found]

    monkeypatch.setattr(ex_mod, "_export2onnx", _fake)

    def _boom(*_a: Any, **_kw: Any) -> float:
        raise RuntimeError("simulated quality-gate failure")

    monkeypatch.setattr(conv_mod, "_fp16_quality_gate", _boom)

    result = conv_mod.convert_pth_to_onnx(pth, out, fp16=True, trust_pickle=True)
    err = capsys.readouterr().err

    assert result == out
    assert out.exists(), "the user's fp16 ONNX must survive a failed gate"
    fp32_ref = out.with_name(out.stem + "_fp32_ref.onnx")
    assert not fp32_ref.exists(), "the fp32 tmp must be cleaned up"
    assert "WARNING: fp16 quality gate skipped" in err
    assert "simulated quality-gate failure" in err
