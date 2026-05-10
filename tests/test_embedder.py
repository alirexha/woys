"""Embedder coverage.

v0.8.0 deleted the fairseq path; only the ONNX contentvec embedder is
supported. These tests cover:
  1. `OnnxContentvec` (upstream stub now implemented) returns correctly-
     shaped feats for both v1 (256-dim) and v2 (768-dim) paths.
  2. The engine reports `active_embedder == "onnx"` regardless of the
     `embedder` config value (legacy "fairseq" string falls back without
     crashing - see B8 / corr-002).
"""

from __future__ import annotations

from pathlib import Path

import pytest

MODELS_DIR = Path.home() / ".local" / "share" / "woys" / "models"
CONTENTVEC_PATH = MODELS_DIR / "contentvec-f.onnx"


def _have_models() -> bool:
    return CONTENTVEC_PATH.exists()


@pytest.mark.gpu
def test_onnx_contentvec_shapes_v2() -> None:
    """v0.2.0 fills the upstream OnnxContentvec stub. RVC v2 uses the
    layer-12 / 768-dim path."""
    if not _have_models():
        pytest.skip(f"contentvec-f.onnx missing at {CONTENTVEC_PATH}")

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "server"))
    import torch
    from voice_changer.RVC.embedder.OnnxContentvec import (  # type: ignore[import-not-found]
        OnnxContentvec,
    )

    dev = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    emb = OnnxContentvec()
    emb.loadModel(str(CONTENTVEC_PATH), dev)

    # 1 second of silence at 16 kHz, fairseq-style (1, T) shape.
    audio = torch.zeros(1, 16_000, dtype=torch.float32)
    out = emb.extractFeatures(audio, embOutputLayer=12, useFinalProj=False)
    assert out.shape[0] == 1
    assert out.shape[2] == 768, f"expected 768-dim feats, got {out.shape[2]}"


@pytest.mark.gpu
def test_onnx_contentvec_shapes_v1() -> None:
    if not _have_models():
        pytest.skip(f"contentvec-f.onnx missing at {CONTENTVEC_PATH}")

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "server"))
    import torch
    from voice_changer.RVC.embedder.OnnxContentvec import (  # type: ignore[import-not-found]
        OnnxContentvec,
    )

    dev = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    emb = OnnxContentvec()
    emb.loadModel(str(CONTENTVEC_PATH), dev)

    audio = torch.zeros(1, 16_000, dtype=torch.float32)
    out = emb.extractFeatures(audio, embOutputLayer=9, useFinalProj=True)
    assert out.shape[2] == 256, f"expected 256-dim feats, got {out.shape[2]}"


@pytest.mark.gpu
def test_engine_embedder_default_is_onnx() -> None:
    from audio.engine import EngineConfig, RealtimeEngine

    eng = RealtimeEngine(EngineConfig(chunk_seconds=0.25, inference_subprocess=False))
    eng._ensure_sessions()
    assert eng.active_embedder == "onnx"


@pytest.mark.gpu
def test_engine_embedder_legacy_fairseq_value_falls_back_safely() -> None:
    """v0.8.0 dropped the fairseq embedder. Old config.toml files with
    `embedder = "fairseq"` must NOT crash the engine - they fall back to
    ONNX with a logged warning."""
    from audio.engine import EngineConfig, RealtimeEngine

    eng = RealtimeEngine(
        EngineConfig(chunk_seconds=0.25, embedder="fairseq", inference_subprocess=False)
    )
    eng._ensure_sessions()
    assert eng.active_embedder == "onnx"
    assert eng.stats.last_error is not None
    assert "fairseq" in eng.stats.last_error.lower() or "onnx" in eng.stats.last_error.lower()
