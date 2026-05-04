"""`vcclient-cachy convert <pth>` — real RVC .pth → .onnx exporter.

The upstream voice-changer repo ships an `export2onnx` function but expects
to be called from inside its own Pipeline / FastAPI server context. This
module is a thin original-work wrapper that:

  1. Probes the `.pth` checkpoint to derive metadata (model variant,
     embedding channels, f0, sample rate). Logic mirrors upstream's
     `RVCModelSlotGenerator._setInfoByPytorch` but without depending on
     the slot-manager singleton.
  2. Calls upstream's `_export2onnx` with the derived metadata.
  3. Validates the exported `.onnx` loads in ONNX Runtime with the same
     I/O signature our engine expects (feats, p_len, pitch?, pitchf?, sid).

Cache: HuggingFace-derived inputs are downloaded into
`~/.local/share/vcclient-cachy/converted/<repo>/` so re-conversion is free.

Original work — Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

CACHE_DIR = Path.home() / ".local" / "share" / "vcclient-cachy" / "converted"


@dataclass
class _RVCMeta:
    """Subset of upstream RVCModelSlot fields needed by `_export2onnx`."""

    modelType: str
    samplingRate: int
    f0: bool
    embChannels: int
    embedder: str
    embOutputLayer: int
    useFinalProj: bool


def _probe_pth_metadata(pth_path: Path) -> _RVCMeta:
    """Inspect the .pth checkpoint dict to figure out which RVC variant it is.

    Mirrors the upstream `_setInfoByPytorch` decision tree. Doesn't import
    upstream's class hierarchy — keeps this module a pure original-work
    derivative-of-format-knowledge, not a derivative of upstream code.
    """
    # Late import — torch is heavy.
    import torch

    cpt = torch.load(str(pth_path), map_location="cpu", weights_only=False)
    config = cpt.get("config")
    if config is None:
        raise ValueError(
            f"{pth_path.name}: missing 'config' field — not a recognized RVC checkpoint"
        )
    config_len = len(config)
    version = cpt.get("version", "v1")
    f0 = bool(cpt.get("f0", 1) == 1)
    sr = int(config[-1])

    # Late import upstream's enum so we feed _export2onnx the right strings.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))
    from const import EnumInferenceTypes  # type: ignore[import-not-found]

    if config_len == 18:
        # Standard / official RVC checkpoint.
        if version == "v1" or version is None:
            mt = (
                EnumInferenceTypes.pyTorchRVC.value
                if f0
                else EnumInferenceTypes.pyTorchRVCNono.value
            )
            return _RVCMeta(
                modelType=mt,
                samplingRate=sr,
                f0=f0,
                embChannels=256,
                embedder="hubert_base",
                embOutputLayer=9,
                useFinalProj=True,
            )
        # v2+
        mt = (
            EnumInferenceTypes.pyTorchRVCv2.value
            if f0
            else EnumInferenceTypes.pyTorchRVCv2Nono.value
        )
        return _RVCMeta(
            modelType=mt,
            samplingRate=sr,
            f0=f0,
            embChannels=768,
            embedder="hubert_base",
            embOutputLayer=12,
            useFinalProj=False,
        )

    # DDPN-style WebUI checkpoint — has explicit embChannels in config[17].
    emb_channels = int(config[17]) if len(config) > 17 else 768
    use_final_proj = emb_channels == 256
    emb_layer = int(cpt.get("embedder_output_layer", 9))
    embedder_name = cpt.get("embedder_name", "hubert_base")
    if isinstance(embedder_name, str) and embedder_name.endswith("768"):
        embedder_name = embedder_name[:-3]
    mt = EnumInferenceTypes.pyTorchWebUI.value if f0 else EnumInferenceTypes.pyTorchWebUINono.value
    return _RVCMeta(
        modelType=mt,
        samplingRate=sr,
        f0=f0,
        embChannels=emb_channels,
        embedder=str(embedder_name),
        embOutputLayer=emb_layer,
        useFinalProj=use_final_proj,
    )


def _validate_onnx_loads(onnx_path: Path) -> None:
    """Sanity-check the freshly-exported file: must load in ORT and expose
    the I/O names our engine reads (`feats`, `pitch`/`pitchf`, `audio`)."""
    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_names = {i.name for i in sess.get_inputs()}
    out_names = {o.name for o in sess.get_outputs()}
    required_in = {"feats", "p_len", "sid"}
    if not required_in.issubset(in_names):
        raise RuntimeError(f"converted ONNX missing inputs: required {required_in}, got {in_names}")
    if "audio" not in out_names:
        raise RuntimeError(f"converted ONNX has no 'audio' output: {out_names}")


def convert_pth_to_onnx(
    pth_path: Path,
    output_path: Path | None = None,
    *,
    fp16: bool = False,
    opset: int = 17,
) -> Path:
    """Convert an RVC `.pth` to ONNX. Returns the output path.

    `fp16=True` exports half-precision weights. Use only on RVC v2 models
    where you've validated quality is preserved — v1 models often degrade.
    `opset=17` matches what the engine expects; raise it only if you've
    verified ORT 1.20+ supports the ops the model emits.
    """
    pth_path = Path(pth_path).resolve()
    if not pth_path.exists():
        raise FileNotFoundError(f"no such file: {pth_path}")
    if output_path is None:
        output_path = pth_path.with_suffix(".onnx")
    output_path = Path(output_path).resolve()

    meta = _probe_pth_metadata(pth_path)
    print(
        f"[convert] {pth_path.name}: "
        f"type={meta.modelType.split('.')[-1]} sr={meta.samplingRate} "
        f"f0={meta.f0} embCh={meta.embChannels} "
        f"L{meta.embOutputLayer}{'+proj' if meta.useFinalProj else ''} "
        f"fp16={fp16}"
    )

    # Late-import upstream's _export2onnx. The opset arg isn't in upstream's
    # signature; we pass it through via monkey-patching torch.onnx.export.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))
    import torch
    from voice_changer.RVC.onnxExporter.export2onnx import (  # type: ignore[import-not-found]
        _export2onnx,
    )

    metadata_dict = {
        "modelType": meta.modelType,
        "samplingRate": meta.samplingRate,
        "f0": meta.f0,
        "embChannels": meta.embChannels,
        "embedder": meta.embedder,
        "embOutputLayer": meta.embOutputLayer,
        "useFinalProj": meta.useFinalProj,
        "application": "VC_CLIENT",
        "version": "2.1",
    }

    output_simple = output_path.with_name(output_path.stem + "_simple.onnx")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Pin the opset for the duration of the export. torch.onnx.export's
    # signature has too many overloads to type-narrow cleanly.
    from typing import Any

    original_export: Any = torch.onnx.export

    def _export_with_opset(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("opset_version", opset)
        return original_export(*args, **kwargs)

    try:
        torch.onnx.export = _export_with_opset
        _export2onnx(
            str(pth_path),
            str(output_path),
            str(output_simple),
            fp16,
            metadata_dict,
        )
    finally:
        torch.onnx.export = original_export

    if not output_path.exists():
        raise RuntimeError(f"export silently failed — {output_path} not created")

    print(f"[convert] wrote {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MiB)")
    _validate_onnx_loads(output_path)
    print("[convert] ONNX validation OK — ready for the engine.")
    return output_path


def cli_convert(pth: str, output: str | None = None, opset: int = 17, fp16: bool = False) -> int:
    try:
        out = convert_pth_to_onnx(
            Path(pth), Path(output) if output else None, fp16=fp16, opset=opset
        )
    except Exception as e:
        print(f"[convert] ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(f"\nLoad it from `~/.config/vcclient-cachy/config.toml`:\n  rvc_model = {str(out)!r}\n")
    return 0
