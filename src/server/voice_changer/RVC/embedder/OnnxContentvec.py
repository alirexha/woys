"""ONNX implementation of the ContentVec embedder.

Upstream shipped this as a stub (`raise Exception("Not implemented")`); this
fork's v0.2.0 fills it in. The corresponding ONNX file is the same
`contentvec-f.onnx` that `EmbedderManager` already references via
`params.content_vec_500_onnx`.

Layer/projection mapping mirrors the FairseqHubert path:
  - embOutputLayer=9,  useFinalProj=True  → ContentVec layer-9 with final projection
                                            (256-dim, used by RVC v1)
  - embOutputLayer=12, useFinalProj=False → ContentVec layer-12 raw   (768-dim,
                                            used by RVC v2)

The contentvec-f.onnx model exposes three outputs:
  * `units9`  shape (1, T', 256)
  * `unit12`  shape (1, T', 768)
  * `unit12s` shape (1, T', 768)  (stop-grad version of unit12)

Inputs/outputs cross host↔device exactly twice per call (numpy in, numpy out),
so this is **not** the lowest-overhead path possible — Phase 5+ IO-binding
work would shave another ~3-5 ms — but it's correct and matches the tensor
shapes the rest of the upstream pipeline expects.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import onnxruntime as ort
import torch

from voice_changer.RVC.embedder.Embedder import Embedder

# ORT-GPU 1.20+ on driver 595 needs the pip-shipped CUDA libs preloaded
# explicitly. Idempotent, no-op on older ORT versions.
if hasattr(ort, "preload_dlls"):
    ort.preload_dlls()


def _select_providers(dev: torch.device) -> list[Any]:
    available = ort.get_available_providers()
    providers: list[Any] = []
    if dev.type == "cuda" and "CUDAExecutionProvider" in available:
        providers.append(
            (
                "CUDAExecutionProvider",
                {
                    "device_id": dev.index if dev.index is not None else 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                },
            )
        )
    providers.append("CPUExecutionProvider")
    return providers


class OnnxContentvec(Embedder):
    def loadModel(self, file: str, dev: torch.device) -> Embedder:
        # half precision is determined by the ONNX file itself; the abstract
        # Embedder API tracks isHalf as a hint to callers.
        super().setProps("hubert_base", file, dev, isHalf=False)

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        self.model: ort.InferenceSession = ort.InferenceSession(  # type: ignore[assignment]
            file, sess_options=so, providers=_select_providers(dev)
        )

        outputs = {o.name for o in self.model.get_outputs()}
        # All known contentvec-f exports we target ship with both names.
        if not ({"units9", "unit12"} & outputs):
            raise RuntimeError(
                f"contentvec ONNX at {file} has unexpected outputs {outputs}; "
                "expected at least one of 'units9' / 'unit12'"
            )
        return self

    def extractFeatures(
        self,
        feats: torch.Tensor,
        embOutputLayer: int = 9,
        useFinalProj: bool = True,
    ) -> torch.Tensor:
        # Map (layer, useFinalProj) → ONNX output name. Anything else falls
        # back to unit12 (the most common RVC v2 path).
        if embOutputLayer == 9 and useFinalProj:
            target = "units9"
        elif embOutputLayer == 12 and not useFinalProj:
            target = "unit12"
        else:
            target = "unit12"

        if feats.ndim == 3:
            # FairseqHubert receives (1, 1, T) — squeeze the channel axis.
            feats = feats.squeeze(1)

        audio = feats.detach().to(torch.float32).cpu().numpy()
        if audio.ndim == 1:
            audio = audio.reshape(1, -1)

        out_np = self.model.run([target], {"audio": audio.astype(np.float32)})[0]
        return torch.from_numpy(out_np).to(self.dev)
