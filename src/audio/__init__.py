"""PipeWire audio integration for woys.

The package attributes resolve lazily (PEP 562): `from audio import
RealtimeEngine` still works, but importing a light submodule such as
`audio.pipewire` no longer drags in `audio.engine` -- numpy, onnxruntime and
its CUDA/cuDNN preload -- which the short control-socket commands never use.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from audio.engine import EngineStats, RealtimeEngine
    from audio.engine_config import EngineConfig
    from audio.pipewire import (
        PipeWireError,
        VirtualMic,
        VirtualMicState,
        ensure_pipewire,
        get_state,
    )

_EXPORTS = {
    "EngineConfig": "audio.engine_config",
    "EngineStats": "audio.engine",
    "RealtimeEngine": "audio.engine",
    "PipeWireError": "audio.pipewire",
    "VirtualMic": "audio.pipewire",
    "VirtualMicState": "audio.pipewire",
    "ensure_pipewire": "audio.pipewire",
    "get_state": "audio.pipewire",
}

__all__ = [
    "EngineConfig",
    "EngineStats",
    "PipeWireError",
    "RealtimeEngine",
    "VirtualMic",
    "VirtualMicState",
    "ensure_pipewire",
    "get_state",
]


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)
