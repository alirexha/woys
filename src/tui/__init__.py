"""Terminal UI for woys (Textual).

The package attributes resolve lazily (PEP 562) so `import tui.control` --
all the control-socket client needs -- does not import Textual or the engine.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tui.app import WoysApp, run_tui
    from tui.config import AppConfig, load_config, save_config

    # v0.13.1 - back-compat alias for any external scripts that imported
    # the pre-rename class name. Safe to remove in a future major when
    # no in-the-wild script can still reference VCClientApp.
    VCClientApp = WoysApp

_EXPORTS = {
    "WoysApp": ("tui.app", "WoysApp"),
    "VCClientApp": ("tui.app", "WoysApp"),
    "run_tui": ("tui.app", "run_tui"),
    "AppConfig": ("tui.config", "AppConfig"),
    "load_config": ("tui.config", "load_config"),
    "save_config": ("tui.config", "save_config"),
}

__all__ = ["AppConfig", "VCClientApp", "WoysApp", "load_config", "run_tui", "save_config"]


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attr = target
    return getattr(importlib.import_module(module), attr)
