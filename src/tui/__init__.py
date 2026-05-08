"""Terminal UI for woys (Textual)."""

from tui.app import WoysApp, run_tui

# v0.13.1 — back-compat alias for any external scripts that imported
# the pre-rename class name. Safe to remove in a future major when
# no in-the-wild script can still reference VCClientApp.
VCClientApp = WoysApp
from tui.config import AppConfig, load_config, save_config

__all__ = ["AppConfig", "VCClientApp", "WoysApp", "load_config", "run_tui", "save_config"]
