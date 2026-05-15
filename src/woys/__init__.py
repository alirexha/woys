"""woys: Linux-native real-time voice changer (RVC-only on ONNX Runtime CUDA).

Single source of truth for the project version. Hatchling reads `__version__`
out of this file at build time (`[tool.hatch.version]` in pyproject.toml);
everything else that needs the version (CLI banner, `--version`, `woys info`)
imports it from here. Bump the literal below, then run
`python scripts/release.py` to propagate it to README/PROGRESS/PKGBUILD.
"""

__version__ = "0.14.3"
