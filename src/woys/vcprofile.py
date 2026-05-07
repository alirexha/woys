"""`.vcprofile` — shareable voice profile bundles (no model weights).

A `.vcprofile` is just a TOML file with three top-level tables:

  [meta]              — format version, woys version, author hint
  [profile]           — full snapshot of the profile fields (same keys as
                        the `[profiles.<name>]` section in config.toml)
  [model]             — file name (basename) + SHA-256 of the .onnx the
                        profile expects. Importer uses the hash to locate
                        the matching local file in the model library, or to
                        complain if the user is missing it.

Crucially, the profile *does not bundle the model weights* — those are user-
contributed and license-encumbered. Sharing a .vcprofile says "use *that*
voice with *these* settings (pitch, sid, chunks, etc.)"; the receiver still
needs the matching .onnx in their library (verified via SHA-256).

Format version: 1.

Original work — Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import tomli_w

VCPROFILE_VERSION = 1
DEFAULT_EXTENSION = ".vcprofile"


def _ensure_path() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def export_profile(name: str, output: Path, *, config_path: Path | None = None) -> Path:
    """Write the named saved profile (from config.toml) to a .vcprofile file."""
    _ensure_path()
    from tui.config import CONFIG_FILE, load_config
    from woys.profiles import _profiles_bag

    cfg = load_config(config_path or CONFIG_FILE)
    bag = _profiles_bag(cfg)
    if name not in bag:
        raise KeyError(f"no such profile: {name!r}")
    snap = dict(bag[name])

    rvc_path_str = snap.get("rvc_model", "")
    rvc_path = Path(rvc_path_str) if rvc_path_str else None

    model_block: dict[str, Any] = {}
    if rvc_path and rvc_path.exists():
        model_block = {
            "filename": rvc_path.name,
            "sha256": _sha256_file(rvc_path),
            "size_bytes": rvc_path.stat().st_size,
        }
    else:
        model_block = {
            "filename": rvc_path.name if rvc_path else "",
            "sha256": "",
            "size_bytes": 0,
            "missing": True,
        }

    # Drop the absolute path from the on-the-wire profile — the receiver will
    # resolve via the model library.
    snap.pop("rvc_model", None)

    # Late import for version string.
    import woys

    payload = {
        "meta": {
            "format_version": VCPROFILE_VERSION,
            "woys_version": woys.__version__,
            "profile_name": name,
        },
        "profile": snap,
        "model": model_block,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix != DEFAULT_EXTENSION:
        output = output.with_suffix(DEFAULT_EXTENSION)
    with output.open("wb") as f:
        tomli_w.dump(payload, f)
    return output


def import_profile(
    path: Path,
    target_name: str | None = None,
    *,
    config_path: Path | None = None,
    models_dir: Path | None = None,
) -> str:
    """Import a .vcprofile into config.toml.

    If the bundled model SHA-256 matches a file in the local model library
    (resolved via models.discover_models), the imported profile's
    `rvc_model` is set to that path. Otherwise we leave `rvc_model = ""`
    and warn the user.

    Returns the name the profile was saved under.
    """
    _ensure_path()
    import tomllib

    from tui.config import CONFIG_FILE, load_config, save_config
    from woys.models import MODELS_DIR as DEFAULT_MODELS_DIR
    from woys.models import discover_models
    from woys.profiles import save_profile

    # v0.6.8 — guard against malformed .vcprofile so a single bad export
    # file doesn't crash the import flow with a stack trace.
    try:
        with path.open("rb") as f:
            raw = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"{path} is malformed TOML — cannot import. Parse error: {e}") from e
    except OSError as e:
        raise ValueError(f"cannot read {path} ({type(e).__name__}: {e})") from e

    if raw.get("meta", {}).get("format_version") != VCPROFILE_VERSION:
        raise ValueError(
            f"unsupported .vcprofile format_version "
            f"{raw.get('meta', {}).get('format_version')!r}; this build expects "
            f"v{VCPROFILE_VERSION}"
        )

    snap_in = dict(raw.get("profile", {}))
    model_meta = raw.get("model", {})
    desired_sha = model_meta.get("sha256", "")
    desired_name = model_meta.get("filename", "")

    actual_cfg_path = config_path or CONFIG_FILE
    actual_models_dir = models_dir or DEFAULT_MODELS_DIR
    cfg = load_config(actual_cfg_path)
    name = target_name or raw.get("meta", {}).get("profile_name") or path.stem
    if not name:
        raise ValueError(".vcprofile has no profile name and none was provided")

    # Try to bind the .vcprofile's model expectation to a local file.
    if desired_sha:
        match = None
        for entry in discover_models(actual_models_dir):
            try:
                if _sha256_file(entry.path) == desired_sha:
                    match = entry.path
                    break
            except OSError:
                continue
        if match is not None:
            snap_in["rvc_model"] = str(match.resolve())
        else:
            print(
                f"  [warn] no local model matches sha256 "
                f"{desired_sha[:12]}…  (expected: {desired_name!r})"
            )
            print(
                "        download or convert it before applying this profile.\n"
                "        for now the profile will be saved with rvc_model unset."
            )
            snap_in["rvc_model"] = ""
    else:
        snap_in["rvc_model"] = ""

    # B30 / corr-014: build the snapshot directly. The pre-v0.8.0 code
    # called `save_profile(cfg, name)` first (which snapshotted the
    # CURRENT cfg, not snap_in's data) and then immediately overwrote
    # the result — dead code with a side effect that left an
    # intermediate-state profile in `cfg._extras` for the millisecond
    # between the save_profile and the bag overwrite.
    from tui.config import AppConfig

    tmp_cfg = AppConfig()
    for k, v in snap_in.items():
        if hasattr(tmp_cfg, k):
            setattr(tmp_cfg, k, v)

    # B29 / corr-013: use the `_profiles_bag` helper so a corrupt
    # `_extras["profiles"]` (non-dict, e.g. user hand-edited config to
    # `profiles = "broken"`) coerces to {} instead of raising mid-save.
    from woys.profiles import _PROFILE_FIELDS, _profiles_bag

    bag = dict(_profiles_bag(cfg))
    full_snap: dict[str, Any] = {}
    for k in _PROFILE_FIELDS:
        if k in snap_in:
            full_snap[k] = snap_in[k]
        elif hasattr(tmp_cfg, k):
            full_snap[k] = getattr(tmp_cfg, k)
    bag[name] = full_snap
    cfg._extras["profiles"] = bag
    save_config(cfg, actual_cfg_path)
    return name


# CLI handlers ------------------------------------------------------------------


def cli_profile_export(name: str, output: str) -> int:
    out_path = Path(output).expanduser()
    try:
        wrote = export_profile(name, out_path)
    except Exception as e:
        print(f"[profile export] ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(f"[profile export] wrote {wrote}")
    print(f"  share this file to give someone your `{name}` voice settings.")
    print("  the recipient will need a local .onnx with the same SHA-256.")
    return 0


def cli_profile_import(path: str, name: str | None = None) -> int:
    in_path = Path(path).expanduser()
    if not in_path.exists():
        print(f"[profile import] ERROR: no such file: {in_path}", file=sys.stderr)
        return 1
    try:
        new_name = import_profile(in_path, name)
    except Exception as e:
        print(f"[profile import] ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(f"[profile import] saved profile {new_name!r}.")
    print(f"  apply with: woys profile use {new_name}")
    return 0
