"""`.vcprofile` - shareable voice profile bundles (no model weights).

A `.vcprofile` is just a TOML file with three top-level tables:

  [meta]              - format version, woys version, author hint
  [profile]           - full snapshot of the profile fields (same keys as
                        the `[profiles.<name>]` section in config.toml)
  [model]             - file name (basename) + SHA-256 of the .onnx the
                        profile expects. Importer uses the hash to locate
                        the matching local file in the model library, or to
                        complain if the user is missing it.

Crucially, the profile *does not bundle the model weights* - those are user-
contributed and license-encumbered. Sharing a .vcprofile says "use *that*
voice with *these* settings (pitch, sid, chunks, etc.)"; the receiver still
needs the matching .onnx in their library (verified via SHA-256).

Format version: 1.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import tomli_w

VCPROFILE_VERSION = 1
DEFAULT_EXTENSION = ".vcprofile"

# review F-16-08: per-version migration ladder for `.vcprofile`
# import. Each entry migrates a `raw` TOML dict from key K to K+1.
# Pre-fix `import_profile` raised on any version mismatch -- a share/
# interop format breaking hard on its first revision. The reader now
# walks `meta.format_version + 1 .. VCPROFILE_VERSION` and applies
# each registered leg with a stderr warning. A missing leg raises a
# specific "no migration registered for vN -> vN+1" error so a future
# author knows exactly which entry to add. Receiver-side files newer
# than `VCPROFILE_VERSION` raise an "upgrade woys" message.
#
# The ladder is empty today (v1 is the only released format). The
# DELIVERABLE for this commit is the mechanism; entries land alongside
# real format revisions.
_VCPROFILE_MIGRATIONS: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {}


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


def _migrate_vcprofile_raw(
    raw: dict[str, Any],
    *,
    source: Path | None = None,
) -> dict[str, Any]:
    """Walk the `.vcprofile` format-version migration ladder.

    review F-16-08: pre-fix `import_profile` did exact-equality
    on `meta.format_version` and raised on any mismatch. A share/
    interop format whose whole purpose is cross-user/cross-version
    distribution cannot fail-hard on its first revision. The new
    contract:

    - `format_version > VCPROFILE_VERSION`: refuse with a clear
      "upgrade woys" message. We cannot safely interpret a future
      schema (a key could have been renamed, a value semantically
      reinterpreted) -- the only honest response is "this file was
      exported by a newer build; upgrade".
    - `format_version == VCPROFILE_VERSION`: pass through unchanged.
    - `format_version < VCPROFILE_VERSION`: walk
      `[version + 1 .. VCPROFILE_VERSION]` and apply each registered
      migration leg from `_VCPROFILE_MIGRATIONS`. Print a stderr
      warning per leg so the user knows their import migrated. If a
      leg is missing, raise -- a future maintainer adding format vN
      must register the migration for vN-1 -> vN at the same time.
    - missing `meta.format_version` or non-integer value: raise
      with a clear message (a malformed .vcprofile is not a version
      mismatch; we cannot guess which era the file came from).

    The `source` arg is just used in error messages so the user knows
    which file they passed.
    """
    where = f" ({source})" if source is not None else ""
    meta = raw.get("meta", {})
    raw_version: object = meta.get("format_version") if isinstance(meta, dict) else None
    if not isinstance(raw_version, int) or isinstance(raw_version, bool):
        raise ValueError(
            f".vcprofile{where} has missing or non-integer "
            f"meta.format_version (got {raw_version!r}); cannot determine "
            f"which schema era it was exported from. Expected an integer "
            f"between 1 and {VCPROFILE_VERSION}."
        )
    if raw_version > VCPROFILE_VERSION:
        raise ValueError(
            f".vcprofile{where} was exported by a newer build "
            f"(format_version={raw_version}); this woys understands up to "
            f"v{VCPROFILE_VERSION}. Upgrade woys to import this file."
        )
    if raw_version == VCPROFILE_VERSION:
        return raw
    # raw_version < VCPROFILE_VERSION: walk the ladder.
    migrated = raw
    for step in range(raw_version, VCPROFILE_VERSION):
        next_step = step + 1
        leg = _VCPROFILE_MIGRATIONS.get(step)
        if leg is None:
            raise ValueError(
                f".vcprofile{where} format_version={raw_version} requires "
                f"a v{step} -> v{next_step} migration that is not registered "
                f"in `_VCPROFILE_MIGRATIONS`. Re-export from a current woys, "
                f"or file an issue."
            )
        print(
            f"[profile import] migrating v{step} -> v{next_step}{where}",
            file=sys.stderr,
        )
        migrated = leg(migrated)
    # Stamp the new version onto the migrated copy so downstream code
    # treats it as the current schema.
    meta_out = dict(migrated.get("meta", {}))
    meta_out["format_version"] = VCPROFILE_VERSION
    migrated["meta"] = meta_out
    return migrated


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

    # Drop the absolute path from the on-the-wire profile - the receiver will
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

    # v0.6.8 - guard against malformed .vcprofile so a single bad export
    # file doesn't crash the import flow with a stack trace.
    try:
        with path.open("rb") as f:
            raw = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"{path} is malformed TOML - cannot import. Parse error: {e}") from e
    except OSError as e:
        raise ValueError(f"cannot read {path} ({type(e).__name__}: {e})") from e

    raw = _migrate_vcprofile_raw(raw, source=path)

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
    # the result - dead code with a side effect that left an
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
