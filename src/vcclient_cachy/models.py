"""Model-library management — list, download, and set the active RVC voice model.

Backing store
-------------
Models live in `~/.local/share/vcclient-cachy/models/`. Anything matching
`*.onnx` (and not already a foundation file) is treated as an RVC voice.
Foundation files (contentvec, rmvpe, hubert) are filtered out by name.

Hugging Face download
---------------------
`vcclient-cachy models download <repo>` uses `huggingface_hub`'s snapshot
API to fetch all `.onnx` (and any `.index`) files from a repo into the
cache. Re-runs are free thanks to HF's content-addressable cache.

Original work — Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

MODELS_DIR = Path.home() / ".local" / "share" / "vcclient-cachy" / "models"

# Foundation files — these are infrastructure, not user voices. Hide from `list`
# and skip when picking a default `use` target.
FOUNDATION_NAMES = frozenset(
    {
        "rmvpe.onnx",
        "rmvpe-fp16.onnx",
        "rmvpe_wrapped.onnx",
        "rmvpe_wrapped-fp16.onnx",
        "contentvec-f.onnx",
        "contentvec-f-fp16.onnx",
        "hubert_base.onnx",
    }
)


@dataclass
class ModelEntry:
    name: str  # display name (file stem)
    path: Path
    size_mib: float
    sample_rate: int | None  # None when probe fails
    is_v2: bool | None  # None when probe fails
    f0: bool | None  # None when probe fails


def _probe_onnx(path: Path) -> tuple[int | None, bool | None, bool | None]:
    """Cheap ORT-load to read input/output shapes. Returns (sr, is_v2, f0).
    All Nones if the file isn't a recognized RVC voice ONNX."""
    try:
        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        in_names = {i.name: i for i in sess.get_inputs()}
        if "feats" not in in_names:
            return None, None, None
        feats = in_names["feats"]
        # feats shape is [1, dynamic, embChannels].
        try:
            emb_ch = int(feats.shape[2]) if len(feats.shape) >= 3 else None
        except (TypeError, ValueError):
            emb_ch = None
        is_v2 = emb_ch == 768 if emb_ch is not None else None
        f0 = "pitch" in in_names and "pitchf" in in_names
        # Sample rate is not directly probable from the ONNX file; it lives in
        # custom metadata if the upstream exporter populated it.
        sr: int | None = None
        meta = sess.get_modelmeta()
        if meta and meta.custom_metadata_map:
            try:
                import json

                if "metadata" in meta.custom_metadata_map:
                    sr_meta = json.loads(meta.custom_metadata_map["metadata"]).get("samplingRate")
                    if isinstance(sr_meta, int):
                        sr = sr_meta
            except (ValueError, KeyError, json.JSONDecodeError):
                pass
        return sr, is_v2, f0
    except Exception:
        return None, None, None


def discover_models(models_dir: Path = MODELS_DIR) -> list[ModelEntry]:
    """Enumerate .onnx files that look like RVC voice models."""
    if not models_dir.exists():
        return []
    out: list[ModelEntry] = []
    for path in sorted(models_dir.glob("*.onnx")):
        if path.name in FOUNDATION_NAMES:
            continue
        size_mib = path.stat().st_size / (1024 * 1024)
        sr, is_v2, f0 = _probe_onnx(path)
        out.append(
            ModelEntry(
                name=path.stem,
                path=path,
                size_mib=size_mib,
                sample_rate=sr,
                is_v2=is_v2,
                f0=f0,
            )
        )
    return out


def find_by_name(name: str, models_dir: Path = MODELS_DIR) -> Path | None:
    """Resolve `name` against the model library. Match either the file stem
    (e.g. `amitaro_v2_16k`) or the bare filename (with or without .onnx)."""
    target_stem = name[:-5] if name.endswith(".onnx") else name
    for entry in discover_models(models_dir):
        if entry.name == target_stem:
            return entry.path
    # Fallback: treat the input as a literal path.
    p = Path(name).expanduser()
    if p.exists() and p.suffix == ".onnx":
        return p
    return None


def download_repo(repo: str, models_dir: Path = MODELS_DIR) -> list[Path]:
    """Snapshot-download all `.onnx` (and `.index`) files from a HF repo.

    Returns the list of downloaded files copied into `models_dir`.
    """
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as e:
        raise RuntimeError("huggingface_hub missing — `uv pip install huggingface_hub`") from e

    api = HfApi()
    info = api.repo_info(repo)
    siblings = getattr(info, "siblings", None) or []
    entries = [s.rfilename for s in siblings if s.rfilename.endswith((".onnx", ".index"))]
    if not entries:
        raise RuntimeError(f"no .onnx / .index files found in {repo}")

    models_dir.mkdir(parents=True, exist_ok=True)
    landed: list[Path] = []
    for rfn in entries:
        cached = hf_hub_download(repo_id=repo, filename=rfn)
        # Drop into models_dir under a sensible local name (basename only,
        # de-conflict with prefix if needed).
        base = Path(rfn).name
        local = models_dir / base
        if local.exists() and local.stat().st_size == Path(cached).stat().st_size:
            print(f"  [skip] {local.name} already present")
            landed.append(local)
            continue
        # Use a hard link if the cache is on the same filesystem; copy otherwise.
        try:
            local.unlink(missing_ok=True)
            local.hardlink_to(cached)
        except OSError:
            import shutil

            shutil.copy2(cached, local)
        size_mib = local.stat().st_size / (1024 * 1024)
        print(f"  [get ] {local.name}  ({size_mib:.1f} MiB)")
        landed.append(local)
    return landed


def cli_models_list(models_dir: Path = MODELS_DIR) -> int:
    entries = discover_models(models_dir)
    active = _read_active_model()
    if not entries:
        print(f"no RVC voice models found under {models_dir}")
        print("hint: drop a .onnx in there, or run `vcclient-cachy models download <hf-repo>`")
        return 0
    print(f"{'  ':2}{'name':32s} {'size':>9s} {'sr':>6s} {'v':>3s} {'f0':>3s}")
    for e in entries:
        marker = " *" if active and active.name == e.path.name else "  "
        sr = f"{e.sample_rate}" if e.sample_rate else "?"
        ver = "v2" if e.is_v2 else "v1" if e.is_v2 is False else "?"
        f0 = "y" if e.f0 else "n" if e.f0 is False else "?"
        print(f"{marker}{e.name:32s} {e.size_mib:>7.1f}M {sr:>6s} {ver:>3s} {f0:>3s}")
    return 0


def _read_active_model() -> Path | None:
    try:
        from tui.config import load_config
    except ImportError:
        # Tests / standalone runs may not have src/tui on path.
        repo_root = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(repo_root))
        from tui.config import load_config
    cfg = load_config()
    if cfg.rvc_model:
        p = Path(cfg.rvc_model)
        return p if p.exists() else None
    return None


def cli_models_download(repo: str, models_dir: Path = MODELS_DIR) -> int:
    print(f"models download from huggingface.co/{repo}:")
    try:
        files = download_repo(repo, models_dir)
    except Exception as e:
        print(f"[models] ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(f"  done — {len(files)} file(s) in {models_dir}")
    return 0


def cli_models_use(name: str, models_dir: Path = MODELS_DIR) -> int:
    """Set the active RVC model. v0.4.1: hot-swap if the engine is running,
    otherwise write config and let the next start pick it up."""
    path = find_by_name(name, models_dir)
    if path is None:
        print(f"[models] no such model: {name!r}", file=sys.stderr)
        print("  use `vcclient-cachy models list` to see what's available", file=sys.stderr)
        return 1

    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))
    from tui.config import load_config, save_config
    from tui.control import send_command

    # 1. Try to hot-swap a running engine first. The TUI's MODEL handler
    # also persists config, so on success we're fully done.
    socket_reply = send_command(f"MODEL {path.stem}", timeout=2.0)
    if socket_reply.startswith("OK"):
        print(f"[models] hot-swapped → {path.name}")
        print(f"         engine reply: {socket_reply}")
        return 0

    # 2. Engine not running — write config so the next `vcclient-cachy run`
    # picks it up. No more "restart the engine" messaging since the engine
    # would, in fact, see this on the next boot.
    cfg = load_config()
    cfg.rvc_model = str(path.resolve())
    save_config(cfg)
    print(f"[models] config updated → {path.name}")
    print("         (engine not running; the next `vcclient-cachy run` will load it)")
    return 0
