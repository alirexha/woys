"""Model-library management — list, download, and set the active RVC voice model.

Backing store
-------------
Models live in `~/.local/share/woys/models/`. Anything matching
`*.onnx` (and not already a foundation file) is treated as an RVC voice.
Foundation files (contentvec, rmvpe, hubert) are filtered out by name.

Hugging Face download
---------------------
`woys models download <repo>` uses `huggingface_hub`'s snapshot
API to fetch all `.onnx` (and any `.index`) files from a repo into the
cache. Re-runs are free thanks to HF's content-addressable cache.

Original work — Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

MODELS_DIR = Path.home() / ".local" / "share" / "woys" / "models"

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
    # B25 / sec-009: build a name → expected SHA map from the HF API. lfs files
    # have a `lfs` blob with `sha256`; small (non-lfs) files surface their git
    # blob_id in `blob_id` (which is *not* SHA256 but *is* a stable content
    # hash that detects mid-flight tampering). We verify lfs SHAs hard, treat
    # blob_id as informational.
    lfs_sha_by_name: dict[str, str] = {}
    for s in siblings:
        lfs_obj = getattr(s, "lfs", None)
        if lfs_obj is not None:
            sha = getattr(lfs_obj, "sha256", None)
            if isinstance(sha, str) and len(sha) == 64:
                lfs_sha_by_name[s.rfilename] = sha
    entries = [s.rfilename for s in siblings if s.rfilename.endswith((".onnx", ".index"))]
    if not entries:
        raise RuntimeError(f"no .onnx / .index files found in {repo}")

    models_dir.mkdir(parents=True, exist_ok=True)
    landed: list[Path] = []
    for rfn in entries:
        cached = hf_hub_download(repo_id=repo, filename=rfn)
        # B25: verify SHA256 of the cache file against the HF API's reported
        # value. If the cache was tampered with after hf_hub_download wrote
        # it, this catches it; legitimate upstream rehashes invalidate the
        # local cache (user re-downloads via `--force` of huggingface_hub).
        expected = lfs_sha_by_name.get(rfn)
        if expected is not None:
            actual = _sha256_of(Path(cached))
            if actual != expected:
                raise RuntimeError(
                    f"SHA256 mismatch for {rfn} from {repo}:\n"
                    f"  expected {expected}\n"
                    f"  actual   {actual}\n"
                    f"Refusing to install. Try `huggingface-cli download "
                    f"{repo} --force-download` to refresh the cache."
                )
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
        verified = " [sha256 verified]" if expected else ""
        print(f"  [get ] {local.name}  ({size_mib:.1f} MiB){verified}")
        landed.append(local)
    return landed


def _sha256_of(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cli_models_list(models_dir: Path = MODELS_DIR) -> int:
    entries = discover_models(models_dir)
    active = _read_active_model()
    if not entries:
        print(f"no RVC voice models found under {models_dir}")
        print("hint: drop a .onnx in there, or run `woys models download <hf-repo>`")
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
    """Set the active RVC model.

    v0.5.0: uses the async JOB protocol — `MODEL` returns a job id, we
    poll `JOB <id>` until the engine worker actually picked up the swap.
    Default overall timeout 30 s (covers cold-cache cudnn-tune of any
    voice; cached swaps complete in < 200 ms).
    """
    path = find_by_name(name, models_dir)
    if path is None:
        print(f"[models] no such model: {name!r}", file=sys.stderr)
        print("  use `woys models list` to see what's available", file=sys.stderr)
        return 1

    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))
    from tui.config import load_config, save_config
    from tui.control import submit_and_wait

    # Try the running engine first. submit_and_wait handles the JOB poll.
    reply = submit_and_wait(f"MODEL {path.stem}", overall_timeout=30.0)
    if reply.startswith("OK") and " state=done" in reply:
        # Pull the elapsed_ms field for a friendlier print.
        ms = "?"
        for tok in reply.split():
            if tok.startswith("elapsed_ms="):
                ms = tok.split("=", 1)[1]
                break
        print(f"[models] hot-swapped → {path.name}  ({ms} ms)")
        return 0
    if reply.startswith("OK") and "state=error" in reply:
        print(f"[models] swap failed: {reply}", file=sys.stderr)
        return 1
    if reply.startswith("OK") and "job=" not in reply:
        # Old synchronous handler — shouldn't happen post-v0.5.0 but keep
        # backward compat for the rare mixed-version scenario.
        print(f"[models] hot-swapped → {path.name}")
        return 0
    if reply.startswith("ERR control socket not found"):
        # Engine not running — write config so the next `woys run`
        # picks it up.
        cfg = load_config()
        cfg.rvc_model = str(path.resolve())
        save_config(cfg)
        print(f"[models] config updated → {path.name}")
        print("         (engine not running; the next `woys run` will load it)")
        return 0
    print(f"[models] swap reply: {reply}", file=sys.stderr)
    return 1
