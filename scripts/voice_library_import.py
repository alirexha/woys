"""Batch-import a starter voice library — see VOICE_LIBRARY_BRIEF.md.

Driver for the 9-model batch import. Runs each model through:
  download → verify zip → extract → find .pth → convert → validate inference
  → register profile.

Outcomes are printed at the end as a table. /tmp/vcclib-stage is cleaned up
only if every model in the batch validated; otherwise it stays for diagnosis.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))

STAGE_DIR = Path("/tmp/vcclib-stage")
MODELS_DIR = Path.home() / ".local" / "share" / "woys" / "models"
SOURCES_MD = REPO_ROOT / "voice-library" / "SOURCES.md"
BLOCKERS_MD = REPO_ROOT / "BLOCKERS.md"


@dataclass
class Voice:
    slug: str
    display: str
    url: str
    note: str = ""


VOICES: list[Voice] = [
    Voice(
        "donald_trump",
        "Donald Trump (POTUS)",
        "https://huggingface.co/Hazza1/DonaldTrump/resolve/main/Trump.zip",
    ),
    Voice(
        "e_girl",
        "E-Girl (HQ Female)",
        "https://huggingface.co/ZokaxDesu/e-girl/resolve/main/e-girl.zip",
    ),
    Voice(
        "alfred_pennyworth",
        "Alfred Pennyworth (Arkham)",
        "https://huggingface.co/Homiebear/AlfredPennyworth_465e_8835s/resolve/main/AlfredPennyworth_465e_8835s.zip",
    ),
    Voice(
        "lana_del_rey",
        "Lana Del Rey (NFR Era)",
        "https://huggingface.co/pinguG/Lana-Del-Rey/resolve/main/LanaDelReyV2.zip",
    ),
    Voice(
        "harley_quinn",
        "Harley Quinn V2 (Enemy Within, Titan Pretrain)",
        "https://huggingface.co/Cauthess/HarleyQuinnTitanPretrain/resolve/main/Harley%20Quinn%20Version%202%20-%20Enemy%20Within.zip",
        note="Titan pretrain — may fail to convert. 30 min cap, then skip.",
    ),
    Voice(
        "catwoman",
        "Catwoman (Laura Bailey)",
        "https://huggingface.co/Cauthess/CatwomanLauraBailey/resolve/main/Catwoman%20-%20Laura%20Bailey.zip",
    ),
    Voice(
        "megan_fox",
        "Megan Fox",
        "https://huggingface.co/dragoncrack/https___www_donationalerts_com_r_crack_dragon/resolve/main/MeganFox.zip",
        note="Sketchy uploader repo name — verify quality, log if garbage.",
    ),
    Voice(
        "batman_troy_baker",
        "Batman / Bruce Wayne (Troy Baker, Telltale)",
        "https://huggingface.co/Zogii/zogiiRVC/resolve/main/Bruce%20Wayne%20(Troy%20Baker)%20Batman%20The%20Telltale%20Series%20(RVC%20v2)%20400%20Epochs.zip",
    ),
    Voice(
        "spongebob_persian",
        "SpongeBob Persian Dub (Bab Asfanji)",
        "https://huggingface.co/PlushymehereJC/Spongebob_Persian_dub/resolve/main/Bab_Asfanj.zip",
        note="Trained on Farsi audio; works best when speaking Persian.",
    ),
]


@dataclass
class Outcome:
    voice: Voice
    status: str  # "ok" | "skipped"
    reason: str = ""
    onnx_path: Path | None = None
    onnx_size_mib: float = 0.0


def _log_blocker(line: str) -> None:
    BLOCKERS_MD.parent.mkdir(parents=True, exist_ok=True)
    if not BLOCKERS_MD.exists():
        BLOCKERS_MD.write_text("# Blockers\n\n", encoding="utf-8")
    with BLOCKERS_MD.open("a", encoding="utf-8") as f:
        f.write(f"- {line}\n")


def _download(url: str, dest: Path, *, timeout: int = 120) -> bool:
    """curl with retry + continue-from-partial. Returns True on success."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1024:
        print(f"  [skip] {dest.name} already on disk ({dest.stat().st_size / 1024 / 1024:.1f} MiB)")
        return True
    for attempt in (1, 2):
        try:
            cmd = [
                "curl", "-L", "--fail", "-o", str(dest), "--continue-at", "-",
                "--max-time", str(timeout), url,
            ]  # fmt: skip
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 30)
            if r.returncode == 0 and dest.exists() and dest.stat().st_size > 1024:
                print(f"  [get ] {dest.name} ({dest.stat().st_size / 1024 / 1024:.1f} MiB)")
                return True
            print(f"  [retry {attempt}] curl exit={r.returncode}: {r.stderr.strip()[:160]}")
        except subprocess.TimeoutExpired:
            print(f"  [retry {attempt}] curl timeout")
    return False


def _verify_zip(path: Path) -> bool:
    r = subprocess.run(["unzip", "-tq", str(path)], capture_output=True, text=True)
    return r.returncode == 0


def _extract_zip(path: Path, dest: Path) -> bool:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    r = subprocess.run(["unzip", "-q", str(path), "-d", str(dest)], capture_output=True, text=True)
    return r.returncode == 0


def _find_pth(extracted: Path) -> Path | None:
    """Pick the most likely RVC voice .pth from the extract tree.

    Heuristics: prefer files NOT prefixed with `D_` or `G_` (those are
    discriminator / generator training checkpoints). Among the remainder,
    pick the smallest (released RVC weights are typically 50-60 MB; the
    full training checkpoint can be 200+ MB).
    """
    candidates = list(extracted.rglob("*.pth"))
    voice_pths = [p for p in candidates if not (p.name.startswith("D_") or p.name.startswith("G_"))]
    if not voice_pths:
        return None
    # Prefer non-`G_`/`D_`. Among them, prefer the one with the *smallest* size
    # (the released weight is typically smaller than D_/G_).
    voice_pths.sort(key=lambda p: p.stat().st_size)
    return voice_pths[0]


def _find_index(extracted: Path) -> Path | None:
    """Optional .index file (improves quality, not strictly required)."""
    indices = list(extracted.rglob("added_*.index"))
    if not indices:
        indices = list(extracted.rglob("*.index"))
    return indices[0] if indices else None


def _convert(pth: Path, output_onnx: Path) -> bool:
    """Run `woys convert` as a subprocess so we exercise the same
    code path the user would. fp32 only — fp16 only auto-promoted by
    `fp16-convert` for foundations, and contentvec fp16 quality concerns
    (LESSONS §8) make voice-model fp16 a per-user opt-in too."""
    venv_bin = REPO_ROOT / ".venv" / "bin" / "woys"
    cmd = [str(venv_bin), "convert", str(pth), "-o", str(output_onnx)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        print(f"  [convert FAIL] exit={r.returncode}")
        if r.stderr:
            print(f"    stderr (last 240 chars): {r.stderr[-240:]}")
        return False
    return output_onnx.exists() and output_onnx.stat().st_size > 1024 * 1024


def _validate_inference(onnx_path: Path) -> tuple[bool, str]:
    """Load the converted .onnx through the engine, run a 1 s silent buffer.
    Returns (ok, reason_if_not_ok)."""
    try:
        import numpy as np

        from audio.engine import EngineConfig, RealtimeEngine
    except Exception as e:
        return False, f"import error: {type(e).__name__}: {e}"

    try:
        eng = RealtimeEngine(EngineConfig(chunk_seconds=0.1, rvc_model=onnx_path))
        eng._ensure_sessions()
        # Process_chunk_16k expects (N,) float32 audio at 16 kHz.
        silent = np.zeros(16_000, dtype=np.float32)
        out = eng.process_chunk_16k(silent)
        if out.size == 0:
            return False, "empty output"
        if not np.isfinite(out).all():
            return False, "NaN/Inf in output"
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _register_profile(slug: str, onnx_path: Path, display: str, source_url: str, note: str) -> None:
    """Save a profile with sensible defaults (pitch=0, chunk=0.25, monitor=False).

    Stashes the display name + source URL into the profile's snapshot via
    `_extras` so future tooling can surface provenance.

    v0.5.1: chunk_seconds default raised 0.1 → 0.25. The 0.1 default was
    driving short-chunk SOLA tail trims (~10 % output duration loss) on
    higher-SR voices. 0.25 produces output within ~1 % of input duration
    and stays well under the 100 ms latency target. See
    `docs/07-audio-quality-bug.md`.
    """
    from tui.config import load_config, save_config
    from woys.profiles import save_profile

    cfg = load_config()
    cfg.rvc_model = str(onnx_path.resolve())
    cfg.f0_up_key = 0
    cfg.chunk_seconds = 0.25
    cfg.input_gain_db = 0.0
    cfg.monitor = False
    cfg.embedder = "onnx"
    save_profile(cfg, slug)
    # Inject provenance into the just-saved profile entry.
    bag = dict(cfg._extras.get("profiles", {}))
    snap = dict(bag.get(slug, {}))
    snap["_display"] = display
    snap["_source_url"] = source_url
    if note:
        snap["_note"] = note
    bag[slug] = snap
    cfg._extras["profiles"] = bag
    save_config(cfg)


def _write_sources_doc(outcomes: list[Outcome]) -> None:
    SOURCES_MD.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Voice library — provenance",
        "",
        "Audit trail for the starter voice library imported via `scripts/voice_library_import.py`.",
        "",
        "Models are NOT bundled in this repo (per `LICENSE` / brief §7). They live "
        "in `~/.local/share/woys/models/` on the user's machine. This "
        "file documents the upstream HuggingFace URL + slug for each.",
        "",
        "| Slug | Display | Status | Upstream URL |",
        "|------|---------|--------|--------------|",
    ]
    for o in outcomes:
        status = "✅" if o.status == "ok" else "❌"
        lines.append(f"| `{o.voice.slug}` | {o.voice.display} | {status} | <{o.voice.url}> |")
    lines.append("")
    lines.append(
        "Models are typically OpenRAIL-licensed or unlicensed; treat as "
        "**personal use only**. Do NOT redistribute the weights without checking "
        "each upstream repo's specific license."
    )
    lines.append("")
    SOURCES_MD.write_text("\n".join(lines), encoding="utf-8")


def process(voice: Voice) -> Outcome:
    print(f"\n=== {voice.slug}  ({voice.display}) ===")
    if voice.note:
        print(f"  note: {voice.note}")
    stage = STAGE_DIR / voice.slug
    stage.mkdir(parents=True, exist_ok=True)

    # Filename from URL (urllib-decoded).
    url_basename = urllib.parse.unquote(voice.url.rsplit("/", 1)[-1])
    zip_path = stage / url_basename

    # 1. Download
    if not _download(voice.url, zip_path):
        _log_blocker(f"voice-library: {voice.slug} — download failed after 2 attempts: {voice.url}")
        return Outcome(voice, "skipped", "download failed")

    # 2. Verify
    if not _verify_zip(zip_path):
        zip_path.unlink(missing_ok=True)
        if not _download(voice.url, zip_path):
            _log_blocker(f"voice-library: {voice.slug} — corrupt zip + redownload failed")
            return Outcome(voice, "skipped", "corrupt zip")
        if not _verify_zip(zip_path):
            _log_blocker(f"voice-library: {voice.slug} — corrupt zip after retry")
            return Outcome(voice, "skipped", "corrupt zip after retry")

    # 3. Extract
    extract_dir = stage / "extracted"
    if not _extract_zip(zip_path, extract_dir):
        _log_blocker(f"voice-library: {voice.slug} — unzip failed")
        return Outcome(voice, "skipped", "unzip failed")

    # 4. Find .pth
    pth = _find_pth(extract_dir)
    if pth is None:
        _log_blocker(f"voice-library: {voice.slug} — no usable .pth in archive")
        return Outcome(voice, "skipped", "no .pth")
    print(f"  pth: {pth.name} ({pth.stat().st_size / 1024 / 1024:.1f} MiB)")

    # 5. Index (optional)
    idx = _find_index(extract_dir)
    if idx:
        print(f"  index: {idx.name} (informational; engine doesn't load it yet)")

    # 6. Convert
    output_onnx = MODELS_DIR / f"{voice.slug}.onnx"
    print(f"  converting → {output_onnx.name}")
    if not _convert(pth, output_onnx):
        _log_blocker(f"voice-library: {voice.slug} — convert failed (pth: {pth})")
        return Outcome(voice, "skipped", "convert failed")
    onnx_size_mib = output_onnx.stat().st_size / 1024 / 1024
    print(f"  wrote {onnx_size_mib:.1f} MiB onnx")

    # 7. Validate inference
    ok, reason = _validate_inference(output_onnx)
    if not ok:
        _log_blocker(f"voice-library: {voice.slug} — inference validate failed: {reason}")
        # Don't delete: the brief says keep the .onnx for diagnosis.
        return Outcome(voice, "skipped", f"validate failed: {reason}", output_onnx, onnx_size_mib)

    # 8. Register profile
    _register_profile(voice.slug, output_onnx, voice.display, voice.url, voice.note)
    print("  ✅ installed + profile saved")

    return Outcome(voice, "ok", "", output_onnx, onnx_size_mib)


def main() -> int:
    STAGE_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SOURCES_MD.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    outcomes: list[Outcome] = []
    for v in VOICES:
        outcomes.append(process(v))

    elapsed = time.perf_counter() - t0
    print(f"\n=== Outcome summary  (took {elapsed:.1f} s) ===")
    print(f"{'slug':24s}  {'status':9s}  {'size':>9s}  reason")
    for o in outcomes:
        size = f"{o.onnx_size_mib:>7.1f}M" if o.onnx_size_mib else "       -"
        marker = "✅" if o.status == "ok" else "❌"
        print(f"  {marker} {o.voice.slug:22s}  {o.status:9s}  {size}  {o.reason}")

    _write_sources_doc(outcomes)
    print(f"\nprovenance written to {SOURCES_MD}")

    all_ok = all(o.status == "ok" for o in outcomes)
    if all_ok:
        print(f"\nall green — cleaning {STAGE_DIR}")
        shutil.rmtree(STAGE_DIR, ignore_errors=True)
    else:
        print(f"\nfailures present — keeping {STAGE_DIR} for diagnosis")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
