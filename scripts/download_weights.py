"""Download the foundation weights for RVC inference into the user cache.

Run: `python scripts/download_weights.py [--force]`

Pulls (idempotent unless --force):
  - rmvpe.onnx           — f0 detector
  - contentvec-f.onnx    — ONNX content encoder (future: lets us drop fairseq)
  - hubert_base.pt       — fairseq content encoder (current default)
  - amitaro_v2_16k.onnx  — small public RVC voice model for the smoke test

Destination: ~/.local/share/woys/models/
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

CACHE = Path.home() / ".local" / "share" / "woys" / "models"

WEIGHTS: dict[str, str] = {
    "rmvpe.onnx": ("https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.onnx"),
    "contentvec-f.onnx": (
        "https://huggingface.co/wok000/weights_gpl/resolve/main/content-vec/contentvec-f.onnx"
    ),
    "hubert_base.pt": (
        "https://huggingface.co/ddPn08/rvc-webui-models/resolve/main/embeddings/hubert_base.pt"
    ),
    "amitaro_v2_16k.onnx": (
        "https://huggingface.co/wok000/vcclient_model/resolve/main/"
        "rvc_v2_alpha/amitaro16k/amitaro_v2_16k.onnx"
    ),
}


def fetch(url: str, dest: Path, force: bool) -> None:
    if dest.exists() and not force:
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  [skip] {dest.name}  ({size_mb:.1f} MiB) — already cached")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  [get ] {dest.name}  ← {url}")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        chunk = 1 << 16
        while True:
            data = r.read(chunk)
            if not data:
                break
            f.write(data)
    tmp.rename(dest)
    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"  [done] {dest.name}  ({size_mb:.1f} MiB)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true", help="re-download even if cached")
    args = p.parse_args(argv)
    print(f"weights cache: {CACHE}")
    for name, url in WEIGHTS.items():
        fetch(url, CACHE / name, force=args.force)
    print("ok.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
