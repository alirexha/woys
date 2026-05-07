"""Download the foundation weights for RVC inference into the user cache.

Run: `python scripts/download_weights.py [--force]`

Pulls (idempotent unless --force):
  - rmvpe_wrapped.onnx   — f0 detector (waveform-input variant; matches
                           upstream's RMVPEOnnxPitchExtractor contract:
                           inputs `waveform`, `threshold`; output `pitchf`)
  - contentvec-f.onnx    — ONNX content encoder
  - amitaro_v2_16k.onnx  — small public RVC voice model for the smoke test

Destination: ~/.local/share/woys/models/

Note on the rmvpe variant: the engine's session contract expects the
**wrapped** rmvpe (waveform → pitchf), not the bare mel-input variant
(`lj1995/VoiceConversionWebUI/rmvpe.onnx`). The wok000/weights_gpl URL
below is the wrapped graph upstream uses; switching downloaders without
this distinction would cause `_make_session` to bind a `waveform` input
to a `[1, 128, time]` mel tensor, which fails.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

CACHE = Path.home() / ".local" / "share" / "woys" / "models"

WEIGHTS: dict[str, str] = {
    "rmvpe_wrapped.onnx": (
        "https://huggingface.co/wok000/weights_gpl/resolve/main/rmvpe/rmvpe_20231006.onnx"
    ),
    "contentvec-f.onnx": (
        "https://huggingface.co/wok000/weights_gpl/resolve/main/content-vec/contentvec-f.onnx"
    ),
    "amitaro_v2_16k.onnx": (
        "https://huggingface.co/wok000/vcclient_model/resolve/main/"
        "rvc_v2_alpha/amitaro16k/amitaro_v2_16k.onnx"
    ),
}

# Optional SHA-256 integrity table. When present, downloaded files are
# verified post-fetch; mismatch raises and the partial file is removed.
# Run `woys download-weights --print-hashes` after a successful fetch to
# regenerate this table when upstream files legitimately change.
WEIGHTS_SHA256: dict[str, str] = {
    # Populated empirically when the dev machine has known-good copies.
    # Empty values mean "no verification" (transparent). To enable
    # verification, run `python scripts/download_weights.py --print-hashes`
    # against your current files, paste below, then re-run the install.
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path, force: bool, *, skip_verify: bool = False) -> None:
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
    expected = WEIGHTS_SHA256.get(dest.name)
    if expected and not skip_verify:
        actual = _sha256(tmp)
        if actual != expected:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"SHA256 mismatch for {dest.name}:\n"
                f"  expected {expected}\n"
                f"  actual   {actual}\n"
                f"Re-download or pass --skip-verify if you trust the source."
            )
    tmp.rename(dest)
    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"  [done] {dest.name}  ({size_mb:.1f} MiB)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true", help="re-download even if cached")
    p.add_argument(
        "--skip-verify",
        action="store_true",
        help="skip SHA256 integrity check (use only if upstream rehashed legitimately)",
    )
    p.add_argument(
        "--print-hashes",
        action="store_true",
        help="print SHA256 of each cached weight (for regenerating WEIGHTS_SHA256)",
    )
    args = p.parse_args(argv)
    if args.print_hashes:
        print(f"weights cache: {CACHE}")
        for name in WEIGHTS:
            path = CACHE / name
            if path.exists():
                print(f'    "{name}": "{_sha256(path)}",')
            else:
                print(f"  # {name}: missing", file=sys.stderr)
        return 0
    print(f"weights cache: {CACHE}")
    for name, url in WEIGHTS.items():
        fetch(url, CACHE / name, force=args.force, skip_verify=args.skip_verify)
    print("ok.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
