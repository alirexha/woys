"""Download the foundation weights for RVC inference into the user cache.

Run: `python scripts/download_weights.py [--force]`

Pulls (idempotent unless --force):
  - rmvpe_wrapped.onnx   - f0 detector (waveform-input variant; matches
                           upstream's RMVPEOnnxPitchExtractor contract:
                           inputs `waveform`, `threshold`; output `pitchf`)
  - contentvec-f.onnx    - ONNX content encoder
  - amitaro_v2_16k.onnx  - small public RVC voice model for the smoke test

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

# review F-merged-003: these URLs use the moving ref `resolve/main`.
# The load-bearing integrity protection is now `WEIGHTS_SHA256` below, which
# is populated and fail-closed -- if `main` ever serves bytes that differ
# from the pinned hash (tampering, or a legitimate upstream re-publish),
# `fetch()` raises and the file is rejected, so the moving ref is no longer
# a silent integrity hole. Pinning each URL to a literal commit SHA as
# defence-in-depth is a documented residual: it needs a known-good revision
# string for the wok000/* repos that cannot be obtained offline without
# fabricating it. See docs/26-review/phase-6-fixes/commit-010.md.
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

# SHA-256 integrity table. review F-merged-003 (P1): this used to ship
# empty, so `if expected` was always falsy and verification was dead code --
# `install.sh` ran the unverified download on every fresh install. It is now
# populated and the gate is **fail-closed**: a foundation weight with no
# entry here is a hard error in `fetch()` (see below), not a silent pass.
#
# Hashes computed from the dev machine's known-good copies (the same files
# that passed the review Phase 7 listener gate on real hardware). When
# upstream legitimately re-publishes a weight, a maintainer must re-run
# `python scripts/download_weights.py --print-hashes` and bump the entry --
# that is the correct trade for integrity.
WEIGHTS_SHA256: dict[str, str] = {
    "rmvpe_wrapped.onnx": "84f0586308e36157f75b77c8591bf636d6719c0c4ba95f8faf3df479e7566219",
    "contentvec-f.onnx": "4b31ed3d95a568fab7952de923ff7f7d3d17128ea6fce69f665509d24c3156db",
    "amitaro_v2_16k.onnx": "2d74e2fce8d5770e4b640ab45355286396b8308f5b09b059fb41a167592990c5",
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
        print(f"  [skip] {dest.name}  ({size_mb:.1f} MiB) - already cached")
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
    # review F-merged-003 (P1): fail-closed. A known foundation weight
    # with no SHA256 entry is a hard error -- never a silent unverified pass
    # (the pre-fix `if expected and ...` skipped verification whenever the
    # table lacked an entry, which it always did).
    expected = WEIGHTS_SHA256.get(dest.name)
    if not skip_verify:
        if expected is None:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"no SHA256 entry for {dest.name} in WEIGHTS_SHA256 - refusing to "
                f"install an unverified foundation weight. Add its hash (run "
                f"`python scripts/download_weights.py --print-hashes` against a "
                f"known-good copy) or pass --skip-verify if you explicitly trust "
                f"the source."
            )
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
