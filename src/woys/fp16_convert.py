"""ONNX fp32 → fp16 weight conversion for the foundation models.

Halves the on-disk size of `rmvpe_wrapped.onnx` and (optionally)
`contentvec-f.onnx`. Outputs land alongside the inputs as
`<name>-fp16.onnx`, which the engine auto-picks at start time.

Quality notes (validated on this CachyOS / RTX 2070 host):

  * **rmvpe (pitch detector)** - fp16 inference produces pitch values within
    0.1 Hz of the fp32 baseline on a 220 Hz sustained voiced test. Safe to
    promote by default.
  * **contentvec (content encoder)** - fp16 inference produces feats with
    cosine similarity ~0.75 vs fp32, which is *low enough* to materially
    change RVC voice quality downstream. **Not promoted by default**; the
    `--include-contentvec` flag emits the file but the engine still loads
    fp32 unless the user explicitly points `contentvec_model` at it.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterable
from pathlib import Path

MODELS_DIR = Path.home() / ".local" / "share" / "woys" / "models"


def _convert_one(src: Path, dst: Path, *, op_block_list: list[str] | None = None) -> int:
    import onnx
    from onnxconverter_common import float16

    print(f"  [convert] {src.name} → {dst.name}")
    model = onnx.load(str(src))
    fp16_model = float16.convert_float_to_float16(
        model,
        keep_io_types=False,
        disable_shape_infer=True,
        op_block_list=op_block_list or None,
    )
    # The engine loads `<name>-fp16.onnx` whenever it exists, so a write cut
    # short (ENOSPC, Ctrl-C) must never land under that name: write beside
    # it, check the result parses as a model, then rename into place.
    tmp = dst.with_name(dst.name + ".part")
    try:
        onnx.save(fp16_model, str(tmp))
        onnx.checker.check_model(str(tmp))
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)
    return dst.stat().st_size


def _existing_output_ok(dst: Path) -> bool:
    """True if `dst` holds a readable model. Files from before the atomic
    write can be truncated; those get regenerated instead of skipped."""
    import onnx

    try:
        onnx.checker.check_model(str(dst))
    except Exception as exc:
        print(f"  [warn] {dst.name} is unreadable ({type(exc).__name__}); regenerating")
        return False
    return True


def convert_rmvpe(models_dir: Path | None = None, *, force: bool = False) -> Path | None:
    src = (models_dir or MODELS_DIR) / "rmvpe_wrapped.onnx"
    dst = src.with_name(src.stem + "-fp16" + src.suffix)
    if not src.exists():
        print(f"  [skip] {src} missing")
        return None
    if dst.exists() and not force and _existing_output_ok(dst):
        print(f"  [skip] {dst.name} already present ({dst.stat().st_size / 1024 / 1024:.1f} MiB)")
        return dst
    # rmvpe has Cast nodes that confuse fp16 conversion under
    # `keep_io_types=False`; block them.
    sz = _convert_one(src, dst, op_block_list=["Cast"])
    print(f"    wrote {sz / 1024 / 1024:.1f} MiB (was {src.stat().st_size / 1024 / 1024:.1f})")
    return dst


def convert_contentvec(models_dir: Path | None = None, *, force: bool = False) -> Path | None:
    src = (models_dir or MODELS_DIR) / "contentvec-f.onnx"
    dst = src.with_name(src.stem + "-fp16" + src.suffix)
    if not src.exists():
        print(f"  [skip] {src} missing")
        return None
    if dst.exists() and not force and _existing_output_ok(dst):
        print(f"  [skip] {dst.name} already present ({dst.stat().st_size / 1024 / 1024:.1f} MiB)")
        return dst
    print(
        "  [warn] contentvec fp16 lowers quality (cosine ~0.75 vs fp32) - "
        "not auto-loaded by the engine; point contentvec_model at it explicitly."
    )
    sz = _convert_one(src, dst)
    print(f"    wrote {sz / 1024 / 1024:.1f} MiB (was {src.stat().st_size / 1024 / 1024:.1f})")
    return dst


def cli_fp16_convert(targets: Iterable[str], force: bool = False) -> int:
    targets = set(targets)
    if not targets:
        targets = {"rmvpe"}
    print(f"fp16 convert targets: {sorted(targets)}")
    missing: list[str] = []
    try:
        if "rmvpe" in targets and convert_rmvpe(force=force) is None:
            missing.append("rmvpe")
        if "contentvec" in targets and convert_contentvec(force=force) is None:
            missing.append("contentvec")
    except Exception as e:
        print(f"[fp16-convert] ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    if missing:
        print(
            f"[fp16-convert] ERROR: source model missing for {', '.join(missing)} "
            "(see the [skip] line above); fetch it with scripts/download_weights.py.",
            file=sys.stderr,
        )
        return 1
    print("[fp16-convert] done. Restart the engine to pick up the fp16 model(s).")
    return 0


def _main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--include-contentvec",
        action="store_true",
        help="also fp16-convert contentvec (lower quality - not auto-loaded)",
    )
    p.add_argument("--force", action="store_true", help="overwrite existing fp16 files")
    args = p.parse_args()
    targets = ["rmvpe"]
    if args.include_contentvec:
        targets.append("contentvec")
    return cli_fp16_convert(targets, force=args.force)


if __name__ == "__main__":
    raise SystemExit(_main())
