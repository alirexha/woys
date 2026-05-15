"""`woys convert <pth>` - real RVC .pth → .onnx exporter.

The upstream voice-changer repo ships an `export2onnx` function but expects
to be called from inside its own Pipeline / FastAPI server context. This
module is a thin original-work wrapper that:

  1. Probes the `.pth` checkpoint to derive metadata (model variant,
     embedding channels, f0, sample rate). Logic mirrors upstream's
     `RVCModelSlotGenerator._setInfoByPytorch` but without depending on
     the slot-manager singleton.
  2. Calls upstream's `_export2onnx` with the derived metadata.
  3. Validates the exported `.onnx` loads in ONNX Runtime with the same
     I/O signature our engine expects (feats, p_len, pitch?, pitchf?, sid).

Cache: HuggingFace-derived inputs are downloaded into
`~/.local/share/woys/converted/<repo>/` so re-conversion is free.

Security: `.pth` files are pickle archives. `torch.load(weights_only=False)`
will execute arbitrary Python on load. We try `weights_only=True` first;
if torch's safe-load mode rejects the checkpoint (older RVC formats with
custom unpickle constructors do), we require explicit consent via the
`--yes-i-trust-the-pickle` flag (or `WOYS_YES_I_TRUST_THE_PICKLE=1`)
before falling back. Only consent for files you trust - RVC checkpoints
shared on Discord / unknown forks are an RCE vector.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import contextlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CACHE_DIR = Path.home() / ".local" / "share" / "woys" / "converted"


def _user_trusts_pickle(flag: bool) -> bool:
    """Has the user explicitly opted into unsafe pickle loading FOR
    THIS CALL?

    review F-05-04: pre-fix this also honored
    `WOYS_YES_I_TRUST_THE_PICKLE=1` from the environment. The env-var
    path collapsed per-invocation consent to per-environment-lifetime
    consent: set once in a shell rc, it auto-trusted EVERY
    `woys convert` -- including batch flows like
    `voice_library_import.py` that iterate ~9 third-party models.
    `weights_only=False` is genuine arbitrary-code-execution, so the
    silent expansion of "trust this file" to "trust every file
    forever" was an unacceptable consent regression.

    Post-fix the only consent path is the per-invocation
    `--yes-i-trust-the-pickle` CLI flag. Batch callers pass the flag
    explicitly per-file. The env var is no longer honored at all (it
    is documented as REMOVED so users coming from older configs see
    a clear error if they try to set it).
    """
    return bool(flag)


# review F-05-04: keep the constant name for back-compat error
# messages but the env var itself is NOT consulted. If a user has
# `WOYS_YES_I_TRUST_THE_PICKLE=1` in their shell rc, woys convert
# now ignores it and falls into the consent-required error path.
_TRUST_PICKLE_ENV_NAME_REMOVED = "WOYS_YES_I_TRUST_THE_PICKLE"


def _safe_torch_load(pth_path: Path, *, trust_pickle: bool) -> Any:
    """Load a torch checkpoint with weights_only=True first; on failure,
    require explicit consent before falling back to weights_only=False
    (the unsafe pickle-deserialize mode).
    """
    import torch

    try:
        return torch.load(str(pth_path), map_location="cpu", weights_only=True)
    except Exception as safe_err:
        # Many RVC v1 checkpoints have custom unpickle constructors that
        # weights_only rejects. Fall back ONLY with explicit consent.
        if not _user_trusts_pickle(trust_pickle):
            raise RuntimeError(
                f"\n[security] Refusing to load {pth_path.name} via the unsafe pickle path.\n"
                f"  Safe-load failed with: {type(safe_err).__name__}: {safe_err}\n"
                f"\n"
                f"  This .pth is a Python pickle. torch.load(weights_only=False)\n"
                f"  will execute arbitrary code on import. Only proceed if you\n"
                f"  trust the source (a model you trained, or a verified fork).\n"
                f"\n"
                f"  To proceed, re-run with --yes-i-trust-the-pickle. The env var\n"
                f"  {_TRUST_PICKLE_ENV_NAME_REMOVED} is no longer honored\n"
                f"  (review F-05-04: it collapsed per-invocation consent to\n"
                f"  per-environment-lifetime consent for an arbitrary-code-\n"
                f"  execution operation; the flag forces per-file consent).\n"
            ) from safe_err
        # review F-05-04: log every unsafe load loudly so a future
        # audit can grep for it. The consent has already been granted
        # (via the CLI flag); the log line is forensic.
        print(
            f"[security] UNSAFE pickle load of {pth_path} (--yes-i-trust-the-pickle granted)",
            file=sys.stderr,
        )
        return torch.load(str(pth_path), map_location="cpu", weights_only=False)


@dataclass
class _RVCMeta:
    """Subset of upstream RVCModelSlot fields needed by `_export2onnx`."""

    modelType: str
    samplingRate: int
    f0: bool
    embChannels: int
    embedder: str
    embOutputLayer: int
    useFinalProj: bool


def _probe_pth_metadata(pth_path: Path, *, trust_pickle: bool = False) -> _RVCMeta:
    """Inspect the .pth checkpoint dict to figure out which RVC variant it is.

    Mirrors the upstream `_setInfoByPytorch` decision tree. Doesn't import
    upstream's class hierarchy - keeps this module a pure original-work
    derivative-of-format-knowledge, not a derivative of upstream code.

    `trust_pickle=True` allows fall-through to the unsafe `torch.load
    (weights_only=False)` path; default-False makes the consent boundary
    explicit at every call site.
    """
    cpt = _safe_torch_load(pth_path, trust_pickle=trust_pickle)
    config = cpt.get("config")
    if config is None:
        raise ValueError(
            f"{pth_path.name}: missing 'config' field - not a recognized RVC checkpoint"
        )
    config_len = len(config)
    version = cpt.get("version", "v1")
    f0 = bool(cpt.get("f0", 1) == 1)
    sr = int(config[-1])

    # Late import upstream's enum so we feed _export2onnx the right strings.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))
    from const import EnumInferenceTypes  # type: ignore[import-not-found]

    if config_len == 18:
        # Standard / official RVC checkpoint.
        if version == "v1" or version is None:
            mt = (
                EnumInferenceTypes.pyTorchRVC.value
                if f0
                else EnumInferenceTypes.pyTorchRVCNono.value
            )
            return _RVCMeta(
                modelType=mt,
                samplingRate=sr,
                f0=f0,
                embChannels=256,
                embedder="hubert_base",
                embOutputLayer=9,
                useFinalProj=True,
            )
        # v2+
        mt = (
            EnumInferenceTypes.pyTorchRVCv2.value
            if f0
            else EnumInferenceTypes.pyTorchRVCv2Nono.value
        )
        return _RVCMeta(
            modelType=mt,
            samplingRate=sr,
            f0=f0,
            embChannels=768,
            embedder="hubert_base",
            embOutputLayer=12,
            useFinalProj=False,
        )

    # DDPN-style WebUI checkpoint - has explicit embChannels in config[17].
    emb_channels = int(config[17]) if len(config) > 17 else 768
    use_final_proj = emb_channels == 256
    emb_layer = int(cpt.get("embedder_output_layer", 9))
    embedder_name = cpt.get("embedder_name", "hubert_base")
    if isinstance(embedder_name, str) and embedder_name.endswith("768"):
        embedder_name = embedder_name[:-3]
    mt = EnumInferenceTypes.pyTorchWebUI.value if f0 else EnumInferenceTypes.pyTorchWebUINono.value
    return _RVCMeta(
        modelType=mt,
        samplingRate=sr,
        f0=f0,
        embChannels=emb_channels,
        embedder=str(embedder_name),
        embOutputLayer=emb_layer,
        useFinalProj=use_final_proj,
    )


# review F-31-09: post-export fp16 numerical quality gate.
# Pre-fix the only validation on `--fp16` exports was `_validate_onnx_loads`
# (load + I/O names). The docstring on `convert_pth_to_onnx` admitted that
# v1 models "often degrade", but nothing in the export pipeline measured
# the actual degradation -- the user discovered it by ear, possibly mid-
# call. The gate runs the fp16 model and an fp32 reference on a fixed
# seeded input, computes SNR (signal-to-noise ratio in dB), and prints a
# loud stderr warning when SNR falls below the threshold.
#
# Threshold rationale: fp16 vs fp32 SNR for a healthy RVC v2 conversion
# is typically 35-55 dB on the engine's typical input. 30 dB is the
# audibility floor (the gap between perceptually-identical and "you can
# tell" on a clean tone). 20 dB is severe -- catastrophic clipping or a
# broken op. The default surfaces the value either way so the user has a
# numeric quality readout to compare against future builds.
_FP16_SNR_THRESHOLD_DB = 30.0


def _fp16_quality_gate(
    fp16_path: Path,
    fp32_path: Path,
    *,
    is_f0: bool,
    emb_channels: int,
    snr_threshold_db: float = _FP16_SNR_THRESHOLD_DB,
    n_frames: int = 100,
    seed: int = 42,
) -> float:
    """Run ORT on both ONNX files with seeded synthetic inputs and return
    the fp16-vs-fp32 SNR in dB.

    A SNR below `snr_threshold_db` emits a loud stderr warning naming
    the measurement and the threshold. The caller can decide whether to
    abort the export, but this function never raises on quality (only
    on shape mismatches between the two outputs, which would be a real
    structural bug).

    Inputs follow the engine's `_infer` signature:
    - `feats` (1, T, emb_channels) -- float32 (or fp16 cast for the
      fp16 model if its input type says so).
    - `p_len` (1,) -- int64.
    - `sid` (1,) -- int64.
    - `pitch` (1, T) / `pitchf` (1, T) when `is_f0=True`.

    Seeded with `np.random.default_rng(seed)` so two runs of the same
    pair of ONNX files produce the same SNR -- the gate is meant to be
    a reproducible quality signal, not a sample of in-the-wild audio.
    """
    import numpy as np
    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()

    rng = np.random.default_rng(seed)
    feats = rng.standard_normal((1, n_frames, emb_channels)).astype(np.float32)
    p_len = np.array([n_frames], dtype=np.int64)
    sid = np.array([0], dtype=np.int64)
    inputs: dict[str, np.ndarray[Any, Any]] = {"feats": feats, "p_len": p_len, "sid": sid}
    if is_f0:
        pitch = np.full((1, n_frames), 50, dtype=np.int64)
        pitchf = np.full((1, n_frames), 440.0, dtype=np.float32)
        inputs["pitch"] = pitch
        inputs["pitchf"] = pitchf

    sess_fp16 = ort.InferenceSession(str(fp16_path), providers=["CPUExecutionProvider"])
    sess_fp32 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])

    def _coerce(
        d: dict[str, np.ndarray[Any, Any]],
        type_map: dict[str, str],
    ) -> dict[str, np.ndarray[Any, Any]]:
        """Cast each input to the dtype the session declares (fp16 if the
        model's feats input is fp16; everything else stays as-is)."""
        out: dict[str, np.ndarray[Any, Any]] = {}
        for k, v in d.items():
            t = type_map.get(k, "")
            if "float16" in t and v.dtype == np.float32:
                out[k] = v.astype(np.float16)
            else:
                out[k] = v
        return out

    types16 = {i.name: i.type for i in sess_fp16.get_inputs()}
    types32 = {i.name: i.type for i in sess_fp32.get_inputs()}
    out16 = sess_fp16.run(None, _coerce(inputs, types16))[0]
    out32 = sess_fp32.run(None, _coerce(inputs, types32))[0]

    a16 = np.asarray(out16, dtype=np.float64)
    a32 = np.asarray(out32, dtype=np.float64)
    if a16.shape != a32.shape:
        raise RuntimeError(f"fp16/fp32 output shape mismatch: {a16.shape} vs {a32.shape}")
    noise = a16 - a32
    sig_power = float((a32**2).mean())
    noise_power = float((noise**2).mean())
    if noise_power == 0.0:
        snr_db = float("inf")
    elif sig_power == 0.0:
        snr_db = float("-inf")
    else:
        snr_db = 10.0 * float(np.log10(sig_power / noise_power))

    print(
        f"[convert] fp16 quality gate: SNR = {snr_db:.1f} dB (threshold: {snr_threshold_db:.1f} dB)"
    )
    if snr_db < snr_threshold_db:
        print(
            f"[convert] WARNING: fp16 export degraded -- SNR {snr_db:.1f} dB "
            f"is below the {snr_threshold_db:.1f} dB threshold. The model "
            f"may sound noticeably worse than the fp32 export. Consider "
            f"keeping fp32 (drop --fp16) or validating perceptually before "
            f"shipping.",
            file=sys.stderr,
        )
    return snr_db


def _validate_onnx_loads(onnx_path: Path) -> None:
    """Sanity-check the freshly-exported file: must load in ORT and expose
    the I/O names our engine reads (`feats`, `pitch`/`pitchf`, `audio`)."""
    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    in_names = {i.name for i in sess.get_inputs()}
    out_names = {o.name for o in sess.get_outputs()}
    required_in = {"feats", "p_len", "sid"}
    if not required_in.issubset(in_names):
        raise RuntimeError(f"converted ONNX missing inputs: required {required_in}, got {in_names}")
    if "audio" not in out_names:
        raise RuntimeError(f"converted ONNX has no 'audio' output: {out_names}")


def convert_pth_to_onnx(
    pth_path: Path,
    output_path: Path | None = None,
    *,
    fp16: bool = False,
    opset: int = 17,
    trust_pickle: bool = False,
) -> Path:
    """Convert an RVC `.pth` to ONNX. Returns the output path.

    `fp16=True` exports half-precision weights. Use only on RVC v2 models
    where you've validated quality is preserved - v1 models often degrade.
    `opset=17` matches what the engine expects; raise it only if you've
    verified ORT 1.20+ supports the ops the model emits.

    `trust_pickle=True` permits the unsafe `torch.load(weights_only=False)`
    fall-through for older RVC checkpoints (see module docstring). Default
    False makes safe-load attempt-then-fail unless the user opted in via
    the CLI flag or env var.
    """
    pth_path = Path(pth_path).resolve()
    if not pth_path.exists():
        raise FileNotFoundError(f"no such file: {pth_path}")
    if output_path is None:
        output_path = pth_path.with_suffix(".onnx")
    output_path = Path(output_path).resolve()

    meta = _probe_pth_metadata(pth_path, trust_pickle=trust_pickle)
    # v0.14.0 (Lens 6 / C015): consent state captured here, used by
    # `_gated_torch_load` (defined below) to gate the unsafe load that
    # _export2onnx performs internally.
    pth_already_consented = _user_trusts_pickle(trust_pickle)
    print(
        f"[convert] {pth_path.name}: "
        f"type={meta.modelType.split('.')[-1]} sr={meta.samplingRate} "
        f"f0={meta.f0} embCh={meta.embChannels} "
        f"L{meta.embOutputLayer}{'+proj' if meta.useFinalProj else ''} "
        f"fp16={fp16}"
    )

    # Late-import upstream's _export2onnx. The opset arg isn't in upstream's
    # signature; we pass it through via monkey-patching torch.onnx.export.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))
    import torch
    from voice_changer.RVC.onnxExporter.export2onnx import (  # type: ignore[import-not-found]
        _export2onnx,
    )

    metadata_dict = {
        "modelType": meta.modelType,
        "samplingRate": meta.samplingRate,
        "f0": meta.f0,
        "embChannels": meta.embChannels,
        "embedder": meta.embedder,
        "embOutputLayer": meta.embOutputLayer,
        "useFinalProj": meta.useFinalProj,
        "application": "VC_CLIENT",
        "version": "2.1",
    }

    output_simple = output_path.with_name(output_path.stem + "_simple.onnx")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Pin the opset for the duration of the export. torch.onnx.export's
    # signature has too many overloads to type-narrow cleanly.
    original_export: Any = torch.onnx.export

    def _export_with_opset(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("opset_version", opset)
        return original_export(*args, **kwargs)

    # v0.14.0 (Lens 6 / C015): the pickle gate in _safe_torch_load gates
    # _probe_pth_metadata, but upstream's _export2onnx (line 61 in
    # src/server/voice_changer/RVC/onnxExporter/export2onnx.py) calls
    # torch.load(input_model, map_location="cpu") with NO weights_only
    # arg. On torch < 2.6 the default is weights_only=False, which
    # executes arbitrary pickle code on import (CVE class). Without this
    # patch, woys probed the file safely, asked for consent, then loaded
    # it unsafely anyway during the export step.
    #
    # We monkey-patch torch.load for the duration of _export2onnx with
    # the same `_safe_torch_load`-style gate. Trust state is determined
    # by whether _probe_pth_metadata was called with trust_pickle=True
    # (it had to be, otherwise we'd never have reached this point with
    # a non-weights_only-loadable .pth). The flag is captured by
    # `pth_already_consented` from the calling scope below.
    original_torch_load: Any = torch.load

    def _gated_torch_load(*args: Any, **kwargs: Any) -> Any:
        # First try weights_only=True if not already specified.
        if "weights_only" not in kwargs:
            try:
                return original_torch_load(*args, weights_only=True, **kwargs)
            except Exception as safe_err:
                # The upstream caller wants weights_only=False semantics.
                # Allow only with already-granted consent.
                if not pth_already_consented:
                    raise RuntimeError(
                        "[security] _export2onnx attempted unsafe torch.load "
                        f"on {args[0] if args else '<?>'} but no consent was "
                        f"granted. Re-run with --yes-i-trust-the-pickle "
                        f"(the {_TRUST_PICKLE_ENV_NAME_REMOVED} env var "
                        f"is no longer honored -- review F-05-04)."
                    ) from safe_err
                return original_torch_load(*args, weights_only=False, **kwargs)
        return original_torch_load(*args, **kwargs)

    def _run_one_export(fp16_arg: bool, out_path: Path, simple_path: Path) -> None:
        """Single _export2onnx call with the torch.load + torch.onnx.export
        monkey-patches scoped to its lifetime. Factored so the fp16
        quality gate (F-31-09) can run a second fp32 reference export
        without duplicating the patching boilerplate."""
        try:
            torch.onnx.export = _export_with_opset
            torch.load = _gated_torch_load
            _export2onnx(
                str(pth_path),
                str(out_path),
                str(simple_path),
                fp16_arg,
                metadata_dict,
            )
        finally:
            torch.onnx.export = original_export
            torch.load = original_torch_load

    _run_one_export(fp16, output_path, output_simple)

    if not output_path.exists():
        raise RuntimeError(f"export silently failed - {output_path} not created")

    print(f"[convert] wrote {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MiB)")
    _validate_onnx_loads(output_path)
    print("[convert] ONNX validation OK - ready for the engine.")

    # review F-31-09: post-export fp16 numerical quality gate.
    # The docstring on this function says "v1 models often degrade" --
    # measure that, don't just warn in docs. We do a second fp32 export
    # to a tmp path, run both through ORT on a seeded reference input,
    # compute SNR, and emit a loud stderr warning if SNR is below the
    # audibility floor. The fp32 reference is deleted after the gate.
    if fp16:
        # Advisory: v1 models historically degrade under fp16. Embed
        # channels == 256 is the v1 signature (vs 768 on v2). The
        # actual SNR will tell the user how bad it is for this model.
        if meta.embChannels == 256:
            print(
                "[convert] note: this is an RVC v1 checkpoint "
                "(embChannels=256). v1 + fp16 historically produces "
                "lower SNR than v2 + fp16 -- the gate below measures it.",
                file=sys.stderr,
            )
        fp32_ref = output_path.with_name(output_path.stem + "_fp32_ref.onnx")
        fp32_ref_simple = fp32_ref.with_name(fp32_ref.stem + "_simple.onnx")
        try:
            _run_one_export(False, fp32_ref, fp32_ref_simple)
            if not fp32_ref.exists():
                raise RuntimeError(
                    "fp16 quality gate: fp32 reference export silently failed; skipping SNR check"
                )
            _fp16_quality_gate(
                output_path,
                fp32_ref,
                is_f0=meta.f0,
                emb_channels=meta.embChannels,
            )
        except Exception as e:
            # The quality gate is a safety net; failure to run it must
            # NOT lose the user's already-completed fp16 export. Surface
            # what went wrong so they know to validate by ear.
            print(
                f"[convert] WARNING: fp16 quality gate skipped "
                f"({type(e).__name__}: {e}). The fp16 ONNX at "
                f"{output_path} is the user's; validate quality "
                f"perceptually before shipping.",
                file=sys.stderr,
            )
        finally:
            for ref_path in (fp32_ref, fp32_ref_simple):
                if ref_path.exists():
                    with contextlib.suppress(OSError):
                        ref_path.unlink()

    # v0.6.6 - `_export2onnx` always writes a `<stem>_simple.onnx` sibling
    # for upstream's stripped-down inference path, but the woys engine only
    # ever loads the regular `.onnx`. Leaving the sibling around bloats the
    # models dir and confuses `woys models list`. Drop it.
    if output_simple.exists():
        with contextlib.suppress(OSError):
            output_simple.unlink()

    return output_path


def cli_convert(
    pth: str,
    output: str | None = None,
    opset: int = 17,
    fp16: bool = False,
    *,
    trust_pickle: bool = False,
) -> int:
    try:
        out = convert_pth_to_onnx(
            Path(pth),
            Path(output) if output else None,
            fp16=fp16,
            opset=opset,
            trust_pickle=trust_pickle,
        )
    except Exception as e:
        print(f"[convert] ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(f"\nLoad it from `~/.config/woys/config.toml`:\n  rvc_model = {str(out)!r}\n")
    return 0
