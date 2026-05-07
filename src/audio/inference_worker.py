"""v0.8.0 — inference subprocess.

Closes the LESSONS §19 threading tax (~23 ms typical-case overhead
when ORT inference runs in the engine's daemon thread alongside the
writer / watchdog / stderr-reader threads, all contending for the
GIL during numpy ops between ONNX sessions).

The child process:
  - Owns its own CUDA context (forced via `spawn` start method, never
    `fork` — CUDA contexts don't survive fork).
  - Loads cv + rmvpe + rvc ONNX sessions for life. Hot-swaps RVC
    sessions via the same `RvcSessionPool` the in-process path
    uses.
  - Inherits all rc7 → rc12 wins:
      gc.disable() during inference loop                          (rc7)
      EXHAUSTIVE cuDNN search                                     (rc10)
      kSameAsRequested arena strategy + max workspace             (rc12)
      SCHED_FIFO prio 60 if RLIMIT_RTPRIO allows                  (rc11)
      Broader pre-warm covering every soxr-emitted shape          (rc9)
  - Talks to parent over a pair of multiprocessing Pipes (control
    + metadata) plus two `SharedMemory` regions (raw audio bytes
    parent→child for input, result bytes child→parent for output).
    Pickle is on the small control messages only; the hot-path
    audio arrays are zero-copy via `np.ndarray(buffer=shm.buf)`.
  - Watchdogs the parent: if `os.getppid()` becomes 1 (parent
    died, child reparented to init) the child exits gracefully.

The protocol is defined in `audio.inference_proto` so both sides
agree on message shapes without a circular import.
"""

from __future__ import annotations

import contextlib
import gc
import multiprocessing as mp
import os
import sys
import time
import traceback
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

NDArrayF32 = npt.NDArray[np.float32]
NDArrayI64 = npt.NDArray[np.int64]


# Protocol message kinds (must match inference_client). Defined here
# inline rather than in a shared module to keep the child's import
# graph tiny — child doesn't need to import the engine module just
# to read these constants.
CMD_INFER = "infer"
CMD_SWAP_MODEL = "swap_model"
CMD_STOP = "stop"
RESP_READY = "ready"
RESP_DONE = "done"
RESP_SWAP_DONE = "swap_done"
RESP_ERROR = "error"


def _ort_preload_dlls() -> None:
    """Mirror `engine.py`'s ORT preload step. ORT-GPU 1.20+ on driver
    595 needs explicit preload of the pip-shipped CUDA libs before
    any session creation."""
    import onnxruntime as ort  # noqa: PLC0415 — child-process-only

    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()


def _try_set_rt_priority(label: str) -> str | None:
    """Same logic as engine._apply_thread_priority — try SCHED_FIFO
    prio 60, fall back to nice(-10), then to a logged warning."""
    try:
        param = os.sched_param(60)
        os.sched_setscheduler(0, os.SCHED_FIFO, param)
        return None
    except (OSError, PermissionError, AttributeError) as rt_err:
        try:
            os.nice(-10)
            return None
        except (OSError, PermissionError) as nice_err:
            return (
                f"realtime_priority[{label}] denied "
                f"(SCHED_FIFO: {type(rt_err).__name__}: {rt_err}; "
                f"nice -10: {type(nice_err).__name__}: {nice_err}); "
                f"needs CAP_SYS_NICE or RLIMIT_RTPRIO ≥ 60"
            )


def child_main(
    input_shm_name: str,
    output_shm_name: str,
    parent_recv: Any,
    child_send: Any,
    cfg_dict: dict[str, Any],
    parent_pid: int,
) -> None:
    """Entry point for the inference child process.

    Started via `multiprocessing.Process(target=child_main, args=(...))`
    with the `spawn` start method. Pipes are passed as
    `multiprocessing.connection.Connection` objects — multiprocessing
    pickles them correctly across spawn so the child receives live
    Connection handles (NOT raw fds, which would be invalid in the
    new process).

    Args:
      parent_recv: Connection the child reads commands FROM (parent's
        write end — i.e. parent calls `.send(...)` on the other end,
        child calls `.recv()` here).
      child_send: Connection the child writes responses TO.

    The child loop:
      1. Open the SharedMemory regions by name.
      2. Load + warm ORT sessions.
      3. Send RESP_READY to parent.
      4. Loop: recv command, dispatch, send response.
      5. Exit on CMD_STOP or parent-death.
    """
    # No reconstruction needed — Connection objects come through spawn intact.

    # Step 1: open the shared memory regions the parent created.
    try:
        input_shm = shared_memory.SharedMemory(name=input_shm_name)
        output_shm = shared_memory.SharedMemory(name=output_shm_name)
    except FileNotFoundError as e:
        with contextlib.suppress(Exception):
            child_send.send({"cmd": RESP_ERROR, "error": f"shm open failed: {e}"})
        return

    # Step 2: apply rc7 / rc11 wins inside the child.
    if cfg_dict.get("realtime_priority", True):
        _try_set_rt_priority("inference-child")
    if cfg_dict.get("inference_subprocess_disable_gc", True):
        gc.disable()

    # Step 3: load ORT sessions. We import + use engine helpers from
    # within the child so we share session-creation logic exactly with
    # the in-process path. Importing `engine` here is fine because the
    # child has its own Python interpreter — there's no shared ORT
    # state with the parent.
    try:
        _ort_preload_dlls()
        # Make `audio.engine` importable from the child.
        _src_root = Path(__file__).resolve().parent.parent
        if str(_src_root) not in sys.path:
            sys.path.insert(0, str(_src_root))

        from audio.engine import (
            EngineConfig,
            RvcSessionPool,
            _FairseqEmbedder,
            _interpolate_voiced_gaps_np,
            _make_session,
            _to_pitch_coarse,
        )

        cfg = EngineConfig(
            **{k: v for k, v in cfg_dict.items() if k in EngineConfig.__dataclass_fields__}
        )

        cv = _make_session(cfg.contentvec_model)
        rmvpe = _make_session(cfg.rmvpe_model)
        rvc_pool = RvcSessionPool(max_size=cfg.session_pool_size)
        rvc = rvc_pool.get_or_create(cfg.rvc_model)

        cv_input_dtype = cv.get_inputs()[0].type
        rmvpe_input_dtype = rmvpe.get_inputs()[0].type
        is_half = rvc.get_inputs()[0].type != "tensor(float)"
        rvc_output_sr = _probe_rvc_rate(rvc)

        # Step 4: optional fairseq embedder. Mirrors engine's lazy load.
        fairseq_embedder: _FairseqEmbedder | None = None
        active_embedder = "onnx"
        if cfg.embedder == "fairseq":
            try:
                from pathlib import Path as _P  # noqa: PLC0415

                hubert_path = _P(str(cfg.contentvec_model)).parent / "hubert_base.pt"
                if hubert_path.exists():
                    fairseq_embedder = _FairseqEmbedder(hubert_path)
                    active_embedder = "fairseq"
            except (ImportError, FileNotFoundError, RuntimeError):
                active_embedder = "onnx"

        # Step 5: pre-warm. Same logic as engine._warmup_realtime_pipeline
        # — probe soxr to get the realtime shape set, then run _infer_impl
        # for each shape so EXHAUSTIVE cuDNN benchmarks every algo at
        # warmup time, not realtime.
        _warmup_in_child(
            cv=cv,
            rmvpe=rmvpe,
            rvc=rvc,
            cfg=cfg,
            cv_input_dtype=cv_input_dtype,
            rmvpe_input_dtype=rmvpe_input_dtype,
            is_half=is_half,
            fairseq_embedder=fairseq_embedder,
            active_embedder=active_embedder,
        )
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        with contextlib.suppress(Exception):
            child_send.send(
                {
                    "cmd": RESP_ERROR,
                    "error": f"child startup failed: {type(e).__name__}: {e}",
                }
            )
        return

    # Step 6: announce ready. Includes the probed RVC output rate so
    # the parent knows what sink_rate to set for the resampler_out.
    with contextlib.suppress(Exception):
        child_send.send(
            {
                "cmd": RESP_READY,
                "rvc_output_sr": rvc_output_sr,
                "active_embedder": active_embedder,
                "is_half": is_half,
            }
        )

    # Step 7: main loop.
    nan_chunks_total = 0
    while True:
        # Watchdog parent. If parent died, our PPID becomes 1 (init).
        if os.getppid() != parent_pid and os.getppid() == 1:
            with contextlib.suppress(Exception):
                child_send.send({"cmd": RESP_ERROR, "error": "parent died, child exiting"})
            break

        try:
            msg = parent_recv.recv()
        except (EOFError, OSError, BrokenPipeError):
            # Parent's pipe closed unexpectedly.
            break

        cmd = msg.get("cmd") if isinstance(msg, dict) else None

        if cmd == CMD_STOP:
            break

        if cmd == CMD_SWAP_MODEL:
            try:
                new_path = Path(msg["path"])
                rvc = rvc_pool.get_or_create(new_path)
                is_half = rvc.get_inputs()[0].type != "tensor(float)"
                rvc_output_sr = _probe_rvc_rate(rvc)
                cfg.rvc_model = new_path
                with contextlib.suppress(Exception):
                    child_send.send(
                        {
                            "cmd": RESP_SWAP_DONE,
                            "rvc_output_sr": rvc_output_sr,
                            "is_half": is_half,
                        }
                    )
            except Exception as e:
                traceback.print_exc(file=sys.stderr)
                with contextlib.suppress(Exception):
                    child_send.send(
                        {
                            "cmd": RESP_ERROR,
                            "error": f"swap failed: {type(e).__name__}: {e}",
                        }
                    )
            continue

        if cmd == CMD_INFER:
            try:
                shape = tuple(msg["input_shape"])
                size = int(np.prod(shape))
                # Read input from shared memory. `np.frombuffer` would
                # give a zero-copy view but it's READ-ONLY, which
                # forces ORT to copy host → device on each call.
                # `np.ndarray(buffer=...)` gives a writable view that
                # ORT can use directly. Cost: still zero-copy from
                # shm; ORT internally manages the host→device copy.
                in_buf = input_shm.buf
                assert in_buf is not None
                audio16k: NDArrayF32 = np.ndarray(shape, dtype=np.float32, buffer=in_buf)
                # Update per-call config knobs from message — these are
                # cheap to ship per-call and let the parent control
                # f0_up_key without restarting the child.
                f0_up_key = int(msg.get("f0_up_key", cfg.f0_up_key))
                sid = int(msg.get("sid", cfg.sid))
                threshold = float(msg.get("threshold", cfg.threshold))

                t_cv0 = time.perf_counter()
                result, cv_ms, rmvpe_ms, rvc_ms, nan_replaced = _infer_impl(
                    audio16k=audio16k,
                    cv=cv,
                    rmvpe=rmvpe,
                    rvc=rvc,
                    is_half=is_half,
                    cv_input_dtype=cv_input_dtype,
                    rmvpe_input_dtype=rmvpe_input_dtype,
                    fairseq_embedder=fairseq_embedder,
                    active_embedder=active_embedder,
                    f0_up_key=f0_up_key,
                    sid=sid,
                    threshold=threshold,
                    interpolate_fn=_interpolate_voiced_gaps_np,
                    pitch_coarse_fn=_to_pitch_coarse,
                    t_cv0=t_cv0,
                )
                if nan_replaced:
                    nan_chunks_total += 1

                # Write result into the output shared memory.
                flat = np.ascontiguousarray(result.reshape(-1)).astype(np.float32, copy=False)
                out_bytes = flat.size * 4
                # Defensive bound check — parent allocated the shm with
                # a known max size; if the model output exceeds that,
                # we send an error rather than overflow.
                if out_bytes > output_shm.size:
                    raise RuntimeError(f"output size {out_bytes} exceeds shm {output_shm.size}")
                out_buf = output_shm.buf
                assert out_buf is not None
                # `np.ndarray(buffer=memoryview)` gives a writable
                # zero-copy view; assign-into-slice writes back to
                # the shared memory the parent will read.
                out_view: NDArrayF32 = np.ndarray((flat.size,), dtype=np.float32, buffer=out_buf)
                out_view[:] = flat

                with contextlib.suppress(Exception):
                    child_send.send(
                        {
                            "cmd": RESP_DONE,
                            "output_shape": tuple(result.shape),
                            "cv_ms": cv_ms,
                            "rmvpe_ms": rmvpe_ms,
                            "rvc_ms": rvc_ms,
                            "nan_replaced": nan_replaced,
                            "nan_chunks_total": nan_chunks_total,
                        }
                    )
            except Exception as e:
                traceback.print_exc(file=sys.stderr)
                with contextlib.suppress(Exception):
                    child_send.send(
                        {
                            "cmd": RESP_ERROR,
                            "error": f"infer failed: {type(e).__name__}: {e}",
                        }
                    )
            continue

        # Unknown command → log and ignore.
        with contextlib.suppress(Exception):
            child_send.send({"cmd": RESP_ERROR, "error": f"unknown cmd: {cmd!r}"})

    # Step 8: graceful teardown.
    with contextlib.suppress(Exception):
        input_shm.close()
    with contextlib.suppress(Exception):
        output_shm.close()


def _probe_rvc_rate(rvc_session: Any) -> int:
    """Probe RVC session for its native output rate. Returns 16000 if
    probing fails — the historical default for amitaro-style v1
    models. Mirrors `engine._cached_rvc_sr` logic minus the cache
    (the child loads sessions one at a time, no need to cache rate
    lookups across instantiations)."""
    try:
        import onnxruntime as ort  # noqa: F401, PLC0415

        # Read from session metadata — RVC v2 ONNX exports embed the
        # output rate as a custom_metadata_map entry.
        meta = rvc_session.get_modelmeta().custom_metadata_map
        if "samplingRate" in meta:
            return int(meta["samplingRate"])
    except Exception:
        pass
    return 16_000


def _warmup_in_child(
    *,
    cv: Any,
    rmvpe: Any,
    rvc: Any,
    cfg: Any,
    cv_input_dtype: str,
    rmvpe_input_dtype: str,
    is_half: bool,
    fairseq_embedder: Any,
    active_embedder: str,
) -> None:
    """Replicates `engine._warmup_realtime_pipeline` using the child's
    own session handles. Same shape probe + per-shape iteration."""
    from audio.engine import _StreamResampler

    chunk_n_mic = round(cfg.chunk_seconds * cfg.mic_rate)
    if chunk_n_mic <= 0:
        return

    rng = np.random.default_rng(42)

    unique_audio16_lens: set[int] = set()
    if cfg.mic_rate != 16_000:
        probe = _StreamResampler(cfg.mic_rate, 16_000)
        for _ in range(20):
            dummy_48k = rng.standard_normal(chunk_n_mic).astype(np.float32) * 0.001
            out_chunk = probe.process(dummy_48k)
            if out_chunk.size > 0:
                unique_audio16_lens.add(int(out_chunk.shape[0]))
    else:
        unique_audio16_lens.add(chunk_n_mic)

    if not unique_audio16_lens:
        unique_audio16_lens.add(round(cfg.chunk_seconds * 16_000))

    # SOLA's history sizing — matches engine._sola_input_cfg.
    from audio.sola import SOLAConfig

    sola_cfg = SOLAConfig(
        rate=16_000,
        crossfade_ms=cfg.sola_crossfade_ms,
        search_ms=cfg.sola_search_ms,
        context_ms=cfg.sola_context_ms,
        corr_threshold=cfg.sola_corr_threshold,
    )
    history_len = sola_cfg.context_samples + sola_cfg.crossfade_samples

    from audio.engine import _interpolate_voiced_gaps_np, _to_pitch_coarse

    for audio16_len in sorted(unique_audio16_lens):
        model_input_len = history_len + audio16_len
        if model_input_len <= 0:
            continue
        dummy = rng.standard_normal(model_input_len).astype(np.float32) * 0.001
        for _ in range(4):
            try:
                _infer_impl(
                    audio16k=dummy,
                    cv=cv,
                    rmvpe=rmvpe,
                    rvc=rvc,
                    is_half=is_half,
                    cv_input_dtype=cv_input_dtype,
                    rmvpe_input_dtype=rmvpe_input_dtype,
                    fairseq_embedder=fairseq_embedder,
                    active_embedder=active_embedder,
                    f0_up_key=cfg.f0_up_key,
                    sid=cfg.sid,
                    threshold=cfg.threshold,
                    interpolate_fn=_interpolate_voiced_gaps_np,
                    pitch_coarse_fn=_to_pitch_coarse,
                    t_cv0=time.perf_counter(),
                )
            except Exception:
                break


def _infer_impl(
    *,
    audio16k: NDArrayF32,
    cv: Any,
    rmvpe: Any,
    rvc: Any,
    is_half: bool,
    cv_input_dtype: str,
    rmvpe_input_dtype: str,
    fairseq_embedder: Any,
    active_embedder: str,
    f0_up_key: int,
    sid: int,
    threshold: float,
    interpolate_fn: Any,
    pitch_coarse_fn: Any,
    t_cv0: float,
) -> tuple[NDArrayF32, float, float, float, bool]:
    """Mirrors `engine._infer` exactly — same pipeline, same NaN
    sanitization, same per-stage timing collection. Returned as a
    plain tuple so the child can ship per-stage ms back to the
    parent over the metadata pipe.
    """
    # Step 1: contentvec features (or fairseq).
    if active_embedder == "fairseq" and fairseq_embedder is not None:
        feats = fairseq_embedder.extract(audio16k.astype(np.float32, copy=False))
    else:
        in_dtype = np.float16 if "float16" in cv_input_dtype else np.float32
        audio_in = audio16k.reshape(1, -1).astype(in_dtype)
        feats_raw = cv.run(["unit12"], {"audio": audio_in})[0]
        feats = feats_raw.astype(np.float32, copy=False)
    if np.isnan(feats).any():
        feats = np.nan_to_num(feats, nan=0.0)
    t_cv1 = time.perf_counter()

    # Step 2: rmvpe pitch.
    rm_dtype = np.float16 if "float16" in rmvpe_input_dtype else np.float32
    pitchf_raw = rmvpe.run(
        ["pitchf"],
        {
            "waveform": audio16k.reshape(1, -1).astype(rm_dtype),
            "threshold": np.array([threshold], dtype=rm_dtype),
        },
    )[0]
    pitchf = pitchf_raw.astype(np.float32).squeeze()
    pitchf = interpolate_fn(pitchf)
    t_rmvpe1 = time.perf_counter()

    # Step 3: rvc vocoder.
    feats_2x = np.repeat(feats, 2, axis=1)
    pitch_coarse, pitchf_aligned = pitch_coarse_fn(pitchf, target_len=feats_2x.shape[1])
    pitch_coarse = pitch_coarse[: feats_2x.shape[1]].reshape(1, -1)
    if f0_up_key != 0:
        pitchf_aligned = pitchf_aligned * (2.0 ** (f0_up_key / 12.0))
    pitchf_aligned = pitchf_aligned[: feats_2x.shape[1]].reshape(1, -1).astype(np.float32)

    feats_dtype = np.float16 if is_half else np.float32
    out = rvc.run(
        ["audio"],
        {
            "feats": feats_2x.astype(feats_dtype),
            "p_len": np.array([feats_2x.shape[1]], dtype=np.int64),
            "pitch": pitch_coarse,
            "pitchf": pitchf_aligned,
            "sid": np.array([sid], dtype=np.int64),
        },
    )[0]
    result = np.array(out).astype(np.float32).squeeze()
    nan_replaced = False
    if np.isnan(result).any() or np.isinf(result).any():
        result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)
        nan_replaced = True
    t_rvc1 = time.perf_counter()

    cv_ms = (t_cv1 - t_cv0) * 1000.0
    rmvpe_ms = (t_rmvpe1 - t_cv1) * 1000.0
    rvc_ms = (t_rvc1 - t_rmvpe1) * 1000.0

    return result, cv_ms, rmvpe_ms, rvc_ms, nan_replaced


__all__ = [
    "CMD_INFER",
    "CMD_STOP",
    "CMD_SWAP_MODEL",
    "RESP_DONE",
    "RESP_ERROR",
    "RESP_READY",
    "RESP_SWAP_DONE",
    "child_main",
]


# Allow direct invocation via `python -m audio.inference_worker` for
# debugging the child in isolation. Real spawning happens through
# `inference_client.InferenceClient`.
if __name__ == "__main__":
    print("audio.inference_worker is meant to be spawned by InferenceClient", file=sys.stderr)
    sys.exit(2)
