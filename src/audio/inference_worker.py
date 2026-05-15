"""v0.8.0 - inference subprocess.

Closes the LESSONS §19 threading tax (~23 ms typical-case overhead
when ORT inference runs in the engine's daemon thread alongside the
writer / watchdog / stderr-reader threads, all contending for the
GIL during numpy ops between ONNX sessions).

The child process:
  - Owns its own CUDA context (forced via `spawn` start method, never
    `fork` - CUDA contexts don't survive fork).
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
# graph tiny - child doesn't need to import the engine module just
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
    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()


# B47 / quality-013: the priority logic moved into `audio.priority`. Kept as
# a thin alias to preserve any external imports that referenced the old name.
from audio.priority import try_set_realtime_priority as _try_set_rt_priority  # noqa: E402


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
    `multiprocessing.connection.Connection` objects - multiprocessing
    pickles them correctly across spawn so the child receives live
    Connection handles (NOT raw fds, which would be invalid in the
    new process).

    Args:
      parent_recv: Connection the child reads commands FROM (parent's
        write end - i.e. parent calls `.send(...)` on the other end,
        child calls `.recv()` here).
      child_send: Connection the child writes responses TO.

    The child loop:
      1. Open the SharedMemory regions by name.
      2. Load + warm ORT sessions.
      3. Send RESP_READY to parent.
      4. Loop: recv command, dispatch, send response.
      5. Exit on CMD_STOP or parent-death.
    """
    # No reconstruction needed - Connection objects come through spawn intact.

    # review F-merged-014 (P1): the inference child is a separate
    # `spawn`ed process -- wire it into the same rotating log file so a
    # child-side crash is on disk next to the parent's records.
    try:
        import logging

        from woys.logsetup import setup_logging

        setup_logging()
        logging.getLogger("woys.inference-worker").info(
            "inference child started (pid=%s, parent=%s)", os.getpid(), parent_pid
        )
    except Exception as e:
        # Logging setup is best-effort observability, not the child's core
        # job -- never let it abort inference. Not silent: the child's
        # stderr is read by the parent's _stderr_reader_loop.
        print(f"[inference-worker] logging setup failed: {e}", file=sys.stderr)

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

    # Step 3: build a RealtimeEngine instance in legacy in-process mode
    # and let it load sessions / pre-warm. The child then routes every
    # CMD_INFER through `eng._infer()` - the SAME method the
    # in-process path runs. This guarantees the child and in-process
    # paths execute byte-identical inference logic; v0.8.0-rc1/rc2's
    # parallel `_infer_impl` had drifted (or appeared to drift via
    # subtle layout / view differences) from `_infer`, producing
    # garbled audio in production despite shape-correct IPC.
    try:
        _ort_preload_dlls()
        # Make `audio.engine` importable from the child.
        _src_root = Path(__file__).resolve().parent.parent
        if str(_src_root) not in sys.path:
            sys.path.append(str(_src_root))

        from audio.engine import EngineConfig, RealtimeEngine

        # Reconstruct the same EngineConfig the parent has, but force
        # `inference_subprocess=False` so this in-child engine uses
        # the legacy in-process inference (no recursive child spawn).
        cfg_kwargs = {k: v for k, v in cfg_dict.items() if k in EngineConfig.__dataclass_fields__}
        cfg_kwargs["inference_subprocess"] = False
        cfg = EngineConfig(**cfg_kwargs)

        eng = RealtimeEngine(cfg)
        eng._ensure_sessions()
        # _ensure_sessions populated _cv / _rmvpe / _rvc / dtypes /
        # _is_half / _rvc_output_sr / active_embedder / _fairseq.
        eng._warmup_realtime_pipeline()  # broader-shape pre-warm (rc9)
        rvc_output_sr = eng._rvc_output_sr
        is_half = eng._is_half
        active_embedder = eng.active_embedder
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
        # Watchdog parent. Any change in PPID means the original parent died:
        # on Linux, PIDs don't migrate, so getppid() != parent_pid is sufficient.
        # The previous `and getppid() == 1` gate was wrong on systemd-userspace
        # systems (Arch, CachyOS, modern Fedora/Ubuntu) where orphaned user
        # processes reparent to `systemd --user`, NOT pid 1 (init). Without
        # this fix the child stays alive after parent crash, holding GPU memory.
        if os.getppid() != parent_pid:
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
                # Same hot-swap path the in-process engine uses -
                # `reload_rvc()` updates the RVC session via the
                # cached pool, refreshes _is_half, recomputes the
                # output sample rate, and resets streaming state.
                eng.reload_rvc(new_path)
                rvc_output_sr = eng._rvc_output_sr
                is_half = eng._is_half
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
                int(np.prod(shape))
                in_buf = input_shm.buf
                assert in_buf is not None
                # Copy out of shared memory into a fresh contiguous
                # buffer. ORT may capture references to numpy arrays
                # internally; if it captures a view into shm, the
                # parent's next write could mutate ORT's input
                # mid-flight. Copying once decouples ORT from shm.
                audio16k_view: NDArrayF32 = np.ndarray(shape, dtype=np.float32, buffer=in_buf)
                audio16k: NDArrayF32 = np.ascontiguousarray(audio16k_view).copy()
                # Per-call knobs - mutate the in-child engine's cfg so
                # `eng._infer` reads the parent's intent. self.cfg is
                # the canonical source for f0/sid/threshold inside
                # `_infer`, so this is the single point of control.
                eng.cfg.f0_up_key = int(msg.get("f0_up_key", cfg.f0_up_key))
                eng.cfg.sid = int(msg.get("sid", cfg.sid))
                eng.cfg.threshold = float(msg.get("threshold", cfg.threshold))

                # Snapshot per-stage timings BEFORE the call (eng's
                # _infer mutates self.stats.last_cv_ms etc.).
                prev_nan_chunks = eng.stats.nan_chunks
                time.perf_counter()
                # SAME _infer the in-process path runs. Guarantees
                # bit-identical inference logic across the two paths;
                # any divergence between paths now lives in the IPC
                # boundary, not the inference algorithm itself.
                result = eng._infer(audio16k)
                # Per-stage timings come from eng.stats, populated by
                # _infer before it returned.
                cv_ms = eng.stats.last_cv_ms
                rmvpe_ms = eng.stats.last_rmvpe_ms
                rvc_ms = eng.stats.last_rvc_ms
                nan_replaced = eng.stats.nan_chunks > prev_nan_chunks
                if nan_replaced:
                    nan_chunks_total += 1

                # Write result into the output shared memory.
                flat = np.ascontiguousarray(result.reshape(-1)).astype(np.float32, copy=False)
                out_bytes = flat.size * 4
                # Defensive bound check - parent allocated the shm with
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
