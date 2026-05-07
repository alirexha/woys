"""v0.8.0 — parent-side client for the inference subprocess.

Wraps the spawn / IPC / shutdown / restart machinery so the engine
can call `client.infer(audio16k)` and get back `(result, timings)`
without knowing whether inference ran in-process or in a child.

Wire protocol (see `audio.inference_worker`):

  Parent → Child (via `_parent_send` Pipe)
    {"cmd": "infer", "input_shape": tuple, "f0_up_key": int,
     "sid": int, "threshold": float}
    {"cmd": "swap_model", "path": str}
    {"cmd": "stop"}

  Child → Parent (via `_parent_recv` Pipe)
    {"cmd": "ready", "rvc_output_sr": int, "active_embedder": str,
     "is_half": bool}
    {"cmd": "done", "output_shape": tuple, "cv_ms": float,
     "rmvpe_ms": float, "rvc_ms": float, "nan_replaced": bool,
     "nan_chunks_total": int}
    {"cmd": "swap_done", "rvc_output_sr": int, "is_half": bool}
    {"cmd": "error", "error": str}

Shared memory layout:
  input_shm:  raw bytes for the audio16k float32 array. Sized to
              hold the largest plausible model_input length —
              `chunk_seconds * mic_rate * 4 + history(~10K) +
              search(~400)` rounded up to 64 KiB.
  output_shm: raw bytes for the inference result. Sized to hold
              the largest plausible vocoder output — same upper
              bound as input plus headroom for RVC v2 40 kHz
              models that emit ~6–8K samples per chunk → 64 KiB
              is comfortable.

Failure modes handled:
  - Child crashes mid-infer  → recv raises EOFError; client marks
    self dead, restarts on next infer call.
  - Parent crashes           → child watchdogs `os.getppid()` and
    exits when it becomes 1 (init reparented).
  - Stop with in-flight chunk → parent sends CMD_STOP after current
    recv; child drains and exits.
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import time
from dataclasses import dataclass, field
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from audio.inference_worker import (
    CMD_INFER,
    CMD_STOP,
    CMD_SWAP_MODEL,
    RESP_DONE,
    RESP_ERROR,
    RESP_READY,
    RESP_SWAP_DONE,
    child_main,
)

NDArrayF32 = npt.NDArray[np.float32]


# Default shared memory sizes. Must hold the largest plausible
# model_input / model_output. For chunk_seconds=0.15 + RVC v2 40k
# vocoder, the worst output is ~10 K samples × 4 bytes = 40 KB.
# 64 KB is comfortable for both directions and matches the OS pipe
# default. Bumped to 128 KB for output to handle 48k vocoder voices.
_INPUT_SHM_SIZE = 64 * 1024
_OUTPUT_SHM_SIZE = 128 * 1024


@dataclass
class InferenceTimings:
    """Per-call timing data returned alongside `result`. The engine
    uses these to populate `EngineStats.last_cv_ms / last_rmvpe_ms /
    last_rvc_ms` so woys diag shows the same per-stage breakdown
    regardless of in-process vs subprocess inference."""

    cv_ms: float = 0.0
    rmvpe_ms: float = 0.0
    rvc_ms: float = 0.0
    nan_replaced: bool = False
    nan_chunks_total: int = 0
    # Round-trip time including pipe send + child wake + inference +
    # response. The engine's existing `inference` percentile tracks
    # this — the difference between this and (cv+rmvpe+rvc) is the
    # IPC overhead.
    roundtrip_ms: float = 0.0


@dataclass
class _ChildHandles:
    """Lifecycle handles for one child process generation. We rebuild
    this on every (re)start so a crashed child can be cleanly torn
    down and replaced without leaking shm or pipe fds.

    B44 / quality-007: precise types instead of `Any`. mp.Process,
    multiprocessing.connection.Connection, and shared_memory.SharedMemory
    all have stubs in modern Python; the previous `Any` annotations
    let typo'd attribute accesses slip through mypy --strict.
    """

    proc: mp.Process
    parent_send: "mp.connection.Connection"
    parent_recv: "mp.connection.Connection"
    # The child-side ends; parent holds them open until spawn completes,
    # then closes its references so they live only inside the child.
    child_send_remote: "mp.connection.Connection"
    child_recv_remote: "mp.connection.Connection"
    input_shm: shared_memory.SharedMemory
    output_shm: shared_memory.SharedMemory
    rvc_output_sr: int = 16_000
    is_half: bool = False
    active_embedder: str = "onnx"
    last_error: str | None = field(default=None)


class InferenceError(RuntimeError):
    """Raised when the child reports an error or dies unexpectedly."""


class InferenceClient:
    """Parent-side handle to the inference subprocess.

    Single-instance per engine. Not thread-safe — the engine main
    loop is single-threaded so concurrent calls aren't an issue.
    """

    def __init__(self, cfg_dict: dict[str, Any]) -> None:
        self._cfg_dict = dict(cfg_dict)
        self._handles: _ChildHandles | None = None
        self._restart_count = 0

    # ---- lifecycle ---------------------------------------------------------

    def start(self, *, ready_timeout_s: float = 30.0) -> None:
        """Spawn the child, wait for RESP_READY. Raises InferenceError
        on startup failure (bad config, missing model files, etc.)."""
        if self._handles is not None and self._handles.proc.is_alive():
            return
        self._spawn_child(ready_timeout_s=ready_timeout_s)

    def _spawn_child(self, *, ready_timeout_s: float) -> None:
        """Build SharedMemory + Pipes + Process; wait for RESP_READY."""
        # Allocate shared memory regions. The child opens them by name.
        # We use unique names (random suffix in default mp impl) so
        # parallel woys instances don't collide.
        input_shm = shared_memory.SharedMemory(create=True, size=_INPUT_SHM_SIZE)
        output_shm = shared_memory.SharedMemory(create=True, size=_OUTPUT_SHM_SIZE)

        # Two simplex pipes — one each direction. Cleaner than duplex.
        parent_recv, child_send_remote = mp.Pipe(duplex=False)
        child_recv_remote, parent_send = mp.Pipe(duplex=False)

        # Spawn (NOT fork — CUDA contexts don't survive fork).
        ctx = mp.get_context("spawn")
        proc = ctx.Process(
            target=child_main,
            args=(
                input_shm.name,
                output_shm.name,
                child_recv_remote,  # child's read end
                child_send_remote,  # child's write end
                dict(self._cfg_dict),
                # Parent's PID at spawn time. Child watchdogs PPID and
                # exits if it becomes 1 (parent reparented to init).
                # Note: this isn't quite right for nested processes —
                # the child's parent is THIS process, which has pid =
                # mp.current_process().pid. But mp's process model
                # ensures the child's direct parent is us, and our PID
                # is what the child should track.
                __import__("os").getpid(),
            ),
            daemon=True,
            name="woys-inference-child",
        )
        proc.start()

        # The child copies the remote ends to itself via pickle. We
        # close our refs to them so we don't accidentally read/write
        # the wrong direction. (The pipe stays open in the child.)
        child_send_remote.close()
        child_recv_remote.close()

        # Block until the child reports ready or times out.
        deadline = time.perf_counter() + ready_timeout_s
        ready_msg: dict[str, Any] | None = None
        while time.perf_counter() < deadline:
            if not proc.is_alive():
                break
            if parent_recv.poll(timeout=0.1):
                ready_msg = parent_recv.recv()
                break

        if ready_msg is None:
            # Cleanup before raising.
            with contextlib.suppress(Exception):
                proc.terminate()
                proc.join(timeout=1.0)
            for shm in (input_shm, output_shm):
                with contextlib.suppress(Exception):
                    shm.close()
                    shm.unlink()
            raise InferenceError(
                f"child failed to send RESP_READY within {ready_timeout_s}s "
                f"(alive={proc.is_alive()})"
            )

        if ready_msg.get("cmd") == RESP_ERROR:
            err = ready_msg.get("error", "unknown")
            with contextlib.suppress(Exception):
                proc.terminate()
                proc.join(timeout=1.0)
            for shm in (input_shm, output_shm):
                with contextlib.suppress(Exception):
                    shm.close()
                    shm.unlink()
            raise InferenceError(f"child startup failed: {err}")

        if ready_msg.get("cmd") != RESP_READY:
            raise InferenceError(f"unexpected initial msg from child: {ready_msg!r}")

        self._handles = _ChildHandles(
            proc=proc,
            parent_send=parent_send,
            parent_recv=parent_recv,
            child_send_remote=child_send_remote,
            child_recv_remote=child_recv_remote,
            input_shm=input_shm,
            output_shm=output_shm,
            rvc_output_sr=int(ready_msg.get("rvc_output_sr", 16_000)),
            is_half=bool(ready_msg.get("is_half", False)),
            active_embedder=str(ready_msg.get("active_embedder", "onnx")),
        )

    def stop(self, *, timeout_s: float = 2.0) -> None:
        """Send STOP, wait for child to exit, clean up shm + pipes."""
        h = self._handles
        if h is None:
            return
        if h.proc.is_alive():
            with contextlib.suppress(Exception):
                h.parent_send.send({"cmd": CMD_STOP})
            h.proc.join(timeout=timeout_s)
            if h.proc.is_alive():
                h.proc.terminate()
                h.proc.join(timeout=1.0)
            if h.proc.is_alive():
                h.proc.kill()
                h.proc.join(timeout=1.0)
        # B15 / corr-027: confirm the child actually exited before unlinking
        # the shared-memory regions. mp's `proc.kill(); proc.join()` should
        # reap the PID via the shared resource_tracker, but defensively we
        # also poll `/proc/<pid>` to guarantee it's gone — without this,
        # an unlink while the child still maps the shm produces a
        # ResourceWarning (and on some kernels an OSError) that the
        # contextlib.suppress below would swallow, leaking the segment.
        if h.proc.pid is not None:
            proc_path = Path(f"/proc/{h.proc.pid}")
            wait_deadline = time.perf_counter() + 0.5
            while proc_path.exists() and time.perf_counter() < wait_deadline:
                time.sleep(0.01)
        # Close pipes.
        for c in (h.parent_send, h.parent_recv):
            with contextlib.suppress(Exception):
                c.close()
        # Free shm.
        for shm in (h.input_shm, h.output_shm):
            with contextlib.suppress(Exception):
                shm.close()
            # Narrow the suppress scope: only swallow FileNotFoundError
            # (already-cleaned by another process / kernel), not arbitrary
            # bugs.
            with contextlib.suppress(FileNotFoundError):
                shm.unlink()
        self._handles = None

    # ---- properties ---------------------------------------------------------

    @property
    def is_alive(self) -> bool:
        h = self._handles
        return h is not None and h.proc.is_alive()

    @property
    def rvc_output_sr(self) -> int:
        return self._handles.rvc_output_sr if self._handles else 16_000

    @property
    def is_half(self) -> bool:
        return self._handles.is_half if self._handles else False

    @property
    def active_embedder(self) -> str:
        return self._handles.active_embedder if self._handles else "onnx"

    @property
    def restart_count(self) -> int:
        return self._restart_count

    @property
    def last_error(self) -> str | None:
        return self._handles.last_error if self._handles else None

    # ---- hot path -----------------------------------------------------------

    def infer(
        self,
        audio16k: NDArrayF32,
        *,
        f0_up_key: int = 0,
        sid: int = 0,
        threshold: float = 0.3,
    ) -> tuple[NDArrayF32, InferenceTimings]:
        """Send `audio16k` to the child, block for the result.

        Raises InferenceError on child death; the engine's
        `_safe_process_streaming_16k` wrapper catches this and bumps
        `dropped_chunks` like it does for in-process exceptions.
        """
        h = self._handles
        if h is None:
            raise InferenceError("client not started")
        if not h.proc.is_alive():
            raise InferenceError("child not alive")

        flat = np.ascontiguousarray(audio16k.reshape(-1)).astype(np.float32, copy=False)
        nbytes = flat.size * 4
        if nbytes > h.input_shm.size:
            raise InferenceError(f"input size {nbytes} exceeds shm {h.input_shm.size}")
        in_buf = h.input_shm.buf
        assert in_buf is not None
        in_view: NDArrayF32 = np.ndarray((flat.size,), dtype=np.float32, buffer=in_buf)
        in_view[:] = flat

        t0 = time.perf_counter()
        try:
            h.parent_send.send(
                {
                    "cmd": CMD_INFER,
                    "input_shape": audio16k.shape,
                    "f0_up_key": f0_up_key,
                    "sid": sid,
                    "threshold": threshold,
                }
            )
            msg = h.parent_recv.recv()
        except (EOFError, BrokenPipeError, OSError) as e:
            raise InferenceError(f"child pipe died: {type(e).__name__}: {e}") from e

        if msg.get("cmd") == RESP_ERROR:
            err = msg.get("error", "unknown")
            h.last_error = err
            raise InferenceError(f"child reported error: {err}")
        if msg.get("cmd") != RESP_DONE:
            raise InferenceError(f"unexpected msg from child: {msg!r}")

        out_shape = tuple(msg["output_shape"])
        out_size = int(np.prod(out_shape))
        out_buf = h.output_shm.buf
        assert out_buf is not None
        # Copy out of shared memory before returning so the parent
        # owns the array — child may overwrite shm on the next infer.
        result = np.frombuffer(out_buf, dtype=np.float32, count=out_size).copy()
        result = result.reshape(out_shape)

        timings = InferenceTimings(
            cv_ms=float(msg.get("cv_ms", 0.0)),
            rmvpe_ms=float(msg.get("rmvpe_ms", 0.0)),
            rvc_ms=float(msg.get("rvc_ms", 0.0)),
            nan_replaced=bool(msg.get("nan_replaced", False)),
            nan_chunks_total=int(msg.get("nan_chunks_total", 0)),
            roundtrip_ms=(time.perf_counter() - t0) * 1000.0,
        )
        return result, timings

    def swap_model(self, new_path: Path, *, timeout_s: float = 5.0) -> tuple[int, bool]:
        """Tell the child to swap to `new_path`. Returns
        (new rvc_output_sr, is_half) on success. Cold-swap can take
        ~600 ms (model load + cuDNN tune); hot-swap ~10 ms (pool
        cache hit)."""
        h = self._handles
        if h is None or not h.proc.is_alive():
            raise InferenceError("child not alive")

        try:
            h.parent_send.send({"cmd": CMD_SWAP_MODEL, "path": str(new_path)})
            # Drain anything on the wire then look for swap_done.
            deadline = time.perf_counter() + timeout_s
            while time.perf_counter() < deadline:
                if h.parent_recv.poll(timeout=0.05):
                    msg = h.parent_recv.recv()
                    if msg.get("cmd") == RESP_SWAP_DONE:
                        h.rvc_output_sr = int(msg.get("rvc_output_sr", 16_000))
                        h.is_half = bool(msg.get("is_half", False))
                        return h.rvc_output_sr, h.is_half
                    if msg.get("cmd") == RESP_ERROR:
                        raise InferenceError(f"swap failed: {msg.get('error')}")
                    # Skip stray RESP_DONE from a racing infer (shouldn't
                    # happen with single-threaded engine but be defensive).
            raise InferenceError(f"swap timed out after {timeout_s}s")
        except (EOFError, BrokenPipeError, OSError) as e:
            raise InferenceError(f"swap pipe died: {type(e).__name__}: {e}") from e

    # ---- recovery -----------------------------------------------------------

    def restart(self, *, ready_timeout_s: float = 30.0) -> None:
        """Tear down + respawn the child. Called by the engine when
        InferenceError fires repeatedly. Bumps `restart_count`."""
        self._restart_count += 1
        # Best-effort teardown of the dead child.
        h = self._handles
        if h is not None:
            if h.proc.is_alive():
                with contextlib.suppress(Exception):
                    h.proc.terminate()
                    h.proc.join(timeout=1.0)
                if h.proc.is_alive():
                    h.proc.kill()
                    h.proc.join(timeout=1.0)
            for c in (h.parent_send, h.parent_recv):
                with contextlib.suppress(Exception):
                    c.close()
            for shm in (h.input_shm, h.output_shm):
                with contextlib.suppress(Exception):
                    shm.close()
                with contextlib.suppress(Exception):
                    shm.unlink()
            self._handles = None
        self._spawn_child(ready_timeout_s=ready_timeout_s)


__all__ = ["InferenceClient", "InferenceError", "InferenceTimings"]
