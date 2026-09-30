"""GPU anti-jitter for the engine: the NVIDIA clock lock (with the
SIGTERM/SIGINT handlers that revert it) and the two GPU keep-alive loops.

`_GpuTuningMixin` holds these `RealtimeEngine` methods; the engine's
`__init__` creates the state they share (declared below for type checking).
Split out of `audio.engine`.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import subprocess
import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import onnxruntime as ort

from audio.engine_config import EngineConfig
from audio.engine_stats import EngineStats

NDArrayF32 = npt.NDArray[np.float32]


class _GpuTuningMixin:
    """Clock-lock, signal-handler and keep-alive methods of `RealtimeEngine`."""

    cfg: EngineConfig
    stats: EngineStats
    _stats_lock: Any
    _stop_event: threading.Event
    _cv: ort.InferenceSession | None
    _cv_input_dtype: str
    _keepalive_input: NDArrayF32 | None
    _prior_signal_handlers: dict[int, Any]
    _signal_received: int | None

    if TYPE_CHECKING:

        def record_error(self, msg: str) -> None: ...

        def _apply_thread_priority(self, *, label: str, priority: int = 60) -> None: ...

    def _resolve_anti_jitter_flags(self) -> tuple[bool, bool]:
        """Map `cfg.gpu_anti_jitter_mode` (the user-facing knob) to the
        two underlying booleans (clock_lock, torch_keepalive). The
        booleans take precedence when the mode is "off"; the mode field
        wins when set to anything else.

        Returns (clock_lock_on, torch_keepalive_on)."""
        mode = (self.cfg.gpu_anti_jitter_mode or "off").strip().lower()
        if mode == "off":
            return self.cfg.gpu_clock_lock_enabled, self.cfg.gpu_keepalive_torch_stream
        if mode == "keepalive":
            return False, True
        if mode == "clock_lock":
            return True, False
        if mode == "both":
            return True, True
        # Unknown value - log to last_error, fall back to off.
        self.record_error(
            f"unknown gpu_anti_jitter_mode={mode!r}; expected "
            f"off|keepalive|clock_lock|both. Falling back to off."
        )
        return False, False

    @staticmethod
    def _query_max_graphics_clock_mhz() -> int:
        """Return `clocks.max.graphics` MHz from `nvidia-smi` or 0 on
        failure (caller must handle the 0 case)."""
        try:
            res = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=clocks.max.graphics",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=4.0,
            )
        except (OSError, subprocess.TimeoutExpired):
            return 0
        if res.returncode != 0:
            return 0
        first = res.stdout.strip().splitlines()[0].strip() if res.stdout.strip() else ""
        try:
            value = int(float(first))
        except ValueError:
            return 0
        # Sanity: anything outside [600, 4000] is suspicious for a modern GPU
        # (Pascal-and-newer min ~600, Ada-class peak ~3500).
        if value < 600 or value > 4000:
            return 0
        return value

    def _resolve_clock_lock_range(self) -> tuple[int, int]:
        """Decide the (floor_mhz, ceiling_mhz) pair to pass to
        `nvidia-smi -lgc`. Honors the user's explicit fields; otherwise
        auto-detects from `clocks.max.graphics`.

        Returns (floor, ceiling). Raises RuntimeError if the values
        violate sanity (out-of-range, floor > ceiling, etc.).
        """
        max_graphics = self._query_max_graphics_clock_mhz()
        # Sentinel 0 (or any non-positive) means auto-detect.
        if int(self.cfg.gpu_clock_lock_floor_mhz) > 0:
            floor = int(self.cfg.gpu_clock_lock_floor_mhz)
        else:
            if max_graphics == 0:
                raise RuntimeError(
                    "auto-detect of gpu_clock_lock_floor_mhz failed: "
                    "nvidia-smi --query-gpu=clocks.max.graphics returned no usable value. "
                    "Set gpu_clock_lock_floor_mhz explicitly in config.toml."
                )
            floor = max(600, max_graphics - max(0, self.cfg.gpu_clock_lock_floor_offset_mhz))

        if int(self.cfg.gpu_clock_lock_ceiling_mhz) > 0:
            ceiling = int(self.cfg.gpu_clock_lock_ceiling_mhz)
        elif max_graphics > 0:
            ceiling = max_graphics
        else:
            # Fall back to floor if we somehow have neither.
            ceiling = floor

        if floor < 600 or ceiling < floor or ceiling > 4000:
            raise RuntimeError(
                f"resolved clock-lock range (floor={floor}, ceiling={ceiling}) is out of "
                f"sanity bounds [600, 4000] or floor>ceiling. Check "
                f"gpu_clock_lock_floor_mhz / gpu_clock_lock_ceiling_mhz / "
                f"gpu_clock_lock_floor_offset_mhz in config.toml."
            )

        # The brief's hard constraint: clock-lock must use stock or
        # sub-stock values only. We treat `clocks.max.graphics` as
        # NVIDIA's documented stock ceiling for this card. If the user's
        # explicit ceiling overshoots that, refuse - the assistant will
        # not enable an over-stock-spec lock.
        if max_graphics > 0 and ceiling > max_graphics:
            raise RuntimeError(
                f"gpu_clock_lock_ceiling_mhz={ceiling} exceeds "
                f"clocks.max.graphics={max_graphics}; over-stock locks are "
                f"refused per the v0.11.0 hard-constraint policy."
            )

        return floor, ceiling

    def _run_nvidia_smi(self, args: list[str], *, timeout: float = 6.0) -> tuple[bool, str]:
        """Run `sudo nvidia-smi <args>`, return (ok, message). Captures
        both stdout and stderr; treats nonzero exit OR empty output OR
        the literal string "error" in output as a failure. Refuses to
        run if `nvidia-smi` is not on PATH.
        """
        if shutil.which("nvidia-smi") is None:
            return False, "nvidia-smi not on PATH"
        cmd = ["sudo", "-n", "nvidia-smi", *args]
        try:
            res = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as e:
            return False, f"{type(e).__name__}: {e}"
        out = (res.stdout or "").strip()
        err = (res.stderr or "").strip()
        merged = "\n".join(s for s in (out, err) if s).strip()
        if res.returncode != 0:
            return False, f"exit={res.returncode}: {merged or '<no output>'}"
        # nvidia-smi -lgc happy path includes "All done." in stdout.
        if "error" in merged.lower():
            return False, f"nvidia-smi reported error: {merged}"
        return True, merged

    def _apply_gpu_clock_lock(self) -> None:
        """v0.11.0 - apply nvidia-smi -lgc <floor>,<ceiling>. Hard-fails
        the engine start on any unexpected output / exit code.

        v0.14.0 (area 17 / area 19 / C019): if the previous session's
        revert failed (`gpu_clock_lock_revert_failed=True`), attempt a
        fresh -rgc before applying the new lock. Without this, a stuck
        lock from a prior session compounds with the new -lgc and the
        GPU stays at boost-clocks indefinitely.
        """
        if self.stats.gpu_clock_lock_revert_failed:
            ok, msg = self._run_nvidia_smi(["-rgc"], timeout=4.0)
            if ok:
                self.stats.gpu_clock_lock_revert_failed = False
            self.stats.gpu_clock_lock_last_message = (f"recovery -rgc on prior failure: {msg}")[
                :200
            ]
        floor, ceiling = self._resolve_clock_lock_range()
        ok, msg = self._run_nvidia_smi(["-lgc", f"{floor},{ceiling}"])
        self.stats.gpu_clock_lock_last_message = msg[:200]
        if not ok:
            raise RuntimeError(
                f"gpu_clock_lock_enabled=True but nvidia-smi -lgc {floor},{ceiling} failed:\n"
                f"  {msg}\n"
                f"Check that:\n"
                f"  - nvidia-smi is on PATH and the NVIDIA driver is loaded\n"
                f"  - sudo is configured for `sudo -n nvidia-smi -lgc/-rgc` (see docs/22-gpu-clock-lock.md)\n"
                f"  - the floor/ceiling values are within stock spec for this GPU\n"
                f"To disable, set gpu_clock_lock_enabled=false (or gpu_anti_jitter_mode=off) "
                f"in ~/.config/woys/config.toml."
            )
        self.stats.gpu_clock_lock_active = True
        self.stats.gpu_clock_lock_floor_mhz = floor
        self.stats.gpu_clock_lock_ceiling_mhz = ceiling

    def _install_signal_handlers(self) -> None:
        """v0.14.0 (area 17 / C010): always install SIGTERM/SIGINT handlers
        at engine.start, regardless of clock-lock state. Pre-v0.14.0 the
        handler was installed only inside `_apply_gpu_clock_lock`; with
        the default `gpu_anti_jitter_mode="off"` config, signal-delivered
        death used Python's default (terminate immediately) and orphaned
        the inference subprocess + leaked the writer queue.

        Only install on the main thread (signal.signal raises ValueError
        otherwise). Prior handlers (Textual's, CLI's) are saved and
        re-installed by `_revert_gpu_clock_lock` on clean stop OR by the
        signal handler itself before re-raising.
        """
        if threading.current_thread() is not threading.main_thread():
            return
        for sig in (signal.SIGTERM, signal.SIGINT):
            # Some environments (Textual TUI inside async loop) don't
            # let us install handlers; that's fine, engine.stop() still
            # cleans up on the normal exit path.
            with contextlib.suppress(OSError, ValueError):
                prior = signal.signal(sig, self._signal_handler_revert_lock)
                # Our own handler is still installed when the last stop()
                # ran off the main thread and could not restore; saving it
                # as the "prior" would make SIGTERM re-raise into itself.
                # Keep the real prior from the earlier run instead.
                if prior != self._signal_handler_revert_lock:
                    self._prior_signal_handlers[sig] = prior

    def _restore_prior_signal_handlers(self) -> None:
        """Re-install the SIGTERM/SIGINT handlers that were active before
        the engine started (Textual's, the CLI's).

        split out of `_revert_gpu_clock_lock`
        so the signal handler can restore handlers without also triggering
        the `sudo nvidia-smi` fork. Fast and fork-free -- safe to call from
        the signal handler itself.

        Off the main thread (the TUI stops the engine on a worker thread)
        signal.signal() cannot run, so the saved handlers are kept for the
        next main-thread restore instead of being dropped.
        """
        if threading.current_thread() is not threading.main_thread():
            return
        for sig, prior in self._prior_signal_handlers.items():
            with contextlib.suppress(OSError, ValueError):
                signal.signal(sig, prior)
        self._prior_signal_handlers.clear()

    def _revert_gpu_clock_lock(self) -> None:
        """v0.11.0 - call nvidia-smi -rgc and restore prior signal
        handlers. Idempotent (safe to call multiple times); a second call
        when the lock isn't active just no-ops.

        v0.14.0 (area 17 / area 19 / C019): track revert success in a
        separate `gpu_clock_lock_revert_failed` flag. Pre-v0.14.0 the
        function set `gpu_clock_lock_active=False` whether or not -rgc
        succeeded, so a sudo-revoked failure left the GPU locked but the
        engine flagged it as released. Next start saw "fresh state" and
        applied a new lock on top of the stale one. The new flag lets
        `_apply_gpu_clock_lock` detect and recover.

        the `sudo nvidia-smi -rgc`
        call here forks a subprocess and can block up to its timeout. It is
        therefore called only on a normal stack -- from `stop()` -- never
        from the signal handler. The bound is `subprocess.run(timeout=...)`
        inside `_run_nvidia_smi`: a hung nvidia-smi cannot wedge teardown
        past that 4 s.
        """
        if self.stats.gpu_clock_lock_active:
            ok, msg = self._run_nvidia_smi(["-rgc"], timeout=4.0)
            self.stats.gpu_clock_lock_last_message = msg[:200]
            if ok:
                self.stats.gpu_clock_lock_active = False
                self.stats.gpu_clock_lock_revert_failed = False
            else:
                # Keep gpu_clock_lock_active=True so the next start
                # detects a stale lock and attempts recovery -rgc; set
                # the dedicated failure flag so it isn't ambiguous.
                self.stats.gpu_clock_lock_revert_failed = True
                self.record_error(
                    f"nvidia-smi -rgc failed at engine stop: {msg}. "
                    f"Next engine.start() will retry; or run "
                    f"`sudo nvidia-smi -rgc` manually."
                )

        # Restore prior signal handlers (whether or not -rgc succeeded -
        # signal handlers should always go back to caller's pre-engine
        # state on stop).
        self._restore_prior_signal_handlers()

    def _signal_handler_revert_lock(self, signum: int, frame: object) -> None:
        """SIGTERM / SIGINT handler that coordinates clean shutdown.
        Best-effort: a SIGKILL bypasses this entirely.

        the pre-fix handler called
        `_revert_gpu_clock_lock()` inline, which forks `sudo nvidia-smi`
        and can block the main thread up to 4 s -- async-signal-unsafe
        work run on every SIGTERM/SIGINT. A Ctrl-C could hang the process
        and, if it landed mid-`subprocess.run`, deadlock. The handler now
        does only fast, fork-free work:
          1. Record the signal (also a re-entrancy guard for a repeated
             SIGTERM/SIGINT).
          2. Set `_stop_event` so the engine threads exit their loops
             cleanly (drain pacat, release ORT sessions).
          3. Restore the prior (Textual / CLI) signal handlers.
          4. Re-raise via `os.kill` so that prior handler runs the rest of
             the shutdown.
        The GPU clock-lock revert -- the unsafe part -- now happens on a
        normal stack in `stop()` via `_revert_gpu_clock_lock()`. It is
        idempotent, and `_apply_gpu_clock_lock` recovers a stale lock on
        the next start if a hard kill skips `stop()` entirely.

        v0.14.0 (area 8 / area 17 / C005) note retained: restoring the
        prior handler *before* re-raising (rather than a `SIG_DFL` clobber)
        is what keeps Textual's / the CLI's clean-shutdown chain intact.
        """
        if self._signal_received is not None:
            # A second signal arrived while we were mid-handler. The prior
            # handler is (being) restored; just re-raise and let it take
            # over -- don't redo _stop_event / handler-restore work.
            self._drop_own_handler(signum)
            os.kill(os.getpid(), signum)
            return
        self._signal_received = signum
        self._stop_event.set()
        self._restore_prior_signal_handlers()
        self._drop_own_handler(signum)
        # Re-raise. The prior handler is now installed; the kernel delivers
        # this signal to it. The clock-lock revert runs later, on a normal
        # stack, in stop().
        os.kill(os.getpid(), signum)

    def _drop_own_handler(self, signum: int) -> None:
        """If no prior handler was restored (none was saved), this handler
        is still installed and the re-raise would land right back here
        until RecursionError. Fall back to the default action instead."""
        with contextlib.suppress(OSError, ValueError):
            if signal.getsignal(signum) == self._signal_handler_revert_lock:
                signal.signal(signum, signal.SIG_DFL)

    def _torch_keepalive_loop(self) -> None:
        """v0.11.0 - torch.cuda.Stream() based keepalive.

        Replaces the rc3 ORT-stream keepalive when
        `gpu_keepalive_torch_stream=True` (or
        `gpu_anti_jitter_mode in {"keepalive","both"}`). The op is a tiny
        `tensor.add(1.0)` (1024 fp32 elements ≈ 50 µs of GPU work) issued
        on a torch CUDA stream that is NOT shared with ORT's session
        stream - the GPU scheduler can interleave them without
        serialization, closing the rc3 contention regression.

        Failure-safe: any exception during stream creation or the hot
        loop logs to `stats.last_error` and exits the thread cleanly.
        Engine continues running without keepalive in that case."""
        try:
            import torch
        except ImportError as e:
            self.record_error(
                f"torch import failed; torch keepalive disabled: {e}. "
                f"Install via `pip install torch` or set gpu_anti_jitter_mode=off."
            )
            return

        if not torch.cuda.is_available():
            self.record_error(
                "torch.cuda.is_available() returned False; torch keepalive disabled. "
                "Check that torch was built with CUDA support and an NVIDIA driver is loaded."
            )
            return

        try:
            stream = torch.cuda.Stream()  # type: ignore[no-untyped-call]  # torch's Stream stub lacks annotations
            buf = torch.empty(1024, device="cuda", dtype=torch.float32)
        except Exception as e:
            self.record_error(
                f"torch keepalive setup failed; thread exiting: {type(e).__name__}: {e}"
            )
            return

        # Lower priority than engine + writer so audio path always wins
        # CPU contention. The RT priority is best-effort; failure is
        # captured in priority_warnings, doesn't block the loop.
        self._apply_thread_priority(label="torch-keepalive", priority=40)

        interval_s = max(0.005, self.cfg.gpu_keepalive_torch_interval_ms / 1000.0)
        ema_alpha = 0.05
        running_avg = 0.0
        next_tick = time.perf_counter() + interval_s

        while not self._stop_event.is_set():
            now = time.perf_counter()
            if now < next_tick:
                self._stop_event.wait(timeout=min(0.020, next_tick - now))
                continue
            t0 = time.perf_counter()
            try:
                with torch.cuda.stream(stream):
                    buf = buf.add(1.0)
                # Don't synchronize - we want the GPU command queue to
                # absorb the op without blocking; the kernel launch alone
                # is enough to keep the boost from idling.
            except Exception as e:
                self.record_error(
                    f"torch keepalive crash; retiring thread: {type(e).__name__}: {e}"
                )
                break
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            with self._stats_lock:
                self.stats.torch_keepalive_calls += 1
                self.stats._recent_torch_keepalive_ms.append(elapsed_ms)
            self.stats.torch_keepalive_last_ms = elapsed_ms
            running_avg = ema_alpha * elapsed_ms + (1.0 - ema_alpha) * running_avg
            self.stats.torch_keepalive_avg_ms = running_avg
            next_tick = max(next_tick + interval_s, time.perf_counter())

    def _keepalive_loop(self) -> None:
        """v0.10.0-rc3 - periodic tiny ORT op to keep the GPU at boosted
        clock state during the engine's idle gaps.

        Background: the v0.10.0-rc1/rc2 evidence (LESSONS §29) showed
        the laptop GPU's dynamic boost backs off during the ~98 ms
        mic_read window between chunks. Each chunk's RVC then pays a
        variable reboost-recovery cost (rvc.run p99 = 68 ms vs p50 =
        33 ms). nvidia-smi clock log showed 34 % of samples > 100 MHz
        below median.

        Implementation: at `gpu_keepalive_interval_ms` cadence, run
        the cv (contentvec) ONNX session on a small dummy input
        (`gpu_keepalive_input_len` samples). The session is shared
        with the engine's `_extract_feats` path; ORT serializes
        concurrent `session.run()` calls internally on the same
        CUDA stream, so this op queues if engine is busy and runs
        if engine is idle - which is the desired behavior.

        Cost: ~1-3 ms of GPU work per call. At 25 ms cadence that's
        ~5-12 % continuous GPU duty cycle. The intent is to keep
        utilization above the dynamic-boost deboost threshold.

        Defensive: any exception inside the run is silently dropped.
        Goal is "do something on the GPU", not "produce useful output."
        """
        if self._cv is None or self._keepalive_input is None:
            return
        # Pre-warm the keepalive shape with EXHAUSTIVE-cuDNN-cached
        # algos. If we don't, the first keepalive call hits a cold
        # cuDNN path (~80 ms) which would itself cause a one-off
        # writer jitter spike.
        try:
            in_dtype = np.float16 if "float16" in self._cv_input_dtype else np.float32
            warm_in: np.ndarray = self._keepalive_input.reshape(1, -1).astype(in_dtype)  # type: ignore[type-arg]
            for _ in range(2):
                self._cv.run(["unit12"], {"audio": warm_in})
        except Exception as e:
            with self._stats_lock:
                self.stats.priority_warnings.append(
                    f"gpu-keepalive warmup failed: {type(e).__name__}: {e}; thread will exit"
                )
            return

        # v0.10.0-rc3 - keepalive runs at lower priority than the engine
        # main / writer; the audio path always wins same-class tie-breaks.
        self._apply_thread_priority(label="keepalive", priority=40)

        interval_s = max(0.005, self.cfg.gpu_keepalive_interval_ms / 1000.0)
        in_dtype = np.float16 if "float16" in self._cv_input_dtype else np.float32
        dummy_in: np.ndarray = self._keepalive_input.reshape(1, -1).astype(in_dtype)  # type: ignore[type-arg]

        # Track running average so the diag surface can show keepalive cost.
        ema_alpha = 0.05
        running_avg = 0.0
        next_tick = time.perf_counter() + interval_s

        while not self._stop_event.is_set():
            now = time.perf_counter()
            if now < next_tick:
                # Use a short timeout-based wait so we react quickly to stop_event.
                self._stop_event.wait(timeout=min(0.020, next_tick - now))
                continue
            t0 = time.perf_counter()
            try:
                self._cv.run(["unit12"], {"audio": dummy_in})
            except Exception:
                # Bail on persistent error - don't spam stats.last_error.
                break
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            with self._stats_lock:
                self.stats.keepalive_calls += 1
                self.stats._recent_keepalive_ms.append(elapsed_ms)
            self.stats.last_keepalive_ms = elapsed_ms
            running_avg = ema_alpha * elapsed_ms + (1.0 - ema_alpha) * running_avg
            self.stats.keepalive_avg_ms = running_avg
            next_tick = max(next_tick + interval_s, time.perf_counter())
