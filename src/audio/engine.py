"""Realtime voice-conversion engine.

Wires the Phase 1 ONNX inference path to a real-time mic→infer→sink loop.

Audio routing - IMPORTANT (see v0.1.1 fix)
------------------------------------------
On CachyOS, PortAudio is built with the ALSA host API only (no PulseAudio host
API). `sd.OutputStream()` with no explicit `device=` falls through to the ALSA
*default* device, which routes to the system default sink (laptop speakers /
headphones) - NOT to the named PipeWire sink we want. Setting `PULSE_SINK=…`
in the environment is also ignored, because there's no Pulse host API for
PortAudio to consult.

The fix: instead of `sd.OutputStream`, the engine spawns
`pacat --playback --device=WoysSink …` as a subprocess and pipes
raw float32 PCM to its stdin. `pacat` is the canonical PulseAudio client; it
talks to pipewire-pulse natively, takes an explicit `--device=` argument, and
never auto-routes to the system default. This is the same path that the
acoustic loopback bench (`scripts/bench_loopback.py`) uses - proven on this host.

Input is still `sd.InputStream` against the default mic; that path was always
correct (host mic → 48 kHz capture).

Optional local monitoring
-------------------------
By default, **the engine writes the transformed audio to ONLY the virtual
sink** (which `woys-mic` reads from). Nothing plays out of the laptop
speakers - your housemates / streamers / phone calls don't hear what you're
processing. Pass `monitor=True` to additionally play to the host's default
output for self-monitoring.

Threading
---------
The engine runs across 5-6 threads:
- `woys-engine` worker thread: blocking I/O loop, audio capture, inference
  dispatch, SOLA blending. Owns most reads of `EngineStats`.
- writer thread (`_writer_loop`): drains the converted-audio queue,
  writes to pacat / pw-cat / native_pw stream. Mutates xruns / writer-
  interval counters.
- pacat-watchdog thread: watches the playback subprocess's stderr/exit;
  appends to `stats.helper_exit_reasons` on death.
- (optional) GPU keepalive threads: torch-stream / clock-lock keep-alive
  loops. Each may bump its own counter.
- TUI thread: polls `EngineStats` for live UI.
- Signal-handler thread: SIGTERM/SIGINT coordination.

`EngineStats` is shared mutable
state. `stats.<counter> += 1` is LOAD/BINARY_OP/STORE_ATTR (the GIL
guarantees each bytecode, not the triple -- textbook lost-update);
`helper_exit_reasons.append() + len(...) > 10: pop(0)` from two
threads (engine.py:3084-3086 and :3608-3610) violates the `len <= 10`
invariant on race. `self._stats_lock` (`threading.RLock`) serializes
every `+=` and every `append+len-check+pop` block.
"""

from __future__ import annotations

import contextlib
import gc
import logging
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import onnxruntime as ort

from audio.engine_config import DEFAULT_CONTENTVEC as DEFAULT_CONTENTVEC
from audio.engine_config import DEFAULT_RMVPE as DEFAULT_RMVPE
from audio.engine_config import DEFAULT_RVC_MODEL as DEFAULT_RVC_MODEL
from audio.engine_config import MODELS_DIR as MODELS_DIR
from audio.engine_config import USER_VISIBLE_ENGINE_FIELDS as USER_VISIBLE_ENGINE_FIELDS
from audio.engine_config import EngineConfig as EngineConfig
from audio.engine_stats import _STATS_KEPT_ACROSS_RUNS
from audio.engine_stats import EngineStats as EngineStats
from audio.gpu_tuning import _GpuTuningMixin
from audio.inference import _InferenceMixin
from audio.pitch import _VOICED_GAP_MAX_FRAMES as _VOICED_GAP_MAX_FRAMES
from audio.pitch import _interpolate_voiced_gaps_np as _interpolate_voiced_gaps_np
from audio.pitch import _to_pitch_coarse as _to_pitch_coarse
from audio.pitch import interpolate_voiced_gaps_np as interpolate_voiced_gaps_np
from audio.pitch import to_pitch_coarse as to_pitch_coarse
from audio.playback import _PlaybackMixin
from audio.playback import _set_pdeathsig as _set_pdeathsig
from audio.resample import _resample as _resample
from audio.resample import _StreamResampler as _StreamResampler
from audio.sessions import _CUDNN_ALGO_SEARCH as _CUDNN_ALGO_SEARCH
from audio.sessions import _TRT_ACTIVE_PER_SESSION as _TRT_ACTIVE_PER_SESSION
from audio.sessions import _TRT_CACHE_ROOT as _TRT_CACHE_ROOT
from audio.sessions import _TRT_INIT_ERRORS as _TRT_INIT_ERRORS
from audio.sessions import _TRT_PRELOAD_OK as _TRT_PRELOAD_OK
from audio.sessions import CpuFallbackError as CpuFallbackError
from audio.sessions import RvcSessionPool as RvcSessionPool
from audio.sessions import _assert_session_gpu_bound as _assert_session_gpu_bound
from audio.sessions import _cuda_provider_entry as _cuda_provider_entry
from audio.sessions import _make_session as _make_session
from audio.sessions import _preload_trt_dlls as _preload_trt_dlls
from audio.sessions import _session_is_cpu_only as _session_is_cpu_only
from audio.sessions import _trt_cache_dir_for as _trt_cache_dir_for

NDArrayF32 = npt.NDArray[np.float32]
NDArrayI64 = npt.NDArray[np.int64]


@dataclass
class _SwapRequest:
    """a single queued model-swap with a
    per-call completion event. The TUI / socket caller holds the
    `completion` event after `request_model_swap()` returns and waits
    on IT specifically (not a shared broadcast Event) so two rapid
    swaps cannot collapse into one false-done.

    `error` is set by `_maybe_swap_model` when the swap fails (e.g.,
    subprocess InferenceError) so the caller can distinguish "done"
    from "failed". Pre-fix the single broadcast Event had no way to
    convey failure.
    """

    target: Path
    completion: threading.Event = field(default_factory=threading.Event)
    error: BaseException | None = None


# B60 / audio-012: `_resample_linear` (the known-bad reference baseline)
# was deleted in v0.8.0. Production path used `_resample` (soxr); the linear
# variant existed only to fail v0.5.1 quality tests. No callers in src/ or
# tests/. If you need it back as a benchmark, see `scripts/bench_*.py` or
# git history.


class RealtimeEngine(_InferenceMixin, _PlaybackMixin, _GpuTuningMixin):
    """Owns the 3 ONNX sessions and a worker thread that loops mic→infer→sink."""

    def __init__(self, cfg: EngineConfig | None = None) -> None:
        self.cfg = cfg or EngineConfig()
        self.stats = EngineStats()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        # serialize the whole body of start()
        # and stop() so:
        # (a) two concurrent stop() calls (signal-handler path +
        #     action_quit + CLI teardown, see F-CX3-01) don't both
        #     pass `self._inf_client is not None` and double-tear-down,
        # (b) a stop() arriving during start()'s multi-second warmup
        #     window (SIGTERM during autostart) waits for warmup to
        #     finish before tearing down -- avoiding the "running"
        #     engine spawned after the stop signal.
        # `_stopped` is the idempotence guard inside stop().
        # `_started` is set the moment start() has committed (after
        # is_alive re-check) so a racing stop() knows it must run.
        self._lifecycle_lock = threading.Lock()
        # `_stopped` is "stop() has completed at least once since the
        # last start()". Initial False so the first stop() on a
        # constructed-but-never-started engine still runs cleanup
        # (existing contract -- `test_stop_releases_in_process_sessions`
        # pre-dates this fix and relies on it).
        self._stopped: bool = False
        # lock guarding both the counter
        # `+=` / `helper_exit_reasons` TOCTOU sites AND
        # the rolling-window deque iteration sites.
        # `EngineStats` owns the actual lock (`_internal_lock`); the
        # engine aliases it so existing 040a call-sites
        # (`with self._stats_lock:`) reach the same primitive. RLock
        # so future re-entrant call patterns don't deadlock.
        self._stats_lock = self.stats._internal_lock
        # multi-field profile
        # apply consistency. The engine reads cfg fields at scattered
        # points within a chunk (`cfg.monitor` line 3843 + 4021,
        # `cfg.input_gain_db` line 3908, `cfg.f0_up_key` / `cfg.sid`
        # inside _infer). A profile-apply on the TUI thread that
        # writes 4+ fields one at a time leaves the engine reading a
        # half-applied composite mid-chunk (new monitor, old pitch).
        # The fix routes profile applies through the same chunk-
        # boundary barrier as `request_model_swap`: callers stage a
        # dict of `{field: value}` into `_pending_cfg_updates`; the
        # engine flushes the dict at the top of each chunk iteration
        # under `_cfg_lock`, AFTER `_maybe_swap_model` and BEFORE the
        # mic read. Within a chunk the engine sees a consistent
        # snapshot of all queued fields.
        self._cfg_lock = threading.Lock()
        self._pending_cfg_updates: dict[str, Any] = {}
        # bounded monitor-write queue
        # + dedicated writer thread. Pre-fix the engine called
        # `monitor_stream.write(out48)` on its main chunk-processing
        # thread; a slow host-default sink (e.g., a Bluetooth output
        # that stalled briefly) blocked the engine loop -- mic reads
        # backed up -- audio drops. Post-fix the chunk loop does a
        # non-blocking put_nowait into this 8-slot queue; the
        # `_monitor_writer_loop` thread drains it and owns the
        # `sd.OutputStream` lifecycle. On queue overflow the chunk is
        # dropped (counted in `stats.monitor_drops`) -- a brief
        # monitor glitch is fine; an engine stall is not.
        self._monitor_queue: queue.Queue[NDArrayF32] = queue.Queue(maxsize=8)
        self._monitor_thread: threading.Thread | None = None
        # pre-load swap models on a
        # dedicated background thread so the engine worker's
        # `_apply_one_swap` hits a warm `_rvc_pool` cache (~10 ms)
        # instead of paying a cache-cold `get_or_create` (~600 ms) on
        # the hot path. Pre-fix the engine worker did the slow
        # build-+-cudnn-tune itself; mic-read backed up during those
        # 600 ms; audio drops.
        #
        # `request_model_swap` puts the target on BOTH queues:
        # `_swap_queue` (drained by the engine worker at chunk
        # boundary to actually swap the session in) AND
        # `_swap_preload_queue` (drained by this preloader thread
        # which just primes the cache). When the worker reaches the
        # chunk-boundary swap, the cache is usually warm.
        self._swap_preload_queue: queue.Queue[Path] = queue.Queue()
        self._swap_preload_thread: threading.Thread | None = None
        # v0.7.0-rc7 - track whether GC was enabled before this engine
        # disabled it, so stop() restores the prior state instead of
        # blindly enabling. Lets us nest cleanly inside a parent that
        # had already disabled GC (rare but possible in tests).
        self._gc_was_enabled_before_start: bool = False

        # v0.8.0 - handle to the inference subprocess. Created lazily
        # in `start()` when `cfg.inference_subprocess=True`. Stays None
        # in legacy in-process mode.
        self._inf_client: Any = None

        # Lazy-load sessions; avoid CUDA work if the engine is constructed
        # but never started (e.g., TUI dry-run).
        self._cv: ort.InferenceSession | None = None
        self._rmvpe: ort.InferenceSession | None = None
        self._rvc: ort.InferenceSession | None = None
        self._is_half: bool = False
        self._cv_input_dtype: str = "tensor(float)"
        self._rmvpe_input_dtype: str = "tensor(float)"
        # v0.5.0: each RVC voice ONNX has its own native output rate
        # (16k for amitaro, 40k for most v2 voices, 32k/48k for some).
        # Probed at session load by running a known-length forward pass.
        # Default 16k matches the amitaro-only assumption v0.4.x baked in.
        self._rvc_output_sr: int = 16_000

        # Active embedder mode. Only "onnx" is supported (v0.8.0 removed the
        # fairseq path); kept as an attribute because diag/CLI displays it.
        self.active_embedder: str = "onnx"

        # SOLA streaming state (Phase B). v0.5.0 fix: SOLA operates at the
        # OUTPUT rate (model_sr - varies per voice: 16k for amitaro, 40k for
        # most v2 voices, 32k/48k for some). Input history stays at 16 kHz
        # because contentvec/rmvpe always take 16 kHz audio. Two SOLAConfigs:
        # `_sola_input_cfg` sizes the input history (16 kHz);
        # `_sola_output_cfg` runs the actual crossfade (model_sr, rebuilt on swap).
        from audio.sola import SOLAConfig, SOLAStream

        self._sola_input_cfg = SOLAConfig(
            rate=16_000,
            crossfade_ms=self.cfg.sola_crossfade_ms,
            search_ms=self.cfg.sola_search_ms,
            context_ms=self.cfg.sola_context_ms,
            corr_threshold=self.cfg.sola_corr_threshold,
        )
        self._sola: SOLAStream | None = (
            SOLAStream(self._sola_input_cfg) if self.cfg.sola_enabled else None
        )
        # Past-input buffer at 16 kHz (zero-padded on first call). Sized
        # against the input-side SOLAConfig so the math doesn't change when
        # we swap to a higher-rate output model.
        self._input_history: NDArrayF32 = np.zeros(
            self._sola_input_cfg.context_samples + self._sola_input_cfg.crossfade_samples,
            dtype=np.float32,
        )
        # cross-chunk pitch carry.
        # `_interpolate_voiced_gaps_np` bridges short unvoiced runs (≤8
        # frames ≈80 ms at RMVPE 100 fps) between two voiced anchors
        # within a single pitchf vector. A voiced-run that is followed
        # by a short unvoiced run that straddles the chunk boundary
        # leaves the NEXT chunk's leading unvoiced run with no in-window
        # `last_valid` anchor -- the dropout the bridge exists to
        # prevent surfaces in the listener path.
        #
        # We carry the last-voiced f0 + its age in frames across calls
        # to `_infer` so the next chunk's interpolate-pass can use it
        # as a synthetic `last_valid` for a chunk-leading unvoiced run.
        # State updates only on the legacy in-process inference path;
        # the subprocess path (`inference_subprocess=True`) does not
        # currently carry pitch state across the IPC boundary -- the
        # leading-edge dropout is preserved there (documented in
        # `_infer`). Default to (0.0, -1) meaning "no prior voiced
        # frame known."
        self._pitch_carry_f0: float = 0.0
        self._pitch_carry_age_frames: int = -1

        # v0.4.1 hot-swap: queued model-swap requests with PER-CALL completion
        # events. Pre-fix this was a single `_pending_model_swap: Path
        # | None` slot + a shared `_swap_done: threading.Event`. Two
        # rapid swap requests collapsed (the second overwrote the
        # first in the single slot, so voice-A was silently dropped);
        # the broadcast `_swap_done.set()` released ALL waiters when
        # only ONE swap had actually applied -- so Job A reported
        # "done" even though voice-A never loaded. Defeated B5/
        # corr-003. the project rules silent-failure.
        #
        # F-13-12: `_swap_done.set()` had three setter sites, all
        # inside `_maybe_swap_model` (engine-thread). `stop()` never
        # set it, so if the engine stopped with a swap in flight the
        # JobRegistry daemon thread parked for the full 10 s timeout.
        # Reachable via the ordinary "queue a swap, toggle off"
        # sequence.
        #
        # Post-fix `request_model_swap` enqueues a `_SwapRequest`
        # (target + per-call event) and returns the event. The worker
        # drains the queue at each chunk boundary, applies each swap
        # in order, and sets the event of the swap it just completed.
        # `stop()` resolves every outstanding event in teardown
        # (F-13-12) so callers never park.
        self._swap_queue: queue.Queue[_SwapRequest] = queue.Queue()
        self._swap_lock = threading.Lock()
        # Per-call completion events that the engine still owes a
        # `.set()` to. Used by `stop()` to resolve every outstanding
        # waiter in teardown.
        self._outstanding_swaps: list[_SwapRequest] = []
        # Promoted so _maybe_swap can flush the SOLA tail through the same
        # pacat process the worker already owns. v0.5.2: protected by
        # `_pacat_lock` so the watchdog can swap the handle atomically.
        self._pacat_proc: subprocess.Popen[bytes] | None = None
        self._pacat_lock = threading.Lock()
        # Set by `_open_pacat` to either "pw-cat" or "pacat" - surfaced in
        # `woys diag` so the user can see which backend is live.
        self._player_backend: str = ""

        # v0.5.2 - pacat writer / watchdog / stderr-reader threads.
        # Lifetimes are bound to a single `_run_loop()` invocation: spawned
        # in `_run_loop`'s try, joined in its finally.
        self._writer_queue: queue.Queue[bytes] | None = None
        self._writer_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._watchdog_thread: threading.Thread | None = None
        # v0.10.0-rc3 - GPU keep-alive thread; only started when
        # `cfg.gpu_keepalive_enabled=True`.
        self._keepalive_thread: threading.Thread | None = None
        # The keepalive dummy input - pre-warmed at engine start so cuDNN
        # has a cached algorithm for this shape. Allocated once, reused on
        # every keepalive iteration.
        self._keepalive_input: NDArrayF32 | None = None
        # v0.11.0 - torch separate-stream keepalive thread (replaces the
        # rc3 ORT-stream keepalive when `gpu_keepalive_torch_stream=True`
        # OR `gpu_anti_jitter_mode in {"keepalive","both"}`).
        self._torch_keepalive_thread: threading.Thread | None = None
        # v0.11.0 - best-effort SIGTERM/SIGINT handler so a `kill <pid>`
        # or Ctrl-C reverts an active GPU clock lock instead of leaving
        # the system in a locked state. Installed at engine start when
        # the lock is active; restored to the prior handler at engine
        # stop. SIGKILL (`kill -9`) cannot be caught - that case relies
        # on the user manually running `nvidia-smi -rgc`, documented in
        # docs/22-gpu-clock-lock.md.
        self._prior_signal_handlers: dict[int, Any] = {}
        # which signal (if any) triggered
        # shutdown. Set by the async-signal-safe handler; also a re-entrancy
        # guard so a repeated SIGTERM/SIGINT doesn't redo the handler work.
        self._signal_received: int | None = None
        # Watchdog signal: writer flips this on BrokenPipe so the watchdog
        # respawns immediately instead of waiting for its next poll tick.
        self._pacat_dead_event = threading.Event()
        # Last write timestamp (perf_counter). Writer thread updates it;
        # used to compute `writer_jitter_ms`.
        self._last_writer_ts: float | None = None
        # B14 / corr-015: circuit-breaker counter. Reset on every successful
        # chunk; if it climbs to 50 consecutive failures, `_stop_event` is
        # set so the engine exits cleanly rather than serving silence.
        self._consecutive_drops: int = 0

        # v0.5.0 session pool - shared cache so swap = pointer swap.
        self._rvc_pool = RvcSessionPool(
            max_size=self.cfg.session_pool_size,
            use_tensorrt=self.cfg.use_tensorrt,
            allow_cpu_fallback=self.cfg.allow_cpu_fallback,
        )
        # Probed `model_sr` per voice path so we don't redo the probe each
        # swap. Keys are resolved Paths.
        self._rvc_sr_cache: dict[Path, int] = {}
        # v0.6.7 - stateful per-(src,dst) resamplers. Created in `_run_loop`
        # before the first chunk, replaced when the model output rate
        # changes during hot-swap. See `docs/11-microcuts-bug.md`.
        # this init used to sit as dead code after
        # a `return` inside the `inference_subprocess_pid` property, so the
        # attributes did not exist until `_run_loop` ran -- any earlier
        # access (`_maybe_swap_model`, `reload_rvc`) raised AttributeError.
        self._resampler_in: _StreamResampler | None = None
        self._resampler_out: _StreamResampler | None = None

    # ---- B23 / quality-019: public read-accessors ---------------------------
    # cli.py used to reach into `engine._player_backend`, `engine._inf_client`
    # to render diag info; now goes through these stable surfaces.

    @property
    def player_backend(self) -> str:
        """The active playback backend ('pacat' / 'pw-cat'), or '' before start."""
        return self._player_backend

    @property
    def has_inference_subprocess(self) -> bool:
        """True iff the inference subprocess is currently spawned + alive."""
        return self._inf_client is not None and self._inf_client.is_alive

    @property
    def inference_subprocess_pid(self) -> int | None:
        """Child process PID, or None if running in-process."""
        if self._inf_client is None or self._inf_client._handles is None:
            return None
        pid = self._inf_client._handles.proc.pid
        return int(pid) if pid is not None else None

    # ---- model loading ------------------------------------------------------

    def _ensure_sessions(self) -> None:
        # v0.3.0: prefer fp16 variants if present next to the fp32 file. fp16
        # rmvpe halves its VRAM footprint with no measurable pitch-detection
        # quality loss (validated v0.2.0). fp16 contentvec, by contrast, has
        # cosine sim 0.75 vs fp32 - only auto-promoted if explicitly requested.
        cv_path = self._auto_pick_fp16(self.cfg.contentvec_model, allow=False)
        rmvpe_path = self._auto_pick_fp16(self.cfg.rmvpe_model, allow=True)
        rvc_path = Path(self.cfg.rvc_model)

        # fail with a
        # clear, typed error naming the missing file AND a remediation
        # command, *before* handing the path to ONNX Runtime (which
        # otherwise raises an opaque ORT-internal exception far from
        # the cause). This also makes the top-level traceback guard
        # (F-merged-022) a clean typed catch rather than an error-
        # string heuristic. Only paths whose session still needs
        # loading are checked, so an already-loaded session is
        # unaffected.
        #
        # The remediation tag distinguishes foundation models (fetched
        # by `scripts/download_weights.py` -- usually from a
        # `./install.sh --skip-models` follow-up) from the user's RVC
        # voice model (fetched by `woys models download <repo>` or
        # converted from a .pth via `woys convert`).
        for label, path, needed, remediation in (
            (
                "contentvec model",
                cv_path,
                self._cv is None,
                "scripts/download_weights.py",
            ),
            (
                "rmvpe model",
                rmvpe_path,
                self._rmvpe is None,
                "scripts/download_weights.py",
            ),
            (
                "rvc model",
                rvc_path,
                self._rvc is None,
                "woys models download <hf-repo> -- or -- woys convert <pth>",
            ),
        ):
            if needed and not path.exists():
                raise FileNotFoundError(
                    f"{label} not found at {path}.\n  Fix: run `{remediation}` and try again."
                )

        if self._cv is None:
            self._cv = _make_session(
                cv_path,
                use_tensorrt=self.cfg.use_tensorrt,
                allow_cpu_fallback=self.cfg.allow_cpu_fallback,
            )
            self._cv_input_dtype = self._cv.get_inputs()[0].type
        if self._rmvpe is None:
            self._rmvpe = _make_session(
                rmvpe_path,
                use_tensorrt=self.cfg.use_tensorrt,
                allow_cpu_fallback=self.cfg.allow_cpu_fallback,
            )
            self._rmvpe_input_dtype = self._rmvpe.get_inputs()[0].type
        if self._rvc is None:
            self._rvc = self._rvc_pool.get_or_create(self.cfg.rvc_model)
            self._is_half = self._rvc.get_inputs()[0].type != "tensor(float)"
            self._rvc_output_sr = self._cached_rvc_sr(Path(self.cfg.rvc_model))

        # v0.8.1: snapshot TRT init status from the module-level
        # tracker into stats, so woys diag can show which sessions
        # actually got TRT EP and which fell back to CUDA.
        self.stats.trt_active_for = dict(_TRT_ACTIVE_PER_SESSION)
        self.stats.trt_init_errors = dict(_TRT_INIT_ERRORS)
        # record whether any model session
        # bound CPU-only. Only reachable through the test-only
        # cfg.allow_cpu_fallback -- otherwise `_make_session` raises
        # CpuFallbackError above. Printed by `woys diag`.
        self.stats.cpu_fallback_active = any(
            _session_is_cpu_only(s) for s in (self._cv, self._rmvpe, self._rvc) if s is not None
        )

        # Resolve embedder mode. v0.8.0 removed the fairseq path - only "onnx"
        # is supported. Any non-"onnx" value in config is reported and the
        # engine falls back to onnx (so old config.toml files don't crash).
        if self.cfg.embedder != "onnx":
            msg = (
                f"unknown embedder {self.cfg.embedder!r}; v0.8.0 only supports "
                f'"onnx". Falling back.'
            )
            print(f"[engine] {msg}")
            self.record_error(msg)
        self.active_embedder = "onnx"

    @staticmethod
    def _auto_pick_fp16(fp32_path: Path, *, allow: bool) -> Path:
        """If a `<name>-fp16.onnx` sibling exists and `allow=True`, use it."""
        if not allow:
            return fp32_path
        cand = fp32_path.with_name(fp32_path.stem + "-fp16" + fp32_path.suffix)
        return cand if cand.exists() else fp32_path

    def _probe_rvc_output_sr(self) -> int:
        """Run one forward pass through the loaded RVC session to measure its
        native output sample rate.

        The output of an RVC v2 ONNX is `(N_out,)` audio at the model's
        training rate (16 kHz for amitaro v2_16k, 40 kHz for most v2 voices,
        32 kHz / 48 kHz for some). The convert.py exporter stamps the rate
        into ONNX `custom_metadata_map["metadata"]` as JSON, but reading
        that is brittle - different exporters use different keys. Probing
        is bulletproof: feed a known-length input, count output samples.

        Costs ~20 ms once at session load. Worth it.

        raises `RuntimeError` if the probe
        fails or yields an unrecognised rate -- it never silently guesses,
        because a wrong output rate pitch-shifts the entire session and
        poisons `_rvc_sr_cache`.
        """
        assert self._rvc is not None
        # Feed a 1 s feats window (50 frames after 2x upsample = 100 frames),
        # measure output. Feats dim from the RVC input shape.
        feats_dim = 768
        try:
            shape = self._rvc.get_inputs()[0].shape
            if len(shape) >= 3 and isinstance(shape[2], int):
                feats_dim = shape[2]
        except (IndexError, ValueError):
            pass
        # 1 s of audio at 16 kHz contentvec = 50 frames. Upsample 2x = 100.
        n_frames = 100
        feats_dummy = np.zeros((1, n_frames, feats_dim), dtype=np.float32)
        feats_dtype = np.float16 if self._is_half else np.float32
        feed: dict[str, np.ndarray] = {  # type: ignore[type-arg]
            "feats": feats_dummy.astype(feats_dtype),
            "p_len": np.array([n_frames], dtype=np.int64),
            "pitch": np.zeros((1, n_frames), dtype=np.int64),
            "pitchf": np.zeros((1, n_frames), dtype=np.float32),
            "sid": np.array([0], dtype=np.int64),
        }
        try:
            out = self._rvc.run(["audio"], feed)[0]
        except Exception as e:
            # re-raise -- do NOT silently
            # return 16 kHz. CX2 corrected the original "nono variant"
            # framing: `_infer` builds an *identical* pitch-bearing feed
            # dict, so a genuinely pitchless model would crash there on
            # every chunk -- it cannot produce the silent-chipmunk symptom.
            # The realistic trigger is a transient GPU/cuDNN/shape error on
            # the cold first pass, where swallowing and assuming 16 kHz
            # poisons `_rvc_sr_cache` forever and plays the whole session
            # ~16 semitones off with no `last_error` and no counter.
            raise RuntimeError(
                f"failed to probe the RVC model's output sample rate: "
                f"{type(e).__name__}: {e}. The engine cannot run without a "
                f"known output rate (a wrong guess pitch-shifts the whole "
                f"session), so this aborts start visibly. Retry, or check "
                f"`woys info` / the GPU state."
            ) from e
        n_out = int(np.asarray(out).size)
        # Output for 1 s of feats input ≈ 1 s of audio at the model rate.
        # Round to the nearest known RVC training rate.
        for sr in (16_000, 22_050, 24_000, 32_000, 40_000, 44_100, 48_000):
            if abs(n_out - sr) < sr * 0.05:
                return sr
        # an unrecognised rate is a second silent
        # guess -- raise instead of treating the raw sample count as Hz.
        raise RuntimeError(
            f"RVC model probe produced {n_out} samples for a 1 s input, which "
            f"matches no known RVC training rate (16/22.05/24/32/40/44.1/48 "
            f"kHz). Refusing to guess the output rate."
        )

    def _cached_rvc_sr(self, path: Path) -> int:
        """Probe and remember the model's output sample rate.

        Side-effect: recreates `self._sola` at the new rate so the
        crossfade-window math matches the actual output samples.
        """
        key = Path(path).resolve()
        if key in self._rvc_sr_cache:
            sr = self._rvc_sr_cache[key]
        else:
            sr = self._probe_rvc_output_sr()
            self._rvc_sr_cache[key] = sr
        self._rebuild_sola_for_rate(sr)
        return sr

    def _rebuild_sola_for_rate(self, model_sr: int) -> None:
        """Recreate the output-side SOLAStream for the given rate. Idempotent -
        no-op when the rate is unchanged."""
        from audio.sola import SOLAConfig, SOLAStream

        if not self.cfg.sola_enabled:
            self._sola = None
            return
        if self._sola is not None and self._sola.cfg.rate == model_sr:
            return
        out_cfg = SOLAConfig(
            rate=model_sr,
            crossfade_ms=self.cfg.sola_crossfade_ms,
            search_ms=self.cfg.sola_search_ms,
            context_ms=self.cfg.sola_context_ms,
            corr_threshold=self.cfg.sola_corr_threshold,
        )
        self._sola = SOLAStream(out_cfg)

    def reload_rvc(self, path: Path) -> None:
        """Hot-swap the RVC voice model - synchronous, thread-unsafe.

        Use `request_model_swap()` from any thread other than the engine
        worker; this function is kept for tests + offline use only.
        """
        self.cfg.rvc_model = path
        self._rvc = self._rvc_pool.get_or_create(path)
        self._is_half = self._rvc.get_inputs()[0].type != "tensor(float)"
        self._rvc_output_sr = self._cached_rvc_sr(Path(path))
        self.reset_streaming_state()

    def warmup_voice_library(self, voice_paths: list[Path] | None = None) -> int:
        """Eagerly load + cudnn-warm every cached voice. Returns count warmed.

        If `voice_paths` is None, walks the user's models dir and warms
        every `.onnx` that doesn't look like a foundation file. Costs
        ~600 ms per voice on RTX 2070; subsequent swaps to any of those
        voices are pointer swaps (~10 ms total).

        B61 / perf-007: caps voice_paths at `session_pool_size`. Pre-v0.8.0
        we walked every voice in the dir even though only the last
        `pool_size` survive eviction - so voices 1..N-pool_size were
        warmed and immediately discarded. Wasted startup time.
        """
        if voice_paths is None:
            foundations = {
                "rmvpe.onnx", "rmvpe-fp16.onnx",
                "rmvpe_wrapped.onnx", "rmvpe_wrapped-fp16.onnx",
                "contentvec-f.onnx", "contentvec-f-fp16.onnx",
                "hubert_base.onnx",
            }  # fmt: skip
            voice_paths = sorted(p for p in MODELS_DIR.glob("*.onnx") if p.name not in foundations)
        # B61: only warm what fits in the pool. The first N entries (alphabetical)
        # are the ones the user is most likely to land on - bias toward retaining
        # the deterministic prefix.
        cap = self.cfg.session_pool_size
        if len(voice_paths) > cap:
            print(
                f"[engine] eager-warmup capped at session_pool_size={cap} "
                f"(have {len(voice_paths)} voices; skipping the LRU-evicted tail)"
            )
            voice_paths = voice_paths[:cap]
        for p in voice_paths:
            self._rvc_pool.warmup(p)
            # Also probe + cache the SR for each.
            with contextlib.suppress(Exception):
                self._rvc_sr_cache[p.resolve()] = self._probe_sr_for(p)
        return len(voice_paths)

    def _probe_sr_for(self, path: Path) -> int:
        """Probe the model's output SR via the pool (so the session is cached)."""
        sess = self._rvc_pool.get_or_create(path)
        prev_rvc = self._rvc
        prev_is_half = self._is_half
        self._rvc = sess
        self._is_half = sess.get_inputs()[0].type != "tensor(float)"
        try:
            return self._probe_rvc_output_sr()
        finally:
            self._rvc = prev_rvc
            self._is_half = prev_is_half

    def request_model_swap(self, path: Path) -> _SwapRequest:
        """Thread-safe: queue a model swap and return the per-call
        `_SwapRequest`. The caller waits on `req.completion` and reads
        `req.error` to distinguish "done" from "failed".

        pre-fix this
        method overwrote a single `_pending_model_swap` slot and
        cleared a SHARED `_swap_done` Event -- two rapid swaps
        collapsed (the second overwrote the first; voice A was
        silently dropped) and the broadcast `_swap_done.set()`
        released all waiters when only one swap had applied. Defeated
        B5/corr-003.

        Post-fix every request lands as its own `_SwapRequest` in
        `self._swap_queue`. F-13-12: if the engine is stopped (or
        stopping) when this is called, the event is resolved
        immediately so the caller never parks.

        the return type widened
        from `threading.Event` to `_SwapRequest` so swap *failures*
        also have a single read site. `_maybe_swap_model` sets
        `req.error` before resolving the event; callers (the TUI's
        MODEL / PROFILE handlers, `_apply_profile_named`,
        `action_cycle_profile`) check `req.error` after `req.
        completion.wait(...)` and route a failure to
        `engine.record_error()`. Pre-fix the failure landed only in
        the JobRegistry status_line via the worker's bare-exception
        path, and the TUI never polled its own jobs -- so a swap to
        a corrupted ONNX or a `subprocess swap failed` cascade
        recorded "OK job=... model=...", parked the UI on "loading
        X..." for 10 s, then resumed with the old voice still in
        place. No banner, no toast.
        """
        req = _SwapRequest(target=Path(path))
        with self._swap_lock:
            # `_stop_event` covers an engine that stopped itself (or is
            # mid-stop): its loop is gone and nothing drains the queue.
            if self._stopped or self._stop_event.is_set():
                # F-13-12: a STOPPED engine cannot drain the queue --
                # resolve the event immediately so callers don't park
                # for the full 10 s JobRegistry timeout. A never-
                # started engine still queues (used by tests +
                # script-only flows where `_maybe_swap_model` is
                # called manually).
                req.error = RuntimeError("engine stopped; swap not applied")
                req.completion.set()
                return req
            self._swap_queue.put(req)
            self._outstanding_swaps.append(req)
        # also signal the preloader
        # to prime the pool cache so the worker hits a cache-hit on
        # the chunk-boundary swap. Best-effort -- queue.put never
        # blocks because the queue is unbounded; if the preloader
        # thread isn't running (engine not yet started or already
        # stopped), the worker just pays the cache-miss cost itself.
        with contextlib.suppress(Exception):
            self._swap_preload_queue.put_nowait(Path(path))
        return req

    def request_cfg_update(self, updates: dict[str, Any]) -> None:
        """Thread-safe: queue a multi-field cfg update for the worker to
        apply at the next chunk boundary.

        pre-fix
        `_apply_profile_named` wrote `engine.cfg.f0_up_key`,
        `engine.cfg.sid`, `engine.cfg.monitor`, and `engine.cfg.input_
        gain_db` one at a time. The engine thread reads those same
        fields at scattered points within a single chunk, so an
        apply interleaved with a chunk left the engine reading a
        half-applied composite (e.g., new monitor flag, old pitch).
        The bug class is multi-field *consistency*; lock-around-write
        alone doesn't fix it because the engine's READS were not
        locked.

        This function stages a dict of `{field: value}` updates; the
        engine drains the dict via `_maybe_apply_pending_cfg()` at
        the top of each chunk iteration. Within a chunk the engine
        sees a consistent snapshot.

        Idempotent: repeat calls merge into the dict (later wins on
        a field collision). Callers who write single fields directly
        (TUI pitch keys, monitor toggle) are unaffected -- single-
        field atomicity is already a Python guarantee.
        """
        with self._cfg_lock:
            self._pending_cfg_updates.update(updates)

    def _maybe_apply_pending_cfg(self) -> None:
        """Worker-side hook: drain `_pending_cfg_updates` and apply
        every queued field to `self.cfg` atomically (relative to the
        chunk loop). Called from `_run_loop` at the top of each chunk
        iteration, right after `_maybe_swap_model()` and before the
        mic read."""
        with self._cfg_lock:
            if not self._pending_cfg_updates:
                return
            updates = self._pending_cfg_updates
            self._pending_cfg_updates = {}
        for field_name, value in updates.items():
            if hasattr(self.cfg, field_name):
                setattr(self.cfg, field_name, value)

    def _maybe_swap_model(self) -> None:
        """Worker-side hook: drain any queued swap requests, applying
        each in order. Called from `_run_loop` at the top of each
        chunk.

        pre-fix the single-slot pattern collapsed
        rapid swaps. Post-fix every queued `_SwapRequest` is applied
        in order; each completion event fires when ITS swap finishes
        (not the broadcast Event of the pre-fix code)."""
        requests: list[_SwapRequest] = []
        with self._swap_lock:
            while True:
                try:
                    requests.append(self._swap_queue.get_nowait())
                except queue.Empty:
                    break
        if not requests:
            return
        for i, req in enumerate(requests):
            try:
                self._apply_one_swap(req)
            except Exception as e:
                # `_apply_one_swap` handles the expected failures itself;
                # anything else still crashes the loop, but no caller may
                # be left parked on a request this batch already drained.
                req.error = e
                self._resolve_swap(req)
                for rest in requests[i + 1 :]:
                    rest.error = RuntimeError("swap not applied: an earlier swap failed")
                    self._resolve_swap(rest)
                raise

    def _apply_one_swap(self, req: _SwapRequest) -> None:
        """Apply a single `_SwapRequest`: flush SOLA tail, drain writer,
        swap the session, reset streaming state, set the per-call
        completion event. Failures are recorded on `req.error`; the
        event is set regardless so the caller never parks.

        Extracted from the legacy `_maybe_swap_model` body so each
        queued swap gets the same treatment (SOLA flush + writer
        drain + session reset). For multi-swap drains the SOLA flush
        happens per-swap -- the user requested each one and each
        deserves a clean transition."""
        target = req.target
        resampler_out_was_flushed = False
        if self._sola is not None:
            with contextlib.suppress(Exception):
                tail = self._sola.flush()
                if tail.size > 0 and self._resampler_out is not None:
                    tail48 = self._resampler_out.process(tail)
                    flush48 = self._resampler_out.flush()
                    resampler_out_was_flushed = True
                    full = np.concatenate([tail48, flush48]) if flush48.size else tail48
                    if full.size > 0:
                        self._enqueue_chunk(self._to_sink_bytes(full))

        # B10 / corr-005: wait for the writer queue to drain before swapping
        # the model. Without this barrier, OLD-rate audio sitting in the
        # queue (up to ~2 s worth at queue_size=8 + chunk_seconds=0.25)
        # plays AFTER the swap completes - user hears the old voice for
        # seconds after triggering a swap. 300 ms timeout caps the wait
        # so a stuck pacat/pw-cat doesn't deadlock the swap.
        if self._writer_queue is not None:
            drain_deadline = time.perf_counter() + 0.3
            while time.perf_counter() < drain_deadline and not self._writer_queue.empty():
                time.sleep(0.005)

        # v0.8.0 - subprocess swap path. Child owns the session pool;
        # tell it to swap, wait for the new RVC sample-rate response,
        # then rebuild the output resampler in the parent if the rate
        # changed. On InferenceError (child died, swap timed out) we
        # used to silently `return` and let the engine continue running
        # in a state where every chunk drops (B6 / corr-004). Now: stop
        # the engine cleanly and surface the error to the TUI.
        if self._inf_client is not None:
            from audio.inference_client import InferenceError

            try:
                new_sr, new_is_half = self._inf_client.swap_model(target)
                self.cfg.rvc_model = target
                self._is_half = new_is_half
                # v0.14.0 (C002): rebuild the resampler if rate changed OR
                # if the SOLA flush above finalized the existing soxr stream.
                # Same-rate swap with a non-identity stream pre-v0.14.0
                # crashed the engine on the next chunk.
                # ~5 ms cold-fade-in
                # masks the soxr filter-warmup transient when the rebuild
                # lands inside live audio (model swap). 5 ms at sink_rate
                # = sink_rate // 200; below the ~10 ms perception window
                # for amplitude steps.
                if new_sr != self._rvc_output_sr or resampler_out_was_flushed:
                    self._resampler_out = _StreamResampler(
                        new_sr,
                        self.cfg.sink_rate,
                        cold_fade_in_samples=self.cfg.sink_rate // 200,
                    )
                if new_sr != self._rvc_output_sr:
                    self._rebuild_sola_for_rate(new_sr)
                self._rvc_output_sr = new_sr
                self.reset_streaming_state()
                self._resolve_swap(req)
                return
            except InferenceError as e:
                # B6 / corr-004: do NOT silently fall through. The in-process
                # _rvc isn't loaded in subprocess mode, so the legacy path
                # would fail every chunk. Better to stop and surface the
                # error than serve silence indefinitely.
                req.error = e
                self._resolve_swap(req)
                self._self_abort(
                    f"subprocess swap failed: {e}. Stopping engine - "
                    f"flip `inference_subprocess=false` to fall back to "
                    f"in-process inference."
                )
                return

        # Legacy in-process path. Existing _cv (contentvec) and _rmvpe
        # stay - they're foundation models, not voice-specific.
        # v0.5.0: pool-cached. Cache hit ≈ 10 ms; cache miss ≈ 600 ms.
        # A model that fails to load or probe (corrupt / incompatible
        # export, file gone) must not crash the engine: keep the old
        # voice, report the failure on the request, and only point
        # cfg.rvc_model at the target once it is really running.
        prev_rvc, prev_is_half = self._rvc, self._is_half
        try:
            self._rvc = self._rvc_pool.get_or_create(target)
            self._is_half = self._rvc.get_inputs()[0].type != "tensor(float)"
            # Probes through self._rvc / self._is_half, hence set above.
            new_sr = self._cached_rvc_sr(target)
        except Exception as e:
            self._rvc, self._is_half = prev_rvc, prev_is_half
            self._rebuild_sola_for_rate(self._rvc_output_sr)
            # The SOLA flush above finalized the output stream; the old
            # voice needs a fresh one, exactly as a successful swap would.
            if resampler_out_was_flushed:
                self._resampler_out = _StreamResampler(
                    self._rvc_output_sr,
                    self.cfg.sink_rate,
                    cold_fade_in_samples=self.cfg.sink_rate // 200,
                )
            self.reset_streaming_state()
            req.error = e
            self.record_error(
                f"model swap to {Path(target).name} failed: {type(e).__name__}: {e}. "
                f"Keeping the current voice."
            )
            self._resolve_swap(req)
            return
        self.cfg.rvc_model = target
        # v0.6.7 - rebuild the output resampler if the new model has a
        # different native rate. Identity ratios (e.g. 16k -> 16k -> 48k
        # stays the same) won't reset state, so swaps between same-rate
        # voices used to skip the rebuild.
        # v0.14.0 (C002): also rebuild when the SOLA flush above finalized
        # the existing soxr stream (`last=True`). Without this, a same-rate
        # swap left a finalized stream in place and the next chunk raised
        # `RuntimeError: Input after last input` from soxr, killing engine.
        # cold-fade-in (see subprocess
        # branch above for rationale). 5 ms at sink_rate.
        if new_sr != self._rvc_output_sr or resampler_out_was_flushed:
            self._resampler_out = _StreamResampler(
                new_sr,
                self.cfg.sink_rate,
                cold_fade_in_samples=self.cfg.sink_rate // 200,
            )
        self._rvc_output_sr = new_sr
        self.reset_streaming_state()
        # B5: signal "swap complete" AFTER all the work. TUI poll sites
        # waiting on the per-call event now correctly observe done-state.
        self._resolve_swap(req)

    def _resolve_swap(self, req: _SwapRequest) -> None:
        """Set the per-call completion event and drop the request
        from `_outstanding_swaps`. Safe to call from the engine
        thread (via `_apply_one_swap`) or from `stop()` teardown
        (where `req.error` is set first)."""
        req.completion.set()
        with self._swap_lock, contextlib.suppress(ValueError):
            # already removed (double-resolve) -> ValueError is fine.
            self._outstanding_swaps.remove(req)

    def _fail_pending_swaps(self, msg: str) -> None:
        """Resolve every outstanding or still-queued swap with an error
        so no caller parks on a request that will never be applied."""
        with self._swap_lock:
            pending = list(self._outstanding_swaps)
            self._outstanding_swaps.clear()
            while True:
                try:
                    pending.append(self._swap_queue.get_nowait())
                except queue.Empty:
                    break
        for req in pending:
            if not req.completion.is_set():
                req.error = RuntimeError(msg)
                req.completion.set()

    def _self_abort(self, msg: str) -> None:
        """The engine stopping itself (circuit breaker, respawn cap,
        failed subprocess swap). Marked as a crash, not a clean stop, so
        the TUI stops showing RUNNING and `woys engine` exits non-zero
        instead of printing frozen stats; the loop exit clears
        `stats.running`."""
        self.record_error(msg)
        self.stats.crashed = True
        self._stop_event.set()

    # ---- inference ----------------------------------------------------------

    # ---- realtime loop ------------------------------------------------------

    def start(self) -> None:
        """start() returns
        promptly after spawning the worker thread; the worker does
        the slow cold-start preamble (sessions, warmup, GPU clock
        lock, inference subprocess) and updates `stats.warmup_stage`
        as it goes. Pre-fix start() ran all that work on the
        caller's thread and blocked for up to ~10 s, freezing the
        TUI with a stale "~2s" toast and no progress signal.

        Failure modes that pre-v0.14.x review caught
        synchronously (PipeWireError, FileNotFoundError,
        CpuFallbackError) now land ASYNC via `stats.crashed=True`
        + `record_error(...)` (the bounded ring from F-merged-015).
        The TUI's `_refresh_stats` polls and surfaces the error as
        a notify toast -- the same machinery that handles steady-
        state errors. F-merged-022's outer traceback guard at
        `_start_engine` keeps catching anything start() itself
        raises synchronously (signal-handler setup, etc.).

        F-merged-018: `_lifecycle_lock` still serializes start()
        and stop(). The lock is now held only for the FAST setup
        (gc disable, signal handlers, worker spawn); the slow
        preamble in `_worker_main` re-acquires the lock so a stop()
        that fires mid-preamble waits for the preamble to either
        finish or itself acquire the lock and observe `_stop_event`.
        """
        with self._lifecycle_lock:
            if self._thread and self._thread.is_alive():
                return
            if self._thread is not None and not self._stopped:
                # The previous run ended without stop() (failed warmup,
                # crash, self-stop). Its GC state, signal handlers, clock
                # lock and inference child are still in place; tear them
                # down first or this start() saves the engine's own state
                # as the "prior" to restore and leaks the old child.
                self._teardown_locked(timeout=2.0)
            self._stop_event.clear()
            self._reset_session_stats()
            self.stats.crashed = False
            self._stopped = False
            # Reset the signal-handler re-entrancy guard so a restarted engine
            # (e.g. the TUI stop->start toggle) can still handle SIGINT/SIGTERM;
            # otherwise a signal seen during a prior run would suppress clean
            # teardown (incl. reverting an active GPU clock lock) on the next.
            self._signal_received = None
            self.stats.warmup_stage = "starting"

            # FAST caller-thread setup (sub-millisecond): GC + signal
            # handlers. Anything slower goes to the worker.
            self._gc_was_enabled_before_start = gc.isenabled()
            if self._gc_was_enabled_before_start:
                gc.disable()
            self._install_signal_handlers()

            # Announce intent + spawn worker. `stats.running = True`
            # before the worker enters its chunk loop because the
            # warmup IS the running engine doing work -- the TUI
            # should show "RUNNING" with the warmup_stage substage
            # rather than "STOPPED" while sessions load.
            self.stats.running = True
            self._thread = threading.Thread(
                target=self._worker_main, name="woys-engine", daemon=True
            )
            self._thread.start()

    def _reset_session_stats(self) -> None:
        """Zero the per-session counters for a new run. In place: the TUI,
        the CLI and `_stats_lock` all hold this object. Without it the
        second run inherited chunks_processed >= 10 (the TUI's warmup
        indicator never showed again) and mixed late/max/avg/drop counts
        across sessions."""
        fresh = EngineStats()
        with self._stats_lock:
            for f in fields(EngineStats):
                if f.name not in _STATS_KEPT_ACROSS_RUNS:
                    setattr(self.stats, f.name, getattr(fresh, f.name))

    def _worker_main(self) -> None:
        """Worker-thread entry: cold-start preamble + chunk loop.

        Preamble failures land on `stats.crashed` + `record_error`
        (async surfacing path; the TUI's `_refresh_stats` notify
        toast picks them up). The preamble itself takes
        `_lifecycle_lock` so a concurrent stop() coordinates
        cleanly. F-merged-030.
        """
        try:
            with self._lifecycle_lock:
                if self._stop_event.is_set():
                    # stop() fired between start()'s spawn and the
                    # worker grabbing the lock -- bail clean.
                    self.stats.warmup_stage = ""
                    self.stats.running = False
                    return
                self._worker_preamble()
            self.stats.warmup_stage = "ready"
        except Exception as e:
            # Broad on purpose: onnxruntime's InvalidProtobuf / Fail /
            # NoSuchFile derive straight from Exception, so a corrupt
            # model would otherwise kill this thread silently and leave
            # the engine "running" at "loading sessions" forever.
            # PipeWireError and CpuFallbackError are RuntimeErrors and
            # land here too.
            self.stats.warmup_stage = f"crashed: {type(e).__name__}"
            self.stats.crashed = True
            self.record_error(f"engine warmup: {type(e).__name__}: {e}")
            self.stats.running = False
            return
        # Preamble succeeded; enter the chunk loop. _run_loop has its
        # own outer try/except that records errors and sets running=
        # False / crashed=True on uncaught exceptions.
        self._run_loop()

    def _worker_preamble(self) -> None:
        """Slow cold-start work that pre-fix ran on the caller's
        thread. Updates `stats.warmup_stage` at each step so the TUI
        can show a live substage. F-merged-030.

        Called from `_worker_main` while holding `_lifecycle_lock`."""
        self.stats.warmup_stage = "checking default sink"
        self._warn_if_default_sink_hijacked()

        self.stats.warmup_stage = "applying GPU clock lock"
        clock_lock_on, _ = self._resolve_anti_jitter_flags()
        if clock_lock_on:
            self._apply_gpu_clock_lock()

        if self.cfg.inference_subprocess:
            self.stats.warmup_stage = "spawning inference subprocess"
            from audio.inference_client import InferenceClient

            cfg_dict = self._cfg_dict_for_subprocess()
            self._inf_client = InferenceClient(cfg_dict)
            self._inf_client.start()
            self._rvc_output_sr = self._inf_client.rvc_output_sr
            self._is_half = self._inf_client.is_half
            self.active_embedder = self._inf_client.active_embedder
            self.stats.child_pid = (
                self._inf_client._handles.proc.pid if self._inf_client._handles else None
            )
            self._rebuild_sola_for_rate(self._rvc_output_sr)
        else:
            self.stats.warmup_stage = "loading sessions"
            self._ensure_sessions()
            self.stats.warmup_stage = "warming pipeline"
            self._warmup_realtime_pipeline()

        if self.cfg.eager_warmup and not self.cfg.inference_subprocess:
            self.stats.warmup_stage = "eager warming voice library"
            n = self.warmup_voice_library()
            print(f"[engine] eager-warmed {n} voice models (instant swaps now)")

        # spawn the dedicated
        # monitor-writer thread. Pre-fix the engine main thread did
        # `monitor_stream.write()` synchronously on the hot path;
        # any slow host-default sink blocked the engine.
        self._monitor_thread = threading.Thread(
            target=self._monitor_writer_loop, name="woys-monitor-writer", daemon=True
        )
        self._monitor_thread.start()

        # spawn the swap-preloader
        # thread so request_model_swap callers warm the _rvc_pool
        # cache off the hot path. The worker's chunk-boundary swap
        # then hits a cache-hit (~10ms) instead of paying the
        # cache-cold get_or_create (~600ms).
        self._swap_preload_thread = threading.Thread(
            target=self._swap_preloader_loop, name="woys-swap-preloader", daemon=True
        )
        self._swap_preload_thread.start()

    def _cfg_dict_for_subprocess(self) -> dict[str, Any]:
        """Convert EngineConfig to a dict for spawn pickling.

        v0.8.0-rc3 - DO NOT convert Path → str. Path is picklable;
        converting drops Path's interface (`with_name`, `parent`,
        etc.) so engine code in the child crashes with
        `AttributeError: 'str' object has no attribute 'with_name'`
        in `_auto_pick_fp16`. The crash forced the parent's
        `start()` to fall back to in-process inference silently,
        which is why CC's bash test passed but real Telegram audio
        sounded "broken" - production was running in-process all
        along, but stderr was hijacked by Textual so the fallback's
        `last_error` was never visible.
        """
        from dataclasses import asdict

        return asdict(self.cfg)

    def record_error(self, msg: str) -> None:
        """Record an error to the bounded timestamped ring + mirror to
        `stats.last_error` for back-compat reads.

        pre-fix `self.stats.last_error = msg`
        was the only sink. ~28 write sites across 6 threads clobbered
        the single string, so a cascading failure left only the LAST
        symptom visible in `woys diag`. The ring (`error_history`,
        deque maxlen=20) keeps the most recent 20 entries with
        timestamps + thread names, so the user can reconstruct the
        cascade.

        The append + back-compat field write both happen under
        `_stats_lock` so a concurrent record_error from another thread
        sees a consistent state. The lock is held for microseconds.
        """
        now = time.monotonic()
        entry = (now, threading.current_thread().name, msg)
        with self._stats_lock:
            self.stats.error_history.append(entry)
            self.stats.last_error = msg
            # stamp the write moment so
            # the TUI can render an age and the chunk-success path can
            # auto-clear once `chunk_seconds` has elapsed without a fresh
            # failure. The ring keeps the history; this field is the
            # *current* sticky-error timestamp.
            self.stats.last_error_ts = now
        # Mirror to the rotating file log (outside the lock - file I/O). The
        # error ring above is RAM-only and dies with the process; this leaves a
        # durable, wall-clock-stamped trace so a transient fault is
        # reconstructable from woys.log after the fact.
        logging.getLogger("woys.engine").warning(msg)

    def recent_errors(self, n: int = 5) -> list[tuple[float, str, str]]:
        """Return the most recent up-to-`n` error entries from the ring,
        oldest first. Snapshot semantics: the returned list is a copy;
        concurrent writers do not see it."""
        with self._stats_lock:
            if n >= len(self.stats.error_history):
                return list(self.stats.error_history)
            return list(self.stats.error_history)[-n:]

    def stop(self, timeout: float = 2.0) -> None:
        # lock + idempotence guard.
        # Pre-fix two concurrent stop() callers (signal-handler path +
        # action_quit + CLI teardown -- see F-CX3-01) both passed
        # `self._inf_client is not None` and double-tore-down the
        # InferenceClient. The `_stopped` flag short-circuits the
        # second caller; the lock also makes a stop() that arrives
        # during start()'s warmup wait for warmup to finish before
        # tearing down, preventing the "running engine spawned after
        # the stop signal" hazard.
        with self._lifecycle_lock:
            if self._stopped:
                return
            self._teardown_locked(timeout)

    def _teardown_locked(self, timeout: float) -> None:
        """stop()'s teardown body. Caller holds `_lifecycle_lock` (a plain
        Lock, not re-entrant), so start() can run it for a previous run
        that ended without stop() -- a failed warmup, a crash or a
        self-stop -- before building the next one."""
        self._stop_event.set()
        # resolve every outstanding swap
        # waiter BEFORE we start the slow teardown. Pre-fix
        # `_swap_done` was never set in `stop()`, so a JobRegistry
        # daemon thread parked the full 10 s timeout on the
        # "queue a swap, toggle off" sequence. With per-call
        # events, we walk the outstanding list and resolve each
        # with an "engine stopped" error (queued requests the worker
        # never got to included).
        self._fail_pending_swaps("engine stopped before swap completed")
        if self._thread:
            self._thread.join(timeout=timeout)
        self.stats.running = False
        # join the monitor-
        # writer thread. The thread sees `_stop_event` via its
        # 50ms get-timeout polling loop and exits after closing
        # its sd.OutputStream.
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=1.0)
            self._monitor_thread = None
        # join the swap-
        # preloader thread. Same pattern -- it sees _stop_event
        # via its 100ms get-timeout polling loop.
        if self._swap_preload_thread is not None:
            self._swap_preload_thread.join(timeout=1.0)
            self._swap_preload_thread = None
        # clear warmup_stage so the TUI
        # doesn't show a stale "warming pipeline" indicator after
        # the engine fully stops.
        self.stats.warmup_stage = ""

        # v0.8.0 - tear down the inference subprocess after the engine
        # thread has stopped sending it work. `InferenceClient.stop()`
        # sends CMD_STOP, joins the child, closes pipes, unlinks shm.
        if self._inf_client is not None:
            with contextlib.suppress(Exception):
                self._inf_client.stop(timeout_s=timeout)
            self._inf_client = None
            self.stats.child_pid = None

        # release the in-process ONNX sessions
        # before the gc.collect() below. Pre-fix `stop()` tore down the
        # inference subprocess but never dropped `_cv`/`_rmvpe`/`_rvc` or
        # evicted the RVC pool (`evict_all()` had no caller), so in-process
        # mode accumulated VRAM across start/stop cycles -- and any future
        # VRAM-target measurement was confounded by the leak. This affects
        # the in-process path (`inference_subprocess=False`); subprocess
        # mode frees the sessions by killing the child above.
        self._cv = None
        self._rmvpe = None
        self._rvc = None
        self._rvc_pool.evict_all()

        # v0.7.0-rc7 - restore GC to its prior state and run one
        # collection to free any cyclic references that accumulated
        # during the session. If GC was already disabled before this
        # engine started (nested case), leave it disabled.
        if self._gc_was_enabled_before_start:
            gc.enable()
            gc.collect()
            self._gc_was_enabled_before_start = False

        # v0.11.0 - release the GPU clock lock if active. Idempotent;
        # safe to call when no lock was applied. SIGTERM/SIGINT path
        # may have already reverted, in which case this is a no-op.
        with contextlib.suppress(Exception):
            self._revert_gpu_clock_lock()

        self._stopped = True

    # ---- v0.5.2 writer / watchdog / stderr-reader plumbing ------------------

    def _swap_preloader_loop(self) -> None:
        """daemon thread that
        primes the `_rvc_pool` cache for every queued swap target.

        Pre-fix the engine worker called
        `self._rvc_pool.get_or_create(target)` on the hot path. On a
        cache miss that costs ~600 ms (model load + cuDNN tune) --
        the engine isn't reading the mic, the writer queue drains,
        the user hears a glitch.

        Post-fix this thread drains `_swap_preload_queue` and calls
        `get_or_create` itself. `RvcSessionPool.get_or_create` builds
        OUTSIDE its internal lock, so this thread's call doesn't
        block a concurrent worker call. By the time the worker
        reaches `_apply_one_swap` at the chunk boundary, the cache
        is warm and its `get_or_create` is a ~10 ms cache-hit.

        Failures (FileNotFound, ORT errors) are swallowed here -- the
        engine worker will hit the same failure at the chunk boundary
        and surface it through the normal swap-completion error path
        (`_SwapRequest.error`).
        """
        while not self._stop_event.is_set():
            try:
                target = self._swap_preload_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            with contextlib.suppress(Exception):
                self._rvc_pool.get_or_create(target)

    def _monitor_writer_loop(self) -> None:
        """daemon thread that owns
        the self-monitor `sd.OutputStream` lifecycle and drains the
        bounded `_monitor_queue`. Pre-fix the engine main thread did
        both -- opening / closing the stream when `cfg.monitor`
        toggled AND writing each chunk synchronously. A slow host
        default sink (Bluetooth glitch, ALSA underrun on a busy
        system) blocked the engine and starved the mic-read loop.

        This thread:
          * Opens an `sd.OutputStream` on cfg.monitor going True.
          * Closes it on cfg.monitor going False.
          * Drains `_monitor_queue` with a short `get` timeout so the
            loop also wakes to check `_stop_event` + `cfg.monitor`.
          * Catches per-write exceptions via `record_error` and
            keeps the stream open (a single bad write doesn't tear
            the stream; the engine's main thread is never blocked).
            Only the first three failures and every 100th after that
            are recorded, so a dead device can't flood the error ring.
        """
        import sounddevice as sd

        # `sd.OutputStream` does not have published type stubs; use Any
        # so the .start/.stop/.write/.close calls don't need per-line
        # ignores.
        stream: Any = None
        write_failures = 0
        while not self._stop_event.is_set():
            want_monitor = bool(self.cfg.monitor)
            if want_monitor and stream is None:
                try:
                    stream = sd.OutputStream(
                        samplerate=self.cfg.sink_rate,
                        channels=self.cfg.channels,
                        dtype="float32",
                    )
                    stream.start()
                except Exception as e:
                    self.record_error(f"monitor open: {type(e).__name__}: {e}")
                    stream = None
                    time.sleep(0.05)
                    continue
            elif not want_monitor and stream is not None:
                with contextlib.suppress(Exception):
                    stream.stop()
                    stream.close()
                stream = None
            try:
                chunk = self._monitor_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if stream is None:
                continue  # drain the queue but discard if monitor is off
            try:
                stream.write(chunk.reshape(-1, 1))
            except Exception as e:
                write_failures += 1
                if write_failures <= 3 or write_failures % 100 == 0:
                    self.record_error(
                        f"monitor write failed (#{write_failures}): {type(e).__name__}: {e}"
                    )
        # Stop event set -- tear down.
        if stream is not None:
            with contextlib.suppress(Exception):
                stream.stop()
                stream.close()

    # ---- v0.11.0 - GPU clock lock + torch separate-stream keepalive ----------

    def _apply_thread_priority(self, *, label: str, priority: int = 60) -> None:
        """Pin to `cpu_affinity_core` and optionally raise priority.

        Called from inside whichever thread should be pinned; affinity /
        scheduling class are per-thread on Linux.

        v0.7.0-rc11 - `realtime_priority=True` requests SCHED_FIFO at the
        given `priority` (default 60). On hosts with RLIMIT_RTPRIO ≥ 60
        (or CAP_SYS_NICE), the thread becomes non-preemptible by
        user-space SCHED_OTHER tasks (KDE compositing, picom, browser,
        etc.). Falls back cleanly to nice(-10), then to a logged
        warning, on locked-down systems.

        B19 / perf-009: writer thread now passes `priority=59` so the
        engine main thread (priority 60) wins SCHED_FIFO tie-breaks
        without starving the writer. Same FIFO scheduler class - both
        threads still preempt SCHED_OTHER background work.
        """
        # B28 + B47: shared `audio.priority` helpers; warnings append to
        # `stats.priority_warnings` so the engine main / writer / inference
        # child can all report independent failures without stomping
        # `last_error`.
        from audio.priority import try_set_affinity, try_set_realtime_priority

        aff_warn = try_set_affinity(self.cfg.cpu_affinity_core, label)
        if aff_warn is not None:
            with self._stats_lock:
                self.stats.priority_warnings.append(aff_warn)
        if self.cfg.realtime_priority:
            rt_warn = try_set_realtime_priority(label, priority=priority)
            if rt_warn is not None:
                with self._stats_lock:
                    self.stats.priority_warnings.append(rt_warn)

    def _run_loop(self) -> None:
        import sounddevice as sd

        chunk_mic = int(self.cfg.mic_rate * self.cfg.chunk_seconds)
        # Reset SOLA buffers so a stop/start cycle doesn't leak stale audio.
        self.reset_streaming_state()
        # v0.6.7 - fresh stateful resamplers. Built per `(src, dst)` pair so
        # filter state survives across chunks; hot-swapped if the model SR
        # changes mid-session (see `_maybe_swap_model`).
        self._resampler_in = _StreamResampler(self.cfg.mic_rate, 16_000)
        self._resampler_out = _StreamResampler(self._rvc_output_sr, self.cfg.sink_rate)

        # v0.5.2: pin engine main thread + bump priority if requested.
        self._apply_thread_priority(label="engine")

        try:
            # v0.5.2: open pacat under the lock + start writer/stderr/watchdog
            # threads before the first inference. The writer queue is sized
            # so the engine can sprint ahead of pacat for ~2 s without
            # blocking, then pressures back through queue_full_events.
            initial_proc = self._open_pacat()
            with self._pacat_lock:
                self._pacat_proc = initial_proc
            # v0.6.7 part 3 - prime the playback backend's stream buffer
            # with `prime_silence_seconds` of zeros before any real audio.
            # Without priming, the buffer steady-state oscillates 0 → chunk
            # → 0 → chunk; engine writer jitter (~30 ms std) pushes the
            # buffer to 0 frequently → pacat reports xruns and outputs one
            # PipeWire quantum (~21-43 ms) of silence per underrun. With a
            # 1x chunk pre-roll, the buffer floor lifts above 0 and only
            # outsized jitter (>chunk_seconds) can underrun.
            # Trade-off: this adds prime_silence_seconds to mic-to-app
            # wall-clock latency. Default 0.25 s matches chunk_seconds -
            # smallest pre-roll that fully bridges typical jitter.
            prime_n = int(self.cfg.sink_rate * self.cfg.prime_silence_seconds)
            if prime_n > 0 and initial_proc.stdin is not None:
                silence = np.zeros(prime_n * self.cfg.output_channels, dtype=np.float32).tobytes()
                # B12 / corr-011: take `_pacat_lock` for the prime-silence
                # write. Pre-v0.8.0 this was safe by accident (writer/watchdog
                # threads weren't started yet at this point in start()), but
                # the order was fragile. Locking makes it explicit so a
                # future reorder doesn't introduce a race.
                with self._pacat_lock, contextlib.suppress(BrokenPipeError, OSError):
                    initial_proc.stdin.write(silence)
                    initial_proc.stdin.flush()
            self._writer_queue = queue.Queue(maxsize=self.cfg.pacat_writer_queue_size)
            self._last_writer_ts = None
            self._pacat_dead_event.clear()
            self._writer_thread = threading.Thread(
                target=self._writer_loop, name="woys-pacat-writer", daemon=True
            )
            self._writer_thread.start()
            self._stderr_thread = threading.Thread(
                target=self._stderr_reader_loop,
                args=(initial_proc,),
                name="woys-pacat-stderr",
                daemon=True,
            )
            self._stderr_thread.start()
            self._watchdog_thread = threading.Thread(
                target=self._watchdog_loop, name="woys-pacat-watchdog", daemon=True
            )
            self._watchdog_thread.start()

            # v0.10.0-rc3 - GPU keep-alive thread (default off). Spawns
            # only in legacy in-process mode where we own `_cv` directly;
            # the IPC subprocess mode keeps the GPU warm via its own
            # constant-rate inference and doesn't need the keepalive.
            # v0.11.0 - torch separate-stream keepalive takes precedence
            # over the rc3 ORT-stream version when either is enabled. The
            # rc3 version remains as a no-torch fallback for environments
            # where torch isn't installed.
            _, torch_keepalive_on = self._resolve_anti_jitter_flags()
            if torch_keepalive_on and not self.cfg.inference_subprocess:
                self._torch_keepalive_thread = threading.Thread(
                    target=self._torch_keepalive_loop,
                    name="woys-torch-keepalive",
                    daemon=True,
                )
                self._torch_keepalive_thread.start()
            elif (
                self.cfg.gpu_keepalive_enabled
                and not self.cfg.inference_subprocess
                and self._cv is not None
            ):
                # Legacy rc3 ORT-stream keepalive - only spun up when
                # torch keepalive is OFF AND the rc3 knob is explicitly
                # set. Allocate the dummy input once, here, so the
                # warmup pass in _keepalive_loop doesn't allocate on
                # the hot path.
                self._keepalive_input = np.zeros(self.cfg.gpu_keepalive_input_len, dtype=np.float32)
                self._keepalive_thread = threading.Thread(
                    target=self._keepalive_loop, name="woys-keepalive", daemon=True
                )
                self._keepalive_thread.start()

            in_stream = sd.InputStream(
                samplerate=self.cfg.mic_rate,
                channels=self.cfg.channels,
                blocksize=chunk_mic,
                dtype="float32",
                device=self.cfg.input_device,
            )
            # the monitor stream
            # lifecycle now lives in `_monitor_writer_loop` (spawned
            # from `_worker_preamble`). The pre-fix eager-open here
            # is gone -- the dedicated thread opens / closes the
            # stream as cfg.monitor toggles, and writes go through
            # the bounded queue.

            # v0.7.0-rc4 - input-gate hysteresis state. The gate must
            # observe ≥`input_gate_hysteresis_ms` of continuously-below
            # threshold input before it fires; voice transients (brief
            # dips between syllables, plosive onsets, fricative onsets)
            # no longer trigger zero-emission. `gate_below_since` is
            # the perf_counter time of the first below-threshold sample
            # in the current run (None when above threshold).
            gate_below_since: float | None = None
            hysteresis_s = max(0.0, self.cfg.input_gate_hysteresis_ms / 1000.0)
            # B31 / corr-016: documented disable sentinel is -200.0 dBFS;
            # use it. The pre-v0.8.0 `> -120.0` cutoff was a magic number
            # (and gave qualitatively-different behavior for -120.0 vs
            # -119.999 - a value that should be a no-op kill threshold).
            gate_thresh = (
                10.0 ** (self.cfg.input_gate_dbfs / 20.0)
                if self.cfg.input_gate_dbfs > -200.0
                else 0.0
            )

            with in_stream:
                while not self._stop_event.is_set():
                    # v0.4.1: pick up any queued model swap before reading
                    # the next mic chunk. Owns _rvc on this thread, so no
                    # race with _infer below.
                    self._maybe_swap_model()
                    # drain any
                    # multi-field cfg apply queued by `request_cfg_
                    # update()` so the rest of this chunk sees a
                    # consistent view of all queued fields. Cheap when
                    # the queue is empty (one bool check inside a lock).
                    self._maybe_apply_pending_cfg()
                    # v0.7.0-rc4 - capture the overflow flag PortAudio
                    # returns when its internal ring buffer overran
                    # since the previous read. Pre-rc4 this was tuple-
                    # unpacked into `_` and lost; area 01 of the audit
                    # flagged it as a silent mic-side drop site
                    # invisible to every existing counter.
                    #
                    # v0.7.0-rc6 - wrapped with timing so we can attribute
                    # producer-side cadence variance to the mic read vs
                    # processing vs handoff. Steady-state mic_read_ms
                    # should hover near chunk_seconds * 1000; variance
                    # reflects ALSA period scheduling + USB iso jitter.
                    t_mic_pre = time.perf_counter()
                    data, overflowed = in_stream.read(chunk_mic)
                    mic_read_ms = (time.perf_counter() - t_mic_pre) * 1000.0
                    self.stats.last_mic_read_ms = mic_read_ms
                    with self._stats_lock:
                        self.stats._recent_mic_read_ms.append(mic_read_ms)
                    if overflowed:
                        with self._stats_lock:
                            self.stats.input_overflows += 1
                    audio = data.reshape(-1).astype(np.float32, copy=False)

                    # v0.5.1: software input pre-attenuation. Default 0 dB
                    # is a no-op (skip the multiply). Negative values trim
                    # hot mics so RVC doesn't amplify clipping as harsh
                    # distortion. RMS is measured AFTER the gain so the
                    # stat reflects what the model actually sees.
                    if self.cfg.input_gain_db != 0.0:
                        audio = audio * np.float32(10.0 ** (self.cfg.input_gain_db / 20.0))
                        # B21 / audio-007: positive `input_gain_db` can push
                        # samples beyond ±1.0; the RVC encoders see garbage on
                        # out-of-range input. Hard-clip post-gain so the
                        # vocoder always sees in-range audio. Users who want
                        # non-clipping headroom should attenuate at the mic
                        # (pre-amp side), not via woys.
                        if self.cfg.input_gain_db > 0.0:
                            np.clip(audio, -1.0, 1.0, out=audio)

                    # v0.14.0 (area 3 / C081): np.dot(a,a)/n is ~5x faster
                    # than sqrt(mean(a**2)) and avoids allocating an N-element
                    # squared-intermediate per chunk on the hot path.
                    rms = float(np.sqrt(np.dot(audio, audio) / audio.size))
                    self.stats.last_input_rms = rms

                    # v0.7.0-rc4 - gate with hysteresis. Below threshold
                    # alone is no longer enough to fire; the gate has to
                    # see hysteresis_s of continuous below-threshold
                    # input first. Above threshold resets the timer.
                    now = time.perf_counter()
                    if rms < gate_thresh:
                        if gate_below_since is None:
                            gate_below_since = now
                        if now - gate_below_since >= hysteresis_s:
                            n_silence = round(
                                audio.shape[0] * self.cfg.sink_rate / self.cfg.mic_rate
                            )
                            self._enqueue_chunk(
                                self._to_sink_bytes(np.zeros(n_silence, dtype=np.float32))
                            )
                            with self._stats_lock:
                                self.stats.gated_chunks += 1
                            continue
                        # Sub-hysteresis dip: pass through to inference. RVC
                        # on near-silent input emits near-silence anyway, so
                        # the cost is a few ms of compute we'd otherwise
                        # bypass. The benefit is that voice transients no
                        # longer get replaced with hard zeros.
                    else:
                        gate_below_since = None

                    t_total = time.perf_counter()
                    audio16 = (
                        self._resampler_in.process(audio)
                        if self._resampler_in is not None
                        else _resample(audio, self.cfg.mic_rate, 16_000)
                    )

                    t_inf = time.perf_counter()
                    # Streaming path uses SOLA + input history (Phase B). When
                    # `sola_enabled=False`, _process_streaming_16k still routes
                    # the model call through the history buffer but skips the
                    # crossfade - useful for A/B perf comparisons.
                    out_native = self._safe_process_streaming_16k(audio16)
                    inf_ms = (time.perf_counter() - t_inf) * 1000

                    if out_native is None or out_native.shape[0] == 0:
                        # `None`: inference raised - `_safe_*` already
                        # bumped `stats.dropped_chunks` and updated
                        # `stats.last_error`. Skip the write; SOLA's
                        # held-back tail covers the gap on resume.
                        # `shape[0] == 0`: first-chunk warmup or
                        # resampler buffer fill - emit nothing yet.
                        continue

                    # `out_native` is at the loaded RVC model's native sample
                    # rate (16k for amitaro, 40k for most v2 voices, etc.).
                    # v0.6.7: stream resampling preserves filter state across
                    # chunks so consecutive chunks splice without the 4 Hz
                    # warm-up artifact (`docs/11-microcuts-bug.md`).
                    out48 = (
                        self._resampler_out.process(out_native)
                        if self._resampler_out is not None
                        else _resample(out_native, self._rvc_output_sr, self.cfg.sink_rate)
                    )
                    if out48.size == 0:
                        # Soxr stream might emit nothing on the very first
                        # chunk while the internal buffer fills. Skip the
                        # write - the next chunk will produce extra samples.
                        continue

                    # v0.5.2: hand off to writer thread (non-blocking enqueue).
                    # The watchdog respawns pacat if it dies - main loop
                    # never raises out of the loop on a transient pacat fault.
                    #
                    # v0.7.0-rc6 - wrapped with timing. enqueue_lag_ms
                    # covers _to_sink_bytes (numpy convert) + put_nowait
                    # (queue insert). Should be sub-ms in steady state;
                    # spikes mean GC pause / GIL contention / queue
                    # backpressure (which would also bump
                    # `queue_full_events`).
                    t_enq_pre = time.perf_counter()
                    self._enqueue_chunk(self._to_sink_bytes(out48))
                    enq_lag_ms = (time.perf_counter() - t_enq_pre) * 1000.0
                    self.stats.last_enqueue_lag_ms = enq_lag_ms
                    with self._stats_lock:
                        self.stats._recent_enqueue_lag_ms.append(enq_lag_ms)

                    # v0.7.0-rc5 - pull SOLA's threshold-fallback count
                    # into engine stats. The rc4 `sola_drain_ms` (zero-
                    # pad bookkeeping) is gone because the pad itself is
                    # gone - SOLA emits constant-size chunks now. A
                    # non-zero fallback count means the alignment search
                    # is giving up (peak corr below threshold); it's a
                    # diagnostic, not a cuts driver.
                    if self._sola is not None:
                        self.stats.sola_fallback_count = self._sola.fallback_count
                        # F-31-05: far-edge-clipped peak count.
                        self.stats.sola_search_clipped = self._sola.search_window_clipped

                    # push to the
                    # bounded monitor queue (non-blocking). The
                    # dedicated `_monitor_writer_loop` thread owns
                    # the sd.OutputStream lifecycle + drains the
                    # queue. Queue overflow counts as a monitor
                    # drop (the user's self-monitor glitches but
                    # the engine main thread is NEVER blocked).
                    # Pre-fix the open/close + synchronous write
                    # happened here on the engine main thread; a
                    # slow host sink stalled the engine.
                    if self.cfg.monitor:
                        try:
                            self._monitor_queue.put_nowait(out48)
                        except queue.Full:
                            with self._stats_lock:
                                self.stats.monitor_drops += 1

                    total_ms = (time.perf_counter() - t_total) * 1000
                    with self._stats_lock:
                        self.stats.chunks_processed += 1
                        # clear a stale
                        # `last_error` once one full `chunk_seconds` has
                        # elapsed without a new failure write. The error
                        # ring (`error_history`) keeps the historical
                        # record; this only un-sticks the StatusPanel
                        # banner so a transient one-off doesn't read as
                        # "engine is currently broken" forever.
                        if (
                            self.stats.last_error is not None
                            and self.stats.last_error_ts is not None
                            and (time.monotonic() - self.stats.last_error_ts)
                            > self.cfg.chunk_seconds
                        ):
                            self.stats.last_error = None
                            self.stats.last_error_ts = None
                    # B63 / arch-012: optional periodic gc.collect(0) for users
                    # who run multi-hour sessions and observe heap growth.
                    # Default off (engine_periodic_gc_chunks=0).
                    if self.cfg.engine_periodic_gc_chunks > 0 and (
                        self.stats.chunks_processed % self.cfg.engine_periodic_gc_chunks == 0
                    ):
                        gc.collect(0)
                    self.stats.last_inference_ms = inf_ms
                    self.stats.last_total_ms = total_ms
                    if inf_ms > self.stats.max_inference_ms:
                        self.stats.max_inference_ms = inf_ms
                    if total_ms > self.stats.max_total_ms:
                        self.stats.max_total_ms = total_ms
                    if total_ms > self.cfg.chunk_seconds * 1000.0:
                        with self._stats_lock:
                            self.stats.late_chunks += 1
                        # v0.6.9 round 5 - capture per-stage breakdown for
                        # postmortem of which session caused the outlier.
                        # Capped at 50 entries so memory doesn't grow without
                        # bound on a degraded GPU.
                        self.stats.slow_chunk_log.append(
                            {
                                "chunk_idx": float(self.stats.chunks_processed),
                                "total_ms": total_ms,
                                "inf_ms": inf_ms,
                                "cv_ms": self.stats.last_cv_ms,
                                "rmvpe_ms": self.stats.last_rmvpe_ms,
                                "rvc_ms": self.stats.last_rvc_ms,
                                "input_rms": rms,
                            }
                        )
                        if len(self.stats.slow_chunk_log) > 50:
                            self.stats.slow_chunk_log.pop(0)
                    # v0.7.0-rc8 - tail-chunk capture, gated on inference
                    # time alone (not total_ms). Fires when inf_ms is more
                    # than 2x the running p50 of `_recent_inference`, which
                    # has been pre-this-chunk's-append at the bottom of the
                    # loop. Skip until the deque has ≥16 prior samples so
                    # the threshold is stable. Captures input-shape and
                    # per-session-stage data so we can read what slow
                    # chunks have in common after a Telegram run.
                    if len(self.stats._recent_inference) >= 16:
                        sorted_inf = sorted(self.stats._recent_inference)
                        inf_p50 = sorted_inf[len(sorted_inf) // 2]
                        if inf_p50 > 0 and inf_ms > inf_p50 * 2:
                            self.stats.tail_chunk_log.append(
                                {
                                    "chunk_idx": float(self.stats.chunks_processed),
                                    "inf_ms": inf_ms,
                                    "inf_p50_ref": float(inf_p50),
                                    "cv_ms": self.stats.last_cv_ms,
                                    "rmvpe_ms": self.stats.last_rmvpe_ms,
                                    "rvc_ms": self.stats.last_rvc_ms,
                                    "audio16_len": float(audio16.shape[0]),
                                    "input_rms": rms,
                                    "mic_read_ms": float(mic_read_ms),
                                }
                            )
                            if len(self.stats.tail_chunk_log) > 50:
                                self.stats.tail_chunk_log.pop(0)
                    with self._stats_lock:
                        self.stats._recent_inference.append(inf_ms)
                        self.stats._recent_total.append(total_ms)
                        inf_snapshot = list(self.stats._recent_inference)
                        total_snapshot = list(self.stats._recent_total)
                    if inf_snapshot:
                        self.stats.avg_inference_ms = sum(inf_snapshot) / len(inf_snapshot)
                        self.stats.avg_total_ms = sum(total_snapshot) / len(total_snapshot)
        except Exception as e:
            self.record_error(f"{type(e).__name__}: {e}")
            self.stats.running = False
            # mark this as a crash, not a clean
            # stop, so the headless `cmd_engine` loop can break + exit
            # non-zero instead of printing frozen stats for the full
            # --seconds.
            self.stats.crashed = True
        finally:
            # v0.5.2: flush SOLA tail through the writer queue, then drain
            # the queue, then tear down the writer/watchdog/stderr threads,
            # then close pacat.
            if self._sola is not None:
                with contextlib.suppress(Exception):
                    tail = self._sola.flush()
                    if tail.size > 0 and self._resampler_out is not None:
                        tail48 = self._resampler_out.process(tail)
                        flush48 = self._resampler_out.flush()
                        full = np.concatenate([tail48, flush48]) if flush48.size else tail48
                        if full.size > 0:
                            self._enqueue_chunk(self._to_sink_bytes(full))
            # Wait briefly for the writer to drain its queue.
            if self._writer_queue is not None:
                deadline = time.perf_counter() + 1.0
                while time.perf_counter() < deadline and not self._writer_queue.empty():
                    time.sleep(0.02)
            # monitor stream
            # teardown moved to `_monitor_writer_loop`'s exit. That
            # thread sees `_stop_event` set + closes its own stream
            # before exiting.
            # Tearing down threads + pacat. stop() has usually set
            # _stop_event already; on the crash path nobody has, so set it
            # here (after the drain) or the writer/watchdog/monitor threads
            # outlive this run and the joins below just time out.
            self._stop_event.set()
            with self._pacat_lock:
                final_proc = self._pacat_proc
                self._pacat_proc = None
            if final_proc is not None:
                try:
                    if final_proc.stdin is not None:
                        final_proc.stdin.close()
                except Exception:
                    pass
                try:
                    final_proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    final_proc.terminate()
                    try:
                        final_proc.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        final_proc.kill()
            # Join helper threads so a fast restart sees a clean slate.
            for t in (
                self._writer_thread,
                self._watchdog_thread,
                self._stderr_thread,
                self._keepalive_thread,
                self._torch_keepalive_thread,
            ):
                if t is not None and t.is_alive():
                    t.join(timeout=0.5)
            self._writer_thread = None
            self._watchdog_thread = None
            self._stderr_thread = None
            self._keepalive_thread = None
            self._keepalive_input = None
            self._torch_keepalive_thread = None
            self._writer_queue = None
            # The loop is gone, so nothing will apply a queued swap. After
            # a crash or self-stop no stop() has run yet: resolve them here
            # and stop reporting the engine as running.
            self._fail_pending_swaps("engine stopped before swap completed")
            if self.stats.crashed:
                self.stats.running = False
