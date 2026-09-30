"""`EngineStats`: the counters, rolling windows and error history the engine
shares with the TUI, `woys diag` and `woys engine`.

Split out of `audio.engine` (which re-exports it). Every read-modify-write
of a shared field goes through `_internal_lock`, aliased as the engine's
`_stats_lock`; see the module docstring of `audio.engine` for the threads
involved.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EngineStats:
    running: bool = False
    # set True iff `_run_loop` exited via an
    # *unhandled exception* (not a clean `stop()`). `running` alone cannot
    # distinguish "the worker crashed" from "someone called stop()", and
    # exit-code correctness for the headless / WM-scripting path depends on
    # that distinction.
    crashed: bool = False
    chunks_processed: int = 0
    last_input_rms: float = 0.0
    last_inference_ms: float = 0.0
    avg_inference_ms: float = 0.0
    last_total_ms: float = 0.0
    avg_total_ms: float = 0.0
    # v0.6.9 - outlier visibility. avg_*_ms hides single slow chunks that
    # arrive after the audio sink has already underrun. max_* tracks the
    # worst chunk since session start; late_chunks counts chunks where total
    # processing exceeded the chunk budget (chunk_seconds * 1000).
    max_inference_ms: float = 0.0
    max_total_ms: float = 0.0
    late_chunks: int = 0
    # v0.6.9 round 5 - per-stage timing for the most recent chunk so the
    # slow_chunk_log breakdown points at which ONNX session was responsible.
    last_cv_ms: float = 0.0
    last_rmvpe_ms: float = 0.0
    last_rvc_ms: float = 0.0
    # Last N chunks where total_ms exceeded the chunk budget. Each entry is a
    # dict {chunk_idx, total_ms, inf_ms, cv_ms, rmvpe_ms, rvc_ms, input_rms}.
    # Surface via the SLOW socket command -> /tmp/woys-slow-chunks.txt.
    slow_chunk_log: list[dict[str, float]] = field(default_factory=list)
    # v0.7.0-rc8 - chunks whose inference time was > 2x the running
    # p50 of recent inference, regardless of whether total_ms passed
    # chunk_seconds*1000. The rc7 diag dump showed inference p50=40 ms
    # / p99=96 ms / max=110 ms with overrun_ratio=0 - a 70 ms tail
    # spread that doesn't trip the existing `slow_chunk_log` (gated on
    # total_ms > chunk_seconds*1000 = 150 ms). This list captures the
    # tail chunks so we can correlate their inf_ms with input shape,
    # history size, RMS, and per-session-stage breakdown. If slow
    # chunks share a common signature (specific audio16_len, specific
    # cv vs rmvpe vs rvc dominance, specific RMS band), rc9's fix
    # targets that mechanism. Capped at 50 entries.
    tail_chunk_log: list[dict[str, float]] = field(default_factory=list)
    last_error: str | None = None
    # timestamp of the most recent
    # `last_error` write (monotonic seconds). The TUI uses this to
    # render an age ("error: ... (3 s ago)") instead of a sticky string
    # the reader cannot distinguish from a freshly minted failure --
    # pre-fix `last_error` survived from session start until clobbered,
    # so a user glancing at StatusPanel could not tell whether the
    # engine had crashed five seconds ago or five minutes ago. Set
    # under `_stats_lock` by `record_error()`; cleared together with
    # `last_error` by the chunk-success path one `chunk_seconds` after
    # the most recent failure (so a real cascade still surfaces, but a
    # one-off transient self-clears once the engine is healthy again).
    last_error_ts: float | None = None
    # cold-start progress.
    # Pre-fix start() ran _ensure_sessions + _warmup_realtime_pipeline
    # + (optional) eager_warmup + GPU clock-lock subprocess.run all on
    # the caller's thread before spawning the worker -- the TUI froze
    # for multi-seconds with a stale "~2s" toast. Post-fix start()
    # returns immediately after spawning the worker; the worker does
    # the heavy preamble and updates this field through documented
    # states: "" / "starting" / "checking default sink" / "applying
    # GPU clock lock" / "loading sessions" / "spawning inference
    # subprocess" / "warming pipeline" / "eager warming voice
    # library" / "ready" / "crashed: <ExceptionName>". The TUI polls
    # this field via `_refresh_stats` so the user sees a live stage,
    # not a frozen UI.
    warmup_stage: str = ""
    # count chunks the engine had
    # to drop on its way to the monitor stream because the bounded
    # `_monitor_queue` was full. Each drop means the monitor sink
    # (host default audio device) was slower than the chunk cadence
    # -- the user's self-monitor briefly glitches but the engine's
    # main thread is NOT blocked. Pre-fix the engine wrote synchronously
    # to `monitor_stream.write()` and stalled on a slow sink.
    monitor_drops: int = 0
    # bounded timestamped error history. Pre-
    # fix `last_error` was a single clobberable string with ~28 write
    # sites across 6 threads. A real failure cascade (`subprocess died
    # -> sessions reloading -> pacat respawn failed`) overwrote
    # `last_error` repeatedly so the user saw only the LAST symptom in
    # `woys diag`. The acknowledged-but-ungeneralized fix already
    # existed at engine.py:868 (`helper_exit_reasons`) and the watchdog
    # used a separate list "rather than clobber last_error wholesale";
    # this generalizes the pattern to all errors.
    #
    # Tuple shape: `(monotonic_ts, thread_name, message)`. `maxlen=20`
    # is bounded so memory does not grow without limit on a degraded
    # session. Reads are unlocked (snapshot semantics are OK; the
    # consumer is `woys diag` / TUI status, not a control-flow path).
    # Writes go through `RealtimeEngine.record_error()`, which appends
    # under `_stats_lock` AND mirrors to `last_error` for back-compat.
    error_history: deque[tuple[float, str, str]] = field(default_factory=lambda: deque(maxlen=20))

    # v0.5.2 health counters (Brief §5 - surfaced in TUI + `diag`).
    # xruns: parsed from pacat -v stderr. Closest thing to a true
    #   PulseAudio-side underrun count without reaching into pw-dump.
    # queue_full_events: writer queue was full when the engine tried to
    #   enqueue → engine has out-paced the writer/sink, treat as a
    #   self-detected underrun.
    # player_restarts: watchdog respawned the playback backend
    #   (pacat / pw-cat / native-pw helper) - it died mid-session.
    #   v0.9.0-rc4 rename: was `pacat_restarts` through v0.9.0-rc3;
    #   the legacy attribute alias is provided below for back-compat.
    # writer_jitter_ms: std dev (ms) of recent inter-chunk write
    #   intervals. Exceeding ~5 % of chunk_seconds*1000 is the
    #   underrun precursor we care about.
    xruns: int = 0
    queue_full_events: int = 0
    player_restarts: int = 0
    writer_jitter_ms: float = 0.0
    # v0.9.0 - when the native-pw helper is in use, the helper prints
    # "underruns=N\n" on stderr roughly once per second; the engine's
    # stderr-reader parses those lines into this counter. Closes audit
    # area 09 rank 1 ("pw-cat is silent on underruns; we swapped a
    # metric we could see for one we can't"). Stays 0 in pw-cat /
    # pacat modes (those backends don't emit `underruns=` lines).
    player_underruns: int = 0
    # v0.6.8 - count of chunks the engine had to drop because inference
    # raised (GPU OOM, numerical, transient ORT error). Without this,
    # any single bad chunk crashes the entire engine; with it, we drop
    # the chunk, leave a brief silence (SOLA tail covers most of it),
    # and keep going. First few hits log to `last_error`; subsequent
    # ones increment silently to avoid spamming the TUI.
    dropped_chunks: int = 0
    # v0.7.0-rc4 - instrumentation for the four silent-drop classes the
    # internal notes audit identified as previously
    # invisible to every existing counter. Each is incremented at
    # the exact site that emits zeros / loses samples; together with
    # `dropped_chunks` and `queue_full_events` they cover every
    # silence-emit path the audit catalogued. Surfaced in `woys diag`
    # output and the TUI STATUS reply so the next debug cycle isn't
    # blind.
    #
    #   input_overflows    - sd.InputStream.read() reported
    #                        `overflowed=True` (mic-side ring underflow,
    #                        previously dropped on the floor at the
    #                        tuple-unpack site).
    #   gated_chunks       - input gate fired and emitted a chunk of
    #                        zeros; bypasses SOLA + resamplers +
    #                        inference, so an upstream of every buffer.
    #   nan_chunks         - RVC vocoder output had NaN/inf and was
    #                        sanitized to zero (v0.6.9 path); a
    #                        non-zero rate during real-speech is
    #                        evidence for the C-class hypothesis from
    #                        the audit.
    #   sola_fallback_count - SOLA's alignment search peak correlation
    #                        fell below `corr_threshold`; the algorithm
    #                        used `offset = 0` (centered, no shift). In
    #                        rc5 this no longer affects emit length -
    #                        SOLA always emits `chunk_n` samples per call
    #                        regardless of fallback - so this counter is
    #                        purely a "how often is the search giving up"
    #                        diagnostic, not a cuts driver.
    #   sola_search_clipped -. Counts
    #                        chunks where the alignment search's peak
    #                        landed at the FAR edge of the [0, search]
    #                        window with corr above threshold. Distinct
    #                        from `sola_fallback_count`: the offset was
    #                        trusted, but the peak hit `best_idx ==
    #                        search`, signaling the true alignment may
    #                        lie beyond the one-sided window. A non-zero
    #                        rate on real audio is evidence against the
    #                        "RVC bias is purely toward late emission"
    #                        assumption that motivates the one-sided
    #                        contract (`sola._best_offset` docstring).
    #
    # rc4's `sola_drain_ms` (cumulative ms of zero-padding) was removed
    # in rc5 because the pad path itself was removed. SOLA emits
    # constant-size chunks now (internal notes
    # §"Proposed rc5 scope"). Drain is structurally zero by construction.
    input_overflows: int = 0
    gated_chunks: int = 0
    nan_chunks: int = 0
    sola_fallback_count: int = 0
    sola_search_clipped: int = 0

    # v0.7.0-rc6 - per-stage producer-side timing for the writer-jitter
    # investigation. The rc5 postmortem
    # attributed the
    # live `writer_jitter_ms = 62` to producer-side cadence variance,
    # not consumer-side. These two new stages plus the existing
    # inference timing sum to per-iteration wall time:
    #
    #   mic_read_ms       blocking read of chunk_mic samples (PortAudio
    #                     / ALSA - should hover near chunk_seconds *
    #                     1000 in steady state; variance reflects
    #                     ALSA period scheduling + USB iso jitter)
    #   inference_ms      RVC inference (existing, percentiles new)
    #   enqueue_lag_ms    output resample + _to_sink_bytes + put_nowait
    #                     (should be sub-ms in steady state; spikes
    #                     mean GC pause / GIL contention / queue full)
    #
    # `woys diag` surfaces p50/p95/p99 of each so we can attribute the
    # 62 ms cadence variance to a specific stage in one Telegram run.
    last_mic_read_ms: float = 0.0
    last_enqueue_lag_ms: float = 0.0

    # Rolling latency window for the TUI.
    # B43 / quality-006: 128-deep rolling window for all stat surfaces so
    # p95/p99 readings have enough samples to be stable. Pre-v0.8.0,
    # `_recent_inference` and `_recent_total` were 32-deep, which made
    # their p99 jumpy; mic_read / enqueue_lag / writer_intervals were
    # already 128. Single window size, single mental model.
    _recent_inference: deque[float] = field(default_factory=lambda: deque(maxlen=128))
    _recent_total: deque[float] = field(default_factory=lambda: deque(maxlen=128))
    # v0.5.2 - inter-write intervals in ms (writer thread fills this).
    _writer_intervals_ms: deque[float] = field(default_factory=lambda: deque(maxlen=128))
    # v0.7.0-rc6 - wider window than _recent_inference (32) so p95/p99
    # have enough samples to be stable. 128 chunks ≈ 19 s at
    # chunk_seconds=0.15.
    _recent_mic_read_ms: deque[float] = field(default_factory=lambda: deque(maxlen=128))
    _recent_enqueue_lag_ms: deque[float] = field(default_factory=lambda: deque(maxlen=128))

    # v0.10.0 - per-stage rolling windows for cv / rmvpe / rvc inference.
    # The engine has tracked `last_*_ms` (most-recent only) and `inf_ms`
    # (sum, with rolling p50/p95/p99) since v0.6.9. The aggregated `inf_ms`
    # mixes the contribution of each stage; tail variance attribution
    # requires per-stage percentiles. v0.10.0's writer-jitter investigation
    # uses these to identify which stage owns the p99 tail. Populated by
    # `_infer` in both legacy in-process and IPC-subprocess paths.
    _recent_cv_ms: deque[float] = field(default_factory=lambda: deque(maxlen=128))
    _recent_rmvpe_ms: deque[float] = field(default_factory=lambda: deque(maxlen=128))
    _recent_rvc_ms: deque[float] = field(default_factory=lambda: deque(maxlen=128))
    # v0.10.0-rc2 - RVC stage further split into pre / run / post so we
    # can attribute the rvc tail to GPU work vs Python pre-/post-process
    # (np.repeat, to_pitch_coarse, astype, isnan/isinf scan). Populated
    # only by the legacy in-process path; the IPC child reports an
    # aggregate `rvc_ms` over the wire (rc3 will plumb the split through
    # the protocol if rvc-pre/post turns out to be load-bearing).
    _recent_rvc_pre_ms: deque[float] = field(default_factory=lambda: deque(maxlen=128))
    _recent_rvc_run_ms: deque[float] = field(default_factory=lambda: deque(maxlen=128))
    _recent_rvc_post_ms: deque[float] = field(default_factory=lambda: deque(maxlen=128))
    # Set of `audio16k.shape[-1]` values seen at inference entry. The rc9
    # broader pre-warm targets soxr's polyphase alternation pattern (4
    # shapes seen on a 48 kHz USB condenser mic: 1957/1958/2446/2447). If runtime
    # introduces shapes outside the pre-warm set, cuDNN re-tunes on the
    # cold shape (~80 ms one-off cost vs ~25 ms cached). The brief lists
    # this as v0.10.x candidate #2; counter-evidence is "set size ≤ 4
    # AND ⊆ warmup_shapes after the first 30 s of runtime."
    unique_audio16_lens: set[int] = field(default_factory=set)
    # Snapshot of the warmup-time shape set, taken once at the end of
    # `_warmup_realtime_pipeline`. Compared against `unique_audio16_lens`
    # in `woys diag` to spot the rc9 gap class.
    warmup_audio16_lens: set[int] = field(default_factory=set)

    # v0.8.0 - inference subprocess telemetry. None / 0 when running
    # in-process (legacy path).
    child_pid: int | None = None
    child_restarts: int = 0
    last_ipc_roundtrip_ms: float = 0.0
    _recent_ipc_roundtrip_ms: deque[float] = field(default_factory=lambda: deque(maxlen=128))

    # v0.8.1 - per-session TRT EP status. After `_ensure_sessions`
    # runs, this maps each loaded model's filename to True (TRT
    # active) or False (CUDA EP fallback because TRT init failed).
    # `trt_init_errors` records the failure reason per model so
    # `woys diag` can print it. Empty when `cfg.use_tensorrt=False`.
    trt_active_for: dict[str, bool] = field(default_factory=dict)
    trt_init_errors: dict[str, str] = field(default_factory=dict)

    # True if any model session bound
    # CPU-only. Only reachable through the test-only cfg.allow_cpu_fallback --
    # otherwise `_make_session` raises CpuFallbackError instead.
    # Printed by `woys diag`.
    cpu_fallback_active: bool = False

    # B28 / corr-009: thread priority + affinity warnings. Each entry
    # describes one failure (engine main, writer, child) so a user with
    # multiple priority issues can see all of them, not just the last.
    # bounded deque. Pre-fix
    # `list` + bare `.append()` was unbounded -- a long-running
    # session that hit repeating keepalive / monitor / debug-log
    # failures would grow this list without limit, slowly eating
    # memory. The F-merged-015 error-ring pattern (deque maxlen=20)
    # is the right shape for "rolling diagnostic that bounds itself".
    # Reads in test_pacat_health iterate the list shape, which
    # `collections.deque` also supports.
    priority_warnings: deque[str] = field(default_factory=lambda: deque(maxlen=20))

    # v0.10.0-rc3 - GPU keep-alive thread observability. Stays at zero
    # when `gpu_keepalive_enabled=False` (default).
    keepalive_calls: int = 0
    last_keepalive_ms: float = 0.0
    keepalive_avg_ms: float = 0.0
    _recent_keepalive_ms: deque[float] = field(default_factory=lambda: deque(maxlen=128))

    # v0.11.0 - torch keepalive (separate CUDA stream). Distinct from
    # rc3 ORT keepalive counter so we can A/B them. Reads "torch_*"
    # in `woys diag` so the active backend is unambiguous.
    torch_keepalive_calls: int = 0
    torch_keepalive_last_ms: float = 0.0
    torch_keepalive_avg_ms: float = 0.0
    _recent_torch_keepalive_ms: deque[float] = field(default_factory=lambda: deque(maxlen=128))

    # v0.11.0 - GPU clock-lock state. Set by `_apply_gpu_clock_lock()` on
    # engine start when `cfg.gpu_clock_lock_enabled=True` (or
    # `gpu_anti_jitter_mode in {"clock_lock","both"}`); cleared by
    # `_revert_gpu_clock_lock()`. The lock is reverted on engine.stop()
    # AND on SIGTERM/SIGINT (see RealtimeEngine.__init__ - _signal_handler).
    gpu_clock_lock_active: bool = False
    gpu_clock_lock_floor_mhz: int = 0
    gpu_clock_lock_ceiling_mhz: int = 0
    # Latest nvidia-smi -lgc / -rgc result message; surfaced in
    # `woys diag` so apply / revert failures are visible.
    gpu_clock_lock_last_message: str = ""
    # v0.14.0 (area 17 / area 19 / C019): True iff the most recent
    # `nvidia-smi -rgc` failed (sudo revoked, driver flicker). The next
    # engine.start() inspects this flag and attempts a fresh -rgc before
    # applying a new lock; otherwise the GPU stays locked across
    # sessions with only `last_error` (easily clobbered) as evidence.
    gpu_clock_lock_revert_failed: bool = False

    # the rolling-window deques
    # above (~11 of them) are appended from one thread (engine worker,
    # writer, GPU keepalive thread) and read from ANOTHER thread (TUI
    # poll / `woys diag`) via the `inference_samples()` /
    # `writer_interval_samples_ms()` accessors below. `np.array(deque)`
    # and `list(deque)` iterate, and a concurrent append raises
    # `RuntimeError: deque mutated during iteration`. The reader
    # thread silently dies; in the writer-jitter probe (engine.py:3125
    # below) the dying thread is the writer itself, which means the
    # audio stops -- the most serious of F-merged-017's three bug
    # classes.
    #
    # Both sides take this lock. Append sites in `engine.py` wrap with
    # `with self._stats_lock:`; iteration sites in the methods below
    # wrap with `with self._internal_lock:`. The engine's
    # `_stats_lock` (set in `RealtimeEngine.__init__`) IS this lock
    # (`engine._stats_lock = stats._internal_lock`), so callers on
    # either side reach the same primitive.
    _internal_lock: Any = field(default_factory=threading.RLock, repr=False, compare=False)

    # v0.11.0 - track the helper's last-known exit cause(s) so the
    # watchdog's "respawned" message doesn't clobber the original
    # death reason from `_stderr_reader_loop`. List-of-strings, capped
    # at 10 entries; surfaced in `woys diag` output. Each entry is
    # one of:
    #   "native-pw: error: <reason>"  - from the helper's own stderr
    #   "<backend> exited code=<N> at chunks=<idx>"  - from watchdog
    #     when no stderr-side cause was captured before the exit
    helper_exit_reasons: list[str] = field(default_factory=list)

    # v0.9.0-rc4 - back-compat alias for the field renamed from
    # `pacat_restarts` to `player_restarts`. Retained as a permanent alias:
    # first-party readers (the TUI status panel) and any external scripts
    # that scrape EngineStats by the old name keep working.
    @property
    def pacat_restarts(self) -> int:
        return self.player_restarts

    @pacat_restarts.setter
    def pacat_restarts(self, value: int) -> None:
        self.player_restarts = value

    # B23 / quality-019: public read-accessors for the rolling stat
    # windows. cli.py used to reach into the leading-underscore deques
    # directly, which made any future EngineStats refactor
    # silent-breaking.
    def inference_samples(self) -> list[float]:
        """Snapshot of the recent-inference rolling window in ms.

        copy under
        `_internal_lock` so a concurrent appender doesn't raise
        `RuntimeError: deque mutated during iteration`. Same for the
        ~11 sibling snapshot accessors below.
        """
        with self._internal_lock:
            return list(self._recent_inference)

    def total_samples(self) -> list[float]:
        """Snapshot of the recent-total rolling window in ms."""
        with self._internal_lock:
            return list(self._recent_total)

    def mic_read_samples_ms(self) -> list[float]:
        with self._internal_lock:
            return list(self._recent_mic_read_ms)

    def enqueue_lag_samples_ms(self) -> list[float]:
        with self._internal_lock:
            return list(self._recent_enqueue_lag_ms)

    # v0.10.0 - per-stage inference rolling-window accessors.
    def cv_samples_ms(self) -> list[float]:
        """Snapshot of the rolling per-chunk contentvec inference times in ms."""
        with self._internal_lock:
            return list(self._recent_cv_ms)

    def rmvpe_samples_ms(self) -> list[float]:
        """Snapshot of the rolling per-chunk RMVPE pitch-extraction times in ms."""
        with self._internal_lock:
            return list(self._recent_rmvpe_ms)

    def rvc_samples_ms(self) -> list[float]:
        """Snapshot of the rolling per-chunk RVC vocoder inference times in ms."""
        with self._internal_lock:
            return list(self._recent_rvc_ms)

    def writer_interval_samples_ms(self) -> list[float]:
        """Snapshot of the writer-thread inter-flush intervals in ms.
        The std-dev is `writer_jitter_ms`; p99 is the load-bearing tail
        metric the v0.10.x investigation targets (acceptance gate ≤30 ms)."""
        with self._internal_lock:
            return list(self._writer_intervals_ms)

    def rvc_pre_samples_ms(self) -> list[float]:
        """Time spent in numpy pre-processing between RMVPE done and
        `self._rvc.run` invocation: feats_2x = np.repeat, to_pitch_coarse,
        slice/reshape/astype on coarse + aligned pitch tensors."""
        with self._internal_lock:
            return list(self._recent_rvc_pre_ms)

    def rvc_run_samples_ms(self) -> list[float]:
        """Time spent inside `self._rvc.run` itself (the GPU op).
        Compare against rvc_pre and rvc_post to split the rvc tail
        between GPU and Python overhead."""
        with self._internal_lock:
            return list(self._recent_rvc_run_ms)

    def rvc_post_samples_ms(self) -> list[float]:
        """Time spent in numpy post-processing between rvc.run return
        and the result returned to caller: np.array(out).astype.squeeze,
        isnan/isinf scan, optional nan_to_num replacement."""
        with self._internal_lock:
            return list(self._recent_rvc_post_ms)


# EngineStats fields that describe more than one run, so start() keeps them
# when it resets the per-session counters: the error ring, and the GPU
# clock-lock state (`_apply_gpu_clock_lock` recovers a lock whose revert
# failed on the previous stop).
_STATS_KEPT_ACROSS_RUNS = frozenset(
    {
        "error_history",
        "gpu_clock_lock_active",
        "gpu_clock_lock_floor_mhz",
        "gpu_clock_lock_ceiling_mhz",
        "gpu_clock_lock_last_message",
        "gpu_clock_lock_revert_failed",
        "_internal_lock",
    }
)
