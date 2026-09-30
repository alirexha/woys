"""Engine configuration: `EngineConfig`, the model defaults and the list of
user-visible fields.

Kept free of numpy / onnxruntime so the config layer (`tui.config`,
`woys.profiles`, `woys models`) can import it without loading the engine and
its CUDA libraries. `audio.engine` re-exports every name here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MODELS_DIR = Path.home() / ".local" / "share" / "woys" / "models"

# Defaults pulled from Phase 1 inventory.
DEFAULT_RVC_MODEL = MODELS_DIR / "amitaro_v2_16k.onnx"
DEFAULT_RMVPE = MODELS_DIR / "rmvpe_wrapped.onnx"
DEFAULT_CONTENTVEC = MODELS_DIR / "contentvec-f.onnx"


# B9 / arch-004 / arch-005 - single source of truth for "this EngineConfig
# field is user-visible". AppConfig's forwarded set, profiles._PROFILE_FIELDS,
# vcprofile.py snapshot keys, and the migration code's allowlist all derive
# from this. New EngineConfig field added without listing it here will fail
# tests/test_engine_config_drift.py - that's the design.
#
# Excluded categories:
#   - System-only knobs that need an engine restart anyway (session_pool_size,
#     cpu_affinity_core, realtime_priority, eager_warmup, pacat_writer_queue_size,
#     prime_silence_seconds, pacat_watchdog_interval_s, threshold).
#   - Path-typed model defaults (rvc_model, rmvpe_model, contentvec_model) -
#     handled with bespoke str↔Path conversion at the AppConfig boundary.
#   - Subprocess / TRT toggles (inference_subprocess, use_tensorrt) -
#     experimental, intentionally kept out of the user-tunable surface.
USER_VISIBLE_ENGINE_FIELDS: tuple[str, ...] = (
    "f0_up_key",
    "sid",
    "chunk_seconds",
    "mic_rate",
    "sink_rate",
    "sink_name",
    "monitor",
    "output_latency_ms",
    "output_process_time_ms",
    "embedder",
    "sola_enabled",
    "sola_crossfade_ms",
    "sola_search_ms",
    "sola_context_ms",
    "input_gain_db",
    "input_gate_dbfs",
    "input_gate_hysteresis_ms",
    "prefer_pw_cat",
    "prefer_native_pw",
    "prefer_native_pw_buffer_ms",
    "gpu_keepalive_enabled",
    "gpu_keepalive_interval_ms",
    "gpu_keepalive_input_len",
    "gpu_anti_jitter_mode",
    "gpu_clock_lock_enabled",
    "gpu_clock_lock_floor_mhz",
    "gpu_clock_lock_ceiling_mhz",
    "gpu_clock_lock_floor_offset_mhz",
    "gpu_keepalive_torch_stream",
    "gpu_keepalive_torch_interval_ms",
)


@dataclass
class EngineConfig:
    rvc_model: Path = DEFAULT_RVC_MODEL
    rmvpe_model: Path = DEFAULT_RMVPE
    contentvec_model: Path = DEFAULT_CONTENTVEC

    # Audio I/O
    mic_rate: int = 48_000
    sink_rate: int = 48_000
    # v0.7.0 - dropped from 0.25 → 0.15. Empirical sweep (RTX 2070 Mobile,
    # ORT-CUDA, cuDNN HEURISTIC, full realtime engine, catwoman voice):
    #
    #   chunk   late_chunks/total   inference avg   p99
    #   0.10    13-42 / 100         77-98 ms       103-148 ms
    #   0.15    0 / 80              76-80 ms       104-129 ms
    #   0.20    0 / 60              92-96 ms       115-122 ms
    #   0.25    0 / 50              83 ms          124 ms
    #
    # At chunk=0.10 the per-chunk budget is 100 ms but real engine inference
    # is 77-98 ms (a +50 ms tax over the standalone benchmark, traced to
    # GIL/scheduler effects of running inside the engine sub-thread; see
    # LESSONS §19). 13-42 % of chunks miss budget at 0.10, even on the
    # smallest voice. At chunk=0.15 the budget is 150 ms and zero chunks
    # miss it across both light and heavy voices - that's the practical
    # floor on this hardware. v0.7.0 picks 0.15 (saves 100 ms vs the v0.6.x
    # 0.25 default; doesn't pretend 0.10 is achievable).
    #
    # Historical v0.5.1 reason for raising chunk to 0.25 ("SOLA tail-trim ate
    # 10 % of output duration") was repaired by the v0.6.9 SOLA tuning
    # (search_ms 4.0 → 6.0, corr_threshold 0.25 → 0.10). Historical v0.6.7
    # reason ("dropped chunks during cuDNN warmup") was repaired by the
    # v0.7.0 HEURISTIC switch + broader pre-warm shape coverage.
    # v0.12.4 - bumped 0.15 → 0.25 after a user perceptual A/B against the
    # v0.12.3 sweep's top-1 opt-in config (CHANGELOG.md v0.12.4 entry,
    # LESSONS.md §42). Top-1 has chunk_seconds=0.25 + sola_context_ms=200
    # which together drive the chunk-period spectral autocorrelation to
    # exactly 0.000 - the "train wagon on rails" rhythm the user heard
    # on sustained content disappears. Trade: +100 ms total e2e latency
    # (~540 ms → ~640 ms). Conversational threshold (~700 ms) is still
    # comfortably above this; the perceptual delta dwarfs the latency
    # penalty per the user's listening test.
    #
    # honesty caveat: that
    # "perceptual delta dwarfs the latency penalty" claim is from an
    # **n=1 author A/B on desktop WAV playback** during v0.12.4. It
    # has NOT been validated in the production Discord / CS2 VoIP
    # path, where end-to-end latency stacks (mic capture +
    # transcoder + Discord's voice-activation gate + the listener's
    # output buffer) sit on top of the woys engine. Treat 0.25 as
    # the chosen default for the WAV playback context where the
    # original A/B happened, not a settled fact for every consumer.
    chunk_seconds: float = 0.25
    channels: int = 1

    # SOLA crossfade (Phase B). Disable at your peril - without it, audible
    # clicks at every chunk boundary when chunk_seconds is short.
    sola_enabled: bool = True
    # v0.12.3 - bumped 50.0 → 30.0 (low-latency tier winner of the 50-
    # condition sweep). v0.12.4 - REVERTED to 50.0 because the
    # high-latency-tier top-1 config (the user's listening-test winner)
    # uses 50.0 with chunk_seconds=0.25; at that chunk size, the
    # 30 ms crossfade was no longer optimal. See LESSONS §42.
    sola_crossfade_ms: float = 50.0  # overlap window between consecutive chunks
    # v0.6.9: widened from 4.0 to 6.0 so the search window covers at least one
    # full pitch period for typical voice f0 (>= 167 Hz period at 40 kHz model
    # rate = 240 samples = 6 ms). With a sub-period search, sustained vowels
    # produce phase mismatches SOLA can't reach - manifests as audible
    # dropouts during sustained voicing.
    # v0.12.3 picked 4.0 as the low-latency tier winner.
    # v0.12.4 - bumped 4.0 → 16.0 because the high-latency tier (top-1, the
    # user's listening-test winner) uses 16.0. At chunk_seconds=0.25 (250 ms
    # chunk-period vs 150 ms), a wider search range captures correctly-
    # aligned harmonic peaks that fall outside a 4 ms window - the
    # autocorrelation@chunk_period drops to 0.000, eliminating the
    # "train wagon" rhythm entirely. See LESSONS §42.
    sola_search_ms: float = 16.0  # how far to shift looking for in-phase alignment
    # History fed to the model alongside each new chunk so the embedder /
    # vocoder convolutions don't see edge artifacts. Brief calls this "context".
    # v0.12.4 - bumped 100.0 → 200.0 with chunk_seconds=0.25. The wider
    # context window gives SOLA's correlation search more overlap to
    # work with at the larger chunk size, which is what enables the
    # autocorrelation@chunk_period = 0.000 outcome the user picked from
    # the v0.12.3 sweep top-1 config. See LESSONS §42.
    sola_context_ms: float = 200.0
    # v0.6.9: lowered from sola.py's 0.25 default. With the original threshold,
    # SOLA falls back to centered (offset=0) on borderline cases and produces
    # phase-discontinuous crossfade for sustained content. 0.10 still rejects
    # decorrelated noise but keeps best-effort alignment for periodic signals.
    # v0.12.3 - bumped 0.10 → 0.30 after the 50-condition sweep (LESSONS §41).
    # Stricter rejection threshold: with v0.11.0 anti-jitter holding the
    # producer cadence steady, when SOLA's correlation search is below 0.30
    # the alignment is genuinely unreliable (transient, mostly-silent, or
    # mid-consonant) and falling back to centered (offset=0) introduces less
    # phase artifact than blindly accepting a low-confidence peak. v0.6.9's
    # 0.10 was tuned for noisier producer output.
    sola_corr_threshold: float = 0.30

    # RVC
    f0_up_key: int = 0  # semitones
    sid: int = 0
    # B59 / audio-008: RMVPE voiced-frame confidence threshold. Below this,
    # frames are treated as unvoiced (pitchf=0 → RVC NSF emits noise rather
    # than harmonic content). Smaller values catch more frames as voiced
    # (potential breath-as-pitch confusion); larger miss soft-voiced
    # content. Recommended range [0.1, 0.5]. Upstream's default is 0.3.
    threshold: float = 0.3

    # Embedder selection. Only "onnx" is supported (direct ORT contentvec-f.onnx
    # call). The fairseq PyTorch path was removed in v0.8.0 - it had no tests,
    # was opt-in only via the now-deleted [fairseq] extra, and the
    # `extract_features()[0]` indexing would have broken on fairseq API drift
    # (corr-002). The field stays in EngineConfig for backwards-compat with
    # existing config.toml files (any value other than "onnx" raises early).
    embedder: str = "onnx"

    # Routing
    sink_name: str = "WoysSink"
    input_device: str | int | None = None  # None = default mic
    # When False (default): output goes ONLY to WoysSink → woys-mic.
    # When True: ALSO write a best-effort copy to the host's default output
    # (laptop speakers / headphones) for self-monitoring.
    monitor: bool = False
    # Output latency in ms requested from the playback backend.
    # v0.7.0-rc3: 220 → 280. rc2's 220 still produced audible cuts in
    # real-world Telegram VoIP testing - confirming the rc2 retro point
    # that the synthetic harness over-counts cuts uniformly and can't
    # distinguish real-speech variance within its flat region. 280 ms
    # is the last rung in the rc ladder: 20 ms under the v0.6.x 300 ms
    # default that we already know is audibly clean. If 280 also fails,
    # the structural floor on this hardware is hit and further latency
    # reduction needs the ~80 ms engine threading tax (LESSONS §19)
    # closed first - that's v0.8.x territory, not another rc bump.
    # Wall-clock at rc3: chunk 150 + inference 80 + buffer 280 +
    # codec 30 ≈ 540 ms (vs v0.6.x ~660 ms, -18 %).
    output_latency_ms: int = 280
    # Process-time hint to pacat: write callbacks granulate to this many
    # ms. 20 ms keeps writes from coalescing into bursts that would
    # alternately starve and overrun the buffer. Ignored by pw-cat, which
    # uses PipeWire's quantum negotiation instead.
    output_process_time_ms: int = 20

    # v0.7.0-rc4 - flipped back to False. v0.7.0-rc1 reverted to pw-cat
    # on the reasoning that smaller chunks at chunk_seconds=0.15 would
    # eliminate the per-quantum stdin/PipeWire-callback race v0.6.7
    # documented (~43 ms zero-gaps on bursty writes). The
    # internal notes retro disagreed: rc1's "this won't
    # apply" is hand-wavy and doesn't address the race mechanism, and
    # the symptom we hear in Telegram (sample-exact zeros, voice-
    # correlated, ~40 ms quantized - see area 08) matches pw-cat's
    # documented per-quantum-gap pattern more closely than pacat's
    # underrun pattern. Migration cascade in `tui/config.py` pulls
    # users on the rc1+ default sentinel `True` forward to `False`;
    # users who explicitly set `prefer_pw_cat = true` after the field
    # is exposed in AppConfig keep their override.
    prefer_pw_cat: bool = False

    # v0.9.0 - when True, the engine spawns `bin/woys-pw-out` (native
    # PipeWire client) instead of pw-cat / pacat. The native helper
    # decouples the engine's bursty 150 ms chunk writes from PipeWire's
    # per-quantum (1024/48000 = 21.33 ms) RT callback via a lock-free
    # SPSC ring buffer. NEVER falls back silently if the helper is
    # missing - `_open_pacat` raises so the user sees the install gap
    # instead of mysterious cuts.
    #
    # v0.9.1 - default flipped to True. The v0.9.0-rc4 A/B established
    # that BOTH backends produce equivalent audible results on this
    # stack (engine-side writer jitter at ~80 ms is the dominant cut
    # source, downstream of any output backend). Native-pw still wins
    # on observability (honest per-quantum underrun counter, no
    # mid-session pacat-style respawns) and on architectural cleanliness
    # - flipping the default per the audit's "honest metric" rule.
    prefer_native_pw: bool = True

    # v0.9.2 - minimum ring-buffer slack (in milliseconds) the native
    # helper holds beyond the immediate chunk size. **Default reverted to
    # 0 in v0.9.2** after v0.9.1's 80 ms default proved both ineffective
    # against the audible cuts class AND introduced a ~170 ms echo
    # regression. See `CHANGELOG.md` v0.9.2 + LESSONS.md §28 for the
    # full retrospective; the short version is:
    #
    #   * `player_underruns` measures ring-empty events. The buffer
    #     expansion absorbed those events into the slack window and
    #     reduced the COUNTER, but the listener still heard the same
    #     class of micro-cuts because they're driven by engine writer
    #     jitter (~80 ms std-dev), not by ring underruns directly. A
    #     bigger ring just postpones the gap audibility - it doesn't
    #     fix the producer cadence that creates the gap in the first
    #     place.
    #   * The added latency (191 ms slack at default) pushed the
    #     round-trip past the threshold where Telegram echo cancellation
    #     copes, surfacing a new audible regression.
    #
    # The knob remains tunable for power users who want to trade
    # latency for fewer counter increments (e.g., on a CachyOS box
    # where 21 ms quantum is too tight). Default 0 keeps round-trip
    # at v0.9.0 levels and the counter honest.
    #
    #   buffer_ms   ring frames        ring ms     slack       use case
    #   ---------   -----------------  ---------   ---------   ----------
    #   0 (def)     8192 (chunk_only)  ~170 ms     ~21 ms      v0.9.0 baseline
    #   80          16384              ~341 ms     ~191 ms     latency-tolerant
    #   200         32768 (cap)        ~683 ms     ~533 ms     near-mute
    #
    # The helper's SPSC ring uses a power-of-2 mask so actual size is
    # `next_pow2(chunk_frames + buffer_ms x sink_rate / 1000)`. The
    # producer-side jitter fix lives in v0.10.x; this knob is observability,
    # not the cure.
    prefer_native_pw_buffer_ms: int = 0

    # v0.5.1: software input pre-attenuation, in dB. Default 0.0 (passthrough).
    # Hot mics (USB condenser mic at high volume etc.) clip the signal which
    # RVC amplifies as harsh distortion downstream. Setting a small
    # negative value (-3 to -6 dB) trims headroom without quieting much.
    # Applied per chunk before resample → embedder.
    input_gain_db: float = 0.0

    # v0.5.0 session-pool tuning.
    # Cap on simultaneous cached RVC sessions (each ~150 MiB VRAM).
    # B64 / perf-15: VRAM math on RTX 2070 (8 GiB) - pool_size=4 ≈ 600 MiB
    # of voice models, plus foundation models (700-1500 MiB depending on
    # rmvpe fp16 vs fp32), plus cuDNN handle / arena (~500 MiB). Combined
    # with CS2 wanting ~3-4 GiB, an 8 GiB GPU is tight under contention.
    # Lower this to 2 if you see CUDA OOM. Higher only on >12 GiB cards.
    session_pool_size: int = 4
    # If true, on engine.start() we eagerly create + cudnn-warm sessions for
    # every .onnx in the models dir (minus foundations). Adds ~6-12 s to
    # cold start for a 10-voice library, but every subsequent swap is a
    # pointer swap (~10 ms). Recommended for users with persistent engines.
    eager_warmup: bool = False

    # v0.5.2 - pacat underrun mitigations (see docs/08-pacat-underrun-bug.md).
    # Channels emitted by the engine. The PipeWire null-sink loaded by
    # `woys pw setup` defaults to 2 channels; emitting 2 here
    # avoids an in-graph 1→2 upmix on every chunk.
    output_channels: int = 2
    # Bounded queue between the engine main loop and the pacat writer
    # thread. Size 8 ≈ 2 s of slack at chunk_seconds=0.25; full-queue
    # events are exposed as `queue_full_events` (xrun proxy).
    pacat_writer_queue_size: int = 8
    # v0.6.7 part 3 - initial silence written to the playback backend
    # before any real engine output starts. Empirically didn't reduce
    # xruns in our trials (in fact slightly increased them - pacat seems
    # to apply its prebuf threshold to the silence and trip more
    # frequently). Default 0 (off). Kept as a tunable for users whose
    # backends (or future versions of pacat / pw-cat) might benefit.
    # See `docs/11-microcuts-bug.md` part 3.
    prime_silence_seconds: float = 0.0
    # v0.7.0-rc4 - gate threshold lowered -55 → -75. The audit
    # traced rc1/rc2/rc3
    # cuts to this gate firing on intra-speech RMS dips: -55 dBFS is
    # only ~6 dB below typical room noise on a USB condenser mic, and brief
    # speech valleys (between syllables, on plosive onsets, during
    # fricatives) routinely cross it. Each fire emits a full chunk
    # of zeros directly to the writer, bypassing SOLA, both
    # resamplers, and inference, with no counter incremented - which
    # is why three rcs of output_latency_ms tuning produced a flat
    # audible response. -75 dBFS is well below room ambient; combined
    # with the new hysteresis below, the gate only fires on sustained
    # silence rather than transient voice dips.
    #
    # v0.6.9 original rationale (preserved): when mic RMS is below
    # this floor, emit zeros directly instead of running RVC. Stops
    # the vocoder from hallucinating a ~-24 dBFS "voicing floor" on
    # near-silent input. Set to a very negative number (e.g. -200.0)
    # to disable entirely. See `docs/12-vad-misfire-investigation.md`.
    input_gate_dbfs: float = -75.0
    # v0.7.0-rc4 - hysteresis on the input gate. The gate must observe
    # `input_gate_hysteresis_ms` of continuously-below-threshold input
    # before it fires. Brief dips in voiced speech (typical: 30-150 ms
    # between syllables, on consonant onsets) no longer trigger
    # zero-emission, even if they momentarily cross threshold. Set to
    # 0 for the v0.6.9 behavior (immediate gating with no smoothing).
    # 200 ms is roughly the upper end of natural inter-syllable pause
    # in speech - anything beyond that is genuinely silence and the
    # vocoder-hallucination behavior the gate exists to prevent
    # actually appears.
    input_gate_hysteresis_ms: float = 200.0
    # Watchdog polls the pacat subprocess every N seconds; on death it
    # spawns a replacement and bumps `pacat_restarts`.
    pacat_watchdog_interval_s: float = 0.05
    # If set, pin the engine main thread + writer thread to this CPU core
    # (via os.sched_setaffinity). Reduces L2/L3 cache-miss jitter on the
    # i7-10750H. None = no pinning.
    cpu_affinity_core: int | None = None
    # v0.7.0-rc11 - engine thread runs SCHED_FIFO at priority 60 by
    # default. The rc10 dump showed inference p99 = 84 ms after
    # EXHAUSTIVE cuDNN trimmed shape-driven variance, but a 40 ms
    # p50 → p99 spread remains. The most likely remaining cause is
    # KDE / picom compositor preemption of the engine thread mid-
    # inference. SCHED_FIFO at priority 60 prevents user-space
    # preemption (KDE compositing, browser, etc. run at SCHED_OTHER
    # niced 0) while staying below typical PipeWire/ALSA threads
    # (priority 80-88) and well below kernel RT (98-99).
    #
    # Pre-rc11 this field was named the same and was opt-in (default
    # False) per Brief §6 - but the implementation only called
    # `os.nice(-10)`, which raises priority within SCHED_OTHER and
    # does NOT prevent preemption by another SCHED_OTHER task. rc11
    # rewrites `_apply_thread_priority` to actually call
    # `sched_setscheduler(SCHED_FIFO, 60)`, falling back to nice(-10)
    # then to a logged warning if RT is denied.
    #
    # Falls back cleanly: hosts without `RLIMIT_RTPRIO ≥ 60` (and
    # without CAP_SYS_NICE) get the old nice(-10) behavior. The
    # default `True` is safe - worst case is "no improvement" on
    # locked-down systems, never a hang or crash.
    realtime_priority: bool = True

    # v0.8.0 - run cv → rmvpe → rvc inference in a child process with
    # its own CUDA context. Closes the LESSONS §19 threading tax
    # (~23 ms typical-case overhead from running ORT inference in the
    # engine's daemon thread alongside writer / watchdog / stderr-
    # reader threads, all contending for the GIL during numpy ops
    # between ONNX sessions). Parent audio I/O thread no longer
    # competes; child process gets exclusive GIL + RT priority +
    # gc.disable() + cuDNN EXHAUSTIVE + broader pre-warm (rc7-rc12
    # wins, all preserved).
    #
    # IPC: shared memory for hot-path audio arrays (zero-copy via
    # numpy buffer protocol), Pipes for control + small metadata
    # (pickle overhead ~50-200 µs per call, < 1 % of inference time).
    #
    # v0.8.0-rc4 A/B confirmed multiprocessing is a null result on
    # quiet GPU (subprocess and in-process tied within noise). The
    # rc1 measured win was real but conditional on CS2 contesting
    # the GPU; without contention, in-process inference completes
    # fast enough that the GIL never blocks the writer thread for
    # long. v0.8.1 default flipped to False.
    #
    # Subprocess infrastructure stays as opt-in for users with
    # persistent GPU contention (e.g. CS2 + woys simultaneously) -
    # set `inference_subprocess=True` in `~/.config/woys/config.toml`
    # to spawn the inference child and isolate audio I/O from
    # GIL-bound inference.
    inference_subprocess: bool = False

    # When `inference_subprocess=True`, control whether the CHILD
    # process disables Python GC during its inference loop. Same
    # rc7 logic, just inside the subprocess. Set False if long
    # sessions reveal cyclic-ref memory bloat.
    inference_subprocess_disable_gc: bool = True

    # B63 / arch-012: opt-in periodic gc.collect(0) during the run loop.
    # gc.disable is on for the engine's lifetime, which can be hours; for
    # users who hit cyclic-ref memory bloat on long sessions, set this to
    # e.g. 1000 to run a gen-0 collect every N chunks (~150 s at
    # chunk_seconds=0.15). Cost: ~1-3 ms per collect - small enough to
    # be a non-event for the audio thread; large enough to cause an
    # observable jitter spike on tight chunk_seconds=0.10. Default 0
    # = off (current behavior).
    engine_periodic_gc_chunks: int = 0

    # v0.8.1 - TensorRT execution provider, DISABLED BY DEFAULT.
    #
    # The v0.8.1 TRT pivot was a dead end on this hardware/model
    # combination (ORT 1.22 + TRT 10.16 + RVC v2 + RMVPE):
    #
    #   - RMVPE FP16 STFT fails TRT init outright. TRT 10.16's STFT
    #     importer requires Float32 input; RMVPE has been auto-
    #     promoted to FP16 since v0.3.0. Per-session try/except
    #     catches this and falls back to CUDA EP for RMVPE - but
    #     that means RMVPE doesn't benefit from TRT at all.
    #
    #   - RVC initializes successfully but produces MATHEMATICALLY
    #     WRONG output. Cosine similarity vs CUDA EP across the 4
    #     soxr shapes: 0.02 / 0.44 / 0.48 / 0.28 (target ≥ 0.95).
    #     The Int64 binding warnings from TRT's parser are the
    #     observable symptom; the underlying issue is some
    #     combination of int64 indexing in the NSF source module
    #     and lack of shape inference annotations on the model.
    #
    #   - Speedup, ignoring correctness, is 1.04-1.87x on cv only.
    #     Below the 1.5-3x v0.8.1 target. With cos_sim broken on
    #     RVC, the win is hypothetical.
    #
    # Infrastructure stays in place for users who want to experiment
    # (set `use_tensorrt = true` in `~/.config/woys/config.toml`)
    # and as a path forward when ORT or the RVC export pipeline
    # gains TRT-friendly shape inference / int64 handling. For now,
    # the production default is CUDA EP only - same as rc12 baseline.
    #
    # Per-session TRT init status (success vs CUDA fallback) is
    # surfaced via `EngineStats.trt_active_for` and printed in
    # `woys diag` so the experimenting user sees exactly which
    # sessions take which path.
    use_tensorrt: bool = False

    # when False (default), `_make_session`
    # hard-fails (CpuFallbackError) if a CUDA EP is installed but ONNX
    # Runtime bound a model session CPU-only. Realtime RVC on CPU runs
    # ~10-50x over the chunk-period latency budget -- it is a non-functional
    # state, not a degraded one -- and the pre-fix silent fallback produced
    # a working-looking but unusable engine with no error surfaced anywhere.
    # Test-only seam: True lets the CPU test harnesses build real sessions
    # on a GPU-less box. woys is GPU-only, so this must never be plumbed to
    # config.toml, a profile or a CLI flag; EngineStats.cpu_fallback_active
    # (shown by `woys diag`) reports it if it is ever on.
    allow_cpu_fallback: bool = False

    # v0.10.0-rc3 - GPU keep-alive thread to mitigate dynamic-boost
    # variance.
    #
    # The v0.10.0-rc1/rc2 evidence (`docs/05-perf.md` v0.10.x table,
    # LESSONS §29) established the audible cuts on this stack come
    # from RVC inference tail variance (rvc.run p50=33 ms / p99=68 ms),
    # which correlates 1:1 with GPU clock-state oscillation: 34 % of
    # nvidia-smi clock samples sit > 100 MHz below the median during
    # the engine's bursty workload. The mic_read window (~98 ms / chunk)
    # is a long enough idle gap that the laptop GPU's dynamic boost
    # backs off, and the next chunk's RVC pays a reboost-recovery cost.
    #
    # When enabled, a daemon thread issues a tiny ORT op on the
    # contentvec session every `gpu_keepalive_interval_ms` ms.
    # The op is intentionally cheap (~1-3 ms of GPU work) and uses
    # an input shape pre-warmed at engine start so cuDNN doesn't
    # re-tune. The intent is "keep utilization above the deboost
    # threshold," not "do useful inference." Steady-state cost is
    # ~5-15 % continuous GPU duty cycle.
    #
    # Default off in rc3 - A/B testing planned. If the rc3 5-min run
    # shows writer_jitter p99 dropping toward the ≤ 30 ms gate, rc4
    # will flip the default to True. If the keepalive op QUEUES on
    # the same CUDA stream as engine inference and INCREASES rvc.run
    # p99 instead, we use a separate session in rc4.
    gpu_keepalive_enabled: bool = False
    gpu_keepalive_interval_ms: int = 25
    # Length (in 16 kHz samples) of the keepalive dummy input. 1600 = 100 ms
    # of audio = ~5 features at 50 Hz framerate; tunable down to 320 = 20 ms
    # if the chosen value over-loads the GPU stream. Pre-warmed at engine
    # start with the same EXHAUSTIVE cuDNN search the realtime shapes get,
    # so steady-state keepalive runs hit the cached path.
    gpu_keepalive_input_len: int = 1600

    # v0.11.0 - GPU clock lock + torch separate-stream keepalive.
    #
    # The v0.10.0-partial retrospective (LESSONS §29-§30) located the cuts
    # at NVIDIA dynamic-boost auto-deboost during the engine's mic_read
    # idle window. Two software fixes attack the layer without
    # firmware/hardware risk:
    #
    #   "clock_lock"  - calls `sudo nvidia-smi -lgc <floor>,<ceiling>`
    #                   at engine start and `-rgc` at engine stop. Forces
    #                   the GPU to stay at or above the configured floor
    #                   so no idle-time deboost. SIGTERM/SIGINT-safe.
    #                   Stock specs only; no overclock, no power-limit
    #                   change, no firmware. Sudoers entry needed
    #                   (see docs/22-gpu-clock-lock.md).
    #
    #   "keepalive"   - torch.cuda.Stream() based. Daemon thread issues a
    #                   tiny `tensor.add(1.0)` (~50 µs of GPU work) every
    #                   `gpu_keepalive_interval_ms`. Runs on a CUDA stream
    #                   separate from ORT's, so it doesn't queue against
    #                   engine inference (the rc3 contention-class). No
    #                   sudo. Replaces the ORT-stream keepalive entirely.
    #
    # The user-facing knob is `gpu_anti_jitter_mode`:
    #
    #   "off"        - neither (default; v0.10.0-partial behavior)
    #   "keepalive"  - torch keepalive only
    #   "clock_lock" - clock lock only (sudo)
    #   "both"       - clock lock + torch keepalive (sudo, max effect)
    #
    # The two underlying booleans (gpu_clock_lock_enabled,
    # gpu_keepalive_torch_stream) stay configurable for advanced users
    # but the mode field takes precedence when set to anything other
    # than "off".
    gpu_anti_jitter_mode: str = "off"

    # Lock floor in MHz. 0 means auto-detect from
    # `nvidia-smi --query-gpu=clocks.max.graphics` (returns the GPU's
    # absolute boost ceiling, then subtracts the
    # `gpu_clock_lock_floor_offset_mhz` margin to land on a value the GPU
    # actually sustains under load). On RTX 2070 Mobile this resolves
    # to ~1845 MHz floor with default offset 255. 0 sentinel (instead of
    # None) so the field round-trips through TOML cleanly.
    gpu_clock_lock_enabled: bool = False
    gpu_clock_lock_floor_mhz: int = 0
    gpu_clock_lock_ceiling_mhz: int = 0
    # When auto-detecting the floor, subtract this from
    # `clocks.max.graphics`. Empirical: max-255 lands on the highest
    # clock the GPU naturally sustained during v0.10.x harness runs
    # (RTX 2070 Mobile: max=2100 → floor=1845). Tunable for laptops with
    # different boost behavior.
    gpu_clock_lock_floor_offset_mhz: int = 255

    # Torch separate-stream keepalive (v0.11.0). Replaces the rc3
    # ORT-stream version (`gpu_keepalive_enabled`) - when both are
    # enabled, this one wins (the rc3 ORT-stream version remains
    # available as the no-torch fallback path). Tiny CUDA op
    # (1024-element float32 add) every `gpu_keepalive_torch_interval_ms`
    # on a torch.cuda.Stream() separate from ORT's.
    gpu_keepalive_torch_stream: bool = False
    gpu_keepalive_torch_interval_ms: int = 25
