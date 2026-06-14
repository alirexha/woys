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
import ctypes
import gc
import os
import queue
import shutil
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

# ORT-GPU 1.20+ on driver 595 needs explicit preload of the pip-shipped CUDA libs.
import onnxruntime as ort

NDArrayF32 = npt.NDArray[np.float32]
NDArrayI64 = npt.NDArray[np.int64]

if hasattr(ort, "preload_dlls"):
    ort.preload_dlls()


def _preload_trt_dlls() -> bool:
    """v0.8.1 - preload TensorRT shared libraries so ORT's TRT EP can
    resolve `libnvinfer.so.10`. The pip-installed `tensorrt-cu12`
    package puts its libs under
    `<venv>/lib/python3.11/site-packages/tensorrt_libs/` which isn't
    on the system loader path, so ORT (which dlopens
    `libonnxruntime_providers_tensorrt.so` and that in turn dlopens
    `libnvinfer.so.10`) fails with "cannot open shared object" unless
    we ctypes-preload the .so files into the process's symbol space
    first.

    Returns True if every libnvinfer*.so was loaded successfully,
    False otherwise (TRT EP will silently fall through to CUDA EP
    in that case - the per-session providers list always includes
    CUDA EP as a fallback).
    """
    import ctypes

    try:
        import tensorrt_libs
    except ImportError:
        return False

    libs_dir = os.path.dirname(tensorrt_libs.__file__)
    ok = True
    for fn in sorted(os.listdir(libs_dir)):
        # Only load the libnvinfer* shims; other files in tensorrt_libs
        # (Python source, init helpers) shouldn't be ctypes-loaded.
        if "libnvinfer" not in fn or ".so" not in fn:
            continue
        full = os.path.join(libs_dir, fn)
        try:
            ctypes.CDLL(full, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            ok = False
    return ok


# Best-effort TRT preload at module import. Failure is non-fatal -
# session creation falls back to CUDA EP if TRT can't be initialized.
_TRT_PRELOAD_OK = _preload_trt_dlls()


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
    # Set True only to deliberately run CPU-only (e.g. debugging on a
    # GPU-less box); EngineStats.cpu_fallback_active then reports it.
    # (Not yet plumbed to config.toml -- that is F-merged-008's job.)
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
    # (np.repeat, _to_pitch_coarse, astype, isnan/isinf scan). Populated
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
    # CPU-only. Only reachable when cfg.allow_cpu_fallback is set -- with
    # the default config `_make_session` raises CpuFallbackError instead.
    # Surfaced by `woys diag` so a deliberate CPU-only run is visible.
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
    # `pacat_restarts` to `player_restarts`. External callers (tests,
    # scripts that scrape EngineStats) reading the old name still work
    # for one release; v0.10 deletes the alias.
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
        `self._rvc.run` invocation: feats_2x = np.repeat, _to_pitch_coarse,
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


# v0.7.0-rc10: HEURISTIC → EXHAUSTIVE. The rc8 tail-chunk capture +
# rc9 broader pre-warm together pinned the inference p99 spike to
# cuDNN heuristic algo selection: even after rc9 pre-warmed every
# audio16_len soxr emits (1957/1958/2446/2447), p99 stayed at ~96 ms.
# rc9's tail log showed two distinct slow patterns - `rvc_ms` 64-72
# ms (one shape group) and `rvc_ms` 47-48 ms + `rmvpe_ms` 17 ms
# (other shape group). The heuristic was picking different,
# intrinsically slower, algos for the alternating shapes.
#
# v0.7.0-rc1's pre-rejection of EXHAUSTIVE was based on the autotune
# lump: a 50-100 ms one-time cost per first-encounter shape. At
# chunk_seconds=0.10 / 0.15, paying that cost mid-realtime made the
# first 5-10 chunks miss budget. rc9's broader pre-warm changes
# that calculation: the autotune lump is now paid during warmup
# (engine.start() before _run_loop), not realtime. Net startup
# cost: another ~0.5-1 s on top of rc9's already-extended warmup.
# Acceptable trade for letting cuDNN pick the FASTEST algo per
# shape rather than a heuristic guess.
#
# v0.2.0 - v0.7.0-rc9 history preserved for context:
#   v0.2.0 default - picks fastest steady-state algo per shape but
#   eats 50-100 ms autotune the first time each shape lands.
#   HEURISTIC (rc1+) picked a near-optimal algo from a heuristic
#   without any timed search - slightly slower steady-state but no
#   autotune lump.
#
# Setting can still be flipped back via the env var if EXHAUSTIVE
# regresses or if a future ORT release improves HEURISTIC.
_CUDNN_ALGO_SEARCH = "EXHAUSTIVE"


_TRT_CACHE_ROOT = Path.home() / ".cache" / "woys" / "trt"


def _trt_cache_dir_for(model_path: Path) -> Path:
    """Per-model TRT engine cache directory under
    `~/.cache/woys/trt/<model-stem>/`. Engines for different shapes
    of the same model land in the same directory, keyed by ORT's
    internal shape-aware hash. Different models keep separate
    subdirs so cache invalidation per-model is just `rm -rf`.
    """
    d = _TRT_CACHE_ROOT / model_path.stem
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cuda_provider_entry() -> tuple[str, dict[str, object]]:
    """The CUDA EP config we use everywhere - extracted so the TRT
    fallback path can pull the same options if TRT init fails."""
    return (
        "CUDAExecutionProvider",
        {
            "device_id": 0,
            # v0.7.0-rc12: kNextPowerOfTwo → kSameAsRequested.
            # See engine history below for full context.
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_algo_search": _CUDNN_ALGO_SEARCH,
            "do_copy_in_default_stream": True,
            # B54 / corr-023: bool, not the string "1". ORT's CUDA EP option
            # parser accepts both, but every other entry in this dict uses
            # native types - be consistent.
            "cudnn_conv_use_max_workspace": True,
        },
    )


# Module-level record of which sessions actually got TRT EP. Surfaced
# in `EngineStats.trt_active_for` so woys diag can show which models
# failed TRT init and fell back to CUDA - gives the user one place to
# see the real picture without grepping logs.
_TRT_ACTIVE_PER_SESSION: dict[str, bool] = {}
_TRT_INIT_ERRORS: dict[str, str] = {}

# cap on consecutive playback-helper respawns
# that never stay alive. The watchdog used to retry forever (running=True,
# zero audio). 8 consecutive deaths-without-recovery is unambiguously a
# broken binary/config, not a transient PipeWire hiccup (a one-off death
# respawns once and the counter resets on the next healthy tick).
_PLAYER_RESPAWN_CAP = 8

# parent-death signal for playback-helper
# children. Loaded once at import; None off non-glibc-Linux (then the
# preexec_fn below is a no-op).
try:
    _LIBC: ctypes.CDLL | None = ctypes.CDLL("libc.so.6", use_errno=True)
except OSError:  # pragma: no cover - non-Linux / no glibc
    _LIBC = None
_PR_SET_PDEATHSIG = 1  # <sys/prctl.h>


def _set_pdeathsig() -> None:
    """`preexec_fn` for playback-helper spawns: runs in the forked child,
    before exec, and asks the kernel to send SIGTERM to this process if
    its parent (the engine) dies.

    a `kill -9` of the engine used to orphan the
    playback subprocess, still holding its audio stream. The inference
    child already self-protects via a `getppid()` poll; the playback
    helpers did not. Must stay lock-free (one syscall, no imports, no
    allocation) -- `preexec_fn` runs between fork and exec in a
    multithreaded process.
    """
    if _LIBC is not None:
        # The bare try/except (vs `contextlib.suppress`) is deliberate:
        # `suppress` instantiates an object (a malloc); a bare try/except
        # is allocation-free, which matters in a preexec_fn running
        # between fork and exec.
        try:  # noqa: SIM105
            _LIBC.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM)
        except Exception:
            pass


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


class CpuFallbackError(RuntimeError):
    """ONNX Runtime bound a session CPU-only
    while a CUDA execution provider was available.

    Realtime RVC on CPU runs ~10-50x over the chunk-period latency budget,
    so a CPU-bound session is a non-functional state, not a degraded one.
    Pre-fix `_make_session` appended `CPUExecutionProvider` as an
    unconditional landing pad and never checked `get_providers()`, so a
    broken onnxruntime-gpu wheel / NVIDIA driver / missing `preload_dlls()`
    produced a working-looking but unusable engine with no error anywhere.
    """


def _session_is_cpu_only(sess: ort.InferenceSession) -> bool:
    """True iff the session's active (first) provider is CPUExecutionProvider."""
    bound = sess.get_providers()
    return bool(bound) and bound[0] == "CPUExecutionProvider"


def _assert_session_gpu_bound(
    sess: ort.InferenceSession,
    path: Path,
    *,
    available: list[str],
    allow_cpu_fallback: bool,
) -> None:
    """Hard-fail the silent CUDA->CPU fallback (F-merged-001).

    If a CUDA EP is installed in this ORT build but the session bound
    CPU-only, raise `CpuFallbackError` -- unless `allow_cpu_fallback` is
    set, in which case the CPU binding is left in place (the engine records
    it in `EngineStats.cpu_fallback_active` for `woys diag`).

    When no CUDA EP is present in the build at all, CPU is simply the only
    option -- that is the environment, not a silent *fallback*, so it is
    left alone here; the no-GPU condition is surfaced separately by
    `woys info` (F-merged-013).
    """
    if not _session_is_cpu_only(sess):
        return
    if "CUDAExecutionProvider" not in available:
        return
    if not allow_cpu_fallback:
        raise CpuFallbackError(
            f"{path.name}: a CUDA execution provider is installed but ONNX "
            f"Runtime bound this session CPU-only (providers="
            f"{sess.get_providers()}). Realtime RVC is unusable on CPU. "
            f"Check the onnxruntime-gpu wheel, the NVIDIA driver, and that "
            f"ort.preload_dlls() ran. To deliberately run CPU-only, set "
            f"EngineConfig.allow_cpu_fallback = True."
        )


def _make_session(
    path: Path, *, use_tensorrt: bool = True, allow_cpu_fallback: bool = False
) -> ort.InferenceSession:
    """v0.8.1 - try TensorRT EP first, fall back to CUDA EP per session.

    ORT's TRT EP fails session initialization (not just the TRT
    subgraph) when it encounters operators it can't handle -
    e.g. RMVPE's FP16 STFT, which TRT requires to be FP32. We
    catch that failure and rebuild the session with CUDA EP only.
    The fallback is logged to `_TRT_INIT_ERRORS[path.name]` and
    can be surfaced via `EngineStats.trt_active_for` and woys diag.

    the CUDA->CPU fallback is *not* silent.
    After the session is built, `_assert_session_gpu_bound` raises
    `CpuFallbackError` if a CUDA EP was available but ORT bound CPU-only,
    unless `allow_cpu_fallback` is set.
    """
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    available = ort.get_available_providers()

    if use_tensorrt and _TRT_PRELOAD_OK and "TensorrtExecutionProvider" in available:
        cache_dir = _trt_cache_dir_for(path)
        trt_providers: list[tuple[str, dict[str, object]] | str] = [
            (
                "TensorrtExecutionProvider",
                {
                    "device_id": 0,
                    # Cache engines to disk so the 5-30 s per-shape
                    # compile cost is paid only on the first session
                    # ever (or when the model file changes).
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": str(cache_dir),
                    # 2 GiB workspace for TRT's builder. RTX 2070 has
                    # 8 GiB; cv + rmvpe + rvc resident is ~2 GiB.
                    "trt_max_workspace_size": 2 * 1024 * 1024 * 1024,
                    # FP16 disabled until per-voice quality validation
                    # (cosine sim ≥ 0.95 vs FP32 baseline) is run.
                    "trt_fp16_enable": False,
                    "trt_max_partition_iterations": 1000,
                    "trt_min_subgraph_size": 1,
                    "trt_timing_cache_enable": True,
                    "trt_timing_cache_path": str(cache_dir),
                },
            ),
            _cuda_provider_entry(),
            "CPUExecutionProvider",
        ]
        try:
            sess = ort.InferenceSession(str(path), sess_options=so, providers=trt_providers)
            _assert_session_gpu_bound(
                sess, path, available=available, allow_cpu_fallback=allow_cpu_fallback
            )
            _TRT_ACTIVE_PER_SESSION[path.name] = True
            _TRT_INIT_ERRORS.pop(path.name, None)
            return sess
        except CpuFallbackError:
            # A CpuFallbackError means TRT/CUDA both failed to bind and the
            # session is CPU-only -- that is the F-merged-001 hard-fail, not
            # a "TRT couldn't partition the graph" retry condition. Re-raise;
            # rebuilding with the CUDA-only providers below would just bind
            # CPU again.
            raise
        except Exception as e:
            # TRT couldn't parse / partition the graph. Log the
            # reason and retry with CUDA EP only. Common cause:
            # graph contains an operator TRT doesn't support
            # (FP16 STFT, certain custom ops, dynamic shapes
            # without shape inference annotations).
            _TRT_ACTIVE_PER_SESSION[path.name] = False
            _TRT_INIT_ERRORS[path.name] = f"{type(e).__name__}: {str(e)[:240]}"

    # CUDA EP only path (TRT disabled or TRT init failed).
    cuda_providers: list[tuple[str, dict[str, object]] | str] = []
    if "CUDAExecutionProvider" in available:
        cuda_providers.append(_cuda_provider_entry())
    cuda_providers.append("CPUExecutionProvider")
    sess = ort.InferenceSession(str(path), sess_options=so, providers=cuda_providers)
    _assert_session_gpu_bound(
        sess, path, available=available, allow_cpu_fallback=allow_cpu_fallback
    )
    _TRT_ACTIVE_PER_SESSION.setdefault(path.name, False)
    return sess


# v0.6.9 - pitchf sanitization for the realtime inference path.
# Frames with NaN or f0 <= 0 are treated as "unvoiced" by the RVC vocoder's
# NSF source module; a single such frame mid-utterance zeros the harmonic
# source and produces an audible dropout. We replace NaN with 0 first
# (defensive against extractor bugs), then linearly interpolate runs of
# unvoiced frames up to `_VOICED_GAP_MAX_FRAMES` long between two voiced
# frames. Long unvoiced runs are left as zeros so true silence still
# decodes as silence. See `docs/12-vad-misfire-investigation.md`.
_VOICED_GAP_MAX_FRAMES = 8  # ~80 ms at the RMVPE 100 fps frame rate


def interpolate_voiced_gaps_np(
    pitchf: NDArrayF32,
    *,
    prior_voiced_f0: float = 0.0,
    prior_voiced_age_frames: int = -1,
) -> NDArrayF32:
    """B16 / perf-002: vectorized version. The pre-v0.8.0 implementation
    walked an inner `for k in range(i, j)` Python loop that ran ~50-200
    iterations per chunk under typical RMVPE pitch tracks. numpy slicing
    replaces the loop with a single broadcast multiply per gap.

    Also keeps the dtype path in float32 throughout (pre-v0.8.0 cast to
    float64 for the linspace arithmetic, then back to float32) - minor
    alloc churn reduction for B16's perf-001 partial.

    optional ``prior_voiced_f0`` +
    ``prior_voiced_age_frames`` carry the last-voiced anchor from the
    PREVIOUS chunk so a chunk-leading unvoiced run can still be bridged
    when its in-window `last_valid` is -1. The semantics match the
    in-window path: if the leading run length plus the prior age stays
    within ``_VOICED_GAP_MAX_FRAMES`` and a trailing voiced frame exists
    within this chunk, we synthesize a virtual anchor at index
    ``-prior_voiced_age_frames`` (negative, conceptually "outside the
    chunk to the left") and interpolate from `prior_voiced_f0` through
    the run to the trailing in-window anchor. Defaults ``(0.0, -1)``
    reproduce the pre-F-31-12 fall-back behaviour (no leading-edge
    bridge) -- no caller is forced to thread state through. Engine
    streaming path passes carry state from
    `EngineWorker._pitch_carry_*`.

    bridging interpolates in
    **log-f0** (geometric mean), not in Hz. A glide that is
    perceptually linear (constant semitones / second) is linear in
    log-frequency, not in Hz; bridging 100 Hz → 400 Hz in Hz puts
    the midpoint at 250 Hz, while a perceptually-straight glide
    crosses 200 Hz at midpoint (= sqrt(100 * 400)). Bridging over
    ≤ ``_VOICED_GAP_MAX_FRAMES`` (8 frames ≈ 80 ms) frames in Hz
    produces a sub-perceptual sag in the contour audible on
    voiced->unvoiced->voiced syllable boundaries. One ``log/exp``
    pair per gap; cost is negligible (≤ 8 frame-samples per gap).
    Pitch shift downstream of this function is a constant
    multiplicative factor in Hz, i.e. a constant additive offset
    in log-f0, so the shape of the log-linear bridge is preserved
    by the shift (verified in
    ``test_pitch_shift_modifies_pitchf_and_pitch_coarse_consistently``).
    """
    if pitchf.size == 0:
        return pitchf
    invalid = np.isnan(pitchf) | (pitchf <= 0.0)
    if not invalid.any():
        return pitchf
    if (~invalid).sum() == 0:
        # Whole chunk is unvoiced - preserve so vocoder produces silence.
        return np.nan_to_num(pitchf, nan=0.0).astype(np.float32, copy=False)
    out = np.nan_to_num(pitchf, nan=0.0).astype(np.float32, copy=True)
    n = len(invalid)
    have_prior = (
        prior_voiced_f0 > 0.0
        and prior_voiced_age_frames >= 0
        and prior_voiced_age_frames < _VOICED_GAP_MAX_FRAMES
    )
    # Walk the runs of invalid; bridge each ≤ _VOICED_GAP_MAX_FRAMES gap
    # via vectorized log-linear interpolation between the bracketing
    # voiced frames (F-31-03). Pre-F-31-03 this was linear-in-Hz.
    last_valid = -1
    i = 0
    while i < n:
        if not invalid[i]:
            last_valid = i
            i += 1
            continue
        j = i
        while j < n and invalid[j]:
            j += 1
        run_len = j - i
        if (
            run_len <= _VOICED_GAP_MAX_FRAMES
            and last_valid >= 0
            and j < n
            and out[last_valid] > 0.0
            and out[j] > 0.0
        ):
            # F-31-03: log-linear bridge. alpha vector over the gap,
            # interpolate in log space then exp back to Hz.
            alphas = (np.arange(i, j, dtype=np.float32) - last_valid) / (j - last_valid)
            log_lo = float(np.log(out[last_valid]))
            log_hi = float(np.log(out[j]))
            out[i:j] = np.exp(log_lo * (1.0 - alphas) + log_hi * alphas).astype(np.float32)
        elif (
            # F-31-12 leading-edge bridge: no in-window prior anchor, but
            # the previous chunk left a recent voiced f0 we can use.
            run_len <= _VOICED_GAP_MAX_FRAMES
            and last_valid < 0
            and j < n
            and have_prior
            and out[j] > 0.0
            and prior_voiced_age_frames + run_len <= _VOICED_GAP_MAX_FRAMES
        ):
            # Synthetic anchor at index -prior_voiced_age_frames - 1
            # (one full frame "older" than i==0). Total span from the
            # virtual anchor to j is (prior_voiced_age_frames + 1 + j).
            # F-31-03: log-linear here too.
            virt_anchor_offset = -(prior_voiced_age_frames + 1)
            span = float(j - virt_anchor_offset)
            alphas = (np.arange(i, j, dtype=np.float32) - virt_anchor_offset) / span
            log_lo = float(np.log(prior_voiced_f0))
            log_hi = float(np.log(out[j]))
            out[i:j] = np.exp(log_lo * (1.0 - alphas) + log_hi * alphas).astype(np.float32)
        i = j
    return out


# Backwards-compat alias for the rare external caller. New code uses the
# public name. (B23: encapsulation cleanup; old-style _ prefix retained.)
_interpolate_voiced_gaps_np = interpolate_voiced_gaps_np


def to_pitch_coarse(pitchf: NDArrayF32, target_len: int) -> tuple[NDArrayI64, NDArrayF32]:
    """B24 / quality-020: now a public name (drop leading underscore) so the
    smoke test can `from audio.engine import to_pitch_coarse` instead of
    re-implementing the algorithm. Single source of truth.

    B56 / perf-003: early-exit on all-zero pitchf - the engine's input gate
    fully zeroes audio during sub-hysteresis transitions (engine.py:2184)
    and the resulting RMVPE output is all-zero. Skipping the four numpy
    passes (log, mask multiply, clip, rint) saves ~8 µs per such chunk.

    v0.14.0 (area 7 / C093): clamp negative pitchf at entry. RMVPE in
    practice emits non-negative Hz, but transients / NaN-replaced regions
    can leak negatives. log(1 + pitch/700) at pitch < -700 produces NaN;
    NaN survives the `mask > 0` filter (NaN > 0 is False so the cell is
    untouched), then `clip(NaN, 1, 255)` returns NaN, then
    `rint().astype(int64)` becomes INT64_MIN, which RVC's harmonic-source
    table reads as out-of-bounds garbage. Clamping at entry makes the
    contract explicit and prevents the silent failure mode.
    """
    if pitchf.size == 0:
        return (
            np.zeros(target_len, dtype=np.int64),
            np.zeros(target_len, dtype=np.float32),
        )
    if pitchf.min() < 0.0:
        pitchf = np.clip(pitchf, 0.0, None)
    if float(pitchf.max()) == 0.0:
        return (
            np.zeros(target_len, dtype=np.int64),
            np.zeros(target_len, dtype=np.float32),
        )
    f0_min, f0_max = 50.0, 1100.0
    f0_mel_min = 1127.0 * np.log(1 + f0_min / 700.0)
    f0_mel_max = 1127.0 * np.log(1 + f0_max / 700.0)
    pitch = np.zeros(target_len, dtype=np.float32)
    n = min(len(pitchf), target_len)
    # when pitchf is over-length keep the *last*
    # n frames, not the first. RMVPE and contentvec emit unequal frame
    # counts, so `pitchf[:n]` temporally scrambled the F0 contour against
    # the content features -- frame i of the harmonic source corresponded
    # to a different point in time than frame i of the content. Upstream
    # keeps the trailing frames (Pipeline.py:288, `pitch[:, -feats_len:]`).
    # (When pitchf is not over-length, pitchf[-n:] == pitchf[:n].)
    pitch[-n:] = pitchf[-n:]
    f0_mel = 1127.0 * np.log(1 + pitch / 700.0)
    mask = f0_mel > 0
    f0_mel[mask] = (f0_mel[mask] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
    f0_mel = np.clip(f0_mel, 1.0, 255.0)
    return np.rint(f0_mel).astype(np.int64), pitch


# Backwards-compat alias.
_to_pitch_coarse = to_pitch_coarse


# B60 / audio-012: `_resample_linear` (the known-bad reference baseline)
# was deleted in v0.8.0. Production path used `_resample` (soxr); the linear
# variant existed only to fail v0.5.1 quality tests. No callers in src/ or
# tests/. If you need it back as a benchmark, see `scripts/bench_*.py` or
# git history.


def _resample(audio: NDArrayF32, src_rate: int, dst_rate: int) -> NDArrayF32:
    """High-quality stateless resampler - used for one-shot tests + tail flushes.

    The realtime engine path uses `_StreamResampler` instead so the
    anti-aliasing filter state survives across chunks. See
    `docs/11-microcuts-bug.md` for why per-chunk stateless resampling
    leaks a 4 Hz envelope artifact.

    Cost: ~0.5 ms for a 100 ms chunk on this CPU.
    """
    if src_rate == dst_rate:
        return audio.astype(np.float32, copy=False)
    if audio.size == 0:
        return audio.astype(np.float32, copy=False)
    import soxr  # type: ignore[import-untyped]

    out = soxr.resample(audio, src_rate, dst_rate, quality="HQ")
    return np.asarray(out, dtype=np.float32)


class _StreamResampler:
    """Stateful soxr resampler - preserves filter state across chunks.

    Per-call `soxr.resample(...)` resets the anti-aliasing filter every
    invocation; concatenating the resampled chunks introduces a brief
    filter-transient amplitude dip at every chunk boundary, audible as
    a 4 Hz flutter on sustained content (`docs/11-microcuts-bug.md`).
    `soxr.ResampleStream` carries the filter buffer across calls and
    eliminates the per-chunk warm-up.

    Identity case (`src_rate == dst_rate`) is a passthrough - no soxr
    object created.

    -- ``cold_fade_in_samples``. The
    *steady-state* per-chunk warmup is eliminated by carrying filter
    state, but a freshly-constructed `_StreamResampler` still cold-
    starts on its first emit: the soxr filter delay line begins zero-
    filled, so the first ~few-ms of output has a sub-unity gain.
    Engine.run_loop tolerates this on engine startup (silence before
    the first chunk anyway) but the model-swap path in
    `_apply_one_swap` builds a NEW _StreamResampler when the post-swap
    voice's native rate differs from the previous one -- the cold-
    start blip lands inside live, voiced audio. We mask the transient
    by applying a linear fade-in across the first `cold_fade_in_samples`
    of *output*; the swap path passes ~5 ms of dst_rate samples so the
    listener hears a brief amplitude ramp instead of a hard step.
    Default 0 means "no fade-in" -- the engine-startup constructor
    leaves it zero because the engine emits silence on startup anyway.
    """

    def __init__(
        self,
        src_rate: int,
        dst_rate: int,
        *,
        quality: str = "HQ",
        cold_fade_in_samples: int = 0,
    ) -> None:
        self.src_rate = src_rate
        self.dst_rate = dst_rate
        # F-31-11: remaining cold-fade-in budget. Decremented per emitted
        # sample; once exhausted the resampler is fully primed and
        # subsequent calls bypass the fade-in path entirely.
        self._cold_fade_remaining = int(max(0, cold_fade_in_samples))
        self._cold_fade_total = self._cold_fade_remaining
        if src_rate == dst_rate:
            self._stream = None
            return
        import soxr

        self._stream = soxr.ResampleStream(src_rate, dst_rate, num_channels=1, quality=quality)

    def _apply_cold_fade(self, out: NDArrayF32) -> NDArrayF32:
        """F-31-11: scale the leading samples of `out` by a linear ramp
        that completes the budgeted fade-in. Mutates `out` in place
        (a fresh array each call from soxr) and decrements the
        remaining budget. Cheap path when remaining is 0.
        """
        if self._cold_fade_remaining <= 0 or out.size == 0:
            return out
        total = self._cold_fade_total
        consumed = total - self._cold_fade_remaining
        n_fade_this_chunk = min(self._cold_fade_remaining, out.size)
        if n_fade_this_chunk > 0:
            # Ramp from consumed/total → (consumed + n_fade_this_chunk)/total
            start = consumed / total
            end = (consumed + n_fade_this_chunk) / total
            ramp = np.linspace(start, end, n_fade_this_chunk, endpoint=False, dtype=np.float32)
            out[:n_fade_this_chunk] *= ramp
        self._cold_fade_remaining -= n_fade_this_chunk
        return out

    def process(self, audio: NDArrayF32) -> NDArrayF32:
        """Consume `audio` (1-D float32 mono); return whatever soxr emits
        for this chunk. Output length will lag input length slightly while
        the internal buffer fills - flush() drains the rest."""
        if self._stream is None:
            # Identity path: still honour the cold-fade-in budget so the
            # F-31-11 contract holds even when no rate change is needed.
            out = audio.astype(np.float32, copy=True)
            return self._apply_cold_fade(out)
        if audio.size == 0:
            return np.zeros(0, dtype=np.float32)
        out = self._stream.resample_chunk(audio, last=False)
        out = np.asarray(out, dtype=np.float32).reshape(-1)
        return self._apply_cold_fade(out)

    def flush(self) -> NDArrayF32:
        """Drain any audio held in soxr's internal buffer. Call once before
        discarding (engine stop / model swap when output rate changes)."""
        if self._stream is None:
            return np.zeros(0, dtype=np.float32)
        out = self._stream.resample_chunk(np.zeros(0, dtype=np.float32), last=True)
        out = np.asarray(out, dtype=np.float32).reshape(-1)
        return self._apply_cold_fade(out)


class RvcSessionPool:
    """Per-path cache of `ort.InferenceSession` objects.

    Hot-swap performance was the v0.4.x P0: every `models use` rebuilt the
    session from scratch, including cudnn EXHAUSTIVE algo-tuning, costing
    ~1.5 s + a 305 ms first-chunk inference burst. This pool keeps a small
    set of cached sessions; second swap to an already-seen voice is a
    pointer swap (~10 ms total).

    LRU eviction keeps VRAM bounded - a session uses ~150 MiB resident,
    so the default `max_size=4` caps voice-model VRAM at ~600 MiB on top
    of the foundations. Configurable via `EngineConfig.session_pool_size`.

    Thread-safe. The audio worker calls `get_or_create()` from inside
    `_maybe_swap_model`; tests / TUI may call it from any thread.
    """

    def __init__(
        self,
        max_size: int = 4,
        *,
        use_tensorrt: bool = True,
        allow_cpu_fallback: bool = False,
    ) -> None:
        self._cache: dict[Path, ort.InferenceSession] = {}
        self._access_order: list[Path] = []
        self._max_size = max(1, max_size)
        self._lock = threading.Lock()
        # v0.8.1: pool sessions inherit the engine's TRT preference. The
        # engine constructs the pool with `use_tensorrt=cfg.use_tensorrt`
        # so RVC voice loads share whatever EP path the engine wants.
        self._use_tensorrt = use_tensorrt
        # RVC voice sessions go through the
        # same CUDA->CPU silent-fallback hard-fail as cv/rmvpe.
        self._allow_cpu_fallback = allow_cpu_fallback

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    def __contains__(self, path: Path) -> bool:
        with self._lock:
            return Path(path).resolve() in self._cache

    def get_or_create(self, path: Path) -> ort.InferenceSession:
        """Return a cached session if present, else create + cache.

        Cache hit: ~0.1 ms. Cache miss: ~600 ms (model load + cudnn tune).
        """
        key = Path(path).resolve()
        with self._lock:
            if key in self._cache:
                # Bump LRU.
                self._access_order.remove(key)
                self._access_order.append(key)
                return self._cache[key]

        # Cache miss - build outside the lock (slow); other threads can
        # still get cached sessions while we tune.
        sess = _make_session(
            key,
            use_tensorrt=self._use_tensorrt,
            allow_cpu_fallback=self._allow_cpu_fallback,
        )

        with self._lock:
            # Another thread may have raced us; if so, drop ours and use theirs.
            if key in self._cache:
                return self._cache[key]
            self._cache[key] = sess
            self._access_order.append(key)
            # B26 / corr-006: the pre-v0.8.0 `if evicted != key` guard was
            # dead - `key` was just appended at -1, the pop comes from index
            # 0, so evicted == key only when len == 1 (and then we DO want
            # to evict, never reaching this branch). Just unconditionally
            # drop.
            while len(self._access_order) > self._max_size:
                self._cache.pop(self._access_order.pop(0), None)
        return sess

    def warmup(self, path: Path) -> ort.InferenceSession:
        """Create + run one dummy forward pass so cudnn populates its algo
        cache. Subsequent inferences against the same shape are near-instant.

        The caller is expected to know the model's input shape - we feed the
        widest plausible RVC v2 input (768-dim feats x 100 frames).
        """
        sess = self.get_or_create(path)
        try:
            shape = sess.get_inputs()[0].shape
            feats_dim = int(shape[2]) if len(shape) >= 3 and isinstance(shape[2], int) else 768
        except (IndexError, ValueError):
            feats_dim = 768
        is_half = sess.get_inputs()[0].type != "tensor(float)"
        feats_dt = np.float16 if is_half else np.float32
        n_frames = 100
        feed = {
            "feats": np.zeros((1, n_frames, feats_dim), dtype=feats_dt),
            "p_len": np.array([n_frames], dtype=np.int64),
            "pitch": np.zeros((1, n_frames), dtype=np.int64),
            "pitchf": np.zeros((1, n_frames), dtype=np.float32),
            "sid": np.array([0], dtype=np.int64),
        }
        with contextlib.suppress(Exception):
            sess.run(["audio"], feed)
        return sess

    def warmup_all(self, paths: list[Path]) -> None:
        """Warm a batch of models. Costs ~600 ms per model. Useful at engine
        startup when `eager_warmup` is enabled."""
        for p in paths:
            self.warmup(p)

    def evict_all(self) -> None:
        with self._lock:
            self._cache.clear()
            self._access_order.clear()


class RealtimeEngine:
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
        # bound CPU-only. Only reachable when cfg.allow_cpu_fallback is set --
        # otherwise `_make_session` raises CpuFallbackError above. Surfaced
        # by `woys diag` so a deliberate CPU-only run is visible.
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
            if self._stopped:
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
        for req in requests:
            self._apply_one_swap(req)

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
                self.record_error(
                    f"subprocess swap failed: {e}. Stopping engine - "
                    f"flip `inference_subprocess=false` to fall back to "
                    f"in-process inference."
                )
                req.error = e
                self._resolve_swap(req)
                self._stop_event.set()
                return

        # Legacy in-process path. Existing _cv (contentvec) and _rmvpe
        # stay - they're foundation models, not voice-specific.
        # v0.5.0: pool-cached. Cache hit ≈ 10 ms; cache miss ≈ 600 ms.
        self.cfg.rvc_model = target
        self._rvc = self._rvc_pool.get_or_create(target)
        self._is_half = self._rvc.get_inputs()[0].type != "tensor(float)"
        new_sr = self._cached_rvc_sr(target)
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

    # ---- inference ----------------------------------------------------------

    def _extract_feats(self, audio16k: NDArrayF32) -> NDArrayF32:
        """Embedder dispatch (always ONNX contentvec since v0.8.0)."""
        assert self._cv is not None
        # Cast input to whatever dtype this contentvec ONNX expects (fp16 or fp32).
        in_dtype = np.float16 if "float16" in self._cv_input_dtype else np.float32
        audio_in: np.ndarray = audio16k.reshape(1, -1).astype(in_dtype)  # type: ignore[type-arg]
        feats_raw = self._cv.run(["unit12"], {"audio": audio_in})[0]
        # Always return float32 to the rest of the pipeline.
        feats: NDArrayF32 = feats_raw.astype(np.float32, copy=False)
        return feats

    def process_chunk_16k(self, audio16k: NDArrayF32) -> NDArrayF32:
        """One inference pass on a (N,) float32 chunk at 16 kHz.

        Standalone path - used by tests and by the engine when SOLA is
        disabled. Doesn't touch streaming state. The streaming engine path
        goes through `_process_streaming_16k` instead.
        """
        return self._infer(audio16k)

    def _infer(self, audio16k: NDArrayF32, *, update_pitch_carry: bool = False) -> NDArrayF32:
        """Raw model invocation; no streaming bookkeeping.

        v0.8.0 - when `cfg.inference_subprocess=True` AND the
        `_inf_client` was successfully started, this delegates to the
        child process via `InferenceClient.infer()`. The child runs
        the full cv → rmvpe → rvc pipeline in its own CUDA context;
        per-stage timings come back via the response and populate
        `EngineStats` so woys diag shows the same breakdown.

        On `InferenceError` (child died, pipe broke), we attempt one
        restart-and-retry. If the retry fails too, the exception
        propagates up to `_safe_process_streaming_16k` which catches
        it and bumps `dropped_chunks` like any other inference
        failure.

        when ``update_pitch_carry=True``
        AND we're on the legacy in-process path, read
        `self._pitch_carry_*` to provide a leading-edge anchor to
        `_interpolate_voiced_gaps_np`, and update the carry from the
        trailing voiced frame of this chunk's pitchf. The subprocess
        path does not carry pitch state across the IPC boundary -- the
        leading-edge dropout case (a brief unvoiced run starting a
        chunk where the prior chunk ended voiced) is preserved there.
        Documented limitation; the IPC protocol does not currently
        ship the prior/posterior pitch-carry tuple.
        """
        if self._inf_client is not None:
            from audio.inference_client import InferenceError

            try:
                ipc_result, timings = self._inf_client.infer(
                    audio16k,
                    f0_up_key=self.cfg.f0_up_key,
                    sid=self.cfg.sid,
                    threshold=self.cfg.threshold,
                )
            except InferenceError as e:
                # Try one restart. If THAT fails, propagate.
                self.record_error(f"inference child died ({e}); attempting restart")
                try:
                    self._inf_client.restart()
                    self.stats.child_restarts = self._inf_client.restart_count
                    self.stats.child_pid = (
                        self._inf_client._handles.proc.pid if self._inf_client._handles else None
                    )
                    ipc_result, timings = self._inf_client.infer(
                        audio16k,
                        f0_up_key=self.cfg.f0_up_key,
                        sid=self.cfg.sid,
                        threshold=self.cfg.threshold,
                    )
                except InferenceError:
                    raise

            # Populate stats from child's per-stage timings + the
            # cumulative NaN-replace count. Use the child's running
            # total so cumulative numbers survive a child restart.
            self.stats.last_cv_ms = timings.cv_ms
            self.stats.last_rmvpe_ms = timings.rmvpe_ms
            self.stats.last_rvc_ms = timings.rvc_ms
            self.stats.last_ipc_roundtrip_ms = timings.roundtrip_ms
            # cluster the
            # rolling-window appends under `_stats_lock` so a TUI /
            # diag reader cannot raise on `deque mutated during
            # iteration`.
            with self._stats_lock:
                self.stats._recent_ipc_roundtrip_ms.append(timings.roundtrip_ms)
                self.stats._recent_cv_ms.append(timings.cv_ms)
                self.stats._recent_rmvpe_ms.append(timings.rmvpe_ms)
                self.stats._recent_rvc_ms.append(timings.rvc_ms)
            self.stats.unique_audio16_lens.add(int(audio16k.shape[-1]))
            self.stats.nan_chunks = timings.nan_chunks_total
            ipc_typed: NDArrayF32 = ipc_result
            return ipc_typed

        # Legacy in-process path.
        assert self._cv is not None and self._rmvpe is not None and self._rvc is not None

        # -- documented omission of upstream's
        # `silence_front` lead-in trim. Upstream RVC's `Pipeline.py:254/272/306`
        # tracks `silence_front` (how many leading samples are known-silent)
        # and trims that region from the contentvec features + RMVPE pitchf
        # before inference and index search:
        #     npyOffset = math.floor(silence_front * 16000) // 360
        #     feats = feats[:, npyOffset * 2 :, :]
        # woys deliberately omits this for two reasons:
        #   (1) the input gate (`engine.py` `_safe_process_streaming_16k`) zeros
        #       sub-hysteresis chunks entirely, so silence-only chunks never hit
        #       this path -- the trim is a no-op gain for the most common case;
        #   (2) faiss index retrieval (F-31-01) is currently not implemented, so
        #       the "silent frames pollute the nearest-neighbour search" risk
        #       upstream's trim was protecting against is currently null.
        # The cost we pay: when speech leads with a partially-silent history
        # window, contentvec + RMVPE still run on the silent prefix (small
        # latency tax, no quality impact). If F-31-01 ever lands, this comment
        # is the marker to revisit -- index search WILL be polluted by silent
        # frames at that point. this is a documented design
        # choice, not an unaudited omission.
        t_cv0 = time.perf_counter()
        feats = self._extract_feats(audio16k)
        # v0.6.9: silently zero NaN bursts in feats before they propagate
        # through the inferencer and become NaN samples in the output.
        if np.isnan(feats).any():
            feats = np.nan_to_num(feats, nan=0.0)
        t_cv1 = time.perf_counter()
        rm_dtype = np.float16 if "float16" in self._rmvpe_input_dtype else np.float32
        pitchf_raw = self._rmvpe.run(
            ["pitchf"],
            {
                "waveform": audio16k.reshape(1, -1).astype(rm_dtype),
                "threshold": np.array([self.cfg.threshold], dtype=rm_dtype),
            },
        )[0]
        pitchf = pitchf_raw.astype(np.float32).squeeze()
        # v0.6.9: sanitize + interpolate short voiced→voiced gaps so a transient
        # RMVPE failure mid-utterance doesn't zero the NSF harmonic source.
        # Live diagnostic on e_girl voice traced 8 of 14 dropouts to this path.
        # pass cross-chunk pitch carry so a
        # leading-edge unvoiced run can be bridged using the prior chunk's
        # trailing voiced anchor. Streaming wrapper sets `update_pitch_carry`.
        if update_pitch_carry:
            pitchf = _interpolate_voiced_gaps_np(
                pitchf,
                prior_voiced_f0=self._pitch_carry_f0,
                prior_voiced_age_frames=self._pitch_carry_age_frames,
            )
            # Update carry from the trailing voiced frame of THIS pitchf.
            # `age_frames` measured at the END of this pitchf so the next
            # call's interpretation is conservative-recency (the slight
            # under-aging vs the overlap window is harmless because the
            # next call's in-window `last_valid` catches the same frame
            # if it falls in the history portion -- the carry only fires
            # when it's genuinely outside).
            voiced_mask = pitchf > 0.0
            if bool(voiced_mask.any()):
                last_voiced_idx = int(np.flatnonzero(voiced_mask)[-1])
                self._pitch_carry_f0 = float(pitchf[last_voiced_idx])
                self._pitch_carry_age_frames = (len(pitchf) - 1) - last_voiced_idx
            elif self._pitch_carry_age_frames >= 0:
                # No voiced this chunk; the carry ages by the full pitchf
                # length. Once age >= _VOICED_GAP_MAX_FRAMES the predicate
                # `have_prior` in interpolate_voiced_gaps_np rejects it.
                self._pitch_carry_age_frames += len(pitchf)
        else:
            pitchf = _interpolate_voiced_gaps_np(pitchf)
        t_rmvpe1 = time.perf_counter()

        # v0.10.0-rc2 - split RVC stage into pre / run / post so the
        # tail-attribution data tells us GPU work vs Python overhead.
        feats_2x = np.repeat(feats, 2, axis=1)
        # v0.14.0 (area 4 / area 7 / C001): apply pitch shift in semitones
        # BEFORE deriving pitch_coarse. Upstream's RMVPEOnnxPitchExtractor
        # (src/server/voice_changer/RVC/embedder/RMVPEOnnxPitchExtractor.py)
        # shifts f0 first, then derives BOTH pitch_coarse (mel-bin index)
        # and pitchf (Hz vector) from the shifted result. The pre-v0.14.0
        # engine path multiplied pitchf_aligned AFTER coarse was derived,
        # so RVC saw mismatched harmonic-source vs pitch-class-embedding
        # pairs for any non-zero f0_up_key. Hard to detect aurally because
        # RVC blends the embedding into the residual; cleanest test is
        # A/B'ing pitch shifts against the upstream reference path.
        if self.cfg.f0_up_key != 0:
            pitchf = pitchf * (2.0 ** (self.cfg.f0_up_key / 12.0))
        pitch_coarse, pitchf_aligned = _to_pitch_coarse(pitchf, target_len=feats_2x.shape[1])
        pitch_coarse = pitch_coarse[: feats_2x.shape[1]].reshape(1, -1)
        pitchf_aligned = pitchf_aligned[: feats_2x.shape[1]].reshape(1, -1).astype(np.float32)

        feats_dtype = np.float16 if self._is_half else np.float32
        # Build the input dict (final astype on feats happens here too).
        rvc_inputs = {
            "feats": feats_2x.astype(feats_dtype),
            "p_len": np.array([feats_2x.shape[1]], dtype=np.int64),
            "pitch": pitch_coarse,
            "pitchf": pitchf_aligned,
            "sid": np.array([self.cfg.sid], dtype=np.int64),
        }
        t_rvc_pre1 = time.perf_counter()
        out = self._rvc.run(["audio"], rvc_inputs)[0]
        t_rvc_run1 = time.perf_counter()
        result = np.array(out).astype(np.float32).squeeze()
        # v0.6.9: belt-and-braces NaN sanitize. pacat is fed float32le; NaN
        # would be undefined behavior in PipeWire's mixer chain and the
        # listener hears it as a click + brief gap.
        # B57 / audio-010: posinf=0.0 / neginf=0.0 (not ±1.0). nan_to_num is
        # element-wise - only the rare bad samples are zeroed, not the whole
        # chunk. The pre-v0.8.0 ±1.0 produced full-scale impulses (audible
        # click) on inf samples; zero is a single-sample dropout (~21 µs at
        # 48 kHz), audibly less harsh.
        if np.isnan(result).any() or np.isinf(result).any():
            result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
            # v0.7.0-rc4 - count NaN-sanitize hits so we can attribute
            # voice-correlated cuts to the vocoder rather than the gate
            # or SOLA. Pre-rc4 the path silently zeroed samples; area 06
            # of the audit flagged this as one of three NaN-zero paths
            # that incremented no counter.
            with self._stats_lock:
                self.stats.nan_chunks += 1
        t_rvc1 = time.perf_counter()
        # Per-stage timing surfaces in the slow_chunk_log when a chunk goes late.
        cv_ms = (t_cv1 - t_cv0) * 1000.0
        rmvpe_ms = (t_rmvpe1 - t_cv1) * 1000.0
        rvc_ms = (t_rvc1 - t_rmvpe1) * 1000.0
        # v0.10.0-rc2 - RVC sub-stages.
        rvc_pre_ms = (t_rvc_pre1 - t_rmvpe1) * 1000.0
        rvc_run_ms = (t_rvc_run1 - t_rvc_pre1) * 1000.0
        rvc_post_ms = (t_rvc1 - t_rvc_run1) * 1000.0
        self.stats.last_cv_ms = cv_ms
        self.stats.last_rmvpe_ms = rmvpe_ms
        self.stats.last_rvc_ms = rvc_ms
        # v0.10.0 - per-stage rolling windows for percentile attribution.
        # The pre-v0.10.0 path tracked only `_recent_inference` (sum); the
        # writer-jitter investigation needs to know which stage owns the
        # tail.
        # cluster lock.
        with self._stats_lock:
            self.stats._recent_cv_ms.append(cv_ms)
            self.stats._recent_rmvpe_ms.append(rmvpe_ms)
            self.stats._recent_rvc_ms.append(rvc_ms)
            self.stats._recent_rvc_pre_ms.append(rvc_pre_ms)
            self.stats._recent_rvc_run_ms.append(rvc_run_ms)
            self.stats._recent_rvc_post_ms.append(rvc_post_ms)
        self.stats.unique_audio16_lens.add(int(audio16k.shape[-1]))
        result_typed: NDArrayF32 = result
        return result_typed

    def _safe_process_streaming_16k(self, audio16: NDArrayF32) -> NDArrayF32 | None:
        """v0.6.8 - wrap `_process_streaming_16k` so a transient
        ORT / CUDA / numerical error drops the chunk instead of killing
        the engine.

        Returns the inferred chunk, or `None` if inference failed.
        Caller must check for `None` and skip the playback write - the
        engine's main loop does this with `continue`.

        First three failures log to `stats.last_error` with the
        exception type + message. After that the counter increments
        silently except for every 100th hit (to keep the diagnostic
        line refreshing without spamming).

        Pulled out of `_run_loop` so the failure path is unit-testable
        without spinning up the full audio thread.
        """
        try:
            result = self._process_streaming_16k(audio16)
            self._consecutive_drops = 0
            return result
        except Exception as e:
            with self._stats_lock:
                self.stats.dropped_chunks += 1
            self._consecutive_drops += 1
            n = self.stats.dropped_chunks
            if n <= 3:
                self.record_error(f"inference dropped chunk #{n}: {type(e).__name__}: {e}")
            elif n % 100 == 0:
                self.record_error(
                    f"inference still dropping chunks (total #{n}): {type(e).__name__}: {e}"
                )
            # B14 / corr-015: circuit breaker on sustained inference failure.
            # Voice changer feeding Discord - "stopped" is better than
            # "silently giving them silence." Threshold 50 ≈ 7-12 seconds
            # at chunk_seconds in [0.15, 0.25]; long enough to ride out a
            # transient cuDNN tune but short enough to surface a genuine
            # broken state.
            if self._consecutive_drops >= 50 and not self._stop_event.is_set():
                self.record_error(
                    f"engine stopping: {n} consecutive inference failures. "
                    f"Last: {type(e).__name__}: {e}"
                )
                self._stop_event.set()
            return None

    def _process_streaming_16k(self, new_chunk_16k: NDArrayF32) -> NDArrayF32:
        """Streaming variant. Maintains a sliding input history so the model
        sees overlapping content; SOLA crossfades consecutive outputs.

        Returns audio at 16 kHz that's safe to concatenate with the previous
        emitted chunk (assuming the same SOLA stream). Output length is
        approximately equal to the input length once warmed up.
        """
        # Input-side sizing always uses the 16 kHz config (mic input rate).
        cf = self._sola_input_cfg.crossfade_samples
        ctx = self._sola_input_cfg.context_samples
        history_len = ctx + cf

        # Build model input: last (ctx + cf) of input history + the new chunk.
        model_input = np.concatenate([self._input_history, new_chunk_16k.astype(np.float32)])
        # Update history for next call: keep the last (ctx + cf) samples of
        # the combined buffer (these will be the leading samples next time).
        self._input_history = model_input[-history_len:].copy()

        # the streaming path carries
        # pitch state across `_infer` calls so a leading-edge unvoiced
        # run that straddles a chunk boundary can still be bridged.
        full_out = self._infer(model_input, update_pitch_carry=True)

        # Map the trim from input space to output space proportionally -
        # the model is roughly 1:1 in time, but RVC trims a few samples at
        # the boundaries. Compute the per-sample ratio defensively.
        in_len = model_input.shape[0]
        out_len = full_out.shape[0]
        ratio = out_len / max(in_len, 1)
        # Drop the leading "context" portion in the model output. Keep the
        # last `ctx_drop_out` samples - sized to match SOLA's contract:
        #   chunk_n + cf + search   when SOLA is enabled (rc5 - gives the
        #                           alignment search positional slack so
        #                           emit length stays constant)
        #   chunk_n + cf            when SOLA is disabled (legacy path -
        #                           emit length variable, no slack needed)
        # The pre-rc5 implementation always trimmed to chunk_n + cf even
        # with SOLA enabled; the search then ate samples from the input
        # to find alignment, shrinking the emit. See
        # internal notes for why that was wrong.
        sola_search = self._sola_input_cfg.search_samples if self._sola is not None else 0
        ctx_drop_in = max(history_len - cf - sola_search, 0)
        ctx_drop_out = round(ctx_drop_in * ratio)
        emitted_region = full_out[ctx_drop_out:]

        if self._sola is not None:
            return self._sola.process(emitted_region)
        # SOLA disabled - emit raw, expect chunk-boundary clicks for short chunks.
        return emitted_region

    def reset_streaming_state(self) -> None:
        """Clear SOLA + input history so the engine can resume cleanly after a
        stop / start without leaking stale tail from a previous session."""
        self._input_history = np.zeros(
            self._sola_input_cfg.context_samples + self._sola_input_cfg.crossfade_samples,
            dtype=np.float32,
        )
        if self._sola is not None:
            self._sola.reset()
        # the pitch carry must drop
        # too -- the next session's first chunk is logically the start
        # of a new utterance, no prior voiced anchor is in scope.
        self._pitch_carry_f0 = 0.0
        self._pitch_carry_age_frames = -1

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
            self._stop_event.clear()
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
        except (FileNotFoundError, RuntimeError, OSError) as e:
            # PipeWireError is RuntimeError so it falls into this
            # bucket. We name it in the docstring above for the
            # benefit of grep / IDE.
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

    def _warmup_realtime_pipeline(self, n_chunks_per_shape: int = 4) -> None:
        """v0.6.9 - pre-run synthetic chunks through the *full* realtime
        pipeline (cv → rmvpe → rvc) so cuDNN's algo cache is populated for
        the actual shapes we feed at runtime. `RvcSessionPool.warmup` only
        warms the rvc session; the cv and rmvpe sessions still cold-start
        the first few real chunks otherwise.

        v0.7.0-rc9 - extended to pre-warm EVERY unique input length
        soxr's stream resampler can emit, not just the nominal one. The
        rc8 tail-chunk capture pinned the inference p99=96 ms / max=110 ms
        spike to a shape mismatch: pre-rc9 warmup ran `_infer` with
        `chunk_n = chunk_seconds * 16000 = 2400` samples. The realtime
        path calls `_infer` with `model_input.shape[0] = history_len +
        audio16_len`, where `audio16_len` is whatever
        `_StreamResampler(48k → 16k).process(7200)` emits. Soxr
        alternates between two specific values (1957 / 2447 in a sample 48 kHz USB-condenser session, plus the typical 2400) - every chunk with
        a non-cached shape costs cuDNN a fallback slow path, ~80 ms
        inference vs ~40 ms cached. See
        internal notes and the rc8
        tail_chunk_log dump.

        rc9 fix: drive a probe `_StreamResampler` with synthetic 48k
        input, capture every unique `audio16_len` it emits, and pre-warm
        `_infer` with `history_len + audio16_len` for each. Probe is
        independent of the real `_resampler_in` (which doesn't exist
        until `_run_loop` builds it anyway), so its filter state can't
        leak into realtime.
        """
        if self._cv is None or self._rmvpe is None or self._rvc is None:
            return
        chunk_n_mic = round(self.cfg.chunk_seconds * self.cfg.mic_rate)
        if chunk_n_mic <= 0:
            return

        rng = np.random.default_rng(42)

        # Step 1: probe soxr to enumerate the realtime shape set. Run
        # ~20 chunks so the resampler's polyphase filter settles and we
        # see every steady-state emit length (the alternation pattern in
        # the rc8 dump cycles every ~10 chunks).
        unique_audio16_lens: set[int] = set()
        if self.cfg.mic_rate != 16_000:
            probe = _StreamResampler(self.cfg.mic_rate, 16_000)
            for _ in range(20):
                dummy_48k = rng.standard_normal(chunk_n_mic).astype(np.float32) * 0.001
                out_chunk = probe.process(dummy_48k)
                if out_chunk.size > 0:
                    unique_audio16_lens.add(int(out_chunk.shape[0]))
        else:
            # mic_rate == internal - no resample, audio16_len is fixed.
            unique_audio16_lens.add(chunk_n_mic)

        if not unique_audio16_lens:
            # Fall back to the pre-rc9 single-shape behavior so a
            # surprising probe failure doesn't skip warmup entirely.
            unique_audio16_lens.add(round(self.cfg.chunk_seconds * 16_000))

        # Step 2: pre-warm `_infer` with `history_len + audio16_len`
        # for each unique shape. Matches the realtime concat at
        # `_process_streaming_16k`. Multiple iterations per shape so
        # cuDNN's heuristic cache settles to a stable algo choice.
        history_len = self._sola_input_cfg.context_samples + self._sola_input_cfg.crossfade_samples
        for audio16_len in sorted(unique_audio16_lens):
            model_input_len = history_len + audio16_len
            if model_input_len <= 0:
                continue
            dummy = rng.standard_normal(model_input_len).astype(np.float32) * 0.001
            for _ in range(n_chunks_per_shape):
                try:
                    self._infer(dummy)
                except Exception:
                    # If one shape fails, try the rest - the realtime
                    # path's `_safe_process_streaming_16k` will catch
                    # any inference failure that survives warmup.
                    break

        # v0.10.0 - snapshot the model-input shape set seen during warmup
        # (populated by `_infer` instrumentation as `audio16k.shape[-1]`,
        # which equals `history_len + audio16_len` for every warmup call).
        # Runtime continues adding to `unique_audio16_lens`; the diff
        # surfaces shapes that hit cuDNN cold during the realtime session.
        # Reset the rolling per-stage deques so warmup chunks don't
        # pollute realtime percentile reads.
        self.stats.warmup_audio16_lens = set(self.stats.unique_audio16_lens)
        self.stats._recent_cv_ms.clear()
        self.stats._recent_rmvpe_ms.clear()
        self.stats._recent_rvc_ms.clear()
        self.stats._recent_inference.clear()
        self.stats._recent_total.clear()

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
            self._stop_event.set()
            # resolve every outstanding swap
            # waiter BEFORE we start the slow teardown. Pre-fix
            # `_swap_done` was never set in `stop()`, so a JobRegistry
            # daemon thread parked the full 10 s timeout on the
            # "queue a swap, toggle off" sequence. With per-call
            # events, we walk the outstanding list and resolve each
            # with an "engine stopped" error.
            with self._swap_lock:
                pending = list(self._outstanding_swaps)
                self._outstanding_swaps.clear()
                # Also drain any queued requests that the worker
                # never got to.
                while True:
                    try:
                        pending.append(self._swap_queue.get_nowait())
                    except queue.Empty:
                        break
            for req in pending:
                if not req.completion.is_set():
                    req.error = RuntimeError("engine stopped before swap completed")
                    req.completion.set()
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

    def _warn_if_default_sink_hijacked(self) -> None:
        """one-shot check at engine start.

        If the system default sink is the woys sink the engine writes into,
        all *desktop* audio is being routed into woys plumbing instead of
        the speakers (the v0.14.2 hijack). That is a system-routing problem,
        not an engine fault -- the engine still converts voice fine -- so
        this WARNS (records `stats.last_error`, surfaced by `woys diag` and
        the TUI) rather than refusing to start.

        Best-effort: `get_default_sink()` returns '' on any pactl error, so
        a probe failure is a silent no-op here -- the load-bearing sink
        check is `_assert_sink_loaded`, which hard-fails.
        """
        from audio.pipewire import get_default_sink

        default_sink = get_default_sink()
        if default_sink and default_sink == self.cfg.sink_name:
            self.record_error(
                f"system default sink is {default_sink!r} -- the woys sink the "
                f"engine writes into. Desktop audio is being routed into woys "
                f"plumbing; run `pactl set-default-sink <your-speakers>` to fix."
            )

    def _assert_sink_loaded(self) -> None:
        """v0.6.4 - refuse to start if `cfg.sink_name` isn't a loaded
        PipeWire sink.

        Without this guard, `pw-cat --target=...` and `pacat --device=...`
        treat the named sink as a hint: if it's missing, the session
        manager silently routes the stream to the *default* sink
        (typically laptop speakers). The engine's playback subprocess
        starts cleanly, exits 0, no stderr - and your transformed
        voice plays out of the speakers instead of the virtual mic.
        See docs/10-monitor-leak-diag.md for the full forensic trail.

        v0.14.0 (area 9 / area 17 / area 19 / C014): pre-v0.14.0 the
        function silently skipped the check on three error paths
        (FileNotFoundError -> pactl missing; TimeoutExpired -> daemon
        slow; nonzero rc -> pactl itself errored). Skipping re-opens
        the v0.6.4 routing-to-laptop-speakers bug exactly when the
        environment is most likely to be misconfigured. v0.14.0 hard-
        fails on each: clearer than letting the engine start and serve
        silence to the wrong sink. If a user really has no pactl, they
        should set `sink_name=""` (skip-sink-check mode -- not yet
        implemented; flagged for v0.14.x as an explicit opt-out).
        """
        try:
            result = subprocess.run(
                ["pactl", "list", "short", "sinks"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                "pactl is not on PATH; cannot verify PipeWire sink presence. "
                "Install pipewire-pulse (provides pactl) or use a system with "
                "PulseAudio compatibility -- woys requires pactl for sink-load "
                "verification. Refusing to start rather than risk routing audio "
                "to the wrong sink."
            ) from e
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                "pactl `list short sinks` timed out after 5s. PipeWire daemon "
                "may be unresponsive (mid-restart, hung). Refusing to start "
                "rather than skip the sink-load check. Try `systemctl --user "
                "restart pipewire pipewire-pulse wireplumber` and re-run."
            ) from e
        if result.returncode != 0:
            raise RuntimeError(
                f"pactl `list short sinks` exited {result.returncode}. "
                f"stderr: {result.stderr.strip()[:200] or '<empty>'}. "
                f"Refusing to start without verified sink presence."
            )
        loaded = [line.split("\t")[1] for line in result.stdout.splitlines() if "\t" in line]
        if self.cfg.sink_name not in loaded:
            raise RuntimeError(
                f"PipeWire sink {self.cfg.sink_name!r} is not loaded - refusing to start.\n"
                f"  loaded sinks: {loaded}\n"
                f"  fix: run `woys pw setup` to load the virtual sink, "
                f"or correct `sink_name` in ~/.config/woys/config.toml."
            )

    def _find_native_pw_helper(self) -> Path | None:
        """Locate `woys-pw-out` (the native PipeWire helper introduced in
        v0.9.0). Search order:
          1. $PATH (via `shutil.which`)
          2. <repo>/bin/woys-pw-out (dev checkout, makes `make install`
             optional)
          3. ~/.local/bin/woys-pw-out (default install prefix)

        Returns None if none of those resolve.
        """
        # 1. PATH.
        path_hit = shutil.which("woys-pw-out")
        if path_hit:
            return Path(path_hit)
        # 2. Repo's bin/.
        repo_root = Path(__file__).resolve().parent.parent.parent
        repo_hit = repo_root / "bin" / "woys-pw-out"
        if repo_hit.exists() and os.access(repo_hit, os.X_OK):
            return repo_hit
        # 3. Default install prefix.
        local_hit = Path.home() / ".local" / "bin" / "woys-pw-out"
        if local_hit.exists() and os.access(local_hit, os.X_OK):
            return local_hit
        return None

    def _spawn_checked(self, cmd: list[str]) -> subprocess.Popen[bytes]:
        """Spawn a playback helper and verify it did not die on startup.

        `_open_pacat` used to `return
        subprocess.Popen(...)` with no liveness check. A helper that exits
        immediately (bad args, missing perms, a built-but-broken
        `woys-pw-out`) left the watchdog in an infinite 0.5 s-backoff
        respawn loop with `running=True` and `chunks_processed` climbing
        while zero audio reached the sink. We now sleep briefly and
        `poll()`; an immediate exit raises with the helper's stderr so the
        initial spawn fails the engine loudly and the watchdog's
        consecutive-failure cap can act.

        `preexec_fn=_set_pdeathsig` arms the
        kernel parent-death signal, so a `kill -9` of the engine SIGTERMs
        the playback helper instead of orphaning it with the audio stream
        still open. One change covers all three spawn paths (native-pw /
        pw-cat / pacat) -- they all funnel through here.
        """
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            preexec_fn=_set_pdeathsig,  # lock-free single syscall -- see _set_pdeathsig
        )
        time.sleep(0.05)
        rc = proc.poll()
        if rc is not None:
            stderr = ""
            if proc.stderr is not None:
                with contextlib.suppress(Exception):
                    stderr = proc.stderr.read().decode("utf-8", "replace").strip()
            raise RuntimeError(
                f"playback helper {cmd[0]!r} exited immediately on spawn (rc={rc})"
                + (f": {stderr[:300]}" if stderr else "")
            )
        return proc

    def _open_pacat(self) -> subprocess.Popen[bytes]:
        """Spawn the playback subprocess targeting the named virtual sink.

        v0.5.2: prefers `pw-cat` (PipeWire-native, no underruns under
        bursty 250 ms writes) over `pacat` (PulseAudio compat, drains the
        prebuf/tlength buffer near zero on every chunk → underrun storm).
        Falls back to pacat only if pw-cat is missing.

        v0.6.4: pre-flights sink existence - see `_assert_sink_loaded`.
        Without that guard, `--target` / `--device` silently fall back
        to the default sink when the named sink is missing.

        v0.9.0: `cfg.prefer_native_pw` (default True since v0.9.2 — the
        default flipped after the native helper became the daily-driver
        path) selects the native helper `woys-pw-out`
        (see `bin/woys-pw-out.c`) over pw-cat / pacat. The native helper
        decouples the engine's bursty
        chunk writes from PipeWire's per-quantum RT callback via an
        explicit SPSC ring buffer, closing the audit's area-08 cut
        signature (sample-exact zeros at 21.33/42.67 ms quantum
        cadence). NEVER falls back silently - if `prefer_native_pw=True`
        and the helper is missing, we raise so the user sees an actionable
        error instead of cuts they can't explain.

        The retained name `_open_pacat` is historical - the watchdog and
        writer threads don't care which binary is on the other side, only
        that it accepts raw float32le on stdin.
        """
        self._assert_sink_loaded()
        if self.cfg.prefer_native_pw:
            helper = self._find_native_pw_helper()
            if helper is None:
                raise RuntimeError(
                    "prefer_native_pw=True but `woys-pw-out` was not found. "
                    "Build it with `make -C bin/` from the repo root, then "
                    "either symlink it onto $PATH or run "
                    "`make -C bin/ install` to drop it into ~/.local/bin/. "
                    "Set prefer_native_pw=false in your config to fall back "
                    "to the legacy pw-cat / pacat path."
                )
            self._player_backend = "native-pw"
            # v0.9.1: compute ring frames from prefer_native_pw_buffer_ms.
            # Helper requires power-of-2 ring size (SPSC mask trick), so
            # round up. Need:
            #   chunk_frames + slack_frames
            # where chunk_frames is one engine write (chunk_seconds *
            # sink_rate) and slack_frames absorbs writer-jitter overshoot.
            chunk_frames = int(self.cfg.chunk_seconds * self.cfg.sink_rate)
            slack_frames = int(self.cfg.prefer_native_pw_buffer_ms * self.cfg.sink_rate / 1000)
            needed = chunk_frames + slack_frames
            ring_frames = 1
            while ring_frames < needed:
                ring_frames <<= 1
            # Helper caps ring at 32768 internally as a sanity limit; cap
            # here too with a clear error rather than letting the helper
            # reject the arg later.
            if ring_frames > 32768:
                raise RuntimeError(
                    f"prefer_native_pw_buffer_ms={self.cfg.prefer_native_pw_buffer_ms} "
                    f"computes ring_frames={ring_frames}, above the helper's 32768 cap. "
                    f"Lower the buffer or accept that no realistic engine jitter "
                    f"requires more than 32768/{self.cfg.sink_rate} ≈ "
                    f"{32768 / self.cfg.sink_rate * 1000:.0f} ms."
                )
            cmd = [
                str(helper),
                f"--target={self.cfg.sink_name}",
                f"--rate={self.cfg.sink_rate}",
                f"--channels={self.cfg.output_channels}",
                "--quantum=1024",
                f"--ring-frames={ring_frames}",
            ]
            return self._spawn_checked(cmd)

        if self.cfg.prefer_pw_cat:
            pw_cat = shutil.which("pw-cat")
            if pw_cat is not None:
                self._player_backend = "pw-cat"
                cmd = [
                    pw_cat,
                    "--playback",
                    f"--target={self.cfg.sink_name}",
                    f"--rate={self.cfg.sink_rate}",
                    f"--channels={self.cfg.output_channels}",
                    "--format=f32",
                    "--raw",
                    f"--latency={self.cfg.output_latency_ms}ms",
                    "-",
                ]
                return self._spawn_checked(cmd)

        pacat = shutil.which("pacat")
        if pacat is None:
            raise RuntimeError(
                "neither pw-cat nor pacat found - install pipewire and pipewire-pulse"
            )
        self._player_backend = "pacat"
        cmd = [
            pacat,
            "--playback",
            f"--device={self.cfg.sink_name}",
            f"--rate={self.cfg.sink_rate}",
            f"--channels={self.cfg.output_channels}",
            "--format=float32le",
            f"--latency-msec={self.cfg.output_latency_ms}",
            f"--process-time-msec={self.cfg.output_process_time_ms}",
            "--client-name=woys",
            "--stream-name=engine-out",
            "--raw",
            "-v",
        ]
        return self._spawn_checked(cmd)

    # ---- v0.5.2 writer / watchdog / stderr-reader plumbing ------------------

    def _to_sink_bytes(self, mono: NDArrayF32) -> bytes:
        """Convert a mono float32 chunk at sink_rate into the byte payload
        pacat expects on stdin. With output_channels=2, interleave L=R=mono
        so PipeWire doesn't have to upmix on every chunk.
        """
        if self.cfg.output_channels == 1:
            return mono.tobytes()
        # Stereo: interleave mono into [L0, R0, L1, R1, ...]. np.repeat is
        # the cheapest path: ~50 µs for 250 ms of 48 kHz audio on this CPU,
        # well below the chunk budget.
        stereo = np.repeat(mono.astype(np.float32, copy=False), self.cfg.output_channels)
        return stereo.tobytes()

    def _enqueue_chunk(self, payload: bytes) -> None:
        """Hand a write-ready byte payload to the writer thread. On a full
        queue the engine has out-paced the writer/sink - bump the
        queue_full counter (xrun proxy) and drop the chunk rather than
        block the engine main loop.
        """
        q = self._writer_queue
        if q is None:
            return
        try:
            q.put_nowait(payload)
        except queue.Full:
            with self._stats_lock:
                self.stats.queue_full_events += 1

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
        """
        import sounddevice as sd

        # `sd.OutputStream` does not have published type stubs; use Any
        # so the .start/.stop/.write/.close calls don't need per-line
        # ignores.
        stream: Any = None
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
            with contextlib.suppress(Exception):
                stream.write(chunk.reshape(-1, 1))
        # Stop event set -- tear down.
        if stream is not None:
            with contextlib.suppress(Exception):
                stream.stop()
                stream.close()

    def _writer_loop(self) -> None:
        """Daemon thread: drains _writer_queue into pacat.stdin.

        Decouples the engine main loop from blocking pipe writes (Brief §3
        Fix 2). On BrokenPipeError / OSError the watchdog is signalled to
        respawn pacat; the writer keeps running and reattaches to the new
        handle on the next iteration.
        """
        # Best-effort thread-local affinity so the writer doesn't ping-pong
        # cores away from the main engine thread.
        # B19 / perf-009: writer at priority 59 (engine at 60). Both stay
        # SCHED_FIFO so SCHED_OTHER background work can't starve either,
        # but the engine wins same-class tie-breaks during contention.
        self._apply_thread_priority(label="writer", priority=59)
        while not self._stop_event.is_set():
            try:
                payload = self._writer_queue.get(timeout=0.1) if self._writer_queue else None
            except queue.Empty:
                continue
            if payload is None:
                continue
            with self._pacat_lock:
                proc = self._pacat_proc
            if proc is None or proc.stdin is None:
                # Watchdog hasn't (re)spawned pacat yet; drop and continue.
                continue
            try:
                proc.stdin.write(payload)
                proc.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                # v0.9.0-rc5: distinguish shutdown-race BrokenPipe from
                # a real mid-session helper death. During engine.stop(),
                # the playback subprocess is terminated as part of the
                # finally-block; the writer thread may still have queued
                # bytes and races the helper's exit. That race is normal
                # teardown noise, not a runtime error worth surfacing.
                if self._stop_event.is_set():
                    return
                self.record_error(
                    f"{self._player_backend or 'player'} write failed "
                    f"({type(e).__name__}); respawning"
                )
                self._pacat_dead_event.set()
                # Brief pause so the watchdog has time to respawn before
                # the next iteration tries to write again.
                time.sleep(0.02)
                continue
            now = time.perf_counter()
            if self._last_writer_ts is not None:
                interval_ms = (now - self._last_writer_ts) * 1000.0
                # the append +
                # the np.array(deque) snapshot below both go through
                # `_stats_lock`. Pre-fix the writer thread could die on
                # `RuntimeError: deque mutated during iteration` if a
                # cross-thread `list(deque)` ran via the woys-diag /
                # TUI poll path. The dying thread is the writer
                # itself, which means audio stops silently -- the most
                # serious of the three F-merged-017 sub-bugs.
                snapshot: list[float] | None = None
                with self._stats_lock:
                    self.stats._writer_intervals_ms.append(interval_ms)
                    if len(self.stats._writer_intervals_ms) >= 16:
                        # B27 / corr-008: refresh jitter every chunk
                        # once the deque is full. Cost is ~10 us / chunk
                        # on a 128-deque; trivial.
                        snapshot = list(self.stats._writer_intervals_ms)
                if snapshot is not None:
                    arr = np.array(snapshot, dtype=np.float32)
                    self.stats.writer_jitter_ms = float(arr.std())
            self._last_writer_ts = now

    def _stderr_reader_loop(self, proc: subprocess.Popen[bytes]) -> None:
        """Daemon thread: parses pacat -v stderr for underrun tokens.

        Bound to a single pacat process - when it exits, readline returns
        b'' and the thread terminates. The watchdog spawns a new reader
        for the replacement process.

        B32 / corr-018: in pw-cat mode, parsing for "underrun" is futile
        (pw-cat doesn't emit that token); instead we just drain the pipe
        so it doesn't fill the kernel buffer (~64 KB) and deadlock the
        subprocess. The xruns counter stays 0 in pw-cat mode (already
        documented in `woys diag`).
        """
        if proc.stderr is None:
            return
        is_pacat = self._player_backend == "pacat"
        is_native = self._player_backend == "native-pw"
        # Diagnostic tee: when WOYS_HELPER_STDERR_LOG is set, every line
        # the player backend writes to stderr is also appended to that
        # path with a wall-clock timestamp. Zero overhead when the env
        # var is unset. Useful for forensic post-mortems of "the helper
        # died at some point during a session" cases - we lose nothing
        # to the existing parse-and-overwrite pattern.
        # v0.14.0 (area 6 / area 12 / C021): the env-driven path is
        # security-sensitive. Pre-v0.14.0 it called `open(path, "ab")`
        # with no symlink protection; an attacker who controls the path
        # value could swap a symlink to a victim file (e.g. ~/.bashrc)
        # between checks and corrupt it via "ab" append. Mitigations:
        #   1. Require absolute path (relative paths are caller-confused).
        #   2. Use os.open with O_NOFOLLOW so symlinks at the final
        #      component are refused (we open the actual file, not its
        #      symlink target).
        #   3. Skip silently with a `last_error` warning if the open
        #      fails for any reason -- the diagnostic is opt-in, not
        #      load-bearing.
        debug_log_path = os.environ.get("WOYS_HELPER_STDERR_LOG")
        debug_fp = None
        if debug_log_path:
            try:
                if not os.path.isabs(debug_log_path):
                    raise ValueError(
                        f"WOYS_HELPER_STDERR_LOG must be an absolute path, got {debug_log_path!r}"
                    )
                # refuse paths
                # outside `$XDG_RUNTIME_DIR/woys/` (or the secure
                # `/tmp/woys-{uid}/` fallback). Pre-fix a user
                # innocently setting this to `/tmp/woys-helper.log`
                # opened a symlink-attackable path in a world-
                # traversable directory. The O_NOFOLLOW below still
                # guards the open call, but constraining the
                # location keeps an attacker from positioning a
                # symlink mid-flight in the first place.
                from woys.xdg import safe_runtime_dir

                runtime_dir = safe_runtime_dir()
                resolved = os.path.realpath(debug_log_path)
                if not resolved.startswith(str(runtime_dir.resolve()) + os.sep):
                    raise ValueError(
                        f"WOYS_HELPER_STDERR_LOG must live under {runtime_dir} "
                        f"(refusing {debug_log_path!r}; "
                        f"a path outside the user-private runtime dir is "
                        f"symlink-attackable -- F-05-11)"
                    )
                fd = os.open(
                    debug_log_path,
                    os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW,
                    0o600,
                )
                debug_fp = os.fdopen(fd, "ab", buffering=0)
            except (OSError, ValueError) as e:
                with self._stats_lock:
                    self.stats.priority_warnings.append(
                        f"WOYS_HELPER_STDERR_LOG disabled: {type(e).__name__}: {e}"
                    )
                debug_fp = None
        try:
            for raw in proc.stderr:
                if not raw:
                    break
                if debug_fp is not None:
                    ts = time.strftime("%H:%M:%S", time.localtime())
                    with contextlib.suppress(OSError):
                        debug_fp.write(f"[{ts} {self._player_backend}] ".encode() + raw)
                line = raw.decode("utf-8", errors="replace")
                if is_pacat:
                    # pacat -v prints lines like "Stream underrun.\n" exactly.
                    # We match case-insensitively in case the wording shifts
                    # across PulseAudio versions.
                    if "underrun" in line.lower():
                        with self._stats_lock:
                            self.stats.xruns += 1
                elif is_native:
                    # v0.9.0 - native helper emits:
                    #   "ready"                      once after STREAMING
                    #   "quantum=N rate=M ..."       once after format negotiation
                    #   "underruns=N"                every UNDERRUN_REPORT_SECS
                    #   "error: <msg>"               fatal
                    s = line.strip()
                    if s.startswith("underruns="):
                        try:
                            count = int(s[len("underruns=") :])
                        except ValueError:
                            count = 0
                        self.stats.player_underruns = count
                    elif s.startswith("error:"):
                        # Surface the helper's hard-fail message to woys diag.
                        cause = f"native-pw: {s[len('error:') :].strip()}"
                        self.record_error(cause)
                        # v0.11.0 - also push to helper_exit_reasons so the
                        # watchdog's "respawned" message can't clobber the
                        # cause when the watchdog fires shortly after.
                        with self._stats_lock:
                            self.stats.helper_exit_reasons.append(cause)
                            if len(self.stats.helper_exit_reasons) > 10:
                                self.stats.helper_exit_reasons.pop(0)
                # else: pw-cat or unknown - drain-only.
        except (ValueError, OSError):
            # Pipe closed mid-read during shutdown - expected.
            return
        finally:
            if debug_fp is not None:
                with contextlib.suppress(OSError):
                    debug_fp.close()

    # ---- v0.11.0 - GPU clock lock + torch separate-stream keepalive ----------

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
                self._prior_signal_handlers[sig] = signal.signal(
                    sig, self._signal_handler_revert_lock
                )

    def _restore_prior_signal_handlers(self) -> None:
        """Re-install the SIGTERM/SIGINT handlers that were active before
        the engine started (Textual's, the CLI's).

        split out of `_revert_gpu_clock_lock`
        so the signal handler can restore handlers without also triggering
        the `sudo nvidia-smi` fork. Fast and fork-free -- safe to call from
        the signal handler itself.
        """
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
            os.kill(os.getpid(), signum)
            return
        self._signal_received = signum
        self._stop_event.set()
        self._restore_prior_signal_handlers()
        # Re-raise. The prior handler is now installed; the kernel delivers
        # this signal to it. The clock-lock revert runs later, on a normal
        # stack, in stop().
        os.kill(os.getpid(), signum)

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

    def _watchdog_loop(self) -> None:
        """Daemon thread: respawns pacat if it dies mid-session (Brief §3 Fix 3).

        Polls every `pacat_watchdog_interval_s` (50 ms by default). On dead
        process: opens a replacement under `_pacat_lock`, swaps the handle,
        spawns a fresh stderr reader for the new process, and increments
        `pacat_restarts`. Recovery target ≤ 100 ms.

        the respawn loop is capped. Pre-fix a
        helper that could never stay alive (raised every time, or spawned
        then died immediately) was retried forever with `running=True` and
        zero audio. `consecutive_respawns` counts deaths-without-recovery;
        a healthy tick (`poll() is None`) resets it, so a one-off death
        costs one respawn and does not accumulate. At the cap the engine
        stops cleanly with a definitive `last_error` (mirrors the
        inference circuit breaker).
        """
        consecutive_respawns = 0
        while not self._stop_event.is_set():
            # Wake immediately if the writer signalled BrokenPipe; otherwise
            # poll on the configured interval.
            self._pacat_dead_event.wait(timeout=self.cfg.pacat_watchdog_interval_s)
            self._pacat_dead_event.clear()
            with self._pacat_lock:
                proc = self._pacat_proc
            if proc is None:
                continue
            if proc.poll() is None:
                consecutive_respawns = 0  # alive this tick -> healthy, reset the cap
                continue  # still alive
            # Dead. Count this respawn attempt; a helper that never stays
            # alive must not loop forever.
            consecutive_respawns += 1
            if consecutive_respawns > _PLAYER_RESPAWN_CAP:
                self.record_error(
                    f"engine stopping: playback helper died and was respawned "
                    f"{consecutive_respawns - 1}x without staying alive. "
                    f"Last cause: {self.stats.last_error or 'unknown'}"
                )
                self._stop_event.set()
                return
            # Respawn.
            try:
                new_proc = self._open_pacat()
            except Exception as e:
                self.record_error(
                    f"watchdog respawn failed ({consecutive_respawns}x): {type(e).__name__}: {e}"
                )
                # Back off a bit before retrying so we don't spin.
                time.sleep(0.5)
                continue
            # B11 / corr-007: if stop fired while we were opening the new
            # proc (a slow path - _open_pacat takes ~50-200 ms), do NOT
            # install the new handle. Kill it instead so the engine's
            # finally-block teardown sees a stable `_pacat_proc` and the
            # new proc doesn't leak fds.
            if self._stop_event.is_set():
                with contextlib.suppress(Exception):
                    new_proc.terminate()
                    new_proc.wait(timeout=0.5)
                return
            with self._pacat_lock:
                # Discard the dead handle (caller already detected death).
                self._pacat_proc = new_proc
            with self._stats_lock:
                self.stats.player_restarts += 1
            # v0.11.0 - preserve the helper's own death cause if the
            # stderr reader captured one before the exit. If not, log
            # the watchdog's view (exit code + chunk index) so the user
            # can correlate. Either way, append to helper_exit_reasons
            # rather than clobber last_error wholesale.
            backend = self._player_backend or "player"
            exit_code = proc.returncode
            chunk_idx = self.stats.chunks_processed
            watchdog_msg = (
                f"{backend} exited code={exit_code} at chunk={chunk_idx} "
                f"(restart #{self.stats.player_restarts})"
            )
            with self._stats_lock:
                self.stats.helper_exit_reasons.append(watchdog_msg)
                if len(self.stats.helper_exit_reasons) > 10:
                    self.stats.helper_exit_reasons.pop(0)
            self.record_error(
                f"{backend} respawned (restarts={self.stats.player_restarts}); "
                f"causes={self.stats.helper_exit_reasons[-3:]}"
            )
            # Spawn a fresh stderr reader bound to the new process.
            #
            # join the OLD
            # reader thread before overwriting the reference.
            # Pre-fix we just reassigned `self._stderr_thread = new`
            # -- the prior thread became unreachable, kept its FD,
            # and an external thread inspecting `self._stderr_thread
            # .is_alive()` would only see the new one (the old was a
            # daemon, would eventually exit, but until it did we
            # leaked one Thread object per respawn). The old thread
            # is daemon + reads from the dead process's stderr pipe,
            # which EOFs as soon as the process is terminated -- so
            # the join is at most a few ms.
            old_stderr_t = self._stderr_thread
            if old_stderr_t is not None and old_stderr_t.is_alive():
                old_stderr_t.join(timeout=0.5)
            stderr_t = threading.Thread(
                target=self._stderr_reader_loop,
                args=(new_proc,),
                name="woys-pacat-stderr",
                daemon=True,
            )
            stderr_t.start()
            self._stderr_thread = stderr_t

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
            # Tearing down threads + pacat. _stop_event was set by stop()
            # (or we're here via exception); writer/watchdog will exit on
            # the next loop iteration.
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
