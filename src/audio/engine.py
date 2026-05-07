"""Realtime voice-conversion engine.

Wires the Phase 1 ONNX inference path to a real-time mic→infer→sink loop.

Audio routing — IMPORTANT (see v0.1.1 fix)
------------------------------------------
On CachyOS, PortAudio is built with the ALSA host API only (no PulseAudio host
API). `sd.OutputStream()` with no explicit `device=` falls through to the ALSA
*default* device, which routes to the system default sink (laptop speakers /
headphones) — NOT to the named PipeWire sink we want. Setting `PULSE_SINK=…`
in the environment is also ignored, because there's no Pulse host API for
PortAudio to consult.

The fix: instead of `sd.OutputStream`, the engine spawns
`pacat --playback --device=WoysSink …` as a subprocess and pipes
raw float32 PCM to its stdin. `pacat` is the canonical PulseAudio client; it
talks to pipewire-pulse natively, takes an explicit `--device=` argument, and
never auto-routes to the system default. This is the same path that the
acoustic loopback bench (`scripts/bench_loopback.py`) uses — proven on this host.

Input is still `sd.InputStream` against the default mic; that path was always
correct (host mic → 48 kHz capture).

Optional local monitoring
-------------------------
By default, **the engine writes the transformed audio to ONLY the virtual
sink** (which `woys-mic` reads from). Nothing plays out of the laptop
speakers — your housemates / streamers / phone calls don't hear what you're
processing. Pass `monitor=True` to additionally play to the host's default
output for self-monitoring.

Threading
---------
- Worker thread runs the blocking I/O loop and feeds the pacat subprocess.
- TUI thread polls `EngineStats` for live UI; no shared mutable state beyond
  a few atomic-ish primitives.
"""

from __future__ import annotations

import contextlib
import gc
import os
import queue
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

# ORT-GPU 1.20+ on driver 595 needs explicit preload of the pip-shipped CUDA libs.
import onnxruntime as ort

NDArrayF32 = npt.NDArray[np.float32]
NDArrayI64 = npt.NDArray[np.int64]

if hasattr(ort, "preload_dlls"):
    ort.preload_dlls()


MODELS_DIR = Path.home() / ".local" / "share" / "woys" / "models"

# Defaults pulled from Phase 1 inventory.
DEFAULT_RVC_MODEL = MODELS_DIR / "amitaro_v2_16k.onnx"
DEFAULT_RMVPE = MODELS_DIR / "rmvpe_wrapped.onnx"
DEFAULT_CONTENTVEC = MODELS_DIR / "contentvec-f.onnx"


@dataclass
class EngineConfig:
    rvc_model: Path = DEFAULT_RVC_MODEL
    rmvpe_model: Path = DEFAULT_RMVPE
    contentvec_model: Path = DEFAULT_CONTENTVEC

    # Audio I/O
    mic_rate: int = 48_000
    sink_rate: int = 48_000
    # v0.7.0 — dropped from 0.25 → 0.15. Empirical sweep (RTX 2070 Mobile,
    # ORT-CUDA, cuDNN HEURISTIC, full realtime engine, catwoman voice):
    #
    #   chunk   late_chunks/total   inference avg   p99
    #   0.10    13–42 / 100         77–98 ms       103–148 ms
    #   0.15    0 / 80              76–80 ms       104–129 ms
    #   0.20    0 / 60              92–96 ms       115–122 ms
    #   0.25    0 / 50              83 ms          124 ms
    #
    # At chunk=0.10 the per-chunk budget is 100 ms but real engine inference
    # is 77–98 ms (a +50 ms tax over the standalone benchmark, traced to
    # GIL/scheduler effects of running inside the engine sub-thread; see
    # LESSONS §19). 13–42 % of chunks miss budget at 0.10, even on the
    # smallest voice. At chunk=0.15 the budget is 150 ms and zero chunks
    # miss it across both light and heavy voices — that's the practical
    # floor on this hardware. v0.7.0 picks 0.15 (saves 100 ms vs the v0.6.x
    # 0.25 default; doesn't pretend 0.10 is achievable).
    #
    # Historical v0.5.1 reason for raising chunk to 0.25 ("SOLA tail-trim ate
    # 10 % of output duration") was repaired by the v0.6.9 SOLA tuning
    # (search_ms 4.0 → 6.0, corr_threshold 0.25 → 0.10). Historical v0.6.7
    # reason ("dropped chunks during cuDNN warmup") was repaired by the
    # v0.7.0 HEURISTIC switch + broader pre-warm shape coverage.
    chunk_seconds: float = 0.15
    channels: int = 1

    # SOLA crossfade (Phase B). Disable at your peril — without it, audible
    # clicks at every chunk boundary when chunk_seconds is short.
    sola_enabled: bool = True
    sola_crossfade_ms: float = 50.0  # overlap window between consecutive chunks
    # v0.6.9: widened from 4.0 to 6.0 so the search window covers at least one
    # full pitch period for typical voice f0 (>= 167 Hz period at 40 kHz model
    # rate = 240 samples = 6 ms). With a sub-period search, sustained vowels
    # produce phase mismatches SOLA can't reach — manifests as audible
    # dropouts during sustained voicing.
    sola_search_ms: float = 6.0  # how far to shift looking for in-phase alignment
    # History fed to the model alongside each new chunk so the embedder /
    # vocoder convolutions don't see edge artifacts. Brief calls this "context".
    sola_context_ms: float = 100.0
    # v0.6.9: lowered from sola.py's 0.25 default. With the original threshold,
    # SOLA falls back to centered (offset=0) on borderline cases and produces
    # phase-discontinuous crossfade for sustained content. 0.10 still rejects
    # decorrelated noise but keeps best-effort alignment for periodic signals.
    sola_corr_threshold: float = 0.10

    # RVC
    f0_up_key: int = 0  # semitones
    sid: int = 0
    threshold: float = 0.3

    # Embedder selection (v0.2.0):
    #   "onnx"    — direct ORT contentvec-f.onnx call (default, fastest, no torch+fairseq)
    #   "fairseq" — upstream FairseqHubert PyTorch path; needs the [fairseq] extra installed.
    # Misconfiguration / missing fairseq → fall back to "onnx" with a warning,
    # never crash the engine. (Brief Phase A.)
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
    # real-world Telegram VoIP testing — confirming the rc2 retro point
    # that the synthetic harness over-counts cuts uniformly and can't
    # distinguish real-speech variance within its flat region. 280 ms
    # is the last rung in the rc ladder: 20 ms under the v0.6.x 300 ms
    # default that we already know is audibly clean. If 280 also fails,
    # the structural floor on this hardware is hit and further latency
    # reduction needs the ~80 ms engine threading tax (LESSONS §19)
    # closed first — that's v0.8.x territory, not another rc bump.
    # Wall-clock at rc3: chunk 150 + inference 80 + buffer 280 +
    # codec 30 ≈ 540 ms (vs v0.6.x ~660 ms, −18 %).
    output_latency_ms: int = 280
    # Process-time hint to pacat: write callbacks granulate to this many
    # ms. 20 ms keeps writes from coalescing into bursts that would
    # alternately starve and overrun the buffer. Ignored by pw-cat, which
    # uses PipeWire's quantum negotiation instead.
    output_process_time_ms: int = 20

    # v0.7.0-rc4 — flipped back to False. v0.7.0-rc1 reverted to pw-cat
    # on the reasoning that smaller chunks at chunk_seconds=0.15 would
    # eliminate the per-quantum stdin/PipeWire-callback race v0.6.7
    # documented (~43 ms zero-gaps on bursty writes). The
    # `docs/16-audit/synthesis.md` retro disagreed: rc1's "this won't
    # apply" is hand-wavy and doesn't address the race mechanism, and
    # the symptom we hear in Telegram (sample-exact zeros, voice-
    # correlated, ~40 ms quantized — see lens 08) matches pw-cat's
    # documented per-quantum-gap pattern more closely than pacat's
    # underrun pattern. Migration cascade in `tui/config.py` pulls
    # users on the rc1+ default sentinel `True` forward to `False`;
    # users who explicitly set `prefer_pw_cat = true` after the field
    # is exposed in AppConfig keep their override.
    prefer_pw_cat: bool = False

    # v0.5.1: software input pre-attenuation, in dB. Default 0.0 (passthrough).
    # Hot mics (HyperX QuadCast at high volume etc.) clip the signal which
    # RVC amplifies as harsh distortion downstream. Setting a small
    # negative value (-3 to -6 dB) trims headroom without quieting much.
    # Applied per chunk before resample → embedder.
    input_gain_db: float = 0.0

    # v0.5.0 session-pool tuning.
    # Cap on simultaneous cached RVC sessions (each ~150 MiB VRAM).
    session_pool_size: int = 4
    # If true, on engine.start() we eagerly create + cudnn-warm sessions for
    # every .onnx in the models dir (minus foundations). Adds ~6-12 s to
    # cold start for a 10-voice library, but every subsequent swap is a
    # pointer swap (~10 ms). Recommended for users with persistent engines.
    eager_warmup: bool = False

    # v0.5.2 — pacat underrun mitigations (see docs/08-pacat-underrun-bug.md).
    # Channels emitted by the engine. The PipeWire null-sink loaded by
    # `woys pw setup` defaults to 2 channels; emitting 2 here
    # avoids an in-graph 1→2 upmix on every chunk.
    output_channels: int = 2
    # Bounded queue between the engine main loop and the pacat writer
    # thread. Size 8 ≈ 2 s of slack at chunk_seconds=0.25; full-queue
    # events are exposed as `queue_full_events` (xrun proxy).
    pacat_writer_queue_size: int = 8
    # v0.6.7 part 3 — initial silence written to the playback backend
    # before any real engine output starts. Empirically didn't reduce
    # xruns in our trials (in fact slightly increased them — pacat seems
    # to apply its prebuf threshold to the silence and trip more
    # frequently). Default 0 (off). Kept as a tunable for users whose
    # backends (or future versions of pacat / pw-cat) might benefit.
    # See `docs/11-microcuts-bug.md` part 3.
    prime_silence_seconds: float = 0.0
    # v0.7.0-rc4 — gate threshold lowered -55 → -75. The audit
    # (`docs/16-audit/synthesis.md`, lens 06 / S1) traced rc1/rc2/rc3
    # cuts to this gate firing on intra-speech RMS dips: -55 dBFS is
    # only ~6 dB below typical room noise on a QuadCast, and brief
    # speech valleys (between syllables, on plosive onsets, during
    # fricatives) routinely cross it. Each fire emits a full chunk
    # of zeros directly to the writer, bypassing SOLA, both
    # resamplers, and inference, with no counter incremented — which
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
    # v0.7.0-rc4 — hysteresis on the input gate. The gate must observe
    # `input_gate_hysteresis_ms` of continuously-below-threshold input
    # before it fires. Brief dips in voiced speech (typical: 30–150 ms
    # between syllables, on consonant onsets) no longer trigger
    # zero-emission, even if they momentarily cross threshold. Set to
    # 0 for the v0.6.9 behavior (immediate gating with no smoothing).
    # 200 ms is roughly the upper end of natural inter-syllable pause
    # in speech — anything beyond that is genuinely silence and the
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
    # Opt-in process-priority bump. Requires CAP_SYS_NICE (or root). On
    # PermissionError we log + continue at SCHED_OTHER. OFF by default
    # per Brief §6 — capability requirement is a host-setup concern.
    realtime_priority: bool = False


@dataclass
class EngineStats:
    running: bool = False
    chunks_processed: int = 0
    last_input_rms: float = 0.0
    last_inference_ms: float = 0.0
    avg_inference_ms: float = 0.0
    last_total_ms: float = 0.0
    avg_total_ms: float = 0.0
    # v0.6.9 — outlier visibility. avg_*_ms hides single slow chunks that
    # arrive after the audio sink has already underrun. max_* tracks the
    # worst chunk since session start; late_chunks counts chunks where total
    # processing exceeded the chunk budget (chunk_seconds * 1000).
    max_inference_ms: float = 0.0
    max_total_ms: float = 0.0
    late_chunks: int = 0
    # v0.6.9 round 5 — per-stage timing for the most recent chunk so the
    # slow_chunk_log breakdown points at which ONNX session was responsible.
    last_cv_ms: float = 0.0
    last_rmvpe_ms: float = 0.0
    last_rvc_ms: float = 0.0
    # Last N chunks where total_ms exceeded the chunk budget. Each entry is a
    # dict {chunk_idx, total_ms, inf_ms, cv_ms, rmvpe_ms, rvc_ms, input_rms}.
    # Surface via the SLOW socket command -> /tmp/woys-slow-chunks.txt.
    slow_chunk_log: list[dict[str, float]] = field(default_factory=list)
    # v0.7.0-rc8 — chunks whose inference time was > 2× the running
    # p50 of recent inference, regardless of whether total_ms passed
    # chunk_seconds*1000. The rc7 diag dump showed inference p50=40 ms
    # / p99=96 ms / max=110 ms with overrun_ratio=0 — a 70 ms tail
    # spread that doesn't trip the existing `slow_chunk_log` (gated on
    # total_ms > chunk_seconds*1000 = 150 ms). This list captures the
    # tail chunks so we can correlate their inf_ms with input shape,
    # history size, RMS, and per-session-stage breakdown. If slow
    # chunks share a common signature (specific audio16_len, specific
    # cv vs rmvpe vs rvc dominance, specific RMS band), rc9's fix
    # targets that mechanism. Capped at 50 entries.
    tail_chunk_log: list[dict[str, float]] = field(default_factory=list)
    last_error: str | None = None

    # v0.5.2 health counters (Brief §5 — surfaced in TUI + `diag`).
    # xruns: parsed from pacat -v stderr. Closest thing to a true
    #   PulseAudio-side underrun count without reaching into pw-dump.
    # queue_full_events: writer queue was full when the engine tried to
    #   enqueue → engine has out-paced the writer/sink, treat as a
    #   self-detected underrun.
    # pacat_restarts: watchdog respawned pacat (it died mid-session).
    # writer_jitter_ms: std dev (ms) of recent inter-chunk write
    #   intervals. Exceeding ~5 % of chunk_seconds*1000 is the
    #   underrun precursor we care about.
    xruns: int = 0
    queue_full_events: int = 0
    pacat_restarts: int = 0
    writer_jitter_ms: float = 0.0
    # v0.6.8 — count of chunks the engine had to drop because inference
    # raised (GPU OOM, numerical, transient ORT error). Without this,
    # any single bad chunk crashes the entire engine; with it, we drop
    # the chunk, leave a brief silence (SOLA tail covers most of it),
    # and keep going. First few hits log to `last_error`; subsequent
    # ones increment silently to avoid spamming the TUI.
    dropped_chunks: int = 0
    # v0.7.0-rc4 — instrumentation for the four silent-drop classes the
    # `docs/16-audit/synthesis.md` audit identified as previously
    # invisible to every existing counter. Each is incremented at
    # the exact site that emits zeros / loses samples; together with
    # `dropped_chunks` and `queue_full_events` they cover every
    # silence-emit path the audit catalogued. Surfaced in `woys diag`
    # output and the TUI STATUS reply so the next debug cycle isn't
    # blind.
    #
    #   input_overflows    — sd.InputStream.read() reported
    #                        `overflowed=True` (mic-side ring underflow,
    #                        previously dropped on the floor at the
    #                        tuple-unpack site).
    #   gated_chunks       — input gate fired and emitted a chunk of
    #                        zeros; bypasses SOLA + resamplers +
    #                        inference, so an upstream of every buffer.
    #   nan_chunks         — RVC vocoder output had NaN/inf and was
    #                        sanitized to zero (v0.6.9 path); a
    #                        non-zero rate during real-speech is
    #                        evidence for the C-class hypothesis from
    #                        the audit.
    #   sola_fallback_count — SOLA's alignment search peak correlation
    #                        fell below `corr_threshold`; the algorithm
    #                        used `offset = 0` (centered, no shift). In
    #                        rc5 this no longer affects emit length —
    #                        SOLA always emits `chunk_n` samples per call
    #                        regardless of fallback — so this counter is
    #                        purely a "how often is the search giving up"
    #                        diagnostic, not a cuts driver.
    #
    # rc4's `sola_drain_ms` (cumulative ms of zero-padding) was removed
    # in rc5 because the pad path itself was removed. SOLA emits
    # constant-size chunks now (`docs/16-audit/11-rc4-postmortem.md`
    # §"Proposed rc5 scope"). Drain is structurally zero by construction.
    input_overflows: int = 0
    gated_chunks: int = 0
    nan_chunks: int = 0
    sola_fallback_count: int = 0

    # v0.7.0-rc6 — per-stage producer-side timing for the writer-jitter
    # investigation. The rc5 postmortem
    # (`docs/16-audit/12-rc5-writer-jitter-probe.md`) attributed the
    # live `writer_jitter_ms = 62` to producer-side cadence variance,
    # not consumer-side. These two new stages plus the existing
    # inference timing sum to per-iteration wall time:
    #
    #   mic_read_ms       blocking read of chunk_mic samples (PortAudio
    #                     / ALSA — should hover near chunk_seconds *
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
    _recent_inference: deque[float] = field(default_factory=lambda: deque(maxlen=32))
    _recent_total: deque[float] = field(default_factory=lambda: deque(maxlen=32))
    # v0.5.2 — inter-write intervals in ms (writer thread fills this).
    _writer_intervals_ms: deque[float] = field(default_factory=lambda: deque(maxlen=128))
    # v0.7.0-rc6 — wider window than _recent_inference (32) so p95/p99
    # have enough samples to be stable. 128 chunks ≈ 19 s at
    # chunk_seconds=0.15.
    _recent_mic_read_ms: deque[float] = field(default_factory=lambda: deque(maxlen=128))
    _recent_enqueue_lag_ms: deque[float] = field(default_factory=lambda: deque(maxlen=128))


# v0.7.0: cuDNN convolution algorithm search strategy. EXHAUSTIVE was the
# v0.2.0 default — picks the fastest steady-state algo per shape but eats
# 50–100 ms autotune the first time each shape lands. At chunk_seconds=0.25
# the autotune is amortized over a long session; at chunk_seconds=0.10 the
# first 5–10 chunks miss budget while autotune runs (was the v0.5.1 root
# cause for raising chunk back to 0.25). HEURISTIC picks a near-optimal
# algo from a heuristic without any timed search — slightly slower
# steady-state (typically 1–3 % per chunk on this hardware) but no
# autotune lump, so dropping chunk_seconds becomes safe. The setting can
# still be flipped back via the env var if a future ORT release improves
# EXHAUSTIVE's amortization.
_CUDNN_ALGO_SEARCH = "HEURISTIC"


def _make_session(path: Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    providers: list[tuple[str, dict[str, object]] | str] = []
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.append(
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "cudnn_conv_algo_search": _CUDNN_ALGO_SEARCH,
                    "do_copy_in_default_stream": True,
                },
            )
        )
    providers.append("CPUExecutionProvider")
    return ort.InferenceSession(str(path), sess_options=so, providers=providers)


# v0.6.9 — pitchf sanitization for the realtime inference path.
# Frames with NaN or f0 <= 0 are treated as "unvoiced" by the RVC vocoder's
# NSF source module; a single such frame mid-utterance zeros the harmonic
# source and produces an audible dropout. We replace NaN with 0 first
# (defensive against extractor bugs), then linearly interpolate runs of
# unvoiced frames up to `_VOICED_GAP_MAX_FRAMES` long between two voiced
# frames. Long unvoiced runs are left as zeros so true silence still
# decodes as silence. See `docs/12-vad-misfire-investigation.md`.
_VOICED_GAP_MAX_FRAMES = 8  # ~80 ms at the RMVPE 100 fps frame rate


def _interpolate_voiced_gaps_np(pitchf: NDArrayF32) -> NDArrayF32:
    if pitchf.size == 0:
        return pitchf
    arr = pitchf.astype(np.float64, copy=True)
    invalid = np.isnan(arr) | (arr <= 0.0)
    if not invalid.any():
        return pitchf
    if (~invalid).sum() == 0:
        # Whole chunk is unvoiced — preserve so vocoder produces silence.
        out_silence: NDArrayF32 = np.nan_to_num(arr, nan=0.0).astype(np.float32, copy=False)
        return out_silence
    out = np.nan_to_num(arr, nan=0.0)
    last_valid = -1
    i = 0
    n = len(invalid)
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
            for k in range(i, j):
                alpha = (k - last_valid) / (j - last_valid)
                out[k] = out[last_valid] * (1.0 - alpha) + out[j] * alpha
        i = j
    out_f32: NDArrayF32 = out.astype(np.float32, copy=False)
    return out_f32


def _to_pitch_coarse(pitchf: NDArrayF32, target_len: int) -> tuple[NDArrayI64, NDArrayF32]:
    f0_min, f0_max = 50.0, 1100.0
    f0_mel_min = 1127.0 * np.log(1 + f0_min / 700.0)
    f0_mel_max = 1127.0 * np.log(1 + f0_max / 700.0)
    pitch = np.zeros(target_len, dtype=np.float32)
    n = min(len(pitchf), target_len)
    pitch[-n:] = pitchf[:n]
    f0_mel = 1127.0 * np.log(1 + pitch / 700.0)
    mask = f0_mel > 0
    f0_mel[mask] = (f0_mel[mask] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
    f0_mel = np.clip(f0_mel, 1.0, 255.0)
    return np.rint(f0_mel).astype(np.int64), pitch


def _resample_linear(audio: NDArrayF32, src_rate: int, dst_rate: int) -> NDArrayF32:
    """Cheap linear-interp resampler — kept as a known-bad reference baseline
    for v0.5.1 tests. Production path uses `_resample` (soxr).

    Linear interpolation has no anti-aliasing low-pass: frequencies above
    `dst_rate / 2` fold back as audible high-frequency noise. RMSE on a
    1 kHz sine round-trip 48k→40k→48k is ~30x worse than soxr HQ.
    """
    if src_rate == dst_rate:
        return audio.astype(np.float32, copy=False)
    n_src = len(audio)
    n_dst = round(n_src * dst_rate / src_rate)
    if n_dst <= 0:
        return np.zeros(0, dtype=np.float32)
    src_idx = np.linspace(0, n_src - 1, n_dst, dtype=np.float64)
    floor = np.floor(src_idx).astype(np.int64)
    ceil = np.minimum(floor + 1, n_src - 1)
    frac = (src_idx - floor).astype(np.float32)
    out: NDArrayF32 = ((1 - frac) * audio[floor] + frac * audio[ceil]).astype(np.float32)
    return out


def _resample(audio: NDArrayF32, src_rate: int, dst_rate: int) -> NDArrayF32:
    """High-quality stateless resampler — used for one-shot tests + tail flushes.

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
    """Stateful soxr resampler — preserves filter state across chunks.

    Per-call `soxr.resample(...)` resets the anti-aliasing filter every
    invocation; concatenating the resampled chunks introduces a brief
    filter-transient amplitude dip at every chunk boundary, audible as
    a 4 Hz flutter on sustained content (`docs/11-microcuts-bug.md`).
    `soxr.ResampleStream` carries the filter buffer across calls and
    eliminates the per-chunk warm-up.

    Identity case (`src_rate == dst_rate`) is a passthrough — no soxr
    object created.
    """

    def __init__(self, src_rate: int, dst_rate: int, *, quality: str = "HQ") -> None:
        self.src_rate = src_rate
        self.dst_rate = dst_rate
        if src_rate == dst_rate:
            self._stream = None
            return
        import soxr

        self._stream = soxr.ResampleStream(src_rate, dst_rate, num_channels=1, quality=quality)

    def process(self, audio: NDArrayF32) -> NDArrayF32:
        """Consume `audio` (1-D float32 mono); return whatever soxr emits
        for this chunk. Output length will lag input length slightly while
        the internal buffer fills — flush() drains the rest."""
        if self._stream is None:
            return audio.astype(np.float32, copy=False)
        if audio.size == 0:
            return np.zeros(0, dtype=np.float32)
        out = self._stream.resample_chunk(audio, last=False)
        return np.asarray(out, dtype=np.float32).reshape(-1)

    def flush(self) -> NDArrayF32:
        """Drain any audio held in soxr's internal buffer. Call once before
        discarding (engine stop / model swap when output rate changes)."""
        if self._stream is None:
            return np.zeros(0, dtype=np.float32)
        out = self._stream.resample_chunk(np.zeros(0, dtype=np.float32), last=True)
        return np.asarray(out, dtype=np.float32).reshape(-1)


class _FairseqEmbedder:
    """Wrapper that lazy-loads FairseqHubert; tracks the active mode for stats."""

    def __init__(self, hubert_path: Path) -> None:
        # Imports are deferred until extract() is called so users without the
        # `[fairseq]` extra never pay the torch import cost.
        import torch
        from fairseq import checkpoint_utils  # type: ignore[import-not-found]

        models, _, _ = checkpoint_utils.load_model_ensemble_and_task([str(hubert_path)], suffix="")
        model = models[0]
        model.eval()
        if torch.cuda.is_available():
            model = model.to("cuda:0")
        self._torch = torch
        self.model = model
        self.dev = next(model.parameters()).device

    def extract(self, audio_np: NDArrayF32) -> NDArrayF32:
        torch = self._torch
        with torch.no_grad():
            feats_t = torch.from_numpy(audio_np.reshape(1, -1)).to(self.dev)
            padding_mask = torch.zeros(feats_t.shape, dtype=torch.bool, device=self.dev)
            logits = self.model.extract_features(
                source=feats_t, padding_mask=padding_mask, output_layer=12
            )
            out = logits[0].detach().to(torch.float32).cpu().numpy()
        return out  # type: ignore[no-any-return]


class RvcSessionPool:
    """Per-path cache of `ort.InferenceSession` objects.

    Hot-swap performance was the v0.4.x P0: every `models use` rebuilt the
    session from scratch, including cudnn EXHAUSTIVE algo-tuning, costing
    ~1.5 s + a 305 ms first-chunk inference burst. This pool keeps a small
    set of cached sessions; second swap to an already-seen voice is a
    pointer swap (~10 ms total).

    LRU eviction keeps VRAM bounded — a session uses ~150 MiB resident,
    so the default `max_size=4` caps voice-model VRAM at ~600 MiB on top
    of the foundations. Configurable via `EngineConfig.session_pool_size`.

    Thread-safe. The audio worker calls `get_or_create()` from inside
    `_maybe_swap_model`; tests / TUI may call it from any thread.
    """

    def __init__(self, max_size: int = 4) -> None:
        self._cache: dict[Path, ort.InferenceSession] = {}
        self._access_order: list[Path] = []
        self._max_size = max(1, max_size)
        self._lock = threading.Lock()

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

        # Cache miss — build outside the lock (slow); other threads can
        # still get cached sessions while we tune.
        sess = _make_session(key)

        with self._lock:
            # Another thread may have raced us; if so, drop ours and use theirs.
            if key in self._cache:
                return self._cache[key]
            self._cache[key] = sess
            self._access_order.append(key)
            while len(self._access_order) > self._max_size:
                evicted = self._access_order.pop(0)
                if evicted != key:
                    self._cache.pop(evicted, None)
        return sess

    def warmup(self, path: Path) -> ort.InferenceSession:
        """Create + run one dummy forward pass so cudnn populates its algo
        cache. Subsequent inferences against the same shape are near-instant.

        The caller is expected to know the model's input shape — we feed the
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
        # v0.7.0-rc7 — track whether GC was enabled before this engine
        # disabled it, so stop() restores the prior state instead of
        # blindly enabling. Lets us nest cleanly inside a parent that
        # had already disabled GC (rare but possible in tests).
        self._gc_was_enabled_before_start: bool = False

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

        # Active embedder mode after _ensure_sessions(). Either "onnx" or
        # "fairseq" — falls back from "fairseq" to "onnx" automatically if
        # the fairseq import or model load fails (Phase A spec).
        self.active_embedder: str = "onnx"
        self._fairseq: _FairseqEmbedder | None = None

        # SOLA streaming state (Phase B). v0.5.0 fix: SOLA operates at the
        # OUTPUT rate (model_sr — varies per voice: 16k for amitaro, 40k for
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

        # v0.4.1 hot-swap: the worker thread checks this slot at the top of
        # each chunk. The TUI / socket sets it via `request_model_swap`.
        self._pending_model_swap: Path | None = None
        self._swap_lock = threading.Lock()
        # Promoted so _maybe_swap can flush the SOLA tail through the same
        # pacat process the worker already owns. v0.5.2: protected by
        # `_pacat_lock` so the watchdog can swap the handle atomically.
        self._pacat_proc: subprocess.Popen[bytes] | None = None
        self._pacat_lock = threading.Lock()
        # Set by `_open_pacat` to either "pw-cat" or "pacat" — surfaced in
        # `woys diag` so the user can see which backend is live.
        self._player_backend: str = ""

        # v0.5.2 — pacat writer / watchdog / stderr-reader threads.
        # Lifetimes are bound to a single `_run_loop()` invocation: spawned
        # in `_run_loop`'s try, joined in its finally.
        self._writer_queue: queue.Queue[bytes] | None = None
        self._writer_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._watchdog_thread: threading.Thread | None = None
        # Watchdog signal: writer flips this on BrokenPipe so the watchdog
        # respawns immediately instead of waiting for its next poll tick.
        self._pacat_dead_event = threading.Event()
        # Last write timestamp (perf_counter). Writer thread updates it;
        # used to compute `writer_jitter_ms`.
        self._last_writer_ts: float | None = None

        # v0.5.0 session pool — shared cache so swap = pointer swap.
        self._rvc_pool = RvcSessionPool(max_size=self.cfg.session_pool_size)
        # Probed `model_sr` per voice path so we don't redo the probe each
        # swap. Keys are resolved Paths.
        self._rvc_sr_cache: dict[Path, int] = {}

        # v0.6.7 — stateful per-(src,dst) resamplers. Created in `_run_loop`
        # before the first chunk, replaced when the model output rate
        # changes during hot-swap. See `docs/11-microcuts-bug.md`.
        self._resampler_in: _StreamResampler | None = None
        self._resampler_out: _StreamResampler | None = None

    # ---- model loading ------------------------------------------------------

    def _ensure_sessions(self) -> None:
        # v0.3.0: prefer fp16 variants if present next to the fp32 file. fp16
        # rmvpe halves its VRAM footprint with no measurable pitch-detection
        # quality loss (validated v0.2.0). fp16 contentvec, by contrast, has
        # cosine sim 0.75 vs fp32 — only auto-promoted if explicitly requested.
        cv_path = self._auto_pick_fp16(self.cfg.contentvec_model, allow=False)
        rmvpe_path = self._auto_pick_fp16(self.cfg.rmvpe_model, allow=True)

        if self._cv is None:
            self._cv = _make_session(cv_path)
            self._cv_input_dtype = self._cv.get_inputs()[0].type
        if self._rmvpe is None:
            self._rmvpe = _make_session(rmvpe_path)
            self._rmvpe_input_dtype = self._rmvpe.get_inputs()[0].type
        if self._rvc is None:
            self._rvc = self._rvc_pool.get_or_create(self.cfg.rvc_model)
            self._is_half = self._rvc.get_inputs()[0].type != "tensor(float)"
            self._rvc_output_sr = self._cached_rvc_sr(Path(self.cfg.rvc_model))

        # Resolve embedder mode. ONNX is the default; "fairseq" is opt-in via
        # config and degrades gracefully when the package isn't installed.
        if self.cfg.embedder == "fairseq" and self._fairseq is None:
            hubert_path = MODELS_DIR / "hubert_base.pt"
            try:
                if not hubert_path.exists():
                    raise FileNotFoundError(
                        f"hubert_base.pt missing at {hubert_path} — "
                        "run `scripts/download_weights.py` to fetch it"
                    )
                self._fairseq = _FairseqEmbedder(hubert_path)
                self.active_embedder = "fairseq"
                print(f"[engine] embedder=fairseq (hubert_base.pt @ {hubert_path})")
            except Exception as e:
                msg = (
                    f"fairseq embedder unavailable ({type(e).__name__}: {e}); "
                    "falling back to ONNX contentvec"
                )
                print(f"[engine] {msg}")
                self.stats.last_error = msg
                self.active_embedder = "onnx"
        else:
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
        that is brittle — different exporters use different keys. Probing
        is bulletproof: feed a known-length input, count output samples.

        Costs ~20 ms once at session load. Worth it.
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
        except Exception:
            # Probe failed (model likely doesn't take pitch/pitchf — nono variant).
            # Fall back to 16 kHz; the engine will still work, just possibly chipmunk.
            return 16_000
        n_out = int(np.asarray(out).size)
        # Output for 1 s of feats input ≈ 1 s of audio at the model rate.
        # Round to the nearest known RVC training rate.
        for sr in (16_000, 22_050, 24_000, 32_000, 40_000, 44_100, 48_000):
            if abs(n_out - sr) < sr * 0.05:
                return sr
        # Unknown rate — best effort, treat the raw count as Hz.
        return n_out

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
        """Recreate the output-side SOLAStream for the given rate. Idempotent —
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
        """Hot-swap the RVC voice model — synchronous, thread-unsafe.

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
        """
        if voice_paths is None:
            foundations = {
                "rmvpe.onnx", "rmvpe-fp16.onnx",
                "rmvpe_wrapped.onnx", "rmvpe_wrapped-fp16.onnx",
                "contentvec-f.onnx", "contentvec-f-fp16.onnx",
                "hubert_base.onnx",
            }  # fmt: skip
            voice_paths = sorted(p for p in MODELS_DIR.glob("*.onnx") if p.name not in foundations)
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

    def request_model_swap(self, path: Path) -> None:
        """Thread-safe: queue a model swap for the worker to pick up at the
        next chunk boundary. Returns immediately. Idempotent — repeat calls
        replace the pending target. The SOLA tail is drained before the
        swap so consecutive chunks crossfade cleanly across the boundary.
        """
        with self._swap_lock:
            self._pending_model_swap = Path(path)

    def _maybe_swap_model(self) -> None:
        """Worker-side hook: if a swap was queued, flush SOLA tail then
        replace the RVC session and reset streaming state. Called from
        `_run_loop` at the top of each chunk."""
        with self._swap_lock:
            target = self._pending_model_swap
            self._pending_model_swap = None
        if target is None:
            return
        # Drain SOLA's held-back tail so the last ~50 ms of the *old* voice
        # plays out before the new session takes over. Tail is at the OLD
        # model's output rate, not necessarily 16 kHz. v0.5.2: route through
        # the writer queue so we don't bypass the BrokenPipe / xrun-counter
        # plumbing.
        if self._sola is not None:
            with contextlib.suppress(Exception):
                tail = self._sola.flush()
                if tail.size > 0 and self._resampler_out is not None:
                    tail48 = self._resampler_out.process(tail)
                    flush48 = self._resampler_out.flush()
                    full = np.concatenate([tail48, flush48]) if flush48.size else tail48
                    if full.size > 0:
                        self._enqueue_chunk(self._to_sink_bytes(full))
        # Replace the session. Existing _cv (contentvec) and _rmvpe stay
        # — they're foundation models, not voice-specific.
        # v0.5.0: pool-cached. Cache hit ≈ 10 ms; cache miss ≈ 600 ms.
        self.cfg.rvc_model = target
        self._rvc = self._rvc_pool.get_or_create(target)
        self._is_half = self._rvc.get_inputs()[0].type != "tensor(float)"
        new_sr = self._cached_rvc_sr(target)
        # v0.6.7 — rebuild the output resampler if the new model has a
        # different native rate. Identity ratios (e.g. 16k → 16k → 48k stays
        # the same) won't reset state, so swaps between same-rate voices
        # don't introduce a chunk-boundary artifact.
        if new_sr != self._rvc_output_sr:
            self._resampler_out = _StreamResampler(new_sr, self.cfg.sink_rate)
        self._rvc_output_sr = new_sr
        self.reset_streaming_state()

    # ---- inference ----------------------------------------------------------

    def _extract_feats(self, audio16k: NDArrayF32) -> NDArrayF32:
        """Embedder dispatch — see `EngineConfig.embedder` and `active_embedder`."""
        if self.active_embedder == "fairseq" and self._fairseq is not None:
            return self._fairseq.extract(audio16k.astype(np.float32, copy=False))
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

        Standalone path — used by tests and by the engine when SOLA is
        disabled. Doesn't touch streaming state. The streaming engine path
        goes through `_process_streaming_16k` instead.
        """
        return self._infer(audio16k)

    def _infer(self, audio16k: NDArrayF32) -> NDArrayF32:
        """Raw model invocation; no streaming bookkeeping."""
        assert self._cv is not None and self._rmvpe is not None and self._rvc is not None

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
        pitchf = _interpolate_voiced_gaps_np(pitchf)
        t_rmvpe1 = time.perf_counter()

        feats_2x = np.repeat(feats, 2, axis=1)
        pitch_coarse, pitchf_aligned = _to_pitch_coarse(pitchf, target_len=feats_2x.shape[1])
        pitch_coarse = pitch_coarse[: feats_2x.shape[1]].reshape(1, -1)
        # Apply pitch shift in semitones.
        if self.cfg.f0_up_key != 0:
            pitchf_aligned = pitchf_aligned * (2.0 ** (self.cfg.f0_up_key / 12.0))
        pitchf_aligned = pitchf_aligned[: feats_2x.shape[1]].reshape(1, -1).astype(np.float32)

        feats_dtype = np.float16 if self._is_half else np.float32
        out = self._rvc.run(
            ["audio"],
            {
                "feats": feats_2x.astype(feats_dtype),
                "p_len": np.array([feats_2x.shape[1]], dtype=np.int64),
                "pitch": pitch_coarse,
                "pitchf": pitchf_aligned,
                "sid": np.array([self.cfg.sid], dtype=np.int64),
            },
        )[0]
        result: NDArrayF32 = np.array(out).astype(np.float32).squeeze()
        # v0.6.9: belt-and-braces NaN sanitize. pacat is fed float32le; NaN
        # would be undefined behavior in PipeWire's mixer chain and the
        # listener hears it as a click + brief gap.
        if np.isnan(result).any() or np.isinf(result).any():
            result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)
            # v0.7.0-rc4 — count NaN-sanitize hits so we can attribute
            # voice-correlated cuts to the vocoder rather than the gate
            # or SOLA. Pre-rc4 the path silently zeroed samples; lens 06
            # of the audit flagged this as one of three NaN-zero paths
            # that incremented no counter.
            self.stats.nan_chunks += 1
        t_rvc1 = time.perf_counter()
        # Per-stage timing surfaces in the slow_chunk_log when a chunk goes late.
        self.stats.last_cv_ms = (t_cv1 - t_cv0) * 1000.0
        self.stats.last_rmvpe_ms = (t_rmvpe1 - t_cv1) * 1000.0
        self.stats.last_rvc_ms = (t_rvc1 - t_rmvpe1) * 1000.0
        return result

    def _safe_process_streaming_16k(self, audio16: NDArrayF32) -> NDArrayF32 | None:
        """v0.6.8 — wrap `_process_streaming_16k` so a transient
        ORT / CUDA / numerical error drops the chunk instead of killing
        the engine.

        Returns the inferred chunk, or `None` if inference failed.
        Caller must check for `None` and skip the playback write — the
        engine's main loop does this with `continue`.

        First three failures log to `stats.last_error` with the
        exception type + message. After that the counter increments
        silently except for every 100th hit (to keep the diagnostic
        line refreshing without spamming).

        Pulled out of `_run_loop` so the failure path is unit-testable
        without spinning up the full audio thread.
        """
        try:
            return self._process_streaming_16k(audio16)
        except Exception as e:
            self.stats.dropped_chunks += 1
            n = self.stats.dropped_chunks
            if n <= 3:
                self.stats.last_error = f"inference dropped chunk #{n}: {type(e).__name__}: {e}"
            elif n % 100 == 0:
                self.stats.last_error = (
                    f"inference still dropping chunks (total #{n}): {type(e).__name__}: {e}"
                )
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

        full_out = self._infer(model_input)

        # Map the trim from input space to output space proportionally —
        # the model is roughly 1:1 in time, but RVC trims a few samples at
        # the boundaries. Compute the per-sample ratio defensively.
        in_len = model_input.shape[0]
        out_len = full_out.shape[0]
        ratio = out_len / max(in_len, 1)
        # Drop the leading "context" portion in the model output. Keep the
        # last `ctx_drop_out` samples — sized to match SOLA's contract:
        #   chunk_n + cf + search   when SOLA is enabled (rc5 — gives the
        #                           alignment search positional slack so
        #                           emit length stays constant)
        #   chunk_n + cf            when SOLA is disabled (legacy path —
        #                           emit length variable, no slack needed)
        # The pre-rc5 implementation always trimmed to chunk_n + cf even
        # with SOLA enabled; the search then ate samples from the input
        # to find alignment, shrinking the emit. See
        # `docs/16-audit/11-rc4-postmortem.md` for why that was wrong.
        sola_search = self._sola_input_cfg.search_samples if self._sola is not None else 0
        ctx_drop_in = max(history_len - cf - sola_search, 0)
        ctx_drop_out = round(ctx_drop_in * ratio)
        emitted_region = full_out[ctx_drop_out:]

        if self._sola is not None:
            return self._sola.process(emitted_region)
        # SOLA disabled — emit raw, expect chunk-boundary clicks for short chunks.
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

    # ---- realtime loop ------------------------------------------------------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._ensure_sessions()
        # v0.6.9: warm cv + rmvpe + rvc cuDNN caches with synthetic chunks so
        # the first real chunks don't blow the chunk-time budget. Live
        # diagnostic showed `max_total_ms=456 ms` (1.8x the 250 ms budget) on
        # cold-start chunks — usually the first 3-5 chunks of a session.
        self._warmup_realtime_pipeline()
        # v0.5.0: optionally pre-warm every voice so swaps are instant from
        # the first press of `p`. Costs ~6 s for a 10-voice library.
        if self.cfg.eager_warmup:
            n = self.warmup_voice_library()
            print(f"[engine] eager-warmed {n} voice models (instant swaps now)")
        self.stats.running = True

        # v0.7.0-rc7 — disable Python GC during the engine's lifetime to
        # eliminate generational-collection pauses from the inference
        # tail. The rc6 diag dump (`docs/16-audit/12-rc5-writer-jitter-
        # probe.md` follow-up) showed inference p50=66 ms / p99=98 ms /
        # max=110 ms — a 32 ms tail that owns the writer-cadence
        # variance. mic_read p99 was 160 ms (≈ chunk_seconds, no
        # variance), enqueue_lag p99 was 0.06 ms (sub-ms, clean). Tail
        # is in inference itself.
        #
        # Numpy arrays in the engine hot path are reference-counted —
        # they free as soon as their reference count drops, no GC
        # required. The only thing GC handles is cyclic references,
        # which are rare in this code path. Disabling GC for the
        # session prevents the periodic ~10–30 ms collection pauses
        # that fire on the GIL during inference. stop() re-enables
        # and runs one final collection to clean up any cyclic refs
        # that accumulated.
        self._gc_was_enabled_before_start = gc.isenabled()
        if self._gc_was_enabled_before_start:
            gc.disable()

        self._thread = threading.Thread(target=self._run_loop, name="vcclient-engine", daemon=True)
        self._thread.start()

    def _warmup_realtime_pipeline(self, n_chunks_per_shape: int = 4) -> None:
        """v0.6.9 — pre-run synthetic chunks through the *full* realtime
        pipeline (cv → rmvpe → rvc) so cuDNN's algo cache is populated for
        the actual shapes we feed at runtime. `RvcSessionPool.warmup` only
        warms the rvc session; the cv and rmvpe sessions still cold-start
        the first few real chunks otherwise.

        v0.7.0-rc9 — extended to pre-warm EVERY unique input length
        soxr's stream resampler can emit, not just the nominal one. The
        rc8 tail-chunk capture pinned the inference p99=96 ms / max=110 ms
        spike to a shape mismatch: pre-rc9 warmup ran `_infer` with
        `chunk_n = chunk_seconds * 16000 = 2400` samples. The realtime
        path calls `_infer` with `model_input.shape[0] = history_len +
        audio16_len`, where `audio16_len` is whatever
        `_StreamResampler(48k → 16k).process(7200)` emits. Soxr
        alternates between two specific values (1957 / 2447 in alireza's
        QuadCast 2 S session, plus the typical 2400) — every chunk with
        a non-cached shape costs cuDNN a fallback slow path, ~80 ms
        inference vs ~40 ms cached. See
        `docs/16-audit/12-rc5-writer-jitter-probe.md` and the rc8
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
            # mic_rate == internal — no resample, audio16_len is fixed.
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
                    # If one shape fails, try the rest — the realtime
                    # path's `_safe_process_streaming_16k` will catch
                    # any inference failure that survives warmup.
                    break

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        self.stats.running = False

        # v0.7.0-rc7 — restore GC to its prior state and run one
        # collection to free any cyclic references that accumulated
        # during the session. If GC was already disabled before this
        # engine started (nested case), leave it disabled.
        if self._gc_was_enabled_before_start:
            gc.enable()
            gc.collect()
            self._gc_was_enabled_before_start = False

    def _assert_sink_loaded(self) -> None:
        """v0.6.4 — refuse to start if `cfg.sink_name` isn't a loaded
        PipeWire sink.

        Without this guard, `pw-cat --target=…` and `pacat --device=…`
        treat the named sink as a hint: if it's missing, the session
        manager silently routes the stream to the *default* sink
        (typically laptop speakers). The engine's playback subprocess
        starts cleanly, exits 0, no stderr — and your transformed
        voice plays out of the speakers instead of the virtual mic.
        See docs/10-monitor-leak-diag.md for the full forensic trail.

        If `pactl` itself is unavailable or hangs, we skip the check
        rather than refusing to start (best-effort guard).
        """
        try:
            result = subprocess.run(
                ["pactl", "list", "short", "sinks"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return
        if result.returncode != 0:
            return
        loaded = [line.split("\t")[1] for line in result.stdout.splitlines() if "\t" in line]
        if self.cfg.sink_name not in loaded:
            raise RuntimeError(
                f"PipeWire sink {self.cfg.sink_name!r} is not loaded — refusing to start.\n"
                f"  loaded sinks: {loaded}\n"
                f"  fix: run `woys pw setup` to load the virtual sink, "
                f"or correct `sink_name` in ~/.config/woys/config.toml."
            )

    def _open_pacat(self) -> subprocess.Popen[bytes]:
        """Spawn the playback subprocess targeting the named virtual sink.

        v0.5.2: prefers `pw-cat` (PipeWire-native, no underruns under
        bursty 250 ms writes) over `pacat` (PulseAudio compat, drains the
        prebuf/tlength buffer near zero on every chunk → underrun storm).
        Falls back to pacat only if pw-cat is missing.

        v0.6.4: pre-flights sink existence — see `_assert_sink_loaded`.
        Without that guard, `--target` / `--device` silently fall back
        to the default sink when the named sink is missing.

        The retained name `_open_pacat` is historical — the watchdog and
        writer threads don't care which binary is on the other side, only
        that it accepts raw float32le on stdin.
        """
        self._assert_sink_loaded()
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
                return subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )

        pacat = shutil.which("pacat")
        if pacat is None:
            raise RuntimeError(
                "neither pw-cat nor pacat found — install pipewire and pipewire-pulse"
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
        return subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

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
        queue the engine has out-paced the writer/sink — bump the
        queue_full counter (xrun proxy) and drop the chunk rather than
        block the engine main loop.
        """
        q = self._writer_queue
        if q is None:
            return
        try:
            q.put_nowait(payload)
        except queue.Full:
            self.stats.queue_full_events += 1

    def _writer_loop(self) -> None:
        """Daemon thread: drains _writer_queue into pacat.stdin.

        Decouples the engine main loop from blocking pipe writes (Brief §3
        Fix 2). On BrokenPipeError / OSError the watchdog is signalled to
        respawn pacat; the writer keeps running and reattaches to the new
        handle on the next iteration.
        """
        # Best-effort thread-local affinity so the writer doesn't ping-pong
        # cores away from the main engine thread.
        self._apply_thread_priority(label="writer")
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
                self.stats.last_error = f"pacat write failed ({type(e).__name__}); respawning"
                self._pacat_dead_event.set()
                # Brief pause so the watchdog has time to respawn before
                # the next iteration tries to write again.
                time.sleep(0.02)
                continue
            now = time.perf_counter()
            if self._last_writer_ts is not None:
                interval_ms = (now - self._last_writer_ts) * 1000.0
                self.stats._writer_intervals_ms.append(interval_ms)
                # Update jitter (std dev) periodically — every ~16 chunks
                # is sufficient resolution for a TUI readout.
                if len(self.stats._writer_intervals_ms) >= 16 and (
                    self.stats.chunks_processed % 16 == 0
                ):
                    arr = np.array(self.stats._writer_intervals_ms, dtype=np.float32)
                    self.stats.writer_jitter_ms = float(arr.std())
            self._last_writer_ts = now

    def _stderr_reader_loop(self, proc: subprocess.Popen[bytes]) -> None:
        """Daemon thread: parses pacat -v stderr for underrun tokens.

        Bound to a single pacat process — when it exits, readline returns
        b'' and the thread terminates. The watchdog spawns a new reader
        for the replacement process.
        """
        if proc.stderr is None:
            return
        try:
            for raw in proc.stderr:
                if not raw:
                    break
                # pacat -v prints lines like "Stream underrun.\n" exactly.
                # We match case-insensitively in case the wording shifts
                # across PulseAudio versions.
                line = raw.decode("utf-8", errors="replace")
                if "underrun" in line.lower():
                    self.stats.xruns += 1
        except (ValueError, OSError):
            # Pipe closed mid-read during shutdown — expected.
            return

    def _watchdog_loop(self) -> None:
        """Daemon thread: respawns pacat if it dies mid-session (Brief §3 Fix 3).

        Polls every `pacat_watchdog_interval_s` (50 ms by default). On dead
        process: opens a replacement under `_pacat_lock`, swaps the handle,
        spawns a fresh stderr reader for the new process, and increments
        `pacat_restarts`. Recovery target ≤ 100 ms.
        """
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
                continue  # still alive
            # Respawn.
            try:
                new_proc = self._open_pacat()
            except Exception as e:
                self.stats.last_error = f"watchdog respawn failed: {type(e).__name__}: {e}"
                # Back off a bit before retrying so we don't spin.
                time.sleep(0.5)
                continue
            with self._pacat_lock:
                # Discard the dead handle (caller already detected death).
                self._pacat_proc = new_proc
            self.stats.pacat_restarts += 1
            self.stats.last_error = f"pacat respawned (restarts={self.stats.pacat_restarts})"
            # Spawn a fresh stderr reader bound to the new process. The old
            # reader thread will exit on its own once the dead pipe EOFs.
            stderr_t = threading.Thread(
                target=self._stderr_reader_loop,
                args=(new_proc,),
                name="vcclient-pacat-stderr",
                daemon=True,
            )
            stderr_t.start()
            self._stderr_thread = stderr_t

    def _apply_thread_priority(self, *, label: str) -> None:
        """Pin to `cpu_affinity_core` and optionally raise priority.

        Called from inside whichever thread should be pinned; affinity /
        nice are per-thread on Linux. Permission failures degrade to a
        warning in `last_error` rather than aborting the engine.
        """
        if self.cfg.cpu_affinity_core is not None:
            try:
                os.sched_setaffinity(0, {self.cfg.cpu_affinity_core})
            except (OSError, AttributeError) as e:
                self.stats.last_error = f"affinity[{label}] failed ({type(e).__name__}: {e})"
        if self.cfg.realtime_priority:
            try:
                os.nice(-10)
            except (OSError, PermissionError) as e:
                self.stats.last_error = (
                    f"realtime_priority[{label}] denied ({type(e).__name__}); needs CAP_SYS_NICE"
                )

    def _run_loop(self) -> None:
        import sounddevice as sd

        chunk_mic = int(self.cfg.mic_rate * self.cfg.chunk_seconds)
        # Reset SOLA buffers so a stop/start cycle doesn't leak stale audio.
        self.reset_streaming_state()
        # v0.6.7 — fresh stateful resamplers. Built per `(src, dst)` pair so
        # filter state survives across chunks; hot-swapped if the model SR
        # changes mid-session (see `_maybe_swap_model`).
        self._resampler_in = _StreamResampler(self.cfg.mic_rate, 16_000)
        self._resampler_out = _StreamResampler(self._rvc_output_sr, self.cfg.sink_rate)

        # v0.5.2: pin engine main thread + bump priority if requested.
        self._apply_thread_priority(label="engine")

        monitor_stream = None
        try:
            # v0.5.2: open pacat under the lock + start writer/stderr/watchdog
            # threads before the first inference. The writer queue is sized
            # so the engine can sprint ahead of pacat for ~2 s without
            # blocking, then pressures back through queue_full_events.
            initial_proc = self._open_pacat()
            with self._pacat_lock:
                self._pacat_proc = initial_proc
            # v0.6.7 part 3 — prime the playback backend's stream buffer
            # with `prime_silence_seconds` of zeros before any real audio.
            # Without priming, the buffer steady-state oscillates 0 → chunk
            # → 0 → chunk; engine writer jitter (~30 ms std) pushes the
            # buffer to 0 frequently → pacat reports xruns and outputs one
            # PipeWire quantum (~21-43 ms) of silence per underrun. With a
            # 1x chunk pre-roll, the buffer floor lifts above 0 and only
            # outsized jitter (>chunk_seconds) can underrun.
            # Trade-off: this adds prime_silence_seconds to mic-to-app
            # wall-clock latency. Default 0.25 s matches chunk_seconds —
            # smallest pre-roll that fully bridges typical jitter.
            prime_n = int(self.cfg.sink_rate * self.cfg.prime_silence_seconds)
            if prime_n > 0 and initial_proc.stdin is not None:
                silence = np.zeros(prime_n * self.cfg.output_channels, dtype=np.float32).tobytes()
                with contextlib.suppress(BrokenPipeError, OSError):
                    initial_proc.stdin.write(silence)
                    initial_proc.stdin.flush()
            self._writer_queue = queue.Queue(maxsize=self.cfg.pacat_writer_queue_size)
            self._last_writer_ts = None
            self._pacat_dead_event.clear()
            self._writer_thread = threading.Thread(
                target=self._writer_loop, name="vcclient-pacat-writer", daemon=True
            )
            self._writer_thread.start()
            self._stderr_thread = threading.Thread(
                target=self._stderr_reader_loop,
                args=(initial_proc,),
                name="vcclient-pacat-stderr",
                daemon=True,
            )
            self._stderr_thread.start()
            self._watchdog_thread = threading.Thread(
                target=self._watchdog_loop, name="vcclient-pacat-watchdog", daemon=True
            )
            self._watchdog_thread.start()

            in_stream = sd.InputStream(
                samplerate=self.cfg.mic_rate,
                channels=self.cfg.channels,
                blocksize=chunk_mic,
                dtype="float32",
                device=self.cfg.input_device,
            )
            if self.cfg.monitor:
                # Best-effort self-monitor stream; failures here don't stop the engine.
                try:
                    monitor_stream = sd.OutputStream(
                        samplerate=self.cfg.sink_rate,
                        channels=self.cfg.channels,
                        dtype="float32",
                    )
                    monitor_stream.start()
                except Exception as e:
                    self.stats.last_error = f"monitor: {type(e).__name__}: {e}"
                    monitor_stream = None

            # v0.7.0-rc4 — input-gate hysteresis state. The gate must
            # observe ≥`input_gate_hysteresis_ms` of continuously-below
            # threshold input before it fires; voice transients (brief
            # dips between syllables, plosive onsets, fricative onsets)
            # no longer trigger zero-emission. `gate_below_since` is
            # the perf_counter time of the first below-threshold sample
            # in the current run (None when above threshold).
            gate_below_since: float | None = None
            hysteresis_s = max(0.0, self.cfg.input_gate_hysteresis_ms / 1000.0)
            gate_thresh = (
                10.0 ** (self.cfg.input_gate_dbfs / 20.0)
                if self.cfg.input_gate_dbfs > -120.0
                else 0.0
            )

            with in_stream:
                while not self._stop_event.is_set():
                    # v0.4.1: pick up any queued model swap before reading
                    # the next mic chunk. Owns _rvc on this thread, so no
                    # race with _infer below.
                    self._maybe_swap_model()
                    # v0.7.0-rc4 — capture the overflow flag PortAudio
                    # returns when its internal ring buffer overran
                    # since the previous read. Pre-rc4 this was tuple-
                    # unpacked into `_` and lost; lens 01 of the audit
                    # flagged it as a silent mic-side drop site
                    # invisible to every existing counter.
                    #
                    # v0.7.0-rc6 — wrapped with timing so we can attribute
                    # producer-side cadence variance to the mic read vs
                    # processing vs handoff. Steady-state mic_read_ms
                    # should hover near chunk_seconds * 1000; variance
                    # reflects ALSA period scheduling + USB iso jitter.
                    t_mic_pre = time.perf_counter()
                    data, overflowed = in_stream.read(chunk_mic)
                    mic_read_ms = (time.perf_counter() - t_mic_pre) * 1000.0
                    self.stats.last_mic_read_ms = mic_read_ms
                    self.stats._recent_mic_read_ms.append(mic_read_ms)
                    if overflowed:
                        self.stats.input_overflows += 1
                    audio = data.reshape(-1).astype(np.float32, copy=False)

                    # v0.5.1: software input pre-attenuation. Default 0 dB
                    # is a no-op (skip the multiply). Negative values trim
                    # hot mics so RVC doesn't amplify clipping as harsh
                    # distortion. RMS is measured AFTER the gain so the
                    # stat reflects what the model actually sees.
                    if self.cfg.input_gain_db != 0.0:
                        audio = audio * np.float32(10.0 ** (self.cfg.input_gain_db / 20.0))

                    rms = float(np.sqrt(np.mean(audio**2)))
                    self.stats.last_input_rms = rms

                    # v0.7.0-rc4 — gate with hysteresis. Below threshold
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
                    # crossfade — useful for A/B perf comparisons.
                    out_native = self._safe_process_streaming_16k(audio16)
                    inf_ms = (time.perf_counter() - t_inf) * 1000

                    if out_native is None or out_native.shape[0] == 0:
                        # `None`: inference raised — `_safe_*` already
                        # bumped `stats.dropped_chunks` and updated
                        # `stats.last_error`. Skip the write; SOLA's
                        # held-back tail covers the gap on resume.
                        # `shape[0] == 0`: first-chunk warmup or
                        # resampler buffer fill — emit nothing yet.
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
                        # write — the next chunk will produce extra samples.
                        continue

                    # v0.5.2: hand off to writer thread (non-blocking enqueue).
                    # The watchdog respawns pacat if it dies — main loop
                    # never raises out of the loop on a transient pacat fault.
                    #
                    # v0.7.0-rc6 — wrapped with timing. enqueue_lag_ms
                    # covers _to_sink_bytes (numpy convert) + put_nowait
                    # (queue insert). Should be sub-ms in steady state;
                    # spikes mean GC pause / GIL contention / queue
                    # backpressure (which would also bump
                    # `queue_full_events`).
                    t_enq_pre = time.perf_counter()
                    self._enqueue_chunk(self._to_sink_bytes(out48))
                    enq_lag_ms = (time.perf_counter() - t_enq_pre) * 1000.0
                    self.stats.last_enqueue_lag_ms = enq_lag_ms
                    self.stats._recent_enqueue_lag_ms.append(enq_lag_ms)

                    # v0.7.0-rc5 — pull SOLA's threshold-fallback count
                    # into engine stats. The rc4 `sola_drain_ms` (zero-
                    # pad bookkeeping) is gone because the pad itself is
                    # gone — SOLA emits constant-size chunks now. A
                    # non-zero fallback count means the alignment search
                    # is giving up (peak corr below threshold); it's a
                    # diagnostic, not a cuts driver.
                    if self._sola is not None:
                        self.stats.sola_fallback_count = self._sola.fallback_count

                    # Optional self-monitor → host default output.
                    if monitor_stream is not None:
                        with contextlib.suppress(Exception):
                            monitor_stream.write(out48.reshape(-1, 1))

                    total_ms = (time.perf_counter() - t_total) * 1000
                    self.stats.chunks_processed += 1
                    self.stats.last_inference_ms = inf_ms
                    self.stats.last_total_ms = total_ms
                    if inf_ms > self.stats.max_inference_ms:
                        self.stats.max_inference_ms = inf_ms
                    if total_ms > self.stats.max_total_ms:
                        self.stats.max_total_ms = total_ms
                    if total_ms > self.cfg.chunk_seconds * 1000.0:
                        self.stats.late_chunks += 1
                        # v0.6.9 round 5 — capture per-stage breakdown for
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
                    # v0.7.0-rc8 — tail-chunk capture, gated on inference
                    # time alone (not total_ms). Fires when inf_ms is more
                    # than 2× the running p50 of `_recent_inference`, which
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
                    self.stats._recent_inference.append(inf_ms)
                    self.stats._recent_total.append(total_ms)
                    if self.stats._recent_inference:
                        self.stats.avg_inference_ms = sum(self.stats._recent_inference) / len(
                            self.stats._recent_inference
                        )
                        self.stats.avg_total_ms = sum(self.stats._recent_total) / len(
                            self.stats._recent_total
                        )
        except Exception as e:
            self.stats.last_error = f"{type(e).__name__}: {e}"
            self.stats.running = False
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
            if monitor_stream is not None:
                try:
                    monitor_stream.stop()
                    monitor_stream.close()
                except Exception:
                    pass
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
            for t in (self._writer_thread, self._watchdog_thread, self._stderr_thread):
                if t is not None and t.is_alive():
                    t.join(timeout=0.5)
            self._writer_thread = None
            self._watchdog_thread = None
            self._stderr_thread = None
            self._writer_queue = None
