#!/usr/bin/env python
"""v0.10.x synthetic engine harness.

Drives the realtime engine end-to-end with a deterministic, file-free
audio source so the writer-jitter investigation can run reproducibly
without a microphone or a real Telegram session.

What the harness does
---------------------
1. Monkey-patches `sounddevice.InputStream` BEFORE the engine constructs
   one. The mock returns deterministic float32 audio chunks at the
   engine's `chunk_seconds` cadence, so the engine sees realistic
   producer-side timing without depending on mic hardware.
2. Starts the engine with the active config (matches `woys diag`).
3. Runs for `--duration` seconds.
4. Dumps the full per-stage timing percentiles, writer-interval
   percentiles, shape-coverage diff, and player-side underrun counters
   to `--out` as JSON. Designed to be diffed across runs / branches.

The signal is a 1.5-second loop of:
  - 0.500 s voiced (sine + low pink noise) at RMS ≈ 0.10  — gate stays on
  - 0.200 s unvoiced (white noise) at RMS ≈ 0.05         — pitch path goes silent
  - 0.100 s silence                                       — gate eventually fires
  - 0.700 s voiced (different pitch)
  Loop frequency: 0.667 Hz; covers gate transitions + RMVPE pitch
  changes, which are two of the brief's candidate jitter sources.

Usage
-----
  ./scripts/v010_harness.py                       # 60s, default chunk
  ./scripts/v010_harness.py --duration 300        # 5-minute run (acceptance gate)
  ./scripts/v010_harness.py --duration 300 --out /tmp/v010_run.json
  ./scripts/v010_harness.py --duration 30 --pyspy /tmp/profile.svg

The optional --pyspy flag wraps the run in py-spy record so the
flamegraph captures every Python frame across the synthetic harness.

Output schema (stable for diff-based regression)
------------------------------------------------
  {"version": "v0.10.x-harness-1",
   "duration_s": 300,
   "chunk_seconds": 0.15,
   "config": {...EngineConfig fields, sanitized...},
   "stats": {
     "chunks_processed": 1972,
     "player_underruns": 12,
     "writer_jitter_ms_stddev": 78.4,
     "writer_interval_p50_ms": 149.8,
     "writer_interval_p95_ms": 198.5,
     "writer_interval_p99_ms": 232.7,
     "writer_jitter_p99_ms": 82.7,            # p99 - chunk_ms_target
     "inference_p50_ms": 51.2,
     "inference_p99_ms": 119.6,
     "cv_p50_ms": 6.0, "cv_p99_ms": 12.4,
     "rmvpe_p50_ms": 8.4, "rmvpe_p99_ms": 17.9,
     "rvc_p50_ms": 35.6, "rvc_p99_ms": 88.2,
     "warmup_audio16_lens": [...], "runtime_audio16_lens": [...],
     "unwarmed_shapes": [...]
   }
  }
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "src" / "server") not in sys.path:
    sys.path.insert(0, str(REPO / "src" / "server"))


# ---- deterministic test signal ----------------------------------------------


def _build_signal(duration_s: float, sample_rate: int, *, seed: int = 42) -> np.ndarray:
    """One contiguous mono float32 buffer covering `duration_s` seconds.

    Loops a 1.5 s pattern that exercises:
      [0.0–0.5)  voiced sine 220 Hz + 1/f noise   (RMS ≈ 0.10)
      [0.5–0.7)  white noise                      (RMS ≈ 0.05)
      [0.7–0.8)  silence
      [0.8–1.5)  voiced sine 330 Hz + 1/f noise   (RMS ≈ 0.12)

    Pitch alternation forces the RMVPE network through different f0
    bins (candidate #4 in the v0.10 brief). Voiced/unvoiced/silence
    transitions force the input gate state machine through every
    branch. Deterministic via seeded RNG so runs are diffable.
    """
    rng = np.random.default_rng(seed)
    pattern_s = 1.5
    n_pattern = int(round(pattern_s * sample_rate))
    pattern = np.zeros(n_pattern, dtype=np.float32)

    def _voiced(start_s: float, end_s: float, freq_hz: float, rms: float) -> None:
        i0 = int(round(start_s * sample_rate))
        i1 = int(round(end_s * sample_rate))
        n = i1 - i0
        if n <= 0:
            return
        t = np.arange(n, dtype=np.float32) / sample_rate
        sine = np.sin(2.0 * np.pi * freq_hz * t).astype(np.float32)
        # Pink-ish noise via per-sample Gaussian filtered through a 1-pole.
        noise = rng.standard_normal(n).astype(np.float32) * 0.5
        accum = 0.0
        for i in range(n):
            accum = 0.97 * accum + 0.03 * noise[i]
            noise[i] = accum
        mix = 0.7 * sine + 0.3 * (noise / max(noise.std(), 1e-6))
        # Normalize to target RMS.
        cur = np.sqrt(np.mean(mix.astype(np.float64) ** 2))
        if cur > 0:
            mix = mix * np.float32(rms / cur)
        pattern[i0:i1] = mix

    def _white(start_s: float, end_s: float, rms: float) -> None:
        i0 = int(round(start_s * sample_rate))
        i1 = int(round(end_s * sample_rate))
        n = i1 - i0
        if n <= 0:
            return
        x = rng.standard_normal(n).astype(np.float32)
        cur = np.sqrt(np.mean(x.astype(np.float64) ** 2))
        if cur > 0:
            x = x * np.float32(rms / cur)
        pattern[i0:i1] = x

    _voiced(0.0, 0.5, freq_hz=220.0, rms=0.10)
    _white(0.5, 0.7, rms=0.05)
    # 0.7-0.8 stays as zeros (silence)
    _voiced(0.8, 1.5, freq_hz=330.0, rms=0.12)

    # Tile out to the requested duration.
    n_total = int(round(duration_s * sample_rate))
    n_loops = math.ceil(n_total / n_pattern)
    out = np.tile(pattern, n_loops)[:n_total]
    return out.astype(np.float32, copy=False)


# ---- mock sounddevice.InputStream -------------------------------------------


class _MockInputStream:
    """Drop-in for `sounddevice.InputStream` used by the engine.

    Reads paced at `chunk_seconds` from the pre-generated buffer so the
    engine experiences mic-side cadence comparable to a real device.
    Returns `(data, overflowed=False)` from `.read(chunk_mic)`.
    """

    def __init__(
        self,
        *,
        samplerate: int,
        channels: int,
        blocksize: int,
        dtype: str = "float32",
        device: object = None,
        signal: np.ndarray,
        chunk_seconds: float,
    ) -> None:
        self.samplerate = samplerate
        self.channels = channels
        self.blocksize = blocksize
        self.dtype = dtype
        self._signal = signal.reshape(-1)
        self._chunk_seconds = chunk_seconds
        self._cursor = 0
        self._next_read_at: float | None = None

    # Context manager protocol; engine uses `with in_stream:`.
    def __enter__(self) -> "_MockInputStream":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, n_frames: int) -> tuple[np.ndarray, bool]:
        # Block until the next chunk_seconds boundary so the engine
        # paces the same way it would against a real ALSA device.
        now = time.perf_counter()
        if self._next_read_at is None:
            self._next_read_at = now + self._chunk_seconds
        elif now < self._next_read_at:
            time.sleep(self._next_read_at - now)
            self._next_read_at = self._next_read_at + self._chunk_seconds
        else:
            # Fell behind — schedule the next one from "now" rather than
            # bursting to catch up. Realistic mic behavior.
            self._next_read_at = time.perf_counter() + self._chunk_seconds

        # Pull n_frames from the signal, looping if we run out.
        end = self._cursor + n_frames
        if end <= self._signal.shape[0]:
            chunk = self._signal[self._cursor : end]
            self._cursor = end
        else:
            n1 = self._signal.shape[0] - self._cursor
            n2 = n_frames - n1
            chunk = np.concatenate(
                [self._signal[self._cursor :], self._signal[: n2 % self._signal.shape[0]]]
            )
            self._cursor = n2 % self._signal.shape[0]
        # Engine expects (n, channels) shape; cast to that.
        if self.channels == 1:
            return chunk.reshape(-1, 1).astype(np.float32, copy=False), False
        return np.tile(chunk[:, None], (1, self.channels)).astype(np.float32, copy=False), False


# ---- main harness driver ----------------------------------------------------


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(values, p))


def _stats_dict(engine: Any, *, chunk_ms_target: float) -> dict[str, Any]:
    s = engine.stats
    inf = s.inference_samples()
    cv = s.cv_samples_ms()
    rm = s.rmvpe_samples_ms()
    rvc = s.rvc_samples_ms()
    rvc_pre = s.rvc_pre_samples_ms()
    rvc_run = s.rvc_run_samples_ms()
    rvc_post = s.rvc_post_samples_ms()
    enq = s.enqueue_lag_samples_ms()
    mic = s.mic_read_samples_ms()
    wri = s.writer_interval_samples_ms()

    def _pct(values: list[float], p: float) -> float:
        return _percentile(values, p)

    return {
        "chunks_processed": int(s.chunks_processed),
        "player_underruns": int(s.player_underruns),
        "player_restarts": int(s.player_restarts),
        "xruns": int(s.xruns),
        "queue_full_events": int(s.queue_full_events),
        "dropped_chunks": int(s.dropped_chunks),
        "input_overflows": int(s.input_overflows),
        "gated_chunks": int(s.gated_chunks),
        "nan_chunks": int(s.nan_chunks),
        "sola_fallback_count": int(s.sola_fallback_count),
        "late_chunks": int(s.late_chunks),
        "writer_jitter_ms_stddev": float(s.writer_jitter_ms),
        "writer_interval_p50_ms": _pct(wri, 50),
        "writer_interval_p95_ms": _pct(wri, 95),
        "writer_interval_p99_ms": _pct(wri, 99),
        "writer_interval_max_ms": float(max(wri)) if wri else float("nan"),
        "writer_jitter_p99_ms": max(0.0, _pct(wri, 99) - chunk_ms_target),
        "inference_p50_ms": _pct(inf, 50),
        "inference_p95_ms": _pct(inf, 95),
        "inference_p99_ms": _pct(inf, 99),
        "inference_max_ms": float(s.max_inference_ms),
        "inference_avg_ms": float(s.avg_inference_ms),
        "cv_p50_ms": _pct(cv, 50),
        "cv_p95_ms": _pct(cv, 95),
        "cv_p99_ms": _pct(cv, 99),
        "rmvpe_p50_ms": _pct(rm, 50),
        "rmvpe_p95_ms": _pct(rm, 95),
        "rmvpe_p99_ms": _pct(rm, 99),
        "rvc_p50_ms": _pct(rvc, 50),
        "rvc_p95_ms": _pct(rvc, 95),
        "rvc_p99_ms": _pct(rvc, 99),
        "rvc_pre_p50_ms": _pct(rvc_pre, 50),
        "rvc_pre_p99_ms": _pct(rvc_pre, 99),
        "rvc_run_p50_ms": _pct(rvc_run, 50),
        "rvc_run_p95_ms": _pct(rvc_run, 95),
        "rvc_run_p99_ms": _pct(rvc_run, 99),
        "rvc_post_p50_ms": _pct(rvc_post, 50),
        "rvc_post_p99_ms": _pct(rvc_post, 99),
        "mic_read_p50_ms": _pct(mic, 50),
        "mic_read_p99_ms": _pct(mic, 99),
        "enqueue_lag_p50_ms": _pct(enq, 50),
        "enqueue_lag_p99_ms": _pct(enq, 99),
        "warmup_audio16_lens": sorted(s.warmup_audio16_lens),
        "runtime_audio16_lens": sorted(s.unique_audio16_lens),
        "unwarmed_shapes": sorted(s.unique_audio16_lens - s.warmup_audio16_lens),
        "n_inference_samples": len(inf),
        "n_writer_samples": len(wri),
    }


def _run_engine_synthetic(
    *,
    duration_s: float,
    out_path: Path | None,
    enable_sola: bool,
    chunk_seconds: float | None,
    inference_subprocess: bool,
) -> dict[str, Any]:
    """Run the engine for `duration_s` against the synthetic signal,
    return the stats dict (also written to `out_path` if provided)."""
    # Late imports so monkey-patch happens before engine constructs the
    # InputStream.
    from audio.engine import EngineConfig, RealtimeEngine
    from tui.config import load_config

    cfg = load_config()
    engine_cfg = EngineConfig(
        f0_up_key=cfg.f0_up_key,
        sid=cfg.sid,
        chunk_seconds=chunk_seconds if chunk_seconds is not None else cfg.chunk_seconds,
        sink_name=cfg.sink_name,
        monitor=False,  # never self-monitor — would block on default device
        output_latency_ms=cfg.output_latency_ms,
        output_process_time_ms=cfg.output_process_time_ms,
        embedder=cfg.embedder,
        sola_enabled=enable_sola,
        sola_crossfade_ms=cfg.sola_crossfade_ms,
        sola_search_ms=cfg.sola_search_ms,
        sola_context_ms=cfg.sola_context_ms,
        input_gain_db=cfg.input_gain_db,
        input_gate_dbfs=cfg.input_gate_dbfs,
        input_gate_hysteresis_ms=cfg.input_gate_hysteresis_ms,
        prefer_pw_cat=cfg.prefer_pw_cat,
        prefer_native_pw=cfg.prefer_native_pw,
        prefer_native_pw_buffer_ms=cfg.prefer_native_pw_buffer_ms,
    )
    engine_cfg.inference_subprocess = inference_subprocess
    rvc_path = Path(cfg.rvc_model) if cfg.rvc_model and Path(cfg.rvc_model).exists() else None
    if rvc_path is not None:
        engine_cfg.rvc_model = rvc_path

    chunk_mic_frames = int(round(engine_cfg.chunk_seconds * engine_cfg.mic_rate))
    print(
        f"[harness] generating {duration_s:.0f}s signal at {engine_cfg.mic_rate} Hz "
        f"(chunk_mic={chunk_mic_frames} frames)",
        file=sys.stderr,
    )
    signal = _build_signal(duration_s + 5.0, engine_cfg.mic_rate)

    # Monkey-patch sounddevice.InputStream BEFORE the engine constructs one.
    import sounddevice as sd

    def _make_mock(*, samplerate: int, channels: int, blocksize: int, **kwargs: Any) -> _MockInputStream:
        return _MockInputStream(
            samplerate=samplerate,
            channels=channels,
            blocksize=blocksize,
            signal=signal,
            chunk_seconds=engine_cfg.chunk_seconds,
        )

    sd.InputStream = _make_mock  # type: ignore[assignment]

    engine = RealtimeEngine(engine_cfg)
    print(
        f"[harness] starting engine: prefer_native_pw={engine_cfg.prefer_native_pw} "
        f"buffer_ms={engine_cfg.prefer_native_pw_buffer_ms} chunk_seconds={engine_cfg.chunk_seconds} "
        f"inference_subprocess={engine_cfg.inference_subprocess}",
        file=sys.stderr,
    )
    engine.start()

    # Wait for warmup to settle (engine.start() returns immediately;
    # warmup happens on the engine thread before the run loop spins up).
    # Watch `chunks_processed` to confirm the loop is running.
    deadline = time.perf_counter() + duration_s
    last_print = time.perf_counter()
    try:
        while time.perf_counter() < deadline:
            time.sleep(1.0)
            now = time.perf_counter()
            if now - last_print >= 30.0:
                s = engine.stats
                print(
                    f"[harness] t={now - (deadline - duration_s):6.1f}s  "
                    f"chunks={s.chunks_processed:5d}  "
                    f"under={s.player_underruns:4d}  "
                    f"jitter_sigma={s.writer_jitter_ms:5.1f}ms  "
                    f"inf_avg={s.avg_inference_ms:5.1f}ms",
                    file=sys.stderr,
                )
                last_print = now
            if engine.stats.last_error and "respawned" not in (engine.stats.last_error or ""):
                # Surface fatal errors immediately; shorten the run.
                print(
                    f"[harness] engine error mid-run: {engine.stats.last_error}", file=sys.stderr
                )
    finally:
        engine.stop(timeout=3.0)

    chunk_ms_target = engine_cfg.chunk_seconds * 1000.0
    stats = _stats_dict(engine, chunk_ms_target=chunk_ms_target)
    out = {
        "version": "v0.10.x-harness-1",
        "duration_s": duration_s,
        "chunk_seconds": engine_cfg.chunk_seconds,
        "chunk_ms_target": chunk_ms_target,
        "config": {
            k: v
            for k, v in asdict(engine_cfg).items()
            if not k.startswith("_") and isinstance(v, (str, int, float, bool, type(None)))
        },
        "stats": stats,
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2, default=str))
        print(f"[harness] wrote stats → {out_path}", file=sys.stderr)
    return out


def _print_summary(out: dict[str, Any]) -> None:
    s = out["stats"]
    chunk_ms = out["chunk_ms_target"]
    print()
    print(f"==== v0.10.x harness summary ({out['duration_s']:.0f}s) ====")
    print(f"  chunks_processed        {s['chunks_processed']}")
    print(f"  player_underruns        {s['player_underruns']}")
    print(
        f"  underrun rate           {s['player_underruns'] / max(out['duration_s'], 1):.2f}/sec  "
        f"(acceptance gate ≤ 0.5/sec)"
    )
    print(f"  writer_jitter σ         {s['writer_jitter_ms_stddev']:.1f} ms")
    print(
        f"  writer_interval p50/p95/p99  {s['writer_interval_p50_ms']:.1f} / "
        f"{s['writer_interval_p95_ms']:.1f} / {s['writer_interval_p99_ms']:.1f} ms  "
        f"(target {chunk_ms:.0f} ms)"
    )
    print(
        f"  writer_jitter p99 = p99 - chunk    {s['writer_jitter_p99_ms']:.1f} ms  "
        f"(acceptance gate ≤ 30 ms)"
    )
    print(
        f"  inference  p50/p95/p99/max {s['inference_p50_ms']:.1f} / "
        f"{s['inference_p95_ms']:.1f} / {s['inference_p99_ms']:.1f} / "
        f"{s['inference_max_ms']:.1f} ms"
    )
    print(
        f"   .cv      p50/p99   {s['cv_p50_ms']:.1f} / {s['cv_p99_ms']:.1f} ms"
    )
    print(
        f"   .rmvpe   p50/p99   {s['rmvpe_p50_ms']:.1f} / {s['rmvpe_p99_ms']:.1f} ms"
    )
    print(
        f"   .rvc     p50/p99   {s['rvc_p50_ms']:.1f} / {s['rvc_p99_ms']:.1f} ms"
    )
    if "rvc_run_p50_ms" in s:
        print(
            f"     .rvc_pre   p50/p99   {s['rvc_pre_p50_ms']:.1f} / {s['rvc_pre_p99_ms']:.1f} ms"
        )
        print(
            f"     .rvc_run   p50/p99   {s['rvc_run_p50_ms']:.1f} / {s['rvc_run_p99_ms']:.1f} ms"
        )
        print(
            f"     .rvc_post  p50/p99   {s['rvc_post_p50_ms']:.1f} / {s['rvc_post_p99_ms']:.1f} ms"
        )
    print(f"  late_chunks             {s['late_chunks']}  (>{chunk_ms:.0f} ms wall budget)")
    print(f"  dropped_chunks          {s['dropped_chunks']}  (inference exceptions)")
    print(f"  warmup_audio16_lens     {s['warmup_audio16_lens']}")
    print(f"  runtime_audio16_lens    {s['runtime_audio16_lens']}")
    if s["unwarmed_shapes"]:
        print(f"  [!] unwarmed_shapes     {s['unwarmed_shapes']}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--duration", type=float, default=60.0, help="seconds")
    parser.add_argument("--out", type=Path, default=None, help="output JSON path")
    parser.add_argument("--no-sola", action="store_true", help="disable SOLA crossfade")
    parser.add_argument("--chunk-seconds", type=float, default=None, help="override chunk_seconds")
    parser.add_argument("--subprocess", action="store_true", help="enable inference_subprocess")
    parser.add_argument(
        "--pyspy",
        type=Path,
        default=None,
        help="if set, run under py-spy record and emit flamegraph SVG to this path",
    )
    args = parser.parse_args()

    if args.pyspy is not None:
        # Re-exec ourselves under py-spy so the profiler attaches before
        # any heavy imports. We pass the same args minus --pyspy.
        rerun = [
            ".venv/bin/py-spy",
            "record",
            "-o",
            str(args.pyspy),
            "--rate",
            "200",
            "--",
            sys.executable,
            __file__,
            "--duration",
            str(args.duration),
        ]
        if args.out is not None:
            rerun.extend(["--out", str(args.out)])
        if args.no_sola:
            rerun.append("--no-sola")
        if args.chunk_seconds is not None:
            rerun.extend(["--chunk-seconds", str(args.chunk_seconds)])
        if args.subprocess:
            rerun.append("--subprocess")
        print(f"[harness] re-execing under py-spy: {' '.join(rerun)}", file=sys.stderr)
        return subprocess.call(rerun, cwd=str(REPO))

    out = _run_engine_synthetic(
        duration_s=args.duration,
        out_path=args.out,
        enable_sola=not args.no_sola,
        chunk_seconds=args.chunk_seconds,
        inference_subprocess=args.subprocess,
    )
    _print_summary(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
