#!/usr/bin/env python3
"""Per-version benchmark probe for `scripts/benchmark_compare.py`.

Runs INSIDE a worktree of a specific woys version tag. Imports
the local `src/audio.engine` module via sys.path, performs a
cold-start + warmup + measurement loop on an offline WAV
input, and writes a JSON result to a file.

This file is COPIED into each worktree by the orchestrator
before being invoked, so older tags (which don't have this
file) can still be benchmarked against their own engine code.

Contract (orchestrator-facing):

    argv[1]: WAV path (48k mono or 16k mono; resampled to 16k
             inside the probe).
    argv[2]: warmup chunks (int).
    argv[3]: measurement chunks (int).
    argv[4]: output JSON path.
    argv[5]: rep_index (int; for multi-rep aggregation).

Stdout: progress lines (one per phase).
Exit: 0 on success; non-zero if import / load / runtime fails
      (failure details captured into the JSON regardless).

Defensive features for cross-version compatibility:
- Tries `_process_streaming_16k` first; falls back to
  `process_chunk_16k` if the streaming method isn't present.
- Reads `EngineStats` fields via `getattr(stats, name, None)` so
  fields added in v0.15.0 (e.g. `sola_search_clipped`) appear as
  null on older tags without raising AttributeError.
- Writes a partial JSON with `status="failed"` + an `error` field
  if anything raises during import or engine build.

Hard Rule 9: every measured field is a real number from this run.
Fields that can't be measured for a given version are JSON `null`.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _read_proc_status(pid: int) -> dict[str, int]:
    """Parse /proc/<pid>/status for VmRSS, VmPeak, Threads, FDSize."""
    out: dict[str, int] = {}
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    out["rss_kb"] = int(line.split()[1])
                elif line.startswith("VmPeak:"):
                    out["peak_kb"] = int(line.split()[1])
                elif line.startswith("Threads:"):
                    out["threads"] = int(line.split()[1])
                elif line.startswith("FDSize:"):
                    out["fd_size"] = int(line.split()[1])
    except FileNotFoundError:
        pass
    return out


def _count_fds(pid: int) -> int:
    """Actual FD count from /proc/<pid>/fd/. `FDSize` is the
    allocated-slot count which only grows; this is the live count."""
    try:
        return len(os.listdir(f"/proc/{pid}/fd"))
    except (FileNotFoundError, PermissionError):
        return -1


def _read_nvidia_smi() -> tuple[int, float] | None:
    """Returns (vram_mb_used, gpu_util_pct) for the first GPU, or
    None if nvidia-smi is unavailable / fails."""
    import subprocess

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2.0,
        )
        if result.returncode != 0:
            return None
        first_line = result.stdout.strip().splitlines()[0]
        vram_mb, util_pct = (s.strip() for s in first_line.split(","))
        return int(vram_mb), float(util_pct)
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
        return None


class _ResourcePoller:
    """Polls /proc + nvidia-smi at fixed intervals from a daemon thread.

    Stores raw samples; the probe summarises them at the end.
    """

    def __init__(self, pid: int, interval_s: float = 1.0) -> None:
        self.pid = pid
        self.interval_s = interval_s
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._t: threading.Thread | None = None

    def start(self) -> None:
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def stop(self) -> None:
        self._stop.set()
        if self._t is not None:
            self._t.join(timeout=2.0)

    def _loop(self) -> None:
        while not self._stop.is_set():
            sample: dict[str, Any] = {"t": time.perf_counter()}
            status = _read_proc_status(self.pid)
            sample.update(status)
            sample["fd_count"] = _count_fds(self.pid)
            nv = _read_nvidia_smi()
            if nv is not None:
                sample["vram_mb"], sample["gpu_util_pct"] = nv
            self.samples.append(sample)
            self._stop.wait(self.interval_s)

    def summarise(self) -> dict[str, Any]:
        if not self.samples:
            return {}
        keys = ("rss_kb", "peak_kb", "threads", "fd_count", "vram_mb", "gpu_util_pct")
        out: dict[str, Any] = {}
        for k in keys:
            vals = [s[k] for s in self.samples if k in s and s[k] is not None]
            if not vals:
                out[k] = None
                continue
            vals_sorted = sorted(vals)
            out[k] = {
                "min": vals_sorted[0],
                "max": vals_sorted[-1],
                "median": vals_sorted[len(vals_sorted) // 2],
                "samples_n": len(vals),
            }
        out["fd_count_start"] = self.samples[0].get("fd_count") if self.samples else None
        out["fd_count_end"] = self.samples[-1].get("fd_count") if self.samples else None
        return out


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def _load_wav_mono_16k(wav_path: Path) -> Any:
    """Read WAV, convert to mono float32 at 16 kHz."""
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        try:
            import soxr

            audio = soxr.resample(audio, sr, 16000, quality="HQ")
        except ImportError:
            # Older versions may not have soxr; numpy linear resample is
            # crude but enough for benchmark input (we're measuring the
            # ENGINE, not the resampler).
            ratio = 16000 / sr
            new_len = int(len(audio) * ratio)
            old_idx = np.linspace(0, len(audio) - 1, new_len)
            audio = np.interp(old_idx, np.arange(len(audio)), audio).astype(np.float32)
    return np.asarray(audio, dtype=np.float32)


def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _engine_stats_to_dict(stats: Any) -> dict[str, Any]:
    """Read every known counter from EngineStats defensively. Fields
    added across the v0.7..v0.15 range get null on tags that don't
    have them."""
    fields = [
        "sola_fallback_count",
        "sola_search_clipped",  # v0.15.0 / commit-079
        "nan_chunks",
        "dropped_chunks",
        "gated_chunks",
        "input_overflows",
        "feats_nan_chunks",
        "late_chunks",
        "max_inference_ms",
        "max_total_ms",
        "last_cv_ms",
        "last_rmvpe_ms",
        "last_rvc_ms",
        "child_restarts",
        "consecutive_drops",
    ]
    return {f: _safe_getattr(stats, f, None) for f in fields}


def run_offline(
    wav_path: Path,
    warmup_chunks: int,
    measure_chunks: int,
    out_path: Path,
    rep_index: int,
) -> int:
    """Offline benchmark: feed 16k WAV chunks to engine, time each call,
    poll resources in the background, write JSON.

    Returns the exit code (0 = success)."""
    import numpy as np

    pid = os.getpid()
    result: dict[str, Any] = {
        "rep_index": rep_index,
        "started_at": _now_iso(),
        "pid": pid,
        "wav_path": str(wav_path),
        "warmup_chunks": warmup_chunks,
        "measure_chunks": measure_chunks,
        "status": "running",
    }

    # Step 1: load WAV.
    try:
        print(f"[probe rep={rep_index}] loading WAV {wav_path}", flush=True)
        audio_16k = _load_wav_mono_16k(wav_path)
        print(
            f"[probe rep={rep_index}] WAV loaded: {len(audio_16k)} samples ({len(audio_16k) / 16000:.2f} s)",
            flush=True,
        )
    except Exception as e:
        result["status"] = "wav_load_failed"
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
        out_path.write_text(json.dumps(result, indent=2))
        return 1

    # Step 2: import engine.
    t_import_0 = time.perf_counter()
    try:
        print(f"[probe rep={rep_index}] importing audio.engine ...", flush=True)
        from audio.engine import EngineConfig, RealtimeEngine  # type: ignore
    except Exception as e:
        result["status"] = "import_failed"
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
        out_path.write_text(json.dumps(result, indent=2))
        return 2
    t_import_ms = (time.perf_counter() - t_import_0) * 1000.0
    print(f"[probe rep={rep_index}] import took {t_import_ms:.0f} ms", flush=True)

    # Step 3: build engine + load sessions.
    t_build_0 = time.perf_counter()
    try:
        cfg = EngineConfig()
        # Disable the inference subprocess for offline benchmark -- we
        # want in-process timings (per-stage stats from `_infer`).
        if hasattr(cfg, "inference_subprocess"):
            cfg.inference_subprocess = False
        # SOLA stays enabled (default). Other knobs at defaults.
        engine = RealtimeEngine(cfg)
        # Force session load (some versions lazy-load on first chunk).
        if hasattr(engine, "_ensure_sessions"):
            engine._ensure_sessions()
        elif hasattr(engine, "ensure_sessions"):
            engine.ensure_sessions()
        # Newer versions may load via a different entry; if neither is
        # present, the sessions will load on first chunk and the
        # cold-start timing absorbs it.
    except Exception as e:
        result["status"] = "engine_build_failed"
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
        result["import_ms"] = t_import_ms
        out_path.write_text(json.dumps(result, indent=2))
        return 3
    t_build_ms = (time.perf_counter() - t_build_0) * 1000.0
    print(
        f"[probe rep={rep_index}] engine build + session load took {t_build_ms:.0f} ms",
        flush=True,
    )

    # Step 4: pick the streaming method.
    streaming_method = None
    method_name = None
    if hasattr(engine, "_process_streaming_16k"):
        streaming_method = engine._process_streaming_16k
        method_name = "_process_streaming_16k"
    elif hasattr(engine, "_safe_process_streaming_16k"):
        streaming_method = engine._safe_process_streaming_16k
        method_name = "_safe_process_streaming_16k"
    elif hasattr(engine, "process_chunk_16k"):
        streaming_method = engine.process_chunk_16k
        method_name = "process_chunk_16k_fallback"
    else:
        result["status"] = "no_inference_method"
        result["error"] = "engine has no recognised inference entry point"
        out_path.write_text(json.dumps(result, indent=2))
        return 4

    print(f"[probe rep={rep_index}] using inference method: {method_name}", flush=True)

    # Step 5: chunk sizing.
    chunk_seconds = float(_safe_getattr(cfg, "chunk_seconds", 0.25) or 0.25)
    chunk_samples_16k = int(16000 * chunk_seconds)
    total_chunks_needed = warmup_chunks + measure_chunks
    samples_needed = chunk_samples_16k * total_chunks_needed
    # Loop the WAV if needed.
    if len(audio_16k) < samples_needed:
        reps = (samples_needed + len(audio_16k) - 1) // len(audio_16k)
        audio_16k = np.tile(audio_16k, reps)
    audio_16k = audio_16k[:samples_needed]
    print(
        f"[probe rep={rep_index}] chunk_samples_16k={chunk_samples_16k} "
        f"warmup={warmup_chunks} measure={measure_chunks}",
        flush=True,
    )

    # Step 6: start resource poller.
    poller = _ResourcePoller(pid, interval_s=0.5)
    poller.start()

    # Step 7: cold-start time = first inference latency.
    chunk_idx = 0
    t_cold_start_0 = time.perf_counter()
    try:
        first_chunk = audio_16k[chunk_idx * chunk_samples_16k : (chunk_idx + 1) * chunk_samples_16k]
        _ = streaming_method(first_chunk)
        chunk_idx += 1
        t_cold_start_ms = (time.perf_counter() - t_cold_start_0) * 1000.0
        print(
            f"[probe rep={rep_index}] cold-start (first chunk): {t_cold_start_ms:.1f} ms",
            flush=True,
        )
    except Exception as e:
        poller.stop()
        result["status"] = "cold_start_failed"
        result["error"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()
        result["import_ms"] = t_import_ms
        result["build_ms"] = t_build_ms
        out_path.write_text(json.dumps(result, indent=2))
        return 5

    # Step 8: warmup.
    print(
        f"[probe rep={rep_index}] warmup ({warmup_chunks - 1} more chunks) ...",
        flush=True,
    )
    for _ in range(warmup_chunks - 1):
        chunk = audio_16k[chunk_idx * chunk_samples_16k : (chunk_idx + 1) * chunk_samples_16k]
        try:
            _ = streaming_method(chunk)
            chunk_idx += 1
        except Exception as e:
            poller.stop()
            result["status"] = "warmup_failed"
            result["error"] = f"{type(e).__name__}: {e} at warmup chunk {chunk_idx}"
            result["traceback"] = traceback.format_exc()
            out_path.write_text(json.dumps(result, indent=2))
            return 6

    # Step 9: measurement loop.
    print(
        f"[probe rep={rep_index}] measuring {measure_chunks} chunks ...",
        flush=True,
    )
    per_chunk_latency_ms: list[float] = []
    per_stage_cv_ms: list[float] = []
    per_stage_rmvpe_ms: list[float] = []
    per_stage_rvc_ms: list[float] = []
    drops_at_start = _safe_getattr(engine.stats, "dropped_chunks", 0) or 0

    t_measure_0 = time.perf_counter()
    for i in range(measure_chunks):
        chunk = audio_16k[chunk_idx * chunk_samples_16k : (chunk_idx + 1) * chunk_samples_16k]
        t_chunk_0 = time.perf_counter()
        try:
            _ = streaming_method(chunk)
        except Exception as e:
            poller.stop()
            result["status"] = "measure_failed"
            result["error"] = f"{type(e).__name__}: {e} at measure chunk {i}"
            result["traceback"] = traceback.format_exc()
            result["latency_ms_partial"] = per_chunk_latency_ms
            out_path.write_text(json.dumps(result, indent=2))
            return 7
        t_chunk_ms = (time.perf_counter() - t_chunk_0) * 1000.0
        per_chunk_latency_ms.append(t_chunk_ms)
        cv = _safe_getattr(engine.stats, "last_cv_ms", None)
        rmvpe = _safe_getattr(engine.stats, "last_rmvpe_ms", None)
        rvc = _safe_getattr(engine.stats, "last_rvc_ms", None)
        if cv is not None:
            per_stage_cv_ms.append(float(cv))
        if rmvpe is not None:
            per_stage_rmvpe_ms.append(float(rmvpe))
        if rvc is not None:
            per_stage_rvc_ms.append(float(rvc))
        chunk_idx += 1
    t_measure_ms = (time.perf_counter() - t_measure_0) * 1000.0
    poller.stop()

    drops_at_end = _safe_getattr(engine.stats, "dropped_chunks", 0) or 0
    drops_during = drops_at_end - drops_at_start

    sorted_lat = sorted(per_chunk_latency_ms)
    lat_summary = {
        "mean": sum(per_chunk_latency_ms) / len(per_chunk_latency_ms),
        "median": _percentile(sorted_lat, 0.50),
        "p50": _percentile(sorted_lat, 0.50),
        "p95": _percentile(sorted_lat, 0.95),
        "p99": _percentile(sorted_lat, 0.99),
        "min": sorted_lat[0],
        "max": sorted_lat[-1],
    }

    def _mean_or_none(xs: list[float]) -> float | None:
        return (sum(xs) / len(xs)) if xs else None

    per_stage = {
        "cv_ms_mean": _mean_or_none(per_stage_cv_ms),
        "cv_ms_samples": len(per_stage_cv_ms),
        "rmvpe_ms_mean": _mean_or_none(per_stage_rmvpe_ms),
        "rmvpe_ms_samples": len(per_stage_rmvpe_ms),
        "rvc_ms_mean": _mean_or_none(per_stage_rvc_ms),
        "rvc_ms_samples": len(per_stage_rvc_ms),
    }

    throughput_chunks_per_s = measure_chunks / (t_measure_ms / 1000.0)

    result.update(
        {
            "status": "success",
            "method_name": method_name,
            "chunk_seconds": chunk_seconds,
            "chunk_samples_16k": chunk_samples_16k,
            "import_ms": t_import_ms,
            "build_ms": t_build_ms,
            "cold_start_ms": t_cold_start_ms,
            "measure_wall_ms": t_measure_ms,
            "throughput_chunks_per_s": throughput_chunks_per_s,
            "latency_ms": lat_summary,
            "per_stage": per_stage,
            "engine_stats_end": _engine_stats_to_dict(engine.stats),
            "dropped_chunks_during_run": drops_during,
            "resources": poller.summarise(),
        }
    )

    # Clean shutdown timing.
    t_shutdown_0 = time.perf_counter()
    try:
        if hasattr(engine, "stop"):
            engine.stop()
    except Exception:
        pass
    result["shutdown_ms"] = (time.perf_counter() - t_shutdown_0) * 1000.0

    out_path.write_text(json.dumps(result, indent=2))
    print(f"[probe rep={rep_index}] done, wrote {out_path}", flush=True)
    return 0


def main() -> int:
    if len(sys.argv) != 6:
        print(
            "usage: benchmark_probe.py <wav> <warmup> <measure> <out_json> <rep_index>",
            file=sys.stderr,
        )
        return 99
    wav_path = Path(sys.argv[1])
    warmup_chunks = int(sys.argv[2])
    measure_chunks = int(sys.argv[3])
    out_path = Path(sys.argv[4])
    rep_index = int(sys.argv[5])
    return run_offline(wav_path, warmup_chunks, measure_chunks, out_path, rep_index)


if __name__ == "__main__":
    sys.exit(main())
