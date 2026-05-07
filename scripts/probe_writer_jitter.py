#!/usr/bin/env python
"""Probe writer-thread per-stage timing in isolation from the engine.

Background: rc5's `woys diag` reports `writer_jitter_ms = 62 ms` at
chunk_seconds=150 ms cadence + `xruns ≈ 1.8 / s` even though
`overrun_ratio = 0.000` (engine inference fits in budget). The bug is
between "inference complete" and "pacat receives chunk." Three
suspects:

  1. The engine→writer queue (`q.put_nowait` / `q.get(timeout=0.1)`)
  2. The pacat stdin pipe write (write() + flush())
  3. PipeWire / pacat-side scheduling

This script isolates suspect 2 by driving pacat at the engine's
cadence WITHOUT going through the engine. It writes silence chunks at
150 ms intervals and captures per-stage timing:

  write_ms    — wall time spent in `proc.stdin.write(payload)`
  flush_ms    — wall time spent in `proc.stdin.flush()`
  total_ms    — write_ms + flush_ms
  interval_ms — wall time between successive write-completes

If write+flush variance dominates `interval_ms` variance, the pipe is
the bottleneck and `fcntl(F_SETPIPE_SZ)` is the candidate fix. If
write+flush is near-zero and interval variance comes from elsewhere,
the bottleneck is upstream (queue or scheduler).

Usage:

  ./scripts/probe_writer_jitter.py
  ./scripts/probe_writer_jitter.py --duration 30 --pipe-size 1048576

Run with `pactl list short sinks` showing `WoysSink` first; the
script targets that sink. Doesn't depend on the engine being running.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import shutil
import statistics
import subprocess
import sys
import time

import numpy as np

# Linux F_SETPIPE_SZ value — request the kernel resize a pipe.
F_SETPIPE_SZ = 1031


def _build_chunk(chunk_seconds: float, sink_rate: int, output_channels: int) -> bytes:
    """One chunk of silence, byte-formatted as the engine's
    `_to_sink_bytes` would produce it. Matches the engine's
    `_open_pacat` payload format (float32le, interleaved stereo)."""
    n_samples = round(chunk_seconds * sink_rate)
    mono = np.zeros(n_samples, dtype=np.float32)
    if output_channels == 1:
        return mono.tobytes()
    stereo = np.repeat(mono, output_channels)
    return stereo.tobytes()


def _open_pacat(
    sink_name: str,
    sink_rate: int,
    output_channels: int,
    output_latency_ms: int,
    output_process_time_ms: int,
) -> subprocess.Popen[bytes]:
    """Match the engine's `_open_pacat` for prefer_pw_cat=False."""
    pacat = shutil.which("pacat")
    if pacat is None:
        raise RuntimeError("pacat not found on PATH; install pipewire-pulse")
    cmd = [
        pacat,
        "--playback",
        f"--device={sink_name}",
        f"--rate={sink_rate}",
        f"--channels={output_channels}",
        "--format=float32le",
        f"--latency-msec={output_latency_ms}",
        f"--process-time-msec={output_process_time_ms}",
        "--client-name=woys-probe",
        "--stream-name=writer-jitter-probe",
        "--raw",
    ]
    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _try_resize_pipe(pipe_fd: int, target_size: int) -> int:
    """Request the kernel resize the pipe buffer. Returns the resulting
    size (which may be smaller than requested if we hit
    /proc/sys/fs/pipe-max-size). Returns 0 if F_SETPIPE_SZ unsupported."""
    try:
        actual = fcntl.fcntl(pipe_fd, F_SETPIPE_SZ, target_size)
        return int(actual)
    except (OSError, ValueError) as e:
        print(f"[probe] F_SETPIPE_SZ failed: {e}", file=sys.stderr)
        return 0


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(values, p))


def _summarize(label: str, values: list[float]) -> None:
    if not values:
        print(f"  {label:14s} n=0  (no samples)")
        return
    mean = statistics.mean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    print(
        f"  {label:14s} "
        f"n={len(values):3d}  "
        f"mean={mean:6.2f}  std={std:6.2f}  "
        f"p50={_percentile(values, 50):6.2f}  "
        f"p95={_percentile(values, 95):6.2f}  "
        f"p99={_percentile(values, 99):6.2f}  "
        f"max={max(values):6.2f}"
    )


def run_probe(
    sink_name: str,
    duration_s: float,
    chunk_seconds: float,
    sink_rate: int,
    output_channels: int,
    output_latency_ms: int,
    output_process_time_ms: int,
    pipe_size: int | None,
    label: str,
) -> dict[str, list[float]]:
    """Drive pacat at chunk_seconds cadence for duration_s, capturing
    per-stage timing. Returns {stage: [values_ms]}."""
    print(
        f"\n=== {label} ===\n"
        f"  sink={sink_name} chunk_seconds={chunk_seconds}s "
        f"output_latency_ms={output_latency_ms} pipe_size={pipe_size or 'default'}",
        file=sys.stderr,
    )

    proc = _open_pacat(
        sink_name=sink_name,
        sink_rate=sink_rate,
        output_channels=output_channels,
        output_latency_ms=output_latency_ms,
        output_process_time_ms=output_process_time_ms,
    )
    if proc.stdin is None:
        raise RuntimeError("pacat stdin not pipeable")

    # Try to resize the pipe buffer if asked.
    if pipe_size is not None and pipe_size > 0:
        pipe_fd = proc.stdin.fileno()
        actual = _try_resize_pipe(pipe_fd, pipe_size)
        print(
            f"  pipe resize: requested={pipe_size}  actual={actual}",
            file=sys.stderr,
        )

    payload = _build_chunk(chunk_seconds, sink_rate, output_channels)
    print(f"  payload size: {len(payload)} bytes / chunk", file=sys.stderr)

    write_ms: list[float] = []
    flush_ms: list[float] = []
    total_ms: list[float] = []
    interval_ms: list[float] = []

    last_complete: float | None = None
    n_chunks = round(duration_s / chunk_seconds)
    start = time.perf_counter()

    try:
        for i in range(n_chunks):
            # Wait until the next 150 ms tick (mimics engine cadence).
            target = start + (i + 1) * chunk_seconds
            now = time.perf_counter()
            if target > now:
                time.sleep(target - now)

            t_pre_write = time.perf_counter()
            try:
                proc.stdin.write(payload)
            except (BrokenPipeError, OSError) as e:
                print(f"[probe] write failed at chunk {i}: {e}", file=sys.stderr)
                break
            t_post_write = time.perf_counter()
            try:
                proc.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                print(f"[probe] flush failed at chunk {i}: {e}", file=sys.stderr)
                break
            t_post_flush = time.perf_counter()

            write_ms.append((t_post_write - t_pre_write) * 1000.0)
            flush_ms.append((t_post_flush - t_post_write) * 1000.0)
            total_ms.append((t_post_flush - t_pre_write) * 1000.0)
            if last_complete is not None:
                interval_ms.append((t_post_flush - last_complete) * 1000.0)
            last_complete = t_post_flush
    finally:
        with contextlib.suppress(OSError):
            proc.stdin.close()
        proc.wait(timeout=2.0)

    # Drop the first 5 chunks as warmup (pipe filling, pacat coming up).
    return {
        "write_ms": write_ms[5:],
        "flush_ms": flush_ms[5:],
        "total_ms": total_ms[5:],
        "interval_ms": interval_ms[5:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe pacat writer-side timing in isolation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--sink-name", default="WoysSink")
    parser.add_argument("--duration", type=float, default=20.0, help="seconds")
    parser.add_argument("--chunk-seconds", type=float, default=0.15)
    parser.add_argument("--sink-rate", type=int, default=48_000)
    parser.add_argument("--output-channels", type=int, default=2)
    parser.add_argument("--output-latency-ms", type=int, default=280)
    parser.add_argument("--output-process-time-ms", type=int, default=20)
    parser.add_argument(
        "--compare-pipe-sizes",
        action="store_true",
        help="Run twice — once with default pipe size, once with 1 MB — "
        "to A/B test whether F_SETPIPE_SZ is the candidate rc6 fix.",
    )
    args = parser.parse_args()

    runs: list[tuple[str, int | None]] = [("default pipe size", None)]
    if args.compare_pipe_sizes:
        runs.append(("1 MB pipe (F_SETPIPE_SZ)", 1024 * 1024))

    for label, pipe_size in runs:
        results = run_probe(
            sink_name=args.sink_name,
            duration_s=args.duration,
            chunk_seconds=args.chunk_seconds,
            sink_rate=args.sink_rate,
            output_channels=args.output_channels,
            output_latency_ms=args.output_latency_ms,
            output_process_time_ms=args.output_process_time_ms,
            pipe_size=pipe_size,
            label=label,
        )
        print("\n  per-stage timing (warmup-trimmed, ms):")
        _summarize("write_ms", results["write_ms"])
        _summarize("flush_ms", results["flush_ms"])
        _summarize("total_ms", results["total_ms"])
        _summarize("interval_ms", results["interval_ms"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
