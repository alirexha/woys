"""Run the engine with synthetic input + woys-diag capture in parallel,
report cuts/min for the chosen (chunk_seconds, output_latency_ms,
prefer_pw_cat) config. The v0.6.7 / v0.6.9 ground-truth metric.

Used to compare pacat vs pw-cat at low latencies before shipping
v0.7.0 defaults.

Run:
    .venv/bin/python scripts/cuts_per_min_check.py \\
        --chunk 0.15 --latency 80 --backend pw-cat --voice catwoman \\
        --duration 30
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _engine_thread(
    voice: str,
    chunk: float,
    latency: int,
    prefer_pw_cat: bool,
    stop_event: threading.Event,
    stats_holder: dict[str, object],
) -> None:
    import sounddevice as sd

    from scripts.bench_engine_runtime import _SyntheticInputStream

    sd.InputStream = _SyntheticInputStream  # type: ignore[assignment,misc]

    import audio.engine as ae

    cfg = ae.EngineConfig(
        rvc_model=ae.MODELS_DIR / f"{voice}.onnx",
        chunk_seconds=chunk,
        output_latency_ms=latency,
        prefer_pw_cat=prefer_pw_cat,
    )
    eng = ae.RealtimeEngine(cfg)
    eng.start()
    try:
        # Wait until told to stop.
        while not stop_event.is_set():
            time.sleep(0.1)
        s = eng.stats
        stats_holder["xruns"] = s.xruns
        stats_holder["queue_full"] = s.queue_full_events
        stats_holder["late_chunks"] = s.late_chunks
        stats_holder["chunks"] = s.chunks_processed
        stats_holder["avg_inf_ms"] = float(np.mean(list(s._recent_inference) or [0.0]))
        stats_holder["p99_inf_ms"] = (
            float(np.percentile(list(s._recent_inference), 99)) if s._recent_inference else 0.0
        )
    finally:
        eng.stop(timeout=3.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=float, default=0.15)
    ap.add_argument("--latency", type=int, default=80)
    ap.add_argument("--backend", choices=["pacat", "pw-cat"], default="pw-cat")
    ap.add_argument("--voice", type=str, default="catwoman")
    ap.add_argument("--duration", type=int, default=30)
    args = ap.parse_args()

    diag = Path.home() / ".local" / "bin" / "woys-diag"
    if not diag.exists():
        print("ERROR: woys-diag not installed at ~/.local/bin/woys-diag", file=sys.stderr)
        sys.exit(2)

    stop_event = threading.Event()
    stats: dict[str, object] = {}
    t = threading.Thread(
        target=_engine_thread,
        args=(args.voice, args.chunk, args.latency, args.backend == "pw-cat", stop_event, stats),
    )
    t.start()

    # Wait for engine warmup (cudnn autotune + first chunks).
    print(
        f"warming engine (chunk={args.chunk}s, lat={args.latency}ms, backend={args.backend}, "
        f"voice={args.voice})..."
    )
    time.sleep(5)

    # Run woys-diag to capture from woys-mic.
    print(f"running woys-diag for {args.duration}s...")
    proc = subprocess.run(
        [
            str(diag),
            "run",
            "--duration",
            str(args.duration),
            "--source",
            "woys-mic",
            "--voice",
            f"{args.voice}_chunk{args.chunk}_lat{args.latency}_{args.backend}",
            "--no-spectrogram",
            "--quiet-countdown",
        ],
        capture_output=True,
        text=True,
    )

    stop_event.set()
    t.join(timeout=5.0)

    out = proc.stdout + "\n" + proc.stderr
    # Find cuts/min in output.
    cuts_match = re.search(r"cuts[/_]min[:\s=]+([\d.]+)", out, re.IGNORECASE)
    silence_match = re.search(r"silence[/_]min[:\s=]+([\d.]+)", out, re.IGNORECASE)
    print(f"\n=== {args.voice} chunk={args.chunk} lat={args.latency} {args.backend} ===")
    print(
        f"  engine: chunks={stats.get('chunks')} xruns={stats.get('xruns')} "
        f"qfull={stats.get('queue_full')} late={stats.get('late_chunks')} "
        f"avg_inf={stats.get('avg_inf_ms', 0):.1f}ms p99={stats.get('p99_inf_ms', 0):.1f}ms"
    )
    if cuts_match:
        print(f"  cuts/min     : {cuts_match.group(1)}")
    if silence_match:
        print(f"  silence/min  : {silence_match.group(1)}")
    if not cuts_match:
        print("  raw woys-diag output (last 30 lines):")
        for line in out.strip().splitlines()[-30:]:
            print(f"    {line}")


if __name__ == "__main__":
    main()
