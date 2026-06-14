"""Automated output_latency_ms sweep harness.

For each candidate `output_latency_ms` value, run the engine on the
synthetic 60-second fixture (mono speech-like), capture the engine
output from `WoysSink.monitor` via parec, and analyze the capture with
`woys-diag analyze` for cuts/min. Engine `xruns / queue_full /
late_chunks / inference avg+p99` are also recorded.

The point is to find the smallest output_latency_ms where cuts/min
stays below the user's tolerance (target: < 18 / min) without doing
seven manual mic tests.

Run:
    .venv/bin/python scripts/sweep_latency.py \\
        --voice catwoman \\
        --latencies 80,120,150,180,220,250,300 \\
        --duration 60

Outputs:
- docs/sweep_latency_<timestamp>.json - full numeric results
- docs/sweep_latency_<timestamp>.png - latency-vs-cuts/min plot
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import subprocess
import sys
import threading
import time
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import sounddevice as sd  # noqa: E402

import audio.engine as ae  # noqa: E402

DIAG = Path.home() / ".local" / "bin" / "woys-diag"
FIXTURE = REPO / "tests" / "fixtures" / "auto_sweep_input.wav"


class _FixtureInputStream:
    """Drop-in for `sd.InputStream` that streams a WAV at realtime pace.

    Critical detail - mimics how a real mic delivers audio CONCURRENTLY
    with engine processing, not sequentially. Real mic: PortAudio fills
    an internal buffer at sample rate while the engine processes the
    previous chunk; when `read()` is called the buffer already has data
    waiting (or near-waiting) so the call returns quickly. Net per-loop
    cadence = chunk_seconds, NOT chunk_seconds + inference_ms.

    Earlier version naively did `time.sleep(frames / sr)` per call,
    which inflated the engine main-loop cadence by the inference time
    and produced output-buffer underruns that are NOT representative
    of real-mic behavior. v2 tracks wall-clock from `start()` and
    sleeps only as much as needed to keep `read()` returns aligned to
    a wall-clock-paced fixture cursor.
    """

    def __init__(
        self,
        samplerate: int = 48_000,
        channels: int = 1,
        blocksize: int = 0,
        dtype: str = "float32",
        **_kw: object,
    ) -> None:
        self.samplerate = samplerate
        self.channels = channels
        self.blocksize = blocksize
        self.dtype = dtype
        with wave.open(str(FIXTURE), "rb") as w:
            assert w.getframerate() == samplerate, (
                f"fixture rate {w.getframerate()} != engine mic_rate {samplerate}"
            )
            assert w.getnchannels() == 1, "fixture must be mono"
            raw = w.readframes(w.getnframes())
        self._data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32_768.0
        self._pos = 0
        self._start_time: float | None = None

    def __enter__(self) -> _FixtureInputStream:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def start(self) -> None:
        self._start_time = time.perf_counter()

    def stop(self) -> None:
        return None

    def close(self) -> None:
        return None

    def read(self, frames: int) -> tuple[np.ndarray, bool]:
        # Real mic pacing: nth sample is "delivered" at t = n / samplerate
        # from stream start. We sleep only as much as needed to align
        # the next return with that wall-clock target. If the engine is
        # ahead (just finished a fast chunk), we wait. If the engine is
        # behind (slow inference), we don't sleep - the buffer would
        # already have the data ready.
        if self._start_time is None:
            self._start_time = time.perf_counter()
        target = self._start_time + (self._pos + frames) / self.samplerate
        now = time.perf_counter()
        if now < target:
            time.sleep(target - now)
        end = self._pos + frames
        if end <= self._data.size:
            chunk = self._data[self._pos : end]
            self._pos = end
        elif self._pos < self._data.size:
            tail = self._data[self._pos :]
            chunk = np.concatenate([tail, np.zeros(frames - tail.size, dtype=np.float32)])
            self._pos = self._data.size
        else:
            chunk = np.zeros(frames, dtype=np.float32)
        return chunk.reshape(-1, self.channels), False


@dataclass
class _Run:
    output_latency_ms: int
    voice: str
    chunk_seconds: float
    backend: str
    duration_s: float
    capture_path: str
    cuts_per_min: float
    silent_gaps: int
    clicks: int
    capture_rms_dbfs: float
    capture_peak_dbfs: float
    chunks_processed: int
    xruns: int
    queue_full: int
    late_chunks: int
    avg_inference_ms: float
    p99_inference_ms: float
    max_inference_ms: float
    notes: list[str] = field(default_factory=list)


def _run_engine(
    cfg: ae.EngineConfig,
    duration_s: float,
    capture_started: threading.Event,
    stop_event: threading.Event,
    holder: dict[str, object],
) -> None:
    sd.InputStream = _FixtureInputStream  # type: ignore[assignment,misc]
    eng = ae.RealtimeEngine(cfg)
    holder["engine"] = eng
    eng.start()
    try:
        # Wait for the harness to start its capture before snapshotting
        # stats - queue_full_events otherwise accumulates during the
        # pre-capture warmup window when pw-cat / pacat has no drain.
        capture_started.wait(timeout=20.0)
        # Reset the cumulative counters that we want to attribute to the
        # capture window only.
        eng.stats.xruns = 0
        eng.stats.queue_full_events = 0
        eng.stats.late_chunks = 0
        eng.stats._recent_inference.clear()
        eng.stats.max_inference_ms = 0.0
        eng.stats.chunks_processed = 0
        # Hold until told to stop, then snapshot.
        while not stop_event.is_set():
            time.sleep(0.05)
        s = eng.stats
        infs = list(s._recent_inference)
        holder["chunks_processed"] = s.chunks_processed
        holder["xruns"] = s.xruns
        holder["queue_full"] = s.queue_full_events
        holder["late_chunks"] = s.late_chunks
        holder["avg_inference_ms"] = float(np.mean(infs)) if infs else 0.0
        holder["p99_inference_ms"] = float(np.percentile(infs, 99)) if infs else 0.0
        holder["max_inference_ms"] = float(s.max_inference_ms)
    finally:
        eng.stop(timeout=4.0)


def _capture_monitor(
    duration_s: float,
    out_path: Path,
    source: str = "WoysSink.monitor",
) -> None:
    """Run parec to capture from PipeWire source. Blocks until done."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # parec writes raw PCM; we wrap as WAV ourselves so woys-diag can read it.
    raw = out_path.with_suffix(".raw")
    with open(raw, "wb") as raw_fh:
        proc = subprocess.Popen(
            [
                "parec",
                f"--device={source}",
                "--rate=48000",
                "--channels=2",
                "--format=s16le",
                "--raw",
            ],
            stdout=raw_fh,
            stderr=subprocess.DEVNULL,
        )
        try:
            time.sleep(duration_s)
        finally:
            proc.terminate()
            proc.wait(timeout=2.0)
    # Convert raw → wav.
    raw_bytes = raw.read_bytes()
    raw.unlink()
    samples = np.frombuffer(raw_bytes, dtype=np.int16)
    # parec captures stereo; downmix to mono for woys-diag.
    if samples.size % 2 != 0:
        samples = samples[: samples.size - 1]
    stereo = samples.reshape(-1, 2)
    mono = (stereo[:, 0].astype(np.int32) + stereo[:, 1].astype(np.int32)) // 2
    mono = mono.astype(np.int16)
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(48_000)
        w.writeframes(mono.tobytes())


def _capture_levels(wav_path: Path) -> tuple[float, float]:
    with wave.open(str(wav_path), "rb") as w:
        raw = w.readframes(w.getnframes())
    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32_768.0
    if arr.size == 0:
        return -240.0, -240.0
    rms = float(np.sqrt(np.mean(arr * arr)))
    peak = float(np.abs(arr).max())
    rms_db = 20.0 * np.log10(rms) if rms > 0 else -240.0
    peak_db = 20.0 * np.log10(peak) if peak > 0 else -240.0
    return rms_db, peak_db


def _parse_diag(stdout: str, report_path: Path | None = None) -> tuple[float, int, int]:
    """Parse the cuts-per-min number from either the stdout review line
    or the markdown report file. The review line takes the form

        N events across Ds (X.Y/min). Disruptive…

    while the markdown summary table contains rows like

        | Events per minute | X.Y |
        | Silent-gap dropouts | **N** |
        | Clicks / discontinuities | **N** |

    We try both: stdout first (always present), then the report file if
    we have its path (gives us absolute counts).
    """
    cuts = float("nan")
    m = re.search(r"\(([\d.]+)\s*/\s*min\)", stdout)
    if m:
        cuts = float(m.group(1))
    elif "Audio is clean" in stdout or "no silent-gap" in stdout.lower():
        cuts = 0.0
    gaps = -1
    clicks = -1
    if report_path is not None and report_path.exists():
        body = report_path.read_text()
        if cuts != cuts:  # stdout didn't have it
            m2 = re.search(r"Events per minute\s*\|\s*([\d.]+)", body)
            if m2:
                cuts = float(m2.group(1))
        m_g = re.search(r"Silent-gap dropouts\s*\|\s*\*\*?(\d+)\*?\*?", body)
        if m_g:
            gaps = int(m_g.group(1))
        m_c = re.search(r"Clicks / discontinuities\s*\|\s*\*\*?(\d+)\*?\*?", body)
        if m_c:
            clicks = int(m_c.group(1))
    return cuts, gaps, clicks


def run_one(
    output_latency_ms: int,
    voice: str,
    chunk_seconds: float,
    backend: str,
    duration_s: float,
    out_dir: Path,
) -> _Run:
    rvc_path = ae.MODELS_DIR / f"{voice}.onnx"
    if not rvc_path.exists():
        raise FileNotFoundError(f"voice model not found: {rvc_path}")

    cfg = ae.EngineConfig(
        rvc_model=rvc_path,
        chunk_seconds=chunk_seconds,
        output_latency_ms=output_latency_ms,
        prefer_pw_cat=(backend == "pw-cat"),
    )

    holder: dict[str, object] = {}
    capture_started = threading.Event()
    stop = threading.Event()
    eng_t = threading.Thread(
        target=_run_engine,
        args=(cfg, duration_s, capture_started, stop, holder),
        daemon=False,
    )
    eng_t.start()

    # Let the engine warm up - start() blocks on warmup but we still want
    # a couple seconds for the run-loop to lock onto cadence + the
    # fixture's 3 s lead silence to flow through.
    time.sleep(4.0)

    capture_path = out_dir / f"capture_lat{output_latency_ms}_{backend}.wav"
    print(f"  capturing {duration_s:.0f}s to {capture_path.name}…")
    capture_started.set()
    _capture_monitor(duration_s, capture_path)

    stop.set()
    eng_t.join(timeout=8.0)

    rms_db, peak_db = _capture_levels(capture_path)
    notes: list[str] = []
    if peak_db < -60.0:
        notes.append(f"capture appears silent (peak {peak_db:.1f} dBFS)")

    diag = subprocess.run(
        [
            str(DIAG),
            "analyze",
            str(capture_path),
            "--source",
            f"sweep_lat{output_latency_ms}_{backend}",
            "--no-spectrogram",
        ],
        capture_output=True,
        text=True,
    )
    out = diag.stdout + "\n" + diag.stderr
    report_path = capture_path.with_suffix(".md")
    cuts, gaps, clicks = _parse_diag(out, report_path)
    if cuts != cuts:  # NaN check
        notes.append("woys-diag did not produce a per-minute number")

    return _Run(
        output_latency_ms=output_latency_ms,
        voice=voice,
        chunk_seconds=chunk_seconds,
        backend=backend,
        duration_s=duration_s,
        capture_path=str(capture_path),
        cuts_per_min=cuts,
        silent_gaps=gaps,
        clicks=clicks,
        capture_rms_dbfs=rms_db,
        capture_peak_dbfs=peak_db,
        chunks_processed=int(holder.get("chunks_processed", 0)),
        xruns=int(holder.get("xruns", 0)),
        queue_full=int(holder.get("queue_full", 0)),
        late_chunks=int(holder.get("late_chunks", 0)),
        avg_inference_ms=float(holder.get("avg_inference_ms", 0.0)),
        p99_inference_ms=float(holder.get("p99_inference_ms", 0.0)),
        max_inference_ms=float(holder.get("max_inference_ms", 0.0)),
        notes=notes,
    )


def _plot(runs: list[_Run], png_path: Path, threshold: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs_sorted = sorted(runs, key=lambda r: r.output_latency_ms)
    x = [r.output_latency_ms for r in runs_sorted]
    y = [r.cuts_per_min for r in runs_sorted]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, marker="o", linewidth=2, color="#2c7fb8")
    for r in runs_sorted:
        if r.cuts_per_min == r.cuts_per_min:
            ax.annotate(
                f"{r.cuts_per_min:.0f}",
                (r.output_latency_ms, r.cuts_per_min),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=9,
            )
    ax.axhline(threshold, color="#d95f0e", linestyle="--", label=f"acceptance ≤ {threshold:g}/min")
    ax.set_xlabel("output_latency_ms (pacat / pw-cat buffer)")
    ax.set_ylabel("cuts / min (woys-diag analyze)")
    ax.set_title(f"v0.7.0 latency sweep - {runs_sorted[0].voice} on {runs_sorted[0].backend}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="catwoman")
    ap.add_argument(
        "--latencies",
        default="80,120,150,180,220,250,300",
        help="comma-separated list of output_latency_ms values to sweep",
    )
    ap.add_argument("--chunk", type=float, default=0.15)
    ap.add_argument("--backend", choices=["pw-cat", "pacat"], default="pw-cat")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--threshold", type=float, default=18.0, help="cuts/min acceptance threshold")
    args = ap.parse_args()

    if not FIXTURE.exists():
        print("ERROR: fixture missing - run scripts/gen_sweep_fixture.py first", file=sys.stderr)
        sys.exit(2)
    if not DIAG.exists():
        print(f"ERROR: woys-diag not at {DIAG}", file=sys.stderr)
        sys.exit(2)

    latencies = [int(x.strip()) for x in args.latencies.split(",") if x.strip()]
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = REPO / "docs" / f"sweep_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = REPO / "docs" / f"sweep_latency_{stamp}.json"
    png_path = REPO / "docs" / f"sweep_latency_{stamp}.png"

    print(f"sweep config: voice={args.voice} chunk={args.chunk} backend={args.backend}")
    print(f"latencies: {latencies}")
    print(f"output dir: {out_dir}")

    runs: list[_Run] = []
    for lat in latencies:
        print(f"\n=== output_latency_ms = {lat} ===")
        try:
            run = run_one(
                output_latency_ms=lat,
                voice=args.voice,
                chunk_seconds=args.chunk,
                backend=args.backend,
                duration_s=args.duration,
                out_dir=out_dir,
            )
        except Exception as e:
            print(f"  ERROR at lat={lat}: {type(e).__name__}: {e}")
            continue
        runs.append(run)
        print(
            f"  cuts/min={run.cuts_per_min:.1f}  gaps={run.silent_gaps}  clicks={run.clicks}  "
            f"capture_peak={run.capture_peak_dbfs:.1f} dBFS"
        )
        print(
            f"  engine: chunks={run.chunks_processed} xruns={run.xruns} "
            f"qfull={run.queue_full} late={run.late_chunks} "
            f"avg_inf={run.avg_inference_ms:.1f}ms p99={run.p99_inference_ms:.1f}ms"
        )
        # Save partial results after each run so we keep them on crash.
        json_path.write_text(json.dumps([asdict(r) for r in runs], indent=2))
        # Settle PipeWire between runs.
        time.sleep(1.5)

    if not runs:
        print("\nno runs completed - nothing to plot", file=sys.stderr)
        sys.exit(3)

    _plot(runs, png_path, args.threshold)

    # Pick the recommendation.
    runs_sorted = sorted(runs, key=lambda r: r.output_latency_ms)
    accepted = [
        r
        for r in runs_sorted
        if r.cuts_per_min == r.cuts_per_min and r.cuts_per_min < args.threshold
    ]
    print("\n=== summary ===")
    print(f"{'lat':>5}  {'cuts/min':>8}  {'gaps':>5}  {'clicks':>7}  {'late':>5}  {'p99_inf':>8}")
    for r in runs_sorted:
        print(
            f"{r.output_latency_ms:>5}  {r.cuts_per_min:>8.1f}  {r.silent_gaps:>5}  "
            f"{r.clicks:>7}  {r.late_chunks:>5}  {r.p99_inference_ms:>8.1f}"
        )
    if accepted:
        chosen = accepted[0]
        print(
            f"\nrecommendation: output_latency_ms = {chosen.output_latency_ms} "
            f"(cuts/min {chosen.cuts_per_min:.1f}, threshold {args.threshold:g})"
        )
    else:
        print(
            f"\nno setting met cuts/min < {args.threshold:g}; "
            f"either widen the search up or accept a higher threshold"
        )

    print(f"\njson:  {json_path}")
    print(f"plot:  {png_path}")


if __name__ == "__main__":
    main()
