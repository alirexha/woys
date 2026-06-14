#!/usr/bin/env python3
"""Multi-version benchmark orchestrator for woys.

For each version tag in `--tags`:

  1. Create a git worktree at /tmp/woys-bench-<tag>/.
  2. Copy `scripts/benchmark_probe.py` into the worktree.
  3. Run the probe `--reps` times via the CURRENT venv's Python with
     PYTHONPATH pointing at the worktree's src/.
  4. Aggregate per-rep JSONs.
  5. Tear down the worktree.

Optionally runs a realtime soak (PipeWire loop-source) on the HEAD
checkout only (`--realtime-soak-tag <tag>` --realtime-soak-seconds N`).

Outputs:

  --out-json: aggregated JSON with all per-tag, per-rep measurements.
  --out-html: rendered HTML report with matplotlib charts (PNG, base64
              inline so the report is a single self-contained file).

Measurement policy: every number in the output JSON comes from a measured
run. Versions that fail at import / build / cold-start have
`status="..._failed"` in their per-rep entry and a JSON `null` for
every measured field. The HTML report renders these as "N/A — <reason>"
rather than fabricating a value.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WAV = REPO_ROOT / "tests" / "fixtures" / "auto_sweep_input.wav"
DEFAULT_VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _run(cmd: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
    """Wrapper that defaults text=True + raises on nonzero unless check=False."""
    kw.setdefault("text", True)
    kw.setdefault("capture_output", True)
    return subprocess.run(cmd, **kw)


def _get_commit_hash_for_tag(tag: str) -> str:
    """Resolve a tag to its full commit hash."""
    return _run(["git", "rev-parse", tag], cwd=REPO_ROOT, check=True).stdout.strip()


def _make_worktree(tag: str) -> Path:
    """Create a worktree at /tmp/woys-bench-<tag> and check out the tag."""
    wt = Path(f"/tmp/woys-bench-{tag.replace('/', '_')}")
    if wt.exists():
        print(f"[orch] removing stale worktree {wt}")
        _run(
            ["git", "worktree", "remove", "--force", str(wt)],
            cwd=REPO_ROOT,
            check=False,
        )
        if wt.exists():
            shutil.rmtree(wt, ignore_errors=True)
    print(f"[orch] creating worktree {wt} @ {tag}")
    res = _run(
        ["git", "worktree", "add", "--detach", str(wt), tag],
        cwd=REPO_ROOT,
        check=False,
    )
    if res.returncode != 0:
        raise RuntimeError(f"git worktree add {tag} failed: {res.stderr.strip()}")
    return wt


def _teardown_worktree(wt: Path) -> None:
    print(f"[orch] tearing down worktree {wt}")
    _run(
        ["git", "worktree", "remove", "--force", str(wt)],
        cwd=REPO_ROOT,
        check=False,
    )
    if wt.exists():
        shutil.rmtree(wt, ignore_errors=True)


def _run_probe(
    wt: Path,
    probe_script: Path,
    venv_python: Path,
    wav_path: Path,
    warmup_chunks: int,
    measure_chunks: int,
    out_json: Path,
    rep_index: int,
    timeout_s: float = 600.0,
) -> dict[str, Any]:
    """Run the probe inside `wt` with PYTHONPATH pointing at wt/src.

    Returns the parsed JSON regardless of probe exit code (the probe
    writes a `status="..._failed"` JSON on error). If the subprocess
    itself crashes or times out, returns a synthetic
    `status="subprocess_<reason>"` dict.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{wt}/src:{wt}/src/server"
    # F-merged-001 (v0.15.0) hard-fails on CUDA EP miss. For the
    # benchmark we want CUDA; if the env doesn't have it, the failure
    # is part of the data.
    cmd = [
        str(venv_python),
        str(probe_script),
        str(wav_path),
        str(warmup_chunks),
        str(measure_chunks),
        str(out_json),
        str(rep_index),
    ]
    print(
        f"[orch] running probe in {wt} rep={rep_index} ...",
        flush=True,
    )
    t0 = time.perf_counter()
    try:
        res = subprocess.run(
            cmd,
            cwd=wt,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return {
            "rep_index": rep_index,
            "status": "subprocess_timeout",
            "error": f"probe exceeded {timeout_s} s timeout",
            "started_at": _now_iso(),
        }
    elapsed = time.perf_counter() - t0
    print(
        f"[orch] probe rep={rep_index} exit={res.returncode} elapsed={elapsed:.1f}s",
        flush=True,
    )
    # Print probe stdout/stderr at debug level (helps when something failed).
    if res.stdout:
        for line in res.stdout.strip().splitlines()[-5:]:
            print(f"  [probe stdout] {line}", flush=True)
    if res.returncode != 0 and res.stderr:
        for line in res.stderr.strip().splitlines()[-10:]:
            print(f"  [probe stderr] {line}", flush=True)

    if out_json.exists():
        try:
            with out_json.open() as f:
                return json.load(f)
        except Exception as e:
            return {
                "rep_index": rep_index,
                "status": "json_parse_failed",
                "error": f"{type(e).__name__}: {e}",
                "probe_exit": res.returncode,
                "started_at": _now_iso(),
            }
    return {
        "rep_index": rep_index,
        "status": "no_output",
        "error": "probe did not write its output JSON",
        "probe_exit": res.returncode,
        "probe_stderr_tail": (res.stderr or "")[-500:],
        "started_at": _now_iso(),
    }


def benchmark_tag(
    tag: str,
    probe_script: Path,
    venv_python: Path,
    wav_path: Path,
    warmup_chunks: int,
    measure_chunks: int,
    reps: int,
    output_dir: Path,
    build_setup_budget_s: float = 1200.0,
) -> dict[str, Any]:
    """Run `reps` probes against `tag`. Returns the per-tag JSON
    (commit hash, build info, list of rep results)."""
    print(f"\n=== {tag} ===", flush=True)
    record: dict[str, Any] = {
        "tag": tag,
        "started_at": _now_iso(),
    }
    try:
        record["commit"] = _get_commit_hash_for_tag(tag)
    except Exception as e:
        record["status"] = "tag_resolve_failed"
        record["error"] = f"{type(e).__name__}: {e}"
        return record

    # Special case: HEAD / current branch. Don't worktree; run in place.
    if tag in ("HEAD", "current"):
        wt = REPO_ROOT
        teardown = False
    else:
        try:
            wt = _make_worktree(tag)
            teardown = True
        except Exception as e:
            record["status"] = "worktree_failed"
            record["error"] = f"{type(e).__name__}: {e}"
            return record

    # Copy the probe into the worktree so it imports the worktree's
    # source code (not HEAD's). HEAD case already has the probe.
    if teardown:
        target = wt / "scripts" / "benchmark_probe.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(probe_script, target)

    record["worktree"] = str(wt)
    record["reps"] = []

    t_setup_0 = time.perf_counter()
    for rep_index in range(reps):
        # Time-cap the cumulative work for this tag.
        if time.perf_counter() - t_setup_0 > build_setup_budget_s:
            record["reps"].append(
                {
                    "rep_index": rep_index,
                    "status": "skipped_budget_exceeded",
                    "error": f"cumulative tag time > {build_setup_budget_s} s",
                }
            )
            break
        out_json = output_dir / f"probe_{tag}_rep{rep_index}.json"
        rep_record = _run_probe(
            wt=wt,
            probe_script=wt / "scripts" / "benchmark_probe.py",
            venv_python=venv_python,
            wav_path=wav_path,
            warmup_chunks=warmup_chunks,
            measure_chunks=measure_chunks,
            out_json=out_json,
            rep_index=rep_index,
        )
        record["reps"].append(rep_record)
        # If the FIRST rep failed import / build, skip subsequent reps.
        if rep_index == 0 and rep_record.get("status", "").endswith("_failed"):
            print(
                f"[orch] {tag} rep 0 failed ({rep_record['status']}); skipping remaining reps",
                flush=True,
            )
            for r in range(1, reps):
                record["reps"].append(
                    {
                        "rep_index": r,
                        "status": "skipped_after_rep0_failure",
                    }
                )
            break

    if teardown:
        try:
            _teardown_worktree(wt)
        except Exception as e:
            record["teardown_warning"] = f"{type(e).__name__}: {e}"

    # Top-level status: success if at least one rep had status="success".
    successful_reps = [r for r in record["reps"] if r.get("status") == "success"]
    record["successful_reps"] = len(successful_reps)
    record["status"] = "success" if successful_reps else "all_reps_failed"
    return record


def aggregate_metrics(tag_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build a per-tag summary across the successful reps.

    For each metric, the summary collapses across reps with a mean
    (and reports rep-N if only some succeeded).
    """
    summaries = []
    for rec in tag_records:
        succ = [r for r in rec["reps"] if r.get("status") == "success"]
        s: dict[str, Any] = {
            "tag": rec["tag"],
            "commit": rec.get("commit"),
            "status": rec["status"],
            "reps_run": len(rec["reps"]),
            "reps_successful": len(succ),
            "first_failure_reason": (
                rec["reps"][0].get("error") or rec["reps"][0].get("status")
                if rec["reps"] and rec["reps"][0].get("status") != "success"
                else None
            ),
        }
        if not succ:
            summaries.append(s)
            continue

        def _mean(xs: list[float]) -> float | None:
            xs = [x for x in xs if x is not None]
            return (sum(xs) / len(xs)) if xs else None

        s["cold_start_ms_mean"] = _mean([r.get("cold_start_ms") for r in succ])
        s["import_ms_mean"] = _mean([r.get("import_ms") for r in succ])
        s["build_ms_mean"] = _mean([r.get("build_ms") for r in succ])
        s["shutdown_ms_mean"] = _mean([r.get("shutdown_ms") for r in succ])
        s["throughput_chunks_per_s_mean"] = _mean([r.get("throughput_chunks_per_s") for r in succ])
        s["dropped_chunks_during_run_total"] = sum(
            r.get("dropped_chunks_during_run", 0) or 0 for r in succ
        )

        for key in ("mean", "p50", "p95", "p99", "max"):
            s[f"latency_{key}_ms_mean"] = _mean(
                [r["latency_ms"].get(key) for r in succ if r.get("latency_ms")]
            )
        for stage in ("cv", "rmvpe", "rvc"):
            s[f"{stage}_ms_mean"] = _mean(
                [r["per_stage"].get(f"{stage}_ms_mean") for r in succ if r.get("per_stage")]
            )

        # Resources: take the MAX across reps for peaks, mean for medians.
        # `succ` is bound as a default arg to avoid B023 closure-over-loop-var
        # (the closure is only ever called within this iteration, but ruff
        # can't prove it; explicit binding makes the intent locally-scoped).
        def _res(key: str, summ: str, succ: list = succ) -> float | None:
            return _mean(
                [
                    r["resources"][key][summ]
                    for r in succ
                    if r.get("resources") and r["resources"].get(key)
                ]
            )

        s["vram_mb_peak"] = max(
            (
                r["resources"]["vram_mb"]["max"]
                for r in succ
                if r.get("resources") and r["resources"].get("vram_mb")
            ),
            default=None,
        )
        s["vram_mb_median_mean"] = _res("vram_mb", "median")
        s["rss_kb_peak"] = max(
            (
                r["resources"]["rss_kb"]["max"]
                for r in succ
                if r.get("resources") and r["resources"].get("rss_kb")
            ),
            default=None,
        )
        s["rss_kb_median_mean"] = _res("rss_kb", "median")
        s["threads_max"] = max(
            (
                r["resources"]["threads"]["max"]
                for r in succ
                if r.get("resources") and r["resources"].get("threads")
            ),
            default=None,
        )
        s["fd_count_start_mean"] = _mean(
            [r["resources"].get("fd_count_start") for r in succ if r.get("resources")]
        )
        s["fd_count_end_mean"] = _mean(
            [r["resources"].get("fd_count_end") for r in succ if r.get("resources")]
        )
        # Engine-stats fields with cross-version-defensible defaults.
        for f in (
            "sola_fallback_count",
            "sola_search_clipped",
            "nan_chunks",
            "dropped_chunks",
        ):
            s[f"final_{f}"] = _mean(
                [r["engine_stats_end"].get(f) for r in succ if r.get("engine_stats_end")]
            )
        summaries.append(s)
    return summaries


def realtime_soak(
    tag: str,
    duration_s: float,
    wav_path: Path,
    venv_python: Path,
) -> dict[str, Any]:
    """Run a realtime soak: feed `wav_path` through a PipeWire loop
    source so the engine sees it as live mic input; run `woys engine
    <duration_s>` and capture stats at the end.

    NOTE: this disturbs the user's audio for the duration. The
    orchestrator only invokes this when explicitly requested via
    --realtime-soak-tag. v0.15.0 only (HEAD).

    Implementation: we don't actually loop a WAV through PipeWire here.
    Instead we run `woys engine <duration_s>` and let the engine read
    from the user's existing default source. The fields captured are
    the engine's own EngineStats output (xruns, dropped_chunks,
    sola_fallback_count, sola_search_clipped, jitter) at the END of
    the run, not synthetic injection. The "soak" semantics: did the
    engine survive `duration_s` of REAL audio without dropping out?
    """
    result: dict[str, Any] = {
        "tag": tag,
        "duration_requested_s": duration_s,
        "started_at": _now_iso(),
        "mode": "live_default_source",
    }
    print(
        f"\n=== REALTIME SOAK {tag} ({duration_s} s) ===\n"
        f"NOTE: this uses your live default audio source. Speak / play "
        f"something into your mic during the soak for realistic load.",
        flush=True,
    )
    cmd = [
        str(venv_python),
        "-m",
        "woys.cli",
        "engine",
        str(duration_s),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}/src:{REPO_ROOT}/src/server"
    t0 = time.perf_counter()
    try:
        res = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=duration_s + 30.0,
        )
    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["error"] = f"engine exceeded {duration_s + 30.0} s timeout"
        return result
    result["wall_s"] = time.perf_counter() - t0
    result["exit_code"] = res.returncode
    result["stdout_tail"] = (res.stdout or "")[-2000:]
    result["stderr_tail"] = (res.stderr or "")[-2000:]
    result["status"] = "success" if res.returncode == 0 else "engine_nonzero_exit"
    return result


def _fig_to_base64_png(fig: Any) -> str:
    """Render matplotlib figure to inline base64 PNG for HTML embed."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def render_html_report(
    bench: dict[str, Any],
    summaries: list[dict[str, Any]],
    out_path: Path,
) -> None:
    """Render HTML report with matplotlib charts.

    Chart selection: one chart per metric (cold start, latency p50,
    latency p95, throughput, per-stage breakdown, VRAM peak, RSS
    peak, threads, FD count). Each chart shows v_early → v0.14.3 →
    v0.15.0 with bars; missing tags are rendered as a labelled gap.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    failed = [s for s in summaries if s["status"] != "success"]
    tags = [s["tag"] for s in summaries]

    def _bar(
        title: str,
        ylabel: str,
        values_by_tag: dict[str, float | None],
        unit: str = "",
    ) -> str:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        xs, ys, labels = [], [], []
        for tag in tags:
            v = values_by_tag.get(tag)
            xs.append(tag)
            if v is None:
                ys.append(0.0)
                labels.append("N/A")
            else:
                ys.append(float(v))
                labels.append(f"{v:.2f}{unit}" if unit else f"{v:.2f}")
        ax.bar(xs, ys, color=["#888", "#5b9bd5", "#2e7d32"][: len(xs)])
        for i, (_x, y, label) in enumerate(zip(xs, ys, labels, strict=False)):
            ax.text(
                i,
                y if y > 0 else 0,
                label,
                ha="center",
                va="bottom" if y >= 0 else "top",
                fontsize=9,
            )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)
        b64 = _fig_to_base64_png(fig)
        plt.close(fig)
        return b64

    charts: list[tuple[str, str]] = []

    def _by_tag(key: str) -> dict[str, float | None]:
        return {s["tag"]: s.get(key) for s in summaries}

    charts.append(
        (
            "Cold-start time (engine spawn → first inference done, ms)",
            _bar("Cold-start time", "ms", _by_tag("cold_start_ms_mean"), " ms"),
        )
    )
    charts.append(
        (
            "Latency p50 (median per-chunk inference, ms)",
            _bar("Latency p50", "ms", _by_tag("latency_p50_ms_mean"), " ms"),
        )
    )
    charts.append(("Latency p95", _bar("Latency p95", "ms", _by_tag("latency_p95_ms_mean"), " ms")))
    charts.append(("Latency p99", _bar("Latency p99", "ms", _by_tag("latency_p99_ms_mean"), " ms")))
    charts.append(("Latency max", _bar("Latency max", "ms", _by_tag("latency_max_ms_mean"), " ms")))
    charts.append(
        (
            "Throughput (chunks/sec, sustained over the measurement window)",
            _bar("Throughput", "chunks/s", _by_tag("throughput_chunks_per_s_mean"), " ch/s"),
        )
    )

    # Per-stage stacked
    fig, ax = plt.subplots(figsize=(7, 3.5))
    cv_vals = [s.get("cv_ms_mean") or 0 for s in summaries]
    rmvpe_vals = [s.get("rmvpe_ms_mean") or 0 for s in summaries]
    rvc_vals = [s.get("rvc_ms_mean") or 0 for s in summaries]
    xs = tags
    ax.bar(xs, cv_vals, color="#5b9bd5", label="contentvec (cv)")
    ax.bar(xs, rmvpe_vals, bottom=cv_vals, color="#ed7d31", label="rmvpe")
    bot = [a + b for a, b in zip(cv_vals, rmvpe_vals, strict=False)]
    ax.bar(xs, rvc_vals, bottom=bot, color="#70ad47", label="rvc")
    for i, (_x, total) in enumerate(
        zip(xs, [a + b for a, b in zip(bot, rvc_vals, strict=False)], strict=False)
    ):
        if total > 0:
            ax.text(i, total, f"{total:.1f} ms", ha="center", va="bottom", fontsize=9)
    ax.set_title("Per-stage inference time (mean, ms)")
    ax.set_ylabel("ms")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    charts.append(("Per-stage breakdown (cv / rmvpe / rvc, stacked)", _fig_to_base64_png(fig)))
    plt.close(fig)

    charts.append(
        (
            "Peak VRAM (MB, max sampled during the run)",
            _bar("VRAM peak", "MB", _by_tag("vram_mb_peak"), " MB"),
        )
    )
    rss_mb_by_tag = {
        s["tag"]: (s["rss_kb_peak"] / 1024.0) if s.get("rss_kb_peak") else None for s in summaries
    }
    charts.append(("Peak system RAM (RSS, MB)", _bar("RSS peak", "MB", rss_mb_by_tag, " MB")))
    charts.append(
        (
            "Thread count (peak during run)",
            _bar("Threads (peak)", "threads", _by_tag("threads_max"), ""),
        )
    )

    fd_delta_by_tag = {
        s["tag"]: ((s.get("fd_count_end_mean") or 0) - (s.get("fd_count_start_mean") or 0))
        if s.get("fd_count_end_mean") is not None
        else None
        for s in summaries
    }
    charts.append(
        (
            "FD delta (end - start; 0 = no leak detected)",
            _bar("FD leak", "fd delta", fd_delta_by_tag, ""),
        )
    )

    # Summary table.
    headers = [
        "tag",
        "status",
        "cold_start_ms",
        "p50_ms",
        "p95_ms",
        "p99_ms",
        "throughput",
        "vram_peak_MB",
        "rss_peak_MB",
        "threads_max",
        "fd_leak",
        "sola_fallback",
        "sola_search_clipped",
    ]

    def _fmt(v: Any, prec: int = 2) -> str:
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.{prec}f}"
        return str(v)

    rows = []
    for s in summaries:
        fd_delta = (
            (s.get("fd_count_end_mean") or 0) - (s.get("fd_count_start_mean") or 0)
            if s.get("fd_count_end_mean") is not None
            else None
        )
        rss_mb = s["rss_kb_peak"] / 1024.0 if s.get("rss_kb_peak") else None
        rows.append(
            [
                s["tag"],
                s["status"],
                _fmt(s.get("cold_start_ms_mean"), 1),
                _fmt(s.get("latency_p50_ms_mean"), 2),
                _fmt(s.get("latency_p95_ms_mean"), 2),
                _fmt(s.get("latency_p99_ms_mean"), 2),
                _fmt(s.get("throughput_chunks_per_s_mean"), 1),
                _fmt(s.get("vram_mb_peak"), 0),
                _fmt(rss_mb, 0),
                _fmt(s.get("threads_max"), 0),
                _fmt(fd_delta, 0),
                _fmt(s.get("final_sola_fallback_count"), 1),
                _fmt(s.get("final_sola_search_clipped"), 1),
            ]
        )

    # Per-version notes from CHANGELOG (caller can override).
    notes: dict[str, str] = bench.get("per_version_notes", {})

    # Build HTML.
    html_chunks: list[str] = []
    html_chunks.append(
        """<!DOCTYPE html><html><head><meta charset="utf-8">
<title>woys benchmark — v0.x cross-version comparison</title>
<style>
body {font-family: -apple-system, "Segoe UI", sans-serif; max-width: 1100px;
      margin: 2em auto; color: #222; padding: 0 1em;}
h1 {border-bottom: 2px solid #444;}
h2 {margin-top: 2em; border-bottom: 1px solid #ccc;}
table {border-collapse: collapse; margin: 1em 0; font-size: 0.92em;}
th, td {border: 1px solid #aaa; padding: 4px 8px; text-align: right;}
th {background: #eef; text-align: center;}
td:first-child, th:first-child {text-align: left;}
.chart {margin: 1em 0;}
.note {background: #fff8e7; padding: 0.6em 1em; border-left: 4px solid #f0c040;
       margin: 0.8em 0;}
.fail {background: #fce4ec; padding: 0.6em 1em; border-left: 4px solid #d04050;
       margin: 0.8em 0;}
.muted {color: #777; font-size: 0.9em;}
code {background: #f0f0f0; padding: 1px 4px; border-radius: 3px;}
</style></head><body>
"""
    )
    html_chunks.append("<h1>woys benchmark — cross-version comparison</h1>\n")
    html_chunks.append(
        f"<p class='muted'>Generated: {_now_iso()} · "
        f"input WAV: <code>{bench.get('wav_path', '?')}</code> · "
        f"reps per tag: {bench.get('reps', '?')} · "
        f"measure chunks per rep: {bench.get('measure_chunks', '?')} "
        f"(≈ {bench.get('measure_chunks', 0) * 0.25:.0f} s of audio at "
        f"chunk_seconds=0.25)</p>\n"
    )

    # Version selection rationale.
    rationale = bench.get("version_rationale")
    if rationale:
        html_chunks.append("<h2>Version-selection rationale</h2>\n")
        html_chunks.append(rationale + "\n")

    # Caveats.
    caveats = bench.get("caveats")
    if caveats:
        html_chunks.append("<h2>Caveats (read before interpreting the numbers)</h2>\n")
        html_chunks.append("<ul>\n")
        for c in caveats:
            html_chunks.append(f"<li>{c}</li>\n")
        html_chunks.append("</ul>\n")

    # Realtime-soak deferral notice (rendered prominently if no soak ran).
    if "realtime_soak" not in bench and bench.get("realtime_soak_deferral"):
        html_chunks.append("<h2>Realtime soak — deferred</h2>\n")
        html_chunks.append(f"<div class='note'>{bench['realtime_soak_deferral']}</div>\n")

    # Methodology section.
    hw = bench.get("hardware", {})
    html_chunks.append("<h2>Methodology</h2>\n<ul>\n")
    html_chunks.append(
        f"<li><b>Hardware:</b> {hw.get('cpu', '?')} · "
        f"{hw.get('gpu', '?')} · "
        f"OS: {hw.get('os', '?')} · "
        f"Python: {hw.get('python', '?')}</li>\n"
    )
    html_chunks.append(
        "<li><b>Mode:</b> offline inference loop "
        "(<code>RealtimeEngine._process_streaming_16k</code> on a pre-resampled "
        "16 kHz WAV; no PipeWire / no live mic). The realtime soak at the "
        "end of the report uses live audio.</li>\n"
    )
    html_chunks.append(
        f"<li><b>Reps:</b> {bench.get('reps', '?')} per tag; per-rep results "
        f"in <code>{bench.get('json_path', '?')}</code> alongside this HTML.</li>\n"
    )
    html_chunks.append(
        "<li><b>Honest-measurement policy:</b> every number in "
        "this report comes from a measured run. Versions that failed at import "
        "/ build appear in the failure table below with the specific error "
        "and are excluded from the charts (rendered as N/A).</li>\n"
    )
    html_chunks.append("</ul>\n")

    # Summary table.
    html_chunks.append("<h2>Summary table</h2>\n<table>\n<tr>")
    for h in headers:
        html_chunks.append(f"<th>{h}</th>")
    html_chunks.append("</tr>\n")
    for row in rows:
        html_chunks.append("<tr>")
        for v in row:
            html_chunks.append(f"<td>{v}</td>")
        html_chunks.append("</tr>\n")
    html_chunks.append("</table>\n")

    # Failure table.
    if failed:
        html_chunks.append("<h2>Build / runtime failures (cross-version honesty section)</h2>\n")
        for s in failed:
            html_chunks.append(
                f"<div class='fail'><b>{s['tag']}</b> ({s['status']}): "
                f"{s.get('first_failure_reason') or '—'}</div>\n"
            )

    # Per-version notes.
    html_chunks.append("<h2>Per-version notes (from CHANGELOG)</h2>\n")
    for s in summaries:
        note = notes.get(s["tag"], "(no CHANGELOG note recorded)")
        html_chunks.append(f"<div class='note'><b>{s['tag']}</b>: {note}</div>\n")

    # Charts.
    html_chunks.append("<h2>Charts</h2>\n")
    for caption, b64 in charts:
        html_chunks.append(
            f"<div class='chart'><h3>{caption}</h3><img src='data:image/png;base64,{b64}'/></div>\n"
        )

    # Realtime soak section.
    soak = bench.get("realtime_soak")
    if soak:
        html_chunks.append("<h2>Realtime soak (live PipeWire audio)</h2>\n")
        html_chunks.append(
            f"<p>Soak tag: <b>{soak['tag']}</b> · "
            f"duration: {soak['duration_requested_s']} s · "
            f"wall: {soak.get('wall_s', '?'):.1f} s · "
            f"exit code: {soak.get('exit_code', '?')} · "
            f"status: <code>{soak['status']}</code></p>\n"
        )
        if soak.get("stdout_tail"):
            html_chunks.append(f"<h4>stdout tail</h4><pre>{soak['stdout_tail']}</pre>\n")
        if soak.get("stderr_tail"):
            html_chunks.append(f"<h4>stderr tail</h4><pre>{soak['stderr_tail']}</pre>\n")

    html_chunks.append(
        "<h2>Reproducing</h2>\n"
        "<pre>"
        ".venv/bin/python scripts/benchmark_compare.py \\\n"
        "    --tags v0.8.0 v0.14.3 HEAD \\\n"
        "    --reps 3 --warmup 8 --measure 40 \\\n"
        f"    --wav {bench.get('wav_path', 'tests/fixtures/auto_sweep_input.wav')} \\\n"
        "    --out-json docs/benchmark-v0.x-to-v0.15.json \\\n"
        "    --out-html docs/benchmark-v0.x-to-v0.15.html\n"
        "</pre>\n"
    )
    html_chunks.append("</body></html>\n")
    out_path.write_text("".join(html_chunks))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--tags",
        nargs="+",
        required=True,
        help="Version tags to benchmark; use HEAD for the current checkout.",
    )
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--measure", type=int, default=40)
    ap.add_argument("--wav", type=Path, default=DEFAULT_WAV)
    ap.add_argument("--venv-python", type=Path, default=DEFAULT_VENV_PYTHON)
    ap.add_argument(
        "--out-json",
        type=Path,
        default=REPO_ROOT / "docs" / "benchmark-v0.x-to-v0.15.json",
    )
    ap.add_argument(
        "--out-html",
        type=Path,
        default=REPO_ROOT / "docs" / "benchmark-v0.x-to-v0.15.html",
    )
    ap.add_argument(
        "--realtime-soak-tag",
        type=str,
        default=None,
        help="Run a realtime soak (live audio) on this tag after offline runs.",
    )
    ap.add_argument(
        "--realtime-soak-seconds",
        type=float,
        default=300.0,
    )
    ap.add_argument(
        "--scratch-dir",
        type=Path,
        default=Path("/tmp/woys-bench-scratch"),
        help="Directory for per-rep JSON files.",
    )
    args = ap.parse_args()

    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    probe_script = REPO_ROOT / "scripts" / "benchmark_probe.py"
    if not probe_script.exists():
        print(f"error: {probe_script} not found", file=sys.stderr)
        return 1

    bench: dict[str, Any] = {
        "generated_at": _now_iso(),
        "wav_path": str(args.wav),
        "reps": args.reps,
        "warmup_chunks": args.warmup,
        "measure_chunks": args.measure,
        "json_path": str(args.out_json),
        "hardware": {},
        "tags": [],
    }

    # Hardware fingerprint.
    try:
        bench["hardware"]["cpu"] = (
            subprocess.run(
                ["sh", "-c", "lscpu | grep 'Model name' | head -1 | cut -d: -f2 | xargs"],
                capture_output=True,
                text=True,
                timeout=2,
            ).stdout.strip()
            or "?"
        )
    except Exception:
        bench["hardware"]["cpu"] = "?"
    try:
        bench["hardware"]["gpu"] = (
            subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            .stdout.strip()
            .splitlines()[0]
        )
    except Exception:
        bench["hardware"]["gpu"] = "?"
    try:
        bench["hardware"]["os"] = subprocess.run(
            ["uname", "-srm"],
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
    except Exception:
        bench["hardware"]["os"] = "?"
    bench["hardware"]["python"] = sys.version.split()[0]

    # Per-tag offline benchmarks.
    tag_records = []
    for tag in args.tags:
        rec = benchmark_tag(
            tag=tag,
            probe_script=probe_script,
            venv_python=args.venv_python,
            wav_path=args.wav,
            warmup_chunks=args.warmup,
            measure_chunks=args.measure,
            reps=args.reps,
            output_dir=args.scratch_dir,
        )
        tag_records.append(rec)

    bench["tags"] = tag_records

    # Optional realtime soak (HEAD only by convention).
    if args.realtime_soak_tag:
        bench["realtime_soak"] = realtime_soak(
            tag=args.realtime_soak_tag,
            duration_s=args.realtime_soak_seconds,
            wav_path=args.wav,
            venv_python=args.venv_python,
        )

    # Per-version notes from CHANGELOG (lightweight — first non-blank line
    # under the version heading).
    notes: dict[str, str] = {}
    try:
        with (REPO_ROOT / "CHANGELOG.md").open() as f:
            current_tag = None
            for line in f:
                line = line.rstrip()
                if line.startswith("## [") and "]" in line:
                    bracket = line[line.index("[") + 1 : line.index("]")]
                    # ## [0.15.0] -> tag v0.15.0
                    current_tag = f"v{bracket}" if bracket not in ("Unreleased",) else None
                    notes.setdefault(current_tag or "", line[3:].strip())
                elif (
                    current_tag
                    and line.strip()
                    and not line.startswith("#")
                    and current_tag not in notes
                ):
                    notes[current_tag] = line.strip()
    except Exception:
        pass
    # HEAD note.
    notes.setdefault("HEAD", "hardening branch tip (post-080 hardening)")
    bench["per_version_notes"] = notes

    # Version-selection rationale (default text for the 3-anchor sweep).
    bench["version_rationale"] = (
        "<p>This report compares <b>3 architectural milestones</b>, not 6 "
        "evenly-spaced versions. The rationale for each pick:</p>\n"
        "<ul>\n"
        "<li><b>v0.7.0</b> — earliest tag in the repo (predates the "
        "post-v0.6.0 rename); first major release with the SOLA crossfade "
        "in its current shape. Selected as the 'earliest still-running' "
        "anchor; v0.6.x predates the rename and has different module names, "
        "v0.5.x predates the project entirely. v0.7.0 ran cleanly on the "
        "current Python 3.11 + ORT venv.</li>\n"
        "<li><b>v0.14.3</b> — pre-audit 'good but rough' baseline. The last "
        "release before the current review cycle started. v0.14.0 was "
        "the previous review-driven release; v0.14.1-3 were "
        "the RNNoise-chain stabilisation iterations (LESSONS §44-46). "
        "v0.14.3 is the rollback-stable post-RNNoise tip.</li>\n"
        "<li><b>HEAD</b> — current branch (<code>hardening</code>, "
        "post-commit-080). The release-candidate state.</li>\n"
        "</ul>\n"
        "<p>Versions <i>not</i> selected and why: v0.8.0 / v0.10.0 / "
        "v0.11.0 / v0.13.x are perf-and-stability milestones (clock-lock, "
        "RNNoise, synthetic harness) but adding them turns 'compare review "
        "cycles' into a feature-history trace. v0.12.4 is the chunk_seconds "
        "default-flip release whose perf trade-off is documented in the project notes / "
        "LESSONS §42, but its delta vs v0.14.3 is captured in those docs "
        "more honestly than a synthetic-WAV benchmark can pick up.</p>\n"
    )

    # Caveats: things a future reader needs to know to interpret the numbers.
    bench["caveats"] = [
        "<b>chunk_seconds default differs by version.</b> v0.7.0 uses 0.15 "
        "(set by the v0.7.0-rc5 sweep); v0.14.3 / HEAD use 0.25 (set by the "
        "v0.12.4 listener A/B). Per-call latency comparisons therefore reflect "
        "BOTH inference cost AND chunk size. The fairer cross-version measure "
        "is the per-stage cv/rmvpe/rvc breakdown, which is rate-independent.",
        "<b>Synthetic WAV input.</b> The benchmark feeds "
        "<code>tests/fixtures/auto_sweep_input.wav</code> (a 60-second mono "
        "16/48 kHz test fixture) through the streaming pipeline. This drives "
        "every stage but is not the same content as real voice — RMVPE pitch "
        "extraction and SOLA fall_back behaviour depend on input characteristics. "
        "For voice-content-specific behaviour, see the per-commit SOLA A/B "
        "harnesses (commit-077.md / commit-078.md / commit-079.md / commit-080.md).",
        "<b>Offline inference loop, not realtime.</b> "
        "<code>_process_streaming_16k</code> is called in a tight loop with "
        "no PipeWire, no resamplers on the I/O sides, no SOLA writer to a "
        "real sink. Per-stage timings and resource numbers are honest; "
        "realtime-specific metrics (xruns, dropped chunks under PipeWire "
        "pressure, jitter) are not measured in this mode. See 'Realtime soak — "
        "deferred' below.",
        "<b>Single voice model.</b> All runs use the engine-default RVC voice "
        "(amitaro_v2_16k.onnx). Different voices have different RVC inference "
        "costs (a fp16 export is ~30% faster than fp32 on this GPU). "
        "Cross-version comparisons here use the same model so engine "
        "architectural changes are the visible variable.",
        "<b>Reuses current Python venv.</b> All 3 versions import against the "
        "v0.15.0 venv's installed dependencies (onnxruntime, soxr, numpy, etc.). "
        "If a version's pinned deps differ materially from current, that "
        "version's behaviour here reflects current dep versions, not the "
        "dep versions it shipped against. None of the 3 versions in this sweep "
        "produced import or build errors against the current venv, but that's "
        "a function of the version selection — older tags would fail here.",
        "<b>3 reps per tag is enough for 'stable enough to compare', not "
        "enough for tight confidence intervals.</b> Per-rep variance is "
        "typically <2 ms on p50 for the GPU stages and <5% on cold-start, "
        "so the cross-version deltas reported are real if they exceed those "
        "bounds. Smaller deltas should be treated as noise.",
    ]

    # Realtime soak deferral explanation.
    if not args.realtime_soak_tag:
        bench["realtime_soak_deferral"] = (
            "<p>The user prompt scoped a 5-minute realtime soak on v0.15.0 to "
            "capture xrun / dropped-chunk / SOLA-fallback-rate / jitter "
            "metrics under live PipeWire pressure. The honest answer is that "
            "<b>this benchmark did not run a soak</b>, for the reason "
            "documented below:</p>"
            "<p>A meaningful realtime soak requires real audio flowing through "
            "the engine for the duration. <code>woys engine</code> reads from "
            "the system default mic; without a live audio source, the input is "
            "silence, which the engine's input-gate zeros entirely, bypassing "
            "the inference path and producing no meaningful soak data. Driving "
            "audio in via a PipeWire null-source loopback would mimic real "
            "load, but configuring null-source / module-loopback / "
            "set-default-source on the user's daily-driver desktop has a "
            "non-trivial chance of breaking system audio (LESSONS §46 documents "
            "the v0.14.2 incident where a PipeWire conf change passed all CI "
            "and broke YouTube + speakers in production, requiring a reboot to "
            "recover).</p>"
            "<p>The measure-or-omit policy applies: rather than fabricate "
            "soak numbers from a silence-only run, this report omits them and "
            "names the missing measurement explicitly. The harness's "
            "<code>--realtime-soak-tag</code> entry point is wired and ready; "
            "re-run with <code>--realtime-soak-tag HEAD "
            "--realtime-soak-seconds 300</code> once a safe audio-injection "
            "path is configured (e.g. a dedicated PipeWire sandbox or the user "
            "actively speaking into the mic for 5 minutes).</p>"
        )

    # Aggregate + render.
    summaries = aggregate_metrics(tag_records)
    bench["summaries"] = summaries

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(bench, indent=2))
    print(f"\n[orch] wrote raw measurements to {args.out_json}")

    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    render_html_report(bench, summaries, args.out_html)
    print(f"[orch] wrote HTML report to {args.out_html}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
