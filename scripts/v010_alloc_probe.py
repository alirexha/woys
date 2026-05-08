#!/usr/bin/env python
"""v0.10.x — tracemalloc probe for Python alloc churn on the engine hot path.

Wraps the v0.10.x synthetic harness in `tracemalloc` and dumps the top-N
allocation hotspots over a short bounded run. Brief candidate #5 is
"memory allocation churn on the hot path"; this probe gives the data
to confirm or deny.

Usage:
  ./scripts/v010_alloc_probe.py --duration 60 --top 30 --out /tmp/v010_alloc.txt

`tracemalloc` adds non-trivial overhead (~5-15 % on the hot path), so
a 60 s run is enough; longer runs don't add information once the
allocation pattern repeats. Output is a sorted "top N" by size + count
of in-use blocks at snapshot time + by allocations during the window.

Output schema:
  ==== top 30 in-use at end (size, count, location) ====
  ==== top 30 allocations during run (count delta, size delta) ====
  ==== aggregate stats ====
"""

from __future__ import annotations

import argparse
import sys
import time
import tracemalloc
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "src" / "server") not in sys.path:
    sys.path.insert(0, str(REPO / "src" / "server"))


def _format_top(stats: list[tracemalloc.Statistic], top: int) -> list[str]:
    lines = []
    for i, s in enumerate(stats[:top]):
        # Each statistic has size, count, traceback (frames).
        frame = s.traceback[0]
        loc = f"{frame.filename}:{frame.lineno}"
        # Trim path to last 3 components for readability.
        parts = loc.split("/")
        if len(parts) > 3:
            loc = ".../" + "/".join(parts[-3:])
        lines.append(
            f"  #{i + 1:2d}  size={s.size:>10d} B  count={s.count:>6d}  {loc}"
        )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--out", type=Path, default=Path("/tmp/v010_alloc.txt"))
    args = parser.parse_args()

    # Start tracemalloc BEFORE importing engine to capture import-time
    # allocations too. Frames=10 catches enough call stack to attribute
    # numpy ops back to engine.py call sites.
    tracemalloc.start(10)

    # Snapshot AFTER imports so import-time allocs aren't counted as
    # hot-path. We compare snapshot1 (post-warmup) vs snapshot2 (post-run)
    # to attribute allocations to the run window only.
    print(f"[alloc-probe] starting tracemalloc, duration={args.duration:.0f}s", file=sys.stderr)

    # Re-use the harness internals.
    from scripts import v010_harness

    # Take pre-run snapshot after a short warmup so we exclude warmup
    # allocs from the delta (they're already known: model loading,
    # cudnn cache).
    print("[alloc-probe] running 5s warmup ...", file=sys.stderr)
    v010_harness._run_engine_synthetic(
        duration_s=5.0,
        out_path=None,
        enable_sola=True,
        chunk_seconds=None,
        inference_subprocess=False,
    )
    snap_pre = tracemalloc.take_snapshot()
    print(f"[alloc-probe] pre-run snapshot: {len(snap_pre.statistics('lineno')):d} stats", file=sys.stderr)

    print(f"[alloc-probe] running engine for {args.duration:.0f}s ...", file=sys.stderr)
    v010_harness._run_engine_synthetic(
        duration_s=args.duration,
        out_path=None,
        enable_sola=True,
        chunk_seconds=None,
        inference_subprocess=False,
    )
    snap_post = tracemalloc.take_snapshot()

    # Aggregate stats.
    pre_stats = snap_pre.statistics("lineno")
    post_stats = snap_post.statistics("lineno")
    diff = snap_post.compare_to(snap_pre, "lineno")
    cur, peak = tracemalloc.get_traced_memory()

    out_lines = []
    out_lines.append("v0.10.x tracemalloc probe")
    out_lines.append(f"duration_s={args.duration:.0f}  top={args.top}")
    out_lines.append("")
    out_lines.append("==== aggregate ====")
    out_lines.append(f"  current traced memory: {cur:>12d} B  ({cur / (1024 * 1024):.1f} MiB)")
    out_lines.append(f"  peak traced memory:    {peak:>12d} B  ({peak / (1024 * 1024):.1f} MiB)")
    out_lines.append(f"  pre-run statistics:    {len(pre_stats):>6d} unique allocation sites")
    out_lines.append(f"  post-run statistics:   {len(post_stats):>6d} unique allocation sites")
    out_lines.append(f"  diff entries:          {len(diff):>6d}")
    out_lines.append("")

    out_lines.append(f"==== top {args.top} in-use at end ====")
    out_lines.extend(_format_top(post_stats, args.top))
    out_lines.append("")

    out_lines.append(f"==== top {args.top} positive growth (post - pre) ====")
    growth = sorted(
        (s for s in diff if s.size_diff > 0),
        key=lambda x: x.size_diff,
        reverse=True,
    )
    for i, s in enumerate(growth[: args.top]):
        frame = s.traceback[0]
        loc = f"{frame.filename}:{frame.lineno}"
        parts = loc.split("/")
        if len(parts) > 3:
            loc = ".../" + "/".join(parts[-3:])
        out_lines.append(
            f"  #{i + 1:2d}  Δsize={s.size_diff:>+10d} B  Δcount={s.count_diff:>+6d}  {loc}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out_lines) + "\n")
    print(f"[alloc-probe] wrote {args.out}", file=sys.stderr)
    print()
    print("\n".join(out_lines[:60]))
    if len(out_lines) > 60:
        print(f"... (truncated; full output at {args.out})")

    tracemalloc.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
