#!/usr/bin/env python
"""Profile the woys realtime engine with py-spy.

The engine sub-thread's per-chunk timing has been opaque since v0.6.x:
LESSONS §19 documents an ~80 ms threading tax that adds wall-time on
top of standalone-bench inference, but no profile has ever attributed
it to specific functions. The rc4 postmortem
(`docs/16-audit/11-rc4-postmortem.md`) flagged `writer_jitter_ms = 63.8`
at chunk_seconds=150 ms cadence as evidence that the threading tax is
the next P0 to attack — but doing so requires a real profile, not
code reading.

This script wraps py-spy with the right flags for that profile so the
user doesn't have to remember them between debug sessions.

Usage:

  # Start the engine in one terminal (any of these — they all spawn
  # the `woys-engine` sub-thread the profiler attaches to):
  woys run --autostart        # full TUI
  woys diag --duration 60     # CLI diag mode
  python -m woys              # short-form

  # Then in another terminal, while audio is flowing:
  ./scripts/profile_engine.py --duration 30

  # Or with an explicit pid if pgrep can't find it:
  ./scripts/profile_engine.py --pid 12345 --duration 30

Output is written to `/tmp/woys-engine-profile-<timestamp>.svg`. Open
the SVG in a browser for an interactive flame graph.

Requires py-spy (pip install py-spy) and either CAP_SYS_PTRACE on the
binary or `sudo` (the script will exec sudo automatically if the
caller doesn't already have ptrace permission).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def find_engine_pid() -> int | None:
    """Locate the `woys-engine` thread's owning process via pgrep.

    The engine runs as a daemon thread inside the woys CLI / TUI
    process; we want the PID of that parent. pgrep -f matches against
    the full command line, which catches both `woys run` and
    `python -m woys` and `woys diag`.
    """
    try:
        result = subprocess.run(
            ["pgrep", "-fa", "woys"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0:
        return None
    # Output: "<pid> <cmdline>". Filter out our own process and pgrep's.
    self_pid = os.getpid()
    candidates: list[int] = []
    for line in result.stdout.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        cmdline = parts[1]
        if pid == self_pid:
            continue
        if "profile_engine" in cmdline:
            continue
        # Match anything that's running woys (CLI, TUI, diag).
        if "woys" in cmdline.lower():
            candidates.append(pid)
    if not candidates:
        return None
    if len(candidates) > 1:
        print(
            f"[profile_engine] multiple woys processes found: {candidates}; "
            f"picking the first. Pass --pid to target a specific one.",
            file=sys.stderr,
        )
    return candidates[0]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="py-spy wrapper for the woys realtime engine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--pid",
        type=int,
        default=None,
        help="Target PID. Default: pgrep for the running woys process.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Sample duration in seconds. Default 30. The user should be "
        "talking through woys for this whole window so the profile "
        "captures realistic engine load.",
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=200,
        help="Sample rate in Hz. Default 200 (every 5 ms — enough to "
        "resolve the 150 ms chunk cadence and finer GIL events).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output SVG path. Default: /tmp/woys-engine-profile-<ts>.svg",
    )
    args = parser.parse_args()

    py_spy = shutil.which("py-spy")
    if py_spy is None:
        print(
            "[profile_engine] py-spy not found on PATH.\n"
            "  install: pip install py-spy   (or your preferred python pkg manager)\n"
            "  then re-run this script.",
            file=sys.stderr,
        )
        return 1

    pid = args.pid
    if pid is None:
        pid = find_engine_pid()
        if pid is None:
            print(
                "[profile_engine] no woys process found via pgrep.\n"
                "  start the engine first (e.g. `woys run --autostart`),\n"
                "  then re-run this script in another terminal.",
                file=sys.stderr,
            )
            return 2
        print(f"[profile_engine] auto-detected pid={pid}", file=sys.stderr)

    if args.output is None:
        ts = time.strftime("%Y%m%d-%H%M%S")
        output = Path(f"/tmp/woys-engine-profile-{ts}.svg")
    else:
        output = args.output

    cmd = [
        py_spy,
        "record",
        "--pid",
        str(pid),
        "--output",
        str(output),
        "--duration",
        str(args.duration),
        "--rate",
        str(args.rate),
        "--threads",  # capture all threads incl. writer / watchdog / stderr-reader
        "--idle",  # include time spent in syscalls (read, flush, get) — what we
        # actually care about for writer-thread jitter
    ]

    # py-spy needs ptrace — re-exec under sudo if the caller doesn't
    # already have it. /proc/sys/kernel/yama/ptrace_scope=0 would also
    # work but we don't presume to set that.
    if os.geteuid() != 0:
        sudo = shutil.which("sudo")
        if sudo is not None:
            cmd = [sudo, "--preserve-env=PATH", *cmd]

    print(
        f"[profile_engine] recording {args.duration}s at {args.rate}Hz → {output}",
        file=sys.stderr,
    )
    print(f"[profile_engine] cmd: {' '.join(cmd)}", file=sys.stderr)
    print(
        "[profile_engine] talk through woys / Telegram while this runs so the\n"
        "                 profile captures realistic engine load.",
        file=sys.stderr,
    )

    rc = subprocess.run(cmd, check=False).returncode
    if rc != 0:
        print(
            f"[profile_engine] py-spy exited non-zero ({rc}); profile may be incomplete.",
            file=sys.stderr,
        )
        return rc

    print(f"[profile_engine] flame graph at {output}", file=sys.stderr)
    print(
        f"[profile_engine] open in a browser: xdg-open {output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
