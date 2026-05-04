"""Entry point. Subcommands are wired in Phase 1+ as each module lands."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vcclient_cachy import __version__

# Upstream-style imports (from voice_changer.X) require src/server/ on sys.path.
_SERVER_ROOT = Path(__file__).resolve().parent.parent / "server"
if _SERVER_ROOT.is_dir() and str(_SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(_SERVER_ROOT))

# `src/audio` and `src/tui` are top-level packages; ensure src/ is reachable so
# `from audio.pipewire import VirtualMic` works when running from a checkout.
_SRC_ROOT = Path(__file__).resolve().parent.parent
if _SRC_ROOT.is_dir() and str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vcclient-cachy",
        description="Linux-native real-time voice changer (RVC + ONNX + PipeWire).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=False, metavar="COMMAND")

    sub.add_parser("info", help="print runtime info (CUDA, PipeWire, models)")

    pw = sub.add_parser("pw", help="manage the persistent PipeWire virtual mic")
    pw_sub = pw.add_subparsers(dest="pw_cmd", required=True, metavar="ACTION")

    pw_setup = pw_sub.add_parser("setup", help="create vcclient-mic (idempotent)")
    pw_setup.add_argument("--rate", type=int, default=48_000, help="sample rate (Hz)")
    pw_setup.add_argument("--channels", type=int, default=2, help="channel count")

    pw_sub.add_parser("teardown", help="remove vcclient-mic and the sink")
    pw_sub.add_parser("status", help="report whether the mic is currently loaded")

    return parser


def cmd_info() -> int:
    import shutil
    import subprocess

    print(f"vcclient-cachy {__version__}")
    print(f"  python: {sys.version.split()[0]}")
    pactl = shutil.which("pactl")
    if pactl:
        out = subprocess.run([pactl, "info"], capture_output=True, text=True, timeout=3)
        for line in out.stdout.splitlines():
            if "Server Name" in line or "Server Version" in line:
                print(f"  {line.strip()}")
    nvsmi = shutil.which("nvidia-smi")
    if nvsmi:
        out = subprocess.run(
            [nvsmi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if out.returncode == 0:
            print(f"  gpu: {out.stdout.strip()}")
    return 0


def cmd_pw_setup(rate: int, channels: int) -> int:
    from audio.pipewire import PipeWireError, VirtualMic

    try:
        vm = VirtualMic(rate=rate, channels=channels)
        state = vm.ensure()
    except PipeWireError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(
        f"vcclient-mic ready  "
        f"(sink_module={state.sink_module_id}, source_module={state.source_module_id})"
    )
    return 0


def cmd_pw_teardown() -> int:
    from audio.pipewire import PipeWireError, VirtualMic

    try:
        VirtualMic().teardown()
    except PipeWireError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print("vcclient-mic removed.")
    return 0


def cmd_pw_status() -> int:
    from audio.pipewire import PipeWireError, ensure_pipewire, get_state

    try:
        ensure_pipewire()
    except PipeWireError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    state = get_state()
    print(
        f"sink_present  : {state.sink_present}"
        + (f"  (module {state.sink_module_id})" if state.sink_present else "")
    )
    print(
        f"source_present: {state.source_present}"
        + (f"  (module {state.source_module_id})" if state.source_present else "")
    )
    return 0 if state.fully_present else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "info":
        return cmd_info()
    if args.cmd == "pw":
        if args.pw_cmd == "setup":
            return cmd_pw_setup(args.rate, args.channels)
        if args.pw_cmd == "teardown":
            return cmd_pw_teardown()
        if args.pw_cmd == "status":
            return cmd_pw_status()
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
