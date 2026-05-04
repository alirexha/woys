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

    run = sub.add_parser("run", help="launch the TUI engine controller")
    run.add_argument(
        "--no-pw-setup",
        action="store_true",
        help="skip the auto-creation of vcclient-mic (use pavucontrol manually)",
    )
    run.add_argument(
        "--autostart",
        action="store_true",
        help="start the engine immediately on TUI launch",
    )
    run.add_argument(
        "--monitor",
        action="store_true",
        default=None,
        help="also play transformed audio to your default output (self-monitor). "
        "OFF by default — engine writes only to VCClientCachySink.",
    )

    sub.add_parser("toggle", help="toggle a running TUI's engine on/off")
    sub.add_parser("status", help="ask a running TUI for its status")
    pitch_p = sub.add_parser(
        "pitch",
        help="bump pitch shift in a running TUI (e.g. `vcclient-cachy pitch +2`)",
    )
    pitch_p.add_argument("delta", help="signed integer or `0` to reset")

    convert_p = sub.add_parser(
        "convert",
        help="convert a .pth RVC checkpoint to .onnx",
    )
    convert_p.add_argument("pth", help="path to a .pth RVC checkpoint")
    convert_p.add_argument(
        "-o", "--output", default=None, help="output .onnx path (defaults next to input)"
    )
    convert_p.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    convert_p.add_argument(
        "--fp16",
        action="store_true",
        help="export weights in fp16 — RVC v2 only, validate quality before shipping",
    )

    fp16_p = sub.add_parser(
        "fp16-convert",
        help="produce fp16 ONNX siblings of the foundation models (saves VRAM)",
    )
    fp16_p.add_argument(
        "--include-contentvec",
        action="store_true",
        help="also convert contentvec (lower quality — not auto-loaded)",
    )
    fp16_p.add_argument("--force", action="store_true", help="overwrite existing fp16 files")

    models_p = sub.add_parser("models", help="manage the RVC voice-model library")
    models_sub = models_p.add_subparsers(dest="models_cmd", required=True, metavar="ACTION")
    models_sub.add_parser("list", help="show installed voice models")
    dl_p = models_sub.add_parser("download", help="fetch all ONNX models from a HuggingFace repo")
    dl_p.add_argument("repo", help='e.g. "wok000/vcclient_model"')
    use_p = models_sub.add_parser("use", help="set the active RVC model in config.toml")
    use_p.add_argument("name", help="model name (file stem) or absolute path to a .onnx")

    prof_p = sub.add_parser("profile", help="named state snapshots — model + pitch + chunk + ...")
    prof_sub = prof_p.add_subparsers(dest="profile_cmd", required=True, metavar="ACTION")
    p_save = prof_sub.add_parser("save", help="snapshot the current config under a name")
    p_save.add_argument("name")
    p_use = prof_sub.add_parser("use", help="apply a saved profile to the active config")
    p_use.add_argument("name")
    prof_sub.add_parser("list", help="show saved profiles")
    p_del = prof_sub.add_parser("delete", help="remove a saved profile")
    p_del.add_argument("name")

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
    if args.cmd == "run":
        from tui.app import run_tui

        return run_tui(
            no_pw_setup=args.no_pw_setup,
            autostart=args.autostart,
            monitor=args.monitor,
        )
    if args.cmd == "convert":
        from vcclient_cachy.convert import cli_convert

        return cli_convert(
            args.pth,
            output=args.output,
            opset=getattr(args, "opset", 17),
            fp16=getattr(args, "fp16", False),
        )
    if args.cmd == "fp16-convert":
        from vcclient_cachy.fp16_convert import cli_fp16_convert

        targets = ["rmvpe"]
        if args.include_contentvec:
            targets.append("contentvec")
        return cli_fp16_convert(targets, force=args.force)
    if args.cmd == "models":
        from vcclient_cachy.models import cli_models_download, cli_models_list, cli_models_use

        if args.models_cmd == "list":
            return cli_models_list()
        if args.models_cmd == "download":
            return cli_models_download(args.repo)
        if args.models_cmd == "use":
            return cli_models_use(args.name)
    if args.cmd == "profile":
        from vcclient_cachy.profiles import (
            cli_profile_delete,
            cli_profile_list,
            cli_profile_save,
            cli_profile_use,
        )

        if args.profile_cmd == "save":
            return cli_profile_save(args.name)
        if args.profile_cmd == "use":
            return cli_profile_use(args.name)
        if args.profile_cmd == "list":
            return cli_profile_list()
        if args.profile_cmd == "delete":
            return cli_profile_delete(args.name)
    if args.cmd in ("toggle", "status", "pitch"):
        from tui.control import send_command

        if args.cmd == "toggle":
            print(send_command("TOGGLE"))
        elif args.cmd == "status":
            print(send_command("STATUS"))
        else:  # pitch
            delta = args.delta.lstrip("+") if args.delta.startswith("+") else args.delta
            try:
                int(delta)
            except ValueError:
                print(f"error: pitch must be an integer (got {args.delta!r})", file=sys.stderr)
                return 2
            print(send_command(f"PITCH {delta}"))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
