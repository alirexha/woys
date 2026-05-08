"""Entry point. Subcommands are wired in Phase 1+ as each module lands."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from woys import __version__

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
        prog="woys",
        description="Linux-native real-time voice changer (RVC + ONNX + PipeWire).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=False, metavar="COMMAND")

    sub.add_parser("info", help="print runtime info (CUDA, PipeWire, models)")

    pw = sub.add_parser("pw", help="manage the persistent PipeWire virtual mic")
    pw_sub = pw.add_subparsers(dest="pw_cmd", required=True, metavar="ACTION")

    pw_setup = pw_sub.add_parser("setup", help="create woys-mic (idempotent)")
    pw_setup.add_argument("--rate", type=int, default=48_000, help="sample rate (Hz)")
    pw_setup.add_argument("--channels", type=int, default=2, help="channel count")

    pw_sub.add_parser("teardown", help="remove woys-mic and the sink")
    pw_sub.add_parser("status", help="report whether the mic is currently loaded")

    run = sub.add_parser("run", help="launch the TUI engine controller")
    run.add_argument(
        "--no-pw-setup",
        action="store_true",
        help="skip the auto-creation of woys-mic (use pavucontrol manually)",
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
        "OFF by default — engine writes only to WoysSink.",
    )

    sub.add_parser("toggle", help="toggle a running TUI's engine on/off")
    sub.add_parser("status", help="ask a running TUI for its status")
    sub.add_parser(
        "slow",
        help="dump per-stage timing for chunks that ran over budget (v0.6.9)",
    )
    pitch_p = sub.add_parser(
        "pitch",
        help="bump pitch shift in a running TUI (e.g. `woys pitch +2`)",
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
    convert_p.add_argument(
        "--yes-i-trust-the-pickle",
        dest="trust_pickle",
        action="store_true",
        help=(
            "consent to load arbitrary code from this .pth via torch.load — "
            "needed for older RVC checkpoints whose unpickle constructors "
            "torch's safe-load mode rejects. Only pass this for files you "
            "trust (ones you trained, or from a verified source)."
        ),
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
    p_exp = prof_sub.add_parser("export", help="write a profile to a shareable .vcprofile file")
    p_exp.add_argument("name")
    p_exp.add_argument("-o", "--output", required=True, help="output .vcprofile path")
    p_imp = prof_sub.add_parser("import", help="load a .vcprofile into config.toml")
    p_imp.add_argument("path", help="path to a .vcprofile file")
    p_imp.add_argument(
        "--name",
        default=None,
        help="rename the imported profile (default: use the name embedded in the file)",
    )

    sub.add_parser(
        "tray",
        help="launch the optional system-tray icon (requires the [tray] extra)",
    )

    diag_p = sub.add_parser(
        "diag",
        help="run a short engine self-test and report audio-pipeline health "
        "(xruns, queue-fulls, watchdog restarts, write-jitter)",
    )
    diag_p.add_argument(
        "--seconds",
        type=float,
        default=10.0,
        help="how long to run the engine for the self-test (default: 10)",
    )
    diag_p.add_argument(
        "--no-engine",
        action="store_true",
        help="skip the engine run; only report static info (CUDA, PipeWire, sink state)",
    )

    # v0.8.0-rc2 — engine without the TUI. Same RealtimeEngine + same
    # InferenceClient subprocess spawn, but no Textual hijacking
    # stderr — useful for headless smoke testing the production path
    # (autonomous CC iteration, CI, debugging the multiprocessing
    # layer). Behaves like `woys run --autostart` minus the
    # interactive UI: starts engine, prints status every second,
    # runs until SIGINT or --seconds elapses.
    eng_p = sub.add_parser(
        "engine",
        help="run the engine without the TUI (headless smoke / CI / debug)",
    )
    eng_p.add_argument(
        "--seconds",
        type=float,
        default=0.0,
        help="run for this many seconds then exit (0 = run until SIGINT)",
    )
    eng_p.add_argument(
        "--quiet",
        action="store_true",
        help="suppress the per-second status prints; only print the final summary",
    )

    return parser


def cmd_info() -> int:
    import shutil
    import subprocess

    print(f"woys {__version__}")
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
        f"woys-mic ready  "
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
    print("woys-mic removed.")
    return 0


def cmd_diag(seconds: float, no_engine: bool) -> int:
    """v0.5.2 — engine + audio-pipeline self-test.

    Runs the realtime engine against the configured mic for `seconds` and
    prints the v0.5.2 audio-health counters at the end. Useful for the
    user to confirm a clean session before declaring "no underruns" — and
    for diagnosing third-party audio issues (Discord noise suppression,
    aggressive PipeWire suspend, etc.) without launching the TUI.
    """
    import time

    print(f"woys diag — {__version__}")
    print("---- environment ----")
    cmd_info()  # cuda + pipewire-server versions, gpu

    if no_engine:
        return 0

    # Lazy import — diag without --no-engine is the only path that needs ORT.
    from audio.engine import EngineConfig, RealtimeEngine
    from audio.pipewire import PipeWireError, VirtualMic, get_state
    from tui.config import load_config

    print("---- pipewire ----")
    try:
        VirtualMic().ensure()
        st = get_state()
        print(f"  sink={st.sink_present} source={st.source_present}")
    except PipeWireError as e:
        print(f"  error: {e}")
        return 2

    print(f"---- engine self-test ({seconds:.1f} s) ----")
    cfg = load_config()
    rvc_path = Path(cfg.rvc_model) if cfg.rvc_model and Path(cfg.rvc_model).exists() else None
    engine_cfg = EngineConfig(
        f0_up_key=cfg.f0_up_key,
        sid=cfg.sid,
        chunk_seconds=cfg.chunk_seconds,
        sink_name=cfg.sink_name,
        monitor=cfg.monitor,
        output_latency_ms=cfg.output_latency_ms,
        output_process_time_ms=cfg.output_process_time_ms,
        embedder=cfg.embedder,
        sola_enabled=cfg.sola_enabled,
        sola_crossfade_ms=cfg.sola_crossfade_ms,
        sola_search_ms=cfg.sola_search_ms,
        sola_context_ms=cfg.sola_context_ms,
        input_gain_db=cfg.input_gain_db,
        # v0.7.0-rc4 — pre-rc4 these silently fell back to the
        # dataclass defaults, so `woys diag` was always running
        # the rc1+ defaults regardless of what the user had in
        # `config.toml`. See `docs/16-audit/synthesis.md`.
        input_gate_dbfs=cfg.input_gate_dbfs,
        input_gate_hysteresis_ms=cfg.input_gate_hysteresis_ms,
        prefer_pw_cat=cfg.prefer_pw_cat,
        prefer_native_pw=cfg.prefer_native_pw,
    )
    if rvc_path is not None:
        engine_cfg.rvc_model = rvc_path
    engine = RealtimeEngine(engine_cfg)
    engine.start()
    try:
        # Sample stats every 0.5 s so the user sees progress on long runs.
        deadline = time.perf_counter() + seconds
        while time.perf_counter() < deadline:
            time.sleep(0.5)
            s = engine.stats
            if s.last_error and "respawned" not in s.last_error:
                # Surface non-recovery errors immediately.
                print(f"  [warn] {s.last_error}")
    finally:
        engine.stop(timeout=2.0)

    s = engine.stats
    print("---- results ----")
    # v0.8.0-rc4 — surface the inference path explicitly so silent
    # fallbacks (rc2 corruption bug) can never hide again.
    if engine.cfg.inference_subprocess:
        if s.child_pid is not None:
            print(f"  inference path   : SUBPROCESS (child pid={s.child_pid})")
        else:
            print("  inference path   : IN-PROCESS (subprocess requested but NOT running!)")
    else:
        print("  inference path   : IN-PROCESS (legacy, by config)")
    # v0.8.1 — per-session TRT status. Show which sessions actually
    # use TRT EP and which fell back to CUDA EP because TRT init
    # failed (e.g. RMVPE FP16 STFT, RVC Int64 binding edge cases).
    if engine.cfg.use_tensorrt:
        active = s.trt_active_for or {}
        if active:
            for model_name, is_trt in sorted(active.items()):
                tag = "TRT" if is_trt else "CUDA (TRT init failed)"
                print(f"  trt[{model_name}] : {tag}")
            for model_name, err in sorted(s.trt_init_errors.items()):
                print(f"  trt error[{model_name}]: {err[:100]}...")
        else:
            print("  tensorrt         : enabled but no sessions loaded yet")
    else:
        print("  tensorrt         : disabled by config")
    print(f"  player backend   : {engine.player_backend or 'unknown'}")
    print(f"  chunks_processed : {s.chunks_processed}")
    print(f"  avg total e2e    : {s.avg_total_ms:.1f} ms")
    print(f"  avg inference    : {s.avg_inference_ms:.1f} ms")
    print(f"  writer jitter    : {s.writer_jitter_ms:.1f} ms (target <5% of chunk)")
    print(f"  player xruns     : {s.xruns}  (pacat-only — pw-cat does not report)")
    print(f"  native-pw under. : {s.player_underruns}  (native-pw helper only)")
    print(f"  queue-full events: {s.queue_full_events}")
    print(f"  player restarts  : {s.player_restarts}")
    # v0.7.0-rc4/rc5 — silent-drop counters previously invisible to
    # woys-diag. rc5 dropped `sola_drain_ms` (zero-pad bookkeeping) —
    # SOLA emits constant-size chunks now and never pads silence; see
    # `docs/16-audit/11-rc4-postmortem.md`. `sola_fallback_count` is
    # now a pure "alignment search gave up" diagnostic and does NOT
    # imply audio cuts under rc5.
    print(f"  input overflows  : {s.input_overflows}")
    print(f"  gated chunks     : {s.gated_chunks}  (input gate fired)")
    print(f"  nan chunks       : {s.nan_chunks}  (RVC vocoder NaN sanitize)")
    print(
        f"  sola fallbacks   : {s.sola_fallback_count}  "
        f"(alignment search peak corr below threshold; emit length unaffected in rc5+)"
    )
    print(f"  dropped chunks   : {s.dropped_chunks}  (inference exceptions)")
    # v0.7.0-rc5 — inference budget overrun rate. `late_chunks` is
    # already stored; the ratio is the threading-tax visibility surface
    # the rc4 postmortem flagged as missing. > ~0.05 means the engine
    # routinely runs past its chunk budget — that's the v0.8.x
    # threading-tax track, not something rc5 attempts to fix.
    if s.chunks_processed > 0:
        overrun = s.late_chunks / s.chunks_processed
        print(
            f"  overrun ratio    : {overrun:.3f}  "
            f"({s.late_chunks}/{s.chunks_processed} chunks past budget)"
        )

    # v0.7.0-rc6 — per-stage producer-side timing percentiles. The rc5
    # writer-jitter probe (docs/16-audit/12-rc5-writer-jitter-probe.md)
    # ruled out the writer side. Producer-side cadence variance is
    # what writer_jitter_ms reflects; this breakdown attributes it to
    # mic_read vs inference vs enqueue_lag. Pure instrumentation —
    # no behavior change, no fix proposed.
    import numpy as _np

    def _pct(samples: list[float] | None, p: float) -> float:
        if not samples:
            return float("nan")
        return float(_np.percentile(samples, p))

    inf_samples = s.inference_samples()
    mic_samples = s.mic_read_samples_ms()
    enq_samples = s.enqueue_lag_samples_ms()
    print("  ---- per-stage timing (rolling window, ms) ----")
    print(
        f"  inference        p50={_pct(inf_samples, 50):6.2f}  "
        f"p95={_pct(inf_samples, 95):6.2f}  "
        f"p99={_pct(inf_samples, 99):6.2f}  "
        f"max={s.max_inference_ms:6.2f}  (n={len(inf_samples)})"
    )
    print(
        f"  mic_read         p50={_pct(mic_samples, 50):6.2f}  "
        f"p95={_pct(mic_samples, 95):6.2f}  "
        f"p99={_pct(mic_samples, 99):6.2f}  "
        f"(n={len(mic_samples)}; should hover near {s.last_mic_read_ms:.0f}ms = "
        f"chunk_seconds * 1000)"
    )
    print(
        f"  enqueue_lag      p50={_pct(enq_samples, 50):6.2f}  "
        f"p95={_pct(enq_samples, 95):6.2f}  "
        f"p99={_pct(enq_samples, 99):6.2f}  "
        f"(n={len(enq_samples)}; should be sub-ms in steady state)"
    )

    # v0.7.0-rc8 — inference tail samples. Captures chunks where
    # inf_ms > 2× running p50, with the input shape + per-session-
    # stage breakdown so we can read what slow chunks have in common.
    # Empty list means no chunks crossed the 2× threshold during this
    # session (rare; usually means inference was very stable).
    if s.tail_chunk_log:
        print(f"  ---- inference tail samples ({len(s.tail_chunk_log)} entries) ----")
        print("  cols: chunk_idx  inf_ms (vs p50_ref)  cv  rmvpe  rvc  audio16_len  mic_read  rms")
        for r in s.tail_chunk_log:
            print(
                f"  #{int(r['chunk_idx']):4d}  "
                f"{r['inf_ms']:6.1f} (vs {r['inf_p50_ref']:5.1f})  "
                f"cv={r['cv_ms']:5.1f}  "
                f"rmvpe={r['rmvpe_ms']:5.1f}  "
                f"rvc={r['rvc_ms']:5.1f}  "
                f"a16={int(r['audio16_len']):5d}  "
                f"mic={r['mic_read_ms']:5.1f}  "
                f"rms={r['input_rms']:.4f}"
            )

    if s.last_error:
        print(f"  last_error       : {s.last_error}")

    # Exit non-zero if we saw any underruns or restarts — useful for CI
    # / shell scripting on top of this command.
    return 1 if (s.xruns or s.queue_full_events or s.player_restarts) else 0


def cmd_engine(seconds: float, quiet: bool) -> int:
    """v0.8.0-rc2 — headless engine entry. Same engine + InferenceClient
    spawn path the TUI uses, minus Textual's terminal hijacking. SIGINT
    triggers a clean stop+teardown."""
    import signal as _signal
    import time as _time

    from audio.engine import EngineConfig, RealtimeEngine
    from audio.pipewire import PipeWireError, VirtualMic, get_state
    from tui.config import load_config

    print(f"woys engine — {__version__}")
    try:
        VirtualMic().ensure()
        st = get_state()
        if not (st.sink_present and st.source_present):
            print("error: WoysSink + woys-mic not loaded; run `woys pw setup`", file=sys.stderr)
            return 2
    except PipeWireError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    cfg = load_config()
    rvc_path = Path(cfg.rvc_model) if cfg.rvc_model and Path(cfg.rvc_model).exists() else None
    engine_cfg = EngineConfig(
        f0_up_key=cfg.f0_up_key,
        sid=cfg.sid,
        chunk_seconds=cfg.chunk_seconds,
        sink_name=cfg.sink_name,
        monitor=cfg.monitor,
        output_latency_ms=cfg.output_latency_ms,
        output_process_time_ms=cfg.output_process_time_ms,
        embedder=cfg.embedder,
        sola_enabled=cfg.sola_enabled,
        sola_crossfade_ms=cfg.sola_crossfade_ms,
        sola_search_ms=cfg.sola_search_ms,
        sola_context_ms=cfg.sola_context_ms,
        input_gain_db=cfg.input_gain_db,
        input_gate_dbfs=cfg.input_gate_dbfs,
        input_gate_hysteresis_ms=cfg.input_gate_hysteresis_ms,
        prefer_pw_cat=cfg.prefer_pw_cat,
        prefer_native_pw=cfg.prefer_native_pw,
    )
    if rvc_path is not None:
        engine_cfg.rvc_model = rvc_path

    eng = RealtimeEngine(engine_cfg)
    print("starting engine...")
    eng.start()
    print(
        f"engine running. child_pid={eng.stats.child_pid} "
        f"rvc_output_sr={eng._rvc_output_sr} active_embedder={eng.active_embedder}"
    )

    stop = {"now": False}

    def _on_sigint(signum: int, _frame: object) -> None:
        del signum
        stop["now"] = True

    _signal.signal(_signal.SIGINT, _on_sigint)
    _signal.signal(_signal.SIGTERM, _on_sigint)

    # v0.8.0-rc4 — capture child_pid + last_error BEFORE eng.stop()
    # since stop() clears them as part of subprocess teardown. We
    # want to know what the engine actually ran with, not its
    # post-shutdown state.
    started_child_pid = eng.stats.child_pid
    started_with_subprocess = eng.cfg.inference_subprocess

    deadline = _time.perf_counter() + seconds if seconds > 0 else float("inf")
    try:
        while not stop["now"] and _time.perf_counter() < deadline:
            _time.sleep(1.0)
            if not quiet:
                s = eng.stats
                print(
                    f"  chunks={s.chunks_processed} "
                    f"avg_inf={s.avg_inference_ms:.1f}ms "
                    f"writer_jitter={s.writer_jitter_ms:.1f}ms "
                    f"xruns={s.xruns} "
                    f"queue_full={s.queue_full_events} "
                    f"dropped={s.dropped_chunks} "
                    f"child_alive={eng._inf_client.is_alive if eng._inf_client else 'n/a'}"
                )
        running_last_error = eng.stats.last_error
    finally:
        print("stopping engine...")
        eng.stop(timeout=2.0)
        s = eng.stats
        # Use the pre-stop snapshot, not the post-stop state.
        if started_with_subprocess:
            if started_child_pid is not None:
                path = f"SUBPROCESS (child pid={started_child_pid})"
            else:
                path = "IN-PROCESS (subprocess startup never set child_pid)"
        else:
            path = "IN-PROCESS (legacy, by config)"
        print(f"inference path: {path}")
        # Print last_error from BEFORE stop() — stop() may set its own.
        if running_last_error:
            print(f"last_error (during run): {running_last_error}")
        elif s.last_error:
            print(f"last_error (post-stop): {s.last_error}")
        print(
            f"final: chunks={s.chunks_processed} "
            f"avg_inf={s.avg_inference_ms:.1f}ms "
            f"writer_jitter={s.writer_jitter_ms:.1f}ms "
            f"xruns={s.xruns} "
            f"queue_full={s.queue_full_events} "
            f"dropped={s.dropped_chunks}"
        )
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


def _prewarm_mp_resource_tracker() -> None:
    """v0.8.0-rc2 — force `multiprocessing.resource_tracker` to spawn its
    daemon NOW, while `sys.stderr.fileno()` is still a real fd.

    Why this exists: the resource_tracker daemon is lazily started on the
    first `SharedMemory(create=True)` call. Its spawn passes
    `sys.stderr.fileno()` as a `fds_to_keep` entry. Inside Textual's
    `on_mount`, `sys.stderr` has been replaced with a wrapper whose
    `fileno()` returns -1, causing
    `_posixsubprocess.fork_exec` to raise
    `ValueError: bad value(s) in fds_to_keep`. v0.8.0-rc1 hit this
    immediately on `woys run --autostart` because `engine.start()` (and
    its `InferenceClient.start()` → `SharedMemory(create=True)`) fires
    inside `WoysApp.on_mount`, after stderr is hijacked.

    The fix: create + immediately destroy a tiny SharedMemory here, at
    the entry of `cli.main()`, BEFORE any TUI import. The first call
    spawns the resource_tracker daemon with a real stderr fd. Subsequent
    SharedMemory creations (including the ones inside Textual's
    on_mount) reuse the already-running tracker — no respawn, no
    fileno-of-bad-stream needed.

    Cost: one shm create + close + unlink ≈ 200 µs. Once per process
    lifetime. Skipped silently on platforms without SharedMemory or
    when /dev/shm isn't writable (subprocess inference is disabled by
    `cfg.inference_subprocess=False` in those cases).
    """
    try:
        from multiprocessing import shared_memory  # noqa: PLC0415

        _shm = shared_memory.SharedMemory(create=True, size=8)
        _shm.close()
        _shm.unlink()
    except Exception:
        # /dev/shm unwritable, sandbox restrictions, etc. — leave the
        # tracker un-warmed; subprocess inference will fail later with
        # a clearer error message in InferenceClient.start().
        pass


def main(argv: list[str] | None = None) -> int:
    # MUST run before any Textual import. See `_prewarm_mp_resource_tracker`.
    _prewarm_mp_resource_tracker()
    parser = build_parser()
    args = parser.parse_args(argv)
    # `woys` with no subcommand launches the TUI with autostart — same as
    # `woys run --autostart`. Helpful for "type the app name to open it"
    # ergonomics. `woys --help` and `woys --version` still work because
    # argparse intercepts those before reaching this dispatch.
    if args.cmd is None:
        from tui.app import run_tui

        return run_tui(no_pw_setup=False, autostart=True, monitor=None)
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
        from woys.convert import cli_convert

        return cli_convert(
            args.pth,
            output=args.output,
            opset=getattr(args, "opset", 17),
            fp16=getattr(args, "fp16", False),
            trust_pickle=getattr(args, "trust_pickle", False),
        )
    if args.cmd == "fp16-convert":
        from woys.fp16_convert import cli_fp16_convert

        targets = ["rmvpe"]
        if args.include_contentvec:
            targets.append("contentvec")
        return cli_fp16_convert(targets, force=args.force)
    if args.cmd == "models":
        from woys.models import cli_models_download, cli_models_list, cli_models_use

        if args.models_cmd == "list":
            return cli_models_list()
        if args.models_cmd == "download":
            return cli_models_download(args.repo)
        if args.models_cmd == "use":
            return cli_models_use(args.name)
    if args.cmd == "profile":
        from woys.profiles import (
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
        if args.profile_cmd == "export":
            from woys.vcprofile import cli_profile_export

            return cli_profile_export(args.name, args.output)
        if args.profile_cmd == "import":
            from woys.vcprofile import cli_profile_import

            return cli_profile_import(args.path, args.name)
    if args.cmd == "tray":
        from woys.tray import cli_tray

        return cli_tray()
    if args.cmd == "diag":
        return cmd_diag(args.seconds, args.no_engine)
    if args.cmd == "engine":
        return cmd_engine(args.seconds, args.quiet)
    if args.cmd in ("toggle", "status", "pitch", "slow"):
        from tui.control import send_command

        if args.cmd == "toggle":
            print(send_command("TOGGLE"))
        elif args.cmd == "status":
            print(send_command("STATUS"))
        elif args.cmd == "slow":
            print(send_command("SLOW"))
            # B13: read from the same XDG_RUNTIME_DIR location the TUI writes.
            from tui.control import runtime_path

            slow_path = runtime_path("slow-chunks.txt")
            if slow_path.exists():
                print("---")
                print(slow_path.read_text(), end="")
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
