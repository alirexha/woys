"""Minimal Textual TUI for woys.

Shows live engine state (running, mic RMS, latency), exposes pitch shift via
keys, persists changes to `~/.config/woys/config.toml`.

Keys
----
  t       toggle engine on/off
  +/-     pitch shift up/down (1 semitone)
  0       reset pitch shift
  p       cycle through saved profiles  (v0.3.0)
  s       force-save config
  q       quit
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.reactive import reactive
from textual.widgets import Footer, Header, Label, ProgressBar, Static

from audio import EngineConfig, RealtimeEngine
from audio.engine import DEFAULT_RVC_MODEL
from audio.pipewire import PipeWireError, VirtualMic
from tui.config import AppConfig, load_config, save_config
from tui.control import ControlServer, JobRegistry
from woys.profiles import apply_profile, cycle_profile, list_profiles


class StatusPanel(Static):
    """Top status block: model, on/off, pitch, active profile, cold-start hint."""

    DEFAULT_CSS = """
    StatusPanel {
        padding: 1 2;
        border: round $accent;
        height: 8;
    }
    """

    def render_status(
        self,
        *,
        running: bool,
        model: Path,
        pitch: int,
        profile: str | None,
        cold_start: bool,
        swapping: str | None,
        error: str | None,
    ) -> str:
        if swapping:
            light = "[bold blue]◴[/]"
            state = f"loading {swapping}…"
        elif running and cold_start:
            light = "[bold yellow]◐[/]"
            state = "warming up…"
        elif running:
            light = "[bold green]●[/]"
            state = "RUNNING"
        else:
            light = "[dim]○[/]"
            state = "stopped"
        prof = f"[italic]{profile}[/]" if profile else "[dim](none)[/]"
        err = f"\n[bold red]error:[/] {error}" if error else ""
        return (
            f"{light}  status:  [bold]{state}[/]\n"
            f"   model:   [italic]{model.name or '(none)'}[/]\n"
            f"   pitch:   {pitch:+d} st\n"
            f"   profile: {prof}"
            f"{err}"
        )


class LatencyPanel(Static):
    """Mid latency block: avg total, avg inference, v0.5.2 audio-health row."""

    DEFAULT_CSS = """
    LatencyPanel { padding: 1 2; border: round $accent; height: 8; }
    """

    def render_lat(
        self,
        total_ms: float,
        inf_ms: float,
        chunks: int,
        xruns: int = 0,
        queue_full: int = 0,
        restarts: int = 0,
        jitter_ms: float = 0.0,
    ) -> str:
        # v0.5.2: highlight any non-zero xrun count in red — the user
        # listens to the audio in another window, this is their visual
        # check that the session is clean.
        xrun_color = "red" if xruns or queue_full else "green"
        return (
            f"avg total e2e : [bold]{total_ms:6.1f} ms[/]\n"
            f"avg inference : {inf_ms:6.1f} ms\n"
            f"chunks done   : {chunks}\n"
            f"audio health  : "
            f"[{xrun_color}]xruns={xruns}[/] "
            f"qfull={queue_full} "
            f"restarts={restarts} "
            f"jitter={jitter_ms:.1f}ms"
        )


class VCClientApp(App[int]):
    CSS = """
    Screen { background: $surface; }
    #meter { margin: 1 2; }
    Label.k { text-style: dim; }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("t", "toggle_engine", "toggle"),
        Binding("plus,equals_sign", "pitch_up", "pitch +"),
        Binding("minus", "pitch_down", "pitch -"),
        Binding("0", "pitch_reset", "pitch 0"),
        Binding("p", "cycle_profile", "profile"),
        Binding("s", "save_cfg", "save"),
        Binding("q,ctrl+c", "quit", "quit"),
    ]

    rms = reactive(0.0)
    pitch = reactive(0)
    running = reactive(False)

    def __init__(
        self,
        *,
        cfg: AppConfig | None = None,
        engine: RealtimeEngine | None = None,
        no_pw_setup: bool = False,
    ) -> None:
        super().__init__()
        self.cfg = cfg or load_config()
        # v0.4.1: honor cfg.rvc_model on startup. Empty string ⇒ use the
        # engine's hardcoded default (Amitaro). Any path that doesn't exist
        # also falls back so a stale config.toml doesn't brick the engine.
        rvc_path = (
            Path(self.cfg.rvc_model)
            if self.cfg.rvc_model and Path(self.cfg.rvc_model).exists()
            else DEFAULT_RVC_MODEL
        )
        self.engine = engine or RealtimeEngine(
            EngineConfig(
                rvc_model=rvc_path,
                chunk_seconds=self.cfg.chunk_seconds,
                mic_rate=self.cfg.mic_rate,
                sink_rate=self.cfg.sink_rate,
                f0_up_key=self.cfg.f0_up_key,
                sid=self.cfg.sid,
                sink_name=self.cfg.sink_name,
                monitor=self.cfg.monitor,
                output_latency_ms=self.cfg.output_latency_ms,
                output_process_time_ms=self.cfg.output_process_time_ms,
                embedder=self.cfg.embedder,
                sola_enabled=self.cfg.sola_enabled,
                sola_crossfade_ms=self.cfg.sola_crossfade_ms,
                sola_search_ms=self.cfg.sola_search_ms,
                sola_context_ms=self.cfg.sola_context_ms,
                input_gain_db=self.cfg.input_gain_db,
            )
        )
        self.no_pw_setup = no_pw_setup
        self.pitch = self.cfg.f0_up_key
        self._control = ControlServer(self._handle_control)
        # v0.5.0: async job table for slow socket commands (MODEL / PROFILE).
        self._jobs = JobRegistry()
        # v0.3.0: track active profile so the status panel + cycle key know.
        self._active_profile: str | None = None
        # v0.5.0: track the latest swap target so the TUI can show "loading X..."
        # while the swap is in flight (~10 ms cached, ~600 ms cold).
        self._swap_in_flight: str | None = None

    def on_mount(self) -> None:
        if not self.no_pw_setup:
            try:
                VirtualMic().ensure()
            except PipeWireError as e:
                self.engine.stats.last_error = f"PipeWire: {e}"
                self.notify(f"PipeWire: {e}", severity="error", timeout=8)
        if self.cfg.autostart_engine:
            self._start_engine()
        self._control.start()
        self.set_interval(0.25, self._refresh_stats)

    # ---- control socket -----------------------------------------------------

    def _handle_control(self, cmd: str) -> str:
        cmd = cmd.strip()
        if cmd == "TOGGLE":
            self.call_from_thread(self.action_toggle_engine)
            return "OK toggled"
        if cmd.startswith("PITCH "):
            arg = cmd[len("PITCH ") :].strip()
            if arg in ("0", "+0", "-0"):
                self.call_from_thread(self.action_pitch_reset)
                return "OK pitch=0"
            try:
                delta = int(arg)
            except ValueError:
                return f"ERR bad pitch: {arg!r}"
            new = int(self.pitch) + delta

            def apply() -> None:
                self.pitch = new
                self.engine.cfg.f0_up_key = new
                self.cfg.f0_up_key = new

            self.call_from_thread(apply)
            return f"OK pitch={new}"
        if cmd == "STATUS":
            s = self.engine.stats
            model_name = Path(str(self.engine.cfg.rvc_model)).name
            return (
                f"OK running={s.running} "
                f"pitch={int(self.pitch)} "
                f"profile={self._active_profile or '-'} "
                f"model={model_name} "
                f"avg_total_ms={s.avg_total_ms:.1f} "
                f"avg_inf_ms={s.avg_inference_ms:.1f} "
                f"max_total_ms={s.max_total_ms:.1f} "
                f"late_chunks={s.late_chunks}/{s.chunks_processed}"
            )
        if cmd == "SLOW":
            # v0.6.9 round 5 — dump slow_chunk_log to a file the user can cat.
            # Socket reply stays small; full breakdown lives on disk so multi-
            # line output isn't truncated by the recv buffer.
            from pathlib import Path as _P

            log = list(self.engine.stats.slow_chunk_log)
            out_path = _P("/tmp/woys-slow-chunks.txt")
            lines = [
                "# slow chunk log — chunks where total_ms > chunk_seconds * 1000",
                f"# session count: {len(log)} late, "
                f"chunks_processed={self.engine.stats.chunks_processed}",
                "# columns: chunk_idx total_ms inf_ms cv_ms rmvpe_ms rvc_ms input_rms",
            ]
            for r in log:
                lines.append(
                    f"#{int(r['chunk_idx'])}: "
                    f"total={r['total_ms']:.1f}ms "
                    f"inf={r['inf_ms']:.1f}ms "
                    f"cv={r['cv_ms']:.1f}ms "
                    f"rmvpe={r['rmvpe_ms']:.1f}ms "
                    f"rvc={r['rvc_ms']:.1f}ms "
                    f"input_rms={r['input_rms']:.4f}"
                )
            out_path.write_text("\n".join(lines) + "\n")
            return f"OK wrote {len(log)} entries to {out_path}"
        if cmd.startswith("MODEL "):
            arg = cmd[len("MODEL ") :].strip()
            from woys.models import find_by_name

            new_path = find_by_name(arg)
            if new_path is None:
                return f"ERR no such model: {arg!r} (try `models list`)"

            # Async path: submit + return job id. The job body queues the
            # swap and waits for the worker to apply it.
            def do_swap() -> None:
                self._swap_in_flight = new_path.name

                def apply_main() -> None:
                    self.engine.request_model_swap(new_path)
                    self.cfg.rvc_model = str(new_path.resolve())
                    # v0.5.0: model swap also updates the active profile name
                    # if a profile saved that exact rvc_model exists. This
                    # keeps STATUS's profile= field in sync with reality.
                    matched = self._profile_for_model_path(new_path)
                    if matched is not None:
                        self._active_profile = matched
                    save_config(self.cfg)

                self.call_from_thread(apply_main)

                # Wait for the engine worker to actually apply the swap.
                deadline = time.time() + 10.0
                while time.time() < deadline:
                    if (
                        self.engine._pending_model_swap is None
                        and Path(str(self.engine.cfg.rvc_model)) == new_path.resolve()
                    ):
                        break
                    time.sleep(0.02)
                self._swap_in_flight = None

            jid = self._jobs.submit(do_swap)
            return f"OK job={jid} model={new_path.name}"
        if cmd.startswith("PROFILE "):
            target = cmd[len("PROFILE ") :].strip()

            def do_profile() -> None:
                self._swap_in_flight = target

                def apply_main() -> None:
                    self._apply_profile_named(target)

                self.call_from_thread(apply_main)
                # Mirror the swap-complete poll above so the JOB reflects the
                # *audio path's* swap, not just the config write.
                deadline = time.time() + 10.0
                while time.time() < deadline:
                    if self.engine._pending_model_swap is None:
                        break
                    time.sleep(0.02)
                self._swap_in_flight = None

            jid = self._jobs.submit(do_profile)
            return f"OK job={jid} profile={target}"
        if cmd.startswith("JOB "):
            jid = cmd[len("JOB ") :].strip()
            return self._jobs.status_line(jid)
        if cmd == "QUIT":
            # action_quit is async; post a sync shim instead so call_from_thread
            # gets a non-coroutine callable (Textual typing requires it).
            def _quit_shim() -> None:
                self.engine.stop()
                self._control.stop()
                save_config(self.cfg)
                self.exit(0)

            self.call_from_thread(_quit_shim)
            return "OK quitting"
        return f"ERR unknown: {cmd!r}"

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Vertical():
            yield StatusPanel(id="status")
            yield LatencyPanel(id="latency")
            yield Label("input level", classes="k")
            yield ProgressBar(total=100, show_eta=False, show_percentage=False, id="meter")
        yield Footer()

    # ---- actions ------------------------------------------------------------

    def action_toggle_engine(self) -> None:
        if self.engine.stats.running:
            self.engine.stop()
            self.notify("engine stopped", severity="information", timeout=2)
        else:
            self._start_engine()
            self.notify("engine starting (cudnn warmup ~2s)", severity="information", timeout=2)

    def _start_engine(self) -> None:
        self.engine.cfg.f0_up_key = int(self.pitch)
        self.engine.start()

    def action_pitch_up(self) -> None:
        self.pitch = int(self.pitch) + 1
        self.engine.cfg.f0_up_key = int(self.pitch)
        self.cfg.f0_up_key = int(self.pitch)

    def action_pitch_down(self) -> None:
        self.pitch = int(self.pitch) - 1
        self.engine.cfg.f0_up_key = int(self.pitch)
        self.cfg.f0_up_key = int(self.pitch)

    def action_pitch_reset(self) -> None:
        self.pitch = 0
        self.engine.cfg.f0_up_key = 0
        self.cfg.f0_up_key = 0

    def action_cycle_profile(self) -> None:
        """Phase 4 — cycle to the next saved profile.

        v0.4.1 fix: this now actually swaps the loaded RVC model.
        v0.5.0 polish: pressing `p` rapidly queues swaps via JobRegistry so
        the TUI never freezes; each swap completes in order, and the
        StatusPanel shows `loading X…` while one is in flight.
        """
        names = list_profiles(self.cfg)
        if not names:
            self.notify(
                "no saved profiles. Use `woys profile save <name>` first.",
                severity="warning",
                timeout=4,
            )
            return
        next_name = cycle_profile(self.cfg, self._active_profile)
        if next_name is None:
            return

        def _runner() -> None:
            self._swap_in_flight = next_name

            def apply_main() -> None:
                self._apply_profile_named(next_name)

            self.call_from_thread(apply_main)
            deadline = time.time() + 10.0
            while time.time() < deadline:
                if self.engine._pending_model_swap is None:
                    break
                time.sleep(0.02)
            self._swap_in_flight = None

        self._jobs.submit(_runner)

    def _profile_for_model_path(self, path: Path) -> str | None:
        """v0.5.0: reverse-lookup a saved profile whose rvc_model matches `path`.

        Used by the MODEL handler to keep `_active_profile` in sync when
        the user invokes `models use <slug>` directly. Returns the first
        matching profile name (alphabetical via `list_profiles`), or None.
        """
        target = str(path.resolve())
        for name in list_profiles(self.cfg):
            bag = self.cfg._extras.get("profiles", {})
            snap = bag.get(name, {}) if isinstance(bag, dict) else {}
            if isinstance(snap, dict) and snap.get("rvc_model") == target:
                return name
        return None

    def _apply_profile_named(self, name: str) -> None:
        """Apply a saved profile to both `self.cfg` and `self.engine`. The
        RVC model swap is queued via `request_model_swap` and takes effect
        at the next chunk boundary in the worker (≤ chunk_seconds + a few
        ms cudnn-cache lookups for the new shape)."""
        if not apply_profile(self.cfg, name):
            self.notify(f"failed to apply profile {name!r}", severity="error", timeout=4)
            return
        self._active_profile = name
        # Mirror live-tunable fields onto the engine config. chunk_seconds /
        # output_latency_ms still need an engine restart to bite (they're
        # set at sounddevice/pacat init).
        self.engine.cfg.f0_up_key = self.cfg.f0_up_key
        self.engine.cfg.sid = self.cfg.sid
        self.engine.cfg.monitor = self.cfg.monitor
        # v0.5.1: input_gain_db is read by the audio worker every chunk,
        # so updating in place takes effect on the next mic chunk without
        # an engine restart.
        self.engine.cfg.input_gain_db = self.cfg.input_gain_db
        self.pitch = self.cfg.f0_up_key
        # The actual model swap — this is the v0.4.1 fix.
        new_model = (
            Path(self.cfg.rvc_model)
            if self.cfg.rvc_model and Path(self.cfg.rvc_model).exists()
            else None
        )
        if new_model is not None and new_model != self.engine.cfg.rvc_model:
            self.engine.request_model_swap(new_model)
            self.notify(
                f"profile → {name} (loading {new_model.name}, pitch {self.cfg.f0_up_key:+d})",
                severity="information",
                timeout=3,
            )
        else:
            self.notify(
                f"profile → {name} (pitch {self.cfg.f0_up_key:+d})",
                severity="information",
                timeout=2,
            )
        save_config(self.cfg)

    def action_save_cfg(self) -> None:
        save_config(self.cfg)
        self.notify("config saved", severity="information")

    async def action_quit(self) -> None:
        self.engine.stop()
        self._control.stop()
        save_config(self.cfg)
        self.exit(0)

    # ---- live refresh -------------------------------------------------------

    def _refresh_stats(self) -> None:
        s = self.engine.stats
        try:
            # "Cold start" heuristic: engine is running but the rolling
            # latency window hasn't stabilized yet — first ~10 chunks at
            # chunk_seconds=0.1 = roughly 1 second of warmup.
            cold_start = bool(s.running and s.chunks_processed < 10)
            status = self.query_one("#status", StatusPanel)
            status.update(
                status.render_status(
                    running=s.running,
                    model=self.engine.cfg.rvc_model,
                    pitch=int(self.pitch),
                    profile=self._active_profile,
                    cold_start=cold_start,
                    swapping=self._swap_in_flight,
                    error=s.last_error,
                )
            )
            lat = self.query_one("#latency", LatencyPanel)
            lat.update(
                lat.render_lat(
                    s.avg_total_ms,
                    s.avg_inference_ms,
                    s.chunks_processed,
                    xruns=s.xruns,
                    queue_full=s.queue_full_events,
                    restarts=s.pacat_restarts,
                    jitter_ms=s.writer_jitter_ms,
                )
            )
            meter = self.query_one("#meter", ProgressBar)
            meter.update(progress=min(100, int(s.last_input_rms * 4 * 100)))

            # Surface a fresh `last_error` to the user as a toast (not just text).
            if s.last_error and s.last_error != getattr(self, "_last_notified_error", None):
                self.notify(s.last_error, severity="error", timeout=8)
                self._last_notified_error = s.last_error
        except Exception:
            # Widget tree may not be fully realized yet during early ticks.
            pass


def run_tui(
    *,
    no_pw_setup: bool = False,
    autostart: bool = False,
    monitor: bool | None = None,
) -> int:
    cfg = load_config()
    if autostart:
        cfg.autostart_engine = True
    if monitor is not None:
        cfg.monitor = monitor
    app = VCClientApp(cfg=cfg, no_pw_setup=no_pw_setup)
    return app.run() or 0
