"""Minimal Textual TUI for vcclient-cachy.

Shows live engine state (running, mic RMS, latency), exposes pitch shift via
keys, persists changes to `~/.config/vcclient-cachy/config.toml`.

Keys
----
  t       toggle engine on/off
  +/-     pitch shift up/down (1 semitone)
  0       reset pitch shift
  s       force-save config
  q       quit
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.reactive import reactive
from textual.widgets import Footer, Header, Label, ProgressBar, Static

from audio import EngineConfig, RealtimeEngine
from audio.pipewire import PipeWireError, VirtualMic
from tui.config import AppConfig, load_config, save_config
from tui.control import ControlServer


class StatusPanel(Static):
    """Top status block: model, on/off, pitch."""

    DEFAULT_CSS = """
    StatusPanel {
        padding: 1 2;
        border: round $accent;
        height: 7;
    }
    """

    def render_status(self, *, running: bool, model: Path, pitch: int, error: str | None) -> str:
        light = "[bold green]●[/]" if running else "[dim]○[/]"
        err = f"\n[bold red]error:[/] {error}" if error else ""
        return (
            f"{light}  status: [bold]{'RUNNING' if running else 'stopped'}[/]\n"
            f"   model: [italic]{model.name or '(none)'}[/]\n"
            f"   pitch: {pitch:+d} st"
            f"{err}"
        )


class LatencyPanel(Static):
    """Mid latency block: avg total, avg inference."""

    DEFAULT_CSS = """
    LatencyPanel { padding: 1 2; border: round $accent; height: 6; }
    """

    def render_lat(self, total_ms: float, inf_ms: float, chunks: int) -> str:
        return (
            f"avg total e2e : [bold]{total_ms:6.1f} ms[/]\n"
            f"avg inference : {inf_ms:6.1f} ms\n"
            f"chunks done   : {chunks}"
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
        self.engine = engine or RealtimeEngine(
            EngineConfig(
                chunk_seconds=self.cfg.chunk_seconds,
                mic_rate=self.cfg.mic_rate,
                sink_rate=self.cfg.sink_rate,
                f0_up_key=self.cfg.f0_up_key,
                sid=self.cfg.sid,
                sink_name=self.cfg.sink_name,
                monitor=self.cfg.monitor,
                output_latency_ms=self.cfg.output_latency_ms,
                embedder=self.cfg.embedder,
                sola_enabled=self.cfg.sola_enabled,
                sola_crossfade_ms=self.cfg.sola_crossfade_ms,
                sola_search_ms=self.cfg.sola_search_ms,
                sola_context_ms=self.cfg.sola_context_ms,
            )
        )
        self.no_pw_setup = no_pw_setup
        self.pitch = self.cfg.f0_up_key
        self._control = ControlServer(self._handle_control)

    def on_mount(self) -> None:
        if not self.no_pw_setup:
            try:
                VirtualMic().ensure()
            except PipeWireError as e:
                self.engine.stats.last_error = f"PipeWire: {e}"
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
            return (
                f"OK running={s.running} "
                f"pitch={int(self.pitch)} "
                f"avg_total_ms={s.avg_total_ms:.1f} "
                f"avg_inf_ms={s.avg_inference_ms:.1f}"
            )
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
        else:
            self._start_engine()

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
            status = self.query_one("#status", StatusPanel)
            status.update(
                status.render_status(
                    running=s.running,
                    model=self.engine.cfg.rvc_model,
                    pitch=int(self.pitch),
                    error=s.last_error,
                )
            )
            lat = self.query_one("#latency", LatencyPanel)
            lat.update(lat.render_lat(s.avg_total_ms, s.avg_inference_ms, s.chunks_processed))
            meter = self.query_one("#meter", ProgressBar)
            meter.update(progress=min(100, int(s.last_input_rms * 4 * 100)))
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
