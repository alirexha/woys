"""Realtime voice-conversion engine.

Wires the Phase 1 ONNX inference path to a real-time mic→infer→sink loop.

Audio routing — IMPORTANT (see v0.1.1 fix)
------------------------------------------
On CachyOS, PortAudio is built with the ALSA host API only (no PulseAudio host
API). `sd.OutputStream()` with no explicit `device=` falls through to the ALSA
*default* device, which routes to the system default sink (laptop speakers /
headphones) — NOT to the named PipeWire sink we want. Setting `PULSE_SINK=…`
in the environment is also ignored, because there's no Pulse host API for
PortAudio to consult.

The fix: instead of `sd.OutputStream`, the engine spawns
`pacat --playback --device=VCClientCachySink …` as a subprocess and pipes
raw float32 PCM to its stdin. `pacat` is the canonical PulseAudio client; it
talks to pipewire-pulse natively, takes an explicit `--device=` argument, and
never auto-routes to the system default. This is the same path that the
acoustic loopback bench (`scripts/bench_loopback.py`) uses — proven on this host.

Input is still `sd.InputStream` against the default mic; that path was always
correct (host mic → 48 kHz capture).

Optional local monitoring
-------------------------
By default, **the engine writes the transformed audio to ONLY the virtual
sink** (which `vcclient-mic` reads from). Nothing plays out of the laptop
speakers — your housemates / streamers / phone calls don't hear what you're
processing. Pass `monitor=True` to additionally play to the host's default
output for self-monitoring.

Threading
---------
- Worker thread runs the blocking I/O loop and feeds the pacat subprocess.
- TUI thread polls `EngineStats` for live UI; no shared mutable state beyond
  a few atomic-ish primitives.
"""

from __future__ import annotations

import contextlib
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

# ORT-GPU 1.20+ on driver 595 needs explicit preload of the pip-shipped CUDA libs.
import onnxruntime as ort

NDArrayF32 = npt.NDArray[np.float32]
NDArrayI64 = npt.NDArray[np.int64]

if hasattr(ort, "preload_dlls"):
    ort.preload_dlls()


MODELS_DIR = Path.home() / ".local" / "share" / "vcclient-cachy" / "models"

# Defaults pulled from Phase 1 inventory.
DEFAULT_RVC_MODEL = MODELS_DIR / "amitaro_v2_16k.onnx"
DEFAULT_RMVPE = MODELS_DIR / "rmvpe_wrapped.onnx"
DEFAULT_CONTENTVEC = MODELS_DIR / "contentvec-f.onnx"


@dataclass
class EngineConfig:
    rvc_model: Path = DEFAULT_RVC_MODEL
    rmvpe_model: Path = DEFAULT_RMVPE
    contentvec_model: Path = DEFAULT_CONTENTVEC

    # Audio I/O
    mic_rate: int = 48_000
    sink_rate: int = 48_000
    # v0.2.0 dropped the default to 100 ms thanks to SOLA crossfade; consecutive
    # chunks share `sola_crossfade_ms` of overlap, so seams stay inaudible at
    # this size on continuous speech.
    chunk_seconds: float = 0.1
    channels: int = 1

    # SOLA crossfade (Phase B). Disable at your peril — without it, audible
    # clicks at every chunk boundary when chunk_seconds is short.
    sola_enabled: bool = True
    sola_crossfade_ms: float = 50.0  # overlap window between consecutive chunks
    sola_search_ms: float = 4.0  # how far to shift looking for in-phase alignment
    # History fed to the model alongside each new chunk so the embedder /
    # vocoder convolutions don't see edge artifacts. Brief calls this "context".
    sola_context_ms: float = 100.0

    # RVC
    f0_up_key: int = 0  # semitones
    sid: int = 0
    threshold: float = 0.3

    # Embedder selection (v0.2.0):
    #   "onnx"    — direct ORT contentvec-f.onnx call (default, fastest, no torch+fairseq)
    #   "fairseq" — upstream FairseqHubert PyTorch path; needs the [fairseq] extra installed.
    # Misconfiguration / missing fairseq → fall back to "onnx" with a warning,
    # never crash the engine. (Brief Phase A.)
    embedder: str = "onnx"

    # Routing
    sink_name: str = "VCClientCachySink"
    input_device: str | int | None = None  # None = default mic
    # When False (default): output goes ONLY to VCClientCachySink → vcclient-mic.
    # When True: ALSO write a best-effort copy to the host's default output
    # (laptop speakers / headphones) for self-monitoring.
    monitor: bool = False
    # Output latency in ms requested from pacat. Lower = tighter latency,
    # higher = more buffer headroom against scheduler jitter.
    output_latency_ms: int = 30


@dataclass
class EngineStats:
    running: bool = False
    chunks_processed: int = 0
    last_input_rms: float = 0.0
    last_inference_ms: float = 0.0
    avg_inference_ms: float = 0.0
    last_total_ms: float = 0.0
    avg_total_ms: float = 0.0
    last_error: str | None = None

    # Rolling latency window for the TUI.
    _recent_inference: deque[float] = field(default_factory=lambda: deque(maxlen=32))
    _recent_total: deque[float] = field(default_factory=lambda: deque(maxlen=32))


def _make_session(path: Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    providers: list[tuple[str, dict[str, object]] | str] = []
    if "CUDAExecutionProvider" in ort.get_available_providers():
        # Mirror the Phase 5 smoke-test options. cudnn_conv_algo_search=EXHAUSTIVE
        # eats a one-off ~50-100 ms autotune at first call but unlocks ~3-4x
        # steady-state inference speed.
        providers.append(
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                },
            )
        )
    providers.append("CPUExecutionProvider")
    return ort.InferenceSession(str(path), sess_options=so, providers=providers)


def _to_pitch_coarse(pitchf: NDArrayF32, target_len: int) -> tuple[NDArrayI64, NDArrayF32]:
    f0_min, f0_max = 50.0, 1100.0
    f0_mel_min = 1127.0 * np.log(1 + f0_min / 700.0)
    f0_mel_max = 1127.0 * np.log(1 + f0_max / 700.0)
    pitch = np.zeros(target_len, dtype=np.float32)
    n = min(len(pitchf), target_len)
    pitch[-n:] = pitchf[:n]
    f0_mel = 1127.0 * np.log(1 + pitch / 700.0)
    mask = f0_mel > 0
    f0_mel[mask] = (f0_mel[mask] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
    f0_mel = np.clip(f0_mel, 1.0, 255.0)
    return np.rint(f0_mel).astype(np.int64), pitch


def _resample_linear(audio: NDArrayF32, src_rate: int, dst_rate: int) -> NDArrayF32:
    """Cheap linear-interp resampler.

    Adequate for Phase 3 — Phase 5 can swap in scipy.signal.resample_poly for
    quality if the difference is audible at the sink.
    """
    if src_rate == dst_rate:
        return audio.astype(np.float32, copy=False)
    n_src = len(audio)
    n_dst = round(n_src * dst_rate / src_rate)
    if n_dst <= 0:
        return np.zeros(0, dtype=np.float32)
    src_idx = np.linspace(0, n_src - 1, n_dst, dtype=np.float64)
    floor = np.floor(src_idx).astype(np.int64)
    ceil = np.minimum(floor + 1, n_src - 1)
    frac = (src_idx - floor).astype(np.float32)
    out: NDArrayF32 = ((1 - frac) * audio[floor] + frac * audio[ceil]).astype(np.float32)
    return out


class _FairseqEmbedder:
    """Wrapper that lazy-loads FairseqHubert; tracks the active mode for stats."""

    def __init__(self, hubert_path: Path) -> None:
        # Imports are deferred until extract() is called so users without the
        # `[fairseq]` extra never pay the torch import cost.
        import torch
        from fairseq import checkpoint_utils  # type: ignore[import-not-found]

        models, _, _ = checkpoint_utils.load_model_ensemble_and_task([str(hubert_path)], suffix="")
        model = models[0]
        model.eval()
        if torch.cuda.is_available():
            model = model.to("cuda:0")
        self._torch = torch
        self.model = model
        self.dev = next(model.parameters()).device

    def extract(self, audio_np: NDArrayF32) -> NDArrayF32:
        torch = self._torch
        with torch.no_grad():
            feats_t = torch.from_numpy(audio_np.reshape(1, -1)).to(self.dev)
            padding_mask = torch.zeros(feats_t.shape, dtype=torch.bool, device=self.dev)
            logits = self.model.extract_features(
                source=feats_t, padding_mask=padding_mask, output_layer=12
            )
            out = logits[0].detach().to(torch.float32).cpu().numpy()
        return out  # type: ignore[no-any-return]


class RealtimeEngine:
    """Owns the 3 ONNX sessions and a worker thread that loops mic→infer→sink."""

    def __init__(self, cfg: EngineConfig | None = None) -> None:
        self.cfg = cfg or EngineConfig()
        self.stats = EngineStats()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Lazy-load sessions; avoid CUDA work if the engine is constructed
        # but never started (e.g., TUI dry-run).
        self._cv: ort.InferenceSession | None = None
        self._rmvpe: ort.InferenceSession | None = None
        self._rvc: ort.InferenceSession | None = None
        self._is_half: bool = False
        self._cv_input_dtype: str = "tensor(float)"
        self._rmvpe_input_dtype: str = "tensor(float)"

        # Active embedder mode after _ensure_sessions(). Either "onnx" or
        # "fairseq" — falls back from "fairseq" to "onnx" automatically if
        # the fairseq import or model load fails (Phase A spec).
        self.active_embedder: str = "onnx"
        self._fairseq: _FairseqEmbedder | None = None

        # SOLA streaming state (Phase B). At 16 kHz: a rolling buffer of
        # past mic input that we prepend to each new chunk before running
        # the model, plus a SOLAStream that crossfades consecutive outputs.
        from audio.sola import SOLAConfig, SOLAStream

        self._sola_cfg = SOLAConfig(
            rate=16_000,
            crossfade_ms=self.cfg.sola_crossfade_ms,
            search_ms=self.cfg.sola_search_ms,
            context_ms=self.cfg.sola_context_ms,
        )
        self._sola: SOLAStream | None = (
            SOLAStream(self._sola_cfg) if self.cfg.sola_enabled else None
        )
        # Past-input buffer at 16 kHz (zero-padded on first call).
        self._input_history: NDArrayF32 = np.zeros(
            self._sola_cfg.context_samples + self._sola_cfg.crossfade_samples,
            dtype=np.float32,
        )

    # ---- model loading ------------------------------------------------------

    def _ensure_sessions(self) -> None:
        # v0.3.0: prefer fp16 variants if present next to the fp32 file. fp16
        # rmvpe halves its VRAM footprint with no measurable pitch-detection
        # quality loss (validated v0.2.0). fp16 contentvec, by contrast, has
        # cosine sim 0.75 vs fp32 — only auto-promoted if explicitly requested.
        cv_path = self._auto_pick_fp16(self.cfg.contentvec_model, allow=False)
        rmvpe_path = self._auto_pick_fp16(self.cfg.rmvpe_model, allow=True)

        if self._cv is None:
            self._cv = _make_session(cv_path)
            self._cv_input_dtype = self._cv.get_inputs()[0].type
        if self._rmvpe is None:
            self._rmvpe = _make_session(rmvpe_path)
            self._rmvpe_input_dtype = self._rmvpe.get_inputs()[0].type
        if self._rvc is None:
            self._rvc = _make_session(self.cfg.rvc_model)
            self._is_half = self._rvc.get_inputs()[0].type != "tensor(float)"

        # Resolve embedder mode. ONNX is the default; "fairseq" is opt-in via
        # config and degrades gracefully when the package isn't installed.
        if self.cfg.embedder == "fairseq" and self._fairseq is None:
            hubert_path = MODELS_DIR / "hubert_base.pt"
            try:
                if not hubert_path.exists():
                    raise FileNotFoundError(
                        f"hubert_base.pt missing at {hubert_path} — "
                        "run `scripts/download_weights.py` to fetch it"
                    )
                self._fairseq = _FairseqEmbedder(hubert_path)
                self.active_embedder = "fairseq"
                print(f"[engine] embedder=fairseq (hubert_base.pt @ {hubert_path})")
            except Exception as e:
                msg = (
                    f"fairseq embedder unavailable ({type(e).__name__}: {e}); "
                    "falling back to ONNX contentvec"
                )
                print(f"[engine] {msg}")
                self.stats.last_error = msg
                self.active_embedder = "onnx"
        else:
            self.active_embedder = "onnx"

    @staticmethod
    def _auto_pick_fp16(fp32_path: Path, *, allow: bool) -> Path:
        """If a `<name>-fp16.onnx` sibling exists and `allow=True`, use it."""
        if not allow:
            return fp32_path
        cand = fp32_path.with_name(fp32_path.stem + "-fp16" + fp32_path.suffix)
        return cand if cand.exists() else fp32_path

    def reload_rvc(self, path: Path) -> None:
        """Hot-swap the RVC voice model."""
        self.cfg.rvc_model = path
        self._rvc = _make_session(path)
        self._is_half = self._rvc.get_inputs()[0].type != "tensor(float)"

    # ---- inference ----------------------------------------------------------

    def _extract_feats(self, audio16k: NDArrayF32) -> NDArrayF32:
        """Embedder dispatch — see `EngineConfig.embedder` and `active_embedder`."""
        if self.active_embedder == "fairseq" and self._fairseq is not None:
            return self._fairseq.extract(audio16k.astype(np.float32, copy=False))
        assert self._cv is not None
        # Cast input to whatever dtype this contentvec ONNX expects (fp16 or fp32).
        in_dtype = np.float16 if "float16" in self._cv_input_dtype else np.float32
        audio_in: np.ndarray = audio16k.reshape(1, -1).astype(in_dtype)  # type: ignore[type-arg]
        feats_raw = self._cv.run(["unit12"], {"audio": audio_in})[0]
        # Always return float32 to the rest of the pipeline.
        feats: NDArrayF32 = feats_raw.astype(np.float32, copy=False)
        return feats

    def process_chunk_16k(self, audio16k: NDArrayF32) -> NDArrayF32:
        """One inference pass on a (N,) float32 chunk at 16 kHz.

        Standalone path — used by tests and by the engine when SOLA is
        disabled. Doesn't touch streaming state. The streaming engine path
        goes through `_process_streaming_16k` instead.
        """
        return self._infer(audio16k)

    def _infer(self, audio16k: NDArrayF32) -> NDArrayF32:
        """Raw model invocation; no streaming bookkeeping."""
        assert self._cv is not None and self._rmvpe is not None and self._rvc is not None

        feats = self._extract_feats(audio16k)
        rm_dtype = np.float16 if "float16" in self._rmvpe_input_dtype else np.float32
        pitchf_raw = self._rmvpe.run(
            ["pitchf"],
            {
                "waveform": audio16k.reshape(1, -1).astype(rm_dtype),
                "threshold": np.array([self.cfg.threshold], dtype=rm_dtype),
            },
        )[0]
        pitchf = pitchf_raw.astype(np.float32).squeeze()

        feats_2x = np.repeat(feats, 2, axis=1)
        pitch_coarse, pitchf_aligned = _to_pitch_coarse(pitchf, target_len=feats_2x.shape[1])
        pitch_coarse = pitch_coarse[: feats_2x.shape[1]].reshape(1, -1)
        # Apply pitch shift in semitones.
        if self.cfg.f0_up_key != 0:
            pitchf_aligned = pitchf_aligned * (2.0 ** (self.cfg.f0_up_key / 12.0))
        pitchf_aligned = pitchf_aligned[: feats_2x.shape[1]].reshape(1, -1).astype(np.float32)

        feats_dtype = np.float16 if self._is_half else np.float32
        out = self._rvc.run(
            ["audio"],
            {
                "feats": feats_2x.astype(feats_dtype),
                "p_len": np.array([feats_2x.shape[1]], dtype=np.int64),
                "pitch": pitch_coarse,
                "pitchf": pitchf_aligned,
                "sid": np.array([self.cfg.sid], dtype=np.int64),
            },
        )[0]
        result: NDArrayF32 = np.array(out).astype(np.float32).squeeze()
        return result

    def _process_streaming_16k(self, new_chunk_16k: NDArrayF32) -> NDArrayF32:
        """Streaming variant. Maintains a sliding input history so the model
        sees overlapping content; SOLA crossfades consecutive outputs.

        Returns audio at 16 kHz that's safe to concatenate with the previous
        emitted chunk (assuming the same SOLA stream). Output length is
        approximately equal to the input length once warmed up.
        """
        cf = self._sola_cfg.crossfade_samples
        ctx = self._sola_cfg.context_samples
        history_len = ctx + cf

        # Build model input: last (ctx + cf) of input history + the new chunk.
        model_input = np.concatenate([self._input_history, new_chunk_16k.astype(np.float32)])
        # Update history for next call: keep the last (ctx + cf) samples of
        # the combined buffer (these will be the leading samples next time).
        self._input_history = model_input[-history_len:].copy()

        full_out = self._infer(model_input)

        # Map the trim from input space to output space proportionally —
        # the model is roughly 1:1 in time, but RVC trims a few samples at
        # the boundaries. Compute the per-sample ratio defensively.
        in_len = model_input.shape[0]
        out_len = full_out.shape[0]
        ratio = out_len / max(in_len, 1)
        # Drop the leading "context" portion in the model output. Keep the
        # last `ctx_drop_out` samples as the part that overlaps with the
        # previous emit + the new chunk's worth of audio.
        ctx_drop_in = max(
            history_len - cf, 0
        )  # samples of pure history (no overlap with prev emit)
        ctx_drop_out = round(ctx_drop_in * ratio)
        emitted_region = full_out[ctx_drop_out:]

        if self._sola is not None:
            return self._sola.process(emitted_region)
        # SOLA disabled — emit raw, expect chunk-boundary clicks for short chunks.
        return emitted_region

    def reset_streaming_state(self) -> None:
        """Clear SOLA + input history so the engine can resume cleanly after a
        stop / start without leaking stale tail from a previous session."""
        self._input_history = np.zeros(
            self._sola_cfg.context_samples + self._sola_cfg.crossfade_samples,
            dtype=np.float32,
        )
        if self._sola is not None:
            self._sola.reset()

    # ---- realtime loop ------------------------------------------------------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._ensure_sessions()
        self.stats.running = True
        self._thread = threading.Thread(target=self._run_loop, name="vcclient-engine", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        self.stats.running = False

    def _open_pacat(self) -> subprocess.Popen[bytes]:
        """Spawn pacat targeting the named virtual sink. Raises if pacat missing."""
        pacat = shutil.which("pacat")
        if pacat is None:
            raise RuntimeError(
                "pacat not found — install pipewire-pulse (it provides pactl/pacat/parec)"
            )
        cmd = [
            pacat,
            "--playback",
            f"--device={self.cfg.sink_name}",
            f"--rate={self.cfg.sink_rate}",
            f"--channels={self.cfg.channels}",
            "--format=float32le",
            f"--latency-msec={self.cfg.output_latency_ms}",
            "--client-name=vcclient-cachy",
            "--stream-name=engine-out",
            "--raw",
        ]
        return subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _run_loop(self) -> None:
        import sounddevice as sd

        chunk_mic = int(self.cfg.mic_rate * self.cfg.chunk_seconds)
        # Reset SOLA buffers so a stop/start cycle doesn't leak stale audio.
        self.reset_streaming_state()

        pacat_proc: subprocess.Popen[bytes] | None = None
        monitor_stream = None
        try:
            pacat_proc = self._open_pacat()
            in_stream = sd.InputStream(
                samplerate=self.cfg.mic_rate,
                channels=self.cfg.channels,
                blocksize=chunk_mic,
                dtype="float32",
                device=self.cfg.input_device,
            )
            if self.cfg.monitor:
                # Best-effort self-monitor stream; failures here don't stop the engine.
                try:
                    monitor_stream = sd.OutputStream(
                        samplerate=self.cfg.sink_rate,
                        channels=self.cfg.channels,
                        dtype="float32",
                    )
                    monitor_stream.start()
                except Exception as e:
                    self.stats.last_error = f"monitor: {type(e).__name__}: {e}"
                    monitor_stream = None

            with in_stream:
                while not self._stop_event.is_set():
                    data, _ = in_stream.read(chunk_mic)
                    audio = data.reshape(-1).astype(np.float32, copy=False)
                    rms = float(np.sqrt(np.mean(audio**2)))
                    self.stats.last_input_rms = rms

                    t_total = time.perf_counter()
                    audio16 = _resample_linear(audio, self.cfg.mic_rate, 16_000)

                    t_inf = time.perf_counter()
                    # Streaming path uses SOLA + input history (Phase B). When
                    # `sola_enabled=False`, _process_streaming_16k still routes
                    # the model call through the history buffer but skips the
                    # crossfade — useful for A/B perf comparisons.
                    out16 = self._process_streaming_16k(audio16)
                    inf_ms = (time.perf_counter() - t_inf) * 1000

                    if out16.shape[0] == 0:
                        # First-chunk warmup may emit nothing; skip the write.
                        continue

                    out48 = _resample_linear(out16, 16_000, self.cfg.sink_rate)

                    # Primary output → VCClientCachySink via pacat.
                    if pacat_proc.poll() is not None:
                        raise RuntimeError(f"pacat subprocess died (exit {pacat_proc.returncode})")
                    if pacat_proc.stdin is None:
                        raise RuntimeError("pacat stdin is unavailable")
                    pacat_proc.stdin.write(out48.tobytes())
                    pacat_proc.stdin.flush()

                    # Optional self-monitor → host default output.
                    if monitor_stream is not None:
                        with contextlib.suppress(Exception):
                            monitor_stream.write(out48.reshape(-1, 1))

                    total_ms = (time.perf_counter() - t_total) * 1000
                    self.stats.chunks_processed += 1
                    self.stats.last_inference_ms = inf_ms
                    self.stats.last_total_ms = total_ms
                    self.stats._recent_inference.append(inf_ms)
                    self.stats._recent_total.append(total_ms)
                    if self.stats._recent_inference:
                        self.stats.avg_inference_ms = sum(self.stats._recent_inference) / len(
                            self.stats._recent_inference
                        )
                        self.stats.avg_total_ms = sum(self.stats._recent_total) / len(
                            self.stats._recent_total
                        )
        except Exception as e:
            self.stats.last_error = f"{type(e).__name__}: {e}"
            self.stats.running = False
        finally:
            # Flush SOLA's held-back tail through to the sink before tearing
            # down the subprocess — otherwise the last ~50 ms of audio is lost.
            if self._sola is not None and pacat_proc is not None:
                with contextlib.suppress(Exception):
                    tail16 = self._sola.flush()
                    if tail16.size > 0 and pacat_proc.stdin is not None:
                        tail48 = _resample_linear(tail16, 16_000, self.cfg.sink_rate)
                        pacat_proc.stdin.write(tail48.tobytes())
                        pacat_proc.stdin.flush()
            if monitor_stream is not None:
                try:
                    monitor_stream.stop()
                    monitor_stream.close()
                except Exception:
                    pass
            if pacat_proc is not None:
                try:
                    if pacat_proc.stdin is not None:
                        pacat_proc.stdin.close()
                except Exception:
                    pass
                try:
                    pacat_proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    pacat_proc.terminate()
                    try:
                        pacat_proc.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        pacat_proc.kill()
