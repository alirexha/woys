"""Loading the three model sessions and hot-swapping the voice model.

`_ensure_sessions` builds contentvec, rmvpe and the RVC voice; the swap
path queues a `_SwapRequest`, preloads the new voice off the audio thread,
and applies it at a chunk boundary. `_ModelSwapMixin` holds these
`RealtimeEngine` methods; the engine's `__init__` creates the state they
share (declared below for type checking). Split out of `audio.engine`,
which re-exports `_SwapRequest`.
"""

from __future__ import annotations

import contextlib
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import onnxruntime as ort

from audio.engine_config import MODELS_DIR, EngineConfig
from audio.engine_stats import EngineStats
from audio.resample import _StreamResampler
from audio.sessions import (
    _TRT_ACTIVE_PER_SESSION,
    _TRT_INIT_ERRORS,
    RvcSessionPool,
    _make_session,
    _session_is_cpu_only,
)
from audio.sola import SOLAConfig, SOLAStream

NDArrayF32 = npt.NDArray[np.float32]


@dataclass
class _SwapRequest:
    """a single queued model-swap with a
    per-call completion event. The TUI / socket caller holds the
    `completion` event after `request_model_swap()` returns and waits
    on IT specifically (not a shared broadcast Event) so two rapid
    swaps cannot collapse into one false-done.

    `error` is set by `_maybe_swap_model` when the swap fails (e.g.,
    subprocess InferenceError) so the caller can distinguish "done"
    from "failed". Pre-fix the single broadcast Event had no way to
    convey failure.
    """

    target: Path
    completion: threading.Event = field(default_factory=threading.Event)
    error: BaseException | None = None


class _ModelSwapMixin:
    """Model-loading and hot-swap methods of `RealtimeEngine`."""

    cfg: EngineConfig
    stats: EngineStats
    _stop_event: threading.Event
    _stopped: bool
    _inf_client: Any
    _cv: ort.InferenceSession | None
    _rmvpe: ort.InferenceSession | None
    _rvc: ort.InferenceSession | None
    _cv_input_dtype: str
    _rmvpe_input_dtype: str
    _is_half: bool
    _rvc_output_sr: int
    _rvc_pool: RvcSessionPool
    _rvc_sr_cache: dict[Path, int]
    _sola: SOLAStream | None
    _resampler_out: _StreamResampler | None
    _writer_queue: queue.Queue[bytes] | None
    _swap_queue: queue.Queue[_SwapRequest]
    _swap_preload_queue: queue.Queue[Path]
    _swap_lock: threading.Lock
    _outstanding_swaps: list[_SwapRequest]
    active_embedder: str

    if TYPE_CHECKING:

        def record_error(self, msg: str) -> None: ...

        def _self_abort(self, msg: str) -> None: ...

        def reset_streaming_state(self) -> None: ...

        def _to_sink_bytes(self, mono: NDArrayF32) -> bytes: ...

        def _enqueue_chunk(self, payload: bytes) -> None: ...

    def _ensure_sessions(self) -> None:
        # v0.3.0: prefer fp16 variants if present next to the fp32 file. fp16
        # rmvpe halves its VRAM footprint with no measurable pitch-detection
        # quality loss (validated v0.2.0). fp16 contentvec, by contrast, has
        # cosine sim 0.75 vs fp32 - only auto-promoted if explicitly requested.
        cv_path = self._auto_pick_fp16(self.cfg.contentvec_model, allow=False)
        rmvpe_path = self._auto_pick_fp16(self.cfg.rmvpe_model, allow=True)
        rvc_path = Path(self.cfg.rvc_model)

        # fail with a
        # clear, typed error naming the missing file AND a remediation
        # command, *before* handing the path to ONNX Runtime (which
        # otherwise raises an opaque ORT-internal exception far from
        # the cause). This also makes the top-level traceback guard
        # (F-merged-022) a clean typed catch rather than an error-
        # string heuristic. Only paths whose session still needs
        # loading are checked, so an already-loaded session is
        # unaffected.
        #
        # The remediation tag distinguishes foundation models (fetched
        # by `scripts/download_weights.py` -- usually from a
        # `./install.sh --skip-models` follow-up) from the user's RVC
        # voice model (fetched by `woys models download <repo>` or
        # converted from a .pth via `woys convert`).
        for label, path, needed, remediation in (
            (
                "contentvec model",
                cv_path,
                self._cv is None,
                "scripts/download_weights.py",
            ),
            (
                "rmvpe model",
                rmvpe_path,
                self._rmvpe is None,
                "scripts/download_weights.py",
            ),
            (
                "rvc model",
                rvc_path,
                self._rvc is None,
                "woys models download <hf-repo> -- or -- woys convert <pth>",
            ),
        ):
            if needed and not path.exists():
                raise FileNotFoundError(
                    f"{label} not found at {path}.\n  Fix: run `{remediation}` and try again."
                )

        if self._cv is None:
            self._cv = _make_session(
                cv_path,
                use_tensorrt=self.cfg.use_tensorrt,
                allow_cpu_fallback=self.cfg.allow_cpu_fallback,
            )
            self._cv_input_dtype = self._cv.get_inputs()[0].type
        if self._rmvpe is None:
            self._rmvpe = _make_session(
                rmvpe_path,
                use_tensorrt=self.cfg.use_tensorrt,
                allow_cpu_fallback=self.cfg.allow_cpu_fallback,
            )
            self._rmvpe_input_dtype = self._rmvpe.get_inputs()[0].type
        if self._rvc is None:
            self._rvc = self._rvc_pool.get_or_create(self.cfg.rvc_model)
            self._is_half = self._rvc.get_inputs()[0].type != "tensor(float)"
            self._rvc_output_sr = self._cached_rvc_sr(Path(self.cfg.rvc_model))

        # v0.8.1: snapshot TRT init status from the module-level
        # tracker into stats, so woys diag can show which sessions
        # actually got TRT EP and which fell back to CUDA.
        self.stats.trt_active_for = dict(_TRT_ACTIVE_PER_SESSION)
        self.stats.trt_init_errors = dict(_TRT_INIT_ERRORS)
        # record whether any model session
        # bound CPU-only. Only reachable through the test-only
        # cfg.allow_cpu_fallback -- otherwise `_make_session` raises
        # CpuFallbackError above. Printed by `woys diag`.
        self.stats.cpu_fallback_active = any(
            _session_is_cpu_only(s) for s in (self._cv, self._rmvpe, self._rvc) if s is not None
        )

        # Resolve embedder mode. v0.8.0 removed the fairseq path - only "onnx"
        # is supported. Any non-"onnx" value in config is reported and the
        # engine falls back to onnx (so old config.toml files don't crash).
        if self.cfg.embedder != "onnx":
            msg = (
                f"unknown embedder {self.cfg.embedder!r}; v0.8.0 only supports "
                f'"onnx". Falling back.'
            )
            print(f"[engine] {msg}")
            self.record_error(msg)
        self.active_embedder = "onnx"

    @staticmethod
    def _auto_pick_fp16(fp32_path: Path, *, allow: bool) -> Path:
        """If a `<name>-fp16.onnx` sibling exists and `allow=True`, use it."""
        if not allow:
            return fp32_path
        cand = fp32_path.with_name(fp32_path.stem + "-fp16" + fp32_path.suffix)
        return cand if cand.exists() else fp32_path

    def _probe_rvc_output_sr(self) -> int:
        """Run one forward pass through the loaded RVC session to measure its
        native output sample rate.

        The output of an RVC v2 ONNX is `(N_out,)` audio at the model's
        training rate (16 kHz for amitaro v2_16k, 40 kHz for most v2 voices,
        32 kHz / 48 kHz for some). The convert.py exporter stamps the rate
        into ONNX `custom_metadata_map["metadata"]` as JSON, but reading
        that is brittle - different exporters use different keys. Probing
        is bulletproof: feed a known-length input, count output samples.

        Costs ~20 ms once at session load. Worth it.

        raises `RuntimeError` if the probe
        fails or yields an unrecognised rate -- it never silently guesses,
        because a wrong output rate pitch-shifts the entire session and
        poisons `_rvc_sr_cache`.
        """
        assert self._rvc is not None
        # Feed a 1 s feats window (50 frames after 2x upsample = 100 frames),
        # measure output. Feats dim from the RVC input shape.
        feats_dim = 768
        try:
            shape = self._rvc.get_inputs()[0].shape
            if len(shape) >= 3 and isinstance(shape[2], int):
                feats_dim = shape[2]
        except (IndexError, ValueError):
            pass
        # 1 s of audio at 16 kHz contentvec = 50 frames. Upsample 2x = 100.
        n_frames = 100
        feats_dummy = np.zeros((1, n_frames, feats_dim), dtype=np.float32)
        feats_dtype = np.float16 if self._is_half else np.float32
        feed: dict[str, np.ndarray] = {  # type: ignore[type-arg]
            "feats": feats_dummy.astype(feats_dtype),
            "p_len": np.array([n_frames], dtype=np.int64),
            "pitch": np.zeros((1, n_frames), dtype=np.int64),
            "pitchf": np.zeros((1, n_frames), dtype=np.float32),
            "sid": np.array([0], dtype=np.int64),
        }
        try:
            out = self._rvc.run(["audio"], feed)[0]
        except Exception as e:
            # re-raise -- do NOT silently
            # return 16 kHz. CX2 corrected the original "nono variant"
            # framing: `_infer` builds an *identical* pitch-bearing feed
            # dict, so a genuinely pitchless model would crash there on
            # every chunk -- it cannot produce the silent-chipmunk symptom.
            # The realistic trigger is a transient GPU/cuDNN/shape error on
            # the cold first pass, where swallowing and assuming 16 kHz
            # poisons `_rvc_sr_cache` forever and plays the whole session
            # ~16 semitones off with no `last_error` and no counter.
            raise RuntimeError(
                f"failed to probe the RVC model's output sample rate: "
                f"{type(e).__name__}: {e}. The engine cannot run without a "
                f"known output rate (a wrong guess pitch-shifts the whole "
                f"session), so this aborts start visibly. Retry, or check "
                f"`woys info` / the GPU state."
            ) from e
        n_out = int(np.asarray(out).size)
        # Output for 1 s of feats input ≈ 1 s of audio at the model rate.
        # Round to the nearest known RVC training rate.
        for sr in (16_000, 22_050, 24_000, 32_000, 40_000, 44_100, 48_000):
            if abs(n_out - sr) < sr * 0.05:
                return sr
        # an unrecognised rate is a second silent
        # guess -- raise instead of treating the raw sample count as Hz.
        raise RuntimeError(
            f"RVC model probe produced {n_out} samples for a 1 s input, which "
            f"matches no known RVC training rate (16/22.05/24/32/40/44.1/48 "
            f"kHz). Refusing to guess the output rate."
        )

    def _cached_rvc_sr(self, path: Path) -> int:
        """Probe and remember the model's output sample rate.

        Side-effect: recreates `self._sola` at the new rate so the
        crossfade-window math matches the actual output samples.
        """
        key = Path(path).resolve()
        if key in self._rvc_sr_cache:
            sr = self._rvc_sr_cache[key]
        else:
            sr = self._probe_rvc_output_sr()
            self._rvc_sr_cache[key] = sr
        self._rebuild_sola_for_rate(sr)
        return sr

    def _rebuild_sola_for_rate(self, model_sr: int) -> None:
        """Recreate the output-side SOLAStream for the given rate. Idempotent -
        no-op when the rate is unchanged."""
        from audio.sola import SOLAStream

        if not self.cfg.sola_enabled:
            self._sola = None
            return
        if self._sola is not None and self._sola.cfg.rate == model_sr:
            return
        out_cfg = SOLAConfig(
            rate=model_sr,
            crossfade_ms=self.cfg.sola_crossfade_ms,
            search_ms=self.cfg.sola_search_ms,
            context_ms=self.cfg.sola_context_ms,
            corr_threshold=self.cfg.sola_corr_threshold,
        )
        self._sola = SOLAStream(out_cfg)

    def reload_rvc(self, path: Path) -> None:
        """Hot-swap the RVC voice model - synchronous, thread-unsafe.

        Use `request_model_swap()` from any thread other than the engine
        worker; this function is kept for tests + offline use only.
        """
        self.cfg.rvc_model = path
        self._rvc = self._rvc_pool.get_or_create(path)
        self._is_half = self._rvc.get_inputs()[0].type != "tensor(float)"
        self._rvc_output_sr = self._cached_rvc_sr(Path(path))
        self.reset_streaming_state()

    def warmup_voice_library(self, voice_paths: list[Path] | None = None) -> int:
        """Eagerly load + cudnn-warm every cached voice. Returns count warmed.

        If `voice_paths` is None, walks the user's models dir and warms
        every `.onnx` that doesn't look like a foundation file. Costs
        ~600 ms per voice on RTX 2070; subsequent swaps to any of those
        voices are pointer swaps (~10 ms total).

        B61 / perf-007: caps voice_paths at `session_pool_size`. Pre-v0.8.0
        we walked every voice in the dir even though only the last
        `pool_size` survive eviction - so voices 1..N-pool_size were
        warmed and immediately discarded. Wasted startup time.
        """
        if voice_paths is None:
            foundations = {
                "rmvpe.onnx", "rmvpe-fp16.onnx",
                "rmvpe_wrapped.onnx", "rmvpe_wrapped-fp16.onnx",
                "contentvec-f.onnx", "contentvec-f-fp16.onnx",
                "hubert_base.onnx",
            }  # fmt: skip
            voice_paths = sorted(p for p in MODELS_DIR.glob("*.onnx") if p.name not in foundations)
        # B61: only warm what fits in the pool. The first N entries (alphabetical)
        # are the ones the user is most likely to land on - bias toward retaining
        # the deterministic prefix.
        cap = self.cfg.session_pool_size
        if len(voice_paths) > cap:
            print(
                f"[engine] eager-warmup capped at session_pool_size={cap} "
                f"(have {len(voice_paths)} voices; skipping the LRU-evicted tail)"
            )
            voice_paths = voice_paths[:cap]
        for p in voice_paths:
            self._rvc_pool.warmup(p)
            # Also probe + cache the SR for each.
            with contextlib.suppress(Exception):
                self._rvc_sr_cache[p.resolve()] = self._probe_sr_for(p)
        return len(voice_paths)

    def _probe_sr_for(self, path: Path) -> int:
        """Probe the model's output SR via the pool (so the session is cached)."""
        sess = self._rvc_pool.get_or_create(path)
        prev_rvc = self._rvc
        prev_is_half = self._is_half
        self._rvc = sess
        self._is_half = sess.get_inputs()[0].type != "tensor(float)"
        try:
            return self._probe_rvc_output_sr()
        finally:
            self._rvc = prev_rvc
            self._is_half = prev_is_half

    def request_model_swap(self, path: Path) -> _SwapRequest:
        """Thread-safe: queue a model swap and return the per-call
        `_SwapRequest`. The caller waits on `req.completion` and reads
        `req.error` to distinguish "done" from "failed".

        pre-fix this
        method overwrote a single `_pending_model_swap` slot and
        cleared a SHARED `_swap_done` Event -- two rapid swaps
        collapsed (the second overwrote the first; voice A was
        silently dropped) and the broadcast `_swap_done.set()`
        released all waiters when only one swap had applied. Defeated
        B5/corr-003.

        Post-fix every request lands as its own `_SwapRequest` in
        `self._swap_queue`. F-13-12: if the engine is stopped (or
        stopping) when this is called, the event is resolved
        immediately so the caller never parks.

        the return type widened
        from `threading.Event` to `_SwapRequest` so swap *failures*
        also have a single read site. `_maybe_swap_model` sets
        `req.error` before resolving the event; callers (the TUI's
        MODEL / PROFILE handlers, `_apply_profile_named`,
        `action_cycle_profile`) check `req.error` after `req.
        completion.wait(...)` and route a failure to
        `engine.record_error()`. Pre-fix the failure landed only in
        the JobRegistry status_line via the worker's bare-exception
        path, and the TUI never polled its own jobs -- so a swap to
        a corrupted ONNX or a `subprocess swap failed` cascade
        recorded "OK job=... model=...", parked the UI on "loading
        X..." for 10 s, then resumed with the old voice still in
        place. No banner, no toast.
        """
        req = _SwapRequest(target=Path(path))
        with self._swap_lock:
            # `_stop_event` covers an engine that stopped itself (or is
            # mid-stop): its loop is gone and nothing drains the queue.
            if self._stopped or self._stop_event.is_set():
                # F-13-12: a STOPPED engine cannot drain the queue --
                # resolve the event immediately so callers don't park
                # for the full 10 s JobRegistry timeout. A never-
                # started engine still queues (used by tests +
                # script-only flows where `_maybe_swap_model` is
                # called manually).
                req.error = RuntimeError("engine stopped; swap not applied")
                req.completion.set()
                return req
            self._swap_queue.put(req)
            self._outstanding_swaps.append(req)
        # also signal the preloader
        # to prime the pool cache so the worker hits a cache-hit on
        # the chunk-boundary swap. Best-effort -- queue.put never
        # blocks because the queue is unbounded; if the preloader
        # thread isn't running (engine not yet started or already
        # stopped), the worker just pays the cache-miss cost itself.
        with contextlib.suppress(Exception):
            self._swap_preload_queue.put_nowait(Path(path))
        return req

    def _maybe_swap_model(self) -> None:
        """Worker-side hook: drain any queued swap requests, applying
        each in order. Called from `_run_loop` at the top of each
        chunk.

        pre-fix the single-slot pattern collapsed
        rapid swaps. Post-fix every queued `_SwapRequest` is applied
        in order; each completion event fires when ITS swap finishes
        (not the broadcast Event of the pre-fix code)."""
        requests: list[_SwapRequest] = []
        with self._swap_lock:
            while True:
                try:
                    requests.append(self._swap_queue.get_nowait())
                except queue.Empty:
                    break
        if not requests:
            return
        for i, req in enumerate(requests):
            try:
                self._apply_one_swap(req)
            except Exception as e:
                # `_apply_one_swap` handles the expected failures itself;
                # anything else still crashes the loop, but no caller may
                # be left parked on a request this batch already drained.
                req.error = e
                self._resolve_swap(req)
                for rest in requests[i + 1 :]:
                    rest.error = RuntimeError("swap not applied: an earlier swap failed")
                    self._resolve_swap(rest)
                raise

    def _apply_one_swap(self, req: _SwapRequest) -> None:
        """Apply a single `_SwapRequest`: flush SOLA tail, drain writer,
        swap the session, reset streaming state, set the per-call
        completion event. Failures are recorded on `req.error`; the
        event is set regardless so the caller never parks.

        Extracted from the legacy `_maybe_swap_model` body so each
        queued swap gets the same treatment (SOLA flush + writer
        drain + session reset). For multi-swap drains the SOLA flush
        happens per-swap -- the user requested each one and each
        deserves a clean transition."""
        target = req.target
        resampler_out_was_flushed = False
        if self._sola is not None:
            with contextlib.suppress(Exception):
                tail = self._sola.flush()
                if tail.size > 0 and self._resampler_out is not None:
                    tail48 = self._resampler_out.process(tail)
                    flush48 = self._resampler_out.flush()
                    resampler_out_was_flushed = True
                    full = np.concatenate([tail48, flush48]) if flush48.size else tail48
                    if full.size > 0:
                        self._enqueue_chunk(self._to_sink_bytes(full))

        # B10 / corr-005: wait for the writer queue to drain before swapping
        # the model. Without this barrier, OLD-rate audio sitting in the
        # queue (up to ~2 s worth at queue_size=8 + chunk_seconds=0.25)
        # plays AFTER the swap completes - user hears the old voice for
        # seconds after triggering a swap. 300 ms timeout caps the wait
        # so a stuck pacat/pw-cat doesn't deadlock the swap.
        if self._writer_queue is not None:
            drain_deadline = time.perf_counter() + 0.3
            while time.perf_counter() < drain_deadline and not self._writer_queue.empty():
                time.sleep(0.005)

        # v0.8.0 - subprocess swap path. Child owns the session pool;
        # tell it to swap, wait for the new RVC sample-rate response,
        # then rebuild the output resampler in the parent if the rate
        # changed. On InferenceError (child died, swap timed out) we
        # used to silently `return` and let the engine continue running
        # in a state where every chunk drops (B6 / corr-004). Now: stop
        # the engine cleanly and surface the error to the TUI.
        if self._inf_client is not None:
            from audio.inference_client import InferenceError

            try:
                new_sr, new_is_half = self._inf_client.swap_model(target)
                self.cfg.rvc_model = target
                self._is_half = new_is_half
                # v0.14.0 (C002): rebuild the resampler if rate changed OR
                # if the SOLA flush above finalized the existing soxr stream.
                # Same-rate swap with a non-identity stream pre-v0.14.0
                # crashed the engine on the next chunk.
                # ~5 ms cold-fade-in
                # masks the soxr filter-warmup transient when the rebuild
                # lands inside live audio (model swap). 5 ms at sink_rate
                # = sink_rate // 200; below the ~10 ms perception window
                # for amplitude steps.
                if new_sr != self._rvc_output_sr or resampler_out_was_flushed:
                    self._resampler_out = _StreamResampler(
                        new_sr,
                        self.cfg.sink_rate,
                        cold_fade_in_samples=self.cfg.sink_rate // 200,
                    )
                if new_sr != self._rvc_output_sr:
                    self._rebuild_sola_for_rate(new_sr)
                self._rvc_output_sr = new_sr
                self.reset_streaming_state()
                self._resolve_swap(req)
                return
            except InferenceError as e:
                # B6 / corr-004: do NOT silently fall through. The in-process
                # _rvc isn't loaded in subprocess mode, so the legacy path
                # would fail every chunk. Better to stop and surface the
                # error than serve silence indefinitely.
                req.error = e
                self._resolve_swap(req)
                self._self_abort(
                    f"subprocess swap failed: {e}. Stopping engine - "
                    f"flip `inference_subprocess=false` to fall back to "
                    f"in-process inference."
                )
                return

        # Legacy in-process path. Existing _cv (contentvec) and _rmvpe
        # stay - they're foundation models, not voice-specific.
        # v0.5.0: pool-cached. Cache hit ≈ 10 ms; cache miss ≈ 600 ms.
        # A model that fails to load or probe (corrupt / incompatible
        # export, file gone) must not crash the engine: keep the old
        # voice, report the failure on the request, and only point
        # cfg.rvc_model at the target once it is really running.
        prev_rvc, prev_is_half = self._rvc, self._is_half
        try:
            self._rvc = self._rvc_pool.get_or_create(target)
            self._is_half = self._rvc.get_inputs()[0].type != "tensor(float)"
            # Probes through self._rvc / self._is_half, hence set above.
            new_sr = self._cached_rvc_sr(target)
        except Exception as e:
            self._rvc, self._is_half = prev_rvc, prev_is_half
            self._rebuild_sola_for_rate(self._rvc_output_sr)
            # The SOLA flush above finalized the output stream; the old
            # voice needs a fresh one, exactly as a successful swap would.
            if resampler_out_was_flushed:
                self._resampler_out = _StreamResampler(
                    self._rvc_output_sr,
                    self.cfg.sink_rate,
                    cold_fade_in_samples=self.cfg.sink_rate // 200,
                )
            self.reset_streaming_state()
            req.error = e
            self.record_error(
                f"model swap to {Path(target).name} failed: {type(e).__name__}: {e}. "
                f"Keeping the current voice."
            )
            self._resolve_swap(req)
            return
        self.cfg.rvc_model = target
        # v0.6.7 - rebuild the output resampler if the new model has a
        # different native rate. Identity ratios (e.g. 16k -> 16k -> 48k
        # stays the same) won't reset state, so swaps between same-rate
        # voices used to skip the rebuild.
        # v0.14.0 (C002): also rebuild when the SOLA flush above finalized
        # the existing soxr stream (`last=True`). Without this, a same-rate
        # swap left a finalized stream in place and the next chunk raised
        # `RuntimeError: Input after last input` from soxr, killing engine.
        # cold-fade-in (see subprocess
        # branch above for rationale). 5 ms at sink_rate.
        if new_sr != self._rvc_output_sr or resampler_out_was_flushed:
            self._resampler_out = _StreamResampler(
                new_sr,
                self.cfg.sink_rate,
                cold_fade_in_samples=self.cfg.sink_rate // 200,
            )
        self._rvc_output_sr = new_sr
        self.reset_streaming_state()
        # B5: signal "swap complete" AFTER all the work. TUI poll sites
        # waiting on the per-call event now correctly observe done-state.
        self._resolve_swap(req)

    def _resolve_swap(self, req: _SwapRequest) -> None:
        """Set the per-call completion event and drop the request
        from `_outstanding_swaps`. Safe to call from the engine
        thread (via `_apply_one_swap`) or from `stop()` teardown
        (where `req.error` is set first)."""
        req.completion.set()
        with self._swap_lock, contextlib.suppress(ValueError):
            # already removed (double-resolve) -> ValueError is fine.
            self._outstanding_swaps.remove(req)

    def _fail_pending_swaps(self, msg: str) -> None:
        """Resolve every outstanding or still-queued swap with an error
        so no caller parks on a request that will never be applied."""
        with self._swap_lock:
            pending = list(self._outstanding_swaps)
            self._outstanding_swaps.clear()
            while True:
                try:
                    pending.append(self._swap_queue.get_nowait())
                except queue.Empty:
                    break
        for req in pending:
            if not req.completion.is_set():
                req.error = RuntimeError(msg)
                req.completion.set()

    def _swap_preloader_loop(self) -> None:
        """daemon thread that
        primes the `_rvc_pool` cache for every queued swap target.

        Pre-fix the engine worker called
        `self._rvc_pool.get_or_create(target)` on the hot path. On a
        cache miss that costs ~600 ms (model load + cuDNN tune) --
        the engine isn't reading the mic, the writer queue drains,
        the user hears a glitch.

        Post-fix this thread drains `_swap_preload_queue` and calls
        `get_or_create` itself. `RvcSessionPool.get_or_create` builds
        OUTSIDE its internal lock, so this thread's call doesn't
        block a concurrent worker call. By the time the worker
        reaches `_apply_one_swap` at the chunk boundary, the cache
        is warm and its `get_or_create` is a ~10 ms cache-hit.

        Failures (FileNotFound, ORT errors) are swallowed here -- the
        engine worker will hit the same failure at the chunk boundary
        and surface it through the normal swap-completion error path
        (`_SwapRequest.error`).
        """
        while not self._stop_event.is_set():
            try:
                target = self._swap_preload_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            with contextlib.suppress(Exception):
                self._rvc_pool.get_or_create(target)
