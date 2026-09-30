"""The engine's inference core: contentvec -> rmvpe -> rvc on one block
(`_infer`), the streaming wrapper that keeps the input history and runs
SOLA (`_process_streaming_16k`), its failure guard, and the warmup that
pre-runs every input shape the realtime path will see.

`_InferenceMixin` holds these `RealtimeEngine` methods; the engine's
`__init__` creates the state they share (declared below for type checking).
Split out of `audio.engine`.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import onnxruntime as ort

from audio.engine_config import EngineConfig
from audio.engine_stats import EngineStats
from audio.pitch import interpolate_voiced_gaps_np, to_pitch_coarse
from audio.resample import _StreamResampler
from audio.sola import SOLAConfig, SOLAStream

NDArrayF32 = npt.NDArray[np.float32]


class _InferenceMixin:
    """Inference and streaming methods of `RealtimeEngine`."""

    cfg: EngineConfig
    stats: EngineStats
    _stats_lock: Any
    _stop_event: threading.Event
    _inf_client: Any
    _cv: ort.InferenceSession | None
    _rmvpe: ort.InferenceSession | None
    _rvc: ort.InferenceSession | None
    _cv_input_dtype: str
    _rmvpe_input_dtype: str
    _is_half: bool
    _sola: SOLAStream | None
    _sola_input_cfg: SOLAConfig
    _input_history: NDArrayF32
    _pitch_carry_f0: float
    _pitch_carry_age_frames: int
    _consecutive_drops: int

    if TYPE_CHECKING:

        def record_error(self, msg: str) -> None: ...

        def _self_abort(self, msg: str) -> None: ...

    def _extract_feats(self, audio16k: NDArrayF32) -> NDArrayF32:
        """Embedder dispatch (always ONNX contentvec since v0.8.0)."""
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

        Standalone, offline-only helper: the realtime loop never calls it
        (with or without SOLA it goes through `_process_streaming_16k`), and
        its only caller is the fallback in scripts/benchmark_probe.py.
        Doesn't touch streaming state.
        """
        return self._infer(audio16k)

    def _infer(self, audio16k: NDArrayF32, *, update_pitch_carry: bool = False) -> NDArrayF32:
        """Raw model invocation; no streaming bookkeeping.

        v0.8.0 - when `cfg.inference_subprocess=True` AND the
        `_inf_client` was successfully started, this delegates to the
        child process via `InferenceClient.infer()`. The child runs
        the full cv → rmvpe → rvc pipeline in its own CUDA context;
        per-stage timings come back via the response and populate
        `EngineStats` so woys diag shows the same breakdown.

        On `InferenceError` (child died, pipe broke), we attempt one
        restart-and-retry. If the retry fails too, the exception
        propagates up to `_safe_process_streaming_16k` which catches
        it and bumps `dropped_chunks` like any other inference
        failure.

        when ``update_pitch_carry=True``
        AND we're on the legacy in-process path, read
        `self._pitch_carry_*` to provide a leading-edge anchor to
        `interpolate_voiced_gaps_np`, and update the carry from the
        trailing voiced frame of this chunk's pitchf. The subprocess
        path does not carry pitch state across the IPC boundary -- the
        leading-edge dropout case (a brief unvoiced run starting a
        chunk where the prior chunk ended voiced) is preserved there.
        Documented limitation; the IPC protocol does not currently
        ship the prior/posterior pitch-carry tuple.
        """
        if self._inf_client is not None:
            from audio.inference_client import InferenceError

            try:
                ipc_result, timings = self._inf_client.infer(
                    audio16k,
                    f0_up_key=self.cfg.f0_up_key,
                    sid=self.cfg.sid,
                    threshold=self.cfg.threshold,
                )
            except InferenceError as e:
                # Try one restart. If THAT fails, propagate.
                self.record_error(f"inference child died ({e}); attempting restart")
                try:
                    self._inf_client.restart()
                    self.stats.child_restarts = self._inf_client.restart_count
                    self.stats.child_pid = (
                        self._inf_client._handles.proc.pid if self._inf_client._handles else None
                    )
                    ipc_result, timings = self._inf_client.infer(
                        audio16k,
                        f0_up_key=self.cfg.f0_up_key,
                        sid=self.cfg.sid,
                        threshold=self.cfg.threshold,
                    )
                except InferenceError:
                    raise

            # Populate stats from child's per-stage timings + the
            # cumulative NaN-replace count. Use the child's running
            # total so cumulative numbers survive a child restart.
            self.stats.last_cv_ms = timings.cv_ms
            self.stats.last_rmvpe_ms = timings.rmvpe_ms
            self.stats.last_rvc_ms = timings.rvc_ms
            self.stats.last_ipc_roundtrip_ms = timings.roundtrip_ms
            # cluster the
            # rolling-window appends under `_stats_lock` so a TUI /
            # diag reader cannot raise on `deque mutated during
            # iteration`.
            with self._stats_lock:
                self.stats._recent_ipc_roundtrip_ms.append(timings.roundtrip_ms)
                self.stats._recent_cv_ms.append(timings.cv_ms)
                self.stats._recent_rmvpe_ms.append(timings.rmvpe_ms)
                self.stats._recent_rvc_ms.append(timings.rvc_ms)
                self.stats.unique_audio16_lens.add(int(audio16k.shape[-1]))
            self.stats.nan_chunks = timings.nan_chunks_total
            ipc_typed: NDArrayF32 = ipc_result
            return ipc_typed

        # Legacy in-process path.
        assert self._cv is not None and self._rmvpe is not None and self._rvc is not None

        # -- documented omission of upstream's
        # `silence_front` lead-in trim. Upstream RVC's `Pipeline.py:254/272/306`
        # tracks `silence_front` (how many leading samples are known-silent)
        # and trims that region from the contentvec features + RMVPE pitchf
        # before inference and index search:
        #     npyOffset = math.floor(silence_front * 16000) // 360
        #     feats = feats[:, npyOffset * 2 :, :]
        # woys deliberately omits this for two reasons:
        #   (1) the input gate (`engine.py` `_safe_process_streaming_16k`) zeros
        #       sub-hysteresis chunks entirely, so silence-only chunks never hit
        #       this path -- the trim is a no-op gain for the most common case;
        #   (2) faiss index retrieval (F-31-01) is currently not implemented, so
        #       the "silent frames pollute the nearest-neighbour search" risk
        #       upstream's trim was protecting against is currently null.
        # The cost we pay: when speech leads with a partially-silent history
        # window, contentvec + RMVPE still run on the silent prefix (small
        # latency tax, no quality impact). If F-31-01 ever lands, this comment
        # is the marker to revisit -- index search WILL be polluted by silent
        # frames at that point. this is a documented design
        # choice, not an unaudited omission.
        t_cv0 = time.perf_counter()
        feats = self._extract_feats(audio16k)
        # v0.6.9: silently zero NaN bursts in feats before they propagate
        # through the inferencer and become NaN samples in the output.
        if np.isnan(feats).any():
            feats = np.nan_to_num(feats, nan=0.0)
        t_cv1 = time.perf_counter()
        rm_dtype = np.float16 if "float16" in self._rmvpe_input_dtype else np.float32
        pitchf_raw = self._rmvpe.run(
            ["pitchf"],
            {
                "waveform": audio16k.reshape(1, -1).astype(rm_dtype),
                "threshold": np.array([self.cfg.threshold], dtype=rm_dtype),
            },
        )[0]
        pitchf = pitchf_raw.astype(np.float32).squeeze()
        # v0.6.9: sanitize + interpolate short voiced→voiced gaps so a transient
        # RMVPE failure mid-utterance doesn't zero the NSF harmonic source.
        # Live diagnostic on e_girl voice traced 8 of 14 dropouts to this path.
        # pass cross-chunk pitch carry so a
        # leading-edge unvoiced run can be bridged using the prior chunk's
        # trailing voiced anchor. Streaming wrapper sets `update_pitch_carry`.
        if update_pitch_carry:
            pitchf = interpolate_voiced_gaps_np(
                pitchf,
                prior_voiced_f0=self._pitch_carry_f0,
                prior_voiced_age_frames=self._pitch_carry_age_frames,
            )
            # Update carry from the trailing voiced frame of THIS pitchf.
            # `age_frames` measured at the END of this pitchf so the next
            # call's interpretation is conservative-recency (the slight
            # under-aging vs the overlap window is harmless because the
            # next call's in-window `last_valid` catches the same frame
            # if it falls in the history portion -- the carry only fires
            # when it's genuinely outside).
            voiced_mask = pitchf > 0.0
            if bool(voiced_mask.any()):
                last_voiced_idx = int(np.flatnonzero(voiced_mask)[-1])
                self._pitch_carry_f0 = float(pitchf[last_voiced_idx])
                self._pitch_carry_age_frames = (len(pitchf) - 1) - last_voiced_idx
            elif self._pitch_carry_age_frames >= 0:
                # No voiced this chunk; the carry ages by the full pitchf
                # length. Once age >= _VOICED_GAP_MAX_FRAMES the predicate
                # `have_prior` in interpolate_voiced_gaps_np rejects it.
                self._pitch_carry_age_frames += len(pitchf)
        else:
            pitchf = interpolate_voiced_gaps_np(pitchf)
        t_rmvpe1 = time.perf_counter()

        # v0.10.0-rc2 - split RVC stage into pre / run / post so the
        # tail-attribution data tells us GPU work vs Python overhead.
        feats_2x = np.repeat(feats, 2, axis=1)
        # v0.14.0 (area 4 / area 7 / C001): apply pitch shift in semitones
        # BEFORE deriving pitch_coarse. Upstream's RMVPEOnnxPitchExtractor
        # (src/server/voice_changer/RVC/embedder/RMVPEOnnxPitchExtractor.py)
        # shifts f0 first, then derives BOTH pitch_coarse (mel-bin index)
        # and pitchf (Hz vector) from the shifted result. The pre-v0.14.0
        # engine path multiplied pitchf_aligned AFTER coarse was derived,
        # so RVC saw mismatched harmonic-source vs pitch-class-embedding
        # pairs for any non-zero f0_up_key. Hard to detect aurally because
        # RVC blends the embedding into the residual; cleanest test is
        # A/B'ing pitch shifts against the upstream reference path.
        if self.cfg.f0_up_key != 0:
            pitchf = pitchf * (2.0 ** (self.cfg.f0_up_key / 12.0))
        pitch_coarse, pitchf_aligned = to_pitch_coarse(pitchf, target_len=feats_2x.shape[1])
        pitch_coarse = pitch_coarse[: feats_2x.shape[1]].reshape(1, -1)
        pitchf_aligned = pitchf_aligned[: feats_2x.shape[1]].reshape(1, -1).astype(np.float32)

        feats_dtype = np.float16 if self._is_half else np.float32
        # Build the input dict (final astype on feats happens here too).
        rvc_inputs = {
            "feats": feats_2x.astype(feats_dtype),
            "p_len": np.array([feats_2x.shape[1]], dtype=np.int64),
            "pitch": pitch_coarse,
            "pitchf": pitchf_aligned,
            "sid": np.array([self.cfg.sid], dtype=np.int64),
        }
        t_rvc_pre1 = time.perf_counter()
        out = self._rvc.run(["audio"], rvc_inputs)[0]
        t_rvc_run1 = time.perf_counter()
        result = np.array(out).astype(np.float32).squeeze()
        # v0.6.9: belt-and-braces NaN sanitize. pacat is fed float32le; NaN
        # would be undefined behavior in PipeWire's mixer chain and the
        # listener hears it as a click + brief gap.
        # B57 / audio-010: posinf=0.0 / neginf=0.0 (not ±1.0). nan_to_num is
        # element-wise - only the rare bad samples are zeroed, not the whole
        # chunk. The pre-v0.8.0 ±1.0 produced full-scale impulses (audible
        # click) on inf samples; zero is a single-sample dropout (~21 µs at
        # 48 kHz), audibly less harsh.
        if np.isnan(result).any() or np.isinf(result).any():
            result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
            # v0.7.0-rc4 - count NaN-sanitize hits so we can attribute
            # voice-correlated cuts to the vocoder rather than the gate
            # or SOLA. Pre-rc4 the path silently zeroed samples; area 06
            # of the audit flagged this as one of three NaN-zero paths
            # that incremented no counter.
            with self._stats_lock:
                self.stats.nan_chunks += 1
        t_rvc1 = time.perf_counter()
        # Per-stage timing surfaces in the slow_chunk_log when a chunk goes late.
        cv_ms = (t_cv1 - t_cv0) * 1000.0
        rmvpe_ms = (t_rmvpe1 - t_cv1) * 1000.0
        rvc_ms = (t_rvc1 - t_rmvpe1) * 1000.0
        # v0.10.0-rc2 - RVC sub-stages.
        rvc_pre_ms = (t_rvc_pre1 - t_rmvpe1) * 1000.0
        rvc_run_ms = (t_rvc_run1 - t_rvc_pre1) * 1000.0
        rvc_post_ms = (t_rvc1 - t_rvc_run1) * 1000.0
        self.stats.last_cv_ms = cv_ms
        self.stats.last_rmvpe_ms = rmvpe_ms
        self.stats.last_rvc_ms = rvc_ms
        # v0.10.0 - per-stage rolling windows for percentile attribution.
        # The pre-v0.10.0 path tracked only `_recent_inference` (sum); the
        # writer-jitter investigation needs to know which stage owns the
        # tail.
        # cluster lock.
        with self._stats_lock:
            self.stats._recent_cv_ms.append(cv_ms)
            self.stats._recent_rmvpe_ms.append(rmvpe_ms)
            self.stats._recent_rvc_ms.append(rvc_ms)
            self.stats._recent_rvc_pre_ms.append(rvc_pre_ms)
            self.stats._recent_rvc_run_ms.append(rvc_run_ms)
            self.stats._recent_rvc_post_ms.append(rvc_post_ms)
            self.stats.unique_audio16_lens.add(int(audio16k.shape[-1]))
        result_typed: NDArrayF32 = result
        return result_typed

    def _safe_process_streaming_16k(self, audio16: NDArrayF32) -> NDArrayF32 | None:
        """v0.6.8 - wrap `_process_streaming_16k` so a transient
        ORT / CUDA / numerical error drops the chunk instead of killing
        the engine.

        Returns the inferred chunk, or `None` if inference failed.
        Caller must check for `None` and skip the playback write - the
        engine's main loop does this with `continue`.

        First three failures log to `stats.last_error` with the
        exception type + message. After that the counter increments
        silently except for every 100th hit (to keep the diagnostic
        line refreshing without spamming).

        Pulled out of `_run_loop` so the failure path is unit-testable
        without spinning up the full audio thread.
        """
        try:
            result = self._process_streaming_16k(audio16)
            self._consecutive_drops = 0
            return result
        except Exception as e:
            with self._stats_lock:
                self.stats.dropped_chunks += 1
            self._consecutive_drops += 1
            n = self.stats.dropped_chunks
            if n <= 3:
                self.record_error(f"inference dropped chunk #{n}: {type(e).__name__}: {e}")
            elif n % 100 == 0:
                self.record_error(
                    f"inference still dropping chunks (total #{n}): {type(e).__name__}: {e}"
                )
            # B14 / corr-015: circuit breaker on sustained inference failure.
            # Voice changer feeding Discord - "stopped" is better than
            # "silently giving them silence." Threshold 50 ≈ 7-12 seconds
            # at chunk_seconds in [0.15, 0.25]; long enough to ride out a
            # transient cuDNN tune but short enough to surface a genuine
            # broken state.
            if self._consecutive_drops >= 50 and not self._stop_event.is_set():
                self._self_abort(
                    f"engine stopping: {n} consecutive inference failures. "
                    f"Last: {type(e).__name__}: {e}"
                )
            return None

    def _process_streaming_16k(self, new_chunk_16k: NDArrayF32) -> NDArrayF32:
        """Streaming variant. Maintains a sliding input history so the model
        sees overlapping content; SOLA crossfades consecutive outputs.

        Returns audio at 16 kHz that's safe to concatenate with the previous
        emitted chunk (assuming the same SOLA stream). Output length is
        approximately equal to the input length once warmed up.
        """
        # Input-side sizing always uses the 16 kHz config (mic input rate).
        cf = self._sola_input_cfg.crossfade_samples
        ctx = self._sola_input_cfg.context_samples
        history_len = ctx + cf

        # Build model input: last (ctx + cf) of input history + the new chunk.
        model_input = np.concatenate([self._input_history, new_chunk_16k.astype(np.float32)])
        # Update history for next call: keep the last (ctx + cf) samples of
        # the combined buffer (these will be the leading samples next time).
        self._input_history = model_input[-history_len:].copy()

        # the streaming path carries
        # pitch state across `_infer` calls so a leading-edge unvoiced
        # run that straddles a chunk boundary can still be bridged.
        full_out = self._infer(model_input, update_pitch_carry=True)

        # Map the trim from input space to output space proportionally -
        # the model is roughly 1:1 in time, but RVC trims a few samples at
        # the boundaries. Compute the per-sample ratio defensively.
        in_len = model_input.shape[0]
        out_len = full_out.shape[0]
        ratio = out_len / max(in_len, 1)
        # Drop the leading "context" portion in the model output. Keep the
        # last `ctx_drop_out` samples - sized to match SOLA's contract:
        #   chunk_n + cf + search   when SOLA is enabled (rc5 - gives the
        #                           alignment search positional slack so
        #                           emit length stays constant)
        #   chunk_n + cf            when SOLA is disabled (legacy path -
        #                           emit length variable, no slack needed)
        # The pre-rc5 implementation always trimmed to chunk_n + cf even
        # with SOLA enabled; the search then ate samples from the input
        # to find alignment, shrinking the emit. See
        # internal notes for why that was wrong.
        sola_search = self._sola_input_cfg.search_samples if self._sola is not None else 0
        ctx_drop_in = max(history_len - cf - sola_search, 0)
        ctx_drop_out = round(ctx_drop_in * ratio)
        emitted_region = full_out[ctx_drop_out:]

        if self._sola is not None:
            return self._sola.process(emitted_region)
        # SOLA disabled - emit raw, expect chunk-boundary clicks for short chunks.
        return emitted_region

    def reset_streaming_state(self) -> None:
        """Clear SOLA + input history so the engine can resume cleanly after a
        stop / start without leaking stale tail from a previous session."""
        self._input_history = np.zeros(
            self._sola_input_cfg.context_samples + self._sola_input_cfg.crossfade_samples,
            dtype=np.float32,
        )
        if self._sola is not None:
            self._sola.reset()
        # the pitch carry must drop
        # too -- the next session's first chunk is logically the start
        # of a new utterance, no prior voiced anchor is in scope.
        self._pitch_carry_f0 = 0.0
        self._pitch_carry_age_frames = -1

    def _warmup_realtime_pipeline(self, n_chunks_per_shape: int = 4) -> None:
        """v0.6.9 - pre-run synthetic chunks through the *full* realtime
        pipeline (cv → rmvpe → rvc) so cuDNN's algo cache is populated for
        the actual shapes we feed at runtime. `RvcSessionPool.warmup` only
        warms the rvc session; the cv and rmvpe sessions still cold-start
        the first few real chunks otherwise.

        v0.7.0-rc9 - extended to pre-warm EVERY unique input length
        soxr's stream resampler can emit, not just the nominal one. The
        rc8 tail-chunk capture pinned the inference p99=96 ms / max=110 ms
        spike to a shape mismatch: pre-rc9 warmup ran `_infer` with
        `chunk_n = chunk_seconds * 16000 = 2400` samples. The realtime
        path calls `_infer` with `model_input.shape[0] = history_len +
        audio16_len`, where `audio16_len` is whatever
        `_StreamResampler(48k → 16k).process(7200)` emits. Soxr
        alternates between two specific values (1957 / 2447 in a sample 48 kHz USB-condenser session, plus the typical 2400) - every chunk with
        a non-cached shape costs cuDNN a fallback slow path, ~80 ms
        inference vs ~40 ms cached. See
        internal notes and the rc8
        tail_chunk_log dump.

        rc9 fix: drive a probe `_StreamResampler` with synthetic 48k
        input, capture every unique `audio16_len` it emits, and pre-warm
        `_infer` with `history_len + audio16_len` for each. Probe is
        independent of the real `_resampler_in` (which doesn't exist
        until `_run_loop` builds it anyway), so its filter state can't
        leak into realtime.
        """
        if self._cv is None or self._rmvpe is None or self._rvc is None:
            return
        chunk_n_mic = round(self.cfg.chunk_seconds * self.cfg.mic_rate)
        if chunk_n_mic <= 0:
            return

        rng = np.random.default_rng(42)

        # Step 1: probe soxr to enumerate the realtime shape set. Run
        # ~20 chunks so the resampler's polyphase filter settles and we
        # see every steady-state emit length (the alternation pattern in
        # the rc8 dump cycles every ~10 chunks).
        unique_audio16_lens: set[int] = set()
        if self.cfg.mic_rate != 16_000:
            probe = _StreamResampler(self.cfg.mic_rate, 16_000)
            for _ in range(20):
                dummy_48k = rng.standard_normal(chunk_n_mic).astype(np.float32) * 0.001
                out_chunk = probe.process(dummy_48k)
                if out_chunk.size > 0:
                    unique_audio16_lens.add(int(out_chunk.shape[0]))
        else:
            # mic_rate == internal - no resample, audio16_len is fixed.
            unique_audio16_lens.add(chunk_n_mic)

        if not unique_audio16_lens:
            # Fall back to the pre-rc9 single-shape behavior so a
            # surprising probe failure doesn't skip warmup entirely.
            unique_audio16_lens.add(round(self.cfg.chunk_seconds * 16_000))

        # Step 2: pre-warm `_infer` with `history_len + audio16_len`
        # for each unique shape. Matches the realtime concat at
        # `_process_streaming_16k`. Multiple iterations per shape so
        # cuDNN's heuristic cache settles to a stable algo choice.
        history_len = self._sola_input_cfg.context_samples + self._sola_input_cfg.crossfade_samples
        for audio16_len in sorted(unique_audio16_lens):
            model_input_len = history_len + audio16_len
            if model_input_len <= 0:
                continue
            dummy = rng.standard_normal(model_input_len).astype(np.float32) * 0.001
            for _ in range(n_chunks_per_shape):
                try:
                    self._infer(dummy)
                except Exception:
                    # If one shape fails, try the rest - the realtime
                    # path's `_safe_process_streaming_16k` will catch
                    # any inference failure that survives warmup.
                    break

        # v0.10.0 - snapshot the model-input shape set seen during warmup
        # (populated by `_infer` instrumentation as `audio16k.shape[-1]`,
        # which equals `history_len + audio16_len` for every warmup call).
        # Runtime continues adding to `unique_audio16_lens`; the diff
        # surfaces shapes that hit cuDNN cold during the realtime session.
        # Reset the rolling per-stage deques so warmup chunks don't
        # pollute realtime percentile reads.
        self.stats.warmup_audio16_lens = set(self.stats.unique_audio16_lens)
        self.stats._recent_cv_ms.clear()
        self.stats._recent_rmvpe_ms.clear()
        self.stats._recent_rvc_ms.clear()
        self.stats._recent_inference.clear()
        self.stats._recent_total.clear()
