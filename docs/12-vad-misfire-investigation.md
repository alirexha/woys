# 12 — VAD misfire investigation (v0.6.9)

**Date:** 2026-05-06
**Status:** root cause identified; fix applied; awaiting live re-verification.

## What we measured

Diagnostic tool: `woys-diag v0.1.0`.
Capture: 60 s through `woys-mic` with the **e_girl** voice loaded, structured 7-block protocol (silence → vowel → counting → plosives → fricative → sentence → silence).

**Headline numbers:**

- 22 silent-gap dropouts in 60 s (22 / min) — verdict: *Significant artifacts*.
- 1 click discontinuity (single sample, not chunk-aligned).
- **0 / 22 gaps within ±25 ms of a 0.25 s chunk boundary.** Distribution clusters at 60–122 ms offsets — i.e., near the *middle* of chunks, not the edges.
- Most-affected block: **vowel — 7 gaps in 10 s** (sustained "aaaaaa", which should be the trivially-easiest signal for any voice changer).
- Tail-silence block (53–60 s): mean RMS −24.7 dBFS with 3 detected gaps, despite the user not speaking.

**The chunk-alignment finding is the kill shot for the prior v0.7.0 ring buffer plan.** A chunk-stitching bug would cluster gaps *at* boundaries; this distribution does the opposite. Whatever the bug is, it is sub-chunk and content-driven, not stitching.

Capture + report archived: `~/.local/share/woys-diag/{captures,reports}/20260506-142615__woys-mic.*`.

## Hypotheses, evidence, verdict

Three parallel investigations of the woys + vendored RVC code path:

### Hypothesis 1 — VAD / silence-gate in the woys engine — **REJECTED**

Searched `src/audio/engine.py`, `src/audio/`, `src/woys/`, `src/server/voice_changer/`. The only RMS measurement is `engine.py:1332` (`self.stats.last_input_rms = rms`) — recorded for telemetry, never used to gate output. There is no input-side mute, no `silence_threshold`, no `gate_db`. `VolumeExtractor.get_mask_from_volume` exists in `src/server/voice_changer/common/` but is dead code (no caller).

Conclusion: the engine itself does not gate. This rules out a misconfigured threshold in our code as the cause of the 22 mid-utterance gaps.

### Hypothesis 2 — f0 NaN / zero → vocoder zeros harmonics — **CONFIRMED, this is the bug**

The RVC vocoder in `src/server/voice_changer/RVC/inferencer/rvc_models/infer_pack/models.py` uses an `NSFSourceModule`-style harmonic-plus-noise source. Its `SineGen._f02uv` mask (line 348):

```python
uv = uv * (f0 > self.voiced_threshold)   # voiced_threshold = 0
```

Then line 388 multiplies the sine source by this mask:

```python
sine_waves = sine_waves * uv + noise
```

When any single frame of `f0` is `NaN` or `≤ 0`, `uv` becomes 0 for that frame, the harmonic source is zeroed, and only the noise term remains. At RVC's frame granularity (~10 ms per f0 frame at 16 kHz × 160-sample hop), a single dropped frame produces a ~10 ms harmonic dropout. RMVPE / FCPE return `NaN` on flat-salience or low-energy frames mid-vowel — 22 such failures across 60 s ≈ 1 every 2.7 s, lining up with the observed gap density.

The pipeline does *not* sanitize `pitchf` between extraction and the vocoder:

- `RMVPEPitchExtractor.extract()` writes raw `f0` (potentially `NaN`) into `pitchf` (`RMVPEPitchExtractor.py:31-34`); next line scales by `2 ** (f0_up_key / 12)` which propagates `NaN`.
- `Pipeline.extractPitch()` (`Pipeline.py:77-100`) wraps the array in a tensor without an `isnan` check.
- `Pipeline.exec()` passes the tensor straight to `self.infer(...)` (`Pipeline.py:254`).

The 22-82 ms observed gap durations are 1-8 frames at 10 ms/frame — exactly the SineGen-zero pattern, not a chunk-boundary pattern.

### Hypothesis 3 — engine emits hallucination during silence — **CONFIRMED, secondary issue**

`engine._run_loop` (`engine.py:1316-1378`) reads every mic chunk and unconditionally passes it through `_safe_process_streaming_16k` → `_infer`. The vocoder, given near-silent input, doesn't emit zero — it reconstructs a baseline "voicing floor" from its embedding + pitch path. Result: −24.7 dBFS phantom output during the user-silent block. Audible as constant low-level hum / breathing-like artifacts during pauses.

Not the cause of the mid-utterance gaps, but a real pause-quality issue worth fixing in the same release.

## Fix plan for v0.6.9

Two changes, both small:

### Fix A — sanitize `pitchf` before vocoder (the kill-shot fix)

File: `src/server/voice_changer/RVC/pipeline/Pipeline.py`. In `extractPitch()`, after the tensor conversion, replace `NaN` values with `0`, then linear-interpolate runs of unvoiced/zero frames *up to N frames long* between two voiced frames. Long unvoiced runs (truly silent) are left as zeros so the vocoder correctly produces silence.

This stops single-frame extractor failures from producing audible dropouts mid-utterance, while preserving the voiced/unvoiced distinction the rest of the pipeline depends on.

License: `src/server/` is vendored MIT — fix lives there and inherits MIT, per `NOTICE`.

### Fix B — input-level gate in the engine

File: `src/audio/engine.py`. Add a configurable `input_gate_dbfs` knob (default `-55.0`); when input RMS is below it, emit zeros directly instead of running inference. Stops the −24.7 dBFS phantom emission during pauses.

Configurable so users with noisy mics can lower it (or set it to `-inf` to disable).

License: original woys code, proprietary.

## Verification gate

The fix is "shipped" only when:

1. woys's pytest stays green (`.venv/bin/python -m pytest tests/ -m "not slow"`).
2. A re-run of `woys-diag run --duration 60 --source woys-mic --voice e_girl` reports cuts/min < 5 (down from 23/min). Anything above that and we iterate.

If verification passes → bump to v0.6.9, append CHANGELOG, commit. If not → re-investigate; the diagnostic harness gives us a tight feedback loop.

## Round 2: 23/min → 16/min, vowel still dominates → NaN→int16 cast

First-round fixes (pitchf interpolation + input gate) cut 23/min → 16/min (~30 % reduction). Tail_silence dropped 3 → 1 (input gate worked); vowel block went 7 → 8 (no help — pitch wasn't the root cause there). Cuts still 0/14 chunk-aligned.

Second investigation followed the *vowel-dominant + chunk-mid offset (55–125 ms)* pattern. Two leads:

### Round-2 root cause — `audio1.to(dtype=torch.int16)` after NaN

`Pipeline.infer()` (`Pipeline.py:122` pre-fix) cast the inferencer's float output to int16 with no NaN sanitization:

```python
audio1 = (audio1 * 32767.5).data.to(dtype=torch.int16)
```

Float→int16 with `NaN` is undefined behavior in C/C++; on x86_64 Linux PyTorch typically lands NaN near `INT16_MIN` (−32768). One NaN sample becomes a peak negative spike — a click + a perceived gap to the listener. RVC inferencers can emit single-frame NaN under input distributions outside training (sustained vowel ≈ pure tone is unusual for a model trained on conversational speech). This is exactly the "8 cuts mid-sustained-vowel" signature.

Fix: `torch.nan_to_num(audio1, nan=0.0, posinf=1.0, neginf=-1.0)` before the cast.

### Round-2 secondary — partial-NaN slipping past `extractFeatures` guard

`Pipeline.extractFeatures()` had `if torch.isnan(feats).all(): raise DeviceCannotSupportHalfPrecisionException()` — only catches *total* NaN (the half-precision-failure signal). A few-frame NaN burst from the embedder slips through, propagates into the inferencer, and becomes more NaN samples in `audio1` → same click-and-silence path.

Fix: keep the `.all()` branch (it's load-bearing for the half-precision retry), and add a `if torch.isnan(feats).any(): feats = torch.nan_to_num(feats, nan=0.0)` line right after.

### What we deliberately punted

SOLA on highly-periodic content (sustained vowel = pure tone): the correlation argmax can land on integer pitch periods, producing crossfade misalignment. But the offsets that pathology would produce (~0–20 ms from chunk boundary) don't match our observed 55–125 ms cluster. Re-investigate only if rounds 1+2 don't get cuts/min below 5.

## Re-verification gate (round 2)

Same as before: pytest stays green, woys-diag re-run shows cuts/min < 5. Stack the fixes; ship combined as v0.6.9.

## Round 3 — round-2 fixes were dead code, the engine has its own inference path

Re-running after round 2: 16 → 11 cuts/min. Vowel block dropped 8 → 4 (passed the round-2 falsification threshold), but a 5-cut cluster appeared in the `normal` block (46–50 s).

**Critical discovery while investigating the residual:** the woys realtime engine **does not call `Pipeline.exec()` from upstream**. `engine._infer()` (`src/audio/engine.py:840`) implements its own ONNX dispatch directly: `self._cv.run(...)` for the embedder, `self._rmvpe.run(...)` for pitch, `self._rvc.run(...)` for the vocoder. The upstream `Pipeline` class is unreachable from the realtime path — it's only used by upstream's offline conversion utilities, which woys doesn't ship.

This means **the round-1 pitchf interpolation and round-2 NaN-cast / partial-NaN feats fixes never executed in the production engine**. The 23 → 16 → 11 trajectory was almost entirely the **input-gate fix** (which lives in the right place, `engine._run_loop`) plus run-to-run variance.

The Pipeline.py edits remain as defensive guards in case some other path ever invokes that class, but they are not load-bearing. The actual fix needs to live in `engine._infer()`.

### Round-3 fix — port the same sanitization to the engine inline path

In `src/audio/engine.py:_infer()`:

1. After `feats = self._extract_feats(audio16k)` — `if np.isnan(feats).any(): feats = np.nan_to_num(feats, nan=0.0)`. Catches partial-NaN bursts from the embedder before they propagate.
2. After `pitchf = pitchf_raw.astype(np.float32).squeeze()` — call a new module-level helper `_interpolate_voiced_gaps_np(pitchf)` that does the same as the (dead) torch helper in `Pipeline.py`: replace NaN with 0, then linearly interpolate runs ≤ `_VOICED_GAP_MAX_FRAMES` between two voiced frames.
3. After the model returns (`result = np.array(out).astype(np.float32).squeeze()`) — `if np.isnan(result).any() or np.isinf(result).any(): result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)`. Pacat is fed `float32le`; NaN bytes in the stream are undefined behavior in PipeWire.

### Round-3 instrumentation — visibility into timing outliers

We don't yet know whether residual cuts are content/numerical (NaN) or load/timing (slow chunks). To distinguish, added three new fields to `EngineStats`:

- `max_inference_ms` — slowest single-chunk model call since session start
- `max_total_ms` — slowest single-chunk end-to-end (read → resample → infer → resample → enqueue) since session start
- `late_chunks` — count of chunks whose total time exceeded `chunk_seconds * 1000` (i.e., the chunk arrived at the sink later than the next chunk's deadline)

Exposed via the existing TUI socket `STATUS` command. After a re-run, compare:

- If `late_chunks` is roughly equal to the woys-diag cut count → timing outliers are the cause; mitigations include lifting `chunk_seconds` (already at 0.25 s — pinned by LESSONS §9 per cuDNN warmup), adding GPU warmup chunks at engine start, or pinning the engine to a real-time scheduling policy.
- If `late_chunks` is near zero → cuts are numerical/content-driven; SOLA crossfade pathology on periodic content moves up the suspect list (already analyzed but punted; offsets don't quite match observed distribution).

### Verification gate (round 3)

Same target: cuts/min < 5. Plus, after the run, capture STATUS so we know whether `late_chunks > 0`. The `STATUS` line now includes `max_total_ms=… late_chunks=N/M`.

## Round 4 — STATUS revealed two stacked causes

After round 3, live `STATUS` returned `avg_total_ms=74.2 max_total_ms=456.2 late_chunks=4/246`. Average chunk processing 74 ms (well under 250 ms budget), but max hit 456 ms (1.8x budget). Four chunks of 246 ran late — 1.6 % of chunks. That accounts for a ~4 cuts/min ceiling from timing alone; the rest must be SOLA / phase-alignment.

### Round-4 fix A — pre-warm cv + rmvpe + rvc together at engine.start()

The existing `RvcSessionPool.warmup` only warms the **rvc** session. The contentvec and rmvpe sessions still cold-start on the first real chunks, which is exactly when the audio sink has the least slack.

Added `RealtimeEngine._warmup_realtime_pipeline(n=4)`: runs four synthetic 250 ms chunks through the entire realtime pipeline (`cv.run → rmvpe.run → rvc.run`) right after `_ensure_sessions()` and before the audio thread starts. cuDNN's algo cache populates for the actual shapes; the first real chunks land at warm-cache speed.

Cost: ~300–500 ms added to engine startup. Acceptable; the user already waits ~1 s for the TUI to come up.

### Round-4 fix B — SOLA defaults that work for sustained periodic content

The SOLA correlation search runs at the model output rate (40 kHz for v2 RVC voices like e_girl). With `search_ms = 4 ms` (the previous default), the search window is **160 samples**. A 200 Hz vowel has a period of 200 samples. The search window is **less than one full period**, so SOLA cannot reach a phase-aligned offset for sustained voicing if the model emits chunks with phase shifts beyond ±144°.

Changes to `EngineConfig` defaults:

- `sola_search_ms`: 4.0 → **6.0**. At 40 kHz that's 240 samples — covers ≥ 1 full period for any voice with f0 ≥ 167 Hz.
- `sola_corr_threshold`: 0.25 → **0.10** (added as a new EngineConfig knob; was previously a hard-coded 0.25 in `SOLAConfig`). With the original threshold, SOLA falls back to centered (`offset = 0`) on borderline cases and produces phase-discontinuous crossfades for sustained content. 0.10 still rejects decorrelated noise but keeps best-effort alignment for periodic signals.

Both threaded through `_ensure_sessions` and `_rebuild_sola_for_rate` so the input-side and output-side SOLA streams pick them up.

### Verification gate (round 4)

Same target: cuts/min < 5. STATUS should now show `late_chunks` close to 0 (warmup eliminated cold-start outliers). If cuts/min stays above 5 with `late_chunks ≈ 0`, the residual is in a layer we haven't touched — possibly RVC's training-data mismatch on sustained content (would need model-level mitigation).

## What this rules out

- v0.7.0 ring buffer rewrite is **the wrong fix**. The data does not support a chunk-stitching root cause. Cancelled.
- Discord's Krisp / OS noise suppression is **not** the cause. The capture is from `woys-mic` directly, before any consumer.
- Microphone-side issue is **not** the cause. The HyperX produces clean voiced audio.

## What it implies for the diagnostic harness

The chunk-offset distribution table in the report is the single highest-signal artifact for distinguishing chunk-stitch bugs from frame-rate bugs. Worth promoting it from a per-row column to a separate "chunk alignment histogram" section in `woys-diag` v0.2.0.
