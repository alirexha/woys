# v0.5.1 — Audio quality bug investigation

> Pre-fix root-cause trace. The user reported "noisy, with micro noises and
> scratches throughout the playback" on every voice except Amitaro after
> v0.5.0. This file documents the diagnostic measurements before a single
> line of fix code was written.

## Diagnostic methodology

Synthesized a 1 s 1 kHz sine at 48 kHz, ran it through each candidate
resampler chain (`48k → 40k → 48k`), measured per-sample RMSE vs the
original signal and the high-frequency-error ratio. Then ran a 3 s
220 Hz harmonic stack through the engine at multiple chunk sizes,
measured high-frequency energy in the output.

## Hypothesis A — SOLA chunk-size artifacts → **NOT the cause**

| chunk_seconds | output dur (3 s input) | HF > 12 kHz % | HF > 4 kHz % |
|---|---|---|---|
| 0.10 | 2.70 s | 0.00 % | 0.08 % |
| 0.15 | 2.70 s | 0.01 % | 0.19 % |
| 0.25 | 2.80 s | 0.01 % | 0.24 % |
| 0.50 | 2.98 s | 0.01 % | 0.15 % |

HF energy stays flat across chunk sizes. SOLA isn't introducing chunk-
boundary impulses. There's a separate output-duration shortfall at small
chunks (10 % at 0.1 s, dropping to 1 % at 0.5 s) — that's the SOLA
crossfade region eating into the emit, not a quality artifact.

**Verdict**: not warranting a fix in this release. The brief's stopgap
(`chunk_seconds = 0.25`) helps duration consistency but isn't the
quality fix.

## Hypothesis B — Resampler aliasing → **CONFIRMED MAJOR ISSUE**

The current `_resample_linear` is a 2-tap linear interpolator. It has no
anti-aliasing low-pass, so frequencies above the destination Nyquist
fold back into the signal as audible high-frequency noise.

Measured against soxr (which we already have installed via librosa):

| Resampler | Round-trip RMSE | HF error fraction (>8 kHz of the 24 kHz Nyquist) |
|---|---:|---:|
| `_resample_linear` (current) | **0.001330** | **8.45 %** |
| `soxr` quality=HQ | 0.000044 | 0.4 % (rest is below noise floor) |
| `soxr` quality=VHQ | 0.000044 | 0.4 % |

**30× worse RMSE on linear**, and ~20× more high-frequency content in the
error signal. That high-frequency error sounds exactly like the "scratchy /
crispy / micro-noise" the user reported.

Worse: the engine resamples **twice** per chunk (mic 48k → 16k for the
embedder, then output `model_sr → 48k` for the sink), so the linear
artifacts compound.

**Verdict**: this is the fix. Swap `_resample_linear` to `soxr` quality=HQ.
Cost: < 1 ms per resample on this CPU, no measurable latency hit.

## Hypothesis C — f0 chunk continuity → **Possible secondary contributor**

Brief: "RMVPE pitch detection needs ~150 ms+ of context for clean f0
tracking. At 100 ms chunks, f0 jumps between adjacent chunks → micro-pitch
jitter."

Reality: my engine already feeds RMVPE the **full** `(input_history + new_chunk)` buffer (~250 ms), not just the new 100 ms chunk. So pitch detection has plenty of context. Pitch values *do* re-compute per chunk, which means there's room for trajectory smoothing across chunks — a future polish.

**Verdict**: not warranting a v0.5.1 change. Mark as v0.6.0 candidate.

## Hypothesis D — Input gain / clipping → **Worth wiring**

Brief: user's HyperX QuadCast was at 70 % input volume. RVC amplifies
clipped input as harsh distortion.

Adding `input_gain_db` is cheap and useful as a safety knob even if it's
not THE bug. Adds an `EngineConfig.input_gain_db` field and a per-profile
override; applied via `audio = audio * 10 ** (gain_db / 20)` on each
chunk before resampling. Default 0 dB.

**Verdict**: ship as a small UX add. Not the root-cause fix.

## Hypothesis E — pacat output underruns → **Likely fine**

The current pacat invocation already includes
`--latency-msec={output_latency_ms}` (default 30 ms). Underruns would
manifest as glitches synchronized to playback timing, not as a continuous
scratch through the audio.

**Verdict**: not investigating further. If a v0.5.1 user still hears
scratches after the resampler fix, this is the next thing to revisit.

## Fix plan for v0.5.1

1. **Replace `_resample_linear` with `_resample_soxr`** at all three call
   sites (mic 48k → 16k input, model_sr → sink 48k output, SOLA tail flush).
   Keep `_resample_linear` as `_resample_linear_legacy` for tests that
   need a known-bad reference.

2. **Add `EngineConfig.input_gain_db: float = 0.0`** + apply pre-resample.
   Plumb through AppConfig + the profile snapshot fields list.

3. **Per-voice `chunk_seconds` autopick on convert**. New voices get
   `chunk_seconds=0.25` baked into their auto-profile (the v0.5.1 stopgap
   default). Existing profiles in user config aren't touched on upgrade.

4. **Extend `tests/test_voice_quality.py`** with:
   - `test_no_chunk_boundary_impulses_per_voice` — short-time energy
     deltas at chunk boundaries vs interior.
   - `test_no_aliasing_above_nyquist` — content above model_sr/2 must be
     ≤ −40 dB rel speech RMS.
   - `test_noise_floor_quiet_vs_active` — SNR ≥ 25 dB on a synth with
     silent intervals.

## What did NOT change in v0.5.1

- SOLA math — works correctly per the chunk-size sweep above.
- f0 detector — already has 250 ms context.
- pacat invocation — latency target unchanged.
- Voice-library import / convert subcommand — out of scope.

## Why v0.5.0 missed this

The v0.5.0 real-audio QA harness asserted *output duration* (catches the
chipmunk bug) and *cross-voice distinguishability* (catches "swap is
cosmetic"). It did NOT measure high-frequency artifact content. That's
the gap v0.5.1's harness extension closes.

The duration-and-band-energy gates were a useful first cut but they pass
even when the audio sounds scratchy, because the *gross* energy
distribution looks fine. Spectral-quality assertions (SNR, aliasing,
boundary impulses) catch the actual user complaint.
