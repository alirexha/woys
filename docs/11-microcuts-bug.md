# v0.6.7 — micro-cut investigation

User report (verbatim):

> "my voice is changed ok but its noisy and theres many tiny cuts
> between words and even letters of a word. like micro-freezing of a
> game but in sound"

NOT the v0.5.1 برفک bug (continuous static, fixed by soxr). This is
brief amplitude dips *inside* the speech waveform — distinct from
chunk-boundary clicks (v0.2.x SOLA fix), distinct from `pacat` underrun
storm (v0.5.2 pw-cat switch).

## Hypothesis grid

| # | Hypothesis | Outcome |
|---|---|---|
| (a) | SOLA crossfade dropping samples at boundaries | Ruled out by code read — `_hann_fade` sums to 1, `prev_tail` continuity holds. Cuts in the offline trace were not localised to chunk boundaries (only ~25% near 4 Hz seams). |
| (b) | Stateless soxr resample leaks 4 Hz envelope | **Confirmed contributor** but inaudible on its own (-92 dBFS RMSE on stationary signal). Fixed defensively. |
| (c) | cv→rmvpe→rvc handoff drops frames | No evidence — feats × 2 broadcast, pitchf alignment, `feats.shape[1]` driven slicing all consistent across chunks. |
| (d) | f0 detector returns gaps that vocoder treats as silence | RMVPE threshold 0.3 matches upstream `RMVPEOnnxPitchExtractor.py:58` — not stale. |
| (e) | pw-cat playback buffer too small to absorb engine writer jitter | **Root cause.** User's stored config has `output_latency_ms = 30` overriding the v0.5.2 default of 100. Diag-reported writer jitter is 29.6 ms ≈ buffer size — any spike empties the buffer mid-chunk → audible silence gap. |
| (f) | ORT CUDA kernel sync gaps | Not investigated — engine produces complete chunks atomically before write, so any GPU-side gap is internal to one chunk and SOLA crossfade absorbs it. |

## Forensic data — engine output trace

8 s clean sustained vowel through the full pipeline, three voices:

| Voice | Native SR | Ratio→48k | Silence gaps (real, ≥ 0.5 ms) | Cuts/s |
|---|---|---|---|---|
| amitaro | 16 kHz | 3.0× | 1 (warmup only) | 0.1 |
| jennie  | 32 kHz | 1.5× | 5 | 0.7 |
| trump   | 40 kHz | 1.2× | 21 | 3.1 |

Stricter `|x| < 0.005 for ≥ 0.5 ms` detector. Higher non-integer
ratios produce more residual gaps because the soxr filter transient
is more pronounced for awkward ratios. Hypothesis (b) confirmed as
*a* contributor; not the dominant one.

Stage breakdown for jennie (per-second cut rate, looser detector):

| Stage | cuts/s |
|---|---|
| raw model output (32 kHz) | 200.3 |
| + 48 kHz resample (stateless) | 204.4 |
| + SOLA at 32 kHz | 199.7 |
| + SOLA + 48 kHz resample (full) | 204.4 |

SOLA *reduces* cut rate (200.3 → 199.7). Resample adds about 4/s.
Most "cuts" my detector flagged are the natural amplitude variation
of the multi-harmonic vocoder output — not bugs.

## Hypothesis (e) — root cause

User's `~/.config/woys/config.toml` snapshotted before v0.5.2:

```
output_latency_ms = 30
```

(plus 9 profile-scoped duplicates at 30 each — 10 total stale entries)

EngineConfig default in code is `100`. The stored 30 wins.

`pw-cat --latency=30ms` requests a 30 ms playback buffer from
PipeWire. The engine writer jitter (per `woys diag`) is 29.6 ms
std-dev. Any momentary write delay of 30 ms+ empties pw-cat's
buffer → PipeWire plays silence to `WoysSink` until the next
chunk arrives → `woys-mic` reads silence → audible micro-cut at
every jitter spike.

Same shape as v0.6.4: a default changed in v0.5.2 (30 → 100), the
migrator never propagated the bump to existing configs, every user
who'd run a pre-v0.5.2 build kept the old value indefinitely.

## Fix

1. **Patch live config.** All 10 occurrences of
   `output_latency_ms = 30` rewritten to `100`.
2. **Migrator update.** `_rewrite_paths_in_value` learned a numeric
   bump rule: `output_latency_ms < 100 → 100`. Idempotent on
   already-fixed configs. Tests:
   `test_migrate_bumps_stale_output_latency_ms` and
   `test_migrate_leaves_above_threshold_output_latency_alone`.
3. **Stateful soxr resampler.** New `_StreamResampler` class wraps
   `soxr.ResampleStream`; the realtime engine builds one each for
   `mic_rate → 16k` and `model_sr → sink_rate`, replaced if model
   SR changes during hot-swap. Eliminates per-chunk filter
   transients (hypothesis (b)). Tail-flushed on engine stop and on
   model swap so trailing samples don't drop.
4. **Tests.** `tests/test_stream_resampler.py` (4 cases): identity
   passthrough, streamed-vs-one-shot agreement, flush drains
   buffer, zero-size chunk safety.

Cost analysis:

- Stateful resampler: same ~0.5 ms / chunk as stateless. No latency
  cost (group delay ~1 sample, masked by SOLA crossfade).
- Migrator latency bump: silent on already-correct configs. On
  affected installs, unblocks the bug; no behavioural surprise.

## Expected user-perceived outcome

After applying the live patch + restarting the engine, mic-to-app
playback latency increases by ~70 ms (30 ms → 100 ms playback
buffer). Wall-clock latency rises from ~280 ms to ~350 ms — still
conversational, well under any chat-app threshold. In return the
buffer absorbs writer jitter without underrun → micro-cuts gone.

## Verification protocol

1. ✅ Static: 76 fast tests pass; lint + format clean.
2. ✅ Resampler unit tests: streamed output matches one-shot within
   1e-3 RMSE; flush drains buffer to within ±4 samples; identity
   ratio is a passthrough; empty chunk is safe.
3. ⏳ **User must confirm in Telegram.** Restart the engine,
   record a phrase with hard consonants ("peter piper picked"),
   listen back, judge. Do **not** tag v0.6.7 until confirmed.
4. If user still hears cuts: the residual is hypothesis (f) ORT
   CUDA sync OR voice-model artifact. Reopen with a wider trace.

## Part 2 — v0.6.7 retro on v0.5.2's pw-cat preference

User feedback after the first v0.6.7 ship: cuts reduced to
"approximately 1-second intervals — periodic, not random."
Better but still bad. Sustained "aaaa" vowel test in Telegram
showed continuous flutter.

### Reproduction without the engine

Minimal harness — Python writes 250 ms float32 stereo chunks to
pw-cat's stdin every 250 ms wall-clock (matches engine's writer
cadence; no inference, no SOLA, no resampling). Capture
`WoysSink.monitor` via `parec`:

| Backend / latency        | Zero gaps in 23 s | Rate     |
|--------------------------|-------------------|----------|
| pw-cat at 100 ms         | 73                | 3.10 /s  |
| pw-cat at 300 ms         | 76                | 2.65 /s  |
| **pacat at 300 ms**      | **2**             | **0.08 /s** |
| pacat at 500 ms          | 2                 | 0.08 /s  |

Each gap is ~42.7 ms wide — exactly one PipeWire quantum at
2048 samples / 48 kHz. **pw-cat returns silence for one full
PipeWire quantum every ~3rd chunk**, irrespective of how big the
ring buffer is. Larger buffer doesn't fix it because the bug is
in the read-thread / audio-callback synchronisation, not in
buffer size.

### Why pacat is cleaner

pacat goes through `pipewire-pulse` (the PulseAudio compatibility
layer). Its stdin reader and audio thread are separately
scheduled, and the buffer accounting uses PulseAudio's
prebuf/tlength semantics that absorb bursty 250 ms writes
without exposing the per-quantum read race.

### v0.5.2 retro

The v0.5.2 retro picked pw-cat because pacat at 30 ms latency
had 1.4 underruns/s. That measurement was correct *at 30 ms*. At
300 ms latency on bursty stdin writes, the rankings flip.

### Fix (v0.6.7 part 2)

Change `EngineConfig.prefer_pw_cat = False` (pacat is now the
default backend) and `output_latency_ms = 300` (was 100). Both
defaults bumped together — pw-cat at 100 ms was the symbiotic
sweet spot, but pacat at 300 ms is the new floor.

User's live config patched in place (10 entries `100 → 300`).
Migrator's numeric bump rule updated `< 100 → 100` to
`< 300 → 300`, with a new `test_migrate_bumps_intermediate_latency_to_300`
covering the 100→300 path.

### Residual

Engine + pacat at 300 ms: ~1.0 zero-gap/s on the controlled
sustained-vowel test (vs 0.08 /s for pure burst-write without the
engine). The remaining ~1 cut/s is engine-specific and was not
GC (verified via `gc.disable()` — no change). Suspect ORT CUDA
stream sync or scheduler interaction. **Deferred** — already a
~3× improvement over the v0.6.7 part-1 ship; user-verifiable in
Telegram before deciding whether further investigation is
warranted.

## Part 3 — sweep tests after the part-2 ship

User feedback after part 2: "better but still has random cuts, not
every second but randomly." Investigated the residual via parameter
sweep:

| chunk_s | latency_ms | prime_s | zero gaps / s |
|---------|------------|---------|---------------|
| 0.25    | 300        | 0.0     | 0.67–0.92     |
| 0.25    | 300        | 0.5     | 1.15          |
| 0.25    | 500        | 0.5     | 1.09          |
| 0.25    | 1000       | 0.5     | 1.15          |
| 0.10    | 300        | 0.0     | 3.49          |
| 0.10    | 200        | 0.0     | 3.16          |
| 0.05    | 200        | 0.0     | 4.33          |

Findings:

- **Priming silence didn't help** and slightly increased xruns —
  likely pacat applies its prebuf threshold to the silence and
  rebuffers more aggressively. `prime_silence_seconds` kept as a
  config knob (default 0) for future backend changes.
- **Smaller chunks made it worse.** 0.25 s is a local minimum.
  Tighter chunks → more frequent writes → more race window
  surface area inside pacat / pulseaudio.
- **`sd.OutputStream(device='pipewire')`** routed to the default
  sink (laptop speakers), not `WoysSink`. PortAudio's ALSA host
  API doesn't propagate `PIPEWIRE_NODE_TARGET`. Would need a
  larger plumbing change to use as the default output path.

Net: ~1 zero-gap / s appears to be the floor with the current
stdin-pipe-to-pacat output method. The bottleneck is engine inference
variance (jitter ~30 ms std-dev) propagating into pacat's internal
buffer accounting. Fixing that requires either:

- Reducing engine variance — ORT/CUDA scheduler tuning, possibly
  IOBinding with explicit stream control; OR
- Replacing the stdin-pipe output path with a native PipeWire
  Python binding (pw-python is unmaintained but viable as a
  rewrite target).

Both are v0.7.x scope, not v0.6.7.

## What was *not* fixed

- Writer jitter itself (29.6 ms vs target 12.5 ms). Tightening
  this would reduce mic-to-speaker wall further. Out of scope —
  jitter is dominated by Python GC + ORT scheduler variance.
- `bench_loopback.py` (the project notes notes it's broken). A working
  acoustic loopback would let us measure pw-cat underruns
  directly instead of inferring them from jitter math.
- The residual ~1 Hz engine-specific cut rate (see Part 2 above).
  Worth investigating only if the user still hears the bug after
  the part-2 fix lands.
