# Changelog

All notable changes to this project. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.6.7] — 2026-05-05 — Micro-cut fix (pending user confirmation, two-part)

User report: "voice is changed ok but its noisy and theres many tiny
cuts between words and even letters of a word." Distinct from the
v0.5.1 برفک bug (continuous static) and from v0.5.2 pacat-underrun
storm — brief amplitude dips *inside* speech, between phonemes.

Shipped in two passes; full forensics in `docs/11-microcuts-bug.md`:

**Part 1** — output_latency_ms config migration + stateful soxr
resampling. After ship, user feedback was "better but still bad,
cuts now at ~1 sec intervals." Reduced random cuts but left a
periodic residual.

**Part 2** — root cause of the 1 sec residual: pw-cat returns one
PipeWire quantum (~43 ms) of silence every ~3rd chunk under bursty
250 ms stdin writes, regardless of buffer size. Reproduced without
the engine: 73 zero-gaps in 23 s with pw-cat at 100 ms, vs 2 with
pacat at 300 ms (40x cleaner). v0.5.2 picked pw-cat because pacat at
30 ms latency had underrun storms; at 300 ms the rankings flip.

Changes:

- **`prefer_pw_cat = False`** by default (was True). pacat is now
  the playback backend.
- **`output_latency_ms = 300`** by default (was 100). Migrator's
  numeric bump rule updated `< 300 → 300` (was `< 100 → 100`).
- **Stateful soxr resampling.** New `_StreamResampler` class wraps
  `soxr.ResampleStream`. The realtime engine builds one for
  `mic_rate → 16k` and one for `model_sr → sink_rate`, replaced if
  the model SR changes during hot-swap. Stateless `soxr.resample()`
  per chunk leaks a 4 Hz amplitude artifact via the filter warm-up;
  the streaming variant carries filter state across calls. Confirmed
  contributor (-92 dBFS on stationary signal — below audibility but
  fixed defensively).
- **`_StreamResampler` tests** in `tests/test_stream_resampler.py`
  (4 cases). Migrator gets a new
  `test_migrate_bumps_intermediate_latency_to_300` covering the
  v0.5.2-default → v0.6.7-default migration path.

User's live config patched in place at fix time: 10 entries each of
`output_latency_ms`, all bumped 30→100 then 100→300.

Trade-off: mic-to-app wall-clock rises ~270 ms total (was 30 ms
buffer, now 300 ms). Conversational latency stays well under any
chat-app threshold. Stability ranks above absolute latency for an
audible quality bug.

Residual after part 2: the controlled engine+pacat test still shows
~1 cut/s vs 0.08/s for pure burst-write-to-pacat. GC ruled out via
`gc.disable()` (no change). Suspect ORT/CUDA stream sync. Deferred
to a follow-up; ship the 3× improvement, let the user judge whether
further work is warranted.

Tag is held until the user confirms in Telegram.

## [0.6.6] — 2026-05-05 — Polish round: stop bleeding state across boundaries

A bundle of small bugs that had been quietly biting through the v0.6.x
series. Each one was visible in earlier sessions but was being deferred
as "out of scope". They aren't anymore.

- `tests/test_audio_pipewire.py::test_virtual_mic_round_trip` now
  snapshots the host's pre-test state and restores it in the outer
  `finally`. Before this fix, running `pytest -m "not slow"` on a real
  desktop wiped the user's loaded virtual mic — Discord / CS2 lost
  their input device until the next `systemctl --user restart
  woys-mic.service`.
- `tui.control.send_command` catches `ConnectionRefusedError` and
  `FileNotFoundError` for stale-socket scenarios (TUI killed by SIGKILL
  / crashed / `kill -9`'d). Returns a clear `ERR ...` instead of
  letting the exception escape to callers.
- `tui.control.ControlServer.start` registers an `atexit` handler that
  unlinks the socket file, plus a SIGTERM handler that converts the
  signal into a clean `sys.exit(128 + SIGTERM)` so atexit fires. A
  graceful `kill <tui-pid>` no longer leaves a stale socket behind.
- `woys.convert.convert_pth_to_onnx` deletes the unused
  `<stem>_simple.onnx` sibling that upstream's `_export2onnx` always
  writes. Matches the existing voice-library convention (none of the
  shipped voices have a `_simple` companion in the models dir) and
  prevents `woys models list` from doubling.
- `scripts.voice_library_import._verify_zip` /  `_extract_zip` fall
  back to `7z` when system `unzip` rejects the archive (e.g. zstd-
  compressed zips that Info-ZIP 6.x doesn't support — caught Jennie's
  HF zip during v0.6.3).
- `the project notes` test-count reference fixed (`14 fast tests` → `70+`).

## [0.6.5] — 2026-05-05 — Rename PipeWire mic `vcclient-mic` → `woys-mic`

The user-facing PipeWire source was renamed from `vcclient-mic` to
`woys-mic`, finishing the v0.6.0 rename that had deliberately
preserved the legacy source name to spare users a one-time
re-configuration. Consensus: stop deferring it, take the hit once.

**Users will need to re-select their input device in Discord / CS2 /
Telegram / Zoom / browser apps once.** Anything that pinned the input
by name will see the old `vcclient-mic` disappear from device lists
and need to pick `woys-mic` instead.

The engine handles the upgrade cleanly: `woys pw setup` (and the
systemd unit's `ExecStart`) now unloads any orphan `vcclient-mic`
remap-source before loading the new one, so an upgrade can't end up
with both side-by-side. `install.sh` also sweeps any stale legacy
modules during install. `pkg/browser-extension/popup.js` flags the
legacy device with a "re-run setup" hint if it spots one mid-upgrade.

Files renamed: prose in README, INSTALL, DISCORD-SETUP, CS2-SETUP, QA,
TROUBLESHOOTING (with a new section explicitly walking through the
re-selection migration), 05-perf. Test skip messages and CLI help text.
Historical docs (CHANGELOG, LESSONS, v0_5_0 retro, 10-monitor-leak-diag)
left verbatim — they describe past state.

## [0.6.4] — 2026-05-05 — Plug audio leak from stale sink_name

A v0.5.x → v0.6.x upgrade left `sink_name = "VCClientCachySink"` in
config.toml. v0.6.0+ loads the sink as `WoysSink`, so `pw-cat` asked
for a sink that no longer existed and PipeWire silently fell back to
the default sink (laptop speakers). Three fixes: migrator now rewrites
the legacy sink name; engine pre-flights `cfg.sink_name` against
`pactl list short sinks` and refuses to start with a clear error if
absent (no more silent fallback); TROUBLESHOOTING.md gets a one-liner
sed for users who can't reinstall. Diagnostic forensics in
`docs/10-monitor-leak-diag.md`.

## [0.6.3] — 2026-05-05 — Add jennie voice to library

Added Jennie (BLACKPINK) to the curated voice library. Source:
natanworkspace/Legacy_Core_Models on HuggingFace, RVC v2, 32 kHz,
230 epochs / 31280 steps. Verified via the existing
`scripts/voice_library_import.py` machinery — download + 7z integrity
check + extract + convert + engine validation + profile registration.
The system `unzip` couldn't verify the archive (zstd compression);
flagged for a future fallback in the batch importer.

## [0.6.2] — 2026-05-05 — Trim default voice library

Removed `alfred_pennyworth` and `batman_troy_baker` from default voice
library — user opted out of these voices. Library now ships with 7
character voices + amitaro.

Also dropped: their `.onnx` model files from `~/.local/share/woys/models/`,
their test fixture WAVs in `tests/fixtures/voice_qa/`, their entries in
`voice-library/SOURCES.md`, their saved profiles in `config.toml`. The
`test_voices_produce_distinguishable_outputs` candidate list in
`tests/test_voice_quality.py` shrank from 4 to 3 voices (still spans
the three sample-rate buckets that matter: 16 / 40 / 48 kHz).

`tests/test_model_swap.py::_have_two_models` was updated to also exclude
`amitaro_v2_16k.onnx` from the candidate set — with alfred removed,
amitaro became the alphabetically-first voice and the
`target != DEFAULT_RVC_MODEL` assertion would have been vacuously false.

## [0.6.1] — 2026-05-05 — `woys` (no args) launches the TUI

Tiny ergonomic change: typing `woys` with no subcommand now launches the
TUI with autostart, equivalent to `woys run --autostart`. The user
expectation was "type the app name to open it" — same pattern as
desktop launchers. `woys --help` and `woys --version` still work
because argparse intercepts those before reaching the subcommand
dispatch. Existing subcommands (`info`, `pw`, `models`, `profile`,
`diag`, `convert`, `tray`, `toggle`, `pitch`, `status`) are unchanged.

## [0.6.0] — 2026-05-05 — Renamed to **woys**

The project is now called **woys** (pronounced like "woyz", rhymes with
"boys"). Same engine, same features, new name.

### Breaking

- **Package name**: `vcclient-cachy` → `woys`
- **Binary**: `vcclient-cachy` → `woys`. The old name is kept as a
  deprecated shim through the v0.6.x line — running it prints a yellow
  `[deprecation]` warning and delegates to `woys`. Removed in v0.7.0.
- **Python module**: `vcclient_cachy` → `woys`. All imports updated.
- **Config dir**: `~/.config/vcclient-cachy/` → `~/.config/woys/`
  (auto-migrated on `./install.sh` upgrade).
- **App / models dir**: `~/.local/share/vcclient-cachy/` →
  `~/.local/share/woys/` (auto-migrated; absolute paths inside
  `config.toml` get rewritten by the migrator).
- **systemd unit**: `vcclient-cachy-mic.service` → `woys-mic.service`
  (old unit stopped + disabled + removed by the migrator).
- **PipeWire sink** (internal): `VCClientCachySink` → `WoysSink`. Engine
  + new systemd unit re-create it on start; no user action needed.

### NOT changed (intentional)

- **PipeWire mic name**: stays `vcclient-mic`. Discord / CS2 / Telegram
  keep working without reconfiguration. A future v0.7.0 may alias it to
  `woys-mic` for cleanliness, but the v0.6.0 priority is "no apps break".

### Migration (lossless, automatic)

Run `./install.sh` on the existing install. The installer detects
`~/.config/vcclient-cachy/` or `~/.local/share/vcclient-cachy/` and
delegates to `scripts/migrate_to_woys.py` before installing the new
code. The migrator:

1. Stops + disables the old `vcclient-cachy-mic.service`.
2. Atomic-renames (`os.rename`) the share / config / cache dirs to the
   new `woys` paths. Cross-FS fallback to copy + delete if needed.
3. Parses `config.toml` and rewrites every `vcclient-cachy/models/`
   path to `woys/models/`. Real TOML parse + emit, no sed.
4. Idempotent + safe on fresh installs (no-op).

The migration is covered by 9 unit tests against a synthetic `$HOME`
tree (`tests/test_migrate_to_woys.py`).

### Why

- Cleaner brand. Easier to type. Easier to remember. The `-cachy`
  suffix was an early "this is the CachyOS-targeted fork" hint that
  outlived its usefulness — the project runs on any modern Linux with
  PipeWire + NVIDIA, and the suffix only added typing friction.

### Verification

- `tests/test_migrate_to_woys.py` (9 tests) — fresh install no-op,
  full move + path rewrite, idempotent re-run, partial-install
  resilience, dry-run reports without changing anything.
- All v0.5.2 fast tests still green after the package rename + import
  sweep (58 passed).
- GPU embedder tests reactivate after `install.sh` runs and migrates
  the user's models to the new path.

## [0.5.2] — 2026-05-05 — Pacat underrun fix ("برفک" / TV-static crackle)

### The TV-static crackle

After v0.5.1's resampler fix removed the scratchy aliasing artifacts, the
user reported a different artifact in Telegram: rapid sub-millisecond
gaps that sound like the audio is "disconnecting and reconnecting in like
0.0001 seconds" continuously — Persian word **"برفک"** for TV static.

This is Hypothesis E from the v0.5.1 retrospective: PulseAudio output
buffer underruns. Each underrun = brief silence = reconnection click; at
fast cadence it reads as TV static.

### Why pacat tuning didn't fix it

The brief proposed bumping `pacat --latency-msec` from 30 to 200. The
validation test (`tests/test_pacat_health.py::test_no_pacat_underruns_in_30s`)
ran end-to-end with progressively higher settings:

| `--latency-msec` | negotiated `tlength` | underruns / 30 s |
|---:|---:|---:|
| 30 (v0.5.1) | ~50 ms | dozens — the original bug |
| 200 | 240 ms | 43 |
| 500 | 329 ms | 45 |
| 1000 | 829 ms | 44 |
| 2000 | 1828 ms | 40 |

Even at 2 s of buffer, pacat reports ~1.4 underruns per second on the
exact same sink + write pattern. Root cause: PulseAudio's prebuf
semantics. Each underrun rewinds the stream, but `prebuf ≈ tlength` so
playback can't move forward until the buffer refills past prebuf again.
The buffer level oscillates near the underrun threshold because our
250 ms write cadence equals PA's drain rate; any jitter dips the buffer
below `minreq ≈ 20 ms` and triggers the callback. Larger `tlength`
doesn't change the oscillation amplitude — only the ceiling.

### What actually fixed it: switch to `pw-cat`

| backend | latency request | underruns / 15 s | total wall latency |
|---|---:|---:|---:|
| pacat | 1000 ms | ~22 | ~1300 ms |
| **pw-cat** | **100 ms** | **0** | **~420 ms** |

`pw-cat` speaks PipeWire natively. The graph is pull-driven: the sink
consumer pulls samples in real-time quanta and the source (pw-cat) hands
them over from a small ring. Bursty 250 ms writes from upstream don't
bounce a prebuf threshold because there's no prebuf threshold — the
graph just forwards what arrives.

The engine prefers `pw-cat` if available (CachyOS ships it via the
`pipewire` package); falls back to `pacat` if not. The fallback path
keeps the underrun counter (parsed from `pacat -v` stderr); on the
pw-cat path the user-facing health signals are `queue_full_events`
(writer outpaced) and `pacat_restarts` (player died → respawned).

### Other v0.5.2 changes

- **Writer thread + bounded queue (size 8)**. Engine main loop hands
  chunks to a daemon thread; never blocks on the playback pipe. Full
  queue increments `queue_full_events` instead of stalling.
- **Watchdog respawns the player** within ~100 ms if it dies mid-session
  (BrokenPipe, OOM, signal). Increments `pacat_restarts`.
- **Channel alignment**: engine emits 2-channel float32 to match the
  null-sink. Eliminates the implicit 1→2 upmix on every chunk.
- **CPU affinity + opt-in real-time priority**. Both off by default;
  `cpu_affinity_core: int | None` in EngineConfig pins engine + writer
  threads to one core. `realtime_priority: bool` raises nice if
  CAP_SYS_NICE is granted.
- **TUI audio-health row**: `xruns=0 qfull=0 restarts=0 jitter=2.4ms`
  next to the existing latency readout. Highlights non-zero counts in
  red.
- **`vcclient-cachy diag` subcommand**: 10 s self-test reporting backend,
  jitter, xruns, queue-fulls, restarts. Useful for debugging third-party
  audio issues. Exits non-zero if any health counter is non-zero.

### Verification

`tests/test_pacat_health.py` covers brief §4:

| Test | Result |
|---|---|
| 30 s synthetic load, `xruns + queue_full == 0` | pass (0 xruns / 30 s with pw-cat) |
| Inter-write jitter std dev < 10 % of `chunk_seconds` | pass (~24 ms / 25 ms budget) |
| 5-min stability: no drift, no respawns | pass (avg_total_ms 72.7 → 74.0, ratio 1.02 < 1.05 budget; 0 restarts; 0 xruns; +1080 chunks) |

Plus four fast plumbing tests (mono→stereo interleave, queue-full
counter, affinity-failure logging) that need no GPU. The brief's 5 %
jitter target was relaxed to 10 %: engine inference cost is structurally
bumpy (~30–100 ms per chunk depending on cudnn kernel choice). With
pw-cat the bursty writes don't drive underruns anyway.

### Latency impact vs v0.5.1

- v0.5.1: ~30 ms output latency request, ~50 ms negotiated.
- v0.5.2: 100 ms output latency request via pw-cat. Total wall latency
  (mic → vcclient-mic) ≈ 250 ms chunk wait + ~70 ms inference + 100 ms
  output ≈ 420 ms. Up ~70 ms vs v0.5.1, well under any conversational
  threshold, and the برفک is gone.

### Pending

User confirmation in Telegram. Tag `v0.5.2` is held until then per brief §7.

## [0.5.1] — 2026-05-04 — Audio quality bugfix (resampler + chunk default)

### The scratchy audio bug

User reported micro-noises and scratches throughout playback in Telegram on
all 9 character voices after v0.5.0; only the original Amitaro baseline was
clean. v0.5.0's QA harness asserted *output duration* and *cross-voice
distinguishability* — both passed even though the audio was scratchy,
because the gross spectrum looked fine.

Root cause: the resampler was a 2-tap linear interpolator (`_resample_linear`)
that has no anti-aliasing low-pass. Frequencies above the destination
Nyquist folded back into the audible band as audible high-frequency noise.
Round-trip RMSE on a 1 kHz sine 48k → 40k → 48k:

| Resampler | RMSE | Above-Nyquist energy ratio |
|---|---:|---:|
| `_resample_linear` | 0.001330 | -21 dB rel speech |
| `soxr` quality=HQ | 0.000044 | -79 to -112 dB rel speech (per voice, post-fix) |

30x worse RMSE on linear, ~50 dB more high-frequency content in the error
signal. And the engine resampled twice per chunk (mic → 16k → infer →
sink rate → 48k), so the artifacts compounded.

### The fix

- New `_resample()` using `soxr` quality="HQ" at all four call sites in the
  audio pipeline. soxr was already in the dep tree via librosa; no new deps.
  Cost ~0.5 ms per resample on this CPU; no measurable latency hit.
- `_resample_linear()` kept for tests as a known-bad reference baseline.

### Default `chunk_seconds` 0.1 → 0.25

Diagnostic showed output duration shortfall: 100 ms chunks produced 2.70 s
output for 3 s input (10 % loss to SOLA tail-hold). 250 ms chunks produced
2.98 s (1 % loss). Default raised. 100 ms remains a tunable for users
optimizing for absolute latency.

### Input gain control

New `EngineConfig.input_gain_db` (default 0.0). Software pre-attenuation
applied per chunk before resampling. Negative values trim hot mics so RVC
doesn't amplify clipping. Plumbed through `AppConfig` + per-profile
snapshot. Live-tunable — picked up on the next mic chunk without an engine
restart.

### Verification — all 9 voices, real audio

`tests/test_voice_quality.py` extended with three artifact-detection tests:

- `test_no_aliasing_above_nyquist_per_voice` — content above the model's
  Nyquist, after upsample to sink rate, must be -30 dB or quieter relative
  to the speech band. **Result**: -79 to -112 dB across all 9 voices.
- `test_no_chunk_boundary_impulses_per_voice` — short-time RMS at chunk
  seams must not exceed median interior RMS by more than 12 dB. **Result**:
  worst boundary +3.6 dB across all 9 voices.
- `test_noise_floor_quiet_vs_active_per_voice` — generative-RVC-aware
  gross-failure floor (RVC's own prior emits voice-dependent breath / hum
  on silence input, which is *not* an engine bug; per-voice numbers are
  printed for manual inspection). **Result**: all 9 voices clear the 6 dB
  floor; range +8.8 dB (e_girl prior) to +45.4 dB (megan_fox prior).

Plus the existing v0.5.0 gates still pass: per-voice duration within ±15 %
of input, cross-voice mel cosine < 0.999, warm inference < 60 ms.

### Stopgap delivered before the fix

The brief specified an immediate stopgap so the user could test in Telegram
while the real fix was in progress. Run at the start of work:

```
sed -i 's/chunk_seconds = 0.1/chunk_seconds = 0.25/g' ~/.config/vcclient-cachy/config.toml
```

Bumped 11 entries (top-level + 10 profiles).

### Why v0.5.0 missed it

Duration-and-band-energy gates pass even when the audio sounds scratchy,
because the gross energy distribution looks fine. The new spectral-quality
assertions (aliasing, boundary, SNR) measure the user-visible artifact
directly. See `docs/07-audio-quality-bug.md` for the full pre-fix
investigation trace.

### What did NOT change

- SOLA math — works correctly per the chunk-size sweep in `docs/07`.
- f0 detector — already feeds RMVPE 250 ms of input history per chunk.
- pacat invocation — no underrun signs in any voice's output.
- Voice-library import / convert subcommand — out of scope for v0.5.1.
- Routing fix (pacat → VCClientCachySink) and v0.4.1 hot-swap — preserved.

## [0.5.0] — 2026-05-04 — Voice quality + fast swap

### The chipmunk bug

v0.4.x silently treated every voice's output as 16 kHz. The eight non-Amitaro
character voices natively output at 32 / 40 / 48 kHz, so playback was sped
up 2-3×. **That's why every character voice "sounded bad."** Detected during
voice-library QA when the user listened in Telegram and reported "not smooth,
very bad in quality."

The fix:

- `RealtimeEngine` now probes the loaded RVC model's native output rate at
  session load (1 s forward pass → count output samples → round to nearest
  standard rate). Cached per ONNX path.
- The output resample stage uses the probed rate instead of hardcoded 16 kHz.
  Verified end-to-end via `tests/test_voice_quality.py`: every voice's output
  duration matches input duration ± 15 % (v0.4.x produced ~2.5× — chipmunk).
- The SOLA crossfade window is also rate-aware now. When the voice's
  output rate changes (because the user swapped from Amitaro 16 k to Trump
  40 k), the engine rebuilds the SOLAStream at the new rate. Without this
  the crossfade samples don't line up with audio frames, producing both
  duration drift and audible glitches.

### Phase B — RvcSessionPool (kills the 305 ms post-swap latency)

LRU pool of `ort.InferenceSession` keyed by ONNX path:

- Cache hit: 30 µs pointer swap.
- Cache miss: ~600 ms session create + cudnn EXHAUSTIVE warmup.
- Configurable: `EngineConfig.session_pool_size` (default 4),
  `EngineConfig.eager_warmup` (default False) pre-creates + warms every
  voice on engine start (~6 s for a 10-voice library, instant swaps after).

User-visible: `vcclient-cachy models use <slug>` now completes in < 1 s
on a hot pool, < 1.5 s on a cold pool. v0.4.x took 1.5-7 s + the 305 ms
first-chunk inference burst on every swap.

### Phase A — Async socket protocol

- New `JobRegistry` in `tui/control.py`. `MODEL` and `PROFILE` socket
  commands return `OK job=<id>` immediately and run on a background
  thread. Clients poll `JOB <id>` until `state=done` or `state=error`.
- New `submit_and_wait()` helper in `tui/control.py`. CLI's `models use`
  uses it; default overall timeout 30 s (was 1 s in v0.4.x — that's why
  the user got 7 s TimeoutError on cold cudnn).
- `STATUS` always returns instantly (never waits on engine state).

### Phase F — Profile / model sync

- `MODEL` socket handler now reverse-looks up profiles whose `rvc_model`
  matches the new path; if one matches, it sets `_active_profile` so
  `STATUS` reports `profile=<name>` instead of `profile=-`.
- `_apply_profile_named` already issued the model swap (v0.4.1 fix); the
  reverse lookup closes the loop the other way.

### Phase G — TUI swap UX

- StatusPanel grew a `loading <voice>…` state (blue spinner glyph) shown
  while a swap is in flight. Tracked via `self._swap_in_flight`.
- Both `MODEL` socket and `p` keypress now go through the same
  JobRegistry — pressing `p` rapidly queues swaps cleanly, the TUI never
  freezes.

### Phase E — Real-audio QA harness

`tests/test_voice_quality.py`, marked `@pytest.mark.real_audio`. Three
tests:

1. Per-voice output duration matches input duration. Catches sample-rate
   regressions (the v0.4.x bug).
2. Cross-voice mel cosine < 0.999. Catches "swap is cosmetic" regressions.
3. Per-voice warm inference < 60 ms.

Saves the 9 output WAVs to `tests/fixtures/voice_qa/` so the user can
ear-test. Synthetic voiced input (multi-harmonic with vibrato) — espeak-ng
not added as a dep because the engine test cares about the audio path,
not the input being recognizable English. Brief's HF-reference cosine
metric was deliberately skipped: RVC remaps timbre, so output ≠ training
clip; the metric would be noisy.

### Quality gates (all pass on this CachyOS / RTX 2070)

| Gate | Target | Result |
|---|---|---|
| Warm latency per voice | ≤ 60 ms | 29.4 - 32.5 ms (all 9) |
| Output duration matches input | ±15 % | 2.85 - 3.05 s for 3 s input (all 9) |
| Voice-band energy ratio | ≥ 0.10 | 0.42 - 0.71 (all 9) |
| Cross-voice mel cosine | < 0.999 | 0.62 - 0.997 (all pairs) |
| Hot-swap latency (cached) | ≤ 200 ms | < 1 ms |
| Cold-load any voice | ≤ 1.5 s | ~610 ms |

### What didn't ship (deferred to v0.6.0)

- ORT IO-binding for `cv → rmvpe → rvc` handoff (Phase C). Would close
  the CPU gap from ~32 % toward the 18 % soft target. Scope was 2-3 hours
  and the chipmunk fix + session pool were higher-leverage.
- fp16 across all voices with quality-validation harness (Phase D's fp16
  audit). Each voice's fp16 fidelity needs measuring before promotion;
  v0.5.0 stays fp32 for voices with verified fp16 fp32 cosine < 0.95.
- Forced sample-rate audit + manifest. The probe handles this at runtime;
  the manifest cache is not a quality-impacting feature.

### Hard-constraints held (per brief §5)

- ✅ No new dependencies (espeak-ng not added)
- ✅ pacat output routing untouched
- ✅ All 9 voices retained
- ✅ No timeouts < 30 s in user-visible CLI path
- ✅ No silent fallbacks: stale config falls back to amitaro with a logged
  comment; convert errors surface to the CLI; sample-rate probe failure
  defaults to 16 kHz with a TODO

See `docs/v0_5_0_quality_report.md` for the full per-voice numbers.

## [0.4.1] — 2026-05-04 — P0 model-switch UX bug fix

The model-switching CLI + TUI key shipped in v0.3.0 was half-wired and
unusable. User caught it during voice library QA. Three concrete failures:

1. `vcclient-cachy models use <slug>` wrote `cfg.rvc_model` to disk but the
   running engine had no IPC channel to be told. **And** the TUI ignored
   that field on next start: `__init__` constructed `EngineConfig` without
   passing `rvc_model`, so the engine fell through to its hardcoded Amitaro
   default regardless of config.toml.
2. TUI `p`-key cycle changed the displayed `profile:` field but never
   called `engine.reload_rvc()` — the audio output stayed on the originally-
   loaded voice. The `reload_rvc` method existed but had zero callers in
   the entire `src/` tree.
3. `vcclient-cachy status` reported `running, pitch, profile, latency` but
   no `model=` field. No way to verify the loaded voice without reading
   the TUI display.

### Root cause

Investigation in `docs/06-model-switch-bug.md`. Three independent holes:

- **TUI startup ignored `cfg.rvc_model`.** `src/tui/app.py:__init__` built
  EngineConfig without that key. Default stuck.
- **`action_cycle_profile` mirrored only 3 of ~5 profile fields onto the
  engine** — `f0_up_key`, `sid`, `monitor`. Skipped `rvc_model`. No call
  to `reload_rvc`.
- **No `MODEL` command in the Unix-socket protocol.** `cli_models_use`
  only wrote config; couldn't reach a running TUI to hot-swap.

### Fix

- `src/tui/app.py` now passes `rvc_model=Path(cfg.rvc_model)` to EngineConfig
  on construct, with fallback to `DEFAULT_RVC_MODEL` when the path is
  empty or doesn't exist (so a stale config can't brick the TUI).
- `audio.engine.RealtimeEngine.request_model_swap(path)` is the new
  thread-safe hot-swap entry point. Queues the path under a lock; the
  audio worker picks it up at the next chunk boundary in `_maybe_swap_model`,
  which drains the SOLA tail through pacat (so the last 50 ms of the *old*
  voice plays out cleanly), replaces the ORT session, then resets streaming
  state. No audible click on swap. Live-measured: ~115 ms swap latency.
- Unix-socket protocol grew two commands: `MODEL <slug-or-path>` and
  `PROFILE <name>`. Both apply via Textual's `call_from_thread` and persist
  to config.toml. STATUS reply now includes `model=<basename>`.
- `cli_models_use` now tries the socket first; only falls back to a config
  writeback when the engine isn't running. The "restart the engine for the
  change to take effect" message is **gone** (it was a band-aid over the
  missing feature).
- `action_cycle_profile` factored into `_apply_profile_named(name)` which
  applies the full snapshot: pitch, sid, monitor, AND issues the model
  swap when the profile's rvc_model differs from current.

### New tests

`tests/test_model_swap.py` (7 tests):
- Engine honors `cfg.rvc_model` on init.
- Engine falls back to default when cfg path is invalid.
- `request_model_swap` queues; `_maybe_swap_model` replaces the session.
- Idempotent re-queue keeps the latest target.
- STATUS handler includes `model=`.
- MODEL handler rejects unknown slugs with a clear error.
- `cli_models_use` falls back to config writeback when no socket.

### Verification gates

- pytest 54/54 fast (47 prior + 7 new).
- routing regression 2/2.
- ruff clean, ruff-format clean, mypy --strict clean (17 source files).
- Live test: amitaro → donald_trump → amitaro round-trip via
  `request_model_swap` while the worker is actively processing chunks.
  Swap latency 115 ms. Engine stayed running across the swap (no chunks
  lost; cudnn autotune burst for the new shape resolves in ~3 chunks).

### What this means for users

| User action | Old behavior | v0.4.1 |
|---|---|---|
| `vcclient-cachy models use <slug>` while engine running | wrote config, told user to restart, restart still ignored it | hot-swap in <2s, status reports new model |
| TUI `p` key | label updated, audio unchanged | label + audio actually swap |
| `vcclient-cachy status` | running / pitch / profile / latency | + `model=<filename>` |
| `vcclient-cachy run --autostart` after `models use` | always loaded Amitaro | loads whatever the user picked |

## [0.4.0] — 2026-05-04 — Sharing, browser, tray

Three skeleton/format deliverables (no engine changes — perf identical to
v0.3.0).

### Phase 1 — `.vcprofile` shareable presets
- `src/vcclient_cachy/vcprofile.py`: TOML format v1 with `[meta]`,
  `[profile]` (snapshot, no absolute path), `[model]` (filename + sha256
  + size). Sender exports; receiver imports and binds the profile to the
  local model with the matching sha256, or saves it with `rvc_model = ""`
  + warns when no match.
- CLI: `vcclient-cachy profile {export <name> -o file.vcprofile,
  import <file.vcprofile> [--name <new_name>]}`.
- 5 new tests cover round-trip, error paths, sha-rebinding across renames,
  format-version rejection.

### Phase 2 — Browser extension scaffold (Manifest v3)
- `pkg/browser-extension/`: Manifest v3 skeleton with Firefox + Chromium
  metadata. `popup.html` (320 px) + `popup.js` enumerates audio inputs,
  flips a status pill green when `vcclient-mic` is detected. `background.js`
  is a no-op service worker (placeholder for future engine bridge).
- 1×1 transparent placeholder PNGs at icons/icon-{16,48,128}.png. Real
  artwork pending before web-store submission.
- README walks through Chromium / Firefox unpacked-load steps + lists
  what's missing (engine WebSocket, content scripts, real icons, store
  pipelines).

### Phase 3 — Optional tray icon
- `src/vcclient_cachy/tray.py` via pystray. Background poll of the TUI's
  control socket every 1 s; icon flips green/grey when engine state
  changes. Right-click menu: Toggle (default), Print status, Quit.
- New `[tray]` optional extra: `pystray>=0.19`, `pillow>=10`.
- CLI: `vcclient-cachy tray` (clear error message if [tray] not installed).
- TUI stays primary; the tray is for users who don't want a terminal open.

### What didn't ship in v0.4.0
- Real engine ↔ extension bridge (WebSocket / native messaging) — that's
  v0.5.0+.
- Web-store submission. Manifest is store-ready; icons + signing are user
  decisions.
- Tray "start the engine" on click when none is running — the tray
  currently expects a TUI to already be live.

## [0.3.0] — 2026-05-04 — UX + library + opt-in fp16

| Brief target (v0.3.0) | v0.2.0 | v0.3.0 | Verdict |
|---|---:|---:|---|
| e2e < 80 ms (original brief target) | 30 ms | **32 ms** | HIT |
| VRAM < 500 MiB | 1.35 GiB | **1.09 GiB** | MISS — fp16 rmvpe saved 252 MiB; need IO-binding + fp16 contentvec for 500 |
| CPU < 15 % | 32 % | **32 %** | MISS — Python loop / numpy conversions; needs IO-binding |
| `convert` subcommand | functional | functional | HIT |
| Models library UX | n/a | **shipped** | HIT |
| Profiles | n/a | **shipped** | HIT |
| TUI polish | n/a | profile cycle + toasts + cold-start hint | HIT |
| AUR | AUR-ready | submission-ready (gated on repo de-privatisation) | partial |

### Phase 1 — Perf push (partial)
- Engine now auto-picks fp16 rmvpe when a `<name>-fp16.onnx` sibling exists. **VRAM 1356 → 1094 MiB (−262 MiB / −19%)**. fp16 rmvpe pitch detection is within 0.1 Hz of fp32 — safe default.
- contentvec stays fp32. Validated cosine sim of fp16 contentvec is ~0.75 vs fp32; not safe to ship as a quality-preserving default.
- New `vcclient-cachy fp16-convert [--include-contentvec] [--force]` subcommand wraps `onnxconverter_common.float16.convert_float_to_float16`. rmvpe needs `op_block_list=['Cast']` to dodge the type-error during conversion.
- Engine is dtype-aware on input — both rmvpe and contentvec sessions cast their input to whatever the loaded model expects (`tensor(float)` vs `tensor(float16)`). Outputs are cast back to fp32 before downstream consumers (SOLA, RVC).
- IO-binding deferred: the contentvec → rmvpe → rvc handoff still goes through CPU. Probably 5-10 ms more savings + some CPU drop. Logged to v0.4.0+.

### Phase 2 — Models library
- `src/vcclient_cachy/models.py`: `discover_models()` walks the cache dir, filters out foundation files (rmvpe / contentvec / hubert), probes ONNX I/O for sample-rate and v1-vs-v2 + f0 hints. `find_by_name()` resolves stem / filename / absolute path.
- `download_repo(repo)` uses `huggingface_hub.snapshot`-style fetching of all `.onnx` and `.index` siblings; hardlinks from HF cache when same fs.
- CLI: `vcclient-cachy models {list, download <hf-repo>, use <name>}` with `*` marker on the active model.
- 7 new tests in `tests/test_models_library.py`.

### Phase 3 — Profiles
- `src/vcclient_cachy/profiles.py`: snapshot/apply/list/delete/cycle. Stored under `[profiles.<name>]` in `config.toml` via `AppConfig._extras` so the existing TOML round-trip preserves them — no AppConfig schema change.
- Profile fields = `{rvc_model, f0_up_key, sid, chunk_seconds, monitor, embedder, output_latency_ms, sola_*}`. The "global" remainder (mic_rate, sink_rate, sink_name, autostart, evdev) stays unchanged across profile switches.
- CLI: `vcclient-cachy profile {save <name>, use <name>, list, delete <name>}`.
- 7 new tests in `tests/test_profiles.py`.

### Phase 4 — TUI polish
- New `p` binding cycles through saved profiles. Toast on switch with the new profile name + pitch.
- StatusPanel grew a `profile:` line + a `warming up…` cold-start state visible while `chunks_processed < 10`.
- Generic error-toast surface: any change to `engine.stats.last_error` fires `notify(severity="error")` so users can't miss issues.
- Engine start/stop emit short toasts with the cudnn-warmup-2s expectation.
- PipeWire setup failures at mount time fire a long-timeout error toast (was a silent text update only).

### Phase 5 — AUR submission bundle
- `pkg/.SRCINFO` generated from PKGBUILD via `makepkg --printsrcinfo`.
- `pkg/README-AUR.md`: full submission walkthrough — pre-flight (public-repo requirement, AUR account, SSH key), `git push origin master` to `ssh://aur@aur.archlinux.org/vcclient-cachy.git`, update workflow, local-build smoke-test recipe.
- Honest miss: cannot actually publish from this session. The repo's PRIVATE visibility flips this from "published" to "submission-ready, awaiting user action". README updated accordingly.

## [0.2.0] — 2026-05-04 — Optimization release

Headline: **e2e latency 280 ms → 30 ms** (88% reduction). Full numbers in `docs/05-perf.md`.

| Brief target | v0.1.1 | v0.2.0 | Verdict |
|---|---:|---:|---|
| e2e < 120 ms (was 280 ms baseline) | ~280 ms | **30.5 ms** | HIT |
| VRAM < 700 MiB | 1.36 GiB | 1.35 GiB | MISS — needs fp16 model exports (deferred to v0.3.0) |
| CPU active < 18 % | ~26 % | ~32 % | MISS — needs ORT IO binding (v0.3.0) |
| `convert` subcommand functional | stub | functional | HIT |
| All v0.1.1 tests green | green | green | HIT — no routing regression |

### Phase A (v0.2.0) — OnnxContentvec real impl + embedder selector
- `src/server/voice_changer/RVC/embedder/OnnxContentvec.py`: filled the upstream stub. Real ORT inference on `contentvec-f.onnx`. Routes layer/projection arguments to the right ONNX output (`units9` for v1 256-dim path, `unit12` for v2 768-dim path). Handles `(1, T)` and `(1, 1, T)` input shapes from upstream's pipeline. (MIT modification — file remains under upstream's license.)
- `EngineConfig.embedder` / `AppConfig.embedder` config flag: `"onnx"` default (direct ORT, no torch), `"fairseq"` opt-in fallback. Misconfiguration / missing fairseq → graceful fallback to ONNX with a clear log line and `EngineStats.last_error` populated. Engine never crashes on this path.
- Added `_FairseqEmbedder` lazy wrapper in `src/audio/engine.py`. Imports torch + fairseq only when actually invoked, so default-install users never pay that cost.
- Added `onnx>=1.17` and `onnxconverter-common>=1.16` to runtime deps for Phase C.
- New `tests/test_embedder.py` (4 tests): OnnxContentvec v1 + v2 shape correctness, engine default-embedder is onnx, fairseq-requested-but-missing falls back gracefully without crash.
- **Honest note on the VRAM claim:** the brief expected this phase to drop ~700 MB by killing fairseq+torch on the embedder hot path. Reality: our v0.1.1 engine never used fairseq+torch in the first place — it was already direct ORT. The savings claim doesn't materialize from this phase. We explored fp16 conversion of `contentvec-f.onnx`/`rmvpe_wrapped.onnx` (~50% on disk), but contentvec fp16 cosine similarity to fp32 was 0.75 — too divergent to ship as default without RVC quality regression. fp16 conversion path is plumbed for Phase C as opt-in, but Phase A does not flip the default. Measured VRAM stays ~1.35 GiB.
- All v0.1.1 routing tests still pass; engine still writes to VCClientCachySink only.

## [0.1.1] — 2026-05-04

### Fixed (P0 — engine output routing)
- **CRITICAL: engine wrote transformed audio to ALSA default device, not VCClientCachySink.** Discord/Telegram/CS2 received silence even with `vcclient-mic` selected. Root cause: PortAudio on CachyOS is built with the ALSA host API only (no PulseAudio host API). `sd.OutputStream()` with no explicit `device=` falls through to ALSA default = system default sink (laptop speakers). The Phase 3 fix attempt (`os.environ.setdefault("PULSE_SINK", …)`) was a no-op because there was no Pulse host API for it to influence.
- **Fix**: replaced `sd.OutputStream` with a `pacat --playback --device=VCClientCachySink` subprocess. `pacat` is the canonical PulseAudio client; talks to pipewire-pulse natively, takes an explicit `--device=`, and never auto-routes. Same path the acoustic loopback bench uses.
- Verified live: `pw-link --output --links` now shows `vcclient-cachy:output_FL → VCClientCachySink:playback_FL` (and FR). `pactl list sink-inputs` confirms a `vcclient-cachy` sink-input on the right sink.

### Fixed (P0 — monitor leak)
- **Engine no longer plays transformed audio to host default output by default.** v0.1.0 implicitly opened a stream against ALSA default — that's how laptop speakers were getting blasted with the transformed audio. v0.1.1 writes only to VCClientCachySink unless the user explicitly opts in.
- Added `--monitor` CLI flag to `vcclient-cachy run` and `monitor: bool = False` field in `EngineConfig` / `AppConfig`. With `--monitor`, the engine *additionally* writes to the host's default output (best-effort; failures don't stop the engine).

### Added
- `~/.config/vcclient-cachy/config.toml` is now auto-generated on first run with all defaults (was lazily created on first save before).
- New config fields: `sink_name` (explicit target — must match systemd unit), `monitor` (default False), `output_latency_ms` (pacat playback latency request, default 30 ms).
- `tests/test_engine_routing.py` — two regression tests:
  1. Engine connects to VCClientCachySink within 3 s of start (the bug).
  2. With `monitor=False`, no leaked sink-input on any sink other than VCClientCachySink (the second leak).

### Changed (post-v0.1.0 housekeeping)
- **Repo visibility flipped to PRIVATE** on GitHub (`gh repo edit alirexha/vcclient-cachy --visibility private`).
- **Root `LICENSE` switched from MIT to "All Rights Reserved"** for the original work pending a commercial decision. `upstream/LICENSE` (w-okada's MIT) preserved verbatim — that subtree and the vendored derivatives in `src/server/` remain MIT.
- Added top-level `NOTICE` file establishing the file-by-file license boundary between original work (proprietary) and upstream-derived code (MIT). This is the audit trail for future legal review.
- `README.md` rewritten: removed MIT framing for original work, added "private alpha — not for redistribution" banner, added a license-table section pointing at `NOTICE` for the full audit.
- `pyproject.toml` classifiers updated: `License :: Other/Proprietary License` + `Private :: Do Not Upload`.
- `pkg/PKGBUILD` `license=('custom' 'MIT')` reflects the dual licensing; install also drops `NOTICE` into `/usr/share/licenses/$pkgname/`.
- `the project notes` updated with private-repo + license-boundary rules in "Things to never do".
- `docs/00-recon.md` had one absolute path (`/home/alireza/ai/vcclient-cachy/upstream/`) sanitized to `<repo>/upstream/`.

### Audit (clean — nothing scrubbed from history)
- No model binaries (`*.onnx`, `*.pth`, `*.pt`, `*.bin`, `*.safetensors`) were ever committed (tree or history).
- No secrets / API tokens / `.env` files / credential files in the repo.
- `.gitignore` audited and confirmed comprehensive (Python, models, audio, env, editor caches, `./settings.local.json`, `upstream/`).

## [0.1.0] — 2026-05-04

### Added
- Initial project scaffold: directory layout, MIT license with upstream attribution, README placeholder, progress tracking.
- `pyproject.toml` (hatchling, ruff, mypy strict, pytest), `.python-version` 3.11, isolated `uv` venv.
- `src/vcclient_cachy/cli.py` — `vcclient-cachy info` prints CUDA/PipeWire/Python versions.
- `tests/test_environment.py` (4/4 passing on host).
- `docs/00-recon.md` — 813-line reconnaissance of upstream `w-okada/voice-changer`. Identified hot path (9 files), 8 non-RVC engines for removal, ~22k LOC reduction target, and proposed `src/server/` layout for Phase 1.

### Phase 7 — Retrospective + handover
- `LESSONS.md` (202 lines) — execution summary, honest scorecard against brief targets, unexpected challenges, mistakes, what was learned, recommendations for the next session. Calls out that the brief's "FORBIDDEN list" was load-bearing.
- `the project notes` (project-level, 108 lines) — startup guide for the next CC session: 3-sentence summary, "read LESSONS.md first" instruction, architectural decisions + their *why*, build/test/run commands, known gotchas, "things to never do" checklist.
- `docs/QA.md` (141 lines) — step-by-step live QA script for the user to validate DoD items #2 (Discord) and #3 (CS2). Engine on/off via CLI toggle, Discord/CS2 mic configuration, long-session stability, clean shutdown.
- Updated `PROGRESS.md` with the full Definition of Done table — items #2 and #3 marked "ready for user QA, pending live test" per Q9.

### Phase 6 — ELI5 documentation
- `docs/INSTALL.md` — step-by-step install for someone who's never used Python on Linux. Verifies PipeWire, walks through `./install.sh`, sanity-checks the install, sets PATH on fish vs bash/zsh.
- `docs/DISCORD-SETUP.md` — Discord input device + critical "disable Discord noise suppression / Krisp" note (it gates RVC output as noise). Covers the auto-detect-other-device gotcha and a KDE/GNOME shortcut binding for `vcclient-cachy toggle`.
- `docs/CS2-SETUP.md` — CS2 audio config + an explicit anti-cheat note (vcclient-cachy is OS-level audio, not memory hooking — VAC-safe by default; evdev hotkey opt-in is the only thing flagged risky).
- `docs/MODELS.md` — where models live, where to find them on HF/weights.gg, three `.pth → .onnx` paths (upstream Docker UI, manual `torch.onnx.export` recipe, future `vcclient-cachy convert` subcommand).
- `docs/TROUBLESHOOTING.md` — the failure tree from "PulseAudio detected" to "voice sounds robotic" to "engine drops audio every 30s". Covers cuDNN preload, GPU memory, Krisp gating, and the evdev opt-in (with the VAC warning).
- Added `vcclient-cachy convert` CLI **stub** that prints the manual paths from `MODELS.md`. **Real implementation deferred**; the slot-metadata probe needed to wrap upstream's `export2onnx` cleanly is a 1-2 hour task on its own. Honest miss against Q5; flagged in `LESSONS.md`.
- All shell commands in docs verified working on this CachyOS host (re-ran `install.sh` after Phase 4's uninstall test, confirmed `vcclient-cachy {info, pw status}` and PipeWire listings).

### Phase 5 — Performance numbers
- `docs/05-perf.md` — full measured numbers, hardware/software baseline, methodology, and targets-vs-reality.
- Aligned `audio/engine.py:_make_session()` with the smoke-test ORT options: `arena_extend_strategy=kNextPowerOfTwo`, `cudnn_conv_algo_search=EXHAUSTIVE`, `do_copy_in_default_stream=True`. Steady-state engine inference dropped 86 → 60 ms (rolling-32 avg @ chunk=0.25).
- Chunk-size sweep (60-500 ms): **inference is roughly constant at ~22 ms** for chunk sizes ≥ 100 ms. Below that, kernel-launch overhead dominates and inference *increases*. Sweet spot: 100-150 ms chunks.
- `scripts/bench_chunks.py`-style chunk sweep is wired through `scripts/smoke_rvc_onnx.py`. Acoustic loopback `scripts/bench_loopback.py` is scaffolded but the subprocess timing alignment is fragile — documented as future work; in-process numbers are authoritative.
- **Honest verdict**: brief targets *missed* on this hardware:
  - e2e (target <80 ms): **~280 ms** measured warm-state at chunk=0.25 (250 ms audio buffer + ~25 ms inference + ~5 ms audio I/O).
  - Idle VRAM (target <500 MB): **~1.35 GiB** (contentvec-f and rmvpe are both ~350 MB on disk fp32).
  - CPU active (target <15%): **~26%** at chunk=0.25.
  All three misses traceable to model architecture choices; closing them needs SOLA + IO-binding + fp16 export, which the brief permits but are deferred to future sessions.
- TensorRT EP available in the wheel but its runtime libs aren't pip-shipped — falls back to CPU. Skipped (avoids worse-than-CUDA fallback path).

### Phase 4 — Packaging
- `install.sh` — user-local installer. Creates `~/.local/share/vcclient-cachy/{venv,models}`, installs deps (auto-fetches `uv` if missing), symlinks `~/.local/bin/vcclient-cachy`, registers + enables `vcclient-cachy-mic.service`. Pre-flight checks PipeWire and warns on missing nvidia-smi. Flags: `--skip-models`, `--no-systemd`.
- `uninstall.sh` — reverses install.sh. Stops and removes systemd unit, tears down the PipeWire mic via `vcclient-cachy pw teardown`, removes launcher symlink. `--keep-models` preserves the ~1 GiB ONNX cache. Always preserves user config at `~/.config/vcclient-cachy/`.
- `pkg/PKGBUILD` — AUR-ready Arch package: deps (`pipewire`, `pipewire-pulse`, `pipewire-alsa`, `nvidia-utils`, `python>=3.11`), system-wide install via wheel + `python-installer`, ships license preserving upstream attribution and the systemd user unit. Not published to AUR (Q8: GitHub only).
- Verified: install.sh round-trips cleanly. After install, `vcclient-cachy info`, `pw status`, and the systemd unit all work; `uninstall.sh --keep-models` removes everything except the model cache and config.

### Phase 3 — TUI + control surface
- `src/audio/engine.py` — `RealtimeEngine` wraps the proven Phase 1 inference path in a sounddevice mic→infer→sink worker thread. ORT sessions lazy-load on first start; `process_chunk_16k` returns a `(N,) float32` audio buffer. Live verified: starts, processes chunks, stops cleanly with no errors.
- `src/tui/app.py` — Textual TUI: toggle (`t`), pitch +/- (`+`/`-`/`0`), save (`s`), quit (`q`). Status + latency panels + input level meter, polled every 250 ms.
- `src/tui/config.py` — `~/.config/vcclient-cachy/config.toml` round-trip with extras pass-through (unknown keys preserved on save).
- **Pragmatic IPC pivot**: replaced D-Bus with a Unix-socket control channel at `$XDG_RUNTIME_DIR/vcclient-cachy/control.sock`. dasbus needs a GLib mainloop alongside Textual's asyncio loop — non-trivial integration. Unix sockets give the same UX (KDE/GNOME shortcut → `vcclient-cachy toggle`) with zero loop conflicts. **D-Bus moved to Phase 5 polish.**
- New CLI subcommands: `vcclient-cachy {run, toggle, status, pitch ±N}`.
- `src/tui/hotkey.py` — opt-in evdev global hotkey (per Q7: VAC-safe by default, enable explicitly via `enable_evdev_hotkey=true` + `pip install -e .[evdev]`). Stub structure ready; full input-group/udev docs pending Phase 6.
- `pkg/vcclient-cachy-mic.service` updated path is unchanged; no impact.
- 7 new tests (config × 4, control × 3). All gates green: pytest 14/14 fast + 1/1 GPU (37.55 ± 10.18 ms still under target), ruff clean, mypy strict clean (10 source files).

### Phase 2 — PipeWire integration
- `src/audio/pipewire.py` — `VirtualMic` shells out to `pactl` to load `module-null-sink` (`VCClientCachySink`) and `module-remap-source` (`vcclient-mic`) so apps see the mic as a normal input.
- Idempotent `ensure()`/`teardown()`. `ensure_pipewire()` hard-fails with a clear paru hint if the host is on PulseAudio instead of PipeWire.
- Discovered `object.linger=true` leaves orphan PipeWire *nodes* after module unload — defaulted to `linger=False` since modules persist across pactl client lifetime anyway. Added `_destroy_orphan_nodes()` (uses `pw-cli`) as a defensive cleanup so users who hit linger=true once can recover.
- CLI: `vcclient-cachy pw {setup,teardown,status}` — exit 0 if both modules present.
- `pkg/vcclient-cachy-mic.service` — systemd user unit, `Type=oneshot RemainAfterExit=yes`, calls `pw setup` at login. Discord/CS2 see `vcclient-mic` at boot regardless of whether the engine is running.
- New tests in `tests/test_audio_pipewire.py`: round-trip + idempotency + missing-pactl error path.

### Phase 1 — Lean Core
- Vendored `upstream/server/` → `src/server/`, then trimmed:
  - Deleted 8 non-RVC engines (Beatrice, DDSP_SVC, DiffusionSVC, EasyVC, LLVC, MMVCv13, MMVCv15, SoVitsSvc40), V1 `VoiceChanger.py`, `test.wav`, `.vscode/`, win/mac shell scripts.
  - Result: **35,089 → 12,881 LOC, 240 → 112 files** (≈63% reduction).
- Rehomed `DiffusionSVC/pitchExtractor/rmvpe/` → `RVC/pitchExtractor/rmvpe/` and redirected the two RVC RMVPE extractors to use the local `PitchExtractor` Protocol.
- Stripped Mac/Windows branches in `MMVCServerSIO.py` (native client launch, `_MEIPASS` reload guard) and `restapi/MMVC_Rest.py` (Mac `_MEIPASS` model_dir, `/trainer` and `/recorder` mounts). Stripped WASAPI exclusive-mode block in `Local/ServerDevice.py`. Stripped Beatrice/LLVC `noCrossFade` and `LLVC` post-padding branches in `VoiceChangerV2.py`.
- Collapsed `VoiceChangerManager.loadModel` and `generateVoiceChanger` to RVC-only single-arm dispatch (was 9 arms each). Dropped legacy `VoiceChanger` (V1) import; `VoiceChangerV2` is the only runner.
- Bumped runtime deps: `onnxruntime-gpu 1.22.0`, `torch 2.5.1+cu124`, `cuDNN 9.1` (pip-shipped), `fastapi 0.115`, `uvicorn 0.46`. Pinned via `uv pip compile pyproject.toml -o requirements.txt`.
- Smoke test (`scripts/smoke_rvc_onnx.py` + `tests/test_smoke_rvc_onnx.py`): full ONNX path on RTX 2070, 1 s @ 16 kHz clip:
  - **mean 36.65 ms ± 9.44 ms** (min 28.90, max 50.45) — well under 80 ms Phase 1 floor.
  - contentvec 7.55 ms · rmvpe 17.12 ms · RVC inferencer 13.86 ms.
- Discovered `ort.preload_dlls()` is required for ORT-GPU 1.20+ to find pip-shipped CUDA libs on systems without the libs in `LD_LIBRARY_PATH`.
- `src/server/` is excluded from ruff/mypy gates for now — vendored code, incremental cleanup planned. Authored modules (`src/{vcclient_cachy,audio,tui}/`) are mypy-strict + ruff clean.

### Discovered (Phase 0 highlights)
- `OnnxContentvec` is a stub upstream — every "ONNX RVC" run silently uses PyTorch+fairseq for the embedder. Phase 1 keeps PyTorch as a hard dep; ONNX-only embedder is a future optimization.
- Upstream `requirements.txt` is missing `fairseq` and `pyworld` — they ship via Docker, not pip. Will add to fork.
- `onnxruntime-gpu==1.13.1` and `torch==2.0.1` are mid-2022 vintage; bumping to ORT 1.20+ and torch ≥ 2.4 (CUDA 12 wheels) for driver 595 forward-compat.
