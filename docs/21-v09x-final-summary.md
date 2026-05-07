# woys v0.9.0 — final summary (Phase C deliverable)

Date: 2026-05-08
Status: shipped, tagged `v0.9.0`, pushed to origin/main.

This document is the Phase C deliverable from `V0_9_X_AUTONOMOUS.md`.
It records each of the three fixes' final state, the measured impact,
an honest assessment of likely audible-cut improvement, and open
questions for a future v0.10.x track.

---

## 1. The three fixes — final state

### Fix 1 (ORT IO binding): **DEFERRED — null on this stack**

Documented in `LESSONS.md §23`. Pre-flight bench (`scripts/bench_iobinding.py`,
200 passes × 2 chunk sizes) measured -1.6% / -0.8% Δavg vs baseline on
RTX 2070 Mobile + RVC v2_16k + ORT 1.22. The brief's expected "10-30%
inference win" was the v0.8.0 review's generic prediction; on this
specific hardware/model, the per-chunk H2D copies are too small
relative to inference compute to register. RVC v2_16k input is
~16 KB/chunk; copy cost is ~µs vs 21 ms p50 inference.

No code shipped. The bench file remains at `scripts/bench_iobinding.py`
for re-measurement when (a) the engine moves to a larger model, (b) ORT
or CUDA driver behavior changes, or (c) chunk_seconds drops to 0.10 and
copy fraction grows.

**Lesson generalized**: for any "perf-N% win" item on a fix list, run
the bench BEFORE committing to scope. Predicate-first, fix-second.

### Fix 2 (native PipeWire client): **SHIPPED as v0.9.0-rc1**

`bin/woys-pw-out.c` — a ~430 LOC C helper that registers a native
PipeWire stream client (Stream/Output/Audio, targeting WoysSink),
decouples the engine's bursty 150 ms chunk writes from PipeWire's
per-quantum (1024/48000 = 21.33 ms) RT callback via a lock-free SPSC
ring buffer.

Engine integration:
- `EngineConfig.prefer_native_pw: bool = False` — opt-in for v0.9.0.
  Per `docs/19-pw-investigation.md` §7 follow-up 7, default flips to
  True in v0.9.1 after one release of soak; v0.10 deletes the legacy
  pacat / pw-cat paths entirely.
- `_open_pacat` dispatches to the helper when the flag is set; hard-
  fails with an actionable error if the helper binary is missing
  (no silent fallback per the brief's load-bearing rule).
- `_stderr_reader_loop` parses the helper's `underruns=N` line into
  `EngineStats.player_underruns`. Closes audit lens 09 rank 1 ("pw-cat
  is silent on underruns") for free.
- `woys diag` displays the new `native-pw under.` counter.

**Bonus catch**: integration testing surfaced the same EngineConfig
drift class the rc4 audit fingered (the new field landed on
EngineConfig + AppConfig defaults, but the explicit `EngineConfig(...)`
constructor calls in cli.py / app.py didn't forward it). Fixed both
construction sites; new AST-walk test asserts every site forwards
every USER_VISIBLE field. As a side effect, `woys diag` now respects
user config for voice-shape fields (f0_up_key, sid, monitor, sola_*)
instead of silently ignoring them.

Verification at ship:
- 120 fast tests pass (was 118; added two parametrized AST tests).
- Helper builds clean (gcc + libpipewire-0.3 dev headers).
- Live engine smoke (6s, prefer_native_pw=true): xruns=0, queue_full=0,
  dropped=0, clean shutdown — vs 11-12 xruns / chunk in pacat mode on
  the same workload.

### Fix 3 (mitigations=off documentation): **SHIPPED as v0.9.0-rc2**

`docs/20-mitigations-tuning.md` — single doc, no code change. Walks
through CachyOS systemd-boot-specific apply procedure, security
tradeoff, revert, and a §5 measurement template using
`woys diag --duration 60` + `pw-record` capture for before/after
comparison. §7 includes a combination table covering the three
independent levers (mitigations off, linux-rt, native-pw client) with
an "apply in sequence, measure after each" recommendation.

Doc explicitly explains why woys does NOT modify boot params for the
user — sudo across user-home boundary, reboot required, security
tradeoff is the user's call.

---

## 2. Measured impact per fix

| Fix | Metric | Before | After | Verdict |
|---|---|---|---|---|
| 1 IO binding | Inference avg @ chunk=0.15 (200pass bench) | 27.59 ms | 28.03 ms | **null / -1.6% — deferred** |
| 1 IO binding | Inference avg @ chunk=0.10 (200pass bench) | 27.38 ms | 27.60 ms | null / -0.8% — deferred |
| 2 native PW | `xruns` per 6 s engine smoke | 11-12 | 0 | **clean — pacat counter no longer fires (different mechanism is now in play; native-pw has its own underrun counter)** |
| 2 native PW | Helper underruns per 6 s smoke | n/a | 0-1 (first quantum only) | **clean** after ring stabilizes (~1 quantum) |
| 2 native PW | Engine writer_jitter_ms | ~70-77 ms | ~76-77 ms | **unchanged** (jitter is engine-side production, not output-stage) |
| 3 mitigations | Doc-only | n/a | n/a | n/a |

Things NOT directly measured at this point:
- Telegram VOIP cut count before/after Fix 2 (requires user-driven
  test with the 60-second waveform-evidence capture).
- `mitigations=off` impact on writer_jitter / xruns (requires user
  reboot + measurement template).
- Long-session (>30 min) underrun rate via Fix 2 vs the pacat baseline
  on a real Telegram call.

---

## 3. Honest assessment: are the audible cuts likely improved?

**Probably yes for the per-quantum gap class. Maybe not for everything.**

The audit's lens 08 evidence — sample-exact zeros, voice-correlated,
~21.33 / 42.67 ms quantized — maps cleanly onto the pw-cat
synchronous-stdin-on-RT-thread mechanism. Native-pw's SPSC ring buffer
removes that specific source of gaps by guaranteeing the RT callback
never blocks on stdin. The 6-second smoke test showed 0 underruns
(after the first quantum's startup gap), which is consistent with
the mechanism being eliminated.

What the fix does NOT touch:
- **Engine-side production jitter.** writer_jitter_ms stayed at ~76 ms
  in the smoke test — the engine still writes at chunk_seconds=0.15
  cadence with ~30 ms std variance. If the helper's ring underflows
  during a worst-case engine spike (inference p99 + GIL contention +
  whatever else), there will still be an underrun. The ring is sized
  8× quantum (≈ 170 ms slack), which absorbs typical jitter but not
  pathological multi-chunk stalls.
- **GPU-side variance.** Inference tail spikes (cuDNN edge-shape
  retraining, NVIDIA boost throttling on quiet GPU) are a different
  layer. v0.7.0 mostly closed cuDNN via EXHAUSTIVE; boost throttling
  is hardware behavior we explicitly don't touch (V0_9_X brief
  ban-listed it).
- **Telegram's downstream variance.** 100-200 ms of tg_owt encode +
  network jitter + their playback buffer is on top of whatever
  reaches woys-mic. Native-pw doesn't fix that, can't fix that.

Best-case prediction: voice-correlated micro-cuts on the engine side
disappear. Telegram-specific cuts that originated downstream are
unchanged. Subjective reduction in "white-click" frequency is
plausible; full elimination is not.

The Telegram self-call test is the only honest verdict; the brief
explicitly schedules it as part of the user's Phase C handoff.

---

## 4. Open questions / what would be next on v0.10.x

If this v0.9.0 is followed by v0.10.x work, the next moves in order
of leverage:

### High-leverage, contingent on the user's Telegram verdict

1. **Default flip to `prefer_native_pw=true` (v0.9.1).** One release
   of opt-in soak per the research recommendation. If the user reports
   cuts substantially reduced, this is the v0.9.1 release.
2. **Delete the legacy pw-cat / pacat paths (v0.10.x).** ~150 LOC of
   the `_open_pacat` else-branch + the watchdog/stderr interaction
   around them. Simpler maintenance. Contingent on (1).

### High-leverage, independent

3. **`from_app_config(cfg)` factory** for EngineConfig construction.
   The third drift catch in three releases (rc4 / B9 / v0.9.0-rc1)
   argues for eliminating the manual kwarg lists in cli.py / app.py
   entirely. AST test pins forwarding correctness; refactor closes
   the class.
4. **Engine-side jitter reduction.** The native-pw helper is now a
   non-ratelimiting stage; the bottleneck moves back to engine
   production cadence. perf-001 (per-chunk numpy alloc churn,
   ~600 KB/chunk) and perf-018 (`_input_history` ring buffer) become
   the next rungs. Both deferred from v0.8.0 review.
5. **Engine-side priority tuning revisit.** writer_jitter_ms ~76 ms
   after Fix 2 suggests SCHED_FIFO 60/59 isn't fully solving the
   producer-side variance the audit hypothesized. cyclictest baseline
   would tell us whether linux-rt is the right next lever or whether
   we're CPU-bound regardless.

### Nice to have

6. **Native helper telemetry expansion.** Currently emits `ready`,
   `quantum=N rate=M ...`, `underruns=N`, `error: ...`. Could add
   per-quantum jitter histogram, ring high/low watermark, queued-
   write-size distribution. Helper is small enough to instrument
   heavily without affecting RT path.
7. **AUR packaging update.** PKGBUILD + .SRCINFO bumped to 0.9.0;
   the helper's build is now a dependency. Confirm the AUR path's
   `gcc + libpipewire-0.3` chain in the next release.
8. **Tests for the helper.** Currently no automated test runs against
   the C binary. A small `tests/test_native_pw_helper.py` that pipes
   1 s of synthetic audio into the helper and asserts ready / no
   error / underrun count under bound would catch regressions.

### Out of scope for any v0.10.x

- GPU clock locking (V0_9_X hard-ban).
- linux-rt kernel install (host responsibility).
- Output buffer ladder tuning (closed-out negative result through
  v0.7.x).

---

## 5. Stats

- Brief listed 3 fixes; shipped 2 (Fix 2 + Fix 3); deferred 1 (Fix 1
  with documented null-result evidence).
- Two intermediate rcs: v0.9.0-rc1 (Fix 2), v0.9.0-rc2 (Fix 3).
- Final tag: v0.9.0.
- 120 fast tests pass (up from 118 in v0.8.0; +2 AST drift tests).
- One new module: `bin/woys-pw-out.c` (~430 LOC) + `bin/Makefile`.
- One new module-class doc: `docs/20-mitigations-tuning.md`.
- One investigation doc: `docs/19-pw-investigation.md`.
- One plan doc: `docs/19-v09x-plan.md`.
- LESSONS §23 + §24 + §25.

---

## 6. Handoff

Per the brief: **the user does ONE Telegram self-call as the final
audible verdict.**

The test:
1. Edit `~/.config/woys/config.toml`. Insert
   `prefer_native_pw = true` at the TOP level (before any
   `[profiles.*]` table — TOML interprets keys after `[profiles.X]`
   as belonging to that profile, not the engine).
2. `pkill -f "woys engine"` (the engine running PID 278049 is on the
   pre-v0.9.0 in-memory bytes).
3. `woys run --autostart` or `woys engine --quiet` — whichever the
   user prefers.
4. Verify in `pw-cli ls Node` that `woys-engine-out` appears with
   `media.class = Stream/Output/Audio`.
5. Make a Telegram self-call (voice message to "Saved Messages"
   works). Speak normally for 30-60 seconds.
6. Listen for the white-click cuts.

**Predictions:**
- Cuts reduced or gone: native-pw is the dominant fix; flip default
  to True for v0.9.1 next release.
- Cuts unchanged: the per-quantum gap mechanism wasn't the dominant
  cause on this hardware after all; debugging shifts to engine-side
  production jitter (perf-001, perf-018, linux-rt).
- Cuts WORSE: helper bug; investigate via the captured stderr
  (`underruns=N` line every second) and the pw-record monitor capture.

**Final notes:**
- v0.9.0 is shipped. All commits + tags pushed to origin/main.
- The user's existing `~/.config/woys/config.toml` is unchanged
  during this autonomous run (any test inserts were rolled back).
- Engine PID 278049 is on the pre-v0.9.0 in-memory bytes; needs a
  restart to pick up v0.9.0 code.

End of v0.9.x autonomous run.
