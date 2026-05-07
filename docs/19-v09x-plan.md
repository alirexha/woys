# woys v0.9.x — execution plan (Phase A output)

Date: 2026-05-08

This file is the execution plan written before any v0.9.x code lands.
Posterity record so future-me / future-reviewer can see the reasoning
that drove the order chosen, given the audit (`docs/16-audit/synthesis.md`)
and the v0.8.0 review verdict (`docs/17-review/02-final-verdict.md`).

---

## The three fixes (per V0_9_X_AUTONOMOUS.md)

1. **ORT IO binding** — port `scripts/bench_iobinding.py`'s pattern into
   the engine's `_infer` for cv / rmvpe / rvc sessions. Eliminates per-
   chunk OrtValue allocation; binds outputs on CUDA to skip premature
   D2H copies. Parity test gate: cosine similarity ≥ 0.99 vs the legacy
   `sess.run` path on synthetic input.

2. **Native PipeWire client** — replace the `pacat` / `pw-cat`
   subprocess shellout with a direct PipeWire client process registered
   at the PipeWire quantum boundary. Closes the per-quantum gap
   pathology the audit fingered (lens 08's 42.67 ms FFT peak; pw-cat's
   documented bursty-write race; pacat's structurally-similar timing
   slop). `prefer_native_pw = True` config flag, hard-fail on
   registration failure (no silent fallback).

3. **mitigations=off documentation** — `docs/20-mitigations-tuning.md`
   only. Doc explains GRUB / systemd-boot edit per CachyOS, security
   tradeoff, before/after measurement template using `woys diag
   --duration 60`. woys never touches boot params.

---

## Order: 1 → 2 → 3

### Why this order

**Cost-vs-reward.** Fix 1 has a working pattern in
`scripts/bench_iobinding.py` (193 lines, fully implemented). Porting
into `engine._infer` is mechanical. Fix 2 is a research-and-
implementation task on a Python ecosystem (PipeWire bindings) that is
genuinely sparse — could land cleanly, could turn into a deferral.
Fix 3 is doc-only. Putting Fix 1 first banks a guaranteed win before
the open-ended Fix 2 starts; Fix 3 last is cooldown.

**Risk of regression.** Fix 1 touches inference path; parity test (cos
sim ≥ 0.99) catches any silent-wrong-output regression cheaply. Fix 2
touches audio output path AND introduces a new subsystem (native PW
client) — much larger surface for "engine works but Discord/Telegram
sees nothing" failure modes. Doing Fix 1 first means Fix 2's
verification can include "inference still produces correct samples"
as a known-good baseline.

**Dependency.** Fix 1 trims inference latency (estimated 10-30%); Fix
2's native PW client integration may want to negotiate tighter buffers
when inference has more headroom. Fix 1 → Fix 2 means Fix 2 designs
against the post-IO-binding latency, not the legacy one.

**Audit corroboration of cause attribution.**

For Fix 1: lens 04 (cuDNN) + perf-001 (per-chunk numpy alloc churn,
~600 KB/chunk + GIL release/acquire) + perf-004 (ORT IO binding
specifically called out as deferred opportunity). The mechanism is
real but hits *throughput*, not the specific tail-spike mechanism the
audit fingered. Fix 1's user-facing benefit in v0.9.0 is freed-up
inference budget, NOT a direct attack on the cut signature.

For Fix 2: this is the one that maps cleanly to the audit's actual cut
signature. Lens 08 found:
- Voice-correlated, sample-exact zero gaps
- Quantized to ~21 / 42.67 ms (top FFT peaks in onset periodicity)
- Inter-gap interval clustering at ~130 ms (chunk-cadenced)
- Count flat across `output_latency_ms` 80→220→280

The 42.67 ms peak is exactly one PipeWire quantum (1024 samples at
44100 / matches default quantum on this system). pw-cat's per-quantum
gap pathology is documented in `engine.py:140-154`. pacat (the
v0.8.0 default after rc4 reverted `prefer_pw_cat=True`) has a
structurally-similar subprocess-pipe-vs-PulseAudio-callback race. A
native client at the quantum boundary is the only fix on this list
that *attacks the actual signature*.

For Fix 3: orthogonal to both audio and inference; user-controllable
host tuning. No audit corroboration needed (it's a doc).

### Why NOT a different order

- 2 → 1 → 3 (hardest first): risks burning lots of context on Fix 2
  open-ended research before banking the guaranteed Fix 1 win. If
  Fix 2 dead-ends, Fix 1 still ships from a clean rc1.
- 3 → 1 → 2 (easiest first): Fix 3 is doc-only and can ship anytime;
  doing it first wastes the warm-context window on the wrong task.
- All three in parallel: contradicts the brief's "no bundling" rule.

---

## Stack-specific concerns

**Engine restart.** PID 278049 currently runs the v0.7.0 in-memory
bytes. Fix 1's real-mic verification needs an engine on v0.9.0-rc1
bytes. Will restart once per rc, with explicit log entry. Telegram
audio dies during restart windows (~10s). Acceptable per the brief's
verification protocol.

**PipeWire Python ecosystem maturity.** `libpipewire-0.3.so.0` is
present on this CachyOS box; `pipewire-python` is NOT in the venv.
Three realistic Fix 2 paths to investigate:

1. `pipewire-python` (PyPI): wrapper around dbus + libpipewire.
   Need to evaluate quality; reported as sparse on PipeWire 1.6.
2. `ctypes` shim against `libpipewire-0.3` directly: more code, no
   external dep, full control over thread scheduling.
3. Tiny native binary (C / Rust) doing the PipeWire client work,
   pipe protocol to Python: middle ground; still has a subprocess
   in the path but at the negotiated quantum boundary, not pw-cat's
   buggy one.

**If all three Fix 2 paths dead-end** (libpipewire ABI gap on this
PipeWire version, etc.), the brief authorizes deferral with a
LESSONS.md entry. Will not ship a half-broken native client.

**The "matches upstream" trap.** v0.8.0 caught me citing this
fabricated four times. For Fix 2 specifically, "upstream" is
`w-okada/voice-changer`, which is a webserver (FastAPI + Socket.IO) —
it doesn't have a Linux-native PipeWire client to crib from. So no
fake-upstream-citation risk for this fix. Will state plainly when
relying on PipeWire's own C examples vs. Python-ecosystem patterns.

---

## Verification gates per rc

Per the brief, every commit must pass:

1. **Diff review.** Re-read the full diff, hunting bugs, edge cases,
   masked type errors, comment-vs-code drift.
2. **Dev tools.** Pytest fast suite (must be ≥ 118 passing; target
   118 + new tests for each fix). `mypy --strict` clean (or with
   intentional, justified `# type: ignore[...]` only). `ruff format
   --check` + `ruff check` clean.
3. **Engine smoke.** `woys diag --no-engine` + `woys engine --quiet --seconds 6`
   in CC's bash. Both must complete cleanly.
4. **Real-time test.**
   - Fix 1 (rc1): parity test, cos sim ≥ 0.99 between legacy and
     IO-bound paths on synthetic input across cv / rmvpe / rvc.
     `pytest tests/test_iobinding_parity.py` (new).
   - Fix 2 (rc2): bring engine up via `woys engine`, verify
     `pactl list source-outputs` (or `pw-cli ls Node`) shows our
     native client registered, capture
     `pw-record --target=woys-mic.monitor` for 30s during synthetic
     audio injection, analyze for non-silence + non-garbage.
   - Fix 3 (rc3): doc-only; render in `mdcat` / `glow`, verify all
     commands run as written.
5. **Trials and errors.** Per brief: 60+ second real-style harness
   capture per fix, compare to v0.8.0 baseline, gate-block on regression.

---

## Stop-the-line conditions

Will pause and surface to user if:

- Output quality regresses vs v0.8.0 baseline on the synthetic
  60-second harness (audio gap rate, RMS distribution, etc.).
- Existing tests fail in ways I cannot fix without modifying their
  expected behavior (a test claiming an invariant my changes broke
  for-real, not just rebased).
- Cosine similarity < 0.99 on Fix 1 parity gate (silent wrong outputs).
- Native PW client cannot register, even after fallback path
  investigation, AND I cannot determine why → document and defer
  Fix 2 (LESSONS.md).
- Hardware-mod temptation arises (already banned, but listed for
  completeness).
- A bundling temptation arises (banned).
- Push to origin/main fails (network, auth, etc.).

---

## Push policy

Per brief: push tags + commits to origin/main as I go (one rc per
push). Different from my usual default but explicitly authorized
by the brief.

---

## Sub-agent usage budget

- **Fix 1 (IO binding):** None planned. Implementation is mechanical
  port from a working bench file; engineering coherence matters more
  than parallelism.
- **Fix 2 (native PW client):** One scoped sub-agent for the
  Python-PipeWire-ecosystem investigation (read pipewire-python
  source / libpipewire docs / find existing patterns). Keeps that
  research out of my main context. Implementation will be sequential.
- **Fix 3 (docs):** None planned. Doc is short.
- **Final summary (Phase C):** Possibly one sub-agent to draft the
  LESSONS.md retros and `docs/21-v09x-final-summary.md` while I
  prep the v0.9.0 final tag.

---

## Confidence calibration

- Fix 1: 80% land cleanly. 20% chance of hot-swap rebinding bug
  requiring careful CUDA-tensor-lifetime work.
- Fix 2: 50% land in this session. 50% chance Python PipeWire surface
  is too sparse; honest deferral per brief is the right outcome.
- Fix 3: 99%.

Three-fix combined probability of full v0.9.0 final tag in this
session: ~50%. The brief explicitly says "If a fix turns out to be a
dead end on this stack, document why in LESSONS.md, move on" — so the
"50% Fix 2 lands" branch is acceptable; the "50% Fix 2 deferred"
branch is also acceptable. Either way v0.9.0 ships with whichever
fixes verified clean.

---

## Out

Begin Fix 1 immediately after this file lands.
