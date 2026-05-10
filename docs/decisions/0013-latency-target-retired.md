# 0013 — The 80 ms latency target is retired; ~640 ms is the user-validated optimum

## Decision

The brief's <80 ms end-to-end latency target (`PROJECT_BRIEF.md` §12,
restated in §18 Definition of Done) is formally retired. The
v0.12.4 listener-validated optimum is ~640 ms total e2e; the
operating budget is the conversational comfort threshold (~700 ms).

## Status

`accepted`

## Context

`PROJECT_BRIEF.md` §12 specified "End-to-end latency < 80 ms (mic →
transformed output) — measured, not claimed." The v0.7.0 perf push
(`docs/14-v070-baseline.md`, `docs/05-perf.md`) confirmed the floor on
this hardware (RTX 2070 Mobile + RVC v2 + ContentVec + RMVPE) is
~280 ms warm steady-state — i.e., the brief's number was unachievable
without rewriting RVC inference (forbidden, §12) or distilling the
model (forbidden, §12). v0.11.0 brought the cuts/min down at ~540 ms
(`LESSONS.md` §34). v0.12.4 promoted the +100 ms config (chunk_seconds
=0.25 + sola_context_ms=200) to the new default after a user
listening A/B (`LESSONS.md` §42). Reality is ~640 ms. The brief target
was abandoned without an explicit "we are no longer targeting <80 ms"
decision until now.

## Decision

We document the retire of the 80 ms target. Current operating budget:
≤700 ms (conversational comfort).

## Alternatives considered

- **Keep 80 ms target, miss it forever** — the v0.1.x → v0.13.x
  release history under-states to the brief target, every release.
  Honest accounting is better than a perpetually missed gate.
- **Retire to 320 ms (Telegram echo-cancellation horizon)** —
  `LESSONS.md` §28 cites Telegram's echo-cancellation horizon as
  ~320 ms. Below that, AEC handles the loop; above it, real-Telegram
  callees may hear an echo on their end. v0.11.0 / chunk_seconds=
  0.15 lands at ~540 ms — already past the echo horizon. Retiring
  to 320 ms would require chunk_seconds < ~0.10, which the v0.6.7
  / v0.6.8 sweep (`LESSONS.md` §16-§18) showed drops chunks during
  cuDNN warmup.
- **Retire to 700 ms (conversational threshold)** — corresponds to
  the engine.py:206 cite ("conversational threshold ~700 ms") and
  matches the user's accept band.

## Rationale

`PROJECT_BRIEF.md` §1's "End-to-end latency < 80 ms" was a
realistic target on the *upstream* (browser-based) numbers it was
derived from, but the actual measurements on this stack show the
inference floor is ~30 ms warm steady-state plus the chunk_seconds
buffer plus PipeWire quantum + AEC + apps' own buffering. The
v0.12.4 listening test made a perceptual finding the score formula
didn't capture (`LESSONS.md` §42): the user accepted +100 ms for a
collapse of the chunk-period autocorrelation peak from 0.067 to
0.000 ("the rhythm is GONE"). Below 700 ms, the conversational loop
still feels live; above ~700 ms, callees report waiting for replies.
The choice is "640 ms with no audible rhythm" over "540 ms with
audible rhythm" — perceptual measurement beat the score formula.

## Trade-offs accepted

We never hit the brief's 80 ms target on this hardware/model stack.
We do not attempt to. The 640 ms working point is well above
Telegram's echo-cancellation horizon (~320 ms), which means callees
on Telegram VoIP may hear a degree of echo back from us; the user's
side mitigates with Discord's noise suppression OFF on output (per
`docs/DISCORD-SETUP.md`) and with the v0.13.x RNNoise chain
(decision 0006) for residual transients. CS2 voice chat behaves
similarly — the gameplay loop tolerates the latency.

## Re-litigation triggers

- Hardware moves to a class without dynamic-boost (RTX 50xx desktop,
  Ada/Hopper datacenter) where chunk_seconds can drop without
  warmup penalties.
- RVC architecture retires NSF or otherwise eliminates the
  chunk-period mechanism (decision 0008) — chunk_seconds can drop
  without re-introducing rhythm.
- A new VC architecture (decision 0002) lands with sub-100 ms warm
  inference on this hardware.
- User feedback indicates 640 ms is too high for a specific use
  case (e.g., music-style real-time singing) — re-evaluate
  chunk_seconds for that branch only.
