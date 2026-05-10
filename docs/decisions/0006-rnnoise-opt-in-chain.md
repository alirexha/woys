# 0006 — RNNoise as an opt-in pipewire-pulse chain after `woys-mic`

## Decision

RNNoise denoising is shipped as an opt-in pipewire-pulse module chain
(setup via `scripts/v013_0_rnnoise_chain.sh setup`) that sources from
`woys-mic` and exposes a cleaned `woys-by-alirexha` source; it is not
integrated into the engine and is not on by default.

## Status

`accepted`

## Context

After v0.12.4 closed the chunk-period rhythm at autocorrelation 0.000
(`LESSONS.md` §42), the user asked whether NoiseTorch / RNNoise after
woys would help with the residual chunk-boundary clicks the user had
already accepted as acceptable. Hypothesis going in: probably not.
Result: a 27 % cuts/min reduction (corrected from the original 13 %
in `LESSONS.md` §43 once the v0.13.2 routing leak in §44 was fixed),
at +40 ms latency, depending on the system package
`noise-suppression-for-voice` (Arch extras). v0.13.3 polished naming
so apps see `woys-by-alirexha` (clean default-pick) and
`woys-no-cleanup` (raw fallback) with internals tagged `_internal-...`
(`LESSONS.md` §45).

## Decision

Ship as an opt-in chain, sourcing from `woys-mic`, exposing
`woys-by-alirexha` as the user-facing cleaned endpoint.

## Alternatives considered

- **Native in-process RNNoise after RVC** — vendor `librnnoise.so`
  (BSD-2-clause), call it inside the engine. Avoids the LADSPA
  dependency and external system package, keeps the audio path in
  one process. But adds ~50 KB of vendored binary, requires a build
  step, and couples the cleanup stage to engine release cadence.
- **Always-on RNNoise (default)** — flips on by default, users get
  the −27 % cut reduction free. Costs every user +40 ms latency on
  top of v0.12.4's already-+100 ms (~680 ms total e2e), pushing
  near the conversational comfort threshold (decision 0013), and
  forces the system package as a hard dep.
- **No RNNoise at all** — accept the v0.12.4 residual. The user's
  v0.12.4 listening verdict was "the rhythm is GONE; this is what
  woys should sound like" — i.e., good enough already.
- **NoiseTorch's built-in chain (`-i`)** — failed with "No such
  entity" on this stack; sink/master ordering incompatible with
  PipeWire 1.6.4 + pipewire-pulse 15.0 (`LESSONS.md` §43).

## Rationale

Opt-in fits four facts. First, the cuts/min reduction is real but
conditional: 27 % (post-§44 fix) is worth shipping but not so large
that every user must pay the +40 ms cost. Second, the user's v0.12.4
verdict already accepted the residual, so we are not denying anyone
audible quality by gating. Third, the dep on
`noise-suppression-for-voice` is non-woys system surface; making it
hard would block users who haven't run `pacman -S` for it. Fourth,
the LADSPA-via-pipewire-pulse architecture is the integration path
that actually works on PipeWire 1.6.4 (`LESSONS.md` §44) — native
in-process integration would be an extra build step solving a problem
v0.13.2 already solved with PipeWire modules.

## Trade-offs accepted

Apps must re-select `woys-by-alirexha` instead of `woys-mic` after
running setup — UX cost for the cleaner output. The chain depends on
a non-woys system package; install instructions document this.
+40 ms latency on top of v0.12.4 lands total e2e near the
conversational threshold (decision 0013). The systemd-managed
`woys-chain.service` (`pkg/woys-chain.service`) keeps the chain
loaded across pipewire-pulse restarts, but adds a unit that must be
maintained alongside the user-service `woys.service`.

## Re-litigation triggers

- A measurement on a future RVC successor (no NSF) shows residual
  clicks below the detection floor — RNNoise stops paying for itself.
- `noise-suppression-for-voice` upstream changes its plugin path or
  ABI; we vendor `librnnoise.so` instead.
- A user listening test at +40 ms latency vs current default returns
  "always on is better" with statistical confidence — flip default.
