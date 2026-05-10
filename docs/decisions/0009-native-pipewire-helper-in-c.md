# 0009 — Native PipeWire output helper written in C

## Decision

The audio output path is a small (~250 LOC) C program
(`bin/woys-pw-out.c`) linked against `libpipewire-0.3` that reads
audio frames from a Unix-domain socket and feeds them into a
PipeWire stream's RT process callback via memcpy from a ring buffer.

## Status

`accepted`

## Context

`docs/19-pw-investigation.md` is the chronological record of how we
got here; `LESSONS.md` §24 summarises the engineering decision.
Lens 08 of the v0.7.x audit (`docs/16-audit/synthesis.md`) found
voice-correlated, sample-exact zero gaps quantised to ~21.33 / 42.67
ms in the cut signature on Telegram VOIP — exactly one PipeWire
quantum at 1024/48000. `pw-cat` reads stdin synchronously inside its
RT process callback chain (verified against upstream `pw-cat.c`); when
the engine's chunk-write timing falls out of phase with the quantum
boundary, that synchronous read hits stdin-empty and the buffer goes
out zero-padded for that quantum. The native helper closes the race
by keeping the RT thread strictly memcpy-from-ring with no I/O.

## Decision

C native helper using `libpipewire-0.3`, with a Unix-domain socket
between engine producer and helper RT consumer.

## Alternatives considered

- **`sounddevice` / PortAudio** — Option A in `docs/19-pw-investigation.md`.
  PortAudio's PipeWire backend goes through the JACK compatibility
  shim, adding a layer of buffering with its own quantum behaviour.
  Doesn't solve the RT-thread-blocks-on-producer problem.
- **`pipewire-python`** — Option A.5. Library is unmaintained; the
  upstream's last release predates PipeWire 1.0. The `pactl
  load-module` shell-out path used elsewhere (decision implicit in
  `src/audio/pipewire.py`) confirmed avoiding pipewire-python is
  the right call for woys; reusing the same conclusion here.
- **`ctypes` / `cffi` against libpipewire from Python** — Option B.
  Possible, but the RT process callback runs in a thread the Python
  GIL doesn't own, and any Python interpreter call from that
  callback is undefined behaviour. The "no Python in the RT thread"
  rule made this a non-starter.
- **Continue with `pw-cat`** — Option D / status quo before v0.9.0.
  Confirmed broken by the v0.7.x cut investigation; retained
  default-off for fallback (`prefer_pw_cat: bool = False`) but
  superseded as the primary path.

## Rationale

The hard constraint is "the RT thread must never block on producer
arrival." In Python, every option either runs Python in the RT
thread (bad) or shells out to a process whose RT-thread design we
don't control (`pw-cat` shells stdin into the RT loop — bad). Writing
~250 LOC of C against `libpipewire-0.3` directly gives us the only
RT-thread shape that satisfies the constraint: a lock-free ring
buffer fed by the engine's writer thread (Python, off the RT path),
drained by the C process callback with `memcpy` only. v0.9.0-rc1
landed this path; v0.11.0's anti-jitter built on top of it; the
v0.12.x / v0.13.x stack continues to use it as the default backend.

## Trade-offs accepted

A C compile step at install time — install.sh detects
`libpipewire-0.3` dev headers and compiles `woys-pw-out` if
present; otherwise falls back to `pacat`. Three backend code paths
co-exist in tree (native helper + pacat + pw-cat), each with its
own watchdog and underrun parser; consolidation is deferred work
documented elsewhere. Watchdog respawn carries helper exit reason
forward via `EngineStats.helper_exit_reasons` (`LESSONS.md` §35) so
mid-session deaths leave a diagnostic trail.

## Re-litigation triggers

- A maintained Python binding for libpipewire-0.3 appears that
  exposes the process callback as a callable that can be wired to
  a no-GIL ring read.
- PipeWire upstream changes the RT-thread contract such that
  blocking is allowed under some condition (extremely unlikely).
- A simpler I/O backend (e.g., a future `pw-cat` rewrite without
  the synchronous stdin read) lands and benchmarks at parity with
  the helper at chunk_seconds=0.25.
