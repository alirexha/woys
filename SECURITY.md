# Security policy

woys is a personal project, shipped under `All Rights Reserved` and
maintained best-effort. This file documents how to report a security
issue without exposing it publicly first.

## Supported versions

Only the latest release tag is supported. Older tags receive no
backports; upgrade to latest before reporting.

## Reporting a vulnerability

Email **alireza@hamayeli.com**. Please do **not** open an issue on the
GitHub tracker for security findings — that makes the report public
before there's a chance to remediate.

What helps:

- The affected woys version (`woys --version`).
- A clear repro path or proof-of-concept.
- Your assessment of impact (privilege escalation, audio-capture
  bypass, model-download MITM, sudoers misuse, etc.).

## Response window

Best-effort acknowledgement within **7 days**. This is a single-
maintainer alpha; longer turnaround during busy stretches or
hardware-gated investigations is possible. If you don't hear back in
14 days, a polite follow-up to the same address is welcome.

## What this project doesn't offer

- **No PGP key / signed channel.** Plain email is the only path. If
  you need encryption, host the encrypted blob as a private Gist and
  link it in your email — most pragmatic option for a project at
  this scale.
- **No bug bounty.** Acknowledgement and credit in the relevant
  `CHANGELOG.md` entry is the extent of recognition the project can
  offer.
- **No published embargo timeline.** Coordinated disclosure is
  welcome on a case-by-case basis; the timeline depends on the issue
  and the complexity of the fix.

## Scope

**In scope** — original work under this repo:

- `src/woys/`, `src/audio/`, `src/tui/`
- `bin/`, `pkg/`, `scripts/`
- `install.sh`, `uninstall.sh`
- The systemd user units (`pkg/woys-mic.service`, the generated
  `woys-chain.service`).
- Sudoers config snippets documented in `docs/22-gpu-clock-lock.md`.

**Out of scope** — report upstream instead:

- `src/server/` is vendored from w-okada/voice-changer (MIT) — file
  there: <https://github.com/w-okada/voice-changer>.
- Third-party dependency vulnerabilities (onnxruntime, torch, etc.)
  go to the dep's own tracker; this project will coordinate pin
  updates once upstream patches.
