# 0007 — `gpu_anti_jitter_mode = "off"` as the default

## Decision

`EngineConfig.gpu_anti_jitter_mode` defaults to `"off"`. The four
modes — `off`, `keepalive`, `clock_lock`, `both` — are user-flippable
in `config.toml`; `"both"` is the validated win and requires a
sudoers entry.

## Status

`accepted`

## Context

`LESSONS.md` §31-§34 documents the v0.11.0 finding that NVIDIA dynamic
boost on this laptop GPU has two closed-loop inputs (power-state
policy + workload demand), and that **only mode `"both"`** —
`nvidia-smi -lgc <floor>,<ceiling>` plus a `torch.cuda.Stream()`
keepalive at 25 ms cadence — drove the synthetic clock p50 to the
locked floor (1845 MHz on RTX 2070 Mobile) and produced the user's
real-Telegram 36× underrun reduction (from 7.3/sec to 0.2/sec). The
keepalive-only and clock-lock-only halves were measured: each landed
at GPU clock p50 ≈ 1680 MHz, essentially indistinguishable from
`mode=off`. Code at `src/audio/engine.py:578-637`.

## Decision

Default `mode="off"`; ship `"both"` as the documented opt-in for users
who set up the sudoers entry per `docs/22-gpu-clock-lock.md`.

## Alternatives considered

- **Default `mode="keepalive"`** — torch-only keepalive, no sudo
  requirement. But `LESSONS.md` §34 shows keepalive alone does not
  pull the GPU clock above natural idle (1680 MHz on this hardware),
  so the user-visible benefit is near-zero. Cost is ~5 W constant GPU
  work for a benefit that doesn't materialise without the lock.
- **Default `mode="clock_lock"`** — needs sudoers; worse, lock alone
  is treated as PERMISSION not COMMAND under bursty workload (`LESSONS.md`
  §34), so alone it also lands at 1680 MHz p50. Sudo prereq for no
  measured win.
- **Default `mode="both"`** — the validated 36× win, but requires
  every user to configure a sudoers entry for `nvidia-smi -lgc/-rgc`,
  burns ~5 W during engine runtime, and assumes an NVIDIA GPU on a
  laptop with dynamic-boost behaviour matching this stack's profile.

## Rationale

`"off"` is the only mode that doesn't make assumptions about the
user's hardware (NVIDIA discrete laptop GPU vs desktop vs AMD vs
integrated) or their willingness to grant `nvidia-smi -lgc/-rgc`
sudoers. It's the floor — engine works without prereqs, no power
burn for users who don't need it. The keepalive-only half is not a
listener-validated 36× win (the v0.11.0 review in
internal notes corrected an earlier misread
that was about to flip the default). Users who hit underruns and
read `docs/22-gpu-clock-lock.md` opt into `"both"` once and keep it.

## Trade-offs accepted

Users on hardware that benefits from anti-jitter (RTX laptops with
dynamic boost) get a worse out-of-box experience than they could
have. We mitigate by surfacing the recommendation in `docs/22-gpu-clock-lock.md`
and in `woys diag` output. Users on AMD GPUs, integrated graphics, or
desktop GPUs with stable clocks get no benefit and no penalty from
the default — `"off"` matches their environment.

## Re-litigation triggers

- A future synthetic harness validates `keepalive` alone as
  ≥10× cuts reduction at p99 on a representative listener-test set —
  flip default to `keepalive` (no sudo prereq).
- NVIDIA driver / `nvidia-smi` ships a non-sudo per-process clock
  pinning interface — flip default to `clock_lock` or `both` since
  the prereq disappears.
- AMD ROCm gains an analogous knob and we add support; the default
  policy may differ per backend.
