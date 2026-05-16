# 22 — GPU clock lock + torch separate-stream keepalive (v0.11.0)

The v0.10.0 retrospective (LESSONS §29-§30) located the dominant cause
of audible cuts on this stack: NVIDIA's dynamic boost auto-deboosts the
GPU during the engine's ~98 ms `mic_read` idle window between chunks,
producing variable reboost-recovery cost on the next chunk's RVC
inference. v0.11.0 ships two opt-in mitigations:

| feature      | what it does                                    | needs sudo? | replaces |
|--------------|-------------------------------------------------|-------------|----------|
| clock_lock   | `nvidia-smi -lgc <floor>,<ceiling>`             | yes         | -        |
| keepalive    | `torch.cuda.Stream()` keepalive every 25 ms     | no          | rc3 ORT keepalive |
| both         | clock_lock + keepalive                          | yes         | -        |

Default off. User explicitly opts in via `gpu_anti_jitter_mode` in
`~/.config/woys/config.toml`.

The v0.11.0 5-min harness shows **mode=both is the only setting that
materially moves writer_jitter p99 toward the gate** — clock_lock
alone or keepalive alone produce within-noise differences on this
specific hardware (RTX 2070 Mobile, NVIDIA driver 595, ORT-CUDA
1.22).

## Quick start

Add to `~/.config/woys/config.toml`:

```toml
gpu_anti_jitter_mode = "both"   # off | keepalive | clock_lock | both
```

For "both" or "clock_lock" you also need a passwordless sudoers entry
(see "Sudoers setup" below).

## Hardware safety statement

Both features use **stock GPU specs only**. The implementation explicitly
refuses any setting that:

- Locks the ceiling above `clocks.max.graphics` (the GPU's NVIDIA-validated
  boost ceiling)
- Locks the floor below 600 MHz
- Touches power limits (`-pl`), memory clocks (`-ac`), or applies
  undervolting

The lock is reverted automatically on:

- `engine.stop()` (normal shutdown, TUI quit, `woys` Ctrl-C from an
  interactive shell)
- `SIGTERM` (e.g., `kill <pid>`, `systemctl --user stop woys-mic`)
- `SIGINT` (e.g., Ctrl-C in `woys engine`)

The lock is **not** reverted on `SIGKILL` (`kill -9`) — the kernel
delivers SIGKILL out-of-band and Python can't intercept. If the user
ever sees the GPU stuck at a locked clock with no woys process
running, run `sudo nvidia-smi -rgc` manually to release.

The torch keepalive issues a 1024-element float32 `tensor.add(1.0)`
(~50 µs of GPU work) every 25 ms on a CUDA stream separate from
ORT's session stream. Total continuous GPU duty cycle ~0.2 %. No
state outside the engine process; the CUDA stream is destroyed when
the engine thread exits.

Neither feature touches firmware, BIOS, the NVIDIA driver, or any
persistent system state. Both can be reverted by:

- Setting `gpu_anti_jitter_mode = "off"` in config.toml (engine restart)
- Or removing `/etc/sudoers.d/woys-gpu-clock` to disable the
  passwordless sudo (then the engine refuses to start with
  clock_lock and surfaces a clear error)
- Or rebooting (the lock auto-clears on driver reload)

## Sudoers setup (clock_lock + both modes)

The clock_lock feature calls `sudo -n nvidia-smi -lgc/-rgc`. For
the engine to invoke this without prompting on every start, add a
passwordless entry for those two specific subcommands.

Create `/etc/sudoers.d/woys-gpu-clock` with the following contents
(replace `<your-username>` with your username):

```
# woys v0.11.0 — passwordless nvidia-smi for GPU clock lock anti-jitter.
<your-username> ALL=(root) NOPASSWD: /usr/bin/nvidia-smi -lgc *
<your-username> ALL=(root) NOPASSWD: /usr/bin/nvidia-smi -rgc
```

Set permissions to 0440 and verify with visudo:

```
sudo chmod 0440 /etc/sudoers.d/woys-gpu-clock
sudo visudo -c -f /etc/sudoers.d/woys-gpu-clock
```

The wildcard `*` after `-lgc` accepts any clock-pair argument; the
engine validates the clock values before invoking, so no over-stock
value can be passed even if the sudoers wildcard were exploited.

## Troubleshooting

**"nvidia-smi -lgc <floor>,<ceiling> failed: sudo: a password is required"**
  
  The sudoers.d entry isn't installed or the user in it doesn't
  match `whoami`. Re-run the install above and confirm with
  `sudo -n nvidia-smi -lgc 1845,2100`.

**"resolved clock-lock range (floor=X, ceiling=Y) is out of sanity bounds"**

  Auto-detection of `clocks.max.graphics` returned a value the engine
  refuses (out of [600, 4000] MHz range). Set
  `gpu_clock_lock_floor_mhz` and `gpu_clock_lock_ceiling_mhz`
  explicitly in config.toml.

**"gpu_clock_lock_active=False after engine start"**

  The lock applied successfully but reverted (either at engine.stop()
  or via SIGTERM/SIGINT). Check `EngineStats.last_error` and the
  engine log for the revert reason.

**Lock applied but `clocks.gr` still varies**

  On laptop GPUs with constrained TGP (e.g., RTX 2070 Mobile in a
  thermally-loaded chassis), `nvidia-smi -lgc <floor>,<ceiling>` is
  treated as a HINT by the GPU's internal boost mechanism. The lock
  ENABLES the GPU to run at the floor when sustained load demands
  it, but doesn't force the floor when the workload is bursty (the
  engine's mic_read window is too long for a hard floor). Combining
  clock_lock with the torch keepalive (`gpu_anti_jitter_mode = "both"`)
  provides the constant workload demand needed to actually keep the
  clock at the floor.

## Measured impact (RTX 2070 Mobile, 5-min synthetic harness)

| metric                | off    | keepalive | clock_lock | both   | gate    |
|-----------------------|-------:|----------:|-----------:|-------:|--------:|
| writer_jitter p99 (ms)|   99.7 |      89.3 |      109.4 |   59.4 |   ≤ 30  |
| inference avg (ms)    |   84.0 |      75.5 |       76.1 |   47.9 |   ≤ 52  |
| inference p99 (ms)    |  154.9 |     152.5 |      152.6 |   97.3 |    -    |
| rvc.run p99 (ms)      |  122.8 |     119.7 |      121.1 |   80.4 |    -    |
| underrun rate (/s)    |   6.77 |      7.55 |       6.56 |   6.85 |   ≤ 0.5 |
| GPU clocks p50 (MHz)  |   1710 |      1680 |       1680 |   1845 |    -    |

Headline: **mode=both delivers ~40 % writer_jitter p99 reduction
(99.7 → 59.4 ms) and the only configuration that holds the GPU at
the lock floor (1845 MHz p50 vs ~1680 in all other modes).** Closes
the inference_avg gate; doesn't yet close the writer_jitter or
underrun gates. v0.11.0 ships as a partial release.

## Why "both" works when neither alone does

The clock lock alone tells the GPU "you may run at 1845 MHz". On a
laptop with bursty workload, the GPU's boost mechanism still
gates that decision on sustained utilization — and the engine's
~50 ms RVC followed by ~100 ms idle isn't sustained enough to
trigger the lock floor.

The torch keepalive alone provides constant 0.2 % GPU duty cycle.
That's enough to register as activity but not enough to trigger
the boost-up — the GPU treats the small workload as background
noise and stays at its bursty-workload natural clock (~1680 MHz).

Combined, the lock tells the GPU "1845 is acceptable" and the
keepalive provides the continuous workload signal that says "we're
busy enough to deserve it." The GPU sustains 1845 MHz floor,
prevents the deboost-recovery cost on real RVC chunks.

This is consistent with the v0.10.0 rc3/rc4 finding that GPU clock
state is sensitive to sub-50 ms idle gaps; the rc3 ORT keepalive
hit that threshold but contention on ORT's stream regressed the
RVC tail. v0.11.0's torch keepalive on a separate stream avoids
that contention while filling the gap.

## How to disable

Set `gpu_anti_jitter_mode = "off"` in `~/.config/woys/config.toml`
and restart the engine. The lock auto-reverts on engine stop and
the torch keepalive thread exits.

To completely remove the sudoers entry:

```
sudo rm /etc/sudoers.d/woys-gpu-clock
```

## Original work — Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
