# `mitigations=off` — host tuning for woys (optional, user-driven)

> **Read this first**: woys does **not** modify boot parameters for you.
> This is a host-tuning recommendation. You apply it manually, you
> revert it manually, you accept the security tradeoff manually. Read
> the entire document, including [§3](#3-security-tradeoff), before
> editing anything.

---

## 1. What this does

The Linux kernel ships with mitigations for hardware-level CPU
vulnerabilities (Spectre v1/v2, Meltdown, MDS, L1TF, retbleed, GDS,
etc.). These mitigations insert guard barriers around speculative
execution to prevent side-channel attacks. The barriers cost cycles.

For audio workloads — many small syscalls per second, frequent
context switches, latency-sensitive — these costs aggregate into
measurable per-call overhead. **5–15% syscall-cost reduction** is a
typical observed delta after disabling mitigations on this class of
workload.

`mitigations=off` is a kernel command-line flag that disables every
mitigation at boot. It's reversible (just remove the flag and reboot).

### What this is NOT

- It is **not** a real-time kernel switch. That's a separate move
  (replacing `linux-cachyos` with `linux-cachyos-rt`).
- It is **not** a GPU clock lock. woys explicitly does not touch GPU
  hardware behavior — the hard rule from `V0_9_X_AUTONOMOUS.md`.
- It is **not** a guarantee that audible cuts disappear. It removes
  one source of jitter (per-syscall mitigation cost). The native
  PipeWire client (v0.9.0-rc1) addresses a different layer; `linux-rt`
  would address yet another. These are independent levers.

### What to expect

On the typical-case audio path:
- `woys diag --duration 60` `inference_overrun_ratio` may drop
  modestly (single-percent territory).
- `writer_jitter_ms` may tighten (lower max, similar p50).
- Subjective listening: small improvement, not transformative.

If you're hearing voice-correlated cuts, the dominant mechanism is
likely the per-quantum pacat/pw-cat gap (lens 08 evidence,
`docs/16-audit/synthesis.md`). `mitigations=off` is a complementary
fix, not the headline one.

---

## 2. Apply on CachyOS (systemd-boot)

CachyOS uses `systemd-boot`. Entries live in
`/boot/loader/entries/*.conf`. Find your active entry:

```fish
sudo bootctl list
```

Look for "(default)" — that's the entry to edit. Open it:

```fish
sudo $EDITOR /boot/loader/entries/<your-entry>.conf
```

The file looks like:

```ini
title   CachyOS Linux
linux   /vmlinuz-linux-cachyos
initrd  /initramfs-linux-cachyos.img
options root=UUID=... rw rootflags=subvol=/@ zswap.enabled=0 nowatchdog quiet splash
```

Append `mitigations=off` to the `options` line:

```ini
options root=UUID=... rw rootflags=subvol=/@ zswap.enabled=0 nowatchdog quiet splash mitigations=off
```

Save. Reboot. Verify the flag took effect:

```fish
cat /proc/cmdline
```

Should now contain `mitigations=off`. Confirm the kernel actually
disabled the mitigations (not just received the flag):

```fish
cat /sys/devices/system/cpu/vulnerabilities/*
```

Most lines should now read `Vulnerable` (instead of
`Mitigation: ...`). That's the expected state with `mitigations=off`.

### Note on dual-boot / kernel choice

If you have multiple kernel entries (e.g. `linux` plus
`linux-cachyos`), each has its own `.conf`. Edit only the entries
where you want the flag. Keeping a non-mitigated entry as default and
a mitigated entry as fallback is reasonable defense-in-depth.

---

## 3. Security tradeoff

Disabling mitigations re-exposes the host to the side-channel
attacks they cover. The practical risk model on a personal laptop:

**Attack vector requires** untrusted code running on the same CPU
core(s) as a process you care about, with the goal of leaking memory
the attacker can't read directly. Concretely:

- A malicious browser script using JIT speculative-execution patterns
  to read another tab's memory (Spectre v1 class).
- A compromised package's post-install hook reading `/etc/shadow`
  fragments via cache side channels (MDS / L1TF class).
- A guest VM escaping into the host (more relevant on shared
  infrastructure than personal Linux).

**Practical risk on this laptop** (single-user, personal use):

- Low, but **not zero**. Modern browsers use site isolation so
  cross-tab leaks via Spectre are mostly closed at the browser level
  even without kernel mitigations.
- If you ever run untrusted code (e.g. random GitHub Python scripts,
  Discord bots, untrusted Steam Workshop content), the risk goes up.
- If you use this machine for sensitive work (banking, dental records,
  password vaults), the risk multiplies.

**Recommended posture**:

- Apply `mitigations=off` to a dedicated kernel entry, not the only
  entry. Boot mitigated by default; switch to the unmitigated entry
  only when running woys + Telegram for an extended call.
- Or: apply globally, but DON'T also run untrusted browser content,
  unsandboxed code, or shared/multi-user workloads on the same host.
- Document for yourself when you flipped it. Future you will forget.

---

## 4. Revert

Remove `mitigations=off` from the `options` line in
`/boot/loader/entries/<entry>.conf`. Reboot. Verify:

```fish
cat /proc/cmdline                                     # should NOT contain mitigations=off
cat /sys/devices/system/cpu/vulnerabilities/spectre_v2  # should read Mitigation: ...
```

Reverting is symmetric. No state survives across reboots.

---

## 5. Measurement template (before vs after)

Don't trust subjective listening alone — `woys diag --duration 60`
gives you numbers that lock the comparison.

**Before** (mitigations enabled, default state):

```fish
# Make sure the engine is running normally.
woys engine --quiet > /tmp/woys-engine-before.log 2>&1 &
sleep 8     # let the engine warm up

# Capture a 60-second diag with the real engine running.
woys diag --duration 60 > /tmp/woys-diag-before.txt 2>&1

# Capture audio output for offline analysis.
pw-record --target=WoysSink.monitor --rate=48000 --format=s16le \
    --channels=2 --file-format=wav /tmp/woys-before.wav &
sleep 60
kill %2     # stop the recording
kill %1     # stop the engine

# Save the before-state.
cat /proc/cmdline > /tmp/cmdline-before.txt
date >> /tmp/cmdline-before.txt
```

**Edit the boot entry, reboot, repeat with `-after` suffixes.**

**Compare**:

```fish
diff /tmp/woys-diag-before.txt /tmp/woys-diag-after.txt
```

Look at:
- `inference_overrun_ratio` (= `late_chunks / chunks_processed`).
  Lower = better.
- `writer_jitter_ms` (max + std). Tighter = better.
- `xruns` (pacat backend only; pw-cat / native-PW report nothing
  via this counter).
- The first few `slow_chunk_log` entries' `total_ms` — outlier tail.

Optionally, run the audit's waveform-evidence script to count cut
events:

```fish
.venv/bin/python docs/16-audit/waveform-evidence/analyze-cut-capture.py \
    /tmp/woys-before.wav --plot /tmp/cuts-before.png

.venv/bin/python docs/16-audit/waveform-evidence/analyze-cut-capture.py \
    /tmp/woys-after.wav --plot /tmp/cuts-after.png
```

Compare the gap counts and gap-duration histograms across the two
captures.

---

## 6. Why woys does NOT modify boot params for you

The brief that drove v0.9.x explicitly bans woys from touching boot
config or other host-wide system state. Reasons:

1. **Boot-param edits are sudo operations on a file outside the
   user's home directory.** woys runs as a regular user; it has no
   business escalating to write `/boot/loader/entries/`.
2. **Reboot is required to apply.** woys can't safely reboot the
   user's machine — they may have unsaved work, an in-progress
   download, a Telegram call, etc. Forcing a reboot to "apply"
   a config change would be hostile.
3. **The security tradeoff is the user's call, not woys's.** A tool
   that silently disables CPU mitigations on a multi-user host (or a
   host the user later shares) has shipped a security regression for
   them. The right move is to surface the option clearly and let the
   user own the decision.
4. **Reversibility belongs in the user's workflow, not woys's.** If
   woys flipped the flag, woys would also need a "revert this" path
   that survives uninstall, package upgrade, kernel upgrade, etc.
   That's complexity for a marginal feature.

If you want a one-command convenience wrapper (e.g.,
`woys host enable-mitigations` / `woys host disable-mitigations`),
you can write one yourself as a shell function in your fish/bash rc.
woys will not ship one.

---

## 7. Combining with other tunings

`mitigations=off` is one of three independent levers documented in
v0.7.0/v0.8.0 LESSONS:

| Lever | Reduces | Cost | Reversibility |
|-------|---------|------|---------------|
| `mitigations=off` (this doc) | per-syscall jitter | security exposure | reboot |
| `linux-cachyos-rt` kernel | scheduler preemption | ~5-15% throughput | reboot, separate package |
| Native PipeWire client (v0.9.0) | per-quantum pacat gap | none | run any version |

Apply them **in sequence**, not together. After each, run the
measurement template above. You learn which lever moves what,
instead of doing all three at once and not knowing what helped.

If the cuts persist after all three, the next lens to investigate is
the GPU-side variance (cuDNN tail spikes, NVIDIA boost throttling on
quiet GPU). Those are out of scope for v0.9.x — see
`V0_9_X_AUTONOMOUS.md` hard constraints.

---

End of doc.
