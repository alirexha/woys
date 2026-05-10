# v0.5.2 — Pacat underrun ("برفک") investigation

> **NOTE: Historical investigation snapshot, captured at v0.5.2 (2026-05-05).**
> The pacat → pw-cat switch shipped here was further superseded by the
> native PipeWire output client (`woys-pw-out`) introduced in v0.9.0-rc1.
> The current canonical reference is `docs/05-perf.md` and `LESSONS.md`
> for chronology. Don't act on this doc as if it reflects current state.

> Pre-fix root-cause trace. After v0.5.1's soxr resampler removed the
> aliasing scratches, the user reported a different artifact in
> Telegram — Persian word **"برفک"** (TV-static crackle): rapid
> sub-millisecond gaps that sound like the audio is "disconnecting and
> reconnecting in like 0.0001 seconds" continuously. This is Hypothesis E
> from the v0.5.1 retro: PulseAudio output buffer underruns. Each
> underrun ≈ a brief silence ≈ a reconnection click; at fast cadence it
> reads as TV static.

## Pacat output buffering — what's currently configured

```python
# src/audio/engine.py @ _open_pacat
cmd = [
    pacat,
    "--playback",
    f"--device={self.cfg.sink_name}",
    f"--rate={self.cfg.sink_rate}",            # 48000
    f"--channels={self.cfg.channels}",         # 1  ← engine default
    "--format=float32le",
    f"--latency-msec={self.cfg.output_latency_ms}",   # 30  ← engine default *as of v0.5.2* (now 300; see v0.6.7)
    ...
]
```

`pactl` then logs (with `pacat -v`):

```
Buffer metrics: maxlength=4194304, tlength=46080, prebuf=38408, minreq=7680
```

`tlength=46080 bytes` at 2ch float32 48k = **240 ms target buffer**.
That's PulseAudio's ceiling for this stream — `--latency-msec=30` is
the *requested* latency, but the daemon settles where it can. With
the default of 30 ms, prebuf is set such that *any* engine stall above
~40 ms drains the buffer to zero → underrun.

The engine's measured wall-time per chunk is `~30 ms infer + 250 ms
chunk wait + ~30 ms write/flush + jitter`. The chunk wait is by design
(we read 250 ms of mic), but the **write/flush+jitter pair is the
underrun trigger** — every time it spikes, pacat sees a pause longer
than its remaining buffer.

## Five contributing causes (and how to confirm each)

### Cause 1 — `--latency-msec` set too low for the chunked-write cadence

Default 30 ms means PulseAudio picks a small target buffer.
With 250 ms chunks arriving every ~280 ms (chunk + infer), the buffer
empties between writes and refills only when the engine writes again.
Any 5 ms scheduler hiccup on either side = underrun.

**Confirmed** via the `Buffer metrics` line above: `tlength=46080`
≈ 240 ms is the daemon's actual choice; we're nowhere near 200 ms of
*useful* slack because the daemon also reserves prebuf.

**Fix** (Brief §3 Fix 1): bump the request to 200 ms; the daemon then
sizes the buffer to ~400–500 ms and the engine's per-chunk jitter
sits well within slack.

### Cause 2 — Engine main loop is the writer (synchronous flush)

```python
pacat_proc.stdin.write(out48.tobytes())   # blocking on full pipe
pacat_proc.stdin.flush()                   # blocking syscall
# ... next iteration: read mic, infer, resample, write again
```

If pacat's reader thread is preempted, the OS pipe buffer fills and
`write()` blocks. While it's blocked, the engine isn't reading the next
mic chunk — so when it eventually resumes, the next mic read hits
the input stream's overflow, the inference hits a tight deadline, and
the *next* pacat write arrives late → underrun.

Symptom: tail-latency (`avg_total_ms`) creeps upward over a long run.

**Fix** (Brief §3 Fix 2): writer thread + bounded queue (size 8). The
main loop only enqueues; a daemon thread does the blocking write. A
full queue = an early warning the engine is too slow, which we expose
as an xrun-proxy counter.

### Cause 3 — pacat death mid-session = silent failure cascade

Currently:

```python
if pacat_proc.poll() is not None:
    raise RuntimeError(f"pacat subprocess died (exit {pacat_proc.returncode})")
```

The engine raises out of the main loop and stops. The user sees `running
= False` and `last_error = "RuntimeError: pacat subprocess died"`. They
have to toggle the engine to recover.

**Fix** (Brief §3 Fix 3): watchdog thread polls `pacat.poll()` every
50 ms. On death: log, spawn replacement, swap stdin handle on the
writer thread. Recovery target: ≤ 100 ms gap.

### Cause 4 — Chunk-write timing jitter

Variance in inter-chunk write intervals is what feeds underruns. With
SCHED_OTHER and no CPU pinning, the engine thread can be migrated
across cores mid-chunk; the L2/L3 cache miss costs add up to ~5–10 ms
of jitter on this i7-10750H (6P+6HT).

Two mitigations:
- **(a)** `os.sched_setaffinity(0, {core})` — pin the engine thread
  to one P-core. Unconditional (no caps required).
- **(b)** `os.nice(-10)` or SCHED_FIFO — only with `CAP_SYS_NICE`.
  Behind a config flag, OFF by default.

Brief §6 explicitly forbids enabling SCHED_FIFO by default.

### Cause 5 — Mono engine, stereo sink → in-graph upmix

Engine writes 1-channel float32; null-sink is `channels=2`. PipeWire
runs a 1→2 channel converter inside the graph on every chunk. Cheap
but not free, and adds another scheduling node that can stall.

**Fix** (Brief §3 Fix 5): engine resamples to **stereo** float32le
@ 48 kHz before writing. Pacat becomes a dumb pass-through.

## How we'll detect underruns going forward

`pacat -v` emits `Stream underrun.\n` to stderr on every callback that
finds the buffer empty. Currently we set `stderr=subprocess.DEVNULL`
and lose the signal entirely.

v0.5.2 captures stderr to a `subprocess.PIPE` and runs a daemon reader
thread that increments `EngineStats.xruns` on every `Underrun` token.
The TUI shows the counter live; the new `woys diag`
subcommand prints it after a 10 s self-test.

That's the closest thing to a true xrun count we can get without
reaching into PipeWire internals via `pw-dump` (heavy, racy).

## Verification budget for the fix (Brief §4)

| Test | Threshold |
|---|---|
| `test_no_pacat_underruns_in_30s` | `engine.stats.xruns == 0` over 30 s of synthetic input |
| `test_chunk_write_jitter_under_5pct` | std dev of `_writer_intervals_ms` < 5 % of `chunk_seconds * 1000` |
| `test_long_run_no_drift` | over 5 min: `avg_total_ms_end / avg_total_ms_start < 1.05`, `pacat_restarts == 0` |

All three must pass before declaring the fix done. **Real-mic + Telegram
verification by the user is still the final gate** — synthetic input
hitting the same scheduler pattern can mask real-world variance.

## Out of scope for v0.5.2

- Native PipeWire client (libpipewire) — too much surface area for one bug
- Replacing pacat entirely with sounddevice's PortAudio→PipeWire path
  for the output side — sounddevice's blocking-write semantics are
  similar and we'd lose the dedicated subprocess isolation
- SCHED_FIFO by default — capability requirement, brief §6 forbids

---

## Update — what actually fixed it: `pw-cat`, not `pacat` tuning

The brief assumed pacat tuning would suffice. It doesn't. The validation
test (`tests/test_pacat_health.py::test_no_pacat_underruns_in_30s`) ran
end-to-end with progressively higher `--latency-msec` settings:

| `--latency-msec` | negotiated `tlength` | underruns / 30 s |
|---:|---:|---:|
| 30 (v0.5.1) | ~50 ms | dozens (the original bug) |
| 200 | 240 ms | 43 |
| 500 | 329 ms | 45 |
| 1000 | 829 ms | 44 |
| 2000 | 1828 ms | 40 |

Even at 2 s of buffer, pacat reported ~1.4 underruns per second on the
exact same sink + write pattern. The root cause: PulseAudio's prebuf
semantics. Each underrun rewinds the stream, but `prebuf ≈ tlength` so
the playback head can't move forward until the buffer refills past
prebuf again. The buffer level oscillates near the underrun threshold
because our 250 ms write cadence equals PA's drain rate; any jitter
makes the buffer dip below `minreq ≈ 20 ms` and triggers the callback.
Larger `tlength` doesn't change the oscillation amplitude — only the
ceiling.

Switching the playback subprocess from `pacat` to `pw-cat`:

| backend | `--latency` | underruns / 15 s | total wall latency |
|---|---:|---:|---:|
| pacat | 1000 ms | ~22 | ~1300 ms |
| **pw-cat** | **100 ms** | **0** | **~420 ms** |

`pw-cat` speaks PipeWire natively. PipeWire's graph is pull-driven: the
sink consumer pulls samples in real-time quanta and the source (us, via
pw-cat) hands them over from a small buffer. Bursty 250 ms writes from
upstream don't bounce a prebuf threshold because there's no prebuf
threshold — the graph simply forwards what arrives. We get the safety
net of the new writer thread + watchdog AND a clean audio output.

`pacat` is kept as a fallback for hosts that have pipewire-pulse but
not pipewire-tools, which is rare on CachyOS but possible elsewhere.
The xrun counter is still wired (parses pacat -v stderr) but only ever
non-zero on the pacat path. With pw-cat the user-facing health signal
is `queue_full_events` (writer outpaced) and `pacat_restarts` (process
death).
