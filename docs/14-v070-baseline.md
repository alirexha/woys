# v0.7.0 — pre-state latency baseline (v0.6.10)

> **NOTE: Historical investigation snapshot, captured at v0.7.0 (2026-05-06).**
> Recommendations like "drop chunk_seconds 0.25 → 0.10" and
> "drop output_latency_ms 300 → 80" are stale: chunk_seconds re-stabilized
> at 0.25 in v0.12.4 (listener A/B), and output_latency_ms shipped at 280
> since v0.7.0-rc3. The current canonical reference is `docs/05-perf.md`
> and `LESSONS.md` for chronology. Don't act on this doc as if it reflects
> current state.

Captured 2026-05-06 against the just-shipped v0.6.10 build (which is
v0.6.9 minus the jennie voice — no engine changes between those tags).
All numbers are from `scripts/bench_inference.py` on this machine
(RTX 2070 Mobile, i7-10750H, ORT-CUDA, cuDNN EXHAUSTIVE search).

The benchmark mirrors `RealtimeEngine._infer()` exactly: cv → rmvpe → rvc
on synthetic noise input, no streaming wrapper, no SOLA crossfade, no
pacat, no audio I/O. Per-stage timings come from `EngineStats.last_*_ms`
which the engine populates on every chunk.

> **Methodology note.** Synthetic noise is the easiest input: pitch
> tracker hits voiced fast, no NaN bursts, no silence-gating decisions.
> Real speech can be 1.3–2× slower in the tail. Treat these numbers as
> a lower bound on `_infer()` cost, not a typical case. The brief's
> "~96ms avg, ~456ms tail" came from real-speech instrumentation in
> v0.6.7 / v0.6.9 (LESSONS §16), so the gap below is real input
> sensitivity, not benchmark error.

## Per-stage inference (synthetic input, 60 timed passes after 8 warmups)

### amitaro_v2_16k (smallest model, ~63 MB ONNX, fp32 RVC)

| chunk_seconds | ctx_ms | cv avg | rmvpe avg | rvc avg | **TOTAL avg** | TOTAL p99 | TOTAL max |
|---|---|---|---|---|---|---|---|
| 0.05 | 100 | 3.64 | 7.36 | 7.87 | **18.88** | 19.36 | 19.40 |
| 0.10 | 100 | 3.67 | 7.49 | 9.26 | **20.43** | 21.10 | 21.20 |
| 0.10 | 0   | 3.52 | 7.19 | 9.58 | **20.30** | 20.83 | 20.86 |
| 0.15 | 100 | 3.83 | 7.49 | 9.30 | **20.63** | 21.19 | 21.21 |
| 0.20 | 100 | 3.71 | 7.64 | 9.27 | **20.63** | 21.26 | 21.28 |
| 0.25 | 100 | 4.45 | 8.02 | 11.06 | **23.55** | 24.06 | 24.06 |

### Per-voice at chunk=0.25 (default, ctx=100)

| voice | rvc avg | TOTAL avg | TOTAL p99 |
|---|---|---|---|
| amitaro_v2_16k | 11.16 | 23.86 | 30.01 |
| catwoman       | 20.30 | 33.02 | 33.90 |
| e_girl         | 19.42 | 32.13 | 32.94 |
| megan_fox      | 19.29 | 32.08 | 32.55 |

### Heavy voice at small chunk

| voice | chunk | TOTAL avg | TOTAL p99 | TOTAL max |
|---|---|---|---|---|
| catwoman | 0.05 | 24.85 | 26.32 | 26.67 |
| catwoman | 0.10 | 26.57 | 29.13 | 29.19 |

## Mic-to-app latency decomposition (theoretical)

| Stage | v0.6.10 default | Fixed by | Notes |
|---|---|---|---|
| Mic → engine input (chunk wait) | **250 ms** | `chunk_seconds * 1000` | Fully under our control — biggest single lever |
| Engine inference (`_infer()`) | ~25 ms avg | model + chunk + ctx | Real-speech tail can hit 60–120 ms (LESSONS §16) |
| Streaming wrapper (resample, SOLA, queue) | ~5–15 ms | engine path overhead | Unmeasured here; goes through `_process_streaming_16k` |
| pacat output buffer | **300 ms** | `output_latency_ms` | Set in v0.6.7 to absorb engine jitter; pre-v0.6.9 fixes |
| PipeWire mixer + Discord codec | ~30 ms | Out of woys's control | RTP framing, Opus packetization |
| **TOTAL wall-clock** | **~605 ms** | | Brief estimated 450; either way, dominant costs are chunk + buffer |

## Where the latency lives

The brief identified pacat (300 ms) and inference (~96 ms) as the two
attack surfaces. The data above tells a different story:

1. **chunk_seconds (250 ms)** is the largest single lever. Dropping it
   to 0.10 saves 150 ms outright with no quality risk *if* inference
   stays under the new budget. Per-voice p99 at chunk=0.10 is 21–29 ms
   on synthetic input — comfortable headroom even doubled for real
   speech.

2. **output_latency_ms (300 ms)** is the second lever. Was set high in
   v0.6.7 to absorb engine inference jitter. Post-v0.6.9 fixes (input
   gate, NaN sanitization, dropped-chunk recovery) the jitter floor is
   lower; the 300 ms ceiling probably has slack.

3. **Inference (25 ms)** is *not* the bottleneck. IOBinding +
   cuDNN-heuristic + fp16-contentvec might shave 5–10 ms total. Worth
   doing for tail-latency reduction (and proportionally bigger at small
   chunks), but won't materially move total mic-to-app.

4. **TensorRT, CUDA graphs, custom kernels** would chase the remaining
   25 ms but at substantial complexity cost. Out of scope for v0.7.0.

## Plan (priority-reordered from brief §3 based on data)

1. ✅ Document baseline (this file)
2. Implement ORT IOBinding for cv → rmvpe → rvc — proportionally bigger
   at small chunk_seconds; needed before chunk reduction is meaningful
3. Switch cuDNN EXHAUSTIVE → HEURISTIC — reduces tail jitter (helps the
   chunk-budget calculus)
4. Drop `chunk_seconds` default from 0.25 → 0.10 (≈ −150 ms wall-clock)
5. Drop `output_latency_ms` default from 300 → 80 ms (≈ −220 ms;
   verify cuts/min stays acceptable)
6. fp16 ContentVec quality audit per voice (cosine ≥ 0.92 ships fp16)
7. Pre-warm broader shape coverage so chunk reduction doesn't bring back
   first-N-chunks slow path
8. Document final floor + write LESSONS §19 retrospective
9. Bump 0.7.0-rc1, ship for user CS2 verification (NOT auto-tag)

Targets after each step are documented in §"Post-change deltas" below.

## Post-change deltas

### Technique-by-technique

| Technique | Expected (brief §3) | Actual delta | Decision |
|---|---|---|---|
| ORT IOBinding for cv → rmvpe → rvc | −30 to −50 ms | **−0.3 ms (−1 %)** within noise | **Skipped.** ORT 1.20+ already handles CUDA copies efficiently; the CPU numpy ops between sessions force data to host anyway. Bench in `scripts/bench_iobinding.py`. |
| cuDNN EXHAUSTIVE → HEURISTIC | −5 ms avg + tail | **steady-state same** (within 1 % across 0.10/0.25 chunks) | **Shipped.** No measurable steady-state cost, removes the 50–100 ms cold-start autotune lump per shape. Unblocks lower chunk_seconds. |
| fp16 ContentVec | −10 to −20 ms | not measured | **Skipped.** Inference is not the bottleneck (50 ms threading penalty dominates); 1–3 ms here doesn't move the user-visible needle. v0.2.0 LESSONS §6 also already showed cosine sim 0.75 audibly degraded. |
| Pre-warm broader shape coverage | (defensive) | n/a | **Skipped for v0.7.0.** Was needed to unblock chunk_seconds=0.10 — but 0.10 is unstable for other reasons, so the prep is moot. |
| Drop chunk_seconds 0.25 → 0.15 | (input-side lever) | **−100 ms input wait** | **Shipped.** Sweep showed chunk=0.10 has 13–42 % late chunks (engine inference 77–98 ms with 50 ms threading tax leaves only 22 ms headroom under a 100 ms budget); chunk=0.15 has 0 late chunks. |
| Drop output_latency_ms 300 → 80 | (output-side lever) | **−220 ms output buffer** | **Shipped.** Paired with prefer_pw_cat=True flip. Empirical sweep (catwoman + pw-cat): queue_full_events=0 across output_latency 50/80/100/150 ms. 80 ms = one PipeWire quantum of safety margin over the ~43 ms quantum default. |
| Backend pacat → pw-cat (revert v0.6.7) | (not in brief) | xruns 65/15s → 0/15s | **Shipped.** Pacat-stderr underrun parser fires constantly on this PipeWire version regardless of latency setting; pw-cat is silent. v0.6.7's reason for switching back (per-quantum gaps with 250 ms bursty writes) doesn't apply at 150 ms writes + v0.6.9 stability fixes. |

### Wall-clock latency change

| Stage | v0.6.10 | v0.7.0 | Delta |
|---|---|---|---|
| Mic → engine input (chunk wait) | 250 ms | **150 ms** | −100 ms |
| Engine inference (real path) | ~80 ms | ~80 ms | 0 ms (threading tax not addressed) |
| pacat / pw-cat output buffer | 300 ms | **80 ms** | −220 ms |
| PipeWire mixer + Discord codec | ~30 ms | ~30 ms | 0 ms (out of reach) |
| **TOTAL** | **~660 ms** | **~340 ms** | **−320 ms (−48 %)** |

The 80 ms inference is unchanged — it's the **structural floor on this
hardware** until the engine threading penalty is addressed (see
LESSONS §19). Every other lever is at its measurable minimum.

### Floor and what's blocking it

- **chunk_seconds = 0.15** is the floor. Lower (0.10) puts engine
  inference inside the per-chunk budget by < 25 ms, and 13–42 % of
  chunks miss budget across both light and heavy voices. Closing the
  gap requires reducing the 50 ms threading penalty (Python GIL
  contention with audio threads) — that's a v0.8.x project.
- **output_latency_ms = 80** is the floor for pw-cat at this PipeWire
  version. 50 ms also worked in the sweep but leaves no margin; 80 ms
  gives one quantum of slack at no perceived latency cost.
- **inference 80 ms (real path)** vs 30 ms (standalone bench) is the
  threading-penalty bottleneck. py-spy / perf-profile work to find the
  source is the highest-leverage next experiment.
- **Discord codec ~30 ms** is the immutable downstream floor. Opus
  framing + RTP packetization is what it is.

Net target: any future <300 ms total latency would require closing the
50 ms threading penalty AND keeping the rest of the wall-clock as-is.
180 ms is the theoretical floor on this hardware (150 chunk + 30
inference + 0 buffer + 30 codec), unrealistic but a useful north star.

### Real-world verification (gating v0.7.0 tag)

Per brief §7 + §8 step 7, v0.7.0 is **NOT tagged** by this development
work. v0.7.0-rc1 ships with these defaults; the user runs CS2 callouts
+ Discord voice chat for ≥ 30 minutes; the audible verdict is the gate.
On positive feedback, tag v0.7.0. On "still feels laggy", investigate
further before tagging.
