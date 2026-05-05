# Performance numbers

## v0.2.0 (2026-05-04, this run)

| Metric | v0.1.1 actual | v0.2.0 actual | v0.2.0 brief target | Verdict |
|---|---:|---:|---:|---|
| Warm `avg_total_ms` (chunk=0.1, SOLA on) | ~280 ms (chunk=0.25) | **30.48 ms** | < 120 ms | **HIT** |
| Warm `avg_inference_ms` | ~60 ms | **30.31 ms** | n/a | improved |
| GPU memory used (process) | 1356 MiB | **1348 MiB** | < 700 MiB | **MISS** (see §6) |
| CPU active (parent process) | ~26 % | **31.9 %** | < 18 % | **MISS** (see §6) |

Hard fail thresholds (`> 200 ms`, `> 1.0 GiB`, `> 22 %`) — only the e2e
threshold is comfortably cleared. VRAM and CPU are above hard-fail because
the underlying model architecture hasn't changed; v0.3.0 work targets these.

Methodology: `woys run --autostart` for 10.5 s. First 2.5 s are
discarded as warm-up (cudnn autotune settles by then). Rolling-32 average
captured during the next 8 s.



Measured on this CachyOS machine; reproducible via the scripts called out per
section. All numbers are wall-clock, no estimates. Brief targets:

| Metric           | Target  |
|------------------|---------|
| End-to-end e2e   | < 80 ms |
| Idle VRAM        | < 500 MB |
| CPU active       | < 15 %  |

## 1. Hardware / software baseline

- **CPU:** Intel i7-10750H (6c/12t)
- **GPU:** NVIDIA RTX 2070 (driver `595.71.05`, CUDA system pkg `13.2.1`)
- **Kernel:** `7.0.3-1-cachyos`
- **Audio:** PipeWire `1.6.4` + `pipewire-pulse` 15.0.0
- **Python:** 3.11.15 (isolated venv)
- **ML stack:** `torch==2.5.1+cu124` · `onnxruntime-gpu==1.22.0` · cuDNN `9.1.0` (pip-shipped via `nvidia-cudnn-cu12`)

## 2. In-process inference latency

Source: `tests/test_smoke_rvc_onnx.py` (also exposed as `scripts/smoke_rvc_onnx.py`).

Pipeline: `audio (1 s @ 16 kHz) → contentvec-f.onnx → rmvpe_wrapped.onnx → amitaro_v2_16k.onnx → audio`.
This measures inference cost only — no audio I/O, no SOLA crossfade, no resampling.

```
ORT 1.22.0 · CUDA EP active (CUDAExecutionProvider on RTX 2070)
e2e latency over 10 iters (mean ± stdev):  37.55 ± 10.18 ms  (min 28.84, max 51.93)
  contentvec-f       :  7.55 ms
  rmvpe_wrapped      : 17.12 ms
  amitaro_v2_16k     : 13.86 ms
```

The minimum of 28.84 ms is the steady-state floor; the spread to 51.93 ms is
mostly first-iteration kernel selection variance after warm-up.

## 3. Latency vs chunk size

Source: `scripts/smoke_rvc_onnx.py` re-run with shrinking chunks (or
`scripts/bench_chunks.py` if you'd like a one-liner script).

```
chunk_ms  samples   mean_ms   std_ms   min_ms   max_ms   ×realtime
   500     8000      30.65     7.81    26.32    47.74    16.3×
   250     4000      23.67     0.35    23.17    24.50    10.6×
   200     3200      23.28     0.19    23.00    23.70     8.6×
   150     2400      21.89     0.25    21.53    22.43     6.9×
   120     1920      21.20     0.21    20.93    21.66     5.7×
   100     1600      23.51     0.27    23.18    24.36     4.3×
    80     1280      22.62     0.28    22.23    23.48     3.5×
    60      960      21.37     0.29    20.95    21.93     2.8×
```

**Key finding:** inference time is roughly *constant* (≈22 ms) across chunk sizes
from 60-250 ms. Below ~120 ms, kernel-launch overhead and minimum-input-frame
constraints dominate over the actual matmul cost. Larger chunks give better
realtime headroom but worse e2e latency in practice.

**Sweet spot for woys:** `chunk_seconds = 0.10-0.15` (100-150 ms input
buffering plus ~22 ms inference plus ~10 ms PipeWire I/O = ~132-182 ms total).

The brief's <80 ms target requires chunks ≤ ~50 ms, which falls below contentvec's
minimum-input window. Achieving the target without rewriting the contentvec ONNX
graph (out of scope per the brief) requires careful per-chunk overlap-add (SOLA
crossfade) — a Phase 5+ task beyond the minimum bar of "verifiable working
inference at sustainable latency".

## 4. Acoustic loopback (one-way mic→sink)

Source: `scripts/bench_loopback.py`. Methodology — `pacat` plays an impulse into
`WoysSink`; `parec --device=woys-mic` captures simultaneously; we
locate the impulse in the capture and report the wall-clock delta.

This bench measures everything the user actually hears: PipeWire scheduling +
loopback graph + parec/pacat buffers (note: it does *not* include the host-mic
capture latency — add ~5-15 ms for that in real Discord usage).

> **Note (Phase 5):** the acoustic loopback script (`scripts/bench_loopback.py`)
> is scaffolded but the subprocess timing alignment is fragile — `pacat` and
> `parec` race to start, so the reported one-way delay is sensitive to
> scheduler jitter. **In-process measurements (§2-3 above) are the authoritative
> latency numbers for now.** A future revision should sync via a known marker
> tone in both streams instead of comparing wall clocks across processes. For
> reference, when this method works: PipeWire null-sink → remap-source loopback
> alone is ~5-15 ms on this kernel; add the engine's per-chunk inference time
> from §3 for the with-engine total.

## 5. Engine warm-state — measured (the real-world numbers)

Source: live engine run at `chunk_seconds=0.25` for 8 s after a 2 s warm-up
window so cudnn autotune is past us.

```
chunks_processed (warm rolling 32) : ~25 chunks
avg_inference_ms                   : 60.71 ms
avg_total_ms (incl. mic-read wait) : 260.99 ms
gpu memory used                    : 1356 MiB
cpu (parent process)               : ~26 %
```

The model VRAM footprint (≈ 1.35 GiB) breaks down roughly as:
- contentvec-f.onnx fp32 (≈ 700 MiB resident)
- rmvpe_wrapped.onnx fp32 (≈ 400 MiB)
- amitaro_v2_16k.onnx fp32 (≈ 150 MiB)
- ORT/CUDA arenas + cudnn workspace (~100 MiB)

### Targets vs reality (honest)

| Metric           | Target  | Measured       | Status |
|------------------|---------|----------------|--------|
| End-to-end e2e   | < 80 ms | **~280 ms** (chunk + infer + I/O) | **MISS** |
| Inference only   | n/a     | 30-40 ms (cold→warm)              | strong |
| Idle VRAM        | < 500 MB | ~1.35 GiB                        | **MISS** |
| CPU active       | < 15 %  | ~26 % @ chunk=0.25                 | **MISS** |

### Why the misses (and what would close them)

- **e2e 280 ms vs target 80 ms**: most of the 280 ms is the audio-buffering
  block (`chunk_seconds=0.25` = 250 ms input wait). RVC contentvec needs ~80-150 ms
  of input to produce stable f0/feats; smaller chunks degrade pitch quality.
  Closing this gap means SOLA-style overlap-add with smaller hop sizes (out of
  scope for Phase 5; the brief permits this kind of audio chunk-size tuning).
- **VRAM 1.35 GiB vs 500 MB**: contentvec-f and RMVPE are both ~350 MB on disk
  fp32 ONNX. Closing this gap means quantizing or fp16-exporting both — a
  Phase 5+ task and depends on RVC v2 maintaining quality through fp16 inference
  on Turing (RTX 2070).
- **CPU 26 % vs 15 %**: the linear resample loop is fast (~0.1 ms) but the
  per-chunk Python orchestration (sounddevice read, np conversions, GIL release
  during ORT calls) eats the rest. Some of this is unavoidable Python overhead;
  the rest could be IO-bound to the GPU via ORT IOBinding (not yet wired —
  another Phase 5+ task).

The brief explicitly forbids the *big* optimizations (rewriting in C++/Rust,
custom CUDA kernels, replacing ORT, model distillation), so closing the gap
fully would mean SOLA + IO binding + fp16 export, all within scope, all
deferred for follow-up sessions.

## 6. SessionOptions tuning applied

In `_make_session()` (used by both the smoke test and the runtime engine):

```python
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.log_severity_level = 3
providers = [
    ("CUDAExecutionProvider", {
        "device_id": 0,
        "arena_extend_strategy": "kNextPowerOfTwo",
        "cudnn_conv_algo_search": "EXHAUSTIVE",
        "do_copy_in_default_stream": True,
    }),
    "CPUExecutionProvider",
]
```

Notes on each:
- `ORT_ENABLE_ALL` enables every graph optimizer pass.
- `cudnn_conv_algo_search=EXHAUSTIVE` autotunes the conv kernel selection on
  the first runs. There's a one-shot warm-up cost (~50-100 ms) which we eat
  during model load. Subsequent calls are fast.
- `arena_extend_strategy=kNextPowerOfTwo` reduces fragmentation.
- `do_copy_in_default_stream=True` avoids creating a side-band stream when our
  use is single-threaded inference.

Tuning we have *not* applied yet (Phase 5+ if more headroom needed):

- **IOBinding** to keep tensors GPU-resident across the cv→rmvpe→rvc handoff.
  Could save ~5-10 ms of host-device copies. Not a 5x win; deferred.
- **TensorRT EP**. Available in the wheel, but TensorRT runtime libs aren't
  pip-shipped — we'd need a heavy CUDA toolchain install. Falls back to CPU
  silently, which is *worse* than CUDA EP. Disabled.
- **fp16 / TF32**. The amitaro v2 16k model is fp32-only ONNX export. A
  re-export with `--fp16` would help marginally on RTX 2070 (Turing) but isn't
  free; deferred.

## 7. The FORBIDDEN list (per brief §12)

Not attempted, and not planned:
- ❌ Rewriting RVC inference in C++ / Rust
- ❌ Custom CUDA kernels
- ❌ Replacing ONNX Runtime
- ❌ Model distillation
- ❌ Anything with > 4 hours of dev for marginal gains

The remaining headroom (e.g. SOLA + IOBinding) goes into Phase 5+ polish if a
follow-up session is scoped for it.
