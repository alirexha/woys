# 0001 — ONNX Runtime CUDA EP, with TensorRT EP retained as opt-in

## Decision

woys ships ONNX Runtime CUDA Execution Provider as the only enabled
inference backend; the TensorRT EP code path is retained but
`use_tensorrt = False` by default.

## Status

`accepted`

## Context

`PROJECT_BRIEF.md` §12 lists "Replacing ONNX Runtime" in the FORBIDDEN
list. The brief is silent on TensorRT specifically because TRT is not
a replacement for ORT — it is an *EP* under ORT. The v0.8.1 release
attempted a TRT pivot to claw back the brief's <80 ms latency target.
The empirical evidence is in `LESSONS.md` §23-§33 and the docstring
on `EngineConfig.use_tensorrt` at `src/audio/engine.py:509-542`.

## Decision

ORT CUDA EP is the production path; the TRT EP code path stays in
tree, default-off, for future re-evaluation.

## Alternatives considered

- **TensorRT EP under ORT (v0.8.1 pivot)** — fails on this stack:
  RMVPE-fp16 STFT importer error (TRT 10.16 wants Float32 input);
  RVC produces mathematically wrong output (cosine sim 0.02-0.48
  vs CUDA EP across the four soxr shapes; target ≥0.95).
- **CPU EP only** — no GPU work, but 5-10× slower than CUDA EP at
  warm steady state; unviable for real-time RVC.
- **Replace ORT with a custom inference runtime (TVM, custom CUDA
  C++)** — explicitly forbidden by `PROJECT_BRIEF.md` §12 and by
  `the project notes`'s "Things to never do" list.

## Rationale

The TRT pivot was tested with measurement, not assumed: the v0.8.1 A/B
showed (a) RMVPE-fp16 fails to load under TRT 10.16's STFT importer,
which requires Float32 input (RMVPE has been auto-promoted to fp16
since v0.3.0); (b) RVC initialises but produces output with cosine
similarity 0.02 / 0.44 / 0.48 / 0.28 vs CUDA EP across the four soxr
shapes, target ≥0.95 — i.e., the model is mathematically wrong even
though it runs; (c) speedup ignoring correctness is 1.04-1.87× on cv
only, below the v0.8.1 pivot's 1.5-3× threshold. The IO-binding
attempt (`LESSONS.md` §23) was likewise a null result: 200-pass bench
showed -1.6% / -0.8% deltas at chunk=0.15s / 0.10s — within noise.
The real latency lever turned out to be the GPU clock + keepalive
synergy (`LESSONS.md` §31-§34, +36× underrun reduction in user
listener test), not the EP swap.

## Trade-offs accepted

The TRT EP code path remains in tree as ~30 lines of session-options
config plus the `use_tensorrt` field. That's review surface for a
default-off path that has never shipped active. Net cost: low
(disabled paths are easy to grep for and prune later), kept because
the v0.8.1 measurement was preserved as an inline docstring and a
later ORT/TRT version may re-validate it cheaply.

## Re-litigation triggers

- ORT bumps to a version that fixes the TRT STFT-importer Float32
  constraint (track ORT 1.23+ release notes).
- RVC ONNX exporter ships with shape inference + int64→int32 changes
  that fix the v0.8.1 binding-warning class.
- Hardware moves to a TRT-favoured class (RTX 50xx, Ada/Hopper) where
  the EP's INT8/FP16 path stops regressing accuracy.
- Any future re-attempt MUST re-measure cosine similarity vs CUDA EP
  across all four soxr shapes before flipping the default.
