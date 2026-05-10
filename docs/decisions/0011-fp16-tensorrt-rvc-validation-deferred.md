# 0011 — TensorRT FP16 for RVC is disabled until per-voice quality validation

## Decision

`use_tensorrt = False` (decision 0001). Even if a future ORT/TRT
version fixes the v0.8.1 STFT and binding issues, we do not flip TRT
on globally — we require per-voice spectral-QA showing cosine
similarity ≥0.95 vs CUDA EP across all soxr shapes for each foundation
voice and the user's daily-driver voices before flipping any default.

## Status

`provisional` — re-evaluate when ORT bumps to a TRT-favoured version.

## Context

The v0.8.1 TRT pivot (decision 0001) measured RVC FP16 inference
under TensorRT 10.16 and found mathematically wrong output: cosine
similarity 0.02 / 0.44 / 0.48 / 0.28 across the four soxr shapes,
target ≥0.95. This is an *output-domain* failure, not a *speedup*
failure — the 1.04-1.87× speedup is real, but the model is producing
a different signal. Some voices may degrade gracefully, others may
fall to 0.02 cosine sim (effectively unrelated audio). FP16 TRT for
RVC therefore cannot be flipped on a single-voice "looks fine to me"
test; it needs a per-voice gate.

## Decision

TRT-FP16-RVC stays disabled and requires per-voice cosine-similarity
≥0.95 validation as a gate before any future flip; this gate is
voice-specific because a global "TRT works" claim is not equivalent
to "TRT works for *this* voice."

## Alternatives considered

- **Flip on globally if any future ORT/TRT version "passes a smoke
  test"** — exactly the failure shape v0.8.1 tested into. A single
  voice passing doesn't validate the foundation set or community
  voices.
- **Flip on per-voice** — viable but requires the foundation voice
  set + each user-imported voice to run through the gate at import
  time. Implementation cost: ~50 LOC added to `src/woys/models.py`'s
  voice-discovery path.
- **Never flip; remove TRT EP code** — reduces surface but loses
  the v0.8.1 instrumentation that made the original measurement
  cheap to reproduce.

## Rationale

Cosine-similarity 0.02-0.48 across soxr shapes is not a "TRT is
slightly less accurate" finding — it's a "TRT is producing a
different signal." We do not know in advance which voices in a
user's library will degrade gracefully and which will collapse.
Flipping the TRT default on without per-voice gating invites the
"sounds fine for amitaro_v2_16k, sounds completely broken for the
voice the user actually uses" failure mode. The gate-per-voice
requirement makes future re-evaluation safer at modest engineering
cost.

## Trade-offs accepted

Even if TRT becomes correct on a future stack, we pay the cost of
running the voice-gate at import time (or at first use) for every
voice. This is a small one-time cost per voice (~10 s of inference
on a 1-second test clip). The TRT EP code path stays in tree as
~30 LOC of session-options config plus the `use_tensorrt` field.

## Re-litigation triggers

- ORT bumps to a version that explicitly fixes the TRT
  STFT-importer Float32 constraint AND the int64 binding warnings
  documented in v0.8.1. Both must be in the release notes.
- A future RVC ONNX exporter ships with shape inference and
  int64→int32 conversion that resolves the v0.8.1 export-time
  issues at the model level.
- Hardware moves to a TRT-favoured GPU class (Ada / Hopper /
  RTX 50xx) where the EP's fp16 path stops regressing on RVC
  architectures generally.

## Test plan if re-evaluated

1. Verify ORT/TRT release notes list the v0.8.1 STFT-importer fix.
2. Re-export RVC with current ONNX exporter; verify int64 bindings
   are gone.
3. For each foundation voice (decision 0012's list) + each
   community-trained voice in the project's catalogue, run a
   1-second test clip through CUDA EP and TRT EP, compute cosine
   similarity per soxr shape (4 shapes per voice).
4. Flip default per-voice only where cosine sim ≥0.95 holds for all
   four shapes. Otherwise voice stays on CUDA EP.
