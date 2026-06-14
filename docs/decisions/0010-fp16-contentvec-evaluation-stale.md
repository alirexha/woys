# 0010 — fp16 ContentVec evaluation is stale

## Decision

ContentVec runs in fp32 by default; the v0.2.0 fp16 ContentVec
evaluation has not been re-measured against current RVC v2 voices,
and we treat fp16 ContentVec as untested on the current stack.

## Status

`provisional` — re-measure if engine VRAM headroom becomes a
constraint.

## Context

v0.2.0 evaluated fp16 ContentVec against the project's foundation
voice set at the time. The result was: VRAM saving was real but
modest (the model is small relative to the 1.35 GiB total footprint),
and quality concerns warranted keeping fp32 as the default. Since
v0.2.0, the foundation voice set has changed (community voices
catalogue grew, RVC v2 export pipeline updated), the soxr resampler
generation chain was rewritten (v0.5.1, `LESSONS.md` §12), and RMVPE
was wrapped + auto-promoted to fp16 (v0.3.0). ContentVec did not
follow.

## Decision

Default fp32; do not flip until a per-voice spectral-QA evaluation
on current RVC v2 voices verifies parity within the listening test's
detection floor.

## Alternatives considered

- **Default fp16 ContentVec** — saves VRAM, possibly a small
  inference speedup. But the v0.2.0 quality finding has not been
  re-validated on the current voice catalogue. Flipping the default
  on a stale measurement is exactly the failure mode `LESSONS.md`
  §23 warned against ("any X% perf-win claim on a benchmarked-once
  result deserves a 200-pass second confirmation").
- **Re-evaluate now** — would resolve `provisional` to `accepted`
  in either direction. Estimated 6 h per the review notes
  (per-voice spectral-QA across the foundation voice set).
- **Drop fp16 ContentVec code path entirely** — reduces surface area.
  But preserves an option for a future evaluation where the saving
  matters (e.g., 4 GB GPU class).

## Rationale

The honest position is that fp16 ContentVec was tested once, three
years ago in project time, on a different voice set, and the result
was conservative ("stay on fp32"). We cannot promote that to a
present-day "we picked fp32 because fp16 hurts quality" claim
without re-measurement. The reverse claim ("fp16 is fine now") is
equally unsubstantiated. The default of fp32 carries no measurable
runtime penalty on this hardware (engine VRAM is 1.35 GiB total —
ContentVec is a small fraction of that), so the conservative default
is also the cheap default.

## Trade-offs accepted

VRAM headroom on 4 GB GPUs is tighter than it could be — relevant
for users running woys alongside CS2 on shared discrete VRAM.
Inference speedup, if any, is unclaimed. The fp16 code path is dead
weight in tree until re-evaluation runs.

## Re-litigation triggers

- A user reports VRAM-pressure OOM with current default voices
  alongside CS2 / streaming — ContentVec fp16 becomes the easy
  saving to evaluate first.
- A new RVC variant or ContentVec replacement requires fp16-only
  inference for export-pipeline reasons.
- The foundation voice set is regenerated (decision 0012's voice
  catalogue updates) — re-measure as part of that work.

## Test plan if re-evaluated

1. Pick 3-5 voices from the current foundation set + user's daily
   driver (e.g., `e_girl`).
2. Run identical input audio through fp32 ContentVec → RVC and
   fp16 ContentVec → RVC.
3. Compute cosine similarity of RVC outputs on matched chunks; gate
   at ≥0.95 across all four soxr shapes (mirroring the
   `LESSONS.md` §23 TRT-EP gate methodology).
4. If ≥0.95: flip default. If not: document the failing voices
   and what cosine sim they hit; status stays `provisional`.
