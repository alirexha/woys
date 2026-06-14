# 0008 — NSF state passing across chunk boundaries is out of scope

## Decision

We do not modify RVC's NSF (neural source-filter) source module to
preserve oscillator state across inference chunks; the chunk-period
periodicity it produces (`LESSONS.md` §36, §40-§42) is masked at the
audio layer (SOLA crossfade, chunk_seconds=0.25, optional RNNoise),
not eliminated at the model layer.

## Status

`deferred`

## Context

`LESSONS.md` §36 (v0.12.0 Phase 1) tested the hypothesis that RVC's
NSF source module phase-resets at every chunk boundary, producing a
periodic discontinuity at chunk_seconds (then 150 ms, 6.67 Hz). The
synthetic Phase 1 result was inconclusive due to a pw-record name-
fallback bug; §39 / §40 document the v0.12.2 methodology correction.
The corrected v0.12.x sweep (§41-§42) showed chunk_seconds and
sola_context_ms together drive the chunk-period spectral
autocorrelation peak to 0.000 — perceptually clean per the user's
listening review (§42). The mechanism's *audible attribution* to
NSF reset specifically was never confirmed; what we know is that
masking it at the audio layer works.

## Decision

The chunk-period periodicity is treated as something to mask, not
to surgically fix in the model.

## Alternatives considered

- **Custom NSF state-export ONNX** — re-export RVC with the NSF
  oscillator's hidden state as an additional ONNX input/output, then
  feed the previous chunk's terminal state as the next chunk's
  initial state. Estimated 12-24 h (per the review notes),
  requires per-voice re-export so every community-trained voice in
  the user's library would need re-running through the export
  pipeline.
- **Increase chunk_seconds to suppress the period** — already done
  (v0.12.4 picked 0.25, see decision implicit in `engine.py:176-208`).
  The +100 ms latency cost was accepted by the user listening test;
  further increases would push past the conversational threshold.
- **Mask with SOLA + RNNoise (current approach)** — drives the
  perceptual artefact below the user's detection threshold; see
  decisions 0006 and the SOLA contract in `src/audio/sola.py`.

## Rationale

Three constraints rule out the model-side surgery path right now.
First, scope: `PROJECT_BRIEF.md` §16 lists "training new RVC models"
as out of scope, and re-exporting voices requires owning each
voice's training checkpoint pipeline — which the project deliberately
doesn't. Second, ecosystem fragmentation: re-exporting only our
foundation voices (decision 0012) leaves user-imported community
voices on the unfixed code path. Third, the alternative was tested
and the audio-layer masking gets us to autocorrelation 0.000, which
the user's v0.12.4 listening review accepted as "the rhythm is
GONE." Spending 12-24 h on model surgery to chase residuals below
the user's detection threshold has a poor cost/benefit ratio versus
shipping anti-jitter (decision 0007) and the RNNoise chain
(decision 0006), which together produce listener-validated wins.

## Trade-offs accepted

We retain a small audible residual on sustained content for users
running with default settings; the v0.12.4 + decision 0006 (RNNoise
chain) combination puts that residual below the cuts/min detection
floor on this stack. Future RVC users who expect a chunk-state-
preserving model will not find one in woys; they'd have to fork and
re-export.

## Re-litigation triggers

- A future RVC variant ships with NSF state-passing as a first-class
  ONNX feature (no re-export burden on user voices).
- Listener tests on a new GPU class show the v0.12.4 residual
  becomes audible again (e.g., chunk_seconds tuning has to drop for
  a different reason).
- A community member completes the per-voice re-export effort and
  publishes a tooling pipeline we can adopt.
