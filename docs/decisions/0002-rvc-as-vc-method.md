# 0002 — RVC as the voice-conversion architecture

## Decision

woys uses Retrieval-based Voice Conversion (RVC) as its sole
voice-conversion architecture, inherited from upstream
`w-okada/voice-changer` and trimmed to RVC-only.

## Status

`accepted`

## Context

The brief (`PROJECT_BRIEF.md` §13) instructs us to fork
`w-okada/voice-changer` and strip non-RVC engines. Upstream supported
RVC, MMVC, Beatrice, so-vits-svc, and DDSP-SVC; the v0.1.x recon and
trim removed ~22 k LOC of non-RVC engine code (`docs/00-recon.md`).
That left RVC as the inheritance, not the considered choice. This doc
captures *why* RVC is the right inheritance to keep.

## Decision

RVC is the only supported voice-conversion model class.

## Alternatives considered

- **FreeVC** — speaker-conditioned VC, single unified model. Quality
  is competitive on benchmarks but the trained checkpoint zoo is
  small; community-trained voices live almost entirely in RVC format.
- **KNN-VC** — non-parametric retrieval over a target speaker corpus.
  Requires per-target dataset prep at inference time; no checkpoint
  ecosystem.
- **OpenVoice / SeedTTS / cross-lingual TTS-style cloners** — these
  are TTS-driven, not real-time mic-in/mic-out converters. Wrong
  shape for the brief's Discord/CS2 streaming use case.
- **so-vits-svc** — present in upstream, was trimmed. Higher quality
  on sustained content but heavier (singer-voice-converter lineage),
  longer warmup, and the community-trained zoo skewed toward
  vocaloid-style targets.

## Rationale

RVC won on three load-bearing axes that the alternatives don't share.
First, model availability: weights.gg, Hugging Face, and the
v0.7.x/v0.13.x community-trained voice catalogue are all RVC `.pth`
checkpoints. The user's daily-driver voice (`e_girl`) and the
foundation-default voice (`amitaro_v2_16k`, see decision 0012) are
both RVC. Second, decoder shape fits real-time chunked inference: RVC
v2 runs at 16-100 ms per 250 ms chunk on RTX 2070 Mobile (`LESSONS.md`
§19), well inside the streaming envelope. Third, upstream provided a
working ONNX export path; FreeVC/KNN-VC would have required us to
build that ourselves — explicitly out of scope per the brief.

The "we don't train models" rule (`PROJECT_BRIEF.md` §16) makes
inheriting an architecture-with-ecosystem the only viable path.

## Trade-offs accepted

RVC has known artefacts: chunk-period rhythmic clicks on sustained
vowels (`LESSONS.md` §36-§42) caused by NSF source-module behaviour
across chunk boundaries. We invest engineering in masking those
artefacts (SOLA crossfade, chunk_seconds=0.25, optional RNNoise
chain) rather than in switching architectures. Per-voice models
mean each voice is a separate ONNX file (~50-200 MB), not a single
shared checkpoint with speaker conditioning.

## Re-litigation triggers

- A new VC architecture ships with: (a) a community-trained voice
  catalogue at RVC's scale, (b) ONNX export tooling, (c) measurably
  fewer chunk-boundary artefacts on sustained content.
- RVC v3 or a successor retires NSF in favour of a non-source-module
  vocoder (would eliminate the §36-§42 mechanism entirely).
- Per-voice file size becomes a deployment concern (e.g., shipping
  voice packs in an installer).
