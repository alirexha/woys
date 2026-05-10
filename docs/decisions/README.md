# woys decision corpus

This directory documents the load-bearing decisions a new maintainer
needs to know before reading code. Each file is a one-pager (50-200
lines) covering one decision: what we picked, why, what we gave up,
and what would make us revisit.

These docs are *decision records*, not roadmap items, fix proposals,
or chronological retrospectives. For chronology, see
`../../CHANGELOG.md` and `../../LESSONS.md`. For audit / review
artefacts that *led* to several of these decisions being formalised,
see `../24-review/`. Decision records survive on their own.

## How to add a decision

1. Copy `0000-template.md` to the next free numbered file
   (`NNNN-short-title.md`, zero-padded, monotonic).
2. Fill every section: Decision, Status, Context, (repeat) Decision,
   Alternatives considered, Rationale, Trade-offs accepted,
   Re-litigation triggers.
3. Don't fabricate citations. If no documented rationale exists, set
   status to `provisional` and write the resolving experiment in the
   Re-litigation triggers / Test plan section.
4. Add an entry to the index below.
5. Cross-link from any code comment that touches this decision.

## Index

| #    | Title | Status |
|------|-------|--------|
| [0001](0001-ort-vs-tensorrt.md) | ONNX Runtime CUDA EP, with TensorRT EP retained as opt-in | accepted |
| [0002](0002-rvc-as-vc-method.md) | RVC as the voice-conversion architecture | accepted |
| [0003](0003-textual-tui-vs-alternatives.md) | Textual for the TUI | accepted |
| [0004](0004-sample-rate-48000.md) | 48 000 Hz as the I/O sample rate | accepted |
| [0005](0005-toml-config.md) | TOML for user-facing configuration files | accepted |
| [0006](0006-rnnoise-opt-in-chain.md) | RNNoise as an opt-in pipewire-pulse chain | accepted |
| [0007](0007-gpu-anti-jitter-default-off.md) | `gpu_anti_jitter_mode = "off"` as the default | accepted |
| [0008](0008-nsf-state-passing-out-of-scope.md) | NSF state passing across chunk boundaries is out of scope | deferred |
| [0009](0009-native-pipewire-helper-in-c.md) | Native PipeWire output helper written in C | accepted |
| [0010](0010-fp16-contentvec-evaluation-stale.md) | fp16 ContentVec evaluation is stale | provisional |
| [0011](0011-fp16-tensorrt-rvc-validation-deferred.md) | TensorRT FP16 for RVC requires per-voice validation | provisional |
| [0012](0012-amitaro-default-voice.md) | `amitaro_v2_16k` as the install-time default voice | accepted |
| [0013](0013-latency-target-retired.md) | The 80 ms latency target is retired; ~640 ms is the user-validated optimum | accepted |

## Status meanings

- `accepted` — current behaviour, evidence-backed.
- `provisional` — current behaviour, but re-evaluate after a
  named experiment.
- `superseded by NNNN` — see the linked successor.
- `deferred` — no decision yet; doc captures the open question.

## Template

See `0000-template.md` for the canonical structure.
