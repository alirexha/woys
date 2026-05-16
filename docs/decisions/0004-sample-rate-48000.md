# 0004 — 48 000 Hz as the I/O sample rate

## Decision

Both `mic_rate` and `sink_rate` default to 48 000 Hz, matching the
PipeWire system-default quantum rate.

## Status

`accepted`

## Context

Three sample rates are at play in the inference graph:
mic capture (`engine.py:174` — `mic_rate: int = 48_000`), the
internal model rate (16 000 Hz, ContentVec's required input), and
the model output rate (varies per voice — 16 / 40 / 48 kHz; see
`engine.py:1604` and decision 0002). The brief (`PROJECT_BRIEF.md`
§9) says woys must publish a virtual mic that downstream apps see as
a normal input device, with Discord/CS2 as the named consumers.
Downstream apps run on PipeWire, and PipeWire's negotiated default
on this stack is 48 000 Hz at quantum 1024.

## Decision

48 000 Hz is the I/O rate; resampling to/from 16 kHz happens
inside the engine on dedicated soxr stream objects.

## Alternatives considered

- **44 100 Hz** — historical CD/audio default, still the macOS /
  Windows / consumer-app default. On Linux + PipeWire, picking 44 100
  forces a mid-graph resample inside PipeWire to land at the
  negotiated 48 000 — undoing the v0.5.1 anti-aliasing + soxr-quality
  fixes (`LESSONS.md` §12) for no perceptual gain.
- **16 000 Hz** — match the model-internal rate, skip soxr entirely.
  But mic capture from PipeWire still ramps to whatever PipeWire
  negotiates with the device (typically 48 000), so the "skip soxr"
  saving is illusory; downstream apps then receive a 16 kHz source
  and either resample (cost) or sound thin (quality).
- **96 000 Hz** — over-sampled relative to ContentVec's 16 kHz input;
  pure cost, no benefit on this pipeline.

## Rationale

PipeWire on Arch / CachyOS negotiates 48 000 Hz / 1024 quantum by
default. Aligning woys's I/O to that rate means: (a) capture from a
typical USB mic (USB condenser mic, 48 000 native) does not pay an
extra resample, (b) the virtual `woys-mic` source presents at 48 000
to apps and matches PipeWire's quantum-native rate so apps don't
re-resample either, (c) `bin/woys-pw-out.c` (decision 0009) operates
at PipeWire's quantum boundary cleanly, which is load-bearing for the
v0.7.x quantum-boundary cut investigation. Picking any other rate
would either force a PipeWire-internal resample step or push that
work into woys for no upstream-app benefit.

## Trade-offs accepted

Inside the engine, soxr resamples mic 48 k → 16 k for ContentVec /
RMVPE, and resamples RVC output (16 / 40 / 48 k depending on voice)
back to 48 k for the sink. That's two resampler instances on the hot
path. The cost is ~µs at warm steady state on this hardware
(`LESSONS.md` §12). Voices whose model output rate matches 48 k
(rare; most v2 models are 40 k) skip the output resample.

## Re-litigation triggers

- PipeWire system default migrates to a different quantum rate (e.g.
  Linux audio convention shifts to 96 kHz). Re-evaluate to align.
- A future ContentVec / RMVPE variant requires a different internal
  rate.
- Hardware ships with a 44 100-native USB mic that resists 48 k
  negotiation — at that point a per-device override becomes
  warranted, but the *default* stays at 48 k.
