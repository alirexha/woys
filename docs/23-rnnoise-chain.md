# 23 — RNNoise chain after woys-mic (v0.13.0 opt-in)

The v0.12.x sweep series eliminated the chunk-period periodic
mechanism on this stack at the spectral level (LESSONS §42:
autocorr@chunk_period = 0.000 in v0.12.4 default). The remaining
~86 cuts/min in TTS-driven measurement are aperiodic transient
artifacts from RVC's vocoder.

v0.13.0 ships an OPT-IN RNNoise chain that takes `woys-mic` and
produces `woys-mic-clean`, a parallel virtual source with the same
audio passed through RNNoise. Apps that select `woys-mic-clean`
hear ~13 % fewer cut events at the cost of ~40 ms additional
latency. The default `woys-mic` source is unchanged.

## Measured impact

60 s TTS-driven engine, v0.12.4 defaults, mode=both, two concurrent
recordings (woys-mic and woys-mic-clean recorded by serial ID):

| metric | woys-mic | woys-mic-clean | Δ |
|---|---:|---:|---:|
| woys-diag cuts/min | 86.5 | **75.2** | **-13 %** |
| woys-diag total events (60 s) | 99 | 86 | -13 |
| spectral autocorr peak at 150 ms | 0.111 | **0.079** | -29 % |
| latency (post-engine) | +0 ms | +40 ms | +40 ms |
| total e2e latency on v0.12.4 | ~640 ms | ~680 ms | +40 ms |

13 % cut reduction is the headline. RNNoise wasn't trained for
chunk-boundary clicks specifically; the suppression comes from
RNNoise classifying short-duration high-frequency transients as
non-voice and attenuating them. Since clicks are sub-perception-
threshold transients, a fraction get caught by the classifier even
though that's not its design intent.

## Setup

### 1. Install the LADSPA plugin

```bash
sudo pacman -S noise-suppression-for-voice
```

This drops `/usr/lib/ladspa/librnnoise_ladspa.so` (the same RNNoise
plugin NoiseTorch uses, packaged independently for direct PA module
loading).

### 2. Load the chain

```bash
cd /path/to/woys
./scripts/v013_0_rnnoise_chain.sh setup
```

The script loads three PipeWire modules:

  1. `module-null-sink sink_name=woys-mic-clean media.class=Audio/Source/Virtual`
     — terminal sink that holds the denoised audio (apps consume it
     as a virtual mic source named `woys-mic-clean`)
  2. `module-ladspa-sink sink_name=woys-mic-rnnoise-bridge
     sink_master=woys-mic-clean plugin=...librnnoise_ladspa.so
     label=noise_suppressor_mono` — the RNNoise filter
  3. `module-loopback source=woys-mic sink=woys-mic-rnnoise-bridge
     latency_msec=30` — feeds the v0.12.4 engine output into the
     filter chain

### 3. Select `woys-mic-clean` in the app

In Discord / Telegram / CS2's input-device picker, choose
`woys-mic-clean` instead of `woys-mic`. The app gets the denoised
audio.

If the app doesn't expose the new source (rare; some apps cache
device lists), restart the app after `setup`.

### 4. Disable when not needed

```bash
./scripts/v013_0_rnnoise_chain.sh teardown
```

`woys-mic` itself is unaffected; teardown only removes the chain
that produces `woys-mic-clean`.

## Architecture

```
HyperX → woys engine → WoysSink → loopback → woys-mic (v0.12.4)
                                        ↓
                                   loopback (30 ms, mono)
                                        ↓
                            woys-mic-rnnoise-bridge (LADSPA-sink, RNNoise applies here)
                                        ↓
                                woys-mic-clean (v0.13.0; apps consume)
```

The chain runs in parallel with the existing v0.12.4 woys-mic
output. Apps can select either source. Latency cost is ~40 ms
across the two added stages (loopback + RNNoise frame).

## Why this isn't shipped as default

  * Latency budget: v0.12.4 already added +100 ms over v0.11.0 to
    get rhythm GONE perceptually. Adding +40 ms more puts total e2e
    latency at ~680 ms, close to conversational comfort threshold.
    Users who picked v0.12.4's tradeoff for clean audio may not
    want a further latency bump.
  * 13 % is real but modest. The headline v0.12.4 win was 25 %
    cuts/min reduction PLUS the chunk-period rhythm vanishing
    perceptually. v0.13.0's 13 % is on top of that, on residual
    transients the user already accepted.
  * It's an opt-in, system-level chain dependent on a non-woys
    package. Easier to document and let users enable than to bake
    in.

## Why it works (when it works)

RNNoise is a recurrent neural network trained on speech vs noise.
Its frame size is 10 ms; per frame it outputs a gain coefficient
that's applied to the spectrum.

For voice content: gain ≈ 1.0 (preserve).
For non-voice content: gain → 0.0 (suppress).

Chunk-boundary clicks in RVC output are short-duration, high-
spectral-tilt transients. Some of them register as "non-voice" in
RNNoise's classifier and get attenuated. Not all — the network
wasn't trained for click suppression specifically. The 13 %
reduction is the fraction of clicks that happen to look enough
like noise to the classifier.

A click suppressor trained specifically on chunk-boundary
discontinuities (e.g., a small CNN trained on RVC chunk-boundary
artifacts) might do better. That's research, not an out-of-box
tool. Out of scope.

## Troubleshooting

**"woys-mic source not present"** — run `woys pw setup` first to
load the v0.12.4 woys-mic chain.

**"librnnoise_ladspa.so not found"** — install
`noise-suppression-for-voice` from the cachyos / extra repo. The
file should appear at `/usr/lib/ladspa/librnnoise_ladspa.so`.

**Apps don't see woys-mic-clean** — check `pactl list short sources`
shows `woys-mic-clean`. Some apps cache device lists at start;
restart them after `setup`.

**Audio sounds robotic / over-suppressed** — the
`noise_suppressor_mono` plugin applies aggressive RNNoise
suppression. Voice that has a lot of breathiness or music-like
content can sound artifact-ish. If problematic, use the v0.12.4
default `woys-mic` source instead (unchanged underneath).

## Original work — Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
