# 23 — RNNoise chain after woys-mic (v0.13.x)

> **v0.13.3 update.** Apps now see `woys-by-alirexha` (cleaned, daily
> driver) and `woys-no-cleanup` (raw fallback) directly in their
> input device dropdown. Internal plumbing nodes are tagged
> `_internal-...` in their descriptions so users know not to pick
> them. No audio path or measurement changes — this is naming polish
> on top of the v0.13.2 architecture.

The v0.12.x sweep series eliminated the chunk-period periodic
mechanism on this stack at the spectral level (LESSONS §42:
autocorr@chunk_period = 0.000 in v0.12.4 default). The remaining
~75 cuts/min in TTS-driven measurement are aperiodic transient
artifacts from RVC's vocoder.

v0.13.x ships an OPT-IN RNNoise chain that takes `woys-mic` and
produces `woys-mic-clean`, a parallel virtual mic with the same audio
passed through RNNoise. Apps that select `woys-mic-clean.monitor`
hear ~27 % fewer cut events at the cost of ~40 ms additional
latency. The default `woys-mic` source is unchanged.

> **Heads-up on `.monitor`** — apps record from
> `woys-mic-clean.monitor`, not bare `woys-mic-clean`. The destination
> is a sink (Audio/Sink class), and its monitor port is the source
> apps see. See "Why the `.monitor` suffix" below.

## v0.13.0 → v0.13.2 — what changed

v0.13.0 was the first cut. Under real use (Telegram, system speakers
unmuted) the chain leaked: audio played through speakers regardless
of woys's monitor toggle, because the LADSPA filter-chain output
stream auto-routed to the user's default ALSA sink AS WELL AS the
intended destination. The "13 % cut reduction" measured against
v0.13.0 was contaminated by that speaker echo.

v0.13.2 fixes the leak (root cause: `media.class=Audio/Source/Virtual`
was rejected by wireplumber as a playback target, so the LADSPA
output became an orphan node and policy-routed to default ALSA;
detail in LESSONS §44). Re-measured against the fixed chain, the
real RNNoise contribution is **−27 %**, double v0.13.0's
contaminated number.

## Measured impact (v0.13.2)

60 s TTS-driven engine, v0.12.4 defaults, mode=both, two concurrent
recordings (woys-mic and woys-mic-clean.monitor recorded by serial ID):

| metric | woys-mic | woys-mic-clean.monitor | Δ |
|---|---:|---:|---:|
| woys-diag cuts/min | 75.4 | **54.7** | **−27 %** |
| ALSA-hardware leaks (pw-link) | 0 | **0** | — |
| latency (post-engine) | +0 ms | +40 ms | +40 ms |
| total e2e latency on v0.12.4 | ~640 ms | ~680 ms | +40 ms |

## Setup (the easy way)

```bash
sudo pacman -S noise-suppression-for-voice
woys chain enable
```

That's it. `woys chain enable` installs a systemd user unit that
loads the chain on every login from now on, AND starts it
immediately for the current session.

To disable / remove later:

```bash
woys chain disable
```

## Setup (one-shot, no systemd)

```bash
sudo pacman -S noise-suppression-for-voice
woys chain setup    # loads now; gone after reboot
woys chain teardown # unloads
```

`woys chain status` shows currently-loaded chain modules, sources
visible to apps, and runs an automatic ALSA-hardware-leak check (so
a future regression of the v0.13.0 bug shows up loud, not silent).

If you don't have `woys` on PATH, the same logic lives in
`scripts/v013_2_rnnoise_chain.sh` with the same actions.

## Selecting the cleaned mic in apps (v0.13.3)

In Discord / Telegram / CS2's input-device picker, choose
**`woys-by-alirexha`**. That's it. If the app doesn't expose the
new source (rare; some apps cache device lists), restart the app
after `setup`.

There's also a fallback option called **`woys-no-cleanup`** (the
raw v0.12.4 engine output, no RNNoise, ~40 ms lower latency). Pick
that one if the cleaned voice sounds over-suppressed for your
content.

The other monitor sources you'll see in the dropdown
(`Monitor of _internal-...`) are internal plumbing — don't pick
them. Their `_internal-` description prefix sorts them visually
distinct from the two daily-use options.

### v0.13.2 → v0.13.3 migration

If you previously selected `woys-mic-clean.monitor` (the v0.13.2
recommendation), it'll still work — that source still exists and
still carries the cleaned audio. But `woys-by-alirexha` is the
recommended endpoint going forward; it's friendlier-named and
doesn't collide with users opening a power-user `.monitor` of the
internal sink.

## Architecture (v0.13.2 — Architecture B)

```
HyperX → woys engine → WoysSink → loopback → woys-mic (v0.12.4)
                                        ↓
                                   module-loopback (30 ms, mono)
                                        ↓
                       woys-mic-rnnoise-bridge (module-ladspa-sink, RNNoise)
                                        ↓ sink_master
                       woys-mic-clean (module-null-sink, Audio/Sink, mono)
                                        ↓ auto-created monitor port
                       woys-mic-clean.monitor (apps consume)
```

The whole chain is mono (`channels=1`) end-to-end. The
`noise_suppressor_mono` plugin processes one channel; if the
LADSPA-sink were stereo, PipeWire would spawn two filter instances
in parallel and the resulting stereo stream would never bind back
to the mono `sink_master`.

## Why the `.monitor` suffix

The two architectures we considered were:

  * **Architecture A — `media.class=Audio/Source/Virtual` on the
    final null-sink.** Apps would record directly from
    `woys-mic-clean`. *This is what v0.13.0 shipped, and it was
    broken.* Wireplumber refused to recognize a Source/Virtual node
    as a valid playback target for the LADSPA filter-chain output,
    so `sink_master=` never bound and the orphan stream got auto-
    routed to the default ALSA sink (= my speakers). The 13 % cut
    reduction we measured against v0.13.0 was 86 % real RNNoise +
    14 % feedback / room reflection contamination.

  * **Architecture B — `media.class=Audio/Sink` on the final null-
    sink.** Apps record from the auto-created `.monitor` port. This
    is what v0.13.2 ships. Wireplumber accepts the destination as a
    valid playback target, `sink_master=` binds cleanly, the
    filter-chain output never escapes to ALSA, and the `.monitor`
    port carries exactly the audio the LADSPA filter wrote. ZERO
    leaks confirmed via `pw-link -l` and the new
    `woys chain status` self-check.

The cost of the fix is one extra word in the device picker
(`.monitor`). The benefit is the chain actually works.

## Why it works (when it works)

RNNoise is a recurrent neural network trained on speech vs noise.
Its frame size is 10 ms; per frame it outputs a gain coefficient
that's applied to the spectrum.

For voice content: gain ≈ 1.0 (preserve).
For non-voice content: gain → 0.0 (suppress).

Chunk-boundary clicks in RVC output are short-duration, high-
spectral-tilt transients. Some of them register as "non-voice" in
RNNoise's classifier and get attenuated. Not all — the network
wasn't trained for click suppression specifically. The 27 %
reduction is the fraction of clicks that happen to look enough
like noise to the classifier.

A click suppressor trained specifically on chunk-boundary
discontinuities (e.g., a small CNN trained on RVC chunk-boundary
artifacts) might do better. That's research, not an out-of-box
tool. Out of scope.

## Why this isn't shipped as default

  * Latency budget: v0.12.4 already added +100 ms over v0.11.0 to
    get rhythm GONE perceptually. Adding +40 ms more puts total e2e
    latency at ~680 ms, close to conversational comfort threshold.
    Users who picked v0.12.4's tradeoff for clean audio may not
    want a further latency bump.
  * 27 % is real but modest. The headline v0.12.4 win was 25 %
    cuts/min reduction PLUS the chunk-period rhythm vanishing
    perceptually. v0.13.x's 27 % is on top of that, on residual
    transients the user already accepted.
  * It's a system-level chain dependent on a non-woys package.
    Easier to document and let users enable than to bake in.

## Troubleshooting

**"woys-mic source not present"** — run `woys pw setup` first to
load the v0.12.4 woys-mic chain.

**"librnnoise_ladspa.so not found"** — install
`noise-suppression-for-voice` from the cachyos / extra repo. The
file should appear at `/usr/lib/ladspa/librnnoise_ladspa.so`.

**Apps don't see woys-mic-clean.monitor** — check
`pactl list short sources`. If it's there but missing in the app,
the app cached its device list at startup; restart it.

**Apps see `woys-mic-clean` but recording is silent** — you picked
the sink instead of its monitor. Pick `woys-mic-clean.monitor` (it
appears as a separate entry in pavucontrol / pactl).

**`woys chain status` reports an ALSA leak** — that's a regression
of the v0.13.0 bug. Run `woys chain teardown` then
`woys chain setup` to reset; if it persists, file a bug with the
output of `pw-link -l` and `pactl list short modules`.

**Audio sounds robotic / over-suppressed** — the
`noise_suppressor_mono` plugin applies aggressive RNNoise
suppression. Voice that has a lot of breathiness or music-like
content can sound artifact-ish. If problematic, use the v0.12.4
default `woys-mic` source instead (unchanged underneath).

## Original work — Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
