# woys

Linux-native real-time voice changer. RVC-only, ONNX Runtime CUDA,
PipeWire-native, terminal-controlled. Tested on CachyOS + RTX 2070;
should work on any modern Linux with PipeWire + an NVIDIA GPU.

> **Honest disclaimer.** This is a personal project. I built it for my
> own daily use; you're welcome to try it, but response time on issues
> is best-effort. The codebase has been through two adversarial-audit
> cycles (the most recent being v0.15.0, 213 findings, 80 fix commits);
> the full audit workspace is preserved at `docs/26-review/` if
> you want to know exactly what's been tested and what hasn't.
> Original work is **All Rights Reserved** (see `LICENSE`); the
> `src/server/` subtree carries MIT (see `src/server/LICENSE` and
> `NOTICE`).

## Hardware requirements

- **GPU:** NVIDIA RTX 2070 or better. The engine has been measured
  on an RTX 2070 Mobile + i7-10750H (`docs/26-review/benchmark-v0.7-to-v0.15.html`).
  CUDA support is **required** — there's no CPU fallback (an honest
  hard-fail since v0.15.0; pre-v0.15.0 falling back to CPU silently
  was a P0 audit finding).
- **System RAM:** measured engine RSS peaks at ~1.5 GB with the
  foundation models + one RVC voice loaded. 8 GB total system RAM
  is comfortable for the engine alongside Discord / a game / a
  browser; 16 GB is generous. Below 4 GB free is not tested.
- **VRAM:** ~1.1 GB peak (foundation models dominate; one RVC voice
  adds ~150 MB).
- **OS / audio:** Linux + PipeWire. PipeWire 1.2+ tested; PulseAudio
  / bare-ALSA are not supported (the engine speaks to PipeWire
  directly via `pw-cat` and `pactl`). The original development
  target was CachyOS; other recent distros with PipeWire should work
  but are unverified.
- **Disk:** ~1 GB for the foundation weights + sample voice (more
  if you bring your own RVC voices, typically 60-150 MB each).
- **Internet:** required on first install to download the foundation
  weights (~700 MB total) and the sample voice (~64 MB).

## Quick Start

Two tiers. Tier 1 is the "did it work?" path; Tier 2 is where you
swap in your own voices once Tier 1 produces sound.

### Tier 1 — install and hear your voice converted (5-10 minutes)

```bash
git clone https://github.com/alirexha/woys.git
cd woys
./install.sh                  # downloads weights, sets up the venv, registers `woys`
woys run --autostart          # starts the engine + TUI; press `m` to monitor your output through your speakers
```

That's it. Speak into your default mic; press `m` inside the TUI to
hear the converted output through your speakers (otherwise you need
an app like Discord pointed at the new `woys-mic` source to hear it
— see `docs/DISCORD-SETUP.md`).

The `./install.sh` step is the slow one: it downloads ~700 MB of
foundation weights (`contentvec-f.onnx`, `rmvpe_wrapped.onnx`) and
a 64 MB sample voice (`amitaro_v2_16k.onnx`) from HuggingFace, then
creates a Python venv and installs the engine. Plan for 5-15
minutes depending on your internet; the foundation download is the
load-bearing leg.

### Tier 2 — add more voices

The default install gives you ONE sample voice. To add more,
download an RVC ONNX repo from HuggingFace and switch to it:

```bash
woys models download wok000/vcclient_model   # the sample repo install.sh already uses
woys models list                              # see what's now in your library
woys models use <model-name>                  # pick one by file stem, e.g. `woys models use amitaro_v2_16k`
```

The TUI picks up the new selection on the next engine restart (`q`,
then `woys run --autostart`).

## What You Get

**The engine:** `woys` ships the inference engine, the PipeWire glue,
a Textual TUI, and a CLI. The smoke-test sample voice
(`amitaro_v2_16k.onnx`) ships via `./install.sh` so Tier 1 above
produces audible output without an extra download.

**No bundled voice models beyond the smoke-test sample.** RVC voice
models are downloaded separately from HuggingFace. They're free.
A typical RVC voice is 60-150 MB; download time is 10-60 seconds
on a decent connection.

**Finding more voices:**

- **Start here:** [`wok000/vcclient_model`](https://huggingface.co/wok000/vcclient_model)
  — the same repo `./install.sh` pulls the sample voice from.
  Multiple known-licensed sample voices. Safe pick to verify your
  setup works with multiple models.
- **Browse more:** [`huggingface.co/models?other=rvc`](https://huggingface.co/models?other=rvc)
  — search for "rvc" or "voice changer"; many community voice
  models live here.
- **Legal:** RVC voice clones are subject to whatever license the
  uploader applied (often unclear). **For your own use anything you
  have rights to is fine; for streaming or recording with someone
  else's voice, get permission. Voice cloning of public figures
  without consent is legally and ethically hazardous.** See
  `docs/MODELS.md` for the full license discussion (including the
  GPL-3.0 status of the foundation weights).

**Converting your own `.pth`:** if you have an RVC checkpoint
`.pth` file (the format most community RVC tools produce), `woys
convert` exports it to `.onnx` for use with this engine. See
`docs/MODELS.md`.

## Status (v0.15.0)

v0.15.0 is the post-review hardening release: 36-lens
adversarial audit, 213 unique findings, 80 fix commits over the
v0.14.3 → v0.15.0 span. The cycle focused on **correctness,
observability, UX, security, and legal hygiene**, not perf — so
the latency / VRAM / cuts-per-minute numbers below carry forward
unchanged from v0.12.4 / v0.14.x. See `CHANGELOG.md` § [0.15.0]
for the full set of fixes, deferred items, and audit transparency.

Daily-use ready on RTX 2070 Mobile. The user's perceptual A/B
test (Desktop WAV listening) ratified the v0.12.3 sweep top-1
config as the new default profile in v0.12.4. Measured (still
current in v0.15.0; the audit didn't move these):

  * **cuts/min (TTS sustained content): 58.2** (was 78.0 in v0.11.0 — −25%)
  * **autocorr@chunk_period: 0.000** (was 0.136 — chunk-period
    rhythm entirely eliminated; the "train wagon on rails" pattern
    is gone at the spectral level)
  * **Total e2e latency: ~640 ms** (was ~540 ms; +100 ms
    is the chunk_seconds=0.15 → 0.25 cost the user accepted in
    exchange for clean output)
  * **Underrun rate (real Telegram, mode=both): 0.2/sec** — unchanged
    from v0.11.0; v0.12.4's improvements are spectral-clean rather
    than throughput

Headline feature still opt-in: `gpu_anti_jitter_mode = "both"` keeps
the GPU's dynamic boost from auto-deboosting during the engine's
mic_read idle window. Stock GPU specs only — no overclock, no
power-limit changes, no firmware. Reverts on engine stop / SIGTERM
/ SIGINT. Documented in `docs/22-gpu-clock-lock.md`.

Configurations that minimise latency at the cost of more cuts (e.g.
`chunk_seconds = 0.15`, the v0.11.0/v0.12.3-low-latency-tier defaults)
remain available via `~/.config/woys/config.toml` for users who want
that tradeoff.

## What it is

A fork-and-trim of [w-okada/voice-changer](https://github.com/w-okada/voice-changer) (MIT) that strips the engine to RVC-only, replaces the web GUI with a Textual TUI, integrates a persistent virtual mic via PipeWire, and ships as a proper Arch package. The fork keeps RVC inference on ONNX Runtime CUDA EP and removes the Beatrice / MMVC / so-vits-svc / DDSP-SVC / Diffusion-SVC / EasyVC / LLVC engine paths along with all Windows/WSL/macOS code.

## Goals (measured, not claimed)

- **Inference floor `< 80 ms`** per chunk — the gate in
  `tests/test_smoke_rvc_onnx.py::LATENCY_FLOOR_MS` (measured on
  RTX 2070, ORT-CUDA, RVC v2 + RMVPE). v0.11.0 mode=both: ~45 ms
  inference average in real Telegram session.
- **End-to-end mic → output**: ~500-540 ms with v0.8.0 / v0.9.0 /
  v0.9.2 / v0.11.0 defaults (chunk 150 + inference ~45 + native-pw
  output ~170 + PipeWire codec ~30). v0.9.0 switched the playback
  backend to a native PipeWire client (closes the per-quantum gap
  class from pw-cat); v0.9.1 expanded the ring-buffer slack to 191 ms
  by default — a regression that added ~170 ms of echo without
  fixing the audible cut class. **v0.9.2 reverts the slack default
  to 0** (one PipeWire-quantum tolerance, ~21 ms slack);
  `prefer_native_pw_buffer_ms` stays tunable. v0.10.0 added per-stage
  instrumentation + a synthetic harness that located the dominant
  cause of audible cuts: NVIDIA dynamic-boost auto-deboost during
  the ~98 ms mic_read window between chunks. **v0.11.0 attacks that
  with `gpu_anti_jitter_mode = "both"` (clock_lock + torch
  separate-stream keepalive).** Default off; opt-in via config.toml.
  See `docs/22-gpu-clock-lock.md` and `docs/05-perf.md` for the
  rc-by-rc latency table.
- Idle VRAM `< 500 MB` (currently misses at ~1.35 GiB — foundation
  models dominate).
- CPU `< 15 %` while active.
- `./install.sh` is the single command; expect 5-15 minutes on a
  fresh CachyOS box depending on download speed (the foundation
  weights + sample voice are ~760 MB total).

See `docs/05-perf.md` for the actual measured numbers (some targets are
currently missed; the path to closing the gap is documented). See
`docs/26-review/benchmark-v0.7-to-v0.15.html` for the v0.15.0
cross-version benchmark on RTX 2070 + i7-10750H.

## Hooking up apps

Once the engine is running, apps see a new audio input device called
`woys-mic`. Point your app's input at it:

- **Discord:** `docs/DISCORD-SETUP.md` (note: disable Discord's
  Krisp noise suppression — it eats RVC output).
- **CS2:** `docs/CS2-SETUP.md`.
- **Any PipeWire-aware app:** pick `woys-mic` from its input-device
  picker.

If you enable the RNNoise post-processing chain (next section), apps
see `woys-clean` (cleaned) and `woys-no-cleanup` (raw) instead
of bare `woys-mic`.

### Optional: RNNoise chain

If you want a further ~27% cut reduction at +40 ms additional
latency cost, install the LADSPA plugin and let `woys` manage the
chain:

```bash
sudo pacman -S noise-suppression-for-voice  # Arch / CachyOS; other distros: look for `rnnoise` LADSPA
woys chain enable    # systemd user unit; loads now + on every login
# in your app, select `woys-clean` (the cleaned daily driver)
# fallback option named `woys-no-cleanup` is the raw v0.12.4 path
woys chain disable   # remove unit + tear down chain
```

For one-shot use without systemd:

```bash
woys chain setup
woys chain status    # shows modules, sources, ALSA-leak self-check
woys chain teardown
```

See `docs/23-rnnoise-chain.md` for the measured impact and the
v0.13.0 → v0.13.3 history.

### AUR

`pkg/PKGBUILD` and `pkg/.SRCINFO` are submission-ready. To push to
`aur.archlinux.org/packages/woys`, follow `pkg/README-AUR.md`.
Until then, `./install.sh` is the supported install path.

## Credits

This fork is built on the work of **[w-okada](https://github.com/w-okada)**
and the original [voice-changer](https://github.com/w-okada/voice-changer)
project. The portions of this repository under `upstream/` and any code
within `src/server/` that descends from upstream remain under the original
MIT license (full text in `src/server/LICENSE`). All original work in `src/woys/`,
`src/audio/`, `src/tui/`, `tests/`, `scripts/`, `pkg/`, and `docs/` is the
proprietary work of Alireza Hamayeli.

## License

This repository contains code under **two distinct licenses**:

| Path                                          | License                  | Source            |
|-----------------------------------------------|--------------------------|-------------------|
| `upstream/`                                   | MIT                      | w-okada/voice-changer |
| `src/server/` (vendored, trimmed)             | MIT (derivative)         | w-okada/voice-changer |
| `src/{woys,audio,tui}/`             | **All Rights Reserved**  | Alireza Hamayeli  |
| `tests/`, `scripts/`, `pkg/`, `docs/`, `*.sh` | **All Rights Reserved**  | Alireza Hamayeli  |
| Top-level configuration & metadata            | **All Rights Reserved**  | Alireza Hamayeli  |

Original-work files are governed by `LICENSE` at the repo root (proprietary,
all rights reserved). Upstream-derived files are MIT-licensed; the full MIT
text ships in `src/server/LICENSE`. See `NOTICE` for the file-by-file audit
trail.

No license is granted to copy, modify, distribute, or sublicense the
original work without prior written permission from the copyright holder.
