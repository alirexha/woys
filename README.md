# woys

> **Status: private alpha — All Rights Reserved. Not for redistribution.**
> This repository is private and proprietary pending a commercial decision.
> See `LICENSE` and `NOTICE` for the boundary between original work and the
> upstream `w-okada/voice-changer` MIT-licensed code.

Linux-native real-time voice changer. RVC-only, ONNX Runtime CUDA, PipeWire-native, terminal-controlled. Originally targeted CachyOS; runs on any modern Linux with PipeWire + an NVIDIA GPU.

## Status (v0.13.2)

Daily-use ready on RTX 2070 Mobile. The user's perceptual A/B
test (Desktop WAV listening) ratified the v0.12.3 sweep top-1
config as the new default profile in v0.12.4. Measured:

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

### v0.13.2 opt-in: RNNoise chain (`woys-mic-clean.monitor`)

If you want a further ~27 % cut reduction at +40 ms additional
latency cost, install the LADSPA plugin and let `woys` manage the
chain:

```bash
sudo pacman -S noise-suppression-for-voice
woys chain enable    # systemd user unit; loads now + on every login
# in your app, select `woys-mic-clean.monitor` (NOT bare `woys-mic-clean`)
woys chain disable   # remove unit + tear down chain
```

For one-shot use without systemd:

```bash
woys chain setup
woys chain status    # shows modules, sources, ALSA-leak self-check
woys chain teardown
```

See `docs/23-rnnoise-chain.md` for the measured impact and the
v0.13.0 → v0.13.2 fix history (v0.13.0 had a routing bug that
leaked filter-chain output to ALSA hardware sinks; v0.13.2 fixes it
and the real RNNoise contribution is twice what we originally
measured).

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
- Single `./install.sh`, runs in under 5 minutes on a fresh CachyOS.

See `docs/05-perf.md` for the actual measured numbers (some targets are
currently missed; the path to closing the gap is documented).

## Quick start

See `docs/INSTALL.md`. Short version:

```
git clone https://github.com/alirexha/woys.git
cd woys
./install.sh
woys run --autostart
```

Then point Discord (`docs/DISCORD-SETUP.md`) or CS2 (`docs/CS2-SETUP.md`)
at the `woys-mic` device that appears in their input-device pickers.

### AUR (pending repo de-privatisation)

`pkg/PKGBUILD` and `pkg/.SRCINFO` are submission-ready. Once the GitHub
repo is public, follow `pkg/README-AUR.md` to push to
`aur.archlinux.org/packages/woys`. Until then, `./install.sh`
is the supported install path.

## Credits

This fork is built on the work of **[w-okada](https://github.com/w-okada)**
and the original [voice-changer](https://github.com/w-okada/voice-changer)
project. The portions of this repository under `upstream/` and any code
within `src/server/` that descends from upstream remain under the original
MIT license (`upstream/LICENSE`). All original work in `src/woys/`,
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
all rights reserved). Upstream-derived files are governed by `upstream/LICENSE`
(MIT). See `NOTICE` for the file-by-file audit trail.

No license is granted to copy, modify, distribute, or sublicense the
original work without prior written permission from the copyright holder.
