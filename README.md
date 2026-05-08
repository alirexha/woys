# woys

> **Status: private alpha — All Rights Reserved. Not for redistribution.**
> This repository is private and proprietary pending a commercial decision.
> See `LICENSE` and `NOTICE` for the boundary between original work and the
> upstream `w-okada/voice-changer` MIT-licensed code.

Linux-native real-time voice changer. RVC-only, ONNX Runtime CUDA, PipeWire-native, terminal-controlled. Originally targeted CachyOS; runs on any modern Linux with PipeWire + an NVIDIA GPU.

## What it is

A fork-and-trim of [w-okada/voice-changer](https://github.com/w-okada/voice-changer) (MIT) that strips the engine to RVC-only, replaces the web GUI with a Textual TUI, integrates a persistent virtual mic via PipeWire, and ships as a proper Arch package. The fork keeps RVC inference on ONNX Runtime CUDA EP and removes the Beatrice / MMVC / so-vits-svc / DDSP-SVC / Diffusion-SVC / EasyVC / LLVC engine paths along with all Windows/WSL/macOS code.

## Goals (measured, not claimed)

- **Inference floor `< 80 ms`** per chunk — the gate in
  `tests/test_smoke_rvc_onnx.py::LATENCY_FLOOR_MS` (measured on
  RTX 2070, ORT-CUDA, RVC v2 + RMVPE).
- **End-to-end mic → output**: ~500-540 ms with v0.8.0/v0.9.0/v0.9.2
  defaults (chunk 150 + inference ~80 + native-pw output ~170 + PipeWire
  codec ~30). v0.9.0 switched the playback backend to a native PipeWire
  client (closes the per-quantum gap class from pw-cat); v0.9.1 made
  it default but also expanded the ring-buffer slack to 191 ms by
  default — a regression that added ~170 ms of echo without fixing
  the audible cut class. **v0.9.2 reverts the slack default to 0** (one
  PipeWire-quantum tolerance, ~21 ms slack); the knob
  (`prefer_native_pw_buffer_ms`) stays tunable for users who prefer
  to absorb writer-jitter at a latency cost. See `docs/05-perf.md`
  for the rc-by-rc latency table and the v0.9.0-rc4 A/B that
  established **both backends produce equivalent audible cut rates
  on this hardware** — the cuts are upstream of the playback layer,
  in the engine's writer-jitter (~80 ms std-dev). v0.10.x attacks that.
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
