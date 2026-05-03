# vcclient-cachy

Linux-native, CachyOS-optimized real-time voice changer. RVC-only, ONNX Runtime CUDA, PipeWire-native, terminal-controlled. Beats upstream on latency, installs via PKGBUILD, runs without a browser.

> **Status:** in active development — see `PROGRESS.md`.

## Why a fork?

[w-okada/voice-changer](https://github.com/w-okada/voice-changer) (MIT) is the best open-source RVC client, but it ships a multi-engine web app with Windows/WSL ergonomics. On Linux + PipeWire we don't need most of that. `vcclient-cachy` strips the engine to RVC-only, replaces the web GUI with a Textual TUI, integrates a persistent virtual mic via PipeWire, and ships as a proper Arch package.

## Goals (measured, not claimed)

- End-to-end latency `< 80 ms` (mic → transformed output)
- Idle VRAM `< 500 MB`
- CPU `< 15 %` while active
- Single `./install.sh`, runs in under 5 minutes on a fresh CachyOS

## Quick start

_See `docs/INSTALL.md` once Phase 6 ships._

## Credits

Built on the work of **[w-okada](https://github.com/w-okada)** and the original [voice-changer](https://github.com/w-okada/voice-changer) project. The original `LICENSE` is preserved at `upstream/LICENSE`. This fork's MIT license is at `LICENSE`.

## License

MIT — see `LICENSE`.
