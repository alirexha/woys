# Changelog

All notable changes to this project. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed (post-v0.1.0 housekeeping)
- **Repo visibility flipped to PRIVATE** on GitHub (`gh repo edit alirexha/vcclient-cachy --visibility private`).
- **Root `LICENSE` switched from MIT to "All Rights Reserved"** for the original work pending a commercial decision. `upstream/LICENSE` (w-okada's MIT) preserved verbatim — that subtree and the vendored derivatives in `src/server/` remain MIT.
- Added top-level `NOTICE` file establishing the file-by-file license boundary between original work (proprietary) and upstream-derived code (MIT). This is the audit trail for future legal review.
- `README.md` rewritten: removed MIT framing for original work, added "private alpha — not for redistribution" banner, added a license-table section pointing at `NOTICE` for the full audit.
- `pyproject.toml` classifiers updated: `License :: Other/Proprietary License` + `Private :: Do Not Upload`.
- `pkg/PKGBUILD` `license=('custom' 'MIT')` reflects the dual licensing; install also drops `NOTICE` into `/usr/share/licenses/$pkgname/`.
- `the project notes` updated with private-repo + license-boundary rules in "Things to never do".
- `docs/00-recon.md` had one absolute path (`/home/alireza/ai/vcclient-cachy/upstream/`) sanitized to `<repo>/upstream/`.

### Audit (clean — nothing scrubbed from history)
- No model binaries (`*.onnx`, `*.pth`, `*.pt`, `*.bin`, `*.safetensors`) were ever committed (tree or history).
- No secrets / API tokens / `.env` files / credential files in the repo.
- `.gitignore` audited and confirmed comprehensive (Python, models, audio, env, editor caches, `./settings.local.json`, `upstream/`).

## [0.1.0] — 2026-05-04

### Added
- Initial project scaffold: directory layout, MIT license with upstream attribution, README placeholder, progress tracking.
- `pyproject.toml` (hatchling, ruff, mypy strict, pytest), `.python-version` 3.11, isolated `uv` venv.
- `src/vcclient_cachy/cli.py` — `vcclient-cachy info` prints CUDA/PipeWire/Python versions.
- `tests/test_environment.py` (4/4 passing on host).
- `docs/00-recon.md` — 813-line reconnaissance of upstream `w-okada/voice-changer`. Identified hot path (9 files), 8 non-RVC engines for removal, ~22k LOC reduction target, and proposed `src/server/` layout for Phase 1.

### Phase 7 — Retrospective + handover
- `LESSONS.md` (202 lines) — execution summary, honest scorecard against brief targets, unexpected challenges, mistakes, what was learned, recommendations for the next session. Calls out that the brief's "FORBIDDEN list" was load-bearing.
- `the project notes` (project-level, 108 lines) — startup guide for the next CC session: 3-sentence summary, "read LESSONS.md first" instruction, architectural decisions + their *why*, build/test/run commands, known gotchas, "things to never do" checklist.
- `docs/QA.md` (141 lines) — step-by-step live QA script for the user to validate DoD items #2 (Discord) and #3 (CS2). Engine on/off via CLI toggle, Discord/CS2 mic configuration, long-session stability, clean shutdown.
- Updated `PROGRESS.md` with the full Definition of Done table — items #2 and #3 marked "ready for user QA, pending live test" per Q9.

### Phase 6 — ELI5 documentation
- `docs/INSTALL.md` — step-by-step install for someone who's never used Python on Linux. Verifies PipeWire, walks through `./install.sh`, sanity-checks the install, sets PATH on fish vs bash/zsh.
- `docs/DISCORD-SETUP.md` — Discord input device + critical "disable Discord noise suppression / Krisp" note (it gates RVC output as noise). Covers the auto-detect-other-device gotcha and a KDE/GNOME shortcut binding for `vcclient-cachy toggle`.
- `docs/CS2-SETUP.md` — CS2 audio config + an explicit anti-cheat note (vcclient-cachy is OS-level audio, not memory hooking — VAC-safe by default; evdev hotkey opt-in is the only thing flagged risky).
- `docs/MODELS.md` — where models live, where to find them on HF/weights.gg, three `.pth → .onnx` paths (upstream Docker UI, manual `torch.onnx.export` recipe, future `vcclient-cachy convert` subcommand).
- `docs/TROUBLESHOOTING.md` — the failure tree from "PulseAudio detected" to "voice sounds robotic" to "engine drops audio every 30s". Covers cuDNN preload, GPU memory, Krisp gating, and the evdev opt-in (with the VAC warning).
- Added `vcclient-cachy convert` CLI **stub** that prints the manual paths from `MODELS.md`. **Real implementation deferred**; the slot-metadata probe needed to wrap upstream's `export2onnx` cleanly is a 1-2 hour task on its own. Honest miss against Q5; flagged in `LESSONS.md`.
- All shell commands in docs verified working on this CachyOS host (re-ran `install.sh` after Phase 4's uninstall test, confirmed `vcclient-cachy {info, pw status}` and PipeWire listings).

### Phase 5 — Performance numbers
- `docs/05-perf.md` — full measured numbers, hardware/software baseline, methodology, and targets-vs-reality.
- Aligned `audio/engine.py:_make_session()` with the smoke-test ORT options: `arena_extend_strategy=kNextPowerOfTwo`, `cudnn_conv_algo_search=EXHAUSTIVE`, `do_copy_in_default_stream=True`. Steady-state engine inference dropped 86 → 60 ms (rolling-32 avg @ chunk=0.25).
- Chunk-size sweep (60-500 ms): **inference is roughly constant at ~22 ms** for chunk sizes ≥ 100 ms. Below that, kernel-launch overhead dominates and inference *increases*. Sweet spot: 100-150 ms chunks.
- `scripts/bench_chunks.py`-style chunk sweep is wired through `scripts/smoke_rvc_onnx.py`. Acoustic loopback `scripts/bench_loopback.py` is scaffolded but the subprocess timing alignment is fragile — documented as future work; in-process numbers are authoritative.
- **Honest verdict**: brief targets *missed* on this hardware:
  - e2e (target <80 ms): **~280 ms** measured warm-state at chunk=0.25 (250 ms audio buffer + ~25 ms inference + ~5 ms audio I/O).
  - Idle VRAM (target <500 MB): **~1.35 GiB** (contentvec-f and rmvpe are both ~350 MB on disk fp32).
  - CPU active (target <15%): **~26%** at chunk=0.25.
  All three misses traceable to model architecture choices; closing them needs SOLA + IO-binding + fp16 export, which the brief permits but are deferred to future sessions.
- TensorRT EP available in the wheel but its runtime libs aren't pip-shipped — falls back to CPU. Skipped (avoids worse-than-CUDA fallback path).

### Phase 4 — Packaging
- `install.sh` — user-local installer. Creates `~/.local/share/vcclient-cachy/{venv,models}`, installs deps (auto-fetches `uv` if missing), symlinks `~/.local/bin/vcclient-cachy`, registers + enables `vcclient-cachy-mic.service`. Pre-flight checks PipeWire and warns on missing nvidia-smi. Flags: `--skip-models`, `--no-systemd`.
- `uninstall.sh` — reverses install.sh. Stops and removes systemd unit, tears down the PipeWire mic via `vcclient-cachy pw teardown`, removes launcher symlink. `--keep-models` preserves the ~1 GiB ONNX cache. Always preserves user config at `~/.config/vcclient-cachy/`.
- `pkg/PKGBUILD` — AUR-ready Arch package: deps (`pipewire`, `pipewire-pulse`, `pipewire-alsa`, `nvidia-utils`, `python>=3.11`), system-wide install via wheel + `python-installer`, ships license preserving upstream attribution and the systemd user unit. Not published to AUR (Q8: GitHub only).
- Verified: install.sh round-trips cleanly. After install, `vcclient-cachy info`, `pw status`, and the systemd unit all work; `uninstall.sh --keep-models` removes everything except the model cache and config.

### Phase 3 — TUI + control surface
- `src/audio/engine.py` — `RealtimeEngine` wraps the proven Phase 1 inference path in a sounddevice mic→infer→sink worker thread. ORT sessions lazy-load on first start; `process_chunk_16k` returns a `(N,) float32` audio buffer. Live verified: starts, processes chunks, stops cleanly with no errors.
- `src/tui/app.py` — Textual TUI: toggle (`t`), pitch +/- (`+`/`-`/`0`), save (`s`), quit (`q`). Status + latency panels + input level meter, polled every 250 ms.
- `src/tui/config.py` — `~/.config/vcclient-cachy/config.toml` round-trip with extras pass-through (unknown keys preserved on save).
- **Pragmatic IPC pivot**: replaced D-Bus with a Unix-socket control channel at `$XDG_RUNTIME_DIR/vcclient-cachy/control.sock`. dasbus needs a GLib mainloop alongside Textual's asyncio loop — non-trivial integration. Unix sockets give the same UX (KDE/GNOME shortcut → `vcclient-cachy toggle`) with zero loop conflicts. **D-Bus moved to Phase 5 polish.**
- New CLI subcommands: `vcclient-cachy {run, toggle, status, pitch ±N}`.
- `src/tui/hotkey.py` — opt-in evdev global hotkey (per Q7: VAC-safe by default, enable explicitly via `enable_evdev_hotkey=true` + `pip install -e .[evdev]`). Stub structure ready; full input-group/udev docs pending Phase 6.
- `pkg/vcclient-cachy-mic.service` updated path is unchanged; no impact.
- 7 new tests (config × 4, control × 3). All gates green: pytest 14/14 fast + 1/1 GPU (37.55 ± 10.18 ms still under target), ruff clean, mypy strict clean (10 source files).

### Phase 2 — PipeWire integration
- `src/audio/pipewire.py` — `VirtualMic` shells out to `pactl` to load `module-null-sink` (`VCClientCachySink`) and `module-remap-source` (`vcclient-mic`) so apps see the mic as a normal input.
- Idempotent `ensure()`/`teardown()`. `ensure_pipewire()` hard-fails with a clear paru hint if the host is on PulseAudio instead of PipeWire.
- Discovered `object.linger=true` leaves orphan PipeWire *nodes* after module unload — defaulted to `linger=False` since modules persist across pactl client lifetime anyway. Added `_destroy_orphan_nodes()` (uses `pw-cli`) as a defensive cleanup so users who hit linger=true once can recover.
- CLI: `vcclient-cachy pw {setup,teardown,status}` — exit 0 if both modules present.
- `pkg/vcclient-cachy-mic.service` — systemd user unit, `Type=oneshot RemainAfterExit=yes`, calls `pw setup` at login. Discord/CS2 see `vcclient-mic` at boot regardless of whether the engine is running.
- New tests in `tests/test_audio_pipewire.py`: round-trip + idempotency + missing-pactl error path.

### Phase 1 — Lean Core
- Vendored `upstream/server/` → `src/server/`, then trimmed:
  - Deleted 8 non-RVC engines (Beatrice, DDSP_SVC, DiffusionSVC, EasyVC, LLVC, MMVCv13, MMVCv15, SoVitsSvc40), V1 `VoiceChanger.py`, `test.wav`, `.vscode/`, win/mac shell scripts.
  - Result: **35,089 → 12,881 LOC, 240 → 112 files** (≈63% reduction).
- Rehomed `DiffusionSVC/pitchExtractor/rmvpe/` → `RVC/pitchExtractor/rmvpe/` and redirected the two RVC RMVPE extractors to use the local `PitchExtractor` Protocol.
- Stripped Mac/Windows branches in `MMVCServerSIO.py` (native client launch, `_MEIPASS` reload guard) and `restapi/MMVC_Rest.py` (Mac `_MEIPASS` model_dir, `/trainer` and `/recorder` mounts). Stripped WASAPI exclusive-mode block in `Local/ServerDevice.py`. Stripped Beatrice/LLVC `noCrossFade` and `LLVC` post-padding branches in `VoiceChangerV2.py`.
- Collapsed `VoiceChangerManager.loadModel` and `generateVoiceChanger` to RVC-only single-arm dispatch (was 9 arms each). Dropped legacy `VoiceChanger` (V1) import; `VoiceChangerV2` is the only runner.
- Bumped runtime deps: `onnxruntime-gpu 1.22.0`, `torch 2.5.1+cu124`, `cuDNN 9.1` (pip-shipped), `fastapi 0.115`, `uvicorn 0.46`. Pinned via `uv pip compile pyproject.toml -o requirements.txt`.
- Smoke test (`scripts/smoke_rvc_onnx.py` + `tests/test_smoke_rvc_onnx.py`): full ONNX path on RTX 2070, 1 s @ 16 kHz clip:
  - **mean 36.65 ms ± 9.44 ms** (min 28.90, max 50.45) — well under 80 ms Phase 1 floor.
  - contentvec 7.55 ms · rmvpe 17.12 ms · RVC inferencer 13.86 ms.
- Discovered `ort.preload_dlls()` is required for ORT-GPU 1.20+ to find pip-shipped CUDA libs on systems without the libs in `LD_LIBRARY_PATH`.
- `src/server/` is excluded from ruff/mypy gates for now — vendored code, incremental cleanup planned. Authored modules (`src/{vcclient_cachy,audio,tui}/`) are mypy-strict + ruff clean.

### Discovered (Phase 0 highlights)
- `OnnxContentvec` is a stub upstream — every "ONNX RVC" run silently uses PyTorch+fairseq for the embedder. Phase 1 keeps PyTorch as a hard dep; ONNX-only embedder is a future optimization.
- Upstream `requirements.txt` is missing `fairseq` and `pyworld` — they ship via Docker, not pip. Will add to fork.
- `onnxruntime-gpu==1.13.1` and `torch==2.0.1` are mid-2022 vintage; bumping to ORT 1.20+ and torch ≥ 2.4 (CUDA 12 wheels) for driver 595 forward-compat.
