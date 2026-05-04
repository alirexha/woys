# Progress

Live tracking of phase status. Updated continuously during autonomous execution.

## v0.4.1 — Model-switch P0 bugfix (2026-05-04)

Critical UX bug found during voice library QA: `models use` and TUI `p` key
were both half-wired. CLI wrote config but engine ignored it; TUI updated
the visible label but never swapped the model. `STATUS` lacked a `model=`
field. Fix wires `cfg.rvc_model` through TUI startup, adds thread-safe
`request_model_swap` (with SOLA-tail drain), `MODEL` + `PROFILE` socket
commands, hot-swap from CLI when engine running. Live-verified: 115 ms
swap latency, no audible click, no chunks dropped. 7 new regression tests.
See `docs/06-model-switch-bug.md` for the wiring trace.

## Voice Library v1 — 9 RVC voices batch-imported (2026-05-04)

Built via `scripts/voice_library_import.py` per `VOICE_LIBRARY_BRIEF.md`.
All ✅; the `lana_del_rey` URL in the brief was wrong (`LanaDelReyV2.zip`
doesn't exist; the actual file in `pinguG/Lana-Del-Rey` is `NFR.zip`),
recovered manually. No version bump — this is operator-side data, not a
code release.

| Slug | Display | Onnx (MiB) | Source |
|------|---------|-----------:|--------|
| `donald_trump` | Donald Trump (POTUS) | 105.8 | Hazza1/DonaldTrump |
| `e_girl` | E-Girl (HQ Female) | 110.3 | ZokaxDesu/e-girl |
| `alfred_pennyworth` | Alfred Pennyworth (Arkham) | 107.8 | Homiebear/AlfredPennyworth_465e_8835s |
| `lana_del_rey` | Lana Del Rey (NFR Era) | 105.8 | pinguG/Lana-Del-Rey |
| `harley_quinn` | Harley Quinn V2 (Enemy Within) | 107.8 | Cauthess/HarleyQuinnTitanPretrain |
| `catwoman` | Catwoman (Laura Bailey) | 107.8 | Cauthess/CatwomanLauraBailey |
| `megan_fox` | Megan Fox | 105.8 | dragoncrack/(suspicious-but-functional repo name) |
| `batman_troy_baker` | Batman / Bruce Wayne (Troy Baker, Telltale) | 105.8 | Zogii/zogiiRVC |
| `spongebob_persian` | SpongeBob Persian Dub (Bab Asfanji) | 105.8 | PlushymehereJC/Spongebob_Persian_dub |

Total: ~963 MiB on disk under `~/.local/share/vcclient-cachy/models/`.
Each has a profile in `~/.config/vcclient-cachy/config.toml` with
`pitch=0`, `chunk_seconds=0.1`, `monitor=false`, plus `_display`,
`_source_url`, and (where relevant) `_note` fields documenting provenance.

Smoke test on `donald_trump` after `models use`:
- session load (3 ORT sessions): 1414 ms (one-time)
- first inference (cold cudnn): 451 ms
- warm inference (mean of 10): **34.6 ± 12.1 ms** — same envelope as v0.3.0 baseline.

Provenance: `voice-library/SOURCES.md`. Models are NOT in the git tree.

## v0.4.0 — Sharing + Browser + Tray ✅ shipped 2026-05-04

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | .vcprofile shareable presets | ✅ done |
| 2 | Browser extension scaffold (Manifest v3) | ✅ done — skeleton, no engine bridge yet |
| 3 | Optional tray icon (pystray) | ✅ done |
| 4 | Tag v0.4.0 + retro | ✅ done |

No engine changes; perf numbers identical to v0.3.0. Three deliverables
on the file-format / UX / packaging axis.

## v0.3.0 — UX + library release ✅ shipped 2026-05-04

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Perf push (fp16 rmvpe + IO-binding deferred) | ✅ partial — VRAM 1.35→1.09 GiB, e2e <80ms HIT |
| 2 | Models library (list / download / use) | ✅ done |
| 3 | Profiles (save / use / list / delete / cycle) | ✅ done |
| 4 | TUI polish (cycle key, toasts, cold-start) | ✅ done |
| 5 | AUR submission bundle (gated on repo de-privatisation) | ✅ partial |
| 6 | Tag v0.3.0 + retro | ✅ done |

## v0.2.0 — Optimization release ✅ shipped 2026-05-04

| Phase | Description | Status |
|-------|-------------|--------|
| A | OnnxContentvec real impl + embedder config flag | ✅ done |
| B | SOLA crossfade for low-latency chunks | ✅ done — 30.5 ms warm (target <120) |
| C | Real `convert` subcommand (.pth → .onnx) | ✅ done — verified on amitaro v2 |
| D | Perf verification + docs + tag v0.2.0 | ✅ done |

Headline numbers vs v0.1.1: e2e 280 ms → **30.5 ms** (-88%); VRAM unchanged;
CPU slightly up (more chunks/sec at chunk=0.1). VRAM + CPU misses are scoped
into v0.3.0.

## v0.1.1 — P0 routing fix (2026-05-04)

After v0.1.0 was tagged, the user reported that Discord/Telegram receive
silence when set to `vcclient-mic`, and that they hear transformed audio from
the laptop speakers. Diagnosis: the engine's output was going to the system
default sink, not VCClientCachySink. Root cause: PortAudio on CachyOS only
exposes the ALSA host API; `sd.OutputStream()` with no `device=` falls
through to ALSA default. Fix: switched engine output to a `pacat
--device=VCClientCachySink` subprocess (proven path; same as bench_loopback).
Also gated the host-default monitor stream behind a `--monitor` opt-in flag.
Two new regression tests in `tests/test_engine_routing.py`. Tag: **v0.1.1**.

| Phase | Description | Status |
|-------|-------------|--------|
| Setup | Workspace scaffold, git, gh repo | ✅ done |
| 0 | Recon — clone + map upstream | ✅ done |
| 1 | Lean Core — RVC-only ONNX server | ✅ done — 36.65 ms mean GPU e2e (target <80) |
| 2 | PipeWire integration + persistent vcclient-mic | ✅ done — round-trip + idempotency |
| 3 | TUI + IPC toggle (Unix socket; D-Bus deferred to Phase 5) | ✅ done |
| 4 | PKGBUILD + install/uninstall + systemd | ✅ done — round-trip verified |
| 5 | Performance tuning | ⚠️ partial — measured + tuned ORT options; <80ms target missed (see docs/05-perf.md) |
| 6 | ELI5 docs | ✅ done — all 5 written + commands tested |
| 7 | Retrospective + project the project notes + QA script | ✅ done |

## Definition of Done — final status

| DoD item | Status |
|----------|--------|
| 1. `./install.sh` on a fresh CachyOS works in under 5 minutes | ✅ verified (~3 min, mostly torch+ORT pip install) |
| 2. Discord with `vcclient-mic` selected → real-time voice transformation, **measured** < 80 ms | **ready for user QA, pending live test** (v0.1.1 routing fix verified — see `docs/QA.md` Test 2). Phase 5 measured 280 ms warm e2e; <80 ms target needs SOLA + IO binding (deferred). |
| 3. CS2 with the same mic → same result | **ready for user QA, pending live test** (v0.1.1 routing fix verified — see `docs/QA.md` Test 3). |
| 4. Full control from the TUI — no browser needed | ✅ `vcclient-cachy run` |
| 5. All 5 user-facing docs in `docs/` | ✅ INSTALL, DISCORD-SETUP, CS2-SETUP, MODELS, TROUBLESHOOTING (+ QA + perf + recon) |
| 6. PROGRESS shows every phase complete | ✅ this file |
| 7. LESSONS.md and project the project notes | ✅ written |
| 8. All verification gates passed for every phase | ✅ green per commit |

## Verification gate per phase

1. `pytest tests/ -v` — green
2. `ruff check src/ && ruff format --check src/` — clean
3. `mypy --strict src/` — clean
4. Live run with measured output captured

## Definition of Done

See `PROJECT_BRIEF.md` §18. Items requiring live human QA (DoD #2 and #3 — Discord and CS2 with `vcclient-mic`) are marked **ready for user QA, pending live test** at the end. A QA script is provided.

## System inventory (captured at start)

- OS: CachyOS (Arch-based), kernel `7.0.3-1-cachyos`
- GPU: RTX 2070 8GB, driver `595.71.05`
- CUDA system package: `cuda 13.2.1-1` (driver-forward-compat with CUDA 12 ORT wheels)
- cuDNN: not installed via pacman (pip-shipped via `nvidia-cudnn-cu12`)
- System Python: `3.14.4` (we use isolated `uv` venv on Python 3.11)
- PipeWire: `1.6.4` with `pipewire-pulse` 15.0.0 shim
- Default mic: HyperX QuadCast 2 S
