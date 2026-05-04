# Changelog

All notable changes to this project. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Initial project scaffold: directory layout, MIT license with upstream attribution, README placeholder, progress tracking.
- `pyproject.toml` (hatchling, ruff, mypy strict, pytest), `.python-version` 3.11, isolated `uv` venv.
- `src/vcclient_cachy/cli.py` — `vcclient-cachy info` prints CUDA/PipeWire/Python versions.
- `tests/test_environment.py` (4/4 passing on host).
- `docs/00-recon.md` — 813-line reconnaissance of upstream `w-okada/voice-changer`. Identified hot path (9 files), 8 non-RVC engines for removal, ~22k LOC reduction target, and proposed `src/server/` layout for Phase 1.

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
