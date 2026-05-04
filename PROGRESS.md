# Progress

Live tracking of phase status. Updated continuously during autonomous execution.

| Phase | Description | Status |
|-------|-------------|--------|
| Setup | Workspace scaffold, git, gh repo | ✅ done |
| 0 | Recon — clone + map upstream | ✅ done |
| 1 | Lean Core — RVC-only ONNX server | ✅ done — 36.65 ms mean GPU e2e (target <80) |
| 2 | PipeWire integration + persistent vcclient-mic | ✅ done — round-trip + idempotency |
| 3 | TUI + D-Bus toggle | in progress |
| 4 | PKGBUILD + install/uninstall | pending |
| 5 | Performance tuning | pending |
| 6 | ELI5 docs | pending |
| 7 | Retrospective + project the project notes + QA script | pending |

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
