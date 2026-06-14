#!/usr/bin/env bash
# woys installer — local user install (no sudo for the venv).
#
# Lays down:
#   $HOME/.local/share/woys/{venv,models}
#   $HOME/.local/bin/woys                     (symlink into the venv)
#   $HOME/.config/systemd/user/woys-mic.service
#
# v0.6.0: detects an existing vcclient-cachy install and migrates it
# losslessly before installing the new code (config + models + systemd
# unit all move).
# v0.6.5: PipeWire mic name renamed `vcclient-mic` → `woys-mic`. Apps
# (Discord / CS2 / Telegram) need to re-select their input device once.
# v0.8.0: the deprecated `vcclient-cachy` shim was removed entirely (kept
# through v0.6.x as a transition tool). The script still cleans up any
# stale shim a previous install left behind.
#
# Pre-reqs (the script checks):
#   - PipeWire + pipewire-pulse running
#   - NVIDIA driver + CUDA-capable GPU
#   - Python 3.11 (uv installs one if missing)
#   - uv (we'll fetch it user-local if absent)
#
# Usage:
#   ./install.sh              # full install
#   ./install.sh --skip-models  # don't pre-fetch ONNX weights (~1 GiB)
#   ./install.sh --no-systemd   # don't register the user service
#   ./install.sh --allow-cpu    # proceed without an NVIDIA GPU (unsupported:
#                               # realtime RVC is not viable on CPU)

set -euo pipefail

# validate $HOME before deriving
# paths from it. The install script writes systemd unit files and
# rmrf's directories under $HOME -- if $HOME is unset, relative, or
# owned by a different UID, those operations land in the wrong
# place. Pre-fix the script trusted $HOME blindly.
if [ -z "${HOME:-}" ]; then
    printf '\n[install] error: HOME is unset; refusing to derive paths.\n' >&2
    exit 1
fi
case "$HOME" in
    /*) ;;  # absolute -- OK
    *)
        printf '\n[install] error: HOME=%s is not an absolute path; refusing.\n' "$HOME" >&2
        exit 1
        ;;
esac
if [ ! -d "$HOME" ]; then
    printf '\n[install] error: HOME=%s does not exist or is not a directory.\n' "$HOME" >&2
    exit 1
fi
# Ownership check: $HOME must be owned by the running UID. A scenario
# where $HOME points at another user's directory (root-installed venv
# pulling our scripts; container with a misconfigured passwd) lands
# our systemd units + rmrf operations in the wrong place. `stat -c
# %u` works on coreutils + busybox + macOS BSD-stat-coreutils.
_home_uid="$(stat -c '%u' "$HOME" 2>/dev/null || stat -f '%u' "$HOME")"
if [ "${_home_uid:-}" != "$(id -u)" ]; then
    printf '\n[install] error: HOME=%s is owned by uid=%s, not our uid=%s; refusing.\n' \
        "$HOME" "${_home_uid:-?}" "$(id -u)" >&2
    exit 1
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_HOME="$HOME/.local/share/woys"
OLD_APP_HOME="$HOME/.local/share/vcclient-cachy"
VENV="$APP_HOME/venv"
BIN_DIR="$HOME/.local/bin"
SYSTEMD_USER_DIR="$HOME/.config/systemd/user"
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"

SKIP_MODELS=0
NO_SYSTEMD=0
ALLOW_CPU=0
for arg in "$@"; do
    case "$arg" in
    --skip-models) SKIP_MODELS=1 ;;
    --no-systemd)  NO_SYSTEMD=1 ;;
    --allow-cpu)   ALLOW_CPU=1 ;;
    -h|--help)
        sed -n '2,28p' "${BASH_SOURCE[0]}" | sed 's/^# //;s/^#//'
        exit 0
        ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
    esac
done

say() { printf '\n[install] %s\n' "$*"; }
fail() { printf '\n[install] error: %s\n' "$*" >&2; exit 1; }

# ---- pre-reqs -----------------------------------------------------------------
#
# the destructive vcclient-cachy→woys migration used
# to run HERE, before the prereq checks and venv build. With `set -e`, a
# venv-build failure aborted the script with the old install already
# dismantled and the new one unbuilt — no rollback. The migration block
# now runs *after* the venv + deps are proven buildable (see below).

say "checking host…"
command -v pactl >/dev/null   || fail "pactl missing — install pipewire-pulse"
pactl info | grep -q PipeWire || fail "PipeWire not running (pactl info reports something else)"
# hard-fail on a missing NVIDIA GPU. The
# pre-fix `say "warn: ... fall back to CPU"` advertised a CPU fallback
# that does not exist — ONNX Runtime CUDA-EP sessions do not silently
# become CPU sessions, and realtime RVC on CPU is not viable. --allow-cpu
# is the explicit, documented opt-out for users who understand that.
if ! command -v nvidia-smi >/dev/null; then
    if [ "$ALLOW_CPU" -eq 1 ]; then
        say "  warn: nvidia-smi missing — proceeding with --allow-cpu (UNSUPPORTED;"
        say "        realtime RVC is not viable on CPU, expect continuous underruns)"
    else
        fail "nvidia-smi not found — woys requires an NVIDIA GPU (ONNX Runtime CUDA EP); realtime RVC on CPU is not viable. Re-run with --allow-cpu to override."
    fi
fi

# Install uv if absent.
#
# pre-fix this was a bare
# `curl | sh` pipeline -- inconsistent with the project's own SHA-
# rigor for model downloads. Hard-fail with a clear message if uv
# isn't already installed; let the user choose the install method
# (pip, distro package, Astral's official script). The previous
# automatic install was a convenience that crossed a security
# boundary the rest of the project enforces.
if [ ! -x "$UV_BIN" ]; then
    fail "uv (Astral) is required but not found at $UV_BIN.
       Install it with ONE of:
         pip install --user uv     # via PyPI (recommended; pinnable)
         pacman -S uv              # CachyOS / Arch
         brew install uv           # macOS
         curl -LsSf https://astral.sh/uv/install.sh | sh   # Astral's own (NOT auto-run anymore -- F-05-09)
       Then re-run ./install.sh."
fi

# ---- venv ---------------------------------------------------------------------

mkdir -p "$APP_HOME" "$BIN_DIR" "$SYSTEMD_USER_DIR"

"$UV_BIN" python install 3.11 >/dev/null
if [ -x "$VENV/bin/python" ]; then
    say "reusing existing venv at $VENV (skip 5 GB re-download)…"
    # B38 / pkg-005: don't blanket-suppress stderr — surface real errors.
    if ! "$VENV/bin/python" -m pip uninstall -y vcclient-cachy >/dev/null; then
        say "warning: failed to uninstall stale vcclient-cachy wheel; continuing"
    fi
else
    say "creating Python 3.11 venv at $VENV…"
    "$UV_BIN" venv --python 3.11 "$VENV" >/dev/null
fi

say "installing woys + runtime deps (long step on fresh installs — torch + ORT-GPU)…"
# install the pinned dependency closure
# (requirements.txt) FIRST, then the woys package itself with --no-deps.
# Pre-fix this was `pip install -e .` then `pip install -r
# requirements.txt` -- an order-dependent double-install where the second
# command silently re-resolved whatever the first installed if the two
# files diverged, and the slow step (torch + ORT-GPU) was paid twice.
# requirements.txt is now the single source of what gets installed.
"$UV_BIN" pip install --python "$VENV/bin/python" -r "$REPO_DIR/requirements.txt" >/dev/null
"$UV_BIN" pip install --python "$VENV/bin/python" --no-deps -e "$REPO_DIR" >/dev/null

# ---- v0.6.0 migration: vcclient-cachy → woys ---------------------------------
#
# runs only AFTER the prereq checks and the venv +
# deps build above have all succeeded — so a failed install can never
# dismantle a working vcclient-cachy setup. The venv python is guaranteed
# to exist at this point.

if [ -d "$OLD_APP_HOME" ] || [ -d "$HOME/.config/vcclient-cachy" ] || [ -d "$HOME/.cache/vcclient-cachy" ]; then
    say "detected an existing vcclient-cachy install — migrating to woys…"
    "$VENV/bin/python" "$REPO_DIR/scripts/migrate_to_woys.py" || fail "migration failed"
fi

# ---- v0.9.0 native PipeWire helper --------------------------------------------

# v0.9.0 ships a small native PipeWire client (~250 LOC C) that replaces
# the pw-cat / pacat subprocess on the playback path. Build it now so
# users can opt in via prefer_native_pw=true. Hard-fail if the build
# tools are missing — the helper is the headline v0.9.0 fix.
say "building native PipeWire helper (bin/woys-pw-out)…"
if ! command -v gcc >/dev/null 2>&1; then
    say "warning: gcc not found; skipping native helper build"
    say "         (install gcc + pipewire-dev to enable prefer_native_pw)"
elif ! pkg-config --exists libpipewire-0.3 2>/dev/null; then
    say "warning: libpipewire-0.3 dev headers missing; skipping native helper"
    say "         (pacman -S pipewire to install)"
else
    if ! make -C "$REPO_DIR/bin" >/dev/null; then
        say "warning: native helper build failed; prefer_native_pw will hard-fail"
    else
        install -Dm755 "$REPO_DIR/bin/woys-pw-out" "$BIN_DIR/woys-pw-out"
        say "installed native helper: $BIN_DIR/woys-pw-out"
    fi
fi

# ---- launcher symlink ---------------------------------------------------------

LINK="$BIN_DIR/woys"
ln -sfn "$VENV/bin/woys" "$LINK"
say "linked launcher: $LINK -> $VENV/bin/woys"

# B39 / pkg-009: the deprecated `vcclient-cachy` shim's own comment said
# "removed in v0.7.0". We are in v0.8.0 now. Stop installing it; remove
# any stale shim that earlier installs left behind.
SHIM_OLD="$BIN_DIR/vcclient-cachy"
if [ -e "$SHIM_OLD" ]; then
    rm -f "$SHIM_OLD"
    say "removed obsolete vcclient-cachy shim (deprecated in v0.6.0, dropped in v0.8.0)"
fi

case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *) say "  note: $BIN_DIR is not on \$PATH — add this to your shell rc:" \
            && printf '          fish: fish_add_path %s\n' "$BIN_DIR" \
            && printf '          bash/zsh: export PATH="%s:\$PATH"\n' "$BIN_DIR" ;;
esac

# ---- models -------------------------------------------------------------------

if [ "$SKIP_MODELS" -eq 0 ]; then
    say "fetching ONNX foundation weights into $APP_HOME/models/…"
    "$VENV/bin/python" "$REPO_DIR/scripts/download_weights.py"
else
    say "skipping model download (run scripts/download_weights.py later)"
fi

# ---- systemd user unit --------------------------------------------------------

if [ "$NO_SYSTEMD" -eq 0 ]; then
    install -m 0644 "$REPO_DIR/pkg/woys-mic.service" "$SYSTEMD_USER_DIR/"
    # B38 / pkg-005: surface failure of the systemctl steps so the user
    # learns about it now, not when the mic doesn't auto-load on next boot.
    if ! systemctl --user daemon-reload; then
        say "warning: systemctl daemon-reload failed; continuing"
    fi
    if ! systemctl --user enable --now woys-mic.service; then
        say "warning: failed to enable woys-mic.service; continuing"
    fi
    say "woys-mic.service enabled (Discord/CS2 see woys-mic at boot)."

    # v0.6.8 — prune accumulated config backups left by prior in-place
    # patches (.bak-leak, .bak-microcut-*, .bak-pacat-*). Keep the single
    # most recent for rollback; delete the rest. Best-effort.
    config_dir="$HOME/.config/woys"
    if [ -d "$config_dir" ]; then
        bak_files=$(find "$config_dir" -maxdepth 1 -name 'config.toml.bak-*' 2>/dev/null | sort)
        bak_count=$(printf '%s\n' "$bak_files" | sed '/^$/d' | wc -l)
        if [ "$bak_count" -gt 1 ]; then
            kept=$(printf '%s\n' "$bak_files" | sed '/^$/d' | tail -1)
            say "pruning $((bak_count - 1)) old config backups (kept latest: $(basename "$kept"))"
            printf '%s\n' "$bak_files" | sed '/^$/d' | head -n -1 | while read -r f; do
                rm -f "$f"
            done
        fi
    fi

    # Migration cleanup — if any legacy module is still loaded from a prior
    # install (sink: VCClientCachySink, source: vcclient-mic), drop it now
    # so PipeWire doesn't carry old and new side-by-side. Best-effort.
    if command -v pactl >/dev/null; then
        for mod_id in $(pactl list short modules 2>/dev/null \
            | awk -F'\t' '/sink_name=VCClientCachySink/ {print $1}'); do
            say "removing orphan VCClientCachySink module ($mod_id)…"
            pactl unload-module "$mod_id" 2>/dev/null || true
        done
        for mod_id in $(pactl list short modules 2>/dev/null \
            | awk -F'\t' '/source_name=vcclient-mic/ {print $1}'); do
            say "removing orphan vcclient-mic source module ($mod_id)…"
            pactl unload-module "$mod_id" 2>/dev/null || true
        done
    fi
else
    say "skipping systemd unit registration (--no-systemd)"
fi

# ---- post-install verification ------------------------------------------------

# B40 / pkg-011: run `woys --version` to confirm the binary actually works.
# Catches transitively-failed deps / broken venvs before the user discovers
# them mid-`woys run`.
if ! "$LINK" --version >/dev/null 2>&1; then
    say "ERROR: $LINK does not start. Inspect:"
    say "    $VENV/bin/python -c 'import woys; print(woys.__version__)'"
    say "    $VENV/bin/python -c 'from audio.engine import EngineConfig'"
    exit 1
fi
INSTALLED_VERSION="$("$LINK" --version 2>/dev/null | head -1)"

# verify ALL THREE foundation weights, not just
# amitaro. A non-`--skip-models` install that is missing any of them is
# broken — fail rather than print a reassuring summary.
MODELS_DIR="$APP_HOME/models"
MISSING_MODELS=""
for m in rmvpe_wrapped.onnx contentvec-f.onnx amitaro_v2_16k.onnx; do
    [ -f "$MODELS_DIR/$m" ] || MISSING_MODELS="${MISSING_MODELS}${m} "
done
if [ "$SKIP_MODELS" -eq 1 ]; then
    MODELS_STATUS="skipped (--skip-models; run scripts/download_weights.py later)"
elif [ -n "$MISSING_MODELS" ]; then
    fail "install incomplete — missing foundation weights: ${MISSING_MODELS}(run scripts/download_weights.py)"
else
    MODELS_STATUS="all 3 foundation weights present"
fi

# ---- summary -----------------------------------------------------------------

cat <<EOF

[install] done.

  launcher : $LINK   ($INSTALLED_VERSION)
  venv     : $VENV
  models   : $MODELS_DIR   ($MODELS_STATUS)
  service  : $SYSTEMD_USER_DIR/woys-mic.service

  next steps:
    woys info            # sanity check
    woys pw status       # confirm woys-mic exists
    woys run             # launch the TUI

  v0.6.5 — PipeWire mic name is now 'woys-mic' (was 'vcclient-mic').
  Re-select the input device once in Discord / CS2 / Telegram / etc.
EOF
