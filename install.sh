#!/usr/bin/env bash
# woys installer — local user install (no sudo for the venv).
#
# Lays down:
#   $HOME/.local/share/woys/{venv,models}
#   $HOME/.local/bin/woys                     (symlink into the venv)
#   $HOME/.local/bin/vcclient-cachy           (deprecated shim — prints warning, delegates to woys)
#   $HOME/.config/systemd/user/woys-mic.service
#
# v0.6.0: detects an existing vcclient-cachy install and migrates it
# losslessly before installing the new code (config + models + systemd
# unit all move). The PipeWire mic name (`vcclient-mic`) is unchanged so
# Discord / CS2 / Telegram don't need re-configuration.
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

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_HOME="$HOME/.local/share/woys"
OLD_APP_HOME="$HOME/.local/share/vcclient-cachy"
VENV="$APP_HOME/venv"
BIN_DIR="$HOME/.local/bin"
SYSTEMD_USER_DIR="$HOME/.config/systemd/user"
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"

SKIP_MODELS=0
NO_SYSTEMD=0
for arg in "$@"; do
    case "$arg" in
    --skip-models) SKIP_MODELS=1 ;;
    --no-systemd)  NO_SYSTEMD=1 ;;
    -h|--help)
        sed -n '2,24p' "${BASH_SOURCE[0]}" | sed 's/^# //;s/^#//'
        exit 0
        ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
    esac
done

say() { printf '\n[install] %s\n' "$*"; }
fail() { printf '\n[install] error: %s\n' "$*" >&2; exit 1; }

# ---- v0.6.0 migration: vcclient-cachy → woys ---------------------------------

if [ -d "$OLD_APP_HOME" ] || [ -d "$HOME/.config/vcclient-cachy" ] || [ -d "$HOME/.cache/vcclient-cachy" ]; then
    say "detected an existing vcclient-cachy install — migrating to woys…"
    PY="$(command -v python3 || command -v python || true)"
    if [ -z "$PY" ]; then
        fail "python3 not found — install python before running this script"
    fi
    "$PY" "$REPO_DIR/scripts/migrate_to_woys.py" || fail "migration failed"
fi

# ---- pre-reqs -----------------------------------------------------------------

say "checking host…"
command -v pactl >/dev/null   || fail "pactl missing — install pipewire-pulse"
pactl info | grep -q PipeWire || fail "PipeWire not running (pactl info reports something else)"
command -v nvidia-smi >/dev/null || say "  warn: nvidia-smi missing — engine will fall back to CPU"

# Install uv if absent.
if [ ! -x "$UV_BIN" ]; then
    say "installing uv (Astral) into $HOME/.local/bin…"
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# ---- venv ---------------------------------------------------------------------

mkdir -p "$APP_HOME" "$BIN_DIR" "$SYSTEMD_USER_DIR"

"$UV_BIN" python install 3.11 >/dev/null
if [ -x "$VENV/bin/python" ]; then
    say "reusing existing venv at $VENV (skip 5 GB re-download)…"
    # Drop the old vcclient-cachy wheel before installing the woys wheel —
    # the v0.6.0 rename means the old console script is dead and needs to
    # not shadow the new one.
    "$VENV/bin/python" -m pip uninstall -y vcclient-cachy >/dev/null 2>&1 || true
else
    say "creating Python 3.11 venv at $VENV…"
    "$UV_BIN" venv --python 3.11 "$VENV" >/dev/null
fi

say "installing woys + runtime deps (long step on fresh installs — torch + ORT-GPU)…"
"$UV_BIN" pip install --python "$VENV/bin/python" -e "$REPO_DIR" >/dev/null
"$UV_BIN" pip install --python "$VENV/bin/python" -r "$REPO_DIR/requirements.txt" >/dev/null

# ---- launcher symlink ---------------------------------------------------------

LINK="$BIN_DIR/woys"
ln -sfn "$VENV/bin/woys" "$LINK"
say "linked launcher: $LINK -> $VENV/bin/woys"

# v0.6.0 — backward-compat shim. `vcclient-cachy` keeps working but prints
# a deprecation warning, then exec's the new binary with the same args.
# Remove any stale symlink first — the old install pointed it at
# venv/bin/vcclient-cachy which no longer exists.
SHIM="$BIN_DIR/vcclient-cachy"
rm -f "$SHIM"
cat > "$SHIM" <<'SHIM_EOF'
#!/usr/bin/env bash
# Deprecated shim — `vcclient-cachy` was renamed to `woys` in v0.6.0.
# This wrapper will be removed in v0.7.0. Update your scripts / muscle memory.
printf '\033[33m[deprecation]\033[0m vcclient-cachy is deprecated, use `woys` (this shim will be removed in v0.7.0)\n' >&2
exec woys "$@"
SHIM_EOF
chmod +x "$SHIM"
say "linked deprecated shim: $SHIM -> woys (delete in v0.7.0)"

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
    systemctl --user daemon-reload || true
    systemctl --user enable --now woys-mic.service || true
    say "woys-mic.service enabled (Discord/CS2 see vcclient-mic at boot)."

    # v0.6.0 migration cleanup — if a legacy VCClientCachySink module is
    # still loaded (the old systemd ExecStop didn't always unload it
    # during the rename), drop it now so PipeWire doesn't carry both the
    # old and new sinks side-by-side. Best-effort, ignore failures.
    if command -v pactl >/dev/null; then
        for mod_id in $(pactl list short modules 2>/dev/null \
            | awk -F'\t' '/sink_name=VCClientCachySink/ {print $1}'); do
            say "removing orphan VCClientCachySink module ($mod_id)…"
            pactl unload-module "$mod_id" 2>/dev/null || true
        done
    fi
else
    say "skipping systemd unit registration (--no-systemd)"
fi

# ---- summary -----------------------------------------------------------------

cat <<EOF

[install] done.

  launcher : $LINK
  shim     : $SHIM  (deprecated, removed in v0.7.0)
  venv     : $VENV
  models   : $APP_HOME/models   ($([ -f "$APP_HOME/models/amitaro_v2_16k.onnx" ] && echo present || echo missing))
  service  : $SYSTEMD_USER_DIR/woys-mic.service

  next steps:
    woys info            # sanity check
    woys pw status       # confirm vcclient-mic exists
    woys run             # launch the TUI

  PipeWire mic name is still 'vcclient-mic' — your Discord/CS2/Telegram
  setup keeps working without re-configuration.
EOF
