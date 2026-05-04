#!/usr/bin/env bash
# vcclient-cachy installer — local user install (no sudo for the venv).
#
# Lays down:
#   $HOME/.local/share/vcclient-cachy/{venv,models}
#   $HOME/.local/bin/vcclient-cachy           (symlink into the venv)
#   $HOME/.config/systemd/user/vcclient-cachy-mic.service
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
APP_HOME="$HOME/.local/share/vcclient-cachy"
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
        sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# //;s/^#//'
        exit 0
        ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
    esac
done

say() { printf '\n[install] %s\n' "$*"; }
fail() { printf '\n[install] error: %s\n' "$*" >&2; exit 1; }

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

say "creating Python 3.11 venv at $VENV…"
"$UV_BIN" python install 3.11 >/dev/null
"$UV_BIN" venv --python 3.11 "$VENV" >/dev/null

say "installing vcclient-cachy + runtime deps (this is the long step — torch + ORT-GPU)…"
"$UV_BIN" pip install --python "$VENV/bin/python" -e "$REPO_DIR" >/dev/null
"$UV_BIN" pip install --python "$VENV/bin/python" -r "$REPO_DIR/requirements.txt" >/dev/null

# ---- launcher symlink ---------------------------------------------------------

LINK="$BIN_DIR/vcclient-cachy"
ln -sfn "$VENV/bin/vcclient-cachy" "$LINK"
say "linked launcher: $LINK -> $VENV/bin/vcclient-cachy"

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
    install -m 0644 "$REPO_DIR/pkg/vcclient-cachy-mic.service" "$SYSTEMD_USER_DIR/"
    systemctl --user daemon-reload || true
    systemctl --user enable --now vcclient-cachy-mic.service || true
    say "vcclient-cachy-mic.service enabled (Discord/CS2 see vcclient-mic at boot)."
else
    say "skipping systemd unit registration (--no-systemd)"
fi

# ---- summary -----------------------------------------------------------------

cat <<EOF

[install] done.

  launcher : $LINK
  venv     : $VENV
  models   : $APP_HOME/models   ($([ -f "$APP_HOME/models/amitaro_v2_16k.onnx" ] && echo present || echo missing))
  service  : $SYSTEMD_USER_DIR/vcclient-cachy-mic.service

  next steps:
    vcclient-cachy info        # sanity check
    vcclient-cachy pw status   # confirm vcclient-mic exists
    vcclient-cachy run         # launch the TUI
EOF
