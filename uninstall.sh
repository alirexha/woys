#!/usr/bin/env bash
# vcclient-cachy uninstaller — reverses what install.sh did.
#
# Removes:
#   $HOME/.local/share/vcclient-cachy/        (venv, models — opt-out with --keep-models)
#   $HOME/.local/bin/vcclient-cachy           (launcher symlink)
#   $HOME/.config/systemd/user/vcclient-cachy-mic.service
#
# Leaves alone:
#   $HOME/.config/vcclient-cachy/config.toml (your settings)
#
# Usage:
#   ./uninstall.sh               # remove everything except your config
#   ./uninstall.sh --keep-models # keep the ~1 GiB ONNX cache

set -euo pipefail

APP_HOME="$HOME/.local/share/vcclient-cachy"
BIN_DIR="$HOME/.local/bin"
SYSTEMD_USER_DIR="$HOME/.config/systemd/user"

KEEP_MODELS=0
for arg in "$@"; do
    case "$arg" in
    --keep-models) KEEP_MODELS=1 ;;
    -h|--help)
        sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# //;s/^#//'
        exit 0
        ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
    esac
done

say() { printf '\n[uninstall] %s\n' "$*"; }

# Stop and disable the systemd unit first.
if systemctl --user list-unit-files vcclient-cachy-mic.service >/dev/null 2>&1; then
    say "disabling vcclient-cachy-mic.service…"
    systemctl --user disable --now vcclient-cachy-mic.service 2>/dev/null || true
fi
rm -f "$SYSTEMD_USER_DIR/vcclient-cachy-mic.service"
systemctl --user daemon-reload 2>/dev/null || true

# Tear down the PipeWire mic if a previous run left it loaded.
if command -v pactl >/dev/null && pactl info 2>/dev/null | grep -q PipeWire; then
    if [ -x "$APP_HOME/venv/bin/vcclient-cachy" ]; then
        "$APP_HOME/venv/bin/vcclient-cachy" pw teardown 2>/dev/null || true
    fi
fi

# Remove launcher symlink.
rm -f "$BIN_DIR/vcclient-cachy"

# Remove app dir (optionally keeping models).
if [ "$KEEP_MODELS" -eq 1 ] && [ -d "$APP_HOME/models" ]; then
    say "preserving models at $APP_HOME/models/"
    rm -rf "$APP_HOME/venv"
else
    say "removing $APP_HOME/"
    rm -rf "$APP_HOME"
fi

cat <<EOF

[uninstall] done.

  Config preserved: $HOME/.config/vcclient-cachy/config.toml
  Models preserved: $([ "$KEEP_MODELS" -eq 1 ] && echo "yes ($APP_HOME/models)" || echo no)

  To wipe everything (including config):
    rm -rf $HOME/.config/vcclient-cachy/
EOF
