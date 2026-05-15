#!/usr/bin/env bash
# woys uninstaller — reverses what install.sh did.
#
# Removes:
#   $HOME/.local/share/woys/                   (venv, models — opt-out with --keep-models)
#   $HOME/.local/share/vcclient-cachy/         (legacy path, if still present from pre-v0.6.0)
#   $HOME/.local/bin/woys                      (launcher symlink)
#   $HOME/.local/bin/vcclient-cachy            (deprecated shim from v0.6.x)
#   $HOME/.config/systemd/user/woys-mic.service
#   $HOME/.config/systemd/user/woys-chain.service    (RNNoise chain user unit, v0.13.x+)
#   $HOME/.config/systemd/user/vcclient-cachy-mic.service  (legacy)
#
# Leaves alone:
#   $HOME/.config/woys/config.toml            (your settings)
#   $HOME/.config/vcclient-cachy/config.toml  (legacy settings, if still present)
#
# Usage:
#   ./uninstall.sh               # remove everything except your config
#   ./uninstall.sh --keep-models # keep the ~1 GiB ONNX cache

set -euo pipefail

APP_HOME="$HOME/.local/share/woys"
LEGACY_APP_HOME="$HOME/.local/share/vcclient-cachy"
BIN_DIR="$HOME/.local/bin"
SYSTEMD_USER_DIR="$HOME/.config/systemd/user"

KEEP_MODELS=0
for arg in "$@"; do
    case "$arg" in
    --keep-models) KEEP_MODELS=1 ;;
    -h|--help)
        sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# //;s/^#//'
        exit 0
        ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
    esac
done

say() { printf '\n[uninstall] %s\n' "$*"; }

# Tear down the RNNoise post-engine chain BEFORE removing the binary —
# `woys chain disable` stops + disables the systemd unit, removes the
# unit file, and unloads the loaded pactl modules. Skipping this leaves
# an enabled woys-chain.service pointing at a missing binary (re-runs on
# every login) and the chain modules persist until reboot.
# F-merged-005 — review Phase 6.
if command -v pactl >/dev/null && pactl info 2>/dev/null | grep -q PipeWire; then
    if [ -x "$APP_HOME/venv/bin/woys" ]; then
        say "tearing down the RNNoise chain (if loaded)…"
        "$APP_HOME/venv/bin/woys" chain disable 2>/dev/null || true
        "$APP_HOME/venv/bin/woys" pw teardown 2>/dev/null || true
    elif [ -x "$LEGACY_APP_HOME/venv/bin/vcclient-cachy" ]; then
        "$LEGACY_APP_HOME/venv/bin/vcclient-cachy" pw teardown 2>/dev/null || true
    fi
fi

# Stop and disable every unit name we have ever shipped. `woys chain disable`
# above is the primary cleanup for woys-chain.service; this loop is the
# belt-and-suspenders pass for the case where the binary is already
# missing (e.g. partial install, manual venv delete) and the unit file
# was orphaned. Idempotent: `systemctl disable --now` on a missing unit
# is a no-op.
for unit in woys-mic.service woys-chain.service vcclient-cachy-mic.service; do
    if systemctl --user list-unit-files "$unit" >/dev/null 2>&1; then
        say "disabling $unit…"
        systemctl --user disable --now "$unit" 2>/dev/null || true
    fi
    rm -f "$SYSTEMD_USER_DIR/$unit"
done
systemctl --user daemon-reload 2>/dev/null || true

# Remove launcher symlinks (current + legacy shim).
rm -f "$BIN_DIR/woys" "$BIN_DIR/vcclient-cachy"

# Remove app dirs (optionally keeping models).
for HOME_DIR in "$APP_HOME" "$LEGACY_APP_HOME"; do
    [ -d "$HOME_DIR" ] || continue
    if [ "$KEEP_MODELS" -eq 1 ] && [ -d "$HOME_DIR/models" ]; then
        say "preserving models at $HOME_DIR/models/"
        rm -rf "$HOME_DIR/venv"
    else
        say "removing $HOME_DIR/"
        rm -rf "$HOME_DIR"
    fi
done

cat <<EOF

[uninstall] done.

  Config preserved (woys)         : $HOME/.config/woys/config.toml
  Config preserved (legacy)       : $HOME/.config/vcclient-cachy/config.toml (if still present)
  Models preserved                : $([ "$KEEP_MODELS" -eq 1 ] && echo "yes" || echo no)

  To wipe everything (including config):
    rm -rf $HOME/.config/woys/ $HOME/.config/vcclient-cachy/
EOF
