# 0003 — Textual for the TUI

## Decision

The interactive control UI (`src/tui/`) is built on
[Textual](https://textual.textualize.io/), an asyncio-based terminal
UI framework.

## Status

`accepted`

## Context

`PROJECT_BRIEF.md` §10 specified Textual without ranking alternatives
(`src/tui/ using Textual`). The brief's authority is the
inheritance, but a future maintainer asking "should this still be
Textual?" finds no comparison. v0.13.x changed how the project is
typically run: the engine commonly lives as a pipewire-pulse module
chain plus `woys-chain.service` (systemd user unit), and many users
control it via `woys toggle` from a WM keybind rather than via the
TUI directly. That shift puts the TUI's role under fresh light.

## Decision

Textual remains the TUI framework; CLI + systemd + Unix-socket
control (decisions 0009 boundary) co-exist as the headless path.

## Alternatives considered

- **Raw curses** — minimal dep, but reactive layout, theming, async
  refresh, and mouse support all become bespoke code. Net LOC for
  feature parity: ~10× Textual's.
- **Qt6 / PySide6** — heavyweight (X11/Wayland deps, ~80 MB install
  surface), and pulls a GUI toolkit into a TUI-shaped product.
- **GTK4** — same heavyweight cost as Qt; CachyOS-native but breaks
  the "runs in any terminal" property the TUI currently has.
- **No TUI, CLI + tray + WM-shortcut only** — the v0.13.x usage
  pattern. Viable, but loses the live latency / level-meter /
  in-process pitch slider for users who like a foreground console.

## Rationale

Textual fits four constraints that hold across foreseeable refactors.
First, asyncio integration: the engine surfaces stats via a Unix
socket (decision 0009 boundary) whose poller naturally lives on the
same event loop as Textual's render loop — D-Bus would have needed
a GLib mainloop alongside, which `LESSONS.md` §3 mistake #6
documented as not worth the integration cost. Second, install
surface: pure-Python wheel, no system deps, lands in the same `uv`
venv as the engine. Third, dev velocity: declarative widgets, hot
reload, theming, mouse — features we use today (the live level
meter, the rolling-average latency readout, the model picker)
would each be ~50 LOC of curses or ~200 LOC of Qt. Fourth,
discoverability: Textual apps run in any terminal, including over
SSH and inside `tmux`, with no display-server requirement.

## Trade-offs accepted

Textual is moving fast (versioned API breakages between minor
releases require occasional shim fixes). The TUI shares its event
loop with the engine's stats poller, so a stats-fetch hang would
pause the UI — mitigated by the Unix-socket non-blocking read.
Headless / SSH-from-shell-script use cases route through the CLI
(`woys toggle`, `woys pitch +N`) rather than the TUI.

## Re-litigation triggers

- Textual ships a major API rework that would require a v0.14+ port
  large enough to be cost-comparable with switching frameworks.
- User telemetry (or self-reported usage) shows the TUI is rarely
  used vs `woys toggle`; the cost of maintaining it begins to exceed
  the value.
- A successor framework (e.g., a curses+ncurses-extended async lib)
  appears that closes Textual's feature gap with materially smaller
  install surface.
