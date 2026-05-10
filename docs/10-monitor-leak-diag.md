# v0.6.4 — monitor-leak diagnostic

> **NOTE: Historical investigation snapshot, captured at v0.6.4 (2026-05-05).**
> Behavior described here may have shifted (the TUI `m` toggle in v0.13.1
> changed how monitor is exposed). The current canonical reference is
> `docs/05-perf.md` and `LESSONS.md` for chronology. Don't act on this doc
> as if it reflects current state.

User reported: with `monitor = false` (the default), engine output was
audibly playing through laptop speakers. Setting `monitor = true` was
not the cause — and explicit monitor-stream code paths were never
opened. This is the diagnostic trail and root cause.

## Symptom

- `~/.config/woys/config.toml`: `monitor = false` (top level + every profile).
- TUI engine running (`woys run --autostart`, PID 17137 at time of capture).
- Default audio output: `alsa_output.pci-0000_00_1f.3.analog-stereo`
  (Built-in Audio Analog Stereo — laptop speakers).
- Transformed voice audible through speakers in real time.

## Live state at capture

`pactl list short sinks`:

```
61   alsa_output.usb-...HyperX_QuadCast_2_S....iec958-stereo  PipeWire  s24le 2ch 48000Hz  SUSPENDED
63   alsa_output.pci-0000_00_1f.3.analog-stereo               PipeWire  s32le 2ch 48000Hz  RUNNING
1367 WoysSink                                                 PipeWire  float32le 2ch 48000Hz  SUSPENDED
```

`pactl list sink-inputs` (only the engine playback stream shown):

```
Sink Input #1394
    application.name = "pw-cat"
    target.object = "VCClientCachySink"   ← REQUESTED target
    Sink: 63                              ← ACTUAL attachment = laptop speakers
    Corked: no
    Sample Specification: float32le 2ch 48000Hz
```

**Smoking gun:** the engine playback subprocess (`pw-cat`) asks PipeWire
for sink `VCClientCachySink`, which **doesn't exist** on the system
(installed sink is named `WoysSink`). PipeWire silently routes the
stream to the default sink (id 63 = laptop speakers).

The `WoysSink` virtual sink is `SUSPENDED` because nobody is feeding it
— the engine's output is going to the wrong place. `vcclient-mic`
(which Discord/CS2 see as a microphone) is therefore receiving silence,
while the laptop speakers are receiving the engine output.

`monitor=true` was never the leak path. The leak is upstream of any
monitor logic — the playback subprocess itself is misrouted.

## Root cause

Three commits compounded:

1. **v0.5.x default**: `sink_name = "VCClientCachySink"` (saved into
   user's config.toml on first run by `tui.config.save_config`).
2. **v0.6.0 rename** (`src/audio/pipewire.py:30`,
   `src/tui/config.py:32`): default sink name changed
   `VCClientCachySink` → `WoysSink`. `install.sh:149` even goes out of
   its way to unload any leftover legacy `VCClientCachySink` modules.
3. **v0.6.0 migrator** (`scripts/migrate_to_woys.py`): rewrote
   `vcclient-cachy/models/` → `woys/models/` in config paths but
   **left `sink_name` untouched**. The migrator's docstring (lines
   22-27) even claimed the sink name was deliberately preserved — a
   stale claim from an earlier design that contradicts what the rest of
   v0.6.0 actually shipped.

Net result: any user upgrading from v0.5.x → v0.6.x ends up with a
`config.toml` that points the engine at a sink name PipeWire no longer
exposes. `pw-cat --target=<missing>` falls back to the default sink
without warning.

## Why pw-cat falls back silently

`pw-cat` treats `--target` as a *hint*: PipeWire's session manager
(`wireplumber`) is free to route the stream wherever the policy allows
if the named target isn't resolvable. There is no `--strict-target`
flag. From the engine's perspective the subprocess starts cleanly,
exit-code 0, no stderr — the misrouting is invisible above the OS
boundary.

## Fix shape (shipped in v0.6.4)

1. **Migrator**: rewrite `sink_name = "VCClientCachySink"` →
   `"WoysSink"` on upgrade. Idempotent re-run leaves correct
   configs alone. Docstring corrected.
2. **Engine pre-flight** (`src/audio/engine.py:_open_pacat`): before
   spawning `pw-cat`, verify `cfg.sink_name` is present in
   `pactl list short sinks`. If not, raise a clear `RuntimeError`
   that names the missing sink and points at the fix
   (`woys pw setup` or config edit). No more silent fallback.
3. **Self-heal on existing installs**: `install.sh` re-runs the
   migrator on every install, so any user re-running `./install.sh`
   picks up the config rewrite.
4. **TROUBLESHOOTING.md**: one-liner for users who can't reinstall.

## Verification protocol

With the fix applied:

1. Restart engine.
2. Confirm in `pactl list sink-inputs`: the `pw-cat` stream's `Sink:`
   matches `WoysSink`'s id (not the default sink).
3. Capture default-sink monitor for 10 s; confirm peak amplitude is
   ≈ floor (whatever other apps add) and **no engine voice is
   audible** in the capture.

If step 1 fails (sink doesn't exist), the new pre-flight raises before
`pw-cat` ever launches — the engine refuses to start instead of
leaking audio to the wrong place. That's the safety net.
