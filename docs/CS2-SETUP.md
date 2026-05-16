# Using woys with Counter-Strike 2

Same idea as Discord: point CS2 at `woys-mic`. CS2 itself doesn't have
"noise suppression" toggles you need to fiddle with — its in-game voice
chat reads from the OS audio input you select.

## ⚠️ One thing first — anti-cheat note

woys is **userspace audio software**. It doesn't read game memory,
hook into Source 2, or touch CS2's process. Pointing the game at a virtual
mic is the same kind of OS-level audio routing Windows users do every day for
mic processing. **VAC has no problem with this.**

What VAC *does* take issue with is `evdev` raw-input grabbing for global
hotkeys. That's why woys ships with the evdev hotkey **off by
default**, exposing the same toggle through:

- the TUI (`t` key while focused)
- the CLI (`woys toggle`) → bind to a KDE/GNOME shortcut

If you really want a global hotkey, see `docs/TROUBLESHOOTING.md` for the
opt-in evdev setup, but understand it's at your own risk.

## Step 1 — start the engine

```
woys pw status         # confirm woys-mic is loaded
woys run --autostart    # start the TUI with engine running
```

Leave the TUI open in a terminal — Alt-Tab back to your game.

## Step 2 — point CS2 at woys-mic

CS2 picks up the system's *default* recording device. The cleanest path:

1. Open `pavucontrol` (a GUI for PipeWire/PulseAudio):

   ```
   pavucontrol
   ```

   Install if missing: `paru -S pavucontrol`.

2. Switch to the **Configuration** tab.
3. Make sure your real microphone (e.g. USB condenser mic) is set to a profile
   that captures audio.
4. Hop to the **Input Devices** tab.
5. Click the **Set as fallback** button (a gray check) on **woys-mic**.
   That makes CS2 prefer it next time it picks an input.

Alternatively, set it from the CLI:

```
pactl set-default-source woys-mic
```

## Step 3 — check in CS2

1. Launch Counter-Strike 2.
2. **Settings → Audio → Voice → Voice Input Device**.
3. The dropdown will show what PipeWire reports as the default. With the
   fallback set above, it should be `woys-mic`.
4. Hit **Open Mic Test**, speak — you should see the meter respond.

If CS2 stubbornly clings to your old mic, restart Steam — Source 2 caches the
audio device list at launch.

## Step 4 — push-to-talk

Set CS2 to **Voice Type: Push to Talk** (in Audio settings). The engine keeps
running in the background — pushing the in-game key just gates whether your
teammates hear you, not whether the engine is processing audio. That's fine;
modern GPU usage stays low when your mic is silent.

If you'd prefer the engine to stop entirely between rounds:

- bind a KDE/GNOME shortcut to `woys toggle`, or
- press `t` in the TUI when you want to mute the voice changer.

## Step 5 — verify in a casual match

The first round of any match is the cheap test bed. If teammates can't hear
you, hop back to the TUI to confirm:

- `status: RUNNING` (engine is on)
- input level meter responds when you speak
- avg total e2e ~640 ms (per the v0.12.4 listener-test optimum); CS2's own
  input pipeline contributes ~80 ms, leaving ~560 ms for the woys path

If everything looks right but they still can't hear you, see
`docs/TROUBLESHOOTING.md`.

## What if CS2 sounds delayed?

The pipeline introduces ~640 ms total e2e latency by default (per the
v0.12.4 listener-test optimum: chunk size + inference + audio I/O + SOLA
crossfade). CS2's voice chat is generally tolerant of this; for
competitive play it can still feel slow. Two knobs in
`~/.config/woys/config.toml`:

- `chunk_seconds = 0.25` is the listener-test optimum. Lower values reduce
  tail latency at the cost of more chunks dropped during cuDNN warmup;
  0.25 was picked by user A/B over 0.15 because the chunk-period rhythm
  artifact at 0.15 was more perceptible than the +100 ms latency win.
  See LESSONS §42 for the full A/B writeup.
- `mic_rate = 48000` matches CS2's expected rate; don't change it.

## Optional — separate woys-mic for game vs. Discord

If you only want the voice changer in CS2 but real voice in Discord, set
**Discord** to your real mic (e.g. USB mic) and **CS2** to woys-mic.
Different apps can use different mics on PipeWire — that's the whole point.
