# Using vcclient-cachy with Discord

You'll point Discord at `vcclient-mic` (the virtual microphone vcclient-cachy
publishes), then turn off Discord's noise suppression so it doesn't fight the
voice changer's output.

## Step 0 — verify vcclient-mic exists

```
vcclient-cachy pw status
```

If both `sink_present` and `source_present` show `True`, you're good. If not:

```
vcclient-cachy pw setup
```

(Or restart the systemd unit: `systemctl --user restart vcclient-cachy-mic.service`.)

## Step 1 — start the engine

In one terminal:

```
vcclient-cachy run --autostart
```

`--autostart` flips the engine on the moment the TUI launches. Once you see
`status: RUNNING` and the latency panel updates, the audio path is live.

## Step 2 — point Discord at vcclient-mic

1. Open Discord → **User Settings** (cog icon next to your name) → **Voice & Video**.
2. **Input Device** dropdown → pick **`vcclient-mic`**.
3. **Output Device** can stay on your headphones — vcclient-cachy doesn't
   touch playback, only input.

A small mic-test bar should respond when you talk.

## Step 3 — disable Discord's noise suppression

This is critical. Discord's built-in noise suppression (Krisp) is trained on
*real* voices. The transformed output of an RVC model can read as "noise" to
Krisp and get gated out, leaving your contacts hearing nothing.

In **User Settings → Voice & Video**:

1. Scroll to **Voice Processing**.
2. Turn **OFF**:
   - Echo Cancellation
   - Noise Suppression (set to **None** — *not* "Krisp" or "Standard")
   - Automatic Gain Control
3. Leave on:
   - **Advanced Voice Activity** (or use Push-to-Talk if you prefer)

Save settings (Discord usually does this automatically).

## Step 4 — try it

Hop into a voice channel, or use Discord's mic test (**Voice & Video → Let's
Check**). Speak normally. Your contacts should hear the transformed voice
with low latency.

## Troubleshooting

| Symptom                                    | What to do                                                                 |
|--------------------------------------------|----------------------------------------------------------------------------|
| Discord says "no input"                    | Check `vcclient-cachy status` — is the engine RUNNING?                     |
| Voice sounds robotic / clipped             | Lower input gain in your mic; the engine handles up to ~0.7 RMS cleanly    |
| Heavy delay (>500 ms)                      | Reduce `chunk_seconds` in `~/.config/vcclient-cachy/config.toml` to 0.25   |
| Mic level meter is dead in Discord         | Try `pavucontrol` → **Recording** tab → confirm Discord is reading vcclient-mic |
| Pitch shift sounds wrong                   | Hit `0` in the TUI to reset, then `+`/`-` one semitone at a time           |
| Engine errors out after model load         | Check `~/.local/share/vcclient-cachy/models/` — re-run `scripts/download_weights.py` |

If Discord auto-detects "another device" each call and switches off
vcclient-mic, lock the input device in Discord's settings (the dropdown shows
"vcclient-mic" with a lock icon when remembered).

## Pro tip — KDE/GNOME shortcut for toggle

Bind a keyboard shortcut in your DE that runs:

```
vcclient-cachy toggle
```

This way you can mute the voice changer mid-call without alt-tabbing to the
TUI. The shortcut talks to the running TUI over its Unix socket.

KDE: **System Settings → Shortcuts → Custom Shortcuts → Add → Command/URL**.
GNOME: **Settings → Keyboard → View and Customize Shortcuts → Custom Shortcuts**.
