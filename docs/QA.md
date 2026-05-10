# Live QA script — Discord + CS2 round-trip

**This is the human-in-the-loop validation for Definition of Done items
#2 and #3** (PROJECT_BRIEF §18). Items requiring you to actually speak into
your mic and listen via the target app — automation can't fake the routing
end-to-end without itself becoming a fake.

Run through this once after install. ~10 minutes. Mark the boxes as you go.

## Pre-flight

- [ ] `woys info` shows your GPU and PipeWire version
- [ ] `woys pw status` exits 0 (both `True`)
- [ ] `pactl list short sources | grep woys-mic` shows one line
- [ ] `pactl list short sinks | grep WoysSink` shows one line
- [ ] `systemctl --user status woys-mic.service` is `active (exited)`

If any of those fail, fix per `docs/TROUBLESHOOTING.md` before going on.

## Test 1 — engine on/off + CLI toggle

In one terminal:

```
woys run --autostart
```

- [ ] TUI displays `RUNNING` after the cold-start chunks finish
- [ ] `avg total e2e` settles below 500 ms within 5 seconds
- [ ] Input level meter responds when you speak

In a *second* terminal:

```
woys status
woys toggle
woys status
woys toggle
```

- [ ] First `status` shows `running=True`
- [ ] After first `toggle`, status shows `running=False`
- [ ] After second `toggle`, back to `running=True`
- [ ] TUI's status panel reflects the toggles in real time

```
woys pitch +2
woys pitch -3
woys pitch 0
```

- [ ] TUI's `pitch` line updates each time
- [ ] After `pitch 0`, it reads `+0 st`

## Test 2 — Discord receives transformed voice (DoD #2)

1. Open Discord (closed before? launch it now so it picks up the new mic list).
2. **User Settings → Voice & Video**:
   - [ ] **Input Device** dropdown contains `woys-mic`
   - [ ] Set **Input Device** to `woys-mic`
   - [ ] Under **Voice Processing**, set **Noise Suppression: None**
   - [ ] Set **Echo Cancellation: OFF**
   - [ ] Set **Automatic Gain Control: OFF**
3. Hit **Let's Check** in Voice & Video.
4. Speak into your real mic for ~3 seconds.

- [ ] Discord's mic level bar moves while you speak
- [ ] When you play back the test, you hear *transformed* (not raw) voice
- [ ] No "Krisp gated everything to silence" effect

5. Optional: hop into a private voice channel and call yourself from a
   second device.

- [ ] Latency between speaking and remote-side hearing feels usable
      (around 640 ms typical with the v0.12.4 defaults — `chunk_seconds=0.25`
      and `output_latency_ms=280`, the latter shipping since v0.7.0-rc3)
- [ ] No audible chunk-boundary clicks or dropouts during continuous speech

## Test 3 — CS2 receives transformed voice (DoD #3)

1. Set the system default input source to `woys-mic`:

   ```
   pactl set-default-source woys-mic
   ```

2. Launch Steam, then CS2 (Source 2 caches the audio device list at launch
   — close and re-open if it was running).
3. **Settings → Audio → Voice**:
   - [ ] **Voice Input Device** shows the system default (which is now
         `woys-mic`)
4. Hit **Open Mic Test**.

- [ ] Mic test meter responds as you speak
- [ ] Playback is the transformed voice, not your raw voice

5. Hop into a casual match.

- [ ] Teammates can hear you (ask in chat or wave at the mic)
- [ ] Your audio doesn't trip Counter-Strike's anti-shouting filter
      (it shouldn't — the engine doesn't do anything weird at the codec layer)

## Test 4 — engine survives a longer session

Leave the TUI running with the engine on for 10+ minutes.

- [ ] No `last_error` ever populates the status panel
- [ ] `chunks_processed` keeps climbing
- [ ] `avg_total_ms` stays roughly constant — no slow degradation
- [ ] `nvidia-smi` shows the woys process around 1.3-1.4 GiB stable

If you see `last_error: OSError ... woys-mic`, the systemd unit may have
restarted; check `journalctl --user -u woys-mic.service`.

## Test 5 — clean shutdown

In the TUI: press `q`.

- [ ] TUI exits within 1-2 seconds
- [ ] Config saved (mtime on `~/.config/woys/config.toml` updates)
- [ ] `pactl list short sources | grep woys-mic` *still* shows the mic
      (the systemd unit owns it; engine quit didn't tear it down — that's
      intentional per Q6)

To fully tear down everything:

```
systemctl --user stop woys-mic.service
woys pw status   # both False
```

## If anything failed

Drop the failure into `docs/TROUBLESHOOTING.md`'s template — open an issue
on GitHub or ping me directly. The exact `journalctl --user -u
woys-mic.service` lines around the failure are usually enough.

---

**When all 5 tests pass, DoD #2 and #3 are met.** The brief's automated
checks (verification gates, latency floor) are gated by `pytest` and ran
green per phase. The human-in-the-loop checks are gated by this script.
