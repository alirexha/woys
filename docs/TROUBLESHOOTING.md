# Troubleshooting

If something's broken, the order to check things in:

1. Audio daemon: `pactl info | head -1` should mention PipeWire.
2. The mic exists: `woys pw status` should report `True / True`.
3. The engine starts: `woys run --autostart` should show RUNNING.
4. Apps see the mic: `pactl list short sources` should include `vcclient-mic`.
5. Discord/CS2 are pointed at it (see DISCORD-SETUP.md / CS2-SETUP.md).

If those five are all green, the rest is probably model or anti-cheat related.

---

## "pactl info" says PulseAudio, not PipeWire

You're on classic PulseAudio. woys needs PipeWire (the modern
PulseAudio-compatible audio daemon).

```
paru -S pipewire pipewire-pulse pipewire-alsa
sudo pacman -Rns pulseaudio pulseaudio-bluetooth
systemctl --user --now disable pulseaudio.socket pulseaudio.service
systemctl --user --now enable pipewire.socket pipewire-pulse.socket
```

Then log out and log back in. Re-check with `pactl info | head -1`.

## "woys: command not found"

`~/.local/bin/` isn't on your `$PATH`. On fish (CachyOS default):

```
fish_add_path ~/.local/bin
```

On bash/zsh, append to your shell rc:

```
export PATH="$HOME/.local/bin:$PATH"
```

Then `source ~/.bashrc` (or open a fresh terminal).

## CUDA EP fails to load — "libcublasLt.so.12 not found"

The pip-shipped CUDA libs that come with `onnxruntime-gpu` aren't on
`LD_LIBRARY_PATH`. We work around this internally via `ort.preload_dlls()`,
which is called automatically. If you see this error, your venv is probably
out of date:

```
cd ~/ai/woys
.venv/bin/python -m pip install --upgrade onnxruntime-gpu
```

Or run `./install.sh --skip-models` to refresh the install.

## "No GPU found, falling back to CPU"

Inference on CPU is real-time-impossible for this pipeline. Verify the GPU is
detectable:

```
nvidia-smi
```

If `nvidia-smi` works but the engine claims no GPU, the venv's `onnxruntime-gpu`
isn't seeing CUDA. Check ORT version is **≥ 1.20**:

```
~/.local/share/woys/venv/bin/python -c "import onnxruntime; print(onnxruntime.__version__)"
```

If it's older, re-install:

```
~/.local/share/woys/venv/bin/pip install -U "onnxruntime-gpu>=1.20"
```

## "vcclient-mic" doesn't appear in Discord/CS2

The systemd unit may not have started. Check:

```
systemctl --user status woys-mic.service
```

If it's not running:

```
systemctl --user start woys-mic.service
```

If it errors out, run the underlying command manually to see the actual
error:

```
woys pw setup
```

Also restart the app — Discord especially caches the device list at launch.

## I hear my own transformed voice through laptop speakers (monitor=false)

Symptom: `monitor = false` in `~/.config/woys/config.toml`, but the
engine output is still audible on the system default sink (laptop
speakers / headphones). Discord / CS2 also report `vcclient-mic` as
silent.

Cause: a stale `sink_name` in your config from before the v0.6.0
rename. The internal sink was `VCClientCachySink` in v0.5.x and is
`WoysSink` in v0.6.0+. The v0.6.0 migrator (before v0.6.4) didn't
rewrite that key, so the engine asked `pw-cat` to play to a sink that
no longer exists. PipeWire silently fell back to the default sink.

Quickest fix:

```
# 1. stop any running woys engine first (Ctrl-C in the TUI, or:)
pkill -f 'woys run'

# 2. rewrite the sink name in your config:
sed -i 's|sink_name = "VCClientCachySink"|sink_name = "WoysSink"|' \
    ~/.config/woys/config.toml

# 3. confirm the WoysSink module is loaded:
woys pw status

# 4. relaunch:
woys
```

If you upgrade to v0.6.4+ and re-run `./install.sh`, the migrator does
this rewrite for you. The engine in v0.6.4+ also pre-flights — it
refuses to start if `sink_name` doesn't resolve to a loaded sink, so
this failure mode can't recur silently.

## Engine starts then stops with `last_error`

The engine's `EngineStats.last_error` will surface the underlying issue. Most
common:

- `OSError: ... vcclient-mic` — the mic was torn down externally. Run
  `woys pw setup` and start the engine again.
- `ModuleNotFoundError: contentvec` — the model files aren't in
  `~/.local/share/woys/models/`. Run:

  ```
  ~/.local/share/woys/venv/bin/python scripts/download_weights.py
  ```

- `RuntimeError: ... CUDA out of memory` — close other GPU-using apps
  (browsers with hardware video, OBS, other ML tools).

## Voice sounds robotic / glitchy

This usually means the pipeline is dropping audio (running below realtime).

1. Check the latency panel in the TUI. If `avg_total_ms` exceeds your
   `chunk_seconds × 1000`, the engine isn't keeping up.
2. Increase `chunk_seconds` in `~/.config/woys/config.toml`:

   ```toml
   chunk_seconds = 0.5   # was 0.25
   ```

3. Restart the TUI.

If that doesn't help, your GPU may be under heavy load from another app.
Check `nvidia-smi` for other CUDA processes.

## Voice is too quiet / too loud at the listener

woys doesn't apply gain — output level matches the voice model's
training distribution. To boost output, raise the **WoysSink** sink
volume in `pavucontrol` (Output Devices tab). The remap-source mirrors that
volume into vcclient-mic.

```
# Or via CLI:
pactl set-sink-volume WoysSink 150%
```

## Discord noise suppression eats the transformed voice

Disable it in Discord → Voice & Video → Voice Processing — set Noise Suppression
to **None** (not Krisp, not Standard). Krisp is trained on real voices and gates
out RVC output as "noise".

## Model file too large for upload

This affects the `.pth → .onnx` conversion via upstream's web UI when the
file is > 100 MB. Use the manual `torch.onnx.export` path in `MODELS.md` Option B.

## Engine drops audio every ~30 seconds

This is usually an ORT cudnn algo cache miss when the input shape varies.
The engine uses fixed `chunk_seconds`, so shapes are stable in normal use.
If you see this:

1. Run `woys info` and confirm the GPU isn't thermal-throttling
   (`nvidia-smi` GPU temp should be < 80°C).
2. Lock the chunk size in config.toml — don't let it auto-vary.

## Can I make the global hotkey work without VAC banning me?

The default woys build does **not** use evdev raw-grabbing — it
exposes toggling via the TUI key (`t`) and the Unix socket
(`woys toggle`). The latter is what you bind to a KDE/GNOME shortcut.

If you need a system-wide hotkey *outside* of CS2, the opt-in evdev path:

```
pip install -e ~/ai/woys[evdev]
sudo usermod -aG input $USER
# logout / login
```

Then in `~/.config/woys/config.toml`:

```toml
enable_evdev_hotkey = true
evdev_hotkey = "ctrl+alt+v"
```

**Don't enable this if you play VAC-protected games.** The brief and the user
Q&A both flagged this as a real risk.

## How do I check my measured latency?

The TUI's latency panel shows `avg_total_ms` and `avg_inference_ms`. To run
the inference-only benchmark used in `docs/05-perf.md`:

```
cd ~/ai/woys
.venv/bin/python scripts/smoke_rvc_onnx.py
```

Or via pytest:

```
.venv/bin/python -m pytest tests/test_smoke_rvc_onnx.py -v -s
```

## Reset to defaults

Wipe config and re-install:

```
rm ~/.config/woys/config.toml
cd ~/ai/woys
./uninstall.sh
./install.sh
```

This nukes the venv and re-creates it. Models cache is in
`~/.local/share/woys/models/` and is preserved unless you also
remove that.
