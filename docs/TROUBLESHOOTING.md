# Troubleshooting

If something's broken, the order to check things in:

1. Audio daemon: `pactl info | head -1` should mention PipeWire.
2. The mic exists: `vcclient-cachy pw status` should report `True / True`.
3. The engine starts: `vcclient-cachy run --autostart` should show RUNNING.
4. Apps see the mic: `pactl list short sources` should include `vcclient-mic`.
5. Discord/CS2 are pointed at it (see DISCORD-SETUP.md / CS2-SETUP.md).

If those five are all green, the rest is probably model or anti-cheat related.

---

## "pactl info" says PulseAudio, not PipeWire

You're on classic PulseAudio. vcclient-cachy needs PipeWire (the modern
PulseAudio-compatible audio daemon).

```
paru -S pipewire pipewire-pulse pipewire-alsa
sudo pacman -Rns pulseaudio pulseaudio-bluetooth
systemctl --user --now disable pulseaudio.socket pulseaudio.service
systemctl --user --now enable pipewire.socket pipewire-pulse.socket
```

Then log out and log back in. Re-check with `pactl info | head -1`.

## "vcclient-cachy: command not found"

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
cd ~/ai/vcclient-cachy
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
~/.local/share/vcclient-cachy/venv/bin/python -c "import onnxruntime; print(onnxruntime.__version__)"
```

If it's older, re-install:

```
~/.local/share/vcclient-cachy/venv/bin/pip install -U "onnxruntime-gpu>=1.20"
```

## "vcclient-mic" doesn't appear in Discord/CS2

The systemd unit may not have started. Check:

```
systemctl --user status vcclient-cachy-mic.service
```

If it's not running:

```
systemctl --user start vcclient-cachy-mic.service
```

If it errors out, run the underlying command manually to see the actual
error:

```
vcclient-cachy pw setup
```

Also restart the app — Discord especially caches the device list at launch.

## Engine starts then stops with `last_error`

The engine's `EngineStats.last_error` will surface the underlying issue. Most
common:

- `OSError: ... vcclient-mic` — the mic was torn down externally. Run
  `vcclient-cachy pw setup` and start the engine again.
- `ModuleNotFoundError: contentvec` — the model files aren't in
  `~/.local/share/vcclient-cachy/models/`. Run:

  ```
  ~/.local/share/vcclient-cachy/venv/bin/python scripts/download_weights.py
  ```

- `RuntimeError: ... CUDA out of memory` — close other GPU-using apps
  (browsers with hardware video, OBS, other ML tools).

## Voice sounds robotic / glitchy

This usually means the pipeline is dropping audio (running below realtime).

1. Check the latency panel in the TUI. If `avg_total_ms` exceeds your
   `chunk_seconds × 1000`, the engine isn't keeping up.
2. Increase `chunk_seconds` in `~/.config/vcclient-cachy/config.toml`:

   ```toml
   chunk_seconds = 0.5   # was 0.25
   ```

3. Restart the TUI.

If that doesn't help, your GPU may be under heavy load from another app.
Check `nvidia-smi` for other CUDA processes.

## Voice is too quiet / too loud at the listener

vcclient-cachy doesn't apply gain — output level matches the voice model's
training distribution. To boost output, raise the **VCClientCachySink** sink
volume in `pavucontrol` (Output Devices tab). The remap-source mirrors that
volume into vcclient-mic.

```
# Or via CLI:
pactl set-sink-volume VCClientCachySink 150%
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

1. Run `vcclient-cachy info` and confirm the GPU isn't thermal-throttling
   (`nvidia-smi` GPU temp should be < 80°C).
2. Lock the chunk size in config.toml — don't let it auto-vary.

## Can I make the global hotkey work without VAC banning me?

The default vcclient-cachy build does **not** use evdev raw-grabbing — it
exposes toggling via the TUI key (`t`) and the Unix socket
(`vcclient-cachy toggle`). The latter is what you bind to a KDE/GNOME shortcut.

If you need a system-wide hotkey *outside* of CS2, the opt-in evdev path:

```
pip install -e ~/ai/vcclient-cachy[evdev]
sudo usermod -aG input $USER
# logout / login
```

Then in `~/.config/vcclient-cachy/config.toml`:

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
cd ~/ai/vcclient-cachy
.venv/bin/python scripts/smoke_rvc_onnx.py
```

Or via pytest:

```
.venv/bin/python -m pytest tests/test_smoke_rvc_onnx.py -v -s
```

## Reset to defaults

Wipe config and re-install:

```
rm ~/.config/vcclient-cachy/config.toml
cd ~/ai/vcclient-cachy
./uninstall.sh
./install.sh
```

This nukes the venv and re-creates it. Models cache is in
`~/.local/share/vcclient-cachy/models/` and is preserved unless you also
remove that.
