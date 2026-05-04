# Installing vcclient-cachy

This guide assumes you've never installed Python software on Linux before.
Every command is copy-paste-able; every step says **what** it does and **why**.

## Step 0 — what you need

| Thing               | What for                                       |
|---------------------|------------------------------------------------|
| CachyOS / Arch      | The fork is Linux-native; non-systemd distros work too if you know what you're doing |
| PipeWire            | The audio routing layer (already on CachyOS) |
| NVIDIA GPU + driver | RVC inference runs on CUDA; tested on RTX 2070 |
| ~5 GB free disk     | Models (~1 GB) + venv with torch+ORT (~3.5 GB) |
| ~5 minutes          | Most of it is downloading torch and ORT      |

Verify CachyOS is on PipeWire (it should be by default):

```
pactl info | head -1
```

You should see `Server Name: PulseAudio (on PipeWire ...)`. If it says
"PulseAudio" without "PipeWire", uninstall PulseAudio and install PipeWire-pulse:

```
paru -S pipewire pipewire-pulse pipewire-alsa
sudo pacman -Rns pulseaudio
```

Then log out and log back in.

## Step 1 — clone the repo

```
cd ~/ai
git clone https://github.com/alirexha/vcclient-cachy.git
cd vcclient-cachy
```

`cd ~/ai` puts you in the AI workspace folder.
`git clone …` copies the source tree from GitHub to disk.
`cd vcclient-cachy` walks into the freshly-cloned directory.

## Step 2 — run install.sh

```
./install.sh
```

What this does, in order:

1. Checks for `pactl` (pipewire-pulse) and `nvidia-smi` (GPU). Warns if missing.
2. Installs `uv` (a fast Python package installer) into `~/.local/bin/` if absent.
3. Creates an isolated Python 3.11 environment under `~/.local/share/vcclient-cachy/venv/`.
4. Installs `vcclient-cachy` and all its dependencies into that environment.
   This is the slow step — it pulls ~3.5 GB of Python wheels (torch, onnxruntime-gpu, etc.).
5. Symlinks `~/.local/bin/vcclient-cachy` to the venv's binary so you can run it from anywhere.
6. Downloads the foundation ONNX weights into `~/.local/share/vcclient-cachy/models/`:
   - `contentvec-f.onnx` (~360 MB — content encoder)
   - `rmvpe_wrapped.onnx` (~345 MB — pitch detector)
   - `hubert_base.pt` (~180 MB — fallback embedder)
   - `amitaro_v2_16k.onnx` (~64 MB — sample voice for testing)
7. Registers `vcclient-cachy-mic.service` as a systemd user unit, then enables and starts it.

If `~/.local/bin` isn't on your `$PATH`, the installer prints how to add it.
On fish (CachyOS default):

```
fish_add_path ~/.local/bin
```

On bash/zsh, append this to `~/.bashrc` or `~/.zshrc`:

```
export PATH="$HOME/.local/bin:$PATH"
```

## Step 3 — sanity-check the install

```
vcclient-cachy info
```

You should see something like:

```
vcclient-cachy 0.1.0
  python: 3.11.15
  Server Name: PulseAudio (on PipeWire 1.6.4)
  gpu: NVIDIA GeForce RTX 2070, 595.71.05, 8192 MiB
```

Then check the persistent virtual mic is loaded:

```
vcclient-cachy pw status
```

Expected:

```
sink_present  : True  (module 536870916)
source_present: True  (module 536870917)
```

`pactl list short sources` should now include a line containing `vcclient-mic`.

## Step 4 — run the TUI

```
vcclient-cachy run --autostart
```

Hotkeys inside the TUI:

| Key  | Action                                   |
|------|------------------------------------------|
| `t`  | Toggle the engine on/off                 |
| `+`  | Pitch shift +1 semitone                  |
| `-`  | Pitch shift -1 semitone                  |
| `0`  | Reset pitch                              |
| `s`  | Save current settings to `config.toml`   |
| `q`  | Quit                                     |

From outside the TUI (e.g. from a KDE/GNOME global shortcut), use:

```
vcclient-cachy toggle
vcclient-cachy pitch +2
vcclient-cachy status
```

These talk to the running TUI over a Unix socket at
`$XDG_RUNTIME_DIR/vcclient-cachy/control.sock`.

## Step 5 — wire it into Discord / CS2

See `docs/DISCORD-SETUP.md` and `docs/CS2-SETUP.md`. The short version: in those
apps' input-device selector, pick `vcclient-mic`. That's it.

## Updating

```
cd ~/ai/vcclient-cachy
git pull
./install.sh --skip-models
```

`--skip-models` avoids re-downloading the ~1 GB model cache.

## Uninstalling

```
cd ~/ai/vcclient-cachy
./uninstall.sh
```

Or `./uninstall.sh --keep-models` to keep the ~1 GB ONNX cache around.
Your config at `~/.config/vcclient-cachy/config.toml` is always preserved;
delete it manually if you want a fully clean slate.
