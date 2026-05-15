# Installing woys

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
git clone https://github.com/alirexha/woys.git
cd woys
```

`cd ~/ai` puts you in the AI workspace folder.
`git clone …` copies the source tree from GitHub to disk.
`cd woys` walks into the freshly-cloned directory.

## Step 2 — run install.sh

```
./install.sh
```

What this does, in order:

1. Checks for `pactl` (pipewire-pulse) and `nvidia-smi` (GPU). Warns if missing.
2. Installs `uv` (a fast Python package installer) into `~/.local/bin/` if absent.
3. Creates an isolated Python 3.11 environment under `~/.local/share/woys/venv/`.
4. Installs `woys` and all its dependencies into that environment.
   This is the slow step — it pulls ~3.5 GB of Python wheels (torch, onnxruntime-gpu, etc.).
5. Symlinks `~/.local/bin/woys` to the venv's binary so you can run it from anywhere.
6. Downloads the foundation ONNX weights into `~/.local/share/woys/models/`:
   - `contentvec-f.onnx` (~360 MB — content encoder)
   - `rmvpe_wrapped.onnx` (~345 MB — pitch detector)
   - `amitaro_v2_16k.onnx` (~64 MB — sample voice for testing)

   Older versions of woys also downloaded `hubert_base.pt` (~180 MB) for the
   fairseq embedder fallback. Since v0.8.0 the embedder is always ONNX
   contentvec; `hubert_base.pt` is no longer needed and is no longer
   downloaded.
7. Registers `woys-mic.service` as a systemd user unit, then enables and starts it.

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
woys info
```

You should see something like:

```
woys 0.13.3
  python: 3.11.15
  Server Name: PulseAudio (on PipeWire 1.6.4)
  gpu: NVIDIA GeForce RTX 2070, 595.71.05, 8192 MiB
```

Then check the persistent virtual mic is loaded:

```
woys pw status
```

Expected:

```
sink_present  : True  (module 536870916)
source_present: True  (module 536870917)
```

`pactl list short sources` should now include a line containing `woys-mic`.

Since v0.13.3, when the optional RNNoise chain is enabled (`woys chain
setup`), apps will additionally see two friendlier-named sources in their
input device dropdown:

- **`woys-by-alirexha`** — RNNoise-cleaned source (the recommended daily
  driver; ~13 % cuts/min reduction at the cost of ~+40 ms latency).
- **`woys-no-cleanup`** — raw v0.12.4 engine output, no RNNoise (the
  low-latency fallback). This is the same node `woys-mic` points at.

## Step 4 — run the TUI

```
woys run --autostart
```

Hotkeys inside the TUI:

| Key  | Action                                   |
|------|------------------------------------------|
| `t`  | Toggle the engine on/off                 |
| `+`  | Pitch shift +1 semitone                  |
| `-`  | Pitch shift -1 semitone                  |
| `0`  | Reset pitch                              |
| `p`  | Cycle through saved profiles             |
| `m`  | Toggle self-monitor (host-output copy)   |
| `s`  | Save current settings to `config.toml`   |
| `q`  | Quit                                     |

From outside the TUI (e.g. from a KDE/GNOME global shortcut), use:

```
woys toggle
woys pitch +2
woys status
```

These talk to the running TUI over a Unix socket at
`$XDG_RUNTIME_DIR/woys/control.sock`.

## Step 5 — wire it into Discord / CS2

See `docs/DISCORD-SETUP.md` and `docs/CS2-SETUP.md`. The short version: in those
apps' input-device selector, pick `woys-mic`. That's it.

## Updating

```
cd ~/woys
git pull
./install.sh --skip-models
```

`--skip-models` avoids re-downloading the ~1 GB model cache.

## Uninstalling

```
cd ~/woys
./uninstall.sh
```

Or `./uninstall.sh --keep-models` to keep the ~1 GB ONNX cache around.
Your config at `~/.config/woys/config.toml` is always preserved;
delete it manually if you want a fully clean slate.
