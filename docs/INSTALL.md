# Installing woys

This guide assumes you've never installed Python software on Linux before.
Every command is copy-paste-able; every step says **what** it does and **why**.

## Step 0 — what you need

| Thing               | What for                                       |
|---------------------|------------------------------------------------|
| CachyOS / Arch      | The fork is Linux-native; non-systemd distros work too if you know what you're doing |
| PipeWire            | The audio routing layer (already on CachyOS) |
| NVIDIA GPU + driver | RVC inference runs on CUDA; tested on RTX 2070 |
| uv                  | Python installer the script uses to fetch Python 3.11 and build the venv (`sudo pacman -S uv`) |
| gcc, make, pkg-config + PipeWire headers | Builds `woys-pw-out`, the native PipeWire playback helper woys uses by default (`sudo pacman -S base-devel pkgconf`; the Arch `pipewire` package ships the headers; Debian/Ubuntu: `build-essential pkg-config libpipewire-0.3-dev`) |
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
cd ~
git clone https://github.com/alirexha/woys.git
cd woys
```

`cd ~` puts you in your home folder, so the checkout lands in `~/woys`
(the path the rest of this guide uses).
`git clone …` copies the source tree from GitHub to disk.
`cd woys` walks into the freshly-cloned directory.

## Step 2 — run install.sh

```
./install.sh
```

What this does, in order:

1. Checks the prerequisites and stops with an error if one is missing:
   `pactl` talking to PipeWire, `nvidia-smi` (the GPU), `uv`, and the
   build tools for step 5. It does not install any of them for you.
   `uv` is looked up on your `PATH`, then at `~/.local/bin/uv`; if yours
   lives elsewhere, run `UV_BIN=/path/to/uv ./install.sh`.
2. Uses `uv` to install Python 3.11 (user-local) if you don't have it.
3. Creates an isolated Python 3.11 environment under `~/.local/share/woys/venv/`.
4. Installs `woys` and all its dependencies into that environment.
   This is the slow step — it pulls ~3.5 GB of Python wheels (torch, onnxruntime-gpu, etc.).
5. Builds the native PipeWire helper (`make -C bin/`) and installs it as
   `~/.local/bin/woys-pw-out`. woys plays audio through it by default, so
   the install fails if gcc, make or the PipeWire headers are missing
   (checked up front, before the slow step) or the build fails.
6. Symlinks `~/.local/bin/woys` to the venv's binary so you can run it from anywhere.
7. Downloads the foundation ONNX weights into `~/.local/share/woys/models/`:
   - `contentvec-f.onnx` (~360 MB — content encoder)
   - `rmvpe_wrapped.onnx` (~345 MB — pitch detector)
   - `amitaro_v2_16k.onnx` (~64 MB — sample voice for testing)

   Older versions of woys also downloaded `hubert_base.pt` (~180 MB) for the
   fairseq embedder fallback. Since v0.8.0 the embedder is always ONNX
   contentvec; `hubert_base.pt` is no longer needed and is no longer
   downloaded.
8. Registers `woys-mic.service` as a systemd user unit, then enables and starts it.

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
woys 0.15.0
  python: 3.11.15
  onnxruntime: 1.22.0
  CUDAExecutionProvider: available
  TensorrtExecutionProvider: not available
  Server Name: PulseAudio (on PipeWire 1.6.4)
  gpu: NVIDIA GeForce RTX 2070, 595.71.05, 8192 MiB
  active rvc model: (none configured)
```

`CUDAExecutionProvider: available` is the line that matters: without it
woys cannot run.

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

When the optional RNNoise chain is enabled (`woys chain setup`), apps
show one friendly-named source in their input device dropdown:

- **`woys-clean`** — RNNoise-cleaned source (the recommended daily
  driver; ~27 % cuts/min reduction at the cost of ~+40 ms latency, see
  `docs/23-rnnoise-chain.md`).

Since v0.14.1 the raw engine output (node `woys-mic`) is relabelled
`_internal-raw-bypass` while the chain is on, so it sorts with the other
internal nodes. Pick it only if you want the raw path; its usual
`woys-no-cleanup` label comes back after `woys chain teardown`.

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
apps' input-device selector, pick `woys-no-cleanup` (the description of
the `woys-mic` source; apps that list node names show `woys-mic`). That's it.

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

This removes the venv, the launcher and the systemd units. It keeps
`~/.local/share/woys/models/` (the ~1 GB foundation weights plus any voice
models you added); `./uninstall.sh --purge-models` deletes that too.
Your config at `~/.config/woys/config.toml` is always preserved;
delete it manually if you want a fully clean slate.
