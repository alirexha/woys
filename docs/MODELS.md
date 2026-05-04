# Voice models — finding, converting, swapping

vcclient-cachy ships **no bundled voice models**. The `install.sh` step pulls
foundation weights (contentvec, RMVPE, hubert) and a single small public RVC
voice model (`amitaro_v2_16k.onnx`) for the smoke test. Everything else you
add yourself.

## Where models live

All weights cache under:

```
~/.local/share/vcclient-cachy/models/
```

| File                      | Role                                                | Size  |
|---------------------------|-----------------------------------------------------|-------|
| `contentvec-f.onnx`       | Content encoder (extracts speaker-agnostic feats)   | 360 MB |
| `rmvpe_wrapped.onnx`      | Pitch (f0) detector — RVC's preferred              | 345 MB |
| `hubert_base.pt`          | Fallback PyTorch embedder (used if contentvec stub fails) | 180 MB |
| `amitaro_v2_16k.onnx`     | Sample RVC voice (publicly licensed)               | 64 MB |
| `<your-voice>.onnx`       | Whatever voice models you drop in here              | varies |

## Finding RVC models

RVC ONNX models float around the internet. Common sources:

- **Hugging Face** — search [`models?other=rvc`](https://huggingface.co/models?other=rvc).
  The `wok000` and `lj1995` repos host the foundation weights and many sample voices.
- **weights.gg** — community hub with thousands of `.pth` voice clones.
- **Rejekts / RVC_PlayGround** Spaces — usable for testing without download.

License rules:
- For *your own* use, anything you have rights to is fine.
- For *streaming or recording* with someone else's voice, get permission.
  Voice cloning of public figures without consent is legally and ethically
  hazardous.

## Adding an ONNX model

If you already have an `.onnx`:

```
cp /path/to/your-voice.onnx ~/.local/share/vcclient-cachy/models/
```

To switch the engine to it, edit `~/.config/vcclient-cachy/config.toml`:

```toml
rvc_model = "/home/<you>/.local/share/vcclient-cachy/models/your-voice.onnx"
```

Restart the TUI (`q` then `vcclient-cachy run --autostart`) — the new model
loads on engine start.

## Converting `.pth` → `.onnx`

vcclient-cachy ships a `convert` subcommand placeholder; the real implementation
is on the follow-up roadmap. Use one of these manual paths in the meantime:

### Option A — upstream voice-changer's web UI

If you have Docker:

```
docker run -d --gpus all --rm -p 18888:18888 \
    --name vcclient-upstream wokad/voice-changer:latest
```

1. Open `http://localhost:18888`.
2. Click **Edit** on a slot, upload your `.pth`, optionally `.index`.
3. Click **Export ONNX**. The result lands in the slot directory.
4. `docker exec vcclient-upstream find /resources -name '*.onnx'` → copy out.
5. `docker stop vcclient-upstream`.

This is the easiest path because upstream's converter handles the metadata
inspection (sample rate, embedder type, f0 flag, etc.) automatically.

### Option B — manual `torch.onnx.export`

For users who don't want Docker, here's the minimal recipe. **You need to know
the model's flavor first** — that's `RVC v1`, `v2`, with-f0 / without-f0, and
`embChannels` of 256 (v1) or 768 (v2). Most models off Hugging Face are v2 / 768
/ with-f0; weights.gg metadata usually mentions it.

```python
# convert_pth_to_onnx.py — drop into vcclient-cachy root and run
import sys, torch
sys.path.insert(0, "src/server")  # so upstream-style imports resolve

from voice_changer.RVC.onnxExporter.SynthesizerTrnMs768NSFsid_ONNX import (
    SynthesizerTrnMs768NSFsid as Synth,
)

PTH = "your-voice.pth"          # input
OUT = "your-voice.onnx"         # output

state = torch.load(PTH, map_location="cpu")
hps = state.get("config")
if hps is None:
    raise SystemExit("model file has no embedded config — use Option A instead")

net = Synth(*hps, is_half=False)
net.load_state_dict(state["weight"], strict=False)
net.eval()

# Dummy inputs — match RVC's expected dtypes/shapes.
feats = torch.randn(1, 200, 768)              # content vec, 768-dim for v2
p_len = torch.LongTensor([200])
pitch = torch.LongTensor([[0] * 200])
pitchf = torch.zeros(1, 200)
sid = torch.LongTensor([0])

torch.onnx.export(
    net, (feats, p_len, pitch, pitchf, sid), OUT,
    input_names=["feats", "p_len", "pitch", "pitchf", "sid"],
    output_names=["audio"],
    opset_version=17,
    dynamic_axes={
        "feats":  {1: "feats_dynamic_axes_1"},
        "pitch":  {1: "pitch_dynamic_axes_1"},
        "pitchf": {1: "pitchf_dynamic_axes_1"},
    },
)
print(f"wrote {OUT}")
```

Run:

```
.venv/bin/python convert_pth_to_onnx.py
```

For 256-dim (v1) models, swap the import to `SynthesizerTrnMs256NSFsid_ONNX`
and change `feats` to `(1, 200, 256)`.

For nono (no-f0) models, swap to `..._nono_ONNX` and drop the `pitch`/`pitchf`
inputs.

### Option C — wait for `vcclient-cachy convert`

If hand-coding ONNX exports isn't your idea of a fun afternoon, the
`vcclient-cachy convert <path.pth>` subcommand is the planned shortcut. It's
currently a stub that prints these instructions; the real implementation lands
in a follow-up release.

## Voice quality tips

- **Use 40k models for quality, 16k for latency.** The amitaro_v2_16k sample
  is ideal for testing latency; for production-quality voice, 40k is better.
- **Pitch shift should match speaker pitch.** A male-to-female voice usually
  needs `+12` semitones; female-to-male `-12`. Start at 0 and tweak with `+`/`-`.
- **Use the model's index file** (`.index`) when available. vcclient-cachy
  doesn't use it yet (Phase 5+ enhancement); upstream's UI does.

## Removing a model

Just delete the `.onnx` file. If it was the active model, the engine falls
back to the configured default on next start.

```
rm ~/.local/share/vcclient-cachy/models/old-voice.onnx
```
