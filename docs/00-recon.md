# Phase 0 — Recon: `upstream/` for `vcclient-cachy`

Scope: trace exactly which code in `upstream/` (a clone of `w-okada/voice-changer`)
implements the **RVC inference path on ONNX Runtime CUDA EP for Linux/PipeWire**,
and what to delete around it. Every claim is anchored to `path:line`. All paths
are relative to `/home/alireza/ai/vcclient-cachy/upstream/` unless otherwise
stated.

---

## 1. Repo overview

### Tree (depth 2)

```
upstream/
├── client/                  React/TS + webpack web UI (the *only* "real" UI)
│   ├── demo/                Application bundle — what gets served by FastAPI
│   ├── lib/                 `@dannadori/voice-changer-client-js` library + worklet
│   ├── python/              Stand-alone Python "thin" client (vc_client.py)
│   └── buildAllDemo.sh
├── docker/                  Legacy MMVC trainer/server image (CUDA 11.8)
├── docker_folder/           Empty mount points (model_dir/, pretrain/)
├── docker_trainer/          Trainer image (separate from voice-changer server)
├── docker_vcclient/         The runtime image actually used to ship vcclient
├── docs/                    Static GitHub Pages bundle (built front-end mirror)
├── docs_i18n/               Translated documentation
├── recorder/                Standalone web "voice recorder" tool (not used at runtime)
├── script/                  3 docker push helper scripts
├── server/                  Python server. THIS is what we fork.
│   ├── data/                ModelSlot dataclasses (per-engine schemas)
│   ├── downloader/          Weight + sample downloader
│   ├── model_dir_static/    Empty (ships static slots, e.g. Beatrice-JVS)
│   ├── mods/                ssl/, log_control, origins helpers
│   ├── restapi/             FastAPI routers (Hello, VoiceChanger, Fileuploader)
│   ├── sio/                 python-socketio app + namespace
│   ├── tmp_dir/             Empty runtime scratch
│   └── voice_changer/       Per-engine implementations
├── signatures/              Codesigning leftovers
├── trainer/                 Empty subdir set
├── tutorials/               Markdown tutorials w/ images
├── start2.sh                Old MMVC docker launcher
├── start_docker.sh          Newer Linux launcher (vcclient docker)
├── start_v0.1.sh            Very old MMVC launcher
├── Hina_*.ipynb / *.ipynb   Colab/Kaggle launcher notebooks
├── package.json             Top-level npm scripts to build docker images
├── README*.md / README_dev_*.md  Multi-language READMEs
└── LICENSE / LICENSE-CLA / LICENSE-NOTICE
```

### Licenses (separate files)

- `upstream/LICENSE` — primary license (MIT-style; first 3 lines say
  `Copyright (c) 2022 wataru-okada`).
- `upstream/LICENSE-CLA` — Contributor License Agreement.
- `upstream/LICENSE-NOTICE` — third-party notices (the hubert / contentvec /
  RVC / etc. attribution chain).

### Top-level dirs — one-liners

- `client/` — React+TS frontend; lib package (`client/lib`) wraps the
  `MediaStream` + worklet glue, demo (`client/demo`) is the actual SPA.
- `docker/` — legacy MMVC training+server image (do **not** use as our base).
- `docker_folder/` — empty mount stubs that the legacy compose used.
- `docker_trainer/` — trainer-only image (irrelevant to inference).
- `docker_vcclient/` — runtime image used by `start_docker.sh`. Closest thing
  upstream has to a "Linux supported" path.
- `docs/` — auto-built static SPA mirror for GitHub Pages.
- `docs_i18n/` — translations of the README/tutorials.
- `recorder/` — separate React tool; not part of the runtime.
- `script/` — three docker push shell scripts.
- `server/` — Python backend. This is the only thing we keep.
- `signatures/` — leftover code-signing artifacts (macOS notarization).
- `trainer/` — placeholder; only empty subdirs.
- `tutorials/` — user-facing docs/screenshots.

---

## 2. Server architecture (the part we keep)

### 2.1 Web framework

**FastAPI** (mounted under a `python-socketio` ASGI wrapper) served by
**uvicorn**.

- App construction: `server/MMVCServerSIO.py:142-144` builds
  `voiceChangerManager`, then `app_fastapi = MMVC_Rest.get_instance(...)`,
  then `app_socketio = MMVC_SocketIOApp.get_instance(...)`.
- FastAPI itself: `server/restapi/MMVC_Rest.py:51` (`app_fastapi = FastAPI()`).
- Route registration: `server/restapi/MMVC_Rest.py:101-106` (Hello, VoiceChanger,
  Fileuploader routers via `app_fastapi.include_router(...)`).
- Static mounts: `server/restapi/MMVC_Rest.py:59-99` (mounts `/front`,
  `/trainer`, `/recorder`, `/tmp`, `/upload_dir`, `/model_dir_static`,
  `/{model_dir}`).
- Socket.IO ASGI wrap: `server/sio/MMVC_SocketIOApp.py:35-74` (mounts the
  frontend's static assets at the root, including
  `/ort-wasm-simd.wasm` line 59 and Beatrice icons line 63-69).
- Uvicorn entry: `server/MMVCServerSIO.py:124-134` (`uvicorn.run(...)`).

### 2.2 Process model

Single uvicorn process — **but** when launched directly (not via Docker),
`MMVCServerSIO.py:253` spawns a **second** process via
`multiprocessing.Process` to also start a native Tauri/electron-style
"voice-changer-native-client" wrapper. That branch is OS-gated
(`sys.platform.startswith("win")` / `darwin`) — see
`server/MMVCServerSIO.py:256-265` — and is dead on Linux.

A separate worker thread is started for local audio:
`server/voice_changer/VoiceChangerManager.py:86-87` (`thread = threading.Thread(target=self.serverDevice.start, ...)`).

So at runtime there is **one** uvicorn process plus **one** sounddevice worker
thread inside it. That's the whole model.

### 2.3 Inference entry points (where audio enters the pipeline)

There are three paths that all converge on `VoiceChangerManager.changeVoice`:

1. **REST `POST /test`** (browser legacy path):
   `server/restapi/MMVC_Rest_VoiceChanger.py:24` registers the route;
   `server/restapi/MMVC_Rest_VoiceChanger.py:46` calls
   `self.voiceChangerManager.changeVoice(unpackedData)`.
2. **Socket.IO `request_message`**:
   `server/sio/MMVC_Namespace.py:41-56` — handler unpacks int16 PCM and
   calls `self.voiceChangerManager.changeVoice(...)` at line 52.
3. **Server-side audio (sounddevice loop)**:
   `server/voice_changer/Local/ServerDevice.py:138`
   (`self.serverDeviceCallbacks.on_request(unpackedData)`); `on_request` is
   wired in `server/voice_changer/VoiceChangerManager.py:56-57`.

`changeVoice` is at `server/voice_changer/VoiceChangerManager.py:364-372`.
It dispatches to `self.voiceChanger.on_request(receivedData)` — for the RVC
path, `voiceChanger` is a `VoiceChangerV2` instance whose `on_request` is at
`server/voice_changer/VoiceChangerV2.py:215-346` (this is where SOLA crossfade,
input/output resampling and timing live).

The actual model call inside `VoiceChangerV2.on_request` happens at
`server/voice_changer/VoiceChangerV2.py:239` (`self.voiceChanger.inference(...)`).
For RVC that resolves to `RVCr2.inference` at
`server/voice_changer/RVC/RVCr2.py:182-264`, which itself calls
`self.pipeline.exec(...)` at `server/voice_changer/RVC/RVCr2.py:228`.

### 2.4 Model loader for RVC

- **Slot generator (parses `.pth`/`.onnx`, fills `RVCModelSlot`)**:
  `server/voice_changer/RVC/RVCModelSlotGenerator.py:14-38` (entry
  `loadModel`); `_setInfoByONNX` at line 121, `_setInfoByPytorch` at line 41.
  The `.onnx` vs `.pth` switch is at line 27:
  `slotInfo.isONNX = slotInfo.modelFile.endswith(".onnx")`.
- **Inferencer dispatch (which class actually loads the file)**:
  `server/voice_changer/RVC/inferencer/InferencerManager.py:30-64`
  (the giant if/elif on `EnumInferenceTypes`). Two ONNX branches:
  `onnxRVC` → `OnnxRVCInferencer` (line 56-57) and `onnxRVCNono` →
  `OnnxRVCInferencerNono` (line 58-59).
- **ONNX inferencer load**:
  `server/voice_changer/RVC/inferencer/OnnxRVCInferencer.py:10-32`
  (constructs `onnxruntime.InferenceSession`, picks fp16 vs fp32 by
  inspecting `first_input_type`).
- **fp16 / CUDA EP selection**: `server/voice_changer/RVC/deviceManager/DeviceManager.py`
  - Provider list for ONNX: `getOnnxExecutionProvider` lines 36-60. CUDA path:
    line 39-41 `return ["CUDAExecutionProvider"], [{"device_id": gpu}]`.
    Falls back to DirectML at line 51 (Windows-only — we cut), else CPU.
  - Half-precision decision (PyTorch path): `halfPrecisionAvailable` lines
    65-90. Hard-codes a blacklist by GPU name (line 76-80: GTX 16xx, P40,
    1070, 1080) and a compute-capability gate (line 87 `cap[0] < 7`).

### 2.5 ONNX Runtime usage (every site)

- `SessionOptions` use (only one place): `server/voice_changer/RVC/pitchExtractor/RMVPEOnnxPitchExtractor.py:26-28`
  (`so = onnxruntime.SessionOptions(); so.log_severity_level = 3`).
- `InferenceSession` constructions (relevant to RVC):
  - RVC inferencer: `server/voice_changer/RVC/inferencer/OnnxRVCInferencer.py:17-19`.
  - RVC slot probe (CPU only, just to read metadata):
    `server/voice_changer/RVC/RVCModelSlotGenerator.py:122`.
  - RMVPE ONNX pitch extractor: `server/voice_changer/RVC/pitchExtractor/RMVPEOnnxPitchExtractor.py:28`.
  - ONNX-Crepe pitch extractor: `server/voice_changer/RVC/pitchExtractor/CrepeOnnxPitchExtractor.py:19`.
- **Providers list**: built in `DeviceManager.getOnnxExecutionProvider`
  (see 2.4); call sites pass it through:
  - `OnnxRVCInferencer.py:13-19`.
  - `RMVPEOnnxPitchExtractor.py:19-28`.
  - `CrepeOnnxPitchExtractor.py:13-19` (read separately to confirm).
- **IO binding** (`io_binding` / `run_with_iobinding`): **not used anywhere**
  in `server/`. Confirmed by grep — only `run([...], {...})` style. This is
  a real perf opportunity for the fork.
- **PyTorch fallback for RVC**: yes, multiple. Even when the user picks
  ONNX models, a PyTorch path is used for the embedder by default because
  `OnnxContentvec` is **a stub that always raises**:
  `server/voice_changer/RVC/embedder/OnnxContentvec.py:7-13`
  (`raise Exception("Not implemented")` in both `loadModel` and
  `extractFeatures`). So even with `--content_vec_500_onnx_on=True` (the
  default), `EmbedderManager.loadEmbedder` falls through the `try/except`
  at `server/voice_changer/RVC/embedder/EmbedderManager.py:39-48` and
  always uses **`FairseqHubert`** (PyTorch + fairseq).
  - `FairseqHubert` at `server/voice_changer/RVC/embedder/FairseqHubert.py:1-46`
    imports `from fairseq import checkpoint_utils` (line 4) — this is the
    biggest non-obvious dep we inherit.
  - The `pyTorchRVC*` inferencer family
    (`server/voice_changer/RVC/inferencer/RVCInferencer*.py`,
    `WebUIInferencer*.py`, `VorasInferencebeta.py`) is used whenever the user
    loads a `.pth`. Pipeline execution still calls
    `torch.cuda.amp.autocast` at `server/voice_changer/RVC/pipeline/Pipeline.py:6, 103, 120`.
    So PyTorch is **load-bearing** even for "ONNX RVC" — we need to either
    finish the OnnxContentvec stub or keep torch around.

### 2.6 f0 detectors (Harvest / Crepe / RMVPE / dio / pm / fcpe)

Registry: `server/voice_changer/RVC/pitchExtractor/PitchExtractorManager.py:30-54`.
The `PitchExtractorType` literal in `server/const.py:85-94` lists all keys.

| Key in const.py | Class | File | Backend |
|---|---|---|---|
| `harvest` | `HarvestPitchExtractor` | `server/voice_changer/RVC/pitchExtractor/HarvestPitchExtractor.py:9` | `pyworld.harvest` (CPU, line 29-35) |
| `dio` | `DioPitchExtractor` | `server/voice_changer/RVC/pitchExtractor/DioPitchExtractor.py:1, 28` | `pyworld.dio` (CPU) |
| `crepe` | `CrepePitchExtractor` | `server/voice_changer/RVC/pitchExtractor/CrepePitchExtractor.py:9` | `torchcrepe` (PyTorch) |
| `crepe_full` / `crepe_tiny` | `CrepeOnnxPitchExtractor` | `server/voice_changer/RVC/pitchExtractor/CrepeOnnxPitchExtractor.py` | ONNX |
| `rmvpe` | `RMVPEPitchExtractor` | `server/voice_changer/RVC/pitchExtractor/RMVPEPitchExtractor.py:8-21` | PyTorch via `RMVPE(model_path=...)` (line 21) — **imports the model class from `voice_changer.DiffusionSVC.pitchExtractor.rmvpe.rmvpe`** at line 4 |
| `rmvpe_onnx` | `RMVPEOnnxPitchExtractor` | `server/voice_changer/RVC/pitchExtractor/RMVPEOnnxPitchExtractor.py:8-28` | ONNX |
| `fcpe` | `FcpePitchExtractor` | `server/voice_changer/RVC/pitchExtractor/FcpePitchExtractor.py` | `torchfcpe` (PyTorch) |

**Is RMVPE ONNX-based already?** Yes when the user selects `rmvpe_onnx` —
that path is fully ONNX (`RMVPEOnnxPitchExtractor.py:28`). The default in
`RVCSettings` is in fact `rmvpe_onnx`:
`server/voice_changer/RVC/RVCSettings.py:9` (`f0Detector: str = "rmvpe_onnx"`).

**Critical bleed**: `RMVPEPitchExtractor.py:3-4` and
`RMVPEOnnxPitchExtractor.py:3` both `from voice_changer.DiffusionSVC.pitchExtractor.PitchExtractor import PitchExtractor`.
`RMVPEPitchExtractor.py:4` *also* imports the actual model class from
`voice_changer.DiffusionSVC.pitchExtractor.rmvpe.rmvpe`. **You cannot delete
`voice_changer/DiffusionSVC/` wholesale without first re-homing the RMVPE
abstract class and the rmvpe model class.** See section 9.

There is no `pm` (PyWorld pm) extractor implemented, despite being a common
RVC option upstream — it's listed in some upstream forks but not here.

### 2.7 Audio I/O on the server side

- Library: **`sounddevice`** (binding to PortAudio).
  - `server/voice_changer/Local/AudioDeviceList.py:1` (`import sounddevice as sd`).
  - `server/voice_changer/Local/ServerDevice.py:11` (`import sounddevice as sd`).
- Streams (server-driven mode):
  `server/voice_changer/Local/ServerDevice.py:235, 236, 249, 250, 263-265`
  use `sd.InputStream` / `sd.OutputStream` / `sd.Stream`.
- Browser-driven mode: the client sends raw int16 PCM over Socket.IO
  (`server/sio/MMVC_Namespace.py:50`) or REST `/test`
  (`server/restapi/MMVC_Rest_VoiceChanger.py:42`). No server-side capture
  in this case.
- Device discovery: `server/voice_changer/Local/AudioDeviceList.py:59-116`
  uses `sd.query_devices()` and `sd.query_hostapis()`.
- Host-API-specific code: WASAPI exclusive-mode toggles at
  `server/voice_changer/Local/ServerDevice.py:307-314` (Windows-only — cut).
  No ALSA/JACK/PipeWire-specific code; PortAudio sees PipeWire as a regular
  ALSA host on Linux when the `libportaudio2` build supports it.
- Sample rate ladder: `server/const.py:24`
  (`SERVER_DEVICE_SAMPLE_RATES = [16000, 32000, 44100, 48000, 96000, 192000]`).

---

## 3. Engines we will REMOVE

Common cross-cutting registries that name every engine:

- `VoiceChangerType` Literal: `server/const.py:8-18` (lists all 9 engines).
- `EnumInferenceTypes`: `server/const.py:68-79` (RVC + WebUI + VoRAS variants;
  most map to RVC sub-architectures so we keep onnxRVC/onnxRVCNono only).
- Engine → loader dispatch (slot creation): `server/voice_changer/VoiceChangerManager.py:167-213` (10-arm if/elif on `params.voiceChangerType`).
- Engine → runtime model dispatch (slot activation): `server/voice_changer/VoiceChangerManager.py:243-329` (8-arm if/elif on `slotInfo.voiceChangerType`).
- Engine → ModelSlot dataclass dispatch: `server/data/ModelSlot.py:166-204` (`loadSlotInfo`).
- Sample downloader engine handling: `server/downloader/SampleDownloader.py`
  has per-engine arms (e.g. line 155-200 for Diffusion-SVC).

These four sites need to be reduced to the RVC arm only.

### 3.1 MMVC (v1.3 + v1.5)

- Dirs: `server/voice_changer/MMVCv13/`, `server/voice_changer/MMVCv15/`.
- Entry classes:
  - `server/voice_changer/MMVCv13/MMVCv13.py` (class `MMVCv13`; module also
    contains an `if sys.platform.startswith("darwin"):` Mac-specific
    `sys.path` hack at line 8-17).
  - `server/voice_changer/MMVCv15/MMVCv15.py` (class `MMVCv15`; same Mac
    `sys.path` hack at line 7).
- Slot generator: `MMVCv13ModelSlotGenerator.py`, `MMVCv15ModelSlotGenerator.py`.
- Registry hits to delete: `VoiceChangerManager.py:172-181, 262-275`,
  `data/ModelSlot.py:46-69, 177-182`.
- Hard-coded sample rates: `server/data/ModelSlot.py:54, 67`
  (both default `samplingRate: int = 24000`).
- Notes: requires a separate runtime checkout of `MMVC_Client_v13` /
  `MMVC_Client_v15` repos (see `docker/Dockerfile:19-20`). Pure dead weight.

### 3.2 Beatrice

- Dir: `server/voice_changer/Beatrice/`.
- Entry class: `server/voice_changer/Beatrice/Beatrice.py:11`. **The class
  is already a stub** — every method `raise RuntimeError("not implemented")`
  (lines 13, 16, 19, 22, 25, 28, 36, 39). It's wired but non-functional;
  removing it is purely a deletion of registry entries.
- Slot generator: `server/voice_changer/Beatrice/BeatriceModelSlotGenerator.py`.
- Static-slot machinery: `Beatrice-JVS` is a *special* slot keyed off the
  `StaticSlot` literal at `server/const.py:20`. It has dedicated branches:
  - `VoiceChangerManager.py:301-306` (`val == "Beatrice-JVS"` static flag).
  - `data/ModelSlot.py:194-196` (`if slotIndex == "Beatrice-JVS"`).
  - `restapi/MMVC_Rest.py:79` (mounts `/model_dir_static`).
  - `sio/MMVC_SocketIOApp.py:63-69` (Beatrice icon static routes).
- Registry hits to delete: `VoiceChangerManager.py:197-201, 297-306`,
  `data/ModelSlot.py:129-134, 192-196`, `const.py:15, 20`,
  `sio/MMVC_SocketIOApp.py:63-69`.
- The crossfade-disable special case in VoiceChangerV2 also keys off
  Beatrice: `server/voice_changer/VoiceChangerV2.py:103-106` (sets
  `noCrossFade = True`). After removing Beatrice, this branch and the
  `if self.noCrossFade:` arm at line 223-231 can collapse.

### 3.3 so-vits-svc-40

- Dir: `server/voice_changer/SoVitsSvc40/`.
- Entry class: `server/voice_changer/SoVitsSvc40/SoVitsSvc40.py:9` has
  Mac `sys.path` hack; the class itself imports `fairseq` at line 38.
- Slot generator: `server/voice_changer/SoVitsSvc40/SoVitsSvc40ModelSlotGenerator.py`.
- Registry hits: `VoiceChangerManager.py:182-186, 276-282`,
  `data/ModelSlot.py:73-86, 183-185`.
- Sample-rate hardcodes: model loader uses `sampling_rate=44100` default at
  `server/voice_changer/SoVitsSvc40/models/models.py:271`.

### 3.4 DDSP-SVC

- Dir: `server/voice_changer/DDSP_SVC/`.
- Entry class: `server/voice_changer/DDSP_SVC/DDSP_SVC.py:11`
  (Mac `sys.path` hack again).
- Slot generator: `server/voice_changer/DDSP_SVC/DDSP_SVCModelSlotGenerator.py`.
- Registry hits: `VoiceChangerManager.py:187-191, 283-289`,
  `data/ModelSlot.py:89-105, 186-188`.
- Bleed: `server/voice_changer/DDSP_SVC/models/ddsp/vocoder.py:10` imports
  `from fairseq import checkpoint_utils`.

### 3.5 Bonus — also remove

These are not in the user-listed four but are equivalent dead weight and
have the same registry footprint:

- **Diffusion-SVC** (`server/voice_changer/DiffusionSVC/`) — entry class at
  `DiffusionSVC.py`. *However* note section 2.6: RVC's RMVPE pitch extractors
  import from this directory. We must move
  `voice_changer/DiffusionSVC/pitchExtractor/PitchExtractor.py` and
  `voice_changer/DiffusionSVC/pitchExtractor/rmvpe/` into a neutral location
  (e.g. `voice_changer/RVC/pitchExtractor/_rmvpe/`) **before** deleting the
  rest of the dir. Registry hits: `VoiceChangerManager.py:192-196, 290-296`,
  `data/ModelSlot.py:108-126, 189-191`.
- **EasyVC** (`server/voice_changer/EasyVC/`) — entry class
  `EasyVC.py`; only ONNX. Registry hits:
  `VoiceChangerManager.py:209-213, 316-323`, `data/ModelSlot.py:144-149,
  200-202`. The `easyVC` branch in
  `server/voice_changer/RVC/inferencer/InferencerManager.py:60-61` and
  `EasyVCInferencerONNX.py` cross-resides under RVC — needs deletion.
- **LLVC** (`server/voice_changer/LLVC/`) — entry class `LLVC.py`. Registry
  hits: `VoiceChangerManager.py:203-207, 307-314`,
  `data/ModelSlot.py:137-142, 197-199`. Triggers
  `server/voice_changer/VoiceChangerV2.py:299-302` (`if self.voiceChanger.voiceChangerType == "LLVC"`).

LOC summary (engines slated for removal, all under `server/voice_changer/`):

```
   $ wc -l Beatrice/*.py MMVCv13/*.py MMVCv15/*.py SoVitsSvc40/*.py \
           DDSP_SVC/*.py DiffusionSVC/*.py EasyVC/*.py LLVC/*.py \
           (recursively)
   ~22,200 LOC across these dirs (vs ~3,200 for RVC + utils).
```

---

## 4. Windows / WSL / macOS code paths

Source/grep evidence inside `server/`:

| Site | Note |
|---|---|
| `server/MMVCServerSIO.py:75-94` | `printMessage()` branches on `platform.system() == "Windows"` to skip ANSI color codes. Trivial — keep on Linux side. |
| `server/MMVCServerSIO.py:256-265` | Spawns native client subprocess on `sys.platform.startswith("win")` and `darwin`. Unused on Linux. Delete. |
| `server/restapi/MMVC_Rest.py:83-93` | Mac-only PyInstaller `_MEIPASS` path mangling for `model_dir`. Delete. |
| `server/downloader/SampleDownloader.py:156` | `if sys.platform.startswith("darwin") is True: continue` — skips DiffusionSVC sample download on Mac. Removed when we delete DiffusionSVC. |
| `server/downloader/SampleDownloader.py:226` | Mirror of the above for the metadata pass. |
| `server/voice_changer/RVC/inferencer/InferencerManager.py:44` | `if sys.platform.startswith("darwin") is False: from VorasInferencebeta...`. We'll delete VoRAS support; the branch goes too. |
| `server/voice_changer/MMVCv13/MMVCv13.py:8-17` | Mac `sys.path` hack. Goes with MMVCv13 deletion. |
| `server/voice_changer/MMVCv15/MMVCv15.py:7` | Same. |
| `server/voice_changer/SoVitsSvc40/SoVitsSvc40.py:9` | Same. |
| `server/voice_changer/DDSP_SVC/DDSP_SVC.py:11` | Same. |
| `server/voice_changer/Local/ServerDevice.py:307-314` | `if "WASAPI" in serverInputAudioDevice.hostAPI: sd.WasapiSettings(exclusive=True)`. Remove (WASAPI is Windows-only). |
| `server/voice_changer/RVC/deviceManager/DeviceManager.py:51-52` | `DmlExecutionProvider` (DirectML) branch — Windows-only. Delete. |
| `server/voice_changer/RVC/deviceManager/DeviceManager.py:17-20, 24-27` | MPS (macOS Metal) branch in `getDevice`. Delete. |
| `server/const.py:32-46, 48` | `NATIVE_CLIENT_FILE_WIN`, `NATIVE_CLIENT_FILE_MAC`, `_MEIPASS` paths. Remove. |

WSL is mentioned only in user-facing READMEs (`README_dev_*.md` lines 7, 65,
77, 82, 87 across en/ja/ko/ru). No code branches on WSL. The
`/usr/lib/wsl/lib` LD_LIBRARY_PATH advice is purely documentation.

`.bat` / `.ps1` / `.cmd` / `.command` files: **none in upstream/** (they
ship inside the released zip but not in the repo). All references are in
tutorial markdown.

`pyaudio`: not used (search returns only `sounddevice`). MMSystem / MME /
ASIO / DirectSound: only mentioned in Korean tutorial markdown
(`tutorials/tutorial_monitor_consept_ko.md:13, 41` and JA equivalent),
no code references.

`pywin32` / Windows-specific deps: not in `server/requirements.txt`.

---

## 5. Web client (kept as fallback, narrowed to RVC)

### 5.1 Where it lives

- `client/lib/` — `@dannadori/voice-changer-client-js` (the audio worklet +
  socket.io plumbing). `package.json` at `client/lib/package.json`.
- `client/demo/` — the actual SPA, depends on the lib via npm.
  `package.json` at `client/demo/package.json:65`
  (`"@dannadori/voice-changer-client-js": "^1.0.182"`). Build is **webpack**
  (not Vite) — `client/demo/webpack.common.js`, `webpack.dev.js`,
  `webpack.prod.js`.
- Pre-built bundle that gets served by the Python server lives at
  `client/demo/dist/index.js` (3.3 MB) — referenced by
  `server/const.py:55-57` (`getFrontendPath` → `../client/demo/dist`).
- Top-level `package.json` only has docker-build scripts; there is no
  Vite anywhere.

### 5.2 Engine-selection switch points (need narrowing)

The client knows about all engines in two source-of-truth files:

- `client/lib/src/const.ts:6-17` — the `VoiceChangerType` const enum (lists
  10 engines including `WebModel`).
- `client/lib/src/const.ts:70-80` — `RVCModelType` enum (the RVC sub-types).

Switch sites in the demo that branch on `voiceChangerType`:

- `client/demo/src/components/demo/904-3_FileUploader.tsx:67-122` — file
  uploader has per-engine arms (mmvcv13, mmvcv15, so-vits-svc-40, DDSP-SVC,
  Diffusion-SVC, Beatrice, LLVC).
- `client/demo/src/components/demo/904-2_SampleDownloader.tsx:89` — handles
  `DiffusionSVCSampleModel`.
- `client/demo/src/components/demo/components2/101-3_SpeakerArea.tsx:25, 86, 89` — branches on engine type for speaker/voice UI.
- `client/demo/src/components/demo/components2/101-2_IndexArea.tsx:12-14` — Beatrice-JVS special case.
- `client/demo/src/components/demo/components2/101-4_F0FactorArea.tsx:3, 13-31` — MMVCv15-only F0 factor UI.
- `client/demo/src/components/demo/components2/100_ModelSlotArea.tsx:44` — Beatrice-JVS slot directory branching.

These all need to either be deleted (preferred) or short-circuited to the
RVC branch only.

The web client also uses `onnxruntime-web` for browser-side inference of
the optional `WebModel` engine: `client/demo/package.json:74`. We can drop
that too since we're keeping only the server RVC path.

---

## 6. Bundled models / weights / huge assets

`find . -type f \( -name '*.onnx' -o -name '*.pth' -o -name '*.pt' -o
 -name '*.bin' -o -name '*.safetensors' -o -name '*.npy' \)` returns
**zero** matches inside `upstream/`. Good — no weights are vendored.

What *is* large:

| Path | Size | Why |
|---|---|---|
| `upstream/server/test.wav` | 1.5 MB | Manual test audio. Drop. |
| `upstream/client/demo/dist/index.js` | 3.3 MB | Pre-built SPA bundle. Keep until we re-trim+rebuild. |
| `upstream/docs/index.js` | 3.1 MB | GitHub Pages static mirror of the same SPA. Drop. |
| `upstream/client/demo/public/settings/*.psd` | several large `.psd` files | Photoshop sources for character icons. Drop. |
| `upstream/recorder/package-lock.json` | 824 KB | Standalone tool. Drop entire `recorder/`. |

Weights are downloaded at first launch by `server/downloader/WeightDownloader.py`:
`hubert_base.pt`, `hubert-soft-0d54a1f4.pt`, `nsf_hifigan/model.bin`,
`crepe_onnx_full.onnx`, `crepe_onnx_tiny.onnx`, `contentvec-f.onnx`,
`rmvpe_20231006.pt`, `rmvpe_20231006.onnx`, `tiny.pt` (Whisper).
For an RVC-only fork we only need: **`hubert_base.pt`** (or the ONNX
contentvec once we implement it), **`rmvpe_20231006.onnx`**, and optionally
**`crepe_onnx_*`** for the alternative pitch detectors.

---

## 7. Dependency graph

### 7.1 `server/requirements.txt` (verbatim, line numbers from file)

```
6  uvicorn==0.21.1
7  pyOpenSSL==23.1.1
8  numpy==1.23.5
9  torch==2.0.1                 # GPU/CPU; we want CUDA build
10 torchaudio==2.0.2            # GPU/CPU
11 resampy==0.4.2
12 python-socketio==5.8.0
13 fastapi==0.95.1
14 python-multipart==0.0.6
15 onnxruntime-gpu==1.13.1      # GPU — REQUIRED for our path
16 scipy==1.10.1
17 matplotlib==3.7.1            # only used for trainer; can drop
18 websockets==11.0.2
19 faiss-cpu==1.7.3             # RVC index search; CPU is fine
20 torchcrepe==0.0.18           # only if we keep `crepe` extractor
21 librosa==0.9.1
22 gin==0.1.6                   # used by DDSP-SVC
23 gin_config==0.5.0            # used by DDSP-SVC
24 einops==0.6.0                # used widely; keep
25 local_attention==1.8.5       # only used by DiffusionSVC PCmer
26 websockets==11.0.2           # duplicate
27 sounddevice==0.4.6
28 dataclasses_json==0.5.7      # not actually imported — check & drop
29 onnxsim==0.4.28              # only used by export2onnx
30 torchfcpe==0.0.3             # only if we keep `fcpe` extractor
```

GPU-only deps flagged:
- `torch==2.0.1` (we need a CUDA wheel — but PyTorch is also CPU-capable,
  so it's a "GPU-when-available" dep).
- `onnxruntime-gpu==1.13.1` — **strict GPU**, this is what binds CUDA EP.
  Note 1.13.1 is **very old** (mid-2022). Phase 1 should bump to a 1.18+
  build that matches our CUDA toolkit on CachyOS.
- `faiss-cpu` — labeled CPU; no GPU faiss in upstream.

### 7.2 Implicit deps not in `requirements.txt`

- **`pyworld`** — not in `requirements.txt` but installed by
  `docker_vcclient/Dockerfile:19`
  (`pip install pyworld==0.3.3 --no-build-isolation`).
  Imported by `Harvest`/`Dio` extractors (`server/voice_changer/RVC/pitchExtractor/HarvestPitchExtractor.py:1`,
  `DioPitchExtractor.py:1`). If we keep `harvest`/`dio`, we need it
  explicitly.
- **`fairseq`** — not in `requirements.txt`! It's installed via the
  shadow MMVC_Client repo's deps inside `docker/Dockerfile:19-20`.
  Imported by `server/voice_changer/RVC/embedder/FairseqHubert.py:4`
  (and similar in deletable engines). We need this explicitly to make the
  PyTorch hubert path work — or finish the `OnnxContentvec` stub and drop
  fairseq entirely.

### 7.3 CUDA / cuDNN version hints

- `upstream/docker/Dockerfile:1` and `:32`: `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04`.
- `upstream/docker_vcclient/Dockerfile:1`: same. Line 2 has a commented-out
  CUDA 12.0 alternative.
- No version pinning inside Python code itself; everything is via the
  base image.

### 7.4 Windows-only deps in upstream's deps

`pywin32` — not present. `winsdk` — not present. `pythonnet` — not present.
The only Windows-flavored dep was `onnxruntime-gpu` itself supporting
`DmlExecutionProvider` at runtime (no Python-side import).

---

## 8. Linux already-supported paths

**There are no native Linux start scripts.** Every path upstream supports
on Linux is via Docker.

- `start_docker.sh` — top-level docker launcher; uses
  `DOCKER_IMAGE=dannadori/voice-changer:...` (read full file to confirm).
- `start2.sh:53-60` — `docker run -it --rm --gpus all --shm-size=128M ...
  $DOCKER_IMAGE`. Linux-shaped.
- `docker_vcclient/Dockerfile:1` — Linux runtime base
  (`nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04`). The `entrypoint`
  invokes `setup.sh` → `exec.sh` → `python3 MMVCServerSIO.py $@`
  (`docker_vcclient/exec.sh:8`). This is the closest-to-Linux path
  upstream offers.
- `docker/exec.sh:29, 33` — copies `.pth`/`.onnx` from a `/resources` mount
  and runs `python3 MMVCServerSIO.py`. Older flavor.
- READMEs explicitly say: "Linux (ubuntu, debian) or WSL2, not tested for
  other linux distributions"
  (`README_dev_en.md:7`, `README_dev_ja.md:7`, `README_dev_ko.md:7`,
  `README_dev_ru.md:9`).

So upstream's "Linux support" is *Ubuntu 22.04 in a CUDA 11.8 container*.
A bare CachyOS install with sounddevice/PortAudio is a new path we need
to build ourselves.

---

## 9. Risk surface for Phase 1 (Lean Core)

Sorted by likely cost-to-fix.

### Load-bearing in non-obvious ways (HIGH risk, fix early)

1. **`OnnxContentvec` is a stub** —
   `server/voice_changer/RVC/embedder/OnnxContentvec.py:7-13`
   raises `Not implemented`. `EmbedderManager` always falls back to
   `FairseqHubert` (`EmbedderManager.py:39-48`). **Reason**: blocks any
   "no torch / no fairseq" goal. Either implement ONNX contentvec
   inference (small; just 12 lines of `onnxruntime.run` against the
   `contentvec-f.onnx` already downloaded by `WeightDownloader.py:107-114`)
   or accept the torch+fairseq dependency.

2. **RVC pitch extractors import from `DiffusionSVC`** —
   `server/voice_changer/RVC/pitchExtractor/RMVPEPitchExtractor.py:3-4`,
   `RMVPEOnnxPitchExtractor.py:3`. **Reason**: deleting `DiffusionSVC/`
   breaks the default f0 detector (`rmvpe_onnx`). Move
   `DiffusionSVC/pitchExtractor/PitchExtractor.py` and
   `DiffusionSVC/pitchExtractor/rmvpe/` into a neutral path (e.g.
   `voice_changer/common/rmvpe/`) before deletion.

3. **PyTorch is required even for the "ONNX" path** —
   `server/voice_changer/RVC/pipeline/Pipeline.py:6, 103, 120` uses
   `torch.cuda.amp.autocast`; `RVCr2.py:216, 244-245` uses `torch.from_numpy`/
   `.detach().cpu().numpy()`. **Reason**: "drop torch" is not a Phase 1
   goal. Keep torch.

4. **`pyworld` is needed for `harvest`/`dio` but absent from
   `requirements.txt`** — `HarvestPitchExtractor.py:1`,
   `DioPitchExtractor.py:1`. **Reason**: a fresh `pip install -r
   requirements.txt` on CachyOS will silently break those f0 detectors.
   Fix by adding `pyworld>=0.3.3` to the fork's requirements or by
   removing `harvest`/`dio` if we're committed to RMVPE-only.

5. **`fairseq` is needed but absent from `requirements.txt`** —
   `FairseqHubert.py:4`. Same risk class as `pyworld`. Either add it or
   finish the OnnxContentvec implementation and delete FairseqHubert/
   FairseqContentvec/FairseqHubertJp wholesale.

6. **VoiceChanger.py (legacy V1) vs VoiceChangerV2** —
   `server/voice_changer/VoiceChanger.py:65` is the old class used by
   MMVCv13/v15/SoVitsSvc40/DDSP_SVC; RVC uses V2 only
   (`VoiceChangerManager.py:259, 295, 305, 312, 321`). After dropping
   non-RVC engines we can delete `VoiceChanger.py` entirely. ~370 LOC.

### Definitely keep

- `server/MMVCServerSIO.py` — entrypoint. Strip the win/darwin native
  client launch (lines 256-265) and the now-dead options (`hubert_base_jp`,
  `hubert_soft`, `nsf_hifigan`, `whisper_tiny`).
- `server/restapi/MMVC_Rest.py` — strip Mac `_MEIPASS` branch (lines 83-93)
  and the `/recorder` and `/trainer` static mounts (lines 65-75).
- `server/restapi/MMVC_Rest_Hello.py` — small.
- `server/restapi/MMVC_Rest_VoiceChanger.py` — REST `/test` endpoint.
- `server/restapi/MMVC_Rest_Fileuploader.py` — file upload + settings;
  keep all routes except `/merge_model` is RVC-specific (good).
- `server/restapi/mods/` — origin/upload helpers.
- `server/sio/MMVC_Namespace.py`, `MMVC_SocketIOApp.py`,
  `MMVC_SocketIOServer.py` — kept whole.
- `server/voice_changer/VoiceChangerManager.py` — strip 9 of 10 engine arms.
- `server/voice_changer/VoiceChangerV2.py` — strip Beatrice/LLVC `noCrossFade`
  branches (lines 103-106, 223-231, 299-302).
- `server/voice_changer/IORecorder.py`, `Local/ServerDevice.py`,
  `Local/AudioDeviceList.py` — keep, with WASAPI removal at
  `ServerDevice.py:307-314`.
- `server/voice_changer/RVC/**` — keep nearly everything **except**:
  - `pitchExtractor/onnxcrepe/` and `pitchExtractor/torchcrepe2/` —
    can drop unless we keep `crepe_full`/`crepe_tiny`.
  - `inferencer/EasyVCInferencerONNX.py` — depends on EasyVC; delete with
    `easyVC` arm in `InferencerManager.py:60-61`.
  - `inferencer/VorasInferencebeta.py` and `inferencer/voras_beta/` —
    optional Linux-only thing (already gated by `darwin` check at
    `InferencerManager.py:44`); can drop.
  - `inferencer/rvc_models/` — keep (used by torch inferencer).
  - `embedder/whisper/`, `embedder/Whisper.py`, `embedder/FairseqHubertJp.py` —
    only used by exotic embedder types. Drop unless we want JP hubert
    or whisper-as-embedder.
- `server/voice_changer/utils/` — keep wholesale, all small.
- `server/voice_changer/common/VolumeExtractor.py` — only used by
  DDSP/DiffusionSVC, can drop.
- `server/data/ModelSlot.py` — strip 8 of 9 dataclasses + `loadSlotInfo`
  arms.
- `server/downloader/SampleDownloader.py` — strip Diffusion-SVC arms;
  keep RVC arm and `Downloader.py`/`WeightDownloader.py`.
- `server/const.py` — strip `MODEL_DIR_STATIC`, `NATIVE_CLIENT_FILE_*`,
  `HUBERT_ONNX_MODEL_PATH`, the `EnumInferenceTypes` non-RVC enums, the
  non-`production` arms of `getSampleJsonAndModelIds` (lines 119-208) and
  the `Beatrice-JVS` static slot literal (line 20).
- `server/Exceptions.py` — keep.
- `server/mods/` — keep `log_control.py`, `origins.py`, `ssl.py`. Already
  Linux-clean.

### Delete outright

- `server/voice_changer/{Beatrice,DDSP_SVC,DiffusionSVC,EasyVC,LLVC,MMVCv13,MMVCv15,SoVitsSvc40}/`
  (after re-homing the RMVPE bits — see risk #2 above).
- `server/voice_changer/VoiceChanger.py` (legacy V1).
- `server/test.wav` (1.5 MB).
- `client/python/` (independent stand-alone Python client; we'll
  build our own).
- `recorder/` (separate tool).
- `docker/`, `docker_trainer/`, `docker_folder/` (legacy / trainer).
- `tutorials/`, `docs_i18n/`, top-level `*.ipynb`, `start_v0.1.sh`,
  `start2.sh`, `start_docker.sh`.
- `signatures/`.

---

## 10. Recommended `src/server/` layout for the fork

Goal: ~10-15 Python files, RVC + ONNX-CUDA only, no Mac/Windows code paths,
PipeWire-friendly.

```
src/server/
├── __init__.py
├── main.py                 # ex-MMVCServerSIO.py — argparse, uvicorn boot
├── config.py               # ex-const.py — minimal (model_dir, weights paths,
│                           #   sample-rate ladder, EnumInferenceTypes(onnx*))
├── exceptions.py           # ex-Exceptions.py
├── ssl_util.py             # ex-mods/ssl.py
├── logging_util.py         # ex-mods/log_control.py
├── origins.py              # ex-mods/origins.py
│
├── api/                    # ex-restapi/
│   ├── __init__.py
│   ├── app.py              # FastAPI factory (was MMVC_Rest.py)
│   ├── routes_health.py    # was MMVC_Rest_Hello.py
│   ├── routes_inference.py # was MMVC_Rest_VoiceChanger.py
│   ├── routes_files.py     # was MMVC_Rest_Fileuploader.py
│   ├── trusted_origin.py   # was restapi/mods/trustedorigin.py
│   └── upload.py           # was restapi/mods/FileUploader.py
│
├── sio/                    # socket.io is small; keep flat
│   ├── __init__.py
│   ├── server.py           # was MMVC_SocketIOServer.py
│   ├── app.py              # was MMVC_SocketIOApp.py (drop Beatrice mounts)
│   └── namespace.py        # was MMVC_Namespace.py
│
├── audio/                  # local audio I/O
│   ├── __init__.py
│   ├── devices.py          # was Local/AudioDeviceList.py (drop WASAPI bits)
│   ├── server_device.py    # was Local/ServerDevice.py (drop WASAPI bits)
│   └── io_recorder.py      # was IORecorder.py
│
├── manager/
│   ├── __init__.py
│   ├── voice_changer_manager.py # was VoiceChangerManager.py — RVC-only,
│   │                            # both dispatch tables collapse to 1 arm
│   ├── voice_changer_runner.py  # was VoiceChangerV2.py — drop Beatrice/LLVC
│   ├── model_slot_manager.py    # was ModelSlotManager.py
│   ├── voice_changer_params.py  # was VoiceChangerParamsManager.py
│   └── slots.py                 # was data/ModelSlot.py — RVCModelSlot only
│
├── rvc/                    # was voice_changer/RVC/ — only path that runs
│   ├── __init__.py
│   ├── settings.py         # was RVCSettings.py
│   ├── slot_generator.py   # was RVCModelSlotGenerator.py
│   ├── pipeline.py         # was pipeline/Pipeline.py + PipelineGenerator.py
│   ├── runner.py           # was RVCr2.py
│   ├── device_manager.py   # was deviceManager/DeviceManager.py
│   │                       #   — strip MPS + DML branches
│   ├── inferencer/
│   │   ├── __init__.py
│   │   ├── base.py             # was Inferencer.py
│   │   ├── manager.py          # was InferencerManager.py — keep all 6
│   │   │                       #   RVC variants + onnxRVC{,Nono} only
│   │   ├── onnx.py             # was OnnxRVCInferencer{,Nono}.py
│   │   └── torch.py            # was RVCInferencer{,Nono,v2,v2Nono}.py
│   │                           #   + WebUIInferencer{,Nono}.py
│   ├── embedder/
│   │   ├── __init__.py
│   │   ├── base.py             # was Embedder.py
│   │   ├── manager.py          # was EmbedderManager.py — only "hubert_base"/
│   │   │                       #   "contentvec" arms
│   │   ├── fairseq_hubert.py   # was FairseqHubert.py + FairseqContentvec.py
│   │   └── onnx_contentvec.py  # was OnnxContentvec.py — IMPLEMENT ME
│   ├── pitch/
│   │   ├── __init__.py
│   │   ├── base.py                  # was PitchExtractor.py
│   │   ├── manager.py               # was PitchExtractorManager.py
│   │   ├── rmvpe_onnx.py            # was RMVPEOnnxPitchExtractor.py
│   │   ├── rmvpe_torch.py           # was RMVPEPitchExtractor.py
│   │   ├── rmvpe_model/             # ex-DiffusionSVC/pitchExtractor/rmvpe/
│   │   │   └── rmvpe.py             #   re-homed!
│   │   ├── crepe_onnx.py            # was CrepeOnnxPitchExtractor.py + onnxcrepe/
│   │   ├── crepe_torch.py           # was CrepePitchExtractor.py
│   │   ├── harvest.py               # was HarvestPitchExtractor.py
│   │   ├── dio.py                   # was DioPitchExtractor.py
│   │   └── fcpe.py                  # was FcpePitchExtractor.py
│   ├── onnx_export.py      # was onnxExporter/export2onnx.py + 6 helpers
│   └── model_merger.py     # was RVCModelMerger.py + modelMerger/
│
└── downloader/
    ├── __init__.py
    ├── http.py             # was Downloader.py
    ├── weights.py          # was WeightDownloader.py — strip jp hubert,
    │                       #   hubert-soft, nsf_hifigan, whisper urls
    └── samples.py          # was SampleDownloader.py — RVC arm only,
                            #   only `production` mode
```

Targets:
- `~10-15 modules` of business code outside `src/server/rvc/`.
- `src/server/rvc/` itself collapses from ~30 files to ~20.
- Deletes ~22k lines of non-RVC engine code in one pass.

Module renames intentionally drop the `MMVC_` prefix (it's misleading
since we no longer support MMVC) and switch to PEP-8 file names.

---

## Appendix A — Quick reference: "what file is the ONNX RVC inference call"

Top-down:
1. Browser sends int16 PCM → `server/sio/MMVC_Namespace.py:50`.
2. → `VoiceChangerManager.changeVoice` at
   `server/voice_changer/VoiceChangerManager.py:364`.
3. → `VoiceChangerV2.on_request` at `server/voice_changer/VoiceChangerV2.py:215`.
4. → `RVCr2.inference` at `server/voice_changer/RVC/RVCr2.py:182`.
5. → `Pipeline.exec` at `server/voice_changer/RVC/pipeline/Pipeline.py:131`.
6. → `Pipeline.extractPitch` (line 77) → `RMVPEOnnxPitchExtractor.extract`
   at `server/voice_changer/RVC/pitchExtractor/RMVPEOnnxPitchExtractor.py:30` →
   `self.onnx_session.run(...)` line 54.
7. → `Pipeline.extractFeatures` (line 102) → `FairseqHubert.extractFeatures`
   at `server/voice_changer/RVC/embedder/FairseqHubert.py:25`
   (PyTorch path even when "ONNX" is selected — see Risk #1).
8. → `Pipeline.infer` (line 117) → `OnnxRVCInferencer.infer` at
   `server/voice_changer/RVC/inferencer/OnnxRVCInferencer.py:34` →
   `self.model.run(["audio"], {...})` line 50 / 61.
9. Output goes back through SOLA in `VoiceChangerV2.on_request` lines 252-285.
10. Result emitted to client at `server/sio/MMVC_Namespace.py:56`.

That's the entire hot path. Nine files. Everything else in
`server/voice_changer/` is either glue, configuration, or another engine.
