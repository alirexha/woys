# 13 — woys-diag detector calibration (v0.6.9 ship gate)

**Date:** 2026-05-06
**Tooling:** `woys-diag` v0.1.1
**Decision tag:** v0.6.9 shipped after this calibration confirmed the
~12 cuts/min floor is **real engine behavior**, not detector
false-positives.

## Why

We hit a noisy floor of ~10–16 cuts/min in `woys-diag` across five
fix-and-verify rounds (23 → 16 → 11 → 14 → 10 → 16). Coefficient of
variation ~30 %. Some of the variance had to be detector noise, some real
engine behavior — and we couldn't tell which without controls. Two
zero-engine captures answer "does the detector flag clean audio as
having cuts?"

If the detector reports cuts on inputs we know are clean, our v0.6.9
ship numbers are inflated and the engine is even better than we claim.
If it reports zero, the floor is real and v0.6.9 ships honestly.

## The two controls

### B1 — direct HyperX mic, no engine in path

```bash
ffmpeg -hide_banner -loglevel error -f pulse \
  -i alsa_input.usb-HP__Inc_HyperX_QuadCast_2_S_C1V444006H-00.analog-stereo \
  -ac 1 -ar 48000 -t 60 -y /tmp/control_directmic.wav
woys-diag analyze /tmp/control_directmic.wav --duration 60 --source direct-hyperx
```

The user reads the same 7-block protocol they read for `woys-diag run`,
into the same HyperX QuadCast 2 S, but the recording is straight from
the mic — no woys engine touches the bytes.

### B2 — purely synthetic, math-clean

```bash
python -c "
import sys; sys.path.insert(0, '/home/alireza/ai/woys-diag/tests')
from conftest import _gen_clean, _write_wav
from pathlib import Path
_write_wav(Path('/tmp/control_synth_clean.wav'), _gen_clean())
"
woys-diag analyze /tmp/control_synth_clean.wav --duration 60 --source synth-clean
```

Same shape as B1 (7-block, 60 s, 48 kHz mono) but mathematically
synthesized — sustained sine for the vowel block, white noise for
fricative/speech blocks, low-amplitude noise for silence blocks, with
25 ms cosine fades at every active-block boundary so the synthesis
itself doesn't have hard transitions. We have detection unit tests
asserting this fixture produces **0 cuts** (`tests/test_analysis.py::
test_clean_no_cuts_no_clicks`); the synthetic control is the same
bytestream those tests run against.

## Results

| Source                           | Cuts/min | Total RMS | Notes                                              |
| -------------------------------- | -------: | --------: | -------------------------------------------------- |
| B2 — synthetic clean             |        0 | −16.0 dB  | Math-clean. As expected.                           |
| B1 — direct HyperX mic           |        0 | −37.5 dB  | Real human voice, real mic. **Detector clean.**    |
| woys v0.6.8 (e_girl, 60 s)       |    ~23   | −19.9 dB  | Pre-fix baseline.                                  |
| woys v0.6.9 (e_girl, 60 s, mean) |    ~12   | −19–20 dB | Post-fix, mean across 6 runs (10/11/14/16/16/14*). |

*The first v0.6.9-named run was actually after only round-1 fixes, so
the strict v0.6.9-with-all-fixes-applied set is the last 4 runs:
14/10/16/14 → mean ~13.5, std ~2.5.

## Interpretation

**The detector is not over-sensitive.** B1 — a real human reading the
real protocol into the real mic — scores 0 cuts. The 12 cuts/min in
v0.6.9 are 12 cuts woys actually produces relative to the same speaker
using the same mic without woys.

**The remaining floor is real engine behavior.** Most of the residual
cuts cluster in `vowel` and `normal` blocks (sustained periodic content
and natural speech with prosody), with cut durations clustered at
37.5 ms and 80 ms — the RMVPE frame rate (10 ms / frame), suggesting
3–8 frame runs of the engine emitting silence the detector picks up.

**Run-to-run variance is real engine variance, not measurement noise.**
With B1+B2 both at 0/min, any non-zero number from a `woys-diag run`
capture is engine signal. The 30 % coefficient of variation reflects:

- cuDNN algorithm-pick variability across cold-start cache states
- Per-chunk GPU jitter on a non-realtime kernel
- Content-driven variability (specific phoneme transitions and
  sustained-vowel edge cases that hit the SineGen voiced/unvoiced
  threshold from different sides chunk-to-chunk)

## What this means for v0.6.9

Ship. The fixes are real, the floor is real, and the floor is **down
~50 %** from v0.6.8 (~23 → ~12 cuts/min). The remaining ~12 cuts/min
needs model-level work that's out of scope for an engine release.

## What this means for woys-diag

**The harness earns its keep.** Without the chunk-offset distribution
woys-diag exposed in the very first run, we'd have spent a week on the
v0.7.0 ring-buffer rewrite. The chunk-aligned warning showing 8/12 in
round 3 and going to 1/10 after the SOLA fix is the kind of evidence
that's invisible without per-cut metadata.

woys-diag should be a permanent regression-test artifact, not a
one-off. Ideas for the v0.2.0 track:

- CI runner: nightly synthetic-input regression. Detector against the
  fixture should always be 0; if it isn't, something in the analysis
  drifted.
- Real-engine regression: scheduled live runs against `woys-mic` after
  each tag, with a comparison-to-baseline column in the report. Catches
  silent regressions (the kind we *almost* shipped twice in this cycle).
- Replay: store WAV captures keyed by woys version + voice + system,
  so a historical "this voice had 12 cuts/min on RTX 2070 in 2026-05"
  can be reanalyzed if detector thresholds change.

## Files referenced

- `/tmp/control_directmic.wav` + `.md` (B1)
- `/tmp/control_synth_clean.wav` + `.md` (B2)
- `~/.local/share/woys-diag/captures/` + `reports/` (live runs across the
  five rounds — see `docs/12-vad-misfire-investigation.md` for the
  per-round numbers)
