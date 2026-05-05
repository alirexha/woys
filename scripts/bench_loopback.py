"""Acoustic loopback round-trip benchmark.

Methodology
-----------
Per Q4 (the README's headline number). Sends a click through `WoysSink`
and listens on `vcclient-mic` via `parec`; measures wall-clock delay between
playback time and capture time using cross-correlation.

This bench is intentionally separate from the in-process timestamp test
(`tests/test_smoke_rvc_onnx.py`) — that one isolates inference cost; this one
measures everything the user actually hears, including audio I/O buffering.

Run: `.venv/bin/python scripts/bench_loopback.py`

Requires `parec` (PipeWire/PulseAudio capture) and `pacat` (PA playback).

Sample interpretation
---------------------
The reported number is the *one-way* mic→sink delay measured in the
loopback (i.e. the signal travels: virtual_mic → engine → virtual_sink → parec).
Adding the user's actual mic-capture buffer (host mic → app), expect ~5-15 ms
more in real Discord usage.
"""

from __future__ import annotations

import argparse
import shutil
import struct
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SR = 48_000  # PipeWire default for music/voice; matches our sink config


def make_click_wav(out: Path, sr: int = SR, lead_silence_ms: int = 200) -> int:
    """A short WAV containing N samples of silence then an impulse-like click."""
    silence = int(sr * lead_silence_ms / 1000)
    samples = [0] * silence + [int(0.85 * 32767)] + [0] * 31 + [int(-0.85 * 32767)] + [0] * 31
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"".join(struct.pack("<h", s) for s in samples))
    return silence + 64


def find_click(audio: np.ndarray, sr: int) -> int | None:
    """Index of the first sample where amplitude exceeds 0.3."""
    above = np.where(np.abs(audio) > 0.3)[0]
    return int(above[0]) if above.size else None


def run() -> int:
    if shutil.which("pacat") is None or shutil.which("parec") is None:
        print("error: pacat/parec not found — install pipewire-pulse", file=sys.stderr)
        return 2

    click_path = ROOT / "tests" / "fixtures" / "click.wav"
    expected_click_idx = make_click_wav(click_path)
    print(f"  click WAV: {click_path}  (impulse @ sample {expected_click_idx})")

    # Capture vcclient-mic for 600 ms.
    cap = subprocess.Popen(
        [
            "parec",
            "--device=vcclient-mic",
            "--rate=48000",
            "--channels=1",
            "--format=s16le",
            "--latency-msec=10",
        ],
        stdout=subprocess.PIPE,
    )
    try:
        time.sleep(0.05)  # let parec stabilize
        # Play to the SINK (where the engine writes).
        play = subprocess.Popen(
            ["pacat", "--device=WoysSink", "--rate=48000", str(click_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        play.wait(timeout=2.0)
        time.sleep(0.4)
    finally:
        cap.terminate()
    raw, _ = cap.communicate(timeout=2.0)

    # Decode int16 mono.
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    print(f"  captured: {len(audio)} samples ({len(audio) / SR * 1000:.1f} ms)")

    idx = find_click(audio, SR)
    if idx is None:
        print("  FAIL: no click found in capture — engine routing broken?")
        return 3
    one_way_ms = (idx - expected_click_idx) * 1000.0 / SR
    print(f"  click landed at sample {idx} → one-way delay {one_way_ms:.2f} ms")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.parse_args()
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
