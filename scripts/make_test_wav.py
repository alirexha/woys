"""Generate a deterministic 1-second test WAV for the smoke test.

Synthesizes a short voiced segment (sum of sinusoids around 200 Hz with
mild AM) at 16 kHz mono int16. Saved to tests/fixtures/sine_voiced_1s.wav.

This is purposely synthetic: no real-voice rights to worry about, deterministic
across machines, and exercises the RVC f0 detector enough that `extract`
returns non-zero pitch values.
"""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "sine_voiced_1s.wav"
SR = 16_000
DURATION_S = 1.0


def main() -> int:
    n = int(SR * DURATION_S)
    samples: list[int] = []
    f0 = 200.0  # Hz - comfortable male/female pitch
    for i in range(n):
        t = i / SR
        # voiced: f0 + 2*f0 + 3*f0 with mild amplitude modulation
        v = (
            0.6 * math.sin(2 * math.pi * f0 * t)
            + 0.25 * math.sin(2 * math.pi * 2 * f0 * t)
            + 0.10 * math.sin(2 * math.pi * 3 * f0 * t)
        )
        v *= 0.7 + 0.3 * math.sin(2 * math.pi * 4.0 * t)  # slow AM
        samples.append(int(max(-1.0, min(1.0, v)) * 0.6 * 32767))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(OUT), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(b"".join(struct.pack("<h", s) for s in samples))
    sz = OUT.stat().st_size
    print(f"wrote {OUT} ({sz} bytes, {DURATION_S}s @ {SR} Hz, mono int16)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
