"""Regression test for the v0.1.1 routing bug.

The engine in v0.1.0 wrote transformed audio to PortAudio's ALSA-default
device (laptop speakers) instead of WoysSink. v0.1.1 spawns
`pacat --device=WoysSink` directly, which is what this test asserts.

If this test fails, Discord/Telegram/CS2 will receive silence even though
they're correctly pointed at `woys-mic`. That's the user-visible
symptom of the v0.1.0 bug.
"""

from __future__ import annotations

import shutil
import subprocess
import time

import pytest

from audio.engine import EngineConfig, RealtimeEngine
from audio.pipewire import VirtualMic, get_state


@pytest.mark.gpu
@pytest.mark.pipewire
@pytest.mark.slow
def test_engine_writes_to_vcclientcachysink_within_3s() -> None:
    """Within 3 s of engine start, WoysSink must have a sink-input
    owned by `woys`. If not, audio is silently going to the
    system default sink (the v0.1.0 routing bug)."""
    if shutil.which("pacat") is None or shutil.which("pactl") is None:
        pytest.skip("pacat/pactl missing — pipewire-pulse not installed")

    VirtualMic().ensure()
    state = get_state()
    if not state.fully_present:
        pytest.skip("WoysSink + woys-mic not loaded — `woys pw setup`")

    eng = RealtimeEngine(EngineConfig(chunk_seconds=0.5))
    eng.start()
    try:
        deadline = time.perf_counter() + 3.0
        found = False
        while time.perf_counter() < deadline:
            out = subprocess.check_output(["pactl", "list", "sink-inputs"], text=True, timeout=2.0)
            # Walk through each "Sink Input #..." block, check sink name + owner.
            blocks = out.split("Sink Input #")[1:]
            for block in blocks:
                if "woys" in block.lower() and "WoysSink" in subprocess.check_output(
                    ["pactl", "list", "short", "sinks"], text=True, timeout=2.0
                ):
                    # Cross-reference: the block's "Sink:" line points at our sink id.
                    sink_id_line = next(
                        (ln.strip() for ln in block.splitlines() if ln.strip().startswith("Sink:")),
                        None,
                    )
                    if sink_id_line is None:
                        continue
                    target_id = sink_id_line.split(":", 1)[1].strip()
                    short_sinks = subprocess.check_output(
                        ["pactl", "list", "short", "sinks"], text=True, timeout=2.0
                    ).splitlines()
                    target_name = next(
                        (
                            line.split("\t", 2)[1]
                            for line in short_sinks
                            if line.startswith(f"{target_id}\t")
                        ),
                        None,
                    )
                    if target_name == "WoysSink":
                        found = True
                        break
            if found:
                break
            time.sleep(0.2)
        assert found, (
            "engine did not connect to WoysSink within 3 s — audio is "
            "leaking to a different sink (the v0.1.0 routing bug)."
        )
    finally:
        eng.stop()


@pytest.mark.gpu
@pytest.mark.pipewire
@pytest.mark.slow
def test_engine_does_not_open_default_alsa_output_when_monitor_off() -> None:
    """With monitor=False (default), the engine must NOT also open a stream
    against the host's default output. v0.1.0 leaked there; v0.1.1 opens
    the monitor stream only when the user opts in."""
    VirtualMic().ensure()
    state = get_state()
    if not state.fully_present:
        pytest.skip("WoysSink + woys-mic not loaded")

    eng = RealtimeEngine(EngineConfig(chunk_seconds=0.5, monitor=False))
    eng.start()
    try:
        time.sleep(2.0)
        out = subprocess.check_output(["pactl", "list", "sink-inputs"], text=True, timeout=2.0)
        blocks = out.split("Sink Input #")[1:]
        leaked: list[str] = []
        short_sinks = subprocess.check_output(
            ["pactl", "list", "short", "sinks"], text=True, timeout=2.0
        ).splitlines()
        for block in blocks:
            if "woys" not in block.lower():
                continue
            sink_id_line = next(
                (ln.strip() for ln in block.splitlines() if ln.strip().startswith("Sink:")),
                None,
            )
            if sink_id_line is None:
                continue
            target_id = sink_id_line.split(":", 1)[1].strip()
            target_name = next(
                (
                    line.split("\t", 2)[1]
                    for line in short_sinks
                    if line.startswith(f"{target_id}\t")
                ),
                "<unknown>",
            )
            if target_name != "WoysSink":
                leaked.append(f"sink={target_id} name={target_name!r}")
        assert not leaked, (
            f"engine opened sink-input(s) on non-WoysSink targets with monitor=False: {leaked}"
        )
    finally:
        eng.stop()
