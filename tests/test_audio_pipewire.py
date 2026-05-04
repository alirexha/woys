"""PipeWire VirtualMic round-trip + idempotency.

Marked `pipewire` so the harness skips on CI / hosts without a running
pipewire-pulse daemon.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from audio.pipewire import (
    SINK_NAME,
    SOURCE_NAME,
    PipeWireError,
    VirtualMic,
    ensure_pipewire,
    get_state,
)


def _pactl_lists(short: str) -> list[str]:
    return subprocess.check_output(
        ["pactl", "list", "short", short], text=True, timeout=3
    ).splitlines()


def test_ensure_pipewire_passes_on_host() -> None:
    # The conftest already skips this whole file on non-PW hosts via the
    # `pipewire` marker, but ensure_pipewire() should also pass directly.
    ensure_pipewire()


@pytest.mark.pipewire
def test_virtual_mic_round_trip() -> None:
    vm = VirtualMic()
    # Make sure we start clean — earlier runs / orphans get cleared.
    vm.teardown()
    initial = get_state()
    assert not initial.fully_present

    state = vm.ensure()
    try:
        assert state.fully_present, f"ensure didn't load both modules: {state}"
        assert state.sink_module_id is not None
        assert state.source_module_id is not None

        # Verify Discord/CS2-visible names appear in pactl listings.
        sources = _pactl_lists("sources")
        sinks = _pactl_lists("sinks")
        assert any(SOURCE_NAME in line for line in sources), sources
        assert any(SINK_NAME in line for line in sinks), sinks

        # Idempotent — second ensure returns same module IDs.
        again = vm.ensure()
        assert again.sink_module_id == state.sink_module_id
        assert again.source_module_id == state.source_module_id
    finally:
        vm.teardown()

    after = get_state()
    assert not after.fully_present, f"teardown left state behind: {after}"
    sources = _pactl_lists("sources")
    sinks = _pactl_lists("sinks")
    assert not any(SOURCE_NAME in line for line in sources), (
        f"orphan source after teardown: {sources}"
    )
    assert not any(SINK_NAME in line for line in sinks), f"orphan sink after teardown: {sinks}"


@pytest.mark.pipewire
def test_pipewire_error_on_missing_pactl(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate `pactl` being absent.
    real_which = shutil.which
    monkeypatch.setattr(shutil, "which", lambda name: None if name == "pactl" else real_which(name))
    with pytest.raises(PipeWireError, match="pactl"):
        ensure_pipewire()
