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

    # v0.6.6 - snapshot the host's pre-test state so we can restore it.
    # Before this guard, running pytest on a real desktop wiped the user's
    # active virtual mic - the test's final teardown left them without
    # `woys-mic` and Discord / CS2 lost their input device until the next
    # `systemctl --user restart woys-mic.service`. Tests must not bite
    # the hand that runs them.
    pretest_state = get_state()
    try:
        # Make sure we start clean - earlier runs / orphans get cleared.
        vm.teardown()
        initial = get_state()
        assert not initial.fully_present

        state = vm.ensure()
        try:
            assert state.fully_present, f"ensure didn't load both modules: {state}"
            assert state.sink_module_id is not None
            assert state.source_module_id is not None

            # Verify Discord/CS2-visible names appear in pactl listings.
            # v0.13.1 - match exact tab-separated name. Substring match
            # was brittle once v0.13.0 added a `woys-mic-clean` source
            # whose name contains `woys-mic` (the v0.6.5 / v0.12.x
            # virtual-mic name).
            sources = _pactl_lists("sources")
            sinks = _pactl_lists("sinks")

            def _has_named(rows: list[str], name: str) -> bool:
                for row in rows:
                    parts = row.split("\t")
                    if len(parts) >= 2 and parts[1].strip() == name:
                        return True
                return False

            assert _has_named(sources, SOURCE_NAME), sources
            assert _has_named(sinks, SINK_NAME), sinks

            # Idempotent - second ensure returns same module IDs.
            again = vm.ensure()
            assert again.sink_module_id == state.sink_module_id
            assert again.source_module_id == state.source_module_id
        finally:
            vm.teardown()

        after = get_state()
        assert not after.fully_present, f"teardown left state behind: {after}"
        sources = _pactl_lists("sources")
        sinks = _pactl_lists("sinks")

        def _has_named(rows: list[str], name: str) -> bool:
            for row in rows:
                parts = row.split("\t")
                if len(parts) >= 2 and parts[1].strip() == name:
                    return True
            return False

        assert not _has_named(sources, SOURCE_NAME), f"orphan source after teardown: {sources}"
        assert not _has_named(sinks, SINK_NAME), f"orphan sink after teardown: {sinks}"
    finally:
        # Restore the host's pre-test state - if the user had the mic loaded
        # before the test started, give it back to them.
        if pretest_state.fully_present:
            vm.ensure()


@pytest.mark.pipewire
def test_pipewire_error_on_missing_pactl(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate `pactl` being absent.
    real_which = shutil.which
    monkeypatch.setattr(shutil, "which", lambda name: None if name == "pactl" else real_which(name))
    with pytest.raises(PipeWireError, match="pactl"):
        ensure_pipewire()
