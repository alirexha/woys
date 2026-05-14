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


# ---- review F-merged-006: relabel_source must be atomic --------------
# These are pure unit tests (mocked `_run_pactl` / `_list_modules`); no
# pipewire daemon is touched, so they are NOT marked `pipewire`.


def test_relabel_source_rolls_back_on_reload_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pre-fix `relabel_source` unloaded woys-mic then reloaded with no
    rollback, so a reload failure left woys-mic *destroyed* while
    `chain.setup()` reported success (the v0.14.2 bug class). Post-fix a
    reload failure recreates woys-mic from its captured original args and
    raises a PipeWireError saying so."""
    from audio import pipewire

    old_args = (
        "master=WoysSink.monitor source_name=woys-mic "
        "source_properties=device.description=woys-no-cleanup object.linger=false"
    )
    monkeypatch.setattr(pipewire, "_list_modules", lambda: [(42, "module-remap-source", old_args)])
    load_calls: list[list[str]] = []

    def fake_run_pactl(args: list[str]) -> subprocess.CompletedProcess[str]:
        if args[0] == "unload-module":
            return subprocess.CompletedProcess(args, 0, "", "")
        if args[0] == "load-module":
            load_calls.append(args)
            # 1st load-module = the relabel attempt -> fail;
            # 2nd = the rollback -> succeed.
            if len(load_calls) == 1:
                return subprocess.CompletedProcess(args, 1, "", "module load failed")
            return subprocess.CompletedProcess(args, 0, "43", "")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(pipewire, "_run_pactl", fake_run_pactl)

    with pytest.raises(PipeWireError, match="rolled back"):
        pipewire.relabel_source("_internal-raw-bypass", passive=True)

    assert len(load_calls) == 2, "a rollback load-module must follow the failed relabel"
    assert any("woys-no-cleanup" in tok for tok in load_calls[1]), (
        "rollback must recreate woys-mic from its original (captured) args"
    )


def test_relabel_source_succeeds_when_reload_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    """Happy path: a successful reload returns without raising and without
    issuing a rollback load-module."""
    from audio import pipewire

    old_args = (
        "master=WoysSink.monitor source_name=woys-mic source_properties=x object.linger=false"
    )
    monkeypatch.setattr(pipewire, "_list_modules", lambda: [(7, "module-remap-source", old_args)])
    load_calls: list[list[str]] = []

    def fake_run_pactl(args: list[str]) -> subprocess.CompletedProcess[str]:
        if args[0] == "load-module":
            load_calls.append(args)
        return subprocess.CompletedProcess(args, 0, "8", "")

    monkeypatch.setattr(pipewire, "_run_pactl", fake_run_pactl)
    pipewire.relabel_source("woys-no-cleanup", passive=False)  # must not raise
    assert len(load_calls) == 1, "happy path must not issue a rollback load-module"


# ---- review F-08-06 (2nd half): default-sink hijack probe ------------


def test_engine_warns_when_default_sink_is_woys_sink(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the system default sink is the woys sink the engine writes into,
    desktop audio is hijacked (the v0.14.2 class). `start()`'s probe must
    record a `last_error` warning -- not hard-fail, the engine still
    converts voice fine."""
    from audio import engine

    eng = engine.RealtimeEngine(engine.EngineConfig())
    monkeypatch.setattr("audio.pipewire.get_default_sink", lambda: eng.cfg.sink_name)

    eng._warn_if_default_sink_hijacked()

    assert eng.stats.last_error, "a hijacked default sink must set last_error"
    assert eng.cfg.sink_name in eng.stats.last_error


def test_engine_no_warning_when_default_sink_is_a_real_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A normal default sink (real speakers) must not trip the probe."""
    from audio import engine

    eng = engine.RealtimeEngine(engine.EngineConfig())
    monkeypatch.setattr(
        "audio.pipewire.get_default_sink", lambda: "alsa_output.pci-0000_00_1f.3.analog-stereo"
    )

    eng._warn_if_default_sink_hijacked()

    assert not eng.stats.last_error
