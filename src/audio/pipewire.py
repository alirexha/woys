"""PipeWire integration: persistent virtual mic via `pactl` modules.

Architecture (matches Q6 — persistent vcclient-mic):
  Boot/login → systemd user unit `woys-mic.service` runs
  `woys pw setup`, which loads two PipeWire modules:
    1. module-null-sink  →  WoysSink (apps see it as an audio output)
    2. module-remap-source master=WoysSink.monitor → vcclient-mic
       (Discord/CS2 see it as a microphone)

  The engine writes transformed audio to WoysSink; everyone listening
  on vcclient-mic hears it. The modules persist across engine start/stop —
  Discord can lock onto vcclient-mic once and forget about it.

We do NOT use `object.linger=true`: pactl-loaded modules already persist past
the loading client's lifetime (modules are server-side state). Linger would
only leave orphan PipeWire *nodes* after `unload-module`. `_destroy_orphan_nodes()`
is provided as a recovery hook in case a prior install used linger=true.

This module shells out to `pactl` (and optionally `pw-cli` for orphan cleanup);
no native libpipewire bindings required (pw-python is unmaintained).
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

# Names that survive across runs and uninstalls.
SINK_NAME = "WoysSink"
SOURCE_NAME = "vcclient-mic"
# pactl prop strings: replace spaces with `\_` (pactl's escape) so the value
# survives shell-style tokenization inside pipewire-pulse's argument parser.
SINK_DESC = "woys_(sink)"
SOURCE_DESC = "vcclient-mic_(woys)"
_SINK_DESC_ESCAPED = SINK_DESC
_SOURCE_DESC_ESCAPED = SOURCE_DESC

PACTL_TIMEOUT_S = 5.0


class PipeWireError(RuntimeError):
    """Raised when PipeWire / pipewire-pulse is unusable on this host."""


@dataclass(frozen=True)
class VirtualMicState:
    sink_present: bool
    source_present: bool
    sink_module_id: int | None
    source_module_id: int | None

    @property
    def fully_present(self) -> bool:
        return self.sink_present and self.source_present


def _run_pactl(args: list[str]) -> subprocess.CompletedProcess[str]:
    pactl = shutil.which("pactl")
    if pactl is None:
        raise PipeWireError("pactl not found — install `pipewire-pulse`")
    return subprocess.run(
        [pactl, *args],
        capture_output=True,
        text=True,
        timeout=PACTL_TIMEOUT_S,
        check=False,
    )


def ensure_pipewire() -> None:
    """Hard-fail with a clear error if the host isn't running PipeWire."""
    out = _run_pactl(["info"])
    if out.returncode != 0:
        raise PipeWireError(
            f"pactl info failed — is the audio daemon running?\n  stderr: {out.stderr.strip()}"
        )
    if "PipeWire" not in out.stdout:
        raise PipeWireError(
            "woys requires PipeWire (with pipewire-pulse).\n"
            f"  Detected: {out.stdout.splitlines()[0] if out.stdout else 'unknown'}\n"
            "  On Arch/CachyOS: `paru -S pipewire pipewire-pulse pipewire-alsa`"
        )


def _list_modules() -> list[tuple[int, str, str]]:
    """Return [(module_id, name, args_str), ...]."""
    out = _run_pactl(["list", "short", "modules"])
    if out.returncode != 0:
        raise PipeWireError(f"pactl list short modules failed: {out.stderr.strip()}")
    rows: list[tuple[int, str, str]] = []
    for line in out.stdout.splitlines():
        parts = line.split("\t", 2)
        if len(parts) < 2:
            continue
        try:
            mod_id = int(parts[0])
        except ValueError:
            continue
        name = parts[1].strip()
        args = parts[2].strip() if len(parts) > 2 else ""
        rows.append((mod_id, name, args))
    return rows


def _destroy_orphan_nodes() -> None:
    """Remove any PipeWire nodes named SINK_NAME or SOURCE_NAME via pw-cli.

    Used to recover from a `linger=true` shutdown that left nodes alive after
    the owning module was unloaded. No-op if pw-cli is missing or finds none.
    """
    pw_cli = shutil.which("pw-cli")
    if pw_cli is None:
        return
    try:
        out = subprocess.run(
            [pw_cli, "ls", "Node"],
            capture_output=True,
            text=True,
            timeout=PACTL_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return
    if out.returncode != 0:
        return

    targets = {SINK_NAME, SOURCE_NAME}
    current_id: str | None = None
    ids_to_destroy: list[str] = []
    for line in out.stdout.splitlines():
        line = line.rstrip()
        if line.lstrip().startswith("id ") and ", type" in line:
            tok = line.lstrip().split(",", 1)[0]  # "id 89"
            current_id = tok.split()[1]
        elif "node.name = " in line and current_id is not None:
            name = line.split('"')[1] if '"' in line else ""
            if name in targets:
                ids_to_destroy.append(current_id)

    for nid in ids_to_destroy:
        subprocess.run(
            [pw_cli, "destroy", nid],
            capture_output=True,
            text=True,
            timeout=PACTL_TIMEOUT_S,
            check=False,
        )


def get_state() -> VirtualMicState:
    """Detect whether our sink and source are currently loaded."""
    sink_id: int | None = None
    source_id: int | None = None
    for mod_id, name, args in _list_modules():
        if name == "module-null-sink" and f"sink_name={SINK_NAME}" in args:
            sink_id = mod_id
        elif name == "module-remap-source" and f"source_name={SOURCE_NAME}" in args:
            source_id = mod_id
    return VirtualMicState(
        sink_present=sink_id is not None,
        source_present=source_id is not None,
        sink_module_id=sink_id,
        source_module_id=source_id,
    )


@dataclass
class VirtualMic:
    """Idempotent setup/teardown of the vcclient-mic node pair.

    `linger` defaults to False because PipeWire modules persist across pactl
    client disconnects natively — modules are server-side state, not client
    state. Setting `object.linger=true` would leave orphan PipeWire *nodes*
    after `unload-module`, which we'd then need pw-cli to destroy by ID.
    """

    rate: int = 48_000
    channels: int = 2
    linger: bool = False

    def ensure(self) -> VirtualMicState:
        """Load both modules if they aren't already present.

        Also clears orphan nodes (left by a prior `linger=true` run, or by
        a borked pactl invocation) before loading.
        """
        ensure_pipewire()
        state = get_state()
        if state.fully_present:
            return state

        # Clean any orphan nodes from a previous run before reloading.
        _destroy_orphan_nodes()

        if not state.sink_present:
            self._load_sink()
        if not state.source_present:
            self._load_source()
        return get_state()

    def teardown(self) -> None:
        """Unload both modules. Idempotent — silent if nothing to do."""
        state = get_state()
        # Unload source first; it depends on the sink's monitor.
        if state.source_module_id is not None:
            _run_pactl(["unload-module", str(state.source_module_id)])
        if state.sink_module_id is not None:
            _run_pactl(["unload-module", str(state.sink_module_id)])
        # Sweep orphans (defensive — should be a no-op when linger=False).
        _destroy_orphan_nodes()

    # ---- internals ----------------------------------------------------------

    def _common_props(self) -> str:
        return "object.linger=true" if self.linger else "object.linger=false"

    def _load_sink(self) -> int:
        args = [
            "load-module",
            "module-null-sink",
            "media.class=Audio/Sink",
            f"sink_name={SINK_NAME}",
            f"sink_properties=device.description={_SINK_DESC_ESCAPED}",
            f"rate={self.rate}",
            f"channels={self.channels}",
            self._common_props(),
        ]
        out = _run_pactl(args)
        if out.returncode != 0:
            raise PipeWireError(
                f"failed to load null-sink: {out.stderr.strip() or out.stdout.strip()}"
            )
        return int(out.stdout.strip())

    def _load_source(self) -> int:
        args = [
            "load-module",
            "module-remap-source",
            f"master={SINK_NAME}.monitor",
            f"source_name={SOURCE_NAME}",
            f"source_properties=device.description={_SOURCE_DESC_ESCAPED}",
            self._common_props(),
        ]
        out = _run_pactl(args)
        if out.returncode != 0:
            raise PipeWireError(
                f"failed to load remap-source: {out.stderr.strip() or out.stdout.strip()}"
            )
        return int(out.stdout.strip())
