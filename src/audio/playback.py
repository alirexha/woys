"""The playback side of the engine: spawning the player process
(`woys-pw-out`, `pw-cat` or `pacat`), feeding it through the writer thread,
draining its stderr, and the watchdog that respawns it.

`_PlaybackMixin` holds these `RealtimeEngine` methods; the engine's
`__init__` creates the state they share (declared below for type checking).
Split out of `audio.engine`, which re-exports `_set_pdeathsig`.
"""

from __future__ import annotations

import contextlib
import ctypes
import os
import queue
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from audio.engine_config import EngineConfig
from audio.engine_stats import EngineStats

NDArrayF32 = npt.NDArray[np.float32]


# cap on consecutive playback-helper respawns
# that never stay alive. The watchdog used to retry forever (running=True,
# zero audio). 8 consecutive deaths-without-recovery is unambiguously a
# broken binary/config, not a transient PipeWire hiccup (a one-off death
# respawns once and the counter resets on the next healthy tick).
_PLAYER_RESPAWN_CAP = 8

# parent-death signal for playback-helper
# children. Loaded once at import; None off non-glibc-Linux (then the
# preexec_fn below is a no-op).
try:
    _LIBC: ctypes.CDLL | None = ctypes.CDLL("libc.so.6", use_errno=True)
except OSError:  # pragma: no cover - non-Linux / no glibc
    _LIBC = None
_PR_SET_PDEATHSIG = 1  # <sys/prctl.h>


def _set_pdeathsig() -> None:
    """`preexec_fn` for playback-helper spawns: runs in the forked child,
    before exec, and asks the kernel to send SIGTERM to this process if
    its parent (the engine) dies.

    a `kill -9` of the engine used to orphan the
    playback subprocess, still holding its audio stream. The inference
    child already self-protects via a `getppid()` poll; the playback
    helpers did not. Must stay lock-free (one syscall, no imports, no
    allocation) -- `preexec_fn` runs between fork and exec in a
    multithreaded process.
    """
    if _LIBC is not None:
        # The bare try/except (vs `contextlib.suppress`) is deliberate:
        # `suppress` instantiates an object (a malloc); a bare try/except
        # is allocation-free, which matters in a preexec_fn running
        # between fork and exec.
        try:  # noqa: SIM105
            _LIBC.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM)
        except Exception:
            pass


class _PlaybackMixin:
    """Player-process methods of `RealtimeEngine`."""

    cfg: EngineConfig
    stats: EngineStats
    _stats_lock: Any
    _stop_event: threading.Event
    _pacat_proc: subprocess.Popen[bytes] | None
    _pacat_lock: threading.Lock
    _pacat_dead_event: threading.Event
    _player_backend: str
    _writer_queue: queue.Queue[bytes] | None
    _stderr_thread: threading.Thread | None
    _last_writer_ts: float | None

    if TYPE_CHECKING:

        def record_error(self, msg: str) -> None: ...

        def _self_abort(self, msg: str) -> None: ...

        def _apply_thread_priority(self, *, label: str, priority: int = 60) -> None: ...

    def _warn_if_default_sink_hijacked(self) -> None:
        """one-shot check at engine start.

        If the system default sink is the woys sink the engine writes into,
        all *desktop* audio is being routed into woys plumbing instead of
        the speakers (the v0.14.2 hijack). That is a system-routing problem,
        not an engine fault -- the engine still converts voice fine -- so
        this WARNS (records `stats.last_error`, surfaced by `woys diag` and
        the TUI) rather than refusing to start.

        Best-effort: `get_default_sink()` returns '' on any pactl error, so
        a probe failure is a silent no-op here -- the load-bearing sink
        check is `_assert_sink_loaded`, which hard-fails.
        """
        from audio.pipewire import get_default_sink

        default_sink = get_default_sink()
        if default_sink and default_sink == self.cfg.sink_name:
            self.record_error(
                f"system default sink is {default_sink!r} -- the woys sink the "
                f"engine writes into. Desktop audio is being routed into woys "
                f"plumbing; run `pactl set-default-sink <your-speakers>` to fix."
            )

    def _assert_sink_loaded(self) -> None:
        """v0.6.4 - refuse to start if `cfg.sink_name` isn't a loaded
        PipeWire sink.

        Without this guard, `pw-cat --target=...` and `pacat --device=...`
        treat the named sink as a hint: if it's missing, the session
        manager silently routes the stream to the *default* sink
        (typically laptop speakers). The engine's playback subprocess
        starts cleanly, exits 0, no stderr - and your transformed
        voice plays out of the speakers instead of the virtual mic.
        See docs/10-monitor-leak-diag.md for the full forensic trail.

        v0.14.0 (area 9 / area 17 / area 19 / C014): pre-v0.14.0 the
        function silently skipped the check on three error paths
        (FileNotFoundError -> pactl missing; TimeoutExpired -> daemon
        slow; nonzero rc -> pactl itself errored). Skipping re-opens
        the v0.6.4 routing-to-laptop-speakers bug exactly when the
        environment is most likely to be misconfigured. v0.14.0 hard-
        fails on each: clearer than letting the engine start and serve
        silence to the wrong sink. If a user really has no pactl, they
        should set `sink_name=""` (skip-sink-check mode -- not yet
        implemented; flagged for v0.14.x as an explicit opt-out).
        """
        try:
            result = subprocess.run(
                ["pactl", "list", "short", "sinks"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except FileNotFoundError as e:
            raise RuntimeError(
                "pactl is not on PATH; cannot verify PipeWire sink presence. "
                "Install pipewire-pulse (provides pactl) or use a system with "
                "PulseAudio compatibility -- woys requires pactl for sink-load "
                "verification. Refusing to start rather than risk routing audio "
                "to the wrong sink."
            ) from e
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                "pactl `list short sinks` timed out after 5s. PipeWire daemon "
                "may be unresponsive (mid-restart, hung). Refusing to start "
                "rather than skip the sink-load check. Try `systemctl --user "
                "restart pipewire pipewire-pulse wireplumber` and re-run."
            ) from e
        if result.returncode != 0:
            raise RuntimeError(
                f"pactl `list short sinks` exited {result.returncode}. "
                f"stderr: {result.stderr.strip()[:200] or '<empty>'}. "
                f"Refusing to start without verified sink presence."
            )
        loaded = [line.split("\t")[1] for line in result.stdout.splitlines() if "\t" in line]
        if self.cfg.sink_name not in loaded:
            raise RuntimeError(
                f"PipeWire sink {self.cfg.sink_name!r} is not loaded - refusing to start.\n"
                f"  loaded sinks: {loaded}\n"
                f"  fix: run `woys pw setup` to load the virtual sink, "
                f"or correct `sink_name` in ~/.config/woys/config.toml."
            )

    def _find_native_pw_helper(self) -> Path | None:
        """Locate `woys-pw-out` (the native PipeWire helper introduced in
        v0.9.0). Search order:
          1. $PATH (via `shutil.which`)
          2. <repo>/bin/woys-pw-out (dev checkout, makes `make install`
             optional)
          3. ~/.local/bin/woys-pw-out (default install prefix)

        Returns None if none of those resolve.
        """
        # 1. PATH.
        path_hit = shutil.which("woys-pw-out")
        if path_hit:
            return Path(path_hit)
        # 2. Repo's bin/.
        repo_root = Path(__file__).resolve().parent.parent.parent
        repo_hit = repo_root / "bin" / "woys-pw-out"
        if repo_hit.exists() and os.access(repo_hit, os.X_OK):
            return repo_hit
        # 3. Default install prefix.
        local_hit = Path.home() / ".local" / "bin" / "woys-pw-out"
        if local_hit.exists() and os.access(local_hit, os.X_OK):
            return local_hit
        return None

    def _spawn_checked(self, cmd: list[str]) -> subprocess.Popen[bytes]:
        """Spawn a playback helper and verify it did not die on startup.

        `_open_pacat` used to `return
        subprocess.Popen(...)` with no liveness check. A helper that exits
        immediately (bad args, missing perms, a built-but-broken
        `woys-pw-out`) left the watchdog in an infinite 0.5 s-backoff
        respawn loop with `running=True` and `chunks_processed` climbing
        while zero audio reached the sink. We now sleep briefly and
        `poll()`; an immediate exit raises with the helper's stderr so the
        initial spawn fails the engine loudly and the watchdog's
        consecutive-failure cap can act.

        `preexec_fn=_set_pdeathsig` arms the
        kernel parent-death signal, so a `kill -9` of the engine SIGTERMs
        the playback helper instead of orphaning it with the audio stream
        still open. One change covers all three spawn paths (native-pw /
        pw-cat / pacat) -- they all funnel through here.
        """
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            preexec_fn=_set_pdeathsig,  # lock-free single syscall -- see _set_pdeathsig
        )
        time.sleep(0.05)
        rc = proc.poll()
        if rc is not None:
            stderr = ""
            if proc.stderr is not None:
                with contextlib.suppress(Exception):
                    stderr = proc.stderr.read().decode("utf-8", "replace").strip()
            raise RuntimeError(
                f"playback helper {cmd[0]!r} exited immediately on spawn (rc={rc})"
                + (f": {stderr[:300]}" if stderr else "")
            )
        return proc

    def _open_pacat(self) -> subprocess.Popen[bytes]:
        """Spawn the playback subprocess targeting the named virtual sink.

        v0.5.2: prefers `pw-cat` (PipeWire-native, no underruns under
        bursty 250 ms writes) over `pacat` (PulseAudio compat, drains the
        prebuf/tlength buffer near zero on every chunk → underrun storm).
        Falls back to pacat only if pw-cat is missing.

        v0.6.4: pre-flights sink existence - see `_assert_sink_loaded`.
        Without that guard, `--target` / `--device` silently fall back
        to the default sink when the named sink is missing.

        v0.9.0: `cfg.prefer_native_pw` (default True since v0.9.2 — the
        default flipped after the native helper became the daily-driver
        path) selects the native helper `woys-pw-out`
        (see `bin/woys-pw-out.c`) over pw-cat / pacat. The native helper
        decouples the engine's bursty
        chunk writes from PipeWire's per-quantum RT callback via an
        explicit SPSC ring buffer, closing the audit's area-08 cut
        signature (sample-exact zeros at 21.33/42.67 ms quantum
        cadence). NEVER falls back silently - if `prefer_native_pw=True`
        and the helper is missing, we raise so the user sees an actionable
        error instead of cuts they can't explain.

        The retained name `_open_pacat` is historical - the watchdog and
        writer threads don't care which binary is on the other side, only
        that it accepts raw float32le on stdin.
        """
        self._assert_sink_loaded()
        if self.cfg.prefer_native_pw:
            helper = self._find_native_pw_helper()
            if helper is None:
                raise RuntimeError(
                    "prefer_native_pw=True but `woys-pw-out` was not found. "
                    "Build it with `make -C bin/` from the repo root, then "
                    "either symlink it onto $PATH or run "
                    "`make -C bin/ install` to drop it into ~/.local/bin/. "
                    "Set prefer_native_pw=false in your config to fall back "
                    "to the legacy pw-cat / pacat path."
                )
            self._player_backend = "native-pw"
            # v0.9.1: compute ring frames from prefer_native_pw_buffer_ms.
            # Helper requires power-of-2 ring size (SPSC mask trick), so
            # round up. Need:
            #   chunk_frames + slack_frames
            # where chunk_frames is one engine write (chunk_seconds *
            # sink_rate) and slack_frames absorbs writer-jitter overshoot.
            chunk_frames = int(self.cfg.chunk_seconds * self.cfg.sink_rate)
            slack_frames = int(self.cfg.prefer_native_pw_buffer_ms * self.cfg.sink_rate / 1000)
            needed = chunk_frames + slack_frames
            ring_frames = 1
            while ring_frames < needed:
                ring_frames <<= 1
            # Helper caps ring at 32768 internally as a sanity limit; cap
            # here too with a clear error rather than letting the helper
            # reject the arg later.
            if ring_frames > 32768:
                raise RuntimeError(
                    f"prefer_native_pw_buffer_ms={self.cfg.prefer_native_pw_buffer_ms} "
                    f"computes ring_frames={ring_frames}, above the helper's 32768 cap. "
                    f"Lower the buffer or accept that no realistic engine jitter "
                    f"requires more than 32768/{self.cfg.sink_rate} ≈ "
                    f"{32768 / self.cfg.sink_rate * 1000:.0f} ms."
                )
            cmd = [
                str(helper),
                f"--target={self.cfg.sink_name}",
                f"--rate={self.cfg.sink_rate}",
                f"--channels={self.cfg.output_channels}",
                "--quantum=1024",
                f"--ring-frames={ring_frames}",
            ]
            return self._spawn_checked(cmd)

        if self.cfg.prefer_pw_cat:
            pw_cat = shutil.which("pw-cat")
            if pw_cat is not None:
                self._player_backend = "pw-cat"
                cmd = [
                    pw_cat,
                    "--playback",
                    f"--target={self.cfg.sink_name}",
                    f"--rate={self.cfg.sink_rate}",
                    f"--channels={self.cfg.output_channels}",
                    "--format=f32",
                    "--raw",
                    f"--latency={self.cfg.output_latency_ms}ms",
                    "-",
                ]
                return self._spawn_checked(cmd)

        pacat = shutil.which("pacat")
        if pacat is None:
            raise RuntimeError(
                "neither pw-cat nor pacat found - install pipewire and pipewire-pulse"
            )
        self._player_backend = "pacat"
        cmd = [
            pacat,
            "--playback",
            f"--device={self.cfg.sink_name}",
            f"--rate={self.cfg.sink_rate}",
            f"--channels={self.cfg.output_channels}",
            "--format=float32le",
            f"--latency-msec={self.cfg.output_latency_ms}",
            f"--process-time-msec={self.cfg.output_process_time_ms}",
            "--client-name=woys",
            "--stream-name=engine-out",
            "--raw",
            "-v",
        ]
        return self._spawn_checked(cmd)

    def _to_sink_bytes(self, mono: NDArrayF32) -> bytes:
        """Convert a mono float32 chunk at sink_rate into the byte payload
        pacat expects on stdin. With output_channels=2, interleave L=R=mono
        so PipeWire doesn't have to upmix on every chunk.
        """
        if self.cfg.output_channels == 1:
            return mono.tobytes()
        # Stereo: interleave mono into [L0, R0, L1, R1, ...]. np.repeat is
        # the cheapest path: ~50 µs for 250 ms of 48 kHz audio on this CPU,
        # well below the chunk budget.
        stereo = np.repeat(mono.astype(np.float32, copy=False), self.cfg.output_channels)
        return stereo.tobytes()

    def _enqueue_chunk(self, payload: bytes) -> None:
        """Hand a write-ready byte payload to the writer thread. On a full
        queue the engine has out-paced the writer/sink - bump the
        queue_full counter (xrun proxy) and drop the chunk rather than
        block the engine main loop.
        """
        q = self._writer_queue
        if q is None:
            return
        try:
            q.put_nowait(payload)
        except queue.Full:
            with self._stats_lock:
                self.stats.queue_full_events += 1

    def _writer_loop(self) -> None:
        """Daemon thread: drains _writer_queue into pacat.stdin.

        Decouples the engine main loop from blocking pipe writes (Brief §3
        Fix 2). On BrokenPipeError / OSError the watchdog is signalled to
        respawn pacat; the writer keeps running and reattaches to the new
        handle on the next iteration.
        """
        # Best-effort thread-local affinity so the writer doesn't ping-pong
        # cores away from the main engine thread.
        # B19 / perf-009: writer at priority 59 (engine at 60). Both stay
        # SCHED_FIFO so SCHED_OTHER background work can't starve either,
        # but the engine wins same-class tie-breaks during contention.
        self._apply_thread_priority(label="writer", priority=59)
        while not self._stop_event.is_set():
            q = self._writer_queue
            if q is None:
                # The session that owned this writer has torn down its
                # queue; polling None would spin a core until stop().
                break
            try:
                payload = q.get(timeout=0.1)
            except queue.Empty:
                continue
            if payload is None:
                continue
            with self._pacat_lock:
                proc = self._pacat_proc
            if proc is None or proc.stdin is None:
                # Watchdog hasn't (re)spawned pacat yet; drop and continue.
                continue
            try:
                proc.stdin.write(payload)
                proc.stdin.flush()
            except (BrokenPipeError, OSError) as e:
                # v0.9.0-rc5: distinguish shutdown-race BrokenPipe from
                # a real mid-session helper death. During engine.stop(),
                # the playback subprocess is terminated as part of the
                # finally-block; the writer thread may still have queued
                # bytes and races the helper's exit. That race is normal
                # teardown noise, not a runtime error worth surfacing.
                if self._stop_event.is_set():
                    return
                self.record_error(
                    f"{self._player_backend or 'player'} write failed "
                    f"({type(e).__name__}); respawning"
                )
                self._pacat_dead_event.set()
                # Brief pause so the watchdog has time to respawn before
                # the next iteration tries to write again.
                time.sleep(0.02)
                continue
            now = time.perf_counter()
            if self._last_writer_ts is not None:
                interval_ms = (now - self._last_writer_ts) * 1000.0
                # the append +
                # the np.array(deque) snapshot below both go through
                # `_stats_lock`. Pre-fix the writer thread could die on
                # `RuntimeError: deque mutated during iteration` if a
                # cross-thread `list(deque)` ran via the woys-diag /
                # TUI poll path. The dying thread is the writer
                # itself, which means audio stops silently -- the most
                # serious of the three F-merged-017 sub-bugs.
                snapshot: list[float] | None = None
                with self._stats_lock:
                    self.stats._writer_intervals_ms.append(interval_ms)
                    if len(self.stats._writer_intervals_ms) >= 16:
                        # B27 / corr-008: refresh jitter every chunk
                        # once the deque is full. Cost is ~10 us / chunk
                        # on a 128-deque; trivial.
                        snapshot = list(self.stats._writer_intervals_ms)
                if snapshot is not None:
                    arr = np.array(snapshot, dtype=np.float32)
                    self.stats.writer_jitter_ms = float(arr.std())
            self._last_writer_ts = now

    def _stderr_reader_loop(self, proc: subprocess.Popen[bytes]) -> None:
        """Daemon thread: parses pacat -v stderr for underrun tokens.

        Bound to a single pacat process - when it exits, readline returns
        b'' and the thread terminates. The watchdog spawns a new reader
        for the replacement process.

        B32 / corr-018: in pw-cat mode, parsing for "underrun" is futile
        (pw-cat doesn't emit that token); instead we just drain the pipe
        so it doesn't fill the kernel buffer (~64 KB) and deadlock the
        subprocess. The xruns counter stays 0 in pw-cat mode (already
        documented in `woys diag`).
        """
        if proc.stderr is None:
            return
        is_pacat = self._player_backend == "pacat"
        is_native = self._player_backend == "native-pw"
        # Diagnostic tee: when WOYS_HELPER_STDERR_LOG is set, every line
        # the player backend writes to stderr is also appended to that
        # path with a wall-clock timestamp. Zero overhead when the env
        # var is unset. Useful for forensic post-mortems of "the helper
        # died at some point during a session" cases - we lose nothing
        # to the existing parse-and-overwrite pattern.
        # v0.14.0 (area 6 / area 12 / C021): the env-driven path is
        # security-sensitive. Pre-v0.14.0 it called `open(path, "ab")`
        # with no symlink protection; an attacker who controls the path
        # value could swap a symlink to a victim file (e.g. ~/.bashrc)
        # between checks and corrupt it via "ab" append. Mitigations:
        #   1. Require absolute path (relative paths are caller-confused).
        #   2. Use os.open with O_NOFOLLOW so symlinks at the final
        #      component are refused (we open the actual file, not its
        #      symlink target).
        #   3. Skip silently with a `last_error` warning if the open
        #      fails for any reason -- the diagnostic is opt-in, not
        #      load-bearing.
        debug_log_path = os.environ.get("WOYS_HELPER_STDERR_LOG")
        debug_fp = None
        if debug_log_path:
            try:
                if not os.path.isabs(debug_log_path):
                    raise ValueError(
                        f"WOYS_HELPER_STDERR_LOG must be an absolute path, got {debug_log_path!r}"
                    )
                # refuse paths
                # outside `$XDG_RUNTIME_DIR/woys/` (or the secure
                # `/tmp/woys-{uid}/` fallback). Pre-fix a user
                # innocently setting this to `/tmp/woys-helper.log`
                # opened a symlink-attackable path in a world-
                # traversable directory. The O_NOFOLLOW below still
                # guards the open call, but constraining the
                # location keeps an attacker from positioning a
                # symlink mid-flight in the first place.
                from woys.xdg import safe_runtime_dir

                runtime_dir = safe_runtime_dir()
                resolved = os.path.realpath(debug_log_path)
                if not resolved.startswith(str(runtime_dir.resolve()) + os.sep):
                    raise ValueError(
                        f"WOYS_HELPER_STDERR_LOG must live under {runtime_dir} "
                        f"(refusing {debug_log_path!r}; "
                        f"a path outside the user-private runtime dir is "
                        f"symlink-attackable -- F-05-11)"
                    )
                fd = os.open(
                    debug_log_path,
                    os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW,
                    0o600,
                )
                debug_fp = os.fdopen(fd, "ab", buffering=0)
            except (OSError, ValueError) as e:
                with self._stats_lock:
                    self.stats.priority_warnings.append(
                        f"WOYS_HELPER_STDERR_LOG disabled: {type(e).__name__}: {e}"
                    )
                debug_fp = None
        try:
            for raw in proc.stderr:
                if not raw:
                    break
                if debug_fp is not None:
                    ts = time.strftime("%H:%M:%S", time.localtime())
                    with contextlib.suppress(OSError):
                        debug_fp.write(f"[{ts} {self._player_backend}] ".encode() + raw)
                line = raw.decode("utf-8", errors="replace")
                if is_pacat:
                    # pacat -v prints lines like "Stream underrun.\n" exactly.
                    # We match case-insensitively in case the wording shifts
                    # across PulseAudio versions.
                    if "underrun" in line.lower():
                        with self._stats_lock:
                            self.stats.xruns += 1
                elif is_native:
                    # v0.9.0 - native helper emits:
                    #   "ready"                      once after STREAMING
                    #   "quantum=N rate=M ..."       once after format negotiation
                    #   "underruns=N"                every UNDERRUN_REPORT_SECS
                    #   "error: <msg>"               fatal
                    s = line.strip()
                    if s.startswith("underruns="):
                        try:
                            count = int(s[len("underruns=") :])
                        except ValueError:
                            count = 0
                        self.stats.player_underruns = count
                    elif s.startswith("error:"):
                        # Surface the helper's hard-fail message to woys diag.
                        cause = f"native-pw: {s[len('error:') :].strip()}"
                        self.record_error(cause)
                        # v0.11.0 - also push to helper_exit_reasons so the
                        # watchdog's "respawned" message can't clobber the
                        # cause when the watchdog fires shortly after.
                        with self._stats_lock:
                            self.stats.helper_exit_reasons.append(cause)
                            if len(self.stats.helper_exit_reasons) > 10:
                                self.stats.helper_exit_reasons.pop(0)
                # else: pw-cat or unknown - drain-only.
        except (ValueError, OSError):
            # Pipe closed mid-read during shutdown - expected.
            return
        finally:
            if debug_fp is not None:
                with contextlib.suppress(OSError):
                    debug_fp.close()

    def _watchdog_loop(self) -> None:
        """Daemon thread: respawns pacat if it dies mid-session (Brief §3 Fix 3).

        Polls every `pacat_watchdog_interval_s` (50 ms by default). On dead
        process: opens a replacement under `_pacat_lock`, swaps the handle,
        spawns a fresh stderr reader for the new process, and increments
        `pacat_restarts`. Recovery target ≤ 100 ms.

        the respawn loop is capped. Pre-fix a
        helper that could never stay alive (raised every time, or spawned
        then died immediately) was retried forever with `running=True` and
        zero audio. `consecutive_respawns` counts deaths-without-recovery;
        a healthy tick (`poll() is None`) resets it, so a one-off death
        costs one respawn and does not accumulate. At the cap the engine
        stops cleanly with a definitive `last_error` (mirrors the
        inference circuit breaker).
        """
        consecutive_respawns = 0
        while not self._stop_event.is_set():
            # Wake immediately if the writer signalled BrokenPipe; otherwise
            # poll on the configured interval.
            self._pacat_dead_event.wait(timeout=self.cfg.pacat_watchdog_interval_s)
            self._pacat_dead_event.clear()
            with self._pacat_lock:
                proc = self._pacat_proc
            if proc is None:
                continue
            if proc.poll() is None:
                consecutive_respawns = 0  # alive this tick -> healthy, reset the cap
                continue  # still alive
            # Dead. Count this respawn attempt; a helper that never stays
            # alive must not loop forever.
            consecutive_respawns += 1
            if consecutive_respawns > _PLAYER_RESPAWN_CAP:
                self._self_abort(
                    f"engine stopping: playback helper died and was respawned "
                    f"{consecutive_respawns - 1}x without staying alive. "
                    f"Last cause: {self.stats.last_error or 'unknown'}"
                )
                return
            # Respawn.
            try:
                new_proc = self._open_pacat()
            except Exception as e:
                self.record_error(
                    f"watchdog respawn failed ({consecutive_respawns}x): {type(e).__name__}: {e}"
                )
                # Back off a bit before retrying so we don't spin.
                time.sleep(0.5)
                continue
            # B11 / corr-007: if stop fired while we were opening the new
            # proc (a slow path - _open_pacat takes ~50-200 ms), do NOT
            # install the new handle. Kill it instead so the engine's
            # finally-block teardown sees a stable `_pacat_proc` and the
            # new proc doesn't leak fds.
            if self._stop_event.is_set():
                with contextlib.suppress(Exception):
                    new_proc.terminate()
                    new_proc.wait(timeout=0.5)
                return
            with self._pacat_lock:
                # Discard the dead handle (caller already detected death).
                self._pacat_proc = new_proc
            with self._stats_lock:
                self.stats.player_restarts += 1
            # v0.11.0 - preserve the helper's own death cause if the
            # stderr reader captured one before the exit. If not, log
            # the watchdog's view (exit code + chunk index) so the user
            # can correlate. Either way, append to helper_exit_reasons
            # rather than clobber last_error wholesale.
            backend = self._player_backend or "player"
            exit_code = proc.returncode
            chunk_idx = self.stats.chunks_processed
            watchdog_msg = (
                f"{backend} exited code={exit_code} at chunk={chunk_idx} "
                f"(restart #{self.stats.player_restarts})"
            )
            with self._stats_lock:
                self.stats.helper_exit_reasons.append(watchdog_msg)
                if len(self.stats.helper_exit_reasons) > 10:
                    self.stats.helper_exit_reasons.pop(0)
            self.record_error(
                f"{backend} respawned (restarts={self.stats.player_restarts}); "
                f"causes={self.stats.helper_exit_reasons[-3:]}"
            )
            # Spawn a fresh stderr reader bound to the new process.
            #
            # join the OLD
            # reader thread before overwriting the reference.
            # Pre-fix we just reassigned `self._stderr_thread = new`
            # -- the prior thread became unreachable, kept its FD,
            # and an external thread inspecting `self._stderr_thread
            # .is_alive()` would only see the new one (the old was a
            # daemon, would eventually exit, but until it did we
            # leaked one Thread object per respawn). The old thread
            # is daemon + reads from the dead process's stderr pipe,
            # which EOFs as soon as the process is terminated -- so
            # the join is at most a few ms.
            old_stderr_t = self._stderr_thread
            if old_stderr_t is not None and old_stderr_t.is_alive():
                old_stderr_t.join(timeout=0.5)
            stderr_t = threading.Thread(
                target=self._stderr_reader_loop,
                args=(new_proc,),
                name="woys-pacat-stderr",
                daemon=True,
            )
            stderr_t.start()
            self._stderr_thread = stderr_t
