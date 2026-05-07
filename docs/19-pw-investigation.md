# 19 — Native PipeWire output: implementation-path investigation

Date: 2026-05-08
Author: research agent (read-only; no code changes)
System: PipeWire 1.6.4 on CachyOS, libpipewire-0.3 + headers installed,
RTX 2070 Mobile, dev `.venv` has `sounddevice` + `cffi` (no
`pipewire-python`, no `pyaudio`, no `gi`).

---

## 0. Setting the question precisely

The cuts the audit fingers as "voice-correlated, ~40 ms-quantized,
quasi-periodic at PipeWire quantum cadence" (`docs/16-audit/synthesis.md`
P0-4, lens 08 FFT peaks at 21.33 / 42.67 ms) plausibly originate from a
specific failure mode: **the engine writes audio to `pw-cat`'s stdin in
chunky bursts (every ~150 ms), and pw-cat's realtime `process()`
callback reads that stdin synchronously inside the RT thread**. When the
producer's burst arrival doesn't line up with PipeWire's quantum
boundary, `process()` hits a stdin-empty condition and hands PipeWire a
zero-filled (or short) buffer for that quantum. One quantum at
1024/48000 = 21.33 ms; two adjacent quanta missed = 42.67 ms; both
appear in lens 08's onset-periodicity FFT.

I confirmed via WebFetch of `pw-cat.c` on PipeWire master that the fill
function (calling `sf_readf_*` from libsndfile, which on stdin is a
blocking read of the pipe) is invoked **inside the RT process callback
chain**, not in a separate prebuffering thread. There is no decoupling
ring buffer between stdin and the PipeWire stream. The audit's
hypothesis is mechanistically sound, and any fix that keeps a subprocess
+ pipe on the hot path inherits this race unless the helper explicitly
prebuffers off the RT thread.

The single most important property of any fix:

> The thing that PipeWire's `process()` callback reads from MUST be a
> ring buffer or pre-staged buffer that is filled on a separate, NON-RT
> thread by the producer. The RT thread MUST NEVER block on producer
> arrival — only on memcpy from the ring buffer.

Every option below is graded against that property first.

---

## 1. Option A — `sounddevice.OutputStream` (PortAudio)

### Hard blocker on this system

`pacman -Qi portaudio` shows PortAudio 19.7.0 linked only against
`libasound` and `libjack` — **no PulseAudio host API**. The current
engine.py prologue (lines 1–22) documents this: on CachyOS,
`sd.OutputStream(device=…)` cannot target `WoysSink` by name because
PortAudio has no Pulse host to translate that name. Setting
`PULSE_SINK=…` in env is ignored for the same reason. This is exactly
why woys originally moved to `pacat` in v0.1.1.

You could route through ALSA's PipeWire-aware plugin
(`/usr/share/alsa/alsa.conf.d/50-pipewire.conf`) with an ALSA device
string like `"pipewire:NAME=WoysSink"`, but: (a) that path goes
ALSA → libasound PipeWire plugin → PipeWire, adding a new round of
buffer-translation between ALSA's period model and PipeWire's quantum
model; (b) that plugin has its own ring buffer and its own xrun
behavior, which is documented to be flakier than native PipeWire
clients; (c) device-by-name targeting through the ALSA plugin is
historically inconsistent, and woys would lose the explicit
`--target=WoysSink` guarantee.

If you wanted to take A seriously you would have to rebuild PortAudio
locally with the experimental Pulse host API (PortAudio issue #425), or
ship a PortAudio with a PipeWire backend that doesn't yet exist. That
is a worse maintenance position than B or C.

### Per-quantum-gap pathology

Even on a system where PortAudio's PulseAudio host API DID exist (which
it doesn't here), PortAudio's PulseAudio backend uses `pa_stream_write`
into the pulseaudio protocol, and pipewire-pulse translates that into
PipeWire stream buffers. There IS a buffering layer in pipewire-pulse
(`tlength` / `minreq` / `prebuf` per-stream parameters), but it doesn't
fundamentally decouple bursty writes from quantum cadence — it only
sets the average target depth. If the producer is bursty enough to
underflow `tlength` in any window where it's consumed at the quantum
rate, you'll see the same ~21/43 ms gaps. **Same class of bug,
different surface.**

The PortAudio callback model would partially help — PortAudio runs its
own callback on its own host-thread cadence and would be a natural
place to prebuffer — BUT the callback fires from PortAudio's
PulseAudio-callback thread, which itself is downstream of
pipewire-pulse's translation, which itself is downstream of PipeWire's
RT thread. Three buffering layers, only one of which woys controls.
None of them is at the PipeWire quantum boundary, where the actual
race lives.

### Verdict

| Property | Rating |
|---|---|
| Solves quantum-gap pathology | **No** — wrong layer; doesn't reach the quantum boundary |
| LOC | ~50 (theoretical, if portaudio pulse host existed) |
| Brittleness | High — requires non-default PortAudio build OR ALSA-PipeWire plugin gymnastics |
| Already-blocked on this host | **Yes** — no Pulse host API in PortAudio |

Eliminate.

---

## 2. Option A.5 — `pipewire-python` (PyPI)

### Maturity check (web research)

`pipewire-python` (pablodz, last release 0.2.3) explicitly states
"STREAMING NOT SUPPORTED BY NOW" on its docs page. Its implementation
strategy is to **shell out to PipeWire's CLI tools** (`pw-cat`,
`pw-record`, `pw-link`) and parse their output. It's a wrapper around
the very subprocess pattern we're trying to replace.

Loading audio data into a live `pw_stream` from Python (which is the
actual ask) is not supported. The package is dbus/CLI-control-only.

### Verdict

| Property | Rating |
|---|---|
| Solves quantum-gap pathology | **No** — wraps the same `pw-cat` subprocess we already use |
| LOC | n/a — feature missing upstream |
| Brittleness | High (small unmaintained package, version 0.2.x) |

Eliminate.

---

## 3. Option B — `ctypes`/`cffi` shim against `libpipewire-0.3`

### ABI surface needed

The minimum surface to register an Audio/Sink → ... no, to register a
PipeWire output stream that targets `WoysSink` and feeds it raw float32:

**Stream + loop (~12 functions)**
- `pw_init(int *argc, char ***argv)` — once per process
- `pw_thread_loop_new(name, props)` — the RT-aware loop wrapper
- `pw_thread_loop_get_loop(tloop)` → `struct pw_loop *`
- `pw_thread_loop_start(tloop)` / `pw_thread_loop_stop(tloop)`
- `pw_thread_loop_lock(tloop)` / `pw_thread_loop_unlock(tloop)` — for
  config from Python side
- `pw_stream_new_simple(loop, name, props, events, userdata)` — creates
  the stream attached to the loop
- `pw_stream_connect(stream, dir, target_id, flags, params, n_params)`
- `pw_stream_dequeue_buffer(stream)` → `struct pw_buffer *`
- `pw_stream_queue_buffer(stream, buf)`
- `pw_stream_disconnect(stream)`, `pw_stream_destroy(stream)`

**Properties helpers (~3 functions)**
- `pw_properties_new(key1, val1, …, NULL)` — variadic; safer to use
  `pw_properties_new_dict(spa_dict)` instead, where you build a flat
  spa_dict in Python.

**SPA POD format builder — the hard part**
- The format param (which says "I want SPA_AUDIO_FORMAT_F32, 48000 Hz,
  stereo") is built with `spa_format_audio_raw_build(builder, id,
  &SPA_AUDIO_INFO_RAW_INIT(...))`. That macro expands to inline POD
  encoding which is awkward to replicate in pure Python.

**Two practical workarounds for the POD** (in order of preference):
1. Build the POD as a static bytes blob once in C (offline), embed it
   as a Python `bytes` literal, and pass a pointer to the bytes when
   calling `pw_stream_connect`. The POD for `f32 / 48000 / 2ch / mono
   layout` is fixed at woys's known config — no runtime variation.
   Lifting the bytes from a tiny C helper that prints `xxd` output is
   ~30 minutes of work and the POD never has to be rebuilt unless
   format changes.
2. Use `cffi` (already in the venv) in API mode against
   `<spa/param/audio/format-utils.h>` — cffi can compile a tiny stub
   that calls `spa_format_audio_raw_build` and returns the POD bytes.
   This is cleaner but needs `cffi.FFI().set_source(...)` at install
   time, which adds a build step.

**Threading**
- PipeWire calls `process()` from its RT data thread. Python callbacks
  invoked from a non-Python-created thread require the GIL, which
  `ctypes` callbacks acquire automatically (via `CFUNCTYPE`). So the
  callback works, but **acquiring the GIL on the RT thread is exactly
  the kind of priority inversion that produces the gaps we're trying
  to fix**. This is the deal-breaker for the "naive" version of B.
- The fix: do NOT register a `process` callback that runs Python.
  Instead, register a `process` callback that's a tiny C trampoline
  (compiled with cffi, or pre-compiled and loaded as a separate .so)
  that does ONE thing: dequeue a buffer, memcpy from a pre-filled
  lock-free SPSC ring buffer, queue the buffer. The Python side fills
  the ring buffer from its own non-RT writer thread.
- This is essentially Option C with the helper inside our process
  instead of out — same architecture, no subprocess hop, same RT
  hygiene. If you take this path you're writing a small amount of C
  anyway.

### LOC estimate

Pure-Python ctypes glue + format POD as embedded bytes + non-RT writer
thread + Python `process` callback (NOT recommended): **~200–300 LOC**.
Will exhibit Python-GIL-on-RT-thread microstutter under load.

ctypes glue + cffi-compiled C trampoline for the RT callback +
SPSC ring buffer: **~350–500 LOC** total, of which ~80 LOC is the C
trampoline. This is the ACTUALLY-SAFE version and is the right thing.

### Prior-art check

Web search returned no production Python-via-ctypes-pw_stream client.
The `pipewire_python` package wraps the CLI, not the library. CodePal
results are LLM-generated stubs without working format negotiation. A
ctypes binding for `pw_stream` exists nowhere I can find that handles
the POD-builder macros. **You're writing this from scratch.** The
official C reference is `tutorial4.c` (sine generator,
`docs/pipewire.org/page_tutorial4.html`) — pasted in §6 below for
reference. It's ~80 lines of C and is the mental template.

### Brittleness

Medium. PipeWire's stable ABI (`libpipewire-0.3`) has held since 2020
and the ABI versioning model is conservative; `pw_stream`'s public API
is unlikely to break in a `0.3.x` minor. The SPA POD encoding is
considered ABI-stable. The risk is that you embed a binary POD blob
that goes out of date silently if the format-builder macro semantics
change — but that has happened zero times since 0.3.0. Pinning
`libpipewire-0.3` as a runtime dependency (already true on every
modern PipeWire system) makes this medium-low.

### Solves the pathology?

**Yes — IF you do it right.** "Right" means:

1. The RT `process()` callback is C, not Python. Either via cffi
   trampoline or a separately compiled `.so`. Python on RT thread
   guarantees occasional GIL-induced gaps under load.
2. The data path between Python writer and RT callback is a lock-free
   SPSC ring buffer (or `pw_stream_set_active(false)` while filling,
   which is a different pattern but works for non-realtime preroll).
3. The ring buffer is sized at >= 2× quantum (e.g., 4096 frames at
   1024-quantum) so single-quantum jitter never starves it.

If you do A in B: write Python `process` callback, naive — you'll get
the same class of gaps as pw-cat, just with different timing. The
work is in the discipline, not the API surface.

### First-cut sketch

```python
# src/audio/pw_native.py — sketch, NOT working code

import ctypes as C
import threading
from ctypes import c_int, c_void_p, c_char_p, c_uint32, CFUNCTYPE

# Load the library (hard-fail on failure per brief).
_lib = C.CDLL("libpipewire-0.3.so.0")  # not just libpipewire-0.3.so

# Minimal struct opaque types.
class _Loop(C.Structure): pass
class _Stream(C.Structure): pass
class _StreamEvents(C.Structure):
    _fields_ = [
        ("version", c_uint32),
        ("destroy", CFUNCTYPE(None, c_void_p)),
        ("state_changed", CFUNCTYPE(None, c_void_p, c_int, c_int, c_char_p)),
        ("control_info", c_void_p),  # NULL
        ("io_changed", c_void_p),    # NULL
        ("param_changed", c_void_p), # NULL
        ("add_buffer", c_void_p),    # NULL
        ("remove_buffer", c_void_p), # NULL
        ("process", CFUNCTYPE(None, c_void_p)),  # the RT callback
        ("drained", c_void_p),       # NULL
        ("command", c_void_p),       # NULL
        ("trigger_done", c_void_p),  # NULL
    ]

# Pre-built POD bytes for f32 / 48000 / 2ch (built once via tiny C helper)
_FORMAT_POD = bytes.fromhex("....")  # offline-generated

class NativePipewireSink:
    def __init__(self, target_name: str, sample_rate: int, channels: int,
                 ring_frames: int = 8192):
        self._ring = SPSCRing(ring_frames * channels * 4)  # bytes
        self._writer_lock = threading.Lock()
        self._stop = threading.Event()

        _lib.pw_init(None, None)
        self._tloop = _lib.pw_thread_loop_new(b"woys-pw", None)
        self._loop = _lib.pw_thread_loop_get_loop(self._tloop)

        props = _lib.pw_properties_new(
            b"media.type", b"Audio",
            b"media.category", b"Playback",
            b"media.role", b"Production",
            b"node.name", b"woys-engine-out",
            b"target.object", target_name.encode(),
            b"node.latency", b"1024/48000",  # request 1024-frame quantum
            None,
        )

        # Hold a strong ref to events struct + callback functions
        # otherwise Python GCs the trampoline and PipeWire crashes.
        self._on_process = CFUNCTYPE(None, c_void_p)(self._process_cb)
        self._events = _StreamEvents(
            version=2,
            process=self._on_process,
            # ... other fields zeroed
        )

        self._stream = _lib.pw_stream_new_simple(
            self._loop, b"woys-engine-out", props, C.byref(self._events),
            None,  # userdata not needed; we close over self in trampoline
        )

        params = (c_void_p * 1)(C.cast(_FORMAT_POD, c_void_p))
        rc = _lib.pw_stream_connect(
            self._stream, 1,  # PW_DIRECTION_OUTPUT
            0xFFFFFFFF,       # PW_ID_ANY
            (1 << 0) | (1 << 2) | (1 << 4),  # AUTOCONNECT | MAP_BUFFERS | RT_PROCESS
            params, 1,
        )
        if rc < 0:
            raise RuntimeError(f"pw_stream_connect failed: {rc}")

        _lib.pw_thread_loop_start(self._tloop)

    def write(self, mono_f32: bytes) -> None:
        """Called from the engine writer thread (NOT RT)."""
        self._ring.write(mono_f32)

    def _process_cb(self, _userdata):
        """RT callback — keep this VERY tight. No Python allocation, no
        log, no exception handling that could raise. This is still
        Python on the RT thread — see brittleness note: under load this
        will produce occasional GIL gaps. The right v2 is to compile
        this trampoline to C via cffi.set_source()."""
        b = _lib.pw_stream_dequeue_buffer(self._stream)
        if not b: return
        # ... read .size, memcpy from self._ring into spa_buffer.datas[0]
        # ... set chunk.offset / chunk.stride / chunk.size
        _lib.pw_stream_queue_buffer(self._stream, b)
```

The above is ~150 LOC of skeleton. The remaining work is the SPSC ring,
the spa_buffer struct unpacking (just two pointer dereferences and four
field offsets — but they need to be right), and the format POD blob.

---

## 4. Option C — small native helper binary (C or Rust), pipe to Python

### What this is

Same architecture as `pw-cat`, except we write the helper. The helper:
1. Opens stdin (set non-blocking).
2. Allocates a ring buffer of N quanta (e.g., 4× quantum = 4096 frames).
3. Spawns a producer-side thread that reads stdin in a loop and writes
   to the ring.
4. Registers a `pw_stream` against `WoysSink`, with a `process()`
   callback that ONLY does ring-buffer reads (never blocks on stdin).
5. On ring-empty in `process()`, emits silence for that quantum AND
   bumps an underrun counter that's reported on stderr (so woys gets
   visibility — `pw-cat` is silent on underruns, which is one of the
   audit's complaints, lens 09 rank 1).

### Reference: `tutorial4.c`

The official PipeWire tutorial 4 (sine-generator, ~80 lines C, see §6)
is exactly the skeleton: `pw_init` → `pw_main_loop_new` →
`pw_stream_new_simple` → `pw_stream_connect` with the format POD →
`pw_main_loop_run`. The diff from tutorial4 to a pw-cat-replacement is:
replace the sine-fill in `on_process` with a ring-buffer read; add a
producer thread that reads stdin into the ring; switch
`pw_main_loop_*` → `pw_thread_loop_*` so the producer thread doesn't
have to share a thread with the RT callback.

### LOC estimate

Minimum-viable C helper: **~250 LOC**, including:
- ~60 LOC: argument parsing, signal handling, log setup
- ~40 LOC: SPSC ring buffer (or vendor a small one)
- ~40 LOC: stdin producer thread
- ~80 LOC: PipeWire setup (lifted from tutorial4) + format POD via
  `spa_format_audio_raw_build`
- ~30 LOC: process callback (dequeue → memcpy from ring → queue)

In Rust: ~350 LOC with `pipewire-rs` crate (which is mature, used by
helvum etc.); but Rust adds a build dependency that woys doesn't
otherwise have. C keeps the dependency story to "gcc + libpipewire-0.3
headers" which is already required to build any package on this
system.

### Build & ship story

The helper is a single `.c` file. Build with:
```
gcc -O2 -o woys-pw-out woys-pw-out.c \
    $(pkg-config --cflags --libs libpipewire-0.3)
```
Ship the binary in `bin/` of the woys package; `woys run` execs it.
Build can be hooked into woys's existing Makefile or `pyproject`'s
build script (uv handles `setup.py` shells fine).

### Solves the pathology?

**Yes, fully.** This is the strongest of the four options on the
pathology question because:
1. The RT `process()` callback is pure C, no GIL. No Python ever
   touches the RT thread.
2. The stdin-read is decoupled from `process()` by an explicit ring
   buffer that we own. No `sf_readf_*`-on-RT-thread blocking like
   pw-cat has.
3. The ring buffer underrun is observable (a counter that the helper
   prints on stderr, which woys's existing stderr-reader can tail).
   This solves the "pw-cat is silent on underruns" complaint
   (audit lens 09 rank 1) for free.

The remaining failure mode is "engine produces audio bursty enough to
underflow the helper's ring," which is the SAME failure mode any other
non-callback-based engine architecture has; it's an engine-side
problem, not an output-stage one. The output stage cannot create gaps
that aren't really there in the engine output.

### Brittleness

Low. C against `libpipewire-0.3` has been ABI-stable since 2020. The
helper has zero Python dependencies. Surface area is small enough to
audit in one sitting. Failure modes are well-bounded (binary missing →
hard fail at engine start; PipeWire crashes → SIGPIPE on stdin →
helper exits → woys's existing watchdog respawns).

### What it doesn't fix

- It does not eliminate the subprocess hop. Telemetry still has to
  pass over a stderr pipe.
- It does not eliminate the engine→stdin write step. If that write is
  EAGAIN-bursty (it shouldn't be at woys's data rate, but if it is)
  the helper's ring fills slower than it drains and you still get
  underruns. Mitigate: writer threads in woys already exist; just
  make sure they `write()` synchronously, not via `select()`.

### First-cut sketch

```c
// woys-pw-out.c — sketch, NOT working code, ~250 LOC final
#include <pipewire/pipewire.h>
#include <spa/param/audio/format-utils.h>
#include <pthread.h>
#include <stdatomic.h>
#include <unistd.h>

#define RING_FRAMES 4096
#define CHANNELS 2
#define RATE 48000

struct ring {
    float buf[RING_FRAMES * CHANNELS];
    _Atomic uint32_t head;
    _Atomic uint32_t tail;
};

static struct ring g_ring;
static struct pw_thread_loop *g_tloop;
static struct pw_stream *g_stream;
static _Atomic uint64_t g_underruns;

static size_t ring_read(float *dst, size_t n_frames) { /* SPSC pop */ }
static size_t ring_write(const float *src, size_t n_frames) { /* SPSC push */ }

static void on_process(void *userdata) {
    struct pw_buffer *b = pw_stream_dequeue_buffer(g_stream);
    if (!b) return;
    struct spa_buffer *sb = b->buffer;
    float *dst = sb->datas[0].data;
    if (!dst) return;
    size_t maxframes = sb->datas[0].maxsize / (sizeof(float) * CHANNELS);
    size_t want = b->requested ? SPA_MIN(b->requested, maxframes) : maxframes;
    size_t got = ring_read(dst, want);
    if (got < want) {
        memset(dst + got * CHANNELS, 0, (want - got) * CHANNELS * sizeof(float));
        atomic_fetch_add_explicit(&g_underruns, 1, memory_order_relaxed);
    }
    sb->datas[0].chunk->offset = 0;
    sb->datas[0].chunk->stride = sizeof(float) * CHANNELS;
    sb->datas[0].chunk->size = want * sizeof(float) * CHANNELS;
    pw_stream_queue_buffer(g_stream, b);
}

static void *stdin_thread(void *_) {
    float buf[1024 * CHANNELS];
    while (1) {
        ssize_t n = read(0, buf, sizeof(buf));
        if (n <= 0) break;
        // spin/sleep until ring has room — never block in process()
        size_t frames = n / (sizeof(float) * CHANNELS);
        size_t written = 0;
        while (written < frames) {
            size_t w = ring_write(buf + written * CHANNELS, frames - written);
            written += w;
            if (w == 0) usleep(500);  // 0.5 ms backoff
        }
    }
    return NULL;
}

static void on_stderr_tick(void *_) {
    // Every N seconds, print "underruns=X" so woys can read it.
    fprintf(stderr, "underruns=%lu\n",
        atomic_load_explicit(&g_underruns, memory_order_relaxed));
}

int main(int argc, char **argv) {
    pw_init(&argc, &argv);
    g_tloop = pw_thread_loop_new("woys-pw-out", NULL);
    // ... set up stream w/ target.object=WoysSink, format f32/48k/2ch ...
    pthread_t producer;
    pthread_create(&producer, NULL, stdin_thread, NULL);
    pw_thread_loop_start(g_tloop);
    pthread_join(producer, NULL);  // exits when stdin EOF
    pw_thread_loop_stop(g_tloop);
    pw_stream_destroy(g_stream);
    pw_thread_loop_destroy(g_tloop);
    return 0;
}
```

That's the concrete shape. ~80 LOC sketched, ~250 LOC for production
(error handling, exit cleanup, stderr telemetry tick, format POD
construction, target-by-name verification before connect).

---

## 5. Recommendation

**Take Option C.**

### Why C beats B

The brief's actual ask is "close the per-quantum-gap pathology."
Both B (cffi + ctypes shim) and C (helper binary) can do this **if and
only if** the RT `process()` callback is non-Python. Therefore at the
critical path, both options end up with C code. The only real
difference is whether that C code lives:
- Inside woys's process, called via cffi/ctypes (Option B)
- In a sidecar binary, called via subprocess + pipe (Option C)

Given that constraint, the engineering question is "which C-code
location is cheaper to ship, audit, and maintain?" Sidecar wins on
every axis:

1. **Build is trivial.** One `.c` file, one pkg-config invocation. No
   cffi `set_source` step at install time. No ABI mismatch risk
   between the version of libpipewire ctypes was authored against and
   the version present at runtime.

2. **Failure isolation.** PipeWire crashing the helper sends SIGPIPE
   to woys's writer; existing watchdog respawns. PipeWire crashing an
   in-process ctypes call takes the entire engine with it. The
   subprocess hop, which is the audit's hypothesis-source for the
   gaps, is in fact a robustness asset for crash-isolation — and the
   gap pathology comes not from "subprocess" but from "subprocess
   reads stdin in RT thread."

3. **Less Python on the RT-adjacent thread.** Even if Option B's
   `process()` is a cffi-compiled C trampoline, the cffi runtime still
   maintains some Python bookkeeping that runs at module-load time and
   can fragment GIL-cooperative state. A clean subprocess with no
   Python state at all is more predictable.

4. **The helper is an asset for diagnosis.** A ~250 LOC binary is
   small enough to instrument heavily (per-quantum jitter, ring
   high/low watermark, underrun count, queued-write size histogram)
   and report on stderr. The audit's lens 09 rank 1 finding ("pw-cat
   is silent on underruns") becomes a free win.

5. **Brittleness is lowest of all options.** No ABI surface from
   Python; ABI surface from C against `libpipewire-0.3` is small and
   stable. No `cffi` build step, no `ctypes` POD manual encoding, no
   risk of `_FORMAT_POD` blob going stale silently.

6. **It's the smallest amount of code on the hot path.** ~80 LOC of
   PipeWire integration on the RT thread, all in one file, all in C,
   all easy to audit. Compare to Option B's ~80 LOC of C trampoline +
   ~300 LOC of Python ctypes glue + ~50 LOC of POD-blob generation
   tooling.

### Why C beats A and A.5

A is blocked by no-Pulse-host-API in PortAudio on this host (engine.py
prologue documents this; verified via `pacman -Qi portaudio` showing
deps `alsa-lib + jack` only). A.5's `pipewire-python` package
explicitly does not support streaming. Both eliminated upstream of
quality considerations.

### Concrete shape of the v0.9 work

1. `bin/woys-pw-out.c` — ~250 LOC, builds with pkg-config + libpipewire-0.3
2. `bin/Makefile` (or hook into pyproject) — single `gcc` line
3. `src/audio/engine.py::_open_pacat` becomes `_open_player` and:
   - if `cfg.prefer_native_pw`: spawn `bin/woys-pw-out --target=WoysSink --rate=48000 --channels=2 --quantum=1024`
   - hard-fail (raise) on registration failure: parse first stderr line ("ready" / "error") with timeout; if "error" or timeout, RuntimeError. Per brief: never silent fallback.
   - else (legacy): keep existing pw-cat / pacat path for one release as a kill-switch
4. `_pacat_stderr_reader` → `_player_stderr_reader`: parse helper's
   `underruns=N` lines into `EngineStats.player_underruns` — gives woys
   visibility we never had with pw-cat.
5. CI/test: build the helper as part of `make test`, smoke-test it
   with a 1-second f32 sine via stdin against a `pw-loopback` sink in
   the test environment. (CI runner needs PipeWire — which is the
   developer's machine right now; remote CI is a v0.9.x problem.)
6. Make `prefer_native_pw=true` the default once 1 release of soak
   confirms no regressions; then delete pw-cat / pacat paths in v0.10.

Time estimate: ~2 days of work for a v0.9.0-rc1 candidate, dominated
by getting the format POD right, getting the SPSC ring's memory
ordering right, and the test harness. The C is small but RT-adjacent
C demands care.

---

## 6. Reference: `tutorial4.c` (PipeWire upstream, fetched 2026-05-08)

```c
#include <math.h>
#include <spa/param/audio/format-utils.h>
#include <pipewire/pipewire.h>

#define M_PI_M2 (M_PI + M_PI)
#define DEFAULT_RATE 44100
#define DEFAULT_CHANNELS 2
#define DEFAULT_VOLUME 0.7

struct data {
    struct pw_main_loop *loop;
    struct pw_stream *stream;
    double accumulator;
};

static void on_process(void *userdata) {
    struct data *data = userdata;
    struct pw_buffer *b;
    struct spa_buffer *buf;
    int i, c, n_frames, stride;
    int16_t *dst, val;

    if ((b = pw_stream_dequeue_buffer(data->stream)) == NULL) {
        pw_log_warn("out of buffers: %m");
        return;
    }
    buf = b->buffer;
    if ((dst = buf->datas[0].data) == NULL) return;
    stride = sizeof(int16_t) * DEFAULT_CHANNELS;
    n_frames = buf->datas[0].maxsize / stride;
    if (b->requested) n_frames = SPA_MIN(b->requested, n_frames);

    for (i = 0; i < n_frames; i++) {
        data->accumulator += M_PI_M2 * 440 / DEFAULT_RATE;
        if (data->accumulator >= M_PI_M2) data->accumulator -= M_PI_M2;
        val = sin(data->accumulator) * DEFAULT_VOLUME * 32767.0;
        for (c = 0; c < DEFAULT_CHANNELS; c++) *dst++ = val;
    }

    buf->datas[0].chunk->offset = 0;
    buf->datas[0].chunk->stride = stride;
    buf->datas[0].chunk->size = n_frames * stride;
    pw_stream_queue_buffer(data->stream, b);
}

static const struct pw_stream_events stream_events = {
    PW_VERSION_STREAM_EVENTS,
    .process = on_process,
};

int main(int argc, char *argv[]) {
    struct data data = { 0, };
    const struct spa_pod *params[1];
    uint8_t buffer[1024];
    struct spa_pod_builder b = SPA_POD_BUILDER_INIT(buffer, sizeof(buffer));

    pw_init(&argc, &argv);
    data.loop = pw_main_loop_new(NULL);
    data.stream = pw_stream_new_simple(
        pw_main_loop_get_loop(data.loop),
        "audio-src",
        pw_properties_new(
            PW_KEY_MEDIA_TYPE, "Audio",
            PW_KEY_MEDIA_CATEGORY, "Playback",
            PW_KEY_MEDIA_ROLE, "Music",
            NULL),
        &stream_events,
        &data);

    params[0] = spa_format_audio_raw_build(&b, SPA_PARAM_EnumFormat,
        &SPA_AUDIO_INFO_RAW_INIT(
            .format = SPA_AUDIO_FORMAT_S16,
            .channels = DEFAULT_CHANNELS,
            .rate = DEFAULT_RATE));

    pw_stream_connect(data.stream,
        PW_DIRECTION_OUTPUT,
        PW_ID_ANY,
        PW_STREAM_FLAG_AUTOCONNECT |
        PW_STREAM_FLAG_MAP_BUFFERS |
        PW_STREAM_FLAG_RT_PROCESS,
        params, 1);

    pw_main_loop_run(data.loop);
    pw_stream_destroy(data.stream);
    pw_main_loop_destroy(data.loop);
    return 0;
}
```

For woys's helper: replace `on_process`'s sine-fill with a SPSC-ring
read; replace `pw_main_loop_*` with `pw_thread_loop_*`; add a producer
thread that reads stdin into the ring; change format to
`SPA_AUDIO_FORMAT_F32`, rate 48000, mono or stereo per
`cfg.output_channels`; add `PW_KEY_TARGET_OBJECT="WoysSink"` to the
properties; print `underruns=N` on stderr every second.

---

## 7. Open questions / follow-ups

These are unresolved without code or a running test environment:

1. **Does target-by-name still work via `PW_KEY_TARGET_OBJECT` in
   PipeWire 1.6.4?** Older docs say yes; the property has been stable
   since 0.3.64. Mentioning here because woys's existing assert in
   `_assert_sink_loaded` checks that `WoysSink` is a loaded PulseAudio
   sink (via `pactl`); the helper would want an analogous check
   against PipeWire's object registry, perhaps via
   `PW_KEY_OBJECT_SERIAL` lookup at startup. **Not blocking — but
   need a test with `pw-loopback` to confirm target-by-name.**

2. **What quantum does WoysSink negotiate?** Today, `node.latency` on
   the WoysSink graph reports `13440/48000` (lens 05 in the audit) —
   that's 280 ms, woys's current `output_latency_ms` request. The
   helper should request `1024/48000` quantum via `node.latency`
   property, but PipeWire's per-node quantum negotiation gives the
   driver final say. **Need a runtime check that `pw_stream_get_time`
   reports the expected quantum after STREAMING state, and bail
   loud if it doesn't.**

3. **Can the helper safely use `PW_STREAM_FLAG_RT_PROCESS`?** The
   process callback runs in the PipeWire data thread (RR-20 per audit
   lens 07). Our `on_process` does ring-read + memcpy + atomic
   counter. All RT-safe. But if the SPSC ring uses a futex-backed
   condition variable on the writer side (it shouldn't — it should
   be busy-poll with usleep backoff on stdin-empty), that's a
   priority inversion risk. **Helper must use lock-free SPSC, period.**
   Implementation note for the eventual writer.

4. **Telegram-specific test plan.** None of the four options is
   verifiable in CS2/Discord alone — the audit's P1-5 finding is that
   Telegram's tg_owt exposes gaps that other VOIP stacks mask. A
   v0.9.0-rc Telegram self-call test, with the helper's stderr
   `underruns=` count tailed in real time, is the only honest
   verification. **Need a 5-minute Telegram self-call protocol** as
   part of the v0.9 release checklist.

5. **What happens when the helper is missing on the user's system?**
   Brief says hard-fail, never silent fallback. The packaging plan
   must guarantee the helper is built and shipped alongside the
   Python code. If using `uv` / `pip`, this means a `setup.py` shim
   or a `cibuildwheel`-style binary wheel. If using a system package,
   the AUR PKGBUILD ships it. **Simplest first cut: ship as part of
   the woys repo and require source build via `make`; v0.9.x can
   address proper packaging.**

6. **Does woys want to handle the sample-rate conversion in the
   helper or in Python?** Today, the engine resamples to 48 kHz
   before writing. Keeping it in Python is simpler. The helper's
   format negotiation should hard-fix at 48 kHz f32; if the engine
   ever wants to push native model rate, that's a future change.
   **Default: helper accepts only 48 kHz f32, hard-fails if
   negotiation gives anything else.**

7. **Existing `pacat`/`pw-cat` path retention.** The brief says
   `prefer_native_pw=true` default. Recommend: ship v0.9.0 with
   `prefer_native_pw=false` default, run one release of opt-in soak,
   flip default in v0.9.1, delete pw-cat path in v0.10. This is one
   release slower than the brief implies — but the audit's history of
   sleeper-flip regressions (lens 09 rank 1: rc1 silently flipped
   `prefer_pw_cat` and three rcs were spent debugging the result)
   argues for explicit opt-in for the first release.

---

## Sources

- [PipeWire stream.h header (local)](file:///usr/include/pipewire-0.3/pipewire/stream.h)
- [PipeWire tutorial 4 (audio playback)](https://docs.pipewire.org/page_tutorial4.html)
- [PipeWire audio-src.c example](https://github.com/PipeWire/pipewire/blob/master/src/examples/audio-src.c)
- [pw-cat.c upstream source (master)](https://github.com/PipeWire/pipewire/blob/master/src/tools/pw-cat.c)
- [pipewire_python on PyPI (pablodz)](https://pypi.org/project/pipewire_python/) — controller-only, no streaming
- [PortAudio PipeWire backend issue #425](https://github.com/PortAudio/portaudio/issues/425) — backend not yet upstream
- [PortAudio PulseAudio host API source](https://files.portaudio.com/docs/v19-doxydocs-dev/pa__linux__pulseaudio__cb_8c.html) — reference, not a fix here
- [PipeWire Pulse protocol module (tlength/minreq/prebuf)](https://docs.pipewire.org/page_module_protocol_pulse.html)
- woys audit synthesis: `docs/16-audit/synthesis.md` (P0-4, lens 05, lens 08, lens 09 rank 1)
- woys engine prologue: `src/audio/engine.py:1-22` (PortAudio-no-Pulse-host-API constraint)
- woys engine subprocess open: `src/audio/engine.py:1943-2007` (`_open_pacat`)
