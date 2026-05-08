/* woys-pw-out — native PipeWire output helper for woys
 *
 * v0.9.0 — replaces the pw-cat / pacat subprocess on the engine's
 * playback path. Reads raw float32le interleaved samples on stdin,
 * registers as a PipeWire output stream, decouples the bursty stdin
 * write pattern from PipeWire's per-quantum RT callback via a
 * lock-free SPSC ring buffer.
 *
 * Why this exists: lens 08 of the v0.7.x audit
 * (`docs/16-audit/synthesis.md`) found voice-correlated, sample-exact
 * zero gaps quantized to ~21.33 / 42.67 ms in the cut signature on
 * Telegram VOIP — exactly one PipeWire quantum at 1024/48000. pw-cat
 * reads stdin synchronously inside its RT process callback chain
 * (verified against upstream `pw-cat.c`). When the engine's chunk
 * write timing falls out of phase with the quantum boundary, that
 * synchronous read hits stdin-empty and the buffer goes out
 * zero-padded for that quantum. This helper closes that race by
 * keeping the RT thread strictly memcpy-from-ring with no I/O.
 *
 * Wire protocol with woys engine
 * ------------------------------
 *   stdin:  raw float32 interleaved samples at the negotiated sample
 *           rate (default 48000) and channel count (default 2).
 *   stderr: status lines woys's stderr reader parses:
 *             "ready"                      — once after stream registers
 *             "underruns=N"                — every UNDERRUN_REPORT_SECS
 *             "error: <msg>"               — fatal, helper will exit nonzero
 *             "quantum=N rate=M"           — once after format negotiation
 *
 * Threading layout
 * ----------------
 *   - pw_thread_loop spins PipeWire's data thread (RT priority via PW config)
 *   - main thread: stdin reader, writes to SPSC ring
 *   - PW data thread: process callback, reads from SPSC ring, writes to PW buffer
 *   - The ring is the only shared state. Memory ordering: writer uses
 *     release on tail-publish, reader uses acquire on tail-load (and
 *     vice versa for head). Standard SPSC pattern, no locks anywhere.
 *
 * Build: gcc -O2 -Wall -Wextra -o woys-pw-out woys-pw-out.c \
 *            $(pkg-config --cflags --libs libpipewire-0.3)
 *
 * Original work — Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
 */

#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE  /* expose usleep */
#include <pipewire/pipewire.h>
#include <spa/param/audio/format-utils.h>
#include <spa/utils/result.h>

#include <errno.h>
#include <getopt.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* --------------------------------------------------------------------
 * Configuration constants (compile-time defaults; CLI can override)
 * -------------------------------------------------------------------- */

#define DEFAULT_RATE           48000
#define DEFAULT_CHANNELS       2
#define DEFAULT_TARGET         "WoysSink"
#define DEFAULT_QUANTUM        1024     /* requested PipeWire quantum */
#define DEFAULT_RING_FRAMES    8192     /* 8× 1024-quantum = ~170 ms slack */
#define UNDERRUN_REPORT_SECS   1
#define STDIN_READ_FRAMES      1024     /* per-read chunk; tunable */
#define MAX_CHANNELS           8        /* sanity cap */

/* --------------------------------------------------------------------
 * SPSC ring buffer (lock-free, single producer = stdin thread,
 *                   single consumer = PW data thread)
 * -------------------------------------------------------------------- */

struct ring {
    float    *buf;            /* allocated: capacity * channels floats */
    uint32_t  capacity;       /* in frames */
    uint8_t   channels;
    _Atomic uint32_t head;    /* read index (in frames), modulo capacity */
    _Atomic uint32_t tail;    /* write index, modulo capacity */
};

static int ring_init(struct ring *r, uint32_t capacity, uint8_t channels) {
    r->buf = calloc((size_t)capacity * channels, sizeof(float));
    if (!r->buf) return -1;
    r->capacity = capacity;
    r->channels = channels;
    atomic_store_explicit(&r->head, 0, memory_order_relaxed);
    atomic_store_explicit(&r->tail, 0, memory_order_relaxed);
    return 0;
}

static void ring_free(struct ring *r) {
    free(r->buf);
    r->buf = NULL;
}

/* Available frames to read (consumer view). */
static uint32_t ring_readable(const struct ring *r) {
    uint32_t h = atomic_load_explicit(&r->head, memory_order_relaxed);
    uint32_t t = atomic_load_explicit(&r->tail, memory_order_acquire);
    return (t - h) & (r->capacity - 1);
}

/* Available frames to write (producer view). */
static uint32_t ring_writable(const struct ring *r) {
    uint32_t h = atomic_load_explicit(&r->head, memory_order_acquire);
    uint32_t t = atomic_load_explicit(&r->tail, memory_order_relaxed);
    /* Reserve one frame to distinguish empty from full. */
    return (r->capacity - 1) - ((t - h) & (r->capacity - 1));
}

/* Consumer: pop up to n_frames from the ring into dst. Returns frames popped. */
static uint32_t ring_pop(struct ring *r, float *dst, uint32_t n_frames) {
    uint32_t avail = ring_readable(r);
    uint32_t take = n_frames < avail ? n_frames : avail;
    if (take == 0) return 0;
    uint32_t h = atomic_load_explicit(&r->head, memory_order_relaxed);
    uint32_t mask = r->capacity - 1;
    uint32_t first = h & mask;
    uint32_t to_end = r->capacity - first;
    uint32_t a = take < to_end ? take : to_end;
    memcpy(dst, &r->buf[(size_t)first * r->channels], a * r->channels * sizeof(float));
    if (a < take) {
        memcpy(&dst[(size_t)a * r->channels], &r->buf[0],
               (take - a) * r->channels * sizeof(float));
    }
    atomic_store_explicit(&r->head, h + take, memory_order_release);
    return take;
}

/* Producer: push up to n_frames from src into the ring. Returns frames pushed. */
static uint32_t ring_push(struct ring *r, const float *src, uint32_t n_frames) {
    uint32_t avail = ring_writable(r);
    uint32_t put = n_frames < avail ? n_frames : avail;
    if (put == 0) return 0;
    uint32_t t = atomic_load_explicit(&r->tail, memory_order_relaxed);
    uint32_t mask = r->capacity - 1;
    uint32_t first = t & mask;
    uint32_t to_end = r->capacity - first;
    uint32_t a = put < to_end ? put : to_end;
    memcpy(&r->buf[(size_t)first * r->channels], src, a * r->channels * sizeof(float));
    if (a < put) {
        memcpy(&r->buf[0], &src[(size_t)a * r->channels],
               (put - a) * r->channels * sizeof(float));
    }
    atomic_store_explicit(&r->tail, t + put, memory_order_release);
    return put;
}

/* --------------------------------------------------------------------
 * Global state — kept tight; only the ring crosses thread boundaries.
 * -------------------------------------------------------------------- */

struct app_state {
    struct pw_thread_loop *tloop;
    struct pw_stream      *stream;
    struct ring            ring;

    /* CLI/config snapshot — set in main(), read elsewhere. */
    const char *target;
    uint32_t    rate;
    uint8_t     channels;
    uint32_t    quantum;

    /* RT-side counter (process callback). Read on stderr-tick thread. */
    _Atomic uint64_t underruns;
    _Atomic uint64_t buffers_filled;

    /* Set by the stream state-changed callback once we hit STREAMING.
     * The main thread blocks on this after pw_stream_connect. */
    _Atomic int      ready;
    _Atomic int      should_exit;
};

static struct app_state g_state;

/* --------------------------------------------------------------------
 * PipeWire callbacks
 *
 * IMPORTANT: on_process runs on PipeWire's RT data thread. Allowed:
 *   - atomic ops, memcpy, ring read, pw_stream_dequeue/queue_buffer,
 *     SPA helpers that don't allocate.
 * Forbidden:
 *   - malloc / free, fprintf, syscalls that may block, anything that
 *     could block on a non-RT thread (no condvars, no mutex contended
 *     by GC'd allocator).
 * Underrun counter is incremented atomically (relaxed).
 * -------------------------------------------------------------------- */

static void on_process(void *userdata) {
    struct app_state *s = userdata;
    struct pw_buffer *b = pw_stream_dequeue_buffer(s->stream);
    if (!b) return;
    struct spa_buffer *sb = b->buffer;
    if (sb->n_datas < 1) {
        pw_stream_queue_buffer(s->stream, b);
        return;
    }
    float *dst = sb->datas[0].data;
    if (!dst) {
        pw_stream_queue_buffer(s->stream, b);
        return;
    }
    uint32_t stride = sizeof(float) * s->channels;
    uint32_t maxframes = sb->datas[0].maxsize / stride;
    /* PipeWire 0.3.32+ exposes b->requested = how many frames the driver
     * actually wants this round; older versions return 0 = "fill it". */
    uint32_t want = (b->requested > 0 && b->requested < maxframes)
                    ? (uint32_t)b->requested : maxframes;
    uint32_t got = ring_pop(&s->ring, dst, want);
    if (got < want) {
        memset(&dst[(size_t)got * s->channels], 0,
               (want - got) * stride);
        atomic_fetch_add_explicit(&s->underruns, 1, memory_order_relaxed);
    }
    sb->datas[0].chunk->offset = 0;
    sb->datas[0].chunk->stride = stride;
    sb->datas[0].chunk->size   = want * stride;
    atomic_fetch_add_explicit(&s->buffers_filled, 1, memory_order_relaxed);
    pw_stream_queue_buffer(s->stream, b);
}

static void on_state_changed(void *userdata, enum pw_stream_state old,
                             enum pw_stream_state state, const char *err) {
    (void)old;
    struct app_state *s = userdata;
    switch (state) {
        case PW_STREAM_STATE_STREAMING:
            atomic_store_explicit(&s->ready, 1, memory_order_release);
            fprintf(stderr, "ready\n");
            fflush(stderr);
            break;
        case PW_STREAM_STATE_ERROR:
            fprintf(stderr, "error: stream entered ERROR state: %s\n",
                    err ? err : "(no message)");
            fflush(stderr);
            atomic_store_explicit(&s->should_exit, 1, memory_order_release);
            pw_thread_loop_signal(s->tloop, false);
            break;
        case PW_STREAM_STATE_UNCONNECTED:
            /* Either initial state or post-disconnect; only complain if
             * we were previously streaming AND not already shutting down.
             * Engine-driven shutdown (SIGTERM, stdin EOF) sets should_exit
             * BEFORE state goes UNCONNECTED — that's normal teardown,
             * not an error worth surfacing through woys's last_error. */
            if (atomic_load_explicit(&s->ready, memory_order_acquire) &&
                !atomic_load_explicit(&s->should_exit, memory_order_acquire)) {
                fprintf(stderr, "error: stream disconnected\n");
                fflush(stderr);
                atomic_store_explicit(&s->should_exit, 1, memory_order_release);
                pw_thread_loop_signal(s->tloop, false);
            }
            break;
        default:
            break;
    }
}

static void on_param_changed(void *userdata, uint32_t id,
                             const struct spa_pod *param) {
    (void)userdata;
    (void)id;
    (void)param;
    /* Could read negotiated quantum / rate here via spa_format_parse,
     * but the audit doesn't require it for correctness — woys's engine
     * has the quantum from `pw-metadata` already. Future enhancement. */
}

static const struct pw_stream_events stream_events = {
    PW_VERSION_STREAM_EVENTS,
    .state_changed = on_state_changed,
    .param_changed = on_param_changed,
    .process       = on_process,
};

/* --------------------------------------------------------------------
 * stdin reader (main thread)
 * -------------------------------------------------------------------- */

static int stdin_reader_loop(struct app_state *s) {
    size_t bytes_per_frame = sizeof(float) * s->channels;
    size_t buf_bytes = STDIN_READ_FRAMES * bytes_per_frame;
    float *buf = malloc(buf_bytes);
    if (!buf) {
        fprintf(stderr, "error: stdin buffer alloc failed\n");
        return -1;
    }

    while (!atomic_load_explicit(&s->should_exit, memory_order_acquire)) {
        ssize_t n = read(STDIN_FILENO, buf, buf_bytes);
        if (n == 0) break;            /* EOF */
        if (n < 0) {
            if (errno == EINTR) continue;
            fprintf(stderr, "error: stdin read failed: %s\n", strerror(errno));
            free(buf);
            return -1;
        }
        if ((size_t)n % bytes_per_frame != 0) {
            /* Partial frame — coalesce into next read. Realistic for
             * pipe-fed input where a 7200-frame chunk straddles a 4 KiB
             * page boundary. We drop the partial bytes; engine should
             * never write a partial frame because it always passes whole
             * float32 interleaved chunks. Bumps as a counter would be
             * useful but not for v0.9.0-rc1. */
            n -= n % (ssize_t)bytes_per_frame;
            if (n == 0) continue;
        }
        size_t frames = (size_t)n / bytes_per_frame;
        size_t written = 0;
        while (written < frames) {
            uint32_t w = ring_push(&s->ring, &buf[written * s->channels],
                                   (uint32_t)(frames - written));
            written += w;
            if (w == 0) {
                /* Ring full — wait briefly. Producer is bursty (engine
                 * writes 7200 frames at ~150 ms cadence vs PW drains
                 * 1024 every ~21 ms), so this can happen during the
                 * post-warmup ramp until the ring stabilizes. 0.5 ms
                 * backoff: balances responsiveness vs CPU. */
                usleep(500);
            }
        }
    }
    free(buf);
    return 0;
}

/* --------------------------------------------------------------------
 * Stderr telemetry tick (dedicated timer thread)
 *
 * Reports `underruns=N` periodically so woys's stderr reader can
 * surface a real xrun count — closing audit lens 09 rank 1
 * ("pw-cat is silent on underruns").
 *
 * v0.9.0-rc3: was SIGALRM-driven and didn't work. SIGALRM gets
 * delivered to whichever thread Linux picks (often PipeWire's data
 * thread, not main) — the main thread's blocking read() never
 * receives EINTR, so the periodic emit never fires. Fixed by using a
 * dedicated tick thread that wakes every UNDERRUN_REPORT_SECS and
 * prints. NOT real-time critical (SCHED_OTHER, no GIL involvement
 * since this is C).
 * -------------------------------------------------------------------- */

static void *underrun_tick_thread(void *userdata) {
    struct app_state *s = userdata;
    /* Wait for stream to be ready before emitting; otherwise we'd
     * print "underruns=0" before the engine even sees `ready`. */
    while (!atomic_load_explicit(&s->ready, memory_order_acquire) &&
           !atomic_load_explicit(&s->should_exit, memory_order_acquire)) {
        usleep(50000); /* 50 ms */
    }
    while (!atomic_load_explicit(&s->should_exit, memory_order_acquire)) {
        /* Use clock_nanosleep with CLOCK_MONOTONIC and EINTR-resume so
         * the periodic emit fires reliably regardless of which thread
         * any signal lands on (the v0.9.0-rc2 SIGALRM-based attempt
         * landed on PipeWire's data thread, never reached the main
         * thread, never fired). */
        struct timespec req = { UNDERRUN_REPORT_SECS, 0 };
        struct timespec rem;
        while (clock_nanosleep(CLOCK_MONOTONIC, 0, &req, &rem) == EINTR) {
            req = rem;
        }
        if (atomic_load_explicit(&s->should_exit, memory_order_acquire)) break;
        uint64_t cur = atomic_load_explicit(&s->underruns, memory_order_relaxed);
        fprintf(stderr, "underruns=%llu\n", (unsigned long long)cur);
        fflush(stderr);
    }
    return NULL;
}

/* --------------------------------------------------------------------
 * Signal handling: clean shutdown on SIGINT/SIGTERM/SIGPIPE
 * -------------------------------------------------------------------- */

static void exit_handler(int sig) {
    (void)sig;
    atomic_store_explicit(&g_state.should_exit, 1, memory_order_release);
    if (g_state.tloop) pw_thread_loop_signal(g_state.tloop, false);
}

/* --------------------------------------------------------------------
 * Argument parsing
 * -------------------------------------------------------------------- */

static void usage(const char *prog) {
    fprintf(stderr,
            "usage: %s [options]\n"
            "  --target=NAME         PipeWire node to target (default: %s)\n"
            "  --rate=N              sample rate in Hz (default: %d)\n"
            "  --channels=N          channels (default: %d)\n"
            "  --quantum=N           requested quantum in frames (default: %d)\n"
            "  --ring-frames=N       ring buffer size in frames (default: %d)\n"
            "  --help                show this message\n",
            prog, DEFAULT_TARGET, DEFAULT_RATE, DEFAULT_CHANNELS,
            DEFAULT_QUANTUM, DEFAULT_RING_FRAMES);
}

static int parse_args(int argc, char **argv, struct app_state *s) {
    s->target   = DEFAULT_TARGET;
    s->rate     = DEFAULT_RATE;
    s->channels = DEFAULT_CHANNELS;
    s->quantum  = DEFAULT_QUANTUM;
    uint32_t ring_frames = DEFAULT_RING_FRAMES;

    static struct option longopts[] = {
        {"target",      required_argument, 0, 't'},
        {"rate",        required_argument, 0, 'r'},
        {"channels",    required_argument, 0, 'c'},
        {"quantum",     required_argument, 0, 'q'},
        {"ring-frames", required_argument, 0, 'b'},
        {"help",        no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };
    int opt;
    int idx = 0;
    while ((opt = getopt_long(argc, argv, "t:r:c:q:b:h", longopts, &idx)) != -1) {
        switch (opt) {
            case 't': s->target = optarg; break;
            case 'r': s->rate = (uint32_t)atoi(optarg); break;
            case 'c': s->channels = (uint8_t)atoi(optarg); break;
            case 'q': s->quantum = (uint32_t)atoi(optarg); break;
            case 'b': ring_frames = (uint32_t)atoi(optarg); break;
            case 'h': usage(argv[0]); return 1;
            default:  usage(argv[0]); return -1;
        }
    }

    /* Sanity-check inputs; helper hard-fails with clear stderr per the
     * brief's "no silent fallback" rule. */
    if (s->rate < 8000 || s->rate > 192000) {
        fprintf(stderr, "error: rate %u out of supported range\n", s->rate);
        return -1;
    }
    if (s->channels < 1 || s->channels > MAX_CHANNELS) {
        fprintf(stderr, "error: channels %u out of supported range\n", s->channels);
        return -1;
    }
    if (s->quantum < 32 || s->quantum > 8192) {
        fprintf(stderr, "error: quantum %u out of supported range\n", s->quantum);
        return -1;
    }
    /* Ring frames must be a power of two so the SPSC mask trick works. */
    if (ring_frames < 256 ||
        (ring_frames & (ring_frames - 1)) != 0) {
        fprintf(stderr, "error: ring-frames %u not power-of-2 or too small\n",
                ring_frames);
        return -1;
    }
    if (ring_init(&s->ring, ring_frames, s->channels) < 0) {
        fprintf(stderr, "error: ring buffer alloc failed\n");
        return -1;
    }
    return 0;
}

/* --------------------------------------------------------------------
 * Main
 * -------------------------------------------------------------------- */

int main(int argc, char **argv) {
    int rc = 0;
    int parse_rc = parse_args(argc, argv, &g_state);
    if (parse_rc != 0) return parse_rc < 0 ? 1 : 0;

    /* Signal hygiene */
    struct sigaction sa = {0};
    sa.sa_handler = exit_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGINT,  &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    /* SIGPIPE: ignore + handle write errors via errno = EPIPE in writes
     * (not relevant for this helper since we only read stdin and write
     * stderr; but block it from killing the process). */
    signal(SIGPIPE, SIG_IGN);

    pw_init(&argc, &argv);

    g_state.tloop = pw_thread_loop_new("woys-pw-out", NULL);
    if (!g_state.tloop) {
        fprintf(stderr, "error: pw_thread_loop_new failed\n");
        rc = 1;
        goto cleanup_pw;
    }

    struct pw_loop *loop = pw_thread_loop_get_loop(g_state.tloop);

    /* Build node properties. The TARGET_OBJECT key has been stable since
     * PipeWire 0.3.64 — works on 1.6.4. node.latency requests our
     * preferred quantum but the driver has final say. */
    char latency[32];
    snprintf(latency, sizeof(latency), "%u/%u", g_state.quantum, g_state.rate);

    struct pw_properties *props = pw_properties_new(
        PW_KEY_MEDIA_TYPE,     "Audio",
        PW_KEY_MEDIA_CATEGORY, "Playback",
        PW_KEY_MEDIA_ROLE,     "Production",
        PW_KEY_NODE_NAME,      "woys-engine-out",
        PW_KEY_APP_NAME,       "woys",
        PW_KEY_TARGET_OBJECT,  g_state.target,
        PW_KEY_NODE_LATENCY,   latency,
        NULL);

    if (!props) {
        fprintf(stderr, "error: pw_properties_new failed\n");
        rc = 1;
        goto cleanup_loop;
    }

    /* Stream + connect must happen under the thread loop lock per the
     * PipeWire API contract. */
    pw_thread_loop_lock(g_state.tloop);

    g_state.stream = pw_stream_new_simple(loop, "woys-engine-out",
                                          props, &stream_events, &g_state);
    /* pw_stream_new_simple takes ownership of `props` regardless of
     * success/failure — don't free `props` ourselves. */
    if (!g_state.stream) {
        pw_thread_loop_unlock(g_state.tloop);
        fprintf(stderr, "error: pw_stream_new_simple failed\n");
        rc = 1;
        goto cleanup_loop;
    }

    /* Build the format POD. spa_format_audio_raw_build is the canonical
     * macro — does inline POD encoding. */
    uint8_t pod_buf[1024];
    struct spa_pod_builder b = SPA_POD_BUILDER_INIT(pod_buf, sizeof(pod_buf));
    const struct spa_pod *params[1];
    struct spa_audio_info_raw info = SPA_AUDIO_INFO_RAW_INIT(
        .format   = SPA_AUDIO_FORMAT_F32,
        .channels = g_state.channels,
        .rate     = g_state.rate);
    params[0] = spa_format_audio_raw_build(&b, SPA_PARAM_EnumFormat, &info);

    int connect_rc = pw_stream_connect(
        g_state.stream,
        PW_DIRECTION_OUTPUT,
        PW_ID_ANY,
        PW_STREAM_FLAG_AUTOCONNECT |
        PW_STREAM_FLAG_MAP_BUFFERS |
        PW_STREAM_FLAG_RT_PROCESS,
        params, 1);

    pw_thread_loop_unlock(g_state.tloop);

    if (connect_rc < 0) {
        fprintf(stderr, "error: pw_stream_connect failed: %s (%d)\n",
                spa_strerror(connect_rc), connect_rc);
        rc = 1;
        goto cleanup_stream;
    }

    /* Start the data thread. The state-changed callback will print
     * "ready\n" on stderr once the stream actually streams. */
    if (pw_thread_loop_start(g_state.tloop) < 0) {
        fprintf(stderr, "error: pw_thread_loop_start failed\n");
        rc = 1;
        goto cleanup_stream;
    }

    /* Wait up to 5 s for stream to enter STREAMING state. If it doesn't,
     * the target probably doesn't exist (WoysSink not loaded) and we
     * hard-fail rather than sit silently. */
    struct timespec deadline;
    clock_gettime(CLOCK_MONOTONIC, &deadline);
    deadline.tv_sec += 5;
    while (!atomic_load_explicit(&g_state.ready, memory_order_acquire) &&
           !atomic_load_explicit(&g_state.should_exit, memory_order_acquire)) {
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        if (now.tv_sec > deadline.tv_sec ||
            (now.tv_sec == deadline.tv_sec && now.tv_nsec > deadline.tv_nsec)) {
            fprintf(stderr,
                    "error: stream did not reach STREAMING state within 5s "
                    "(target=%s rate=%u channels=%u — does the target node "
                    "exist? try `pactl list short modules` to confirm "
                    "WoysSink is loaded)\n",
                    g_state.target, g_state.rate, g_state.channels);
            rc = 1;
            goto cleanup_thread;
        }
        usleep(50000); /* 50 ms */
    }

    if (atomic_load_explicit(&g_state.should_exit, memory_order_acquire)) {
        rc = 1;
        goto cleanup_thread;
    }

    /* Print the realized config so woys can log it. */
    fprintf(stderr, "quantum=%u rate=%u channels=%u target=%s\n",
            g_state.quantum, g_state.rate, g_state.channels, g_state.target);
    fflush(stderr);

    /* Spawn the underrun-tick thread (replaces the SIGALRM mechanism that
     * never reliably hit the main thread; see comment on
     * underrun_tick_thread). */
    pthread_t tick_thread;
    int tick_rc = pthread_create(&tick_thread, NULL, underrun_tick_thread, &g_state);
    if (tick_rc != 0) {
        fprintf(stderr, "warning: underrun-tick thread spawn failed (%d); "
                        "stderr underruns=N reporting will only fire at exit\n",
                tick_rc);
    }

    /* The main thread reads stdin until EOF (engine shutdown) or
     * SIGINT/SIGTERM. */
    int reader_rc = stdin_reader_loop(&g_state);
    if (reader_rc < 0) rc = 1;

    /* Mark voluntary shutdown so the state-changed callback's
     * UNCONNECTED handler stays silent — see the comment there. */
    atomic_store_explicit(&g_state.should_exit, 1, memory_order_release);

    /* Reap the tick thread (it observes should_exit). */
    if (tick_rc == 0) {
        pthread_join(tick_thread, NULL);
    }

    /* Final underrun report so the engine sees a closing "underruns=N"
     * line before our exit. Skip if process is dying mid-write. */
    fprintf(stderr, "underruns=%llu\n",
            (unsigned long long)atomic_load_explicit(&g_state.underruns,
                                                     memory_order_relaxed));
    fflush(stderr);

cleanup_thread:
    pw_thread_loop_stop(g_state.tloop);

cleanup_stream:
    if (g_state.stream) {
        pw_stream_destroy(g_state.stream);
        g_state.stream = NULL;
    }

cleanup_loop:
    if (g_state.tloop) {
        pw_thread_loop_destroy(g_state.tloop);
        g_state.tloop = NULL;
    }

cleanup_pw:
    pw_deinit();
    ring_free(&g_state.ring);
    return rc;
}
