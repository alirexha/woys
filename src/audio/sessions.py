"""ONNX Runtime session construction for the engine.

Loading this module preloads the pip-shipped CUDA / cuDNN libraries (and
TensorRT's, when installed) -- the same import-time step `audio.engine`
always performed, before any session exists. `_make_session` builds a
TRT -> CUDA session and hard-fails a CPU-only binding (`CpuFallbackError`);
`RvcSessionPool` caches voice-model sessions so a hot-swap is a pointer
swap. Split out of `audio.engine` (which re-exports these names).
"""

from __future__ import annotations

import contextlib
import os
import threading
from pathlib import Path

import numpy as np

# ORT-GPU 1.20+ on driver 595 needs explicit preload of the pip-shipped CUDA libs.
import onnxruntime as ort

if hasattr(ort, "preload_dlls"):
    ort.preload_dlls()


def _preload_trt_dlls() -> bool:
    """v0.8.1 - preload TensorRT shared libraries so ORT's TRT EP can
    resolve `libnvinfer.so.10`. The pip-installed `tensorrt-cu12`
    package puts its libs under
    `<venv>/lib/python3.11/site-packages/tensorrt_libs/` which isn't
    on the system loader path, so ORT (which dlopens
    `libonnxruntime_providers_tensorrt.so` and that in turn dlopens
    `libnvinfer.so.10`) fails with "cannot open shared object" unless
    we ctypes-preload the .so files into the process's symbol space
    first.

    Returns True if every libnvinfer*.so was loaded successfully,
    False otherwise (TRT EP will silently fall through to CUDA EP
    in that case - the per-session providers list always includes
    CUDA EP as a fallback).
    """
    import ctypes

    try:
        import tensorrt_libs
    except ImportError:
        return False

    libs_dir = os.path.dirname(tensorrt_libs.__file__)
    ok = True
    for fn in sorted(os.listdir(libs_dir)):
        # Only load the libnvinfer* shims; other files in tensorrt_libs
        # (Python source, init helpers) shouldn't be ctypes-loaded.
        if "libnvinfer" not in fn or ".so" not in fn:
            continue
        full = os.path.join(libs_dir, fn)
        try:
            ctypes.CDLL(full, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            ok = False
    return ok


# Best-effort TRT preload at module import. Failure is non-fatal -
# session creation falls back to CUDA EP if TRT can't be initialized.
_TRT_PRELOAD_OK = _preload_trt_dlls()


# v0.7.0-rc10: HEURISTIC → EXHAUSTIVE. The rc8 tail-chunk capture +
# rc9 broader pre-warm together pinned the inference p99 spike to
# cuDNN heuristic algo selection: even after rc9 pre-warmed every
# audio16_len soxr emits (1957/1958/2446/2447), p99 stayed at ~96 ms.
# rc9's tail log showed two distinct slow patterns - `rvc_ms` 64-72
# ms (one shape group) and `rvc_ms` 47-48 ms + `rmvpe_ms` 17 ms
# (other shape group). The heuristic was picking different,
# intrinsically slower, algos for the alternating shapes.
#
# v0.7.0-rc1's pre-rejection of EXHAUSTIVE was based on the autotune
# lump: a 50-100 ms one-time cost per first-encounter shape. At
# chunk_seconds=0.10 / 0.15, paying that cost mid-realtime made the
# first 5-10 chunks miss budget. rc9's broader pre-warm changes
# that calculation: the autotune lump is now paid during warmup
# (engine.start() before _run_loop), not realtime. Net startup
# cost: another ~0.5-1 s on top of rc9's already-extended warmup.
# Acceptable trade for letting cuDNN pick the FASTEST algo per
# shape rather than a heuristic guess.
#
# v0.2.0 - v0.7.0-rc9 history preserved for context:
#   v0.2.0 default - picks fastest steady-state algo per shape but
#   eats 50-100 ms autotune the first time each shape lands.
#   HEURISTIC (rc1+) picked a near-optimal algo from a heuristic
#   without any timed search - slightly slower steady-state but no
#   autotune lump.
#
# Setting can still be flipped back via the env var if EXHAUSTIVE
# regresses or if a future ORT release improves HEURISTIC.
_CUDNN_ALGO_SEARCH = "EXHAUSTIVE"


_TRT_CACHE_ROOT = Path.home() / ".cache" / "woys" / "trt"


def _trt_cache_dir_for(model_path: Path) -> Path:
    """Per-model TRT engine cache directory under
    `~/.cache/woys/trt/<model-stem>/`. Engines for different shapes
    of the same model land in the same directory, keyed by ORT's
    internal shape-aware hash. Different models keep separate
    subdirs so cache invalidation per-model is just `rm -rf`.
    """
    d = _TRT_CACHE_ROOT / model_path.stem
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cuda_provider_entry() -> tuple[str, dict[str, object]]:
    """The CUDA EP config we use everywhere - extracted so the TRT
    fallback path can pull the same options if TRT init fails."""
    return (
        "CUDAExecutionProvider",
        {
            "device_id": 0,
            # v0.7.0-rc12: kNextPowerOfTwo → kSameAsRequested.
            # See engine history below for full context.
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_algo_search": _CUDNN_ALGO_SEARCH,
            "do_copy_in_default_stream": True,
            # B54 / corr-023: bool, not the string "1". ORT's CUDA EP option
            # parser accepts both, but every other entry in this dict uses
            # native types - be consistent.
            "cudnn_conv_use_max_workspace": True,
        },
    )


# Module-level record of which sessions actually got TRT EP. Surfaced
# in `EngineStats.trt_active_for` so woys diag can show which models
# failed TRT init and fell back to CUDA - gives the user one place to
# see the real picture without grepping logs.
_TRT_ACTIVE_PER_SESSION: dict[str, bool] = {}


_TRT_INIT_ERRORS: dict[str, str] = {}


class CpuFallbackError(RuntimeError):
    """ONNX Runtime bound a session CPU-only: either the CUDA execution
    provider failed to bind, or this onnxruntime build has none at all.

    Realtime RVC on CPU runs ~10-50x over the chunk-period latency budget,
    so a CPU-bound session is a non-functional state, not a degraded one.
    Pre-fix `_make_session` appended `CPUExecutionProvider` as an
    unconditional landing pad and never checked `get_providers()`, so a
    broken onnxruntime-gpu wheel / NVIDIA driver / missing `preload_dlls()`
    produced a working-looking but unusable engine with no error anywhere.
    """


def _session_is_cpu_only(sess: ort.InferenceSession) -> bool:
    """True iff the session's active (first) provider is CPUExecutionProvider."""
    bound = sess.get_providers()
    return bool(bound) and bound[0] == "CPUExecutionProvider"


def _assert_session_gpu_bound(
    sess: ort.InferenceSession,
    path: Path,
    *,
    available: list[str],
    allow_cpu_fallback: bool,
) -> None:
    """Hard-fail a CPU-bound session (F-merged-001).

    Raise `CpuFallbackError` when the session bound CPU-only, unless the
    test-only `allow_cpu_fallback` is set (the engine then records it in
    `EngineStats.cpu_fallback_active`, printed by `woys diag`).

    A build with no CUDA EP at all -- typically a CPU `onnxruntime` wheel
    shadowing onnxruntime-gpu -- fails too: the session runs on CPU just
    the same, and woys is GPU-only.
    """
    if not _session_is_cpu_only(sess) or allow_cpu_fallback:
        return
    if "CUDAExecutionProvider" not in available:
        raise CpuFallbackError(
            f"{path.name}: this onnxruntime build has no CUDA execution "
            f"provider (available={available}), so the session would run "
            f"CPU-only and realtime RVC is unusable on CPU. woys needs "
            f"onnxruntime-gpu: a CPU `onnxruntime` wheel is probably "
            f"shadowing it. Uninstall onnxruntime and reinstall "
            f"onnxruntime-gpu (or re-run install.sh), then check `woys info`."
        )
    raise CpuFallbackError(
        f"{path.name}: a CUDA execution provider is installed but ONNX "
        f"Runtime bound this session CPU-only (providers="
        f"{sess.get_providers()}). Realtime RVC is unusable on CPU. "
        f"Check the onnxruntime-gpu wheel, the NVIDIA driver, and that "
        f"ort.preload_dlls() ran."
    )


def _make_session(
    path: Path, *, use_tensorrt: bool = True, allow_cpu_fallback: bool = False
) -> ort.InferenceSession:
    """v0.8.1 - try TensorRT EP first, fall back to CUDA EP per session.

    ORT's TRT EP fails session initialization (not just the TRT
    subgraph) when it encounters operators it can't handle -
    e.g. RMVPE's FP16 STFT, which TRT requires to be FP32. We
    catch that failure and rebuild the session with CUDA EP only.
    The fallback is logged to `_TRT_INIT_ERRORS[path.name]` and
    can be surfaced via `EngineStats.trt_active_for` and woys diag.

    the CUDA->CPU fallback is *not* silent.
    After the session is built, `_assert_session_gpu_bound` raises
    `CpuFallbackError` if ORT bound it CPU-only -- including on a build
    with no CUDA EP at all -- unless the test-only `allow_cpu_fallback`
    is set.
    """
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    available = ort.get_available_providers()

    if use_tensorrt and _TRT_PRELOAD_OK and "TensorrtExecutionProvider" in available:
        cache_dir = _trt_cache_dir_for(path)
        trt_providers: list[tuple[str, dict[str, object]] | str] = [
            (
                "TensorrtExecutionProvider",
                {
                    "device_id": 0,
                    # Cache engines to disk so the 5-30 s per-shape
                    # compile cost is paid only on the first session
                    # ever (or when the model file changes).
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": str(cache_dir),
                    # 2 GiB workspace for TRT's builder. RTX 2070 has
                    # 8 GiB; cv + rmvpe + rvc resident is ~2 GiB.
                    "trt_max_workspace_size": 2 * 1024 * 1024 * 1024,
                    # FP16 disabled until per-voice quality validation
                    # (cosine sim ≥ 0.95 vs FP32 baseline) is run.
                    "trt_fp16_enable": False,
                    "trt_max_partition_iterations": 1000,
                    "trt_min_subgraph_size": 1,
                    "trt_timing_cache_enable": True,
                    "trt_timing_cache_path": str(cache_dir),
                },
            ),
            _cuda_provider_entry(),
            "CPUExecutionProvider",
        ]
        try:
            sess = ort.InferenceSession(str(path), sess_options=so, providers=trt_providers)
            _assert_session_gpu_bound(
                sess, path, available=available, allow_cpu_fallback=allow_cpu_fallback
            )
            _TRT_ACTIVE_PER_SESSION[path.name] = True
            _TRT_INIT_ERRORS.pop(path.name, None)
            return sess
        except CpuFallbackError:
            # A CpuFallbackError means TRT/CUDA both failed to bind and the
            # session is CPU-only -- that is the F-merged-001 hard-fail, not
            # a "TRT couldn't partition the graph" retry condition. Re-raise;
            # rebuilding with the CUDA-only providers below would just bind
            # CPU again.
            raise
        except Exception as e:
            # TRT couldn't parse / partition the graph. Log the
            # reason and retry with CUDA EP only. Common cause:
            # graph contains an operator TRT doesn't support
            # (FP16 STFT, certain custom ops, dynamic shapes
            # without shape inference annotations).
            _TRT_ACTIVE_PER_SESSION[path.name] = False
            _TRT_INIT_ERRORS[path.name] = f"{type(e).__name__}: {str(e)[:240]}"

    # CUDA EP only path (TRT disabled or TRT init failed).
    cuda_providers: list[tuple[str, dict[str, object]] | str] = []
    if "CUDAExecutionProvider" in available:
        cuda_providers.append(_cuda_provider_entry())
    cuda_providers.append("CPUExecutionProvider")
    sess = ort.InferenceSession(str(path), sess_options=so, providers=cuda_providers)
    _assert_session_gpu_bound(
        sess, path, available=available, allow_cpu_fallback=allow_cpu_fallback
    )
    _TRT_ACTIVE_PER_SESSION.setdefault(path.name, False)
    return sess


class RvcSessionPool:
    """Per-path cache of `ort.InferenceSession` objects.

    Hot-swap performance was the v0.4.x P0: every `models use` rebuilt the
    session from scratch, including cudnn EXHAUSTIVE algo-tuning, costing
    ~1.5 s + a 305 ms first-chunk inference burst. This pool keeps a small
    set of cached sessions; second swap to an already-seen voice is a
    pointer swap (~10 ms total).

    LRU eviction keeps VRAM bounded - a session uses ~150 MiB resident,
    so the default `max_size=4` caps voice-model VRAM at ~600 MiB on top
    of the foundations. Configurable via `EngineConfig.session_pool_size`.

    Thread-safe. The audio worker calls `get_or_create()` from inside
    `_maybe_swap_model`; tests / TUI may call it from any thread.
    """

    def __init__(
        self,
        max_size: int = 4,
        *,
        use_tensorrt: bool = True,
        allow_cpu_fallback: bool = False,
    ) -> None:
        self._cache: dict[Path, ort.InferenceSession] = {}
        self._access_order: list[Path] = []
        self._max_size = max(1, max_size)
        self._lock = threading.Lock()
        # v0.8.1: pool sessions inherit the engine's TRT preference. The
        # engine constructs the pool with `use_tensorrt=cfg.use_tensorrt`
        # so RVC voice loads share whatever EP path the engine wants.
        self._use_tensorrt = use_tensorrt
        # RVC voice sessions go through the
        # same CUDA->CPU silent-fallback hard-fail as cv/rmvpe.
        self._allow_cpu_fallback = allow_cpu_fallback

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    def __contains__(self, path: Path) -> bool:
        with self._lock:
            return Path(path).resolve() in self._cache

    def get_or_create(self, path: Path) -> ort.InferenceSession:
        """Return a cached session if present, else create + cache.

        Cache hit: ~0.1 ms. Cache miss: ~600 ms (model load + cudnn tune).
        """
        key = Path(path).resolve()
        with self._lock:
            if key in self._cache:
                # Bump LRU.
                self._access_order.remove(key)
                self._access_order.append(key)
                return self._cache[key]

        # Cache miss - build outside the lock (slow); other threads can
        # still get cached sessions while we tune.
        sess = _make_session(
            key,
            use_tensorrt=self._use_tensorrt,
            allow_cpu_fallback=self._allow_cpu_fallback,
        )

        with self._lock:
            # Another thread may have raced us; if so, drop ours and use theirs.
            if key in self._cache:
                return self._cache[key]
            self._cache[key] = sess
            self._access_order.append(key)
            # B26 / corr-006: the pre-v0.8.0 `if evicted != key` guard was
            # dead - `key` was just appended at -1, the pop comes from index
            # 0, so evicted == key only when len == 1 (and then we DO want
            # to evict, never reaching this branch). Just unconditionally
            # drop.
            while len(self._access_order) > self._max_size:
                self._cache.pop(self._access_order.pop(0), None)
        return sess

    def warmup(self, path: Path) -> ort.InferenceSession:
        """Create + run one dummy forward pass so cudnn populates its algo
        cache. Subsequent inferences against the same shape are near-instant.

        The caller is expected to know the model's input shape - we feed the
        widest plausible RVC v2 input (768-dim feats x 100 frames).
        """
        sess = self.get_or_create(path)
        try:
            shape = sess.get_inputs()[0].shape
            feats_dim = int(shape[2]) if len(shape) >= 3 and isinstance(shape[2], int) else 768
        except (IndexError, ValueError):
            feats_dim = 768
        is_half = sess.get_inputs()[0].type != "tensor(float)"
        feats_dt = np.float16 if is_half else np.float32
        n_frames = 100
        feed = {
            "feats": np.zeros((1, n_frames, feats_dim), dtype=feats_dt),
            "p_len": np.array([n_frames], dtype=np.int64),
            "pitch": np.zeros((1, n_frames), dtype=np.int64),
            "pitchf": np.zeros((1, n_frames), dtype=np.float32),
            "sid": np.array([0], dtype=np.int64),
        }
        with contextlib.suppress(Exception):
            sess.run(["audio"], feed)
        return sess

    def evict_all(self) -> None:
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
