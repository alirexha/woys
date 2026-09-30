"""The "loading X…" indicator stays up while any swap job is running.

Each MODEL / PROFILE / `p` job cleared the shared indicator when it
finished, so with two overlapping jobs the first to finish hid the one
still loading.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any


def _wait(pred: Any, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while not pred():
        assert time.monotonic() < deadline
        time.sleep(0.01)


def test_indicator_survives_the_first_of_two_overlapping_swaps(tmp_path: Path) -> None:
    from audio.engine import _SwapRequest
    from tui.app import WoysApp
    from tui.config import AppConfig

    a = tmp_path / "a.onnx"
    b = tmp_path / "b.onnx"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    app = WoysApp(cfg=AppConfig(), no_pw_setup=True)
    app.call_from_thread = lambda fn, *x, **k: fn(*x, **k)  # type: ignore[method-assign]
    app.notify = lambda *_a, **_k: None  # type: ignore[method-assign]
    app._save_cfg = lambda: True  # type: ignore[method-assign]
    reqs: dict[str, _SwapRequest] = {}

    def request_model_swap(p: Path) -> _SwapRequest:
        reqs[p.name] = _SwapRequest(target=p)
        return reqs[p.name]

    app.engine.request_model_swap = request_model_swap  # type: ignore[method-assign]

    app._handle_control(f"MODEL {a}")
    _wait(lambda: "a.onnx" in reqs)
    app._handle_control(f"MODEL {b}")
    _wait(lambda: "b.onnx" in reqs)
    assert app._swap_in_flight == "b.onnx"

    reqs["b.onnx"].completion.set()
    time.sleep(0.1)
    assert app._swap_in_flight == "a.onnx"

    reqs["a.onnx"].completion.set()
    _wait(lambda: app._swap_in_flight is None)
