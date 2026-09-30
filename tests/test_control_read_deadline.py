"""A slow client can't hold a control worker for longer than a second.

The per-recv timeout alone let a client that trickles one byte every
0.3 s keep a worker for hours; four of them starved the pool and every
`woys toggle` / `status` hung.
"""

from __future__ import annotations

import socket
import threading
import time
from pathlib import Path


def test_trickling_clients_do_not_starve_the_pool(tmp_path: Path) -> None:
    from tui.control import ControlServer

    path = tmp_path / "control.sock"
    srv = ControlServer(lambda cmd: f"OK {cmd}", path=path)
    srv.start()
    stop = threading.Event()
    tricklers: list[socket.socket] = []

    def trickle(s: socket.socket) -> None:
        while not stop.is_set():
            try:
                s.sendall(b"A")
            except OSError:
                return
            time.sleep(0.3)

    try:
        for _ in range(ControlServer._WORKER_POOL_SIZE):
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.connect(str(path))
            tricklers.append(s)
            threading.Thread(target=trickle, args=(s,), daemon=True).start()
        time.sleep(0.2)

        t0 = time.monotonic()
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as c:
            c.settimeout(5.0)
            c.connect(str(path))
            c.sendall(b"STATUS\n")
            reply = c.recv(256).decode()
        elapsed = time.monotonic() - t0
    finally:
        stop.set()
        for s in tricklers:
            s.close()
        srv.stop()

    assert reply.strip() == "OK STATUS"
    assert elapsed < 3.0
