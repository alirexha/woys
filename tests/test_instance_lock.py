"""v0.14.0 (Lens 17 / C009): single-instance flock tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Local import - `woys.instance_lock` lives under src/.
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from woys.instance_lock import InstanceLockBusy, acquire_instance_lock  # noqa: E402


@pytest.fixture
def fake_runtime_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point XDG_RUNTIME_DIR at a tmpdir so the test never touches the
    user's real lock file."""
    fake = tmp_path / "runtime"
    fake.mkdir()
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(fake))
    return fake / "woys"


def test_acquire_creates_lock_file_with_pid(fake_runtime_dir: Path) -> None:
    """First acquire creates the lock file and writes our PID inside."""
    with acquire_instance_lock() as lock_path:
        assert lock_path == fake_runtime_dir / "instance.lock"
        assert lock_path.exists()
        # File is mode 0600 (hygiene per Lens 6).
        assert lock_path.stat().st_mode & 0o777 == 0o600
        # PID written inside.
        with open(lock_path) as fh:
            content = fh.read().strip()
        assert content == str(os.getpid())


def test_acquire_releases_on_exit(fake_runtime_dir: Path) -> None:
    """A fresh acquire after release succeeds (lock was actually let go)."""
    with acquire_instance_lock():
        pass
    # Should re-acquire without raising.
    with acquire_instance_lock():
        pass


def test_concurrent_acquire_raises_busy(fake_runtime_dir: Path) -> None:
    """v0.14.0 (C009): the second concurrent acquire MUST raise
    InstanceLockBusy. This is the regression test for the
    "two concurrent woys instances corrupt the control socket"
    bug class."""
    cm1 = acquire_instance_lock()
    cm1.__enter__()
    try:
        with pytest.raises(InstanceLockBusy) as exc_info, acquire_instance_lock():
            pass  # pragma: no cover - lock should reject
        # Error mentions the holder PID for diagnostics.
        assert str(os.getpid()) in str(exc_info.value)
    finally:
        cm1.__exit__(None, None, None)


def test_run_tui_acquires_instance_lock(
    fake_runtime_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """review F-merged-002 (P0): `woys run` -- the TUI entry point --
    MUST acquire the single-instance lock.

    Pre-fix the lock was wired only into `woys engine` (cmd_engine);
    `run_tui` constructed and ran WoysApp unconditionally, so two
    `woys run` instances both bound the control socket and wrote into
    WoysSink (the reproduced double-converted-audio corruption).

    Regression test: with the lock already held, run_tui must return exit
    code 2 *without* constructing WoysApp. On pre-fix code run_tui ignores
    the held lock and reaches WoysApp(...), tripping the AssertionError.
    """
    import tui.app as app_mod
    from tui.config import AppConfig

    # Isolate from the user's real config; keep the test hermetic.
    monkeypatch.setattr(app_mod, "load_config", lambda: AppConfig())

    constructed: list[object] = []

    def _fake_woys_app(**kwargs: object) -> object:
        constructed.append(kwargs)
        raise AssertionError(
            "WoysApp was constructed even though the instance lock is held - "
            "run_tui did not acquire the lock (F-merged-002 regression)"
        )

    monkeypatch.setattr(app_mod, "WoysApp", _fake_woys_app)

    with acquire_instance_lock():
        rc = app_mod.run_tui()

    assert rc == 2
    assert constructed == [], "run_tui must reject the busy lock before building WoysApp"


def test_xdg_unset_falls_back_to_tmp(monkeypatch: pytest.MonkeyPatch) -> None:
    """No XDG_RUNTIME_DIR -> /tmp/woys-{uid}/instance.lock. The lock
    is still acquirable; cleans up after the test."""
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    expected = Path(f"/tmp/woys-{os.getuid()}/instance.lock")
    try:
        with acquire_instance_lock() as lock_path:
            assert lock_path == expected
            assert lock_path.exists()
    finally:
        # Best-effort cleanup so subsequent test runs see clean state.
        if expected.exists():
            expected.unlink()
