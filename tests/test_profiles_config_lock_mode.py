"""config.toml.lock is created 0600, like config.toml itself."""

from __future__ import annotations

import os
import stat
from collections.abc import Iterator

import pytest


@pytest.fixture
def loose_umask() -> Iterator[None]:
    prior = os.umask(0o022)
    try:
        yield
    finally:
        os.umask(prior)


def test_config_lock_file_is_private(loose_umask: None) -> None:
    import tui.config as tc
    from woys.profiles import config_lock

    lock_path = tc.CONFIG_FILE.with_suffix(".toml.lock")
    assert not lock_path.exists()
    with config_lock():
        pass

    assert stat.S_IMODE(lock_path.stat().st_mode) == 0o600
