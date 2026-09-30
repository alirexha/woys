"""Importing the vendored exporter must not litter the repo with ./tmp_dir.

`src/server/const.py` runs `os.makedirs("tmp_dir")` relative to the CWD at
import, and `woys.convert` / the fp16-gate tests import it, so a test run
from the repo root left `./tmp_dir` behind. conftest imports it once from a
throwaway directory and points `TMP_DIR` there.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_vendored_tmp_dir_is_outside_the_checkout() -> None:
    import const  # type: ignore[import-not-found]

    tmp = Path(const.TMP_DIR)
    assert tmp.is_absolute(), f"TMP_DIR {tmp} would resolve against the CWD"
    assert not tmp.resolve().is_relative_to(REPO)
    assert tmp.is_dir()
