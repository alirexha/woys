"""`woys convert` must not leave a `tmp_dir/` in the user's current directory.

The vendored upstream `const.py` runs `os.makedirs("tmp_dir")` relative to
the CWD at import time, and both the metadata probe and the exporter import
it, so every `woys convert` dropped an empty `tmp_dir/` wherever it was run.

Each case runs in a fresh interpreter: `const` is cached per process, so an
in-process import would see whatever an earlier test (or conftest) did.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# 18-entry config = the official RVC layout the probe recognizes. The
# string entries make upstream's model constructor fail at once, so the
# export case stops right after the exporter import instead of building
# and exporting a real network.
_MAKE_CKPT = """
import torch
torch.save({"config": ["x"] * 17 + [40000], "version": "v2", "f0": 1}, "voice.pth")
"""


def _run_in(work: Path, body: str) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(REPO / "src"), str(REPO / "src" / "server")]),
    }
    script = textwrap.dedent(_MAKE_CKPT) + textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


@pytest.fixture
def work(tmp_path: Path) -> Path:
    d = tmp_path / "work"
    d.mkdir()
    return d


def test_probe_leaves_no_tmp_dir_in_cwd(work: Path) -> None:
    res = _run_in(
        work,
        """
        from pathlib import Path
        from woys import convert
        meta = convert._probe_pth_metadata(Path("voice.pth"))
        print(meta.modelType)
        """,
    )
    assert res.returncode == 0, res.stderr
    assert "pyTorchRVCv2" in res.stdout
    assert sorted(os.listdir(work)) == ["voice.pth"]


def test_convert_leaves_no_tmp_dir_in_cwd(work: Path) -> None:
    res = _run_in(
        work,
        """
        import os
        from pathlib import Path
        from woys import convert
        start = os.getcwd()
        try:
            convert.convert_pth_to_onnx(Path("voice.pth"), Path("out/voice.onnx"))
        except Exception as e:
            print("convert failed as intended:", type(e).__name__)
        assert os.getcwd() == start, os.getcwd()
        """,
    )
    assert res.returncode == 0, res.stderr
    assert "convert failed as intended" in res.stdout, res.stdout + res.stderr
    # Relative output paths still resolve against the user's CWD.
    assert sorted(os.listdir(work)) == ["out", "voice.pth"]
