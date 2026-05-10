"""Contract tests for scripts/download_weights.py.

These guard the corr-001 class of bug: the engine defaults to a foundation
file the installer doesn't fetch. The test asserts every model the engine
loads at startup is also produced by `download_weights.WEIGHTS`.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def download_weights():
    return _load_module("download_weights_under_test", SCRIPTS / "download_weights.py")


@pytest.fixture(scope="module")
def engine_defaults():
    # Imported lazily so the test runs without ORT loaded.
    if str(REPO / "src") not in sys.path:
        sys.path.insert(0, str(REPO / "src"))
    if str(REPO / "src" / "server") not in sys.path:
        sys.path.insert(0, str(REPO / "src" / "server"))
    from audio import engine  # type: ignore[import-untyped]

    return {
        "rmvpe": engine.DEFAULT_RMVPE.name,
        "contentvec": engine.DEFAULT_CONTENTVEC.name,
        "rvc": engine.DEFAULT_RVC_MODEL.name,
    }


def test_weights_dict_covers_engine_defaults(download_weights, engine_defaults) -> None:
    """The corr-001 contract: download_weights.WEIGHTS must include every
    file that EngineConfig defaults reach for at startup. If you add a new
    foundation model to the engine, you MUST add it here."""
    weights_names = set(download_weights.WEIGHTS.keys())
    missing = []
    for label, name in engine_defaults.items():
        if name not in weights_names:
            missing.append(f"{label} default '{name}' is not in WEIGHTS")
    assert not missing, "engine startup will crash on a fresh install:\n" + "\n".join(missing)


def test_weights_urls_resolve_to_huggingface(download_weights) -> None:
    """All foundation weights live on HuggingFace; sanity-check the URL
    shape so a bad copy-paste fails the test, not the user's fresh install.
    The local filename and the upstream filename can differ (we rename on
    save) - only require that the URL is on HF and the URL contains a
    distinguishing keyword from the local name."""
    for name, url in download_weights.WEIGHTS.items():
        assert url.startswith("https://huggingface.co/"), (
            f"{name} URL is not on huggingface.co: {url}"
        )
        # First "stem token" of the local filename (e.g. rmvpe_wrapped → rmvpe)
        # must appear somewhere in the URL.
        keyword = name.split("_")[0].split(".")[0].split("-")[0]
        assert keyword in url, f"{name} URL doesn't mention '{keyword}': {url}"


def test_sha256_table_keys_are_subset_of_weights(download_weights) -> None:
    """If WEIGHTS_SHA256 has entries, they must all reference real
    WEIGHTS keys (typo-guard)."""
    sha_keys = set(download_weights.WEIGHTS_SHA256.keys())
    weight_keys = set(download_weights.WEIGHTS.keys())
    extra = sha_keys - weight_keys
    assert not extra, f"WEIGHTS_SHA256 references unknown files: {extra}"


def test_print_hashes_does_not_crash_on_missing_cache(
    download_weights, monkeypatch, capsys
) -> None:
    """`--print-hashes` should report missing files to stderr without raising,
    so a fresh-checkout dev can still inspect what would be hashed."""
    nonexistent = Path("/tmp/woys-test-cache-does-not-exist-12345")
    monkeypatch.setattr(download_weights, "CACHE", nonexistent)
    rc = download_weights.main(["--print-hashes"])
    assert rc == 0
