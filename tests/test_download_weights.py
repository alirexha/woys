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


# ---- review F-merged-003: integrity gate must be real + fail-closed ----


class _FakeResponse:
    """Minimal stand-in for `urllib.request.urlopen(...)`'s context manager."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False

    def read(self, n: int = -1) -> bytes:
        end = self._pos + n if n and n > 0 else len(self._data)
        chunk = self._data[self._pos : end]
        self._pos += len(chunk)
        return chunk


def test_sha256_table_is_populated_for_every_foundation_weight(download_weights) -> None:
    """F-merged-003: the table used to ship EMPTY, making the SHA256 check
    dead code. Every weight in WEIGHTS must now have a pinned hash."""
    missing = set(download_weights.WEIGHTS) - set(download_weights.WEIGHTS_SHA256)
    assert not missing, f"WEIGHTS_SHA256 is missing entries for: {missing}"
    for name, h in download_weights.WEIGHTS_SHA256.items():
        assert len(h) == 64 and all(c in "0123456789abcdef" for c in h), (
            f"{name}: not a sha256 hex digest: {h!r}"
        )


def test_fetch_raises_on_sha_mismatch(download_weights, tmp_path, monkeypatch) -> None:
    """A download whose bytes do not match the pinned SHA256 must raise and
    leave no file behind. Pre-fix (empty table) verification was skipped and
    `fetch()` renamed the corrupt `.part` into place -- no raise."""
    monkeypatch.setattr(
        download_weights.urllib.request,
        "urlopen",
        lambda _url, timeout=0: _FakeResponse(b"corrupted not-the-real-weight bytes"),
    )
    dest = tmp_path / "rmvpe_wrapped.onnx"  # a name that IS in WEIGHTS_SHA256
    assert "rmvpe_wrapped.onnx" in download_weights.WEIGHTS_SHA256

    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        download_weights.fetch("http://example/x", dest, force=True)
    assert not dest.exists(), "the corrupt .part must not be renamed into place"


def test_fetch_fail_closed_on_missing_hash_entry(download_weights, tmp_path, monkeypatch) -> None:
    """A foundation weight with no SHA256 entry must be a hard error, not a
    silent unverified install -- fail-open -> fail-closed."""
    monkeypatch.setattr(
        download_weights.urllib.request,
        "urlopen",
        lambda _url, timeout=0: _FakeResponse(b"some bytes"),
    )
    dest = tmp_path / "unlisted_weight.onnx"  # deliberately NOT in WEIGHTS_SHA256
    assert "unlisted_weight.onnx" not in download_weights.WEIGHTS_SHA256

    with pytest.raises(RuntimeError, match="no SHA256 entry"):
        download_weights.fetch("http://example/x", dest, force=True)
    assert not dest.exists()


def test_fetch_skip_verify_bypasses_the_gate(download_weights, tmp_path, monkeypatch) -> None:
    """--skip-verify is the explicit, documented escape hatch -- it must
    still install (the gate is fail-closed, not un-bypassable)."""
    monkeypatch.setattr(
        download_weights.urllib.request,
        "urlopen",
        lambda _url, timeout=0: _FakeResponse(b"whatever bytes"),
    )
    dest = tmp_path / "unlisted_weight.onnx"
    download_weights.fetch("http://example/x", dest, force=True, skip_verify=True)
    assert dest.exists()
