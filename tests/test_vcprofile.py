"""Tests for the v0.4.0 .vcprofile shareable-preset format."""

from __future__ import annotations

import hashlib
import sys
import tomllib
from pathlib import Path

import pytest


def _write_dummy_onnx(p: Path, payload: bytes = b"\x08\x07ir") -> str:
    """Create a stub file and return its sha256."""
    p.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_export_writes_format_v1_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Export round-trip: create a profile, export to .vcprofile, parse the
    file and verify structure."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

    # Redirect config + models to tmp.
    cfg_path = tmp_path / "config.toml"
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    onnx_path = models_dir / "test_voice.onnx"
    sha = _write_dummy_onnx(onnx_path, b"\x08\x07iramodelweights")

    from tui.config import AppConfig, save_config
    from woys.profiles import save_profile
    from woys.vcprofile import export_profile

    cfg = AppConfig()
    cfg.rvc_model = str(onnx_path)
    cfg.f0_up_key = 7
    cfg.chunk_seconds = 0.15
    save_profile(cfg, "test")
    save_config(cfg, cfg_path)

    monkeypatch.setattr("tui.config.CONFIG_FILE", cfg_path)

    out = export_profile("test", tmp_path / "test.vcprofile")
    assert out.exists()
    assert out.suffix == ".vcprofile"

    with out.open("rb") as f:
        raw = tomllib.load(f)

    assert raw["meta"]["format_version"] == 1
    assert raw["meta"]["profile_name"] == "test"
    assert raw["profile"]["f0_up_key"] == 7
    assert raw["profile"]["chunk_seconds"] == 0.15
    assert "rvc_model" not in raw["profile"]  # absolute path is dropped on export
    assert raw["model"]["filename"] == "test_voice.onnx"
    assert raw["model"]["sha256"] == sha


def test_export_unknown_profile_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    cfg_path = tmp_path / "config.toml"
    monkeypatch.setattr("tui.config.CONFIG_FILE", cfg_path)

    from tui.config import AppConfig, save_config
    from woys.vcprofile import export_profile

    save_config(AppConfig(), cfg_path)
    with pytest.raises(KeyError):
        export_profile("does-not-exist", tmp_path / "x.vcprofile")


def test_import_resolves_model_by_sha(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Import: receiver has a different absolute path but the same SHA-256
    locally; import should rebind to the local file."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

    sender_models = tmp_path / "sender_models"
    sender_models.mkdir()
    voice_payload = b"\x08\x07irpayload-deadbeef"
    sha = _write_dummy_onnx(sender_models / "myvoice.onnx", voice_payload)

    receiver_models = tmp_path / "receiver_models"
    receiver_models.mkdir()
    # Receiver renamed it locally - same content, different name.
    _write_dummy_onnx(receiver_models / "myvoice_renamed.onnx", voice_payload)

    cfg_path = tmp_path / "config.toml"

    from tui.config import AppConfig, load_config, save_config
    from woys.profiles import save_profile
    from woys.vcprofile import export_profile, import_profile

    monkeypatch.setattr("tui.config.CONFIG_FILE", cfg_path)
    monkeypatch.setattr("woys.models.MODELS_DIR", receiver_models)

    cfg = AppConfig()
    cfg.rvc_model = str(sender_models / "myvoice.onnx")
    cfg.f0_up_key = -3
    save_profile(cfg, "shared")
    save_config(cfg, cfg_path)
    bundle = export_profile("shared", tmp_path / "shared.vcprofile")
    assert sha == hashlib.sha256(voice_payload).hexdigest()  # sanity

    # Wipe profiles to simulate a fresh receiver.
    cfg2 = AppConfig()
    save_config(cfg2, cfg_path)

    # Patch discover_models to use the receiver's library.
    from woys import models as models_mod

    real_discover = models_mod.discover_models
    monkeypatch.setattr(
        models_mod,
        "discover_models",
        lambda models_dir=receiver_models: real_discover(models_dir),
    )

    name = import_profile(bundle, "shared")
    assert name == "shared"
    cfg3 = load_config(cfg_path)
    profiles = cfg3._extras.get("profiles", {})
    assert "shared" in profiles
    bound_path = profiles["shared"].get("rvc_model", "")
    assert bound_path != ""
    assert Path(bound_path).resolve() == (receiver_models / "myvoice_renamed.onnx").resolve()
    assert profiles["shared"]["f0_up_key"] == -3


def test_import_missing_model_leaves_rvc_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Receiver has no matching .onnx - import should still succeed but
    leave rvc_model empty + warn."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

    sender_models = tmp_path / "sender"
    sender_models.mkdir()
    receiver_models = tmp_path / "receiver"
    receiver_models.mkdir()  # empty

    _write_dummy_onnx(sender_models / "exotic.onnx", b"\x08\x07iruniquecontent")
    cfg_path = tmp_path / "config.toml"

    from tui.config import AppConfig, load_config, save_config
    from woys.profiles import save_profile
    from woys.vcprofile import export_profile, import_profile

    monkeypatch.setattr("tui.config.CONFIG_FILE", cfg_path)

    cfg = AppConfig()
    cfg.rvc_model = str(sender_models / "exotic.onnx")
    save_profile(cfg, "exotic")
    save_config(cfg, cfg_path)
    bundle = export_profile("exotic", tmp_path / "exotic.vcprofile")

    cfg2 = AppConfig()
    save_config(cfg2, cfg_path)

    from woys import models as models_mod

    real_discover = models_mod.discover_models
    monkeypatch.setattr(
        models_mod,
        "discover_models",
        lambda models_dir=receiver_models: real_discover(models_dir),
    )

    import_profile(bundle, "exotic")
    cfg3 = load_config(cfg_path)
    profiles = cfg3._extras["profiles"]
    assert profiles["exotic"]["rvc_model"] == ""


def test_import_rejects_unsupported_format_version(tmp_path: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    bad = tmp_path / "bad.vcprofile"
    bad.write_text(
        '[meta]\nformat_version = 99\nprofile_name = "x"\n[profile]\n[model]\n', encoding="utf-8"
    )

    from woys.vcprofile import import_profile

    with pytest.raises(ValueError, match="format_version"):
        import_profile(bad, "x")


# --- review F-16-08: .vcprofile forward-compat reader ----------------
# Pre-fix `import_profile` raised on any `format_version` mismatch -- a
# share format whose whole purpose is cross-user / cross-version
# distribution cannot fail-hard on its first revision. The fix adds a
# migration ladder + clearer error messages.


def _write_vcprofile(path: Path, *, format_version: object, profile_name: str = "x") -> None:
    """Synthesize a minimal but valid-shape .vcprofile with a custom
    format_version. Bypasses tomli_w so we can write `format_version =
    "not-an-int"` or other deliberately broken values for the type-
    check test."""
    if isinstance(format_version, int) and not isinstance(format_version, bool):
        fv_lit = str(format_version)
    elif isinstance(format_version, str):
        fv_lit = f'"{format_version}"'
    else:
        fv_lit = repr(format_version)
    path.write_text(
        f'[meta]\nformat_version = {fv_lit}\nprofile_name = "{profile_name}"\n[profile]\n[model]\n',
        encoding="utf-8",
    )


def test_import_newer_than_current_says_upgrade_woys(tmp_path: Path) -> None:
    """Bug-class half-A. A .vcprofile from a future build raises with
    a clear "upgrade woys" message, not a generic 'unsupported'.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from woys.vcprofile import VCPROFILE_VERSION, import_profile

    bad = tmp_path / "from_the_future.vcprofile"
    _write_vcprofile(bad, format_version=VCPROFILE_VERSION + 1)

    with pytest.raises(ValueError, match=r"newer build|Upgrade woys"):
        import_profile(bad, "x")


def test_import_older_than_current_with_no_migration_explains_what_is_missing(
    tmp_path: Path,
) -> None:
    """When a v(current-1) file arrives and the ladder has no
    registered v(current-1) -> v(current) migration, the error names
    the missing leg so a future maintainer knows what to add.
    Today the ladder is empty so v(0) is the test case."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from woys.vcprofile import _VCPROFILE_MIGRATIONS, import_profile

    # Defensive: this test assumes no v0 migration is registered. If
    # a future commit adds one, the assumption needs revisiting.
    assert 0 not in _VCPROFILE_MIGRATIONS

    bad = tmp_path / "ancient.vcprofile"
    _write_vcprofile(bad, format_version=0)

    with pytest.raises(ValueError, match=r"v0 -> v1 migration"):
        import_profile(bad, "x")


def test_import_older_than_current_with_registered_migration_migrates_and_warns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Bug-class half-B. The mechanism: an older format_version with a
    registered migration in `_VCPROFILE_MIGRATIONS` imports successfully
    and emits a stderr warning naming the leg. We exercise the
    mechanism by injecting a fake v0 -> v1 migration that fills in
    the profile-name (in the spirit of how a real migration would
    transform old-shape data into new-shape data)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from woys.vcprofile import VCPROFILE_VERSION, _migrate_vcprofile_raw

    # VCPROFILE_VERSION must currently equal 1 for the v0 -> v1 test
    # leg to map onto the verdict's "current-1" scenario. If the format
    # version is bumped in a future commit, this test needs to track.
    assert VCPROFILE_VERSION == 1, (
        "this test exercises v0 -> v1 migration; bump the source/target "
        "versions if VCPROFILE_VERSION moves"
    )

    def _legacy_v0_to_v1(raw: dict[str, object]) -> dict[str, object]:
        # A test-only migration: pretend v0 had `meta.author` and we now
        # rename it to `meta.author_hint`. Real future migrations will
        # do the same shape of work.
        meta = dict(raw.get("meta", {}))  # type: ignore[arg-type]
        if "author" in meta:
            meta["author_hint"] = meta.pop("author")
        out: dict[str, object] = dict(raw)
        out["meta"] = meta
        return out

    # Inject the test leg without leaking it into other tests.
    from woys import vcprofile

    monkeypatch.setitem(vcprofile._VCPROFILE_MIGRATIONS, 0, _legacy_v0_to_v1)

    raw_in: dict[str, object] = {
        "meta": {"format_version": 0, "author": "test-author"},
        "profile": {},
        "model": {},
    }
    migrated = _migrate_vcprofile_raw(raw_in)
    err = capsys.readouterr().err

    assert migrated["meta"]["format_version"] == VCPROFILE_VERSION, (
        "the migrated dict's meta.format_version must be stamped "
        "to the current version after a successful walk"
    )
    assert migrated["meta"]["author_hint"] == "test-author"
    assert "author" not in migrated["meta"]
    assert "migrating v0 -> v1" in err, (
        f"the reader must print a stderr warning per migration leg; stderr was: {err!r}"
    )


def test_import_missing_format_version_is_clear(tmp_path: Path) -> None:
    """A .vcprofile with no `meta.format_version` raises a clear
    "missing or non-integer" error -- not a `None != 1` confusion."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from woys.vcprofile import import_profile

    bad = tmp_path / "no_version.vcprofile"
    bad.write_text('[meta]\nprofile_name = "x"\n[profile]\n[model]\n', encoding="utf-8")

    with pytest.raises(ValueError, match="missing or non-integer"):
        import_profile(bad, "x")


def test_import_non_integer_format_version_is_clear(tmp_path: Path) -> None:
    """A `format_version = "1"` (string instead of int) raises the
    same "missing or non-integer" error -- a malformed value is not
    a version mismatch."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from woys.vcprofile import import_profile

    bad = tmp_path / "string_version.vcprofile"
    _write_vcprofile(bad, format_version="1")

    with pytest.raises(ValueError, match="missing or non-integer"):
        import_profile(bad, "x")
