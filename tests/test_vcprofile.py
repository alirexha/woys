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
    from vcclient_cachy.profiles import save_profile
    from vcclient_cachy.vcprofile import export_profile

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
    from vcclient_cachy.vcprofile import export_profile

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
    # Receiver renamed it locally — same content, different name.
    _write_dummy_onnx(receiver_models / "myvoice_renamed.onnx", voice_payload)

    cfg_path = tmp_path / "config.toml"

    from tui.config import AppConfig, load_config, save_config
    from vcclient_cachy.profiles import save_profile
    from vcclient_cachy.vcprofile import export_profile, import_profile

    monkeypatch.setattr("tui.config.CONFIG_FILE", cfg_path)
    monkeypatch.setattr("vcclient_cachy.models.MODELS_DIR", receiver_models)

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
    from vcclient_cachy import models as models_mod

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
    """Receiver has no matching .onnx — import should still succeed but
    leave rvc_model empty + warn."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

    sender_models = tmp_path / "sender"
    sender_models.mkdir()
    receiver_models = tmp_path / "receiver"
    receiver_models.mkdir()  # empty

    _write_dummy_onnx(sender_models / "exotic.onnx", b"\x08\x07iruniquecontent")
    cfg_path = tmp_path / "config.toml"

    from tui.config import AppConfig, load_config, save_config
    from vcclient_cachy.profiles import save_profile
    from vcclient_cachy.vcprofile import export_profile, import_profile

    monkeypatch.setattr("tui.config.CONFIG_FILE", cfg_path)

    cfg = AppConfig()
    cfg.rvc_model = str(sender_models / "exotic.onnx")
    save_profile(cfg, "exotic")
    save_config(cfg, cfg_path)
    bundle = export_profile("exotic", tmp_path / "exotic.vcprofile")

    cfg2 = AppConfig()
    save_config(cfg2, cfg_path)

    from vcclient_cachy import models as models_mod

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

    from vcclient_cachy.vcprofile import import_profile

    with pytest.raises(ValueError, match="format_version"):
        import_profile(bad, "x")
