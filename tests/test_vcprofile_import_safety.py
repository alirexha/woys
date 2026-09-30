"""`.vcprofile` import must not silently replace a local profile.

The profile name comes from the shared file (`meta.profile_name`).
Pre-fix, importing someone's `main.vcprofile` overwrote the user's own
`main` profile -- pitch, model path and every other field -- with no
prompt, and the CLI just printed "saved profile 'main'".
"""

from __future__ import annotations

from pathlib import Path

import pytest

_SHARED = (
    '[meta]\nformat_version = 1\nprofile_name = "main"\n'
    "[profile]\nf0_up_key = -3\nmonitor = true\n"
    '[model]\nsha256 = ""\n'
)


def _local_main(cfg_path: Path) -> None:
    from tui.config import AppConfig, save_config
    from woys.profiles import save_profile

    cfg = AppConfig()
    cfg.f0_up_key = 7
    cfg.rvc_model = "/models/mine.onnx"
    save_profile(cfg, "main")
    save_config(cfg, cfg_path)


def _main_profile(cfg_path: Path) -> dict[str, object]:
    from tui.config import load_config
    from woys.profiles import _profiles_bag

    return _profiles_bag(load_config(cfg_path))["main"]


def test_import_refuses_to_replace_an_existing_profile(tmp_path: Path) -> None:
    from woys.vcprofile import import_profile

    cfg_path = tmp_path / "config.toml"
    _local_main(cfg_path)
    vp = tmp_path / "shared.vcprofile"
    vp.write_text(_SHARED)
    with pytest.raises(ValueError, match="already exists"):
        import_profile(vp, config_path=cfg_path, models_dir=tmp_path)
    main = _main_profile(cfg_path)
    assert main["f0_up_key"] == 7
    assert main["rvc_model"] == "/models/mine.onnx"


def test_import_under_another_name_keeps_the_local_one(tmp_path: Path) -> None:
    from tui.config import load_config
    from woys.profiles import _profiles_bag
    from woys.vcprofile import import_profile

    cfg_path = tmp_path / "config.toml"
    _local_main(cfg_path)
    vp = tmp_path / "shared.vcprofile"
    vp.write_text(_SHARED)
    assert import_profile(vp, "theirs", config_path=cfg_path, models_dir=tmp_path) == "theirs"
    bag = _profiles_bag(load_config(cfg_path))
    assert bag["main"]["f0_up_key"] == 7
    assert bag["theirs"]["f0_up_key"] == -3


def test_import_overwrite_replaces_when_asked(tmp_path: Path) -> None:
    from woys.vcprofile import import_profile

    cfg_path = tmp_path / "config.toml"
    _local_main(cfg_path)
    vp = tmp_path / "shared.vcprofile"
    vp.write_text(_SHARED)
    assert import_profile(vp, config_path=cfg_path, models_dir=tmp_path, overwrite=True) == "main"
    assert _main_profile(cfg_path)["f0_up_key"] == -3


def test_cli_import_reports_the_clash_and_changes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import tui.config
    from woys.vcprofile import cli_profile_import

    # conftest points tui.config.CONFIG_FILE at a per-test tmp file.
    cfg_path = tui.config.CONFIG_FILE
    _local_main(cfg_path)
    vp = tmp_path / "shared.vcprofile"
    vp.write_text(_SHARED)
    assert cli_profile_import(str(vp)) == 1
    assert "already exists" in capsys.readouterr().err
    assert _main_profile(cfg_path)["f0_up_key"] == 7


def _vcprofile_named(tmp_path: Path, name_toml: str) -> Path:
    vp = tmp_path / "shared.vcprofile"
    vp.write_text(
        f"[meta]\nformat_version = 1\nprofile_name = {name_toml}\n"
        '[profile]\nf0_up_key = 1\n[model]\nsha256 = ""\n'
    )
    return vp


@pytest.mark.parametrize(
    "name_toml",
    [
        "123",
        '"x[/bold]"',
        '"[@click=app.quit]x[/]"',
        '"a\\nb"',
        '"a]b"',
        '"-rf"',
        '"trailing "',
        '"' + "n" * 65 + '"',
        '"caf\\u00e9"',
    ],
)
def test_import_rejects_unsafe_profile_names(tmp_path: Path, name_toml: str) -> None:
    from tui.config import load_config
    from woys.profiles import _profiles_bag
    from woys.vcprofile import import_profile

    cfg_path = tmp_path / "config.toml"
    vp = _vcprofile_named(tmp_path, name_toml)
    with pytest.raises(ValueError, match="invalid profile name"):
        import_profile(vp, config_path=cfg_path, models_dir=tmp_path)
    assert _profiles_bag(load_config(cfg_path)) == {}


def test_import_rejects_an_unsafe_name_given_on_the_command_line(tmp_path: Path) -> None:
    from woys.vcprofile import import_profile

    vp = _vcprofile_named(tmp_path, '"fine"')
    with pytest.raises(ValueError, match="invalid profile name"):
        import_profile(vp, "x[/bold]", config_path=tmp_path / "c.toml", models_dir=tmp_path)


@pytest.mark.parametrize("name", ["main", "Work 2", "deep_voice-v1.2", "7", "n" * 64])
def test_import_accepts_plain_profile_names(tmp_path: Path, name: str) -> None:
    from woys.vcprofile import import_profile

    vp = _vcprofile_named(tmp_path, f'"{name}"')
    assert import_profile(vp, config_path=tmp_path / "c.toml", models_dir=tmp_path) == name


def test_cli_import_holds_the_config_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`woys profile import` rewrites the whole config.toml; it must hold the
    same lock the TUI and the other profile commands take, or a TUI save
    landing in between loses the imported profile (or the TUI's change)."""
    import contextlib
    from collections.abc import Iterator

    import woys.profiles as profiles_mod
    import woys.vcprofile as vcprofile_mod

    held: list[bool] = []
    in_lock = [False]

    @contextlib.contextmanager
    def _recording_lock() -> Iterator[None]:
        in_lock[0] = True
        try:
            yield
        finally:
            in_lock[0] = False

    real_import = vcprofile_mod.import_profile

    def _spy(*a: object, **kw: object) -> str:
        held.append(in_lock[0])
        return real_import(*a, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(profiles_mod, "config_lock", _recording_lock)
    monkeypatch.setattr(vcprofile_mod, "import_profile", _spy)
    vp = tmp_path / "shared.vcprofile"
    vp.write_text(_SHARED)
    assert vcprofile_mod.cli_profile_import(str(vp)) == 0
    assert held == [True]
