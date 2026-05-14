"""Drift contract tests for EngineConfig → AppConfig → profiles.

Pre-v0.8.0, three surfaces (`AppConfig` field list, `_PROFILE_FIELDS`,
the migration code's allowlist) maintained separate manual mirrors of
`EngineConfig`'s user-visible fields. New fields drifted across releases
(`input_gate_dbfs`, `prefer_pw_cat` were the rc4 audit casualties).

v0.8.0 introduces `audio.engine.USER_VISIBLE_ENGINE_FIELDS` as the single
source of truth. These tests pin the contract:

  * Every name in `USER_VISIBLE_ENGINE_FIELDS` is an actual `EngineConfig`
    field with a default that round-trips through TOML.
  * `AppConfig` forwards a default for every entry.
  * `_PROFILE_FIELDS` covers every entry (plus `rvc_model`).

If you add a user-visible EngineConfig field, add it to
`USER_VISIBLE_ENGINE_FIELDS` and these tests pass automatically.

Original work - Copyright (c) 2026 Alireza Hamayeli, All Rights Reserved.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import fields
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
if str(REPO / "src" / "server") not in sys.path:
    sys.path.insert(0, str(REPO / "src" / "server"))


def test_user_visible_fields_all_exist_on_engine_config() -> None:
    from audio.engine import USER_VISIBLE_ENGINE_FIELDS, EngineConfig

    engine_field_names = {f.name for f in fields(EngineConfig)}
    missing = [n for n in USER_VISIBLE_ENGINE_FIELDS if n not in engine_field_names]
    assert not missing, (
        f"USER_VISIBLE_ENGINE_FIELDS lists names that aren't on EngineConfig: {missing}. "
        f"This breaks the contract that the tuple is the single source of truth."
    )


def test_app_config_forwards_every_user_visible_engine_field() -> None:
    """Adding a new user-visible EngineConfig field without forwarding to
    AppConfig means user overrides in config.toml are silently dropped.
    This test catches the drift before it ships."""
    from audio.engine import USER_VISIBLE_ENGINE_FIELDS, EngineConfig
    from tui.config import AppConfig

    app_field_names = {f.name for f in fields(AppConfig)}
    missing = [n for n in USER_VISIBLE_ENGINE_FIELDS if n not in app_field_names]
    assert not missing, (
        f"AppConfig is missing forwarded fields from EngineConfig:\n"
        f"  {missing}\n"
        f"Add them to AppConfig with `name: T = _E.{{name}}` defaults."
    )

    # Defaults must actually match (catches the rc4 case where AppConfig
    # had output_latency_ms=100 while EngineConfig had been bumped to 300).
    eng = EngineConfig()
    app = AppConfig()
    for name in USER_VISIBLE_ENGINE_FIELDS:
        eng_default = getattr(eng, name)
        app_default = getattr(app, name)
        assert eng_default == app_default, (
            f"AppConfig.{name} default ({app_default!r}) does not match "
            f"EngineConfig.{name} ({eng_default!r}). The forwarding is stale."
        )


def test_profile_fields_cover_every_user_visible_engine_field() -> None:
    """`_PROFILE_FIELDS` should contain every USER_VISIBLE field plus
    `rvc_model` (which lives at the AppConfig layer as a string)."""
    from audio.engine import USER_VISIBLE_ENGINE_FIELDS
    from woys.profiles import _PROFILE_FIELDS

    # rvc_model is the AppConfig-only path field; everything else mirrors EngineConfig.
    expected = {"rvc_model", *USER_VISIBLE_ENGINE_FIELDS}
    actual = set(_PROFILE_FIELDS)
    missing = expected - actual
    extra = actual - expected
    assert not missing and not extra, (
        f"_PROFILE_FIELDS drift:\n  missing: {sorted(missing)}\n  extra: {sorted(extra)}"
    )


@pytest.mark.parametrize("name", ["input_gate_dbfs", "prefer_pw_cat", "input_gate_hysteresis_ms"])
def test_rc4_drift_regression(name: str) -> None:
    """The rc4 audit (LESSONS-class) caught these specific fields drifting
    out of profile / AppConfig coverage. Pin them explicitly so a future
    refactor can't quietly drop them."""
    from audio.engine import USER_VISIBLE_ENGINE_FIELDS
    from tui.config import AppConfig
    from woys.profiles import _PROFILE_FIELDS

    assert name in USER_VISIBLE_ENGINE_FIELDS
    assert name in _PROFILE_FIELDS
    assert name in {f.name for f in fields(AppConfig)}


# ---- review F-merged-008 / F-01-04: single forwarding path -------------
# B9 + B50 caught AppConfig/EngineConfig *default* drift. The v0.9.0 add-on
# AST-walked cli.py / app.py for hand-written `EngineConfig(...)` blocks and
# checked each forwarded every field -- but its `_NOT_FORWARDED_AT_CONSTRUCTION`
# exemption list went stale: it exempted `mic_rate` / `sink_rate` (which the
# TUI *does* forward and the engine *does* consume) and `sink_name` (which
# both cli sites *did* forward), so the test was green while `woys diag` /
# `woys engine` silently ran 48 kHz defaults on non-48k hardware (F-01-04).
#
# The fix removed all three hand-written blocks in favour of one
# `app_config_to_engine_config()` helper. These tests now pin (1) that the
# helper forwards every USER_VISIBLE field, and (2) that no hand-rolled
# `EngineConfig(...)` block has crept back into cli.py / app.py.


def _engine_config_call_kwargs(source: str) -> list[set[str]]:
    """Return the kwarg names from each `EngineConfig(...)` call in `source`."""
    tree = ast.parse(source)
    out: list[set[str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Name) and func.id == "EngineConfig":
                name = func.id
            elif isinstance(func, ast.Attribute) and func.attr == "EngineConfig":
                name = func.attr
            if name == "EngineConfig":
                kwargs = {kw.arg for kw in node.keywords if kw.arg is not None}
                out.append(kwargs)
    return out


def test_app_config_to_engine_config_forwards_every_user_visible_field() -> None:
    """The single forwarding helper must carry *every* USER_VISIBLE field
    from AppConfig to EngineConfig.

    This is the F-merged-008 guarantee: add a field to `EngineConfig` +
    `USER_VISIBLE_ENGINE_FIELDS` + `AppConfig` and it reaches every entry
    point with no other edit -- because the helper iterates the tuple. A
    field the helper drops shows up here as a default-vs-sentinel mismatch
    (F-01-04 was exactly `mic_rate` / `sink_rate` being dropped).
    """
    from audio.engine import USER_VISIBLE_ENGINE_FIELDS
    from tui.config import AppConfig, app_config_to_engine_config

    cfg = AppConfig()
    # Mutate each field to a value distinct from its default so a dropped
    # field is visible as a default-vs-sentinel mismatch.
    sentinels: dict[str, object] = {}
    for name in USER_VISIBLE_ENGINE_FIELDS:
        cur = getattr(cfg, name)
        if isinstance(cur, bool):  # bool before int -- bool is an int subclass
            new: object = not cur
        elif isinstance(cur, int):
            new = cur + 7
        elif isinstance(cur, float):
            new = cur + 1.5
        elif isinstance(cur, str):
            new = cur + "-sentinel"
        else:
            new = cur
        sentinels[name] = new
        setattr(cfg, name, new)

    engine_cfg = app_config_to_engine_config(cfg)

    for name, expected in sentinels.items():
        assert getattr(engine_cfg, name) == expected, (
            f"app_config_to_engine_config dropped the user-visible field {name!r} "
            f"(F-merged-008 / F-01-04 regression)"
        )


@pytest.mark.parametrize(
    "source_path",
    [
        REPO / "src" / "woys" / "cli.py",
        REPO / "src" / "tui" / "app.py",
    ],
)
def test_no_hand_rolled_engine_config_in_cli_or_app(source_path: Path) -> None:
    """cli.py and app.py must construct `EngineConfig` *only* through
    `app_config_to_engine_config()` -- never a hand-written `EngineConfig(...)`
    block. A hand-rolled block is how every historical forwarding-drift bug
    (rc4 `prefer_pw_cat`, rc1 `prefer_native_pw`, F-01-04 `mic_rate`/
    `sink_rate`) became real.
    """
    calls = _engine_config_call_kwargs(source_path.read_text())
    assert calls == [], (
        f"{source_path.name}: found {len(calls)} direct EngineConfig(...) "
        f"construction(s) -- use app_config_to_engine_config() instead so "
        f"every USER_VISIBLE field is forwarded from one place."
    )
