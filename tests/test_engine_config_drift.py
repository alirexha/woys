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


# ---- v0.9.0 add-on: forwarding-site drift ------------------------------------
# B9 + B50 caught the case where AppConfig defaults and EngineConfig defaults
# diverged. They did NOT catch the case where the explicit `EngineConfig(...)`
# construction sites in cli.py / app.py forget to forward a USER_VISIBLE field.
# That's exactly how rc4's `prefer_pw_cat` drift (and v0.9.0-rc1's
# `prefer_native_pw` drift, caught the same way) became real cuts. AST-walk
# the source files and assert every site forwards every field.


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


# Fields that are intentionally not forwarded by callers (they use the
# EngineConfig default; the user can't tune them via config.toml).
# Keep this list tight - anything here is a deliberate exception, not
# a forgotten plumb.
_NOT_FORWARDED_AT_CONSTRUCTION = {
    "mic_rate",
    "sink_rate",
    "sink_name",
}


@pytest.mark.parametrize(
    "source_path",
    [
        REPO / "src" / "woys" / "cli.py",
        REPO / "src" / "tui" / "app.py",
    ],
)
def test_engine_config_construction_forwards_user_visible_fields(
    source_path: Path,
) -> None:
    """Every `EngineConfig(...)` call in cli.py / app.py must explicitly
    forward every USER_VISIBLE field that's expected to be plumbed (i.e.
    not in `_NOT_FORWARDED_AT_CONSTRUCTION`). This catches the rc4-class
    drift bug at the construction site, not just the default-value layer.
    """
    from audio.engine import USER_VISIBLE_ENGINE_FIELDS

    expected_fields = set(USER_VISIBLE_ENGINE_FIELDS) - _NOT_FORWARDED_AT_CONSTRUCTION

    src = source_path.read_text()
    call_kwarg_sets = _engine_config_call_kwargs(src)
    if not call_kwarg_sets:
        pytest.skip(f"no EngineConfig() calls found in {source_path.name}")

    for i, kwargs in enumerate(call_kwarg_sets):
        missing = expected_fields - kwargs
        assert not missing, (
            f"{source_path.name}: EngineConfig() call #{i + 1} is missing "
            f"forwarded fields: {sorted(missing)}\n"
            f"This is the v0.9.0-rc1 prefer_native_pw drift class - every "
            f"user-visible field must be plumbed at every construction site."
        )
