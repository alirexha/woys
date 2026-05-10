# 0005 — TOML for user-facing configuration files

## Decision

User-facing configuration (`~/.config/woys/config.toml`, voice
profiles in `src/woys/profiles.py` and `src/woys/vcprofile.py`) is
stored in TOML.

## Status

`accepted`

## Context

`PROJECT_BRIEF.md` §10 says config is persisted to
`~/.config/vcclient-cachy/config.toml` (now `~/.config/woys/config.toml`
since the v0.6.0 rename). The brief picks TOML by name without
ranking alternatives. Profiles and the engine config share the same
serialiser. The only pickle in the tree is `src/woys/convert.py` for
`.pth` voice-checkpoint conversion, gated behind
`--yes-i-trust-the-pickle`.

## Decision

TOML for everything user-facing; pickle is quarantined behind an
explicit consent flag for `.pth` import only.

## Alternatives considered

- **JSON** — strict syntax, ubiquitous tooling. Loses comments;
  config files for a tunable engine benefit heavily from per-field
  comments explaining trade-offs (the `EngineConfig` docstrings
  would need to be re-emitted as schema docs separately).
- **YAML** — supports comments and multi-line strings, but
  `yaml.load()` defaults to unsafe load with full Python-object
  construction, and PyYAML's release cadence is a maintenance
  hazard. ruamel.yaml fixes both but is heavier than tomllib.
- **INI / configparser** — Python stdlib, minimal, but no nested
  structure. The `[gpu]` / `[pipewire]` / `[engine]` sectioning we
  want naturally maps to TOML tables, not flat INI sections.
- **Pickle** — binary, opaque to users, executes arbitrary Python on
  load. Used only for the `.pth` import path with explicit consent
  (`src/woys/convert.py:18-22`).

## Rationale

Three load-bearing properties came together for TOML on Python 3.11+.
First, `tomllib` is in the stdlib (3.11+) — no third-party dep, no
release-cadence churn. Second, `tomli_w` (writer) is a thin pure-Python
companion. Third — the load-bearing one — `_extras` round-trip
(`src/tui/config.py:86`): unknown TOML keys survive a load → save
cycle, so a user editing `config.toml` to add a field that a future
woys version will recognise doesn't lose data when the current
version writes the file back. This works cleanly in TOML because
keys are typed at parse time, awkwardly in JSON (would need a custom
serialiser), and partially in YAML (comment preservation is
implementation-specific).

## Trade-offs accepted

TOML's syntax for very deeply nested data is awkward; we keep config
shallow (one level of tables) to side-step that. TOML's dotted-key
syntax for nested tables can confuse first-time users; the existing
config has no such patterns. No multi-line strings beyond TOML's
basic triple-quoted form, which is sufficient for our use.

## Re-litigation triggers

- Python's stdlib gains a JSON5 / commented-JSON variant that
  matches TOML's feature set with one less serialiser dep. (Low
  probability on the visible Python roadmap.)
- The config schema grows deeper than two levels of nesting and
  TOML readability degrades; consider switching the sub-tree to
  YAML while keeping TOML for top-level.
- A new Python version retires `tomllib` (extremely unlikely).
