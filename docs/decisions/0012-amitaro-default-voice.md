# 0012 — `amitaro_v2_16k` as the install-time default voice

## Decision

The install-time default voice is `amitaro_v2_16k.onnx`, fetched by
`scripts/download_weights.py` and set via
`DEFAULT_RVC_MODEL = MODELS_DIR / "amitaro_v2_16k.onnx"`
(`src/audio/engine.py:114`).

## Status

`accepted`

## Context

`PROJECT_BRIEF.md` §15 forbids bundling models. `NOTICE` ("MODEL
WEIGHTS" section) lists `amitaro_v2_16k.onnx` as the sample voice
for smoke testing, sourced from
`huggingface.co/wok000/vcclient_model`. The user's actual daily
driver is `e_girl` (referenced in engine retrospective lines, e.g.
`engine.py:1912`'s "Live diagnostic on e_girl voice"), but `e_girl`
is not redistributable as a default — it's a community-trained voice
without a foundation-grade redistribution license. Foundation files
(`FOUNDATION_NAMES` in `src/woys/models.py:28-38`) are the
infrastructure files (rmvpe, contentvec, hubert) plus this sample
voice — they are filtered out of `woys models list` and explicitly
considered "not user voices."

## Decision

Default voice = `amitaro_v2_16k`, inherited by foundation status,
not by user preference.

## Alternatives considered

- **No default voice; first-run wizard prompts pick** — best UX in
  principle. But the engine smoke test (`tests/test_smoke_rvc_onnx.py`)
  needs *some* voice to load; absent a default, the smoke test would
  have to ship its own fixture model (defeats the no-bundled-model
  rule).
- **`e_girl` as default** — the user's real daily driver. Not
  redistributable on a foundation-grade license; we don't have the
  rights to ship it as default, and even if we did the install-time
  Hugging Face fetch URL is community-author-owned and could break.
- **Voice-conditioned default (male user → male voice, etc.)** —
  requires either a first-run probe (which we don't ship; see the
  `f0_up_key=0` rationale family) or a user preference question at
  install time. Adds setup friction for a default that's about to be
  changed by the user anyway.

## Rationale

Two constraints fix the choice. First, `PROJECT_BRIEF.md` §15 forbids
bundling models, so the install-time fetch must hit a stable,
foundation-licensed Hugging Face repo
(`wok000/vcclient_model`). `amitaro_v2_16k` is the only voice in the
project's catalogue that sits in that repo with a clear
redistribution path. Second, the smoke test needs *some* voice to
load to verify the engine boots; that voice is also the right default
because users running `./install.sh` for the first time hear it work
end-to-end without picking a voice. The fact that the install-time
default won't sound like the user is documented in
`docs/INSTALL.md` and `docs/MODELS.md`; users run `woys models use
<voice>` after their first model download.

## Trade-offs accepted

The first-run experience is "you sound like Amitaro" — a Japanese
female voice — which is jarring for a male English-speaking user
default. Setting a different default would require either bundling
(forbidden), shipping a community voice on a foundation-grade license
(don't have rights), or building a first-run wizard (out of scope).
The current state is honest about the constraint. Voice-output
sample rate is 16 kHz only for `amitaro_v2_16k` — most other RVC v2
voices are 40 kHz; the engine handles this per-voice
(`engine.py:1407-1417`), but it's a code-path the default voice
exercises differently from typical user voices.

## Re-litigation triggers

- A foundation-grade redistributable voice that better matches the
  primary user demographic appears (e.g., a Creative-Commons-licensed
  English male RVC v2 model on a stable HF repo).
- The `wok000/vcclient_model` repo is moved or its license changes;
  we'd need a new foundation-default at that point regardless.
- The project ships a first-run wizard that probes the user's mic
  and recommends a voice — at that point, "no default" becomes a
  viable alternative.
