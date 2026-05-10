# v0.4.1 — Model-switch bug — root cause trace

> **NOTE: Historical investigation snapshot, captured at v0.4.1 (2026-05-04).**
> Recommendations and `engine.py:NNN` line numbers may be stale — the engine
> has grown by ~1500 lines since this was written. The current canonical
> reference for engine internals is `docs/05-perf.md` and `LESSONS.md`
> chronology. Don't act on this doc as if it reflects current state; treat
> the line numbers as approximate anchors and re-grep by symbol name.

> Pre-fix investigation per `V0_4_1_BUGFIX_BRIEF.md` §2. Filled before any
> code changed.

## Reported symptoms

1. `woys models use <slug>` writes config but the running engine
   doesn't pick it up. After TUI restart, the boot still loads Amitaro.
2. TUI `p` keypress changes the displayed `profile:` but not the audio voice.
3. `woys status` doesn't include the loaded model name anywhere.

## Wiring trace

### A. TUI startup — which model does it actually load?

`src/tui/app.py:111-136` (`VCClientApp.__init__`) constructs `EngineConfig`
with these fields **only**:

```
chunk_seconds, mic_rate, sink_rate, f0_up_key, sid, sink_name,
monitor, output_latency_ms, embedder, sola_enabled,
sola_crossfade_ms, sola_search_ms, sola_context_ms
```

**`rvc_model` is NOT passed.** Therefore `EngineConfig`'s default kicks in
(`audio/engine.py` — search for `DEFAULT_RVC_MODEL`: `DEFAULT_RVC_MODEL = MODELS_DIR / "amitaro_v2_16k.onnx"`).
Whatever the user has in `~/.config/woys/config.toml`'s
`rvc_model` field is **silently ignored** at TUI startup.

This is failure #1 (post-restart): no matter what `models use` writes, the
TUI boots Amitaro.

### B. `action_cycle_profile` — does the `p` key actually swap?

`src/tui/app.py:240-268`. Walks through:

1. `apply_profile(self.cfg, next_name)` — copies the saved profile's
   fields onto `self.cfg` (including `rvc_model`).
2. Mirrors **only three** fields onto the engine config:
   `self.engine.cfg.f0_up_key`, `self.engine.cfg.sid`,
   `self.engine.cfg.monitor`.
3. **Never touches `self.engine.cfg.rvc_model`.** Never calls
   `engine.reload_rvc()`. The ORT session for the model stays exactly
   where it was when the engine first started.

This is failure #2: the visible `profile:` line updates because
`self._active_profile` and `self.cfg.rvc_model` change, but the audio path
keeps using the original RVC session.

### C. `models use` CLI — what does it write?

`src/woys/models.py:213-229` (`cli_models_use`):

1. Loads the on-disk config.
2. Sets `cfg.rvc_model = str(path.resolve())`.
3. Calls `save_config(cfg)` — writes to `~/.config/woys/config.toml`.
4. Prints a "restart the engine for the change to take effect" message.

It writes the config correctly. But the running engine has no IPC channel
to be told that config changed — it would only honor the new path on next
restart. *Combined with* failure A above, even the next restart ignores it.

### D. Unix socket protocol

`src/tui/control.py` docstring lists: `TOGGLE`, `PITCH ±N`, `STATUS`,
`QUIT`. **There is no `MODEL` or `PROFILE` command.** The CLI's `toggle`,
`pitch`, and `status` subcommands forward to this socket; `models use`
does not — it writes config directly and exits.

### E. `STATUS` response shape

`src/tui/app.py:178-184`:

```python
return (
    f"OK running={s.running} "
    f"pitch={int(self.pitch)} "
    f"profile={self._active_profile or '-'} "
    f"avg_total_ms={s.avg_total_ms:.1f} "
    f"avg_inf_ms={s.avg_inference_ms:.1f}"
)
```

No `model=` field. Failure #3.

### F. The engine *does* have a hot-swap method

`audio/engine.py` — search for `def reload_rvc`:

```python
def reload_rvc(self, path: Path) -> None:
    """Hot-swap the RVC voice model."""
    self.cfg.rvc_model = path
    self._rvc = _make_session(path)
    self._is_half = self._rvc.get_inputs()[0].type != "tensor(float)"
```

Two issues:

- **Thread-unsafe.** The audio loop runs in `_run_loop` on a background
  thread; calling `reload_rvc` from the TUI thread races against the
  `_rvc.run()` call in `_infer`.
- **No SOLA-buffer drain.** The crossfade tail is implicit in
  `self._sola._prev_tail`; replacing the RVC session under it produces an
  audible click as the next chunk's spectrum suddenly differs.
- **No streaming-state reset.** `self._input_history` from the old model
  is now feeding the new model; for one chunk the embedder sees stale
  context for the new RVC voice.

`reload_rvc` is also **never called** from anywhere in `src/`. Confirmed
via `grep -rn reload_rvc src/`:

```
src/audio/engine.py:    def reload_rvc(self, path: Path) -> None:
```
(search `engine.py` for `def reload_rvc`)

Sole reference is the definition. No callers.

## Root cause summary

The `models use` and `p`-key UX shipped without engine integration.
Three holes:

1. TUI doesn't pass `cfg.rvc_model` to `EngineConfig` on startup.
2. `action_cycle_profile` mirrors *some* engine fields but skips
   `rvc_model` and never calls `reload_rvc`.
3. The Unix socket protocol lacks a `MODEL` command, so `models use` can't
   reach a running engine.

`reload_rvc` itself is also incomplete (thread-unsafe, no buffer drain).

## Fix plan

A — **Pass `rvc_model` from `AppConfig` into `EngineConfig` at TUI start.**
Fall back to `DEFAULT_RVC_MODEL` when `cfg.rvc_model` is empty (first run).

B — **Promote `reload_rvc` to a thread-safe `request_model_swap(path)`** that
the engine's worker thread picks up at chunk boundaries. The worker drains
the SOLA tail to pacat, then swaps + resets streaming state.

C — **Wire `action_cycle_profile` to actually swap.** After `apply_profile`,
mirror `rvc_model` to the engine via `request_model_swap`, plus all the
other profile fields onto `self.engine.cfg`.

D — **Add `MODEL <slug-or-path>` to the Unix-socket protocol.** Resolves
via `models.find_by_name`, calls `request_model_swap`, persists.

E — **Make `cli_models_use` try the socket first.** If the engine is
running, hot-swap; otherwise fall back to config writeback (no more
"restart the engine" message).

F — **Add `model=<basename>` to the `STATUS` response.**

G — **Persist on quit.** `action_quit` already saves config; verify
`cfg.rvc_model` reflects the currently-loaded model at that point (it
does, because we update `self.cfg.rvc_model` whenever we swap).

H — **Tests.** New `tests/test_model_swap.py` covers: TUI honors
`cfg.rvc_model` on start; `request_model_swap` replaces the ORT session;
`STATUS` includes `model=`; quit→reload preserves the active model.
