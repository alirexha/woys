# vcclient-cachy mic-helper (browser extension skeleton)

> **Status: skeleton only.** Manifest v3 + popup that detects whether
> `vcclient-mic` is visible to the browser. No engine API integration yet
> — that's future work.

## Why an extension?

When the engine is running and `vcclient-mic` exists, every web app that
records audio (Google Meet, Discord web, Zoom web, web SIP, etc.) can pick
it up — but only after the user manually selects it from a device dropdown.
Browsers also cache the device list per tab, so opening a call before
starting the engine means the user has to refresh.

This extension's eventual job: detect the engine's state, prompt the user
to pick `vcclient-mic` when they're on a known voice/call site, and offer
a one-click "reload tab to refresh device list" button.

## Loading the unpacked extension

### Chromium (Chrome / Brave / Vivaldi / Edge)

1. `chrome://extensions` → enable "Developer mode" (top-right toggle).
2. Click "Load unpacked", point at `pkg/browser-extension/`.
3. The toolbar gets a "vcclient-cachy mic helper" icon. Click it to test.

### Firefox

Manifest v3 in Firefox needs `browser_specific_settings.gecko.id`
(already in the manifest). For temporary load:

1. `about:debugging#/runtime/this-firefox`
2. Click "Load Temporary Add-on…", point at
   `pkg/browser-extension/manifest.json`.
3. The temp add-on persists for the session only.

For permanent install, package + sign through addons.mozilla.org.

## File layout

```
pkg/browser-extension/
├── manifest.json    Manifest v3 declaration
├── popup.html       320×~150px popup
├── popup.js         enumerates audio devices, surfaces vcclient-mic state
├── background.js    service-worker stub (no-op for skeleton)
├── icons/           required by manifest; placeholders if generic
└── README.md        this file
```

## What's missing (future work)

- Real engine state probe: WebSocket / native messaging to the local
  `vcclient-cachy run` process so the popup knows if the engine is up,
  what model is loaded, and current pitch.
- Auto-pick: detect content scripts on supported voice apps; click the
  device dropdown for the user.
- Per-tab override: remember user's preference per origin so the helper
  doesn't pop up on sites where they don't want vcclient-mic active.
- Icon assets at 16/48/128 px. Skeleton ships with placeholder paths;
  drop real PNGs into `icons/` before submitting to the Chrome Web Store
  / addons.mozilla.org.
- Submission to web stores (separate pipelines from AUR).

## Why no permissions in the manifest yet?

Manifest v3's `permissions` array is only requested when the feature
actually needs them. The skeleton reads `navigator.mediaDevices` from the
popup context, which doesn't require `microphone` permission to *list*
devices (only to access them). Add `"microphone"` if a future feature
needs to record.

## License

Same All-Rights-Reserved umbrella as the rest of the original work in
this repository (see root `LICENSE`).
