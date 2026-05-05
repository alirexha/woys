// woys mic-helper background service worker.
//
// v0.4.0 deliverable: skeleton only. No tab-level injection, no message
// passing yet. Lives here to satisfy Manifest v3's `background` slot;
// future revisions can hook chrome.runtime.onInstalled, listen for the
// host page asking for media access, etc.

self.addEventListener("install", (_event) => {
  // No-op for skeleton.
});

self.addEventListener("activate", (_event) => {
  // No-op for skeleton.
});

// Manifest v3 service workers are short-lived; persistent state goes into
// chrome.storage.local. Nothing to persist yet.
