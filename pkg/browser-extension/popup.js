// woys mic-helper popup script.
//
// v0.4.0 deliverable: skeleton only. No API integration with the local engine
// yet — that lands in a future release. For now this script just enumerates
// the browser's input devices and reports whether `vcclient-mic` is visible
// among them. The status pill in popup.html flips green when it is.

async function detectVcclientMic() {
  const status = document.getElementById("status");
  if (!status) return;

  // navigator.mediaDevices.enumerateDevices() requires getUserMedia
  // permission to surface device labels; without it, labels are empty
  // strings. We probe both ways.
  let devices = [];
  try {
    devices = await navigator.mediaDevices.enumerateDevices();
  } catch (err) {
    status.textContent = "browser denied device enumeration: " + err.message;
    status.classList.add("warn");
    return;
  }
  const inputs = devices.filter((d) => d.kind === "audioinput");

  // If labels are empty, the extension hasn't been granted mic permission yet.
  const hasLabels = inputs.some((d) => d.label && d.label.length > 0);
  if (!hasLabels) {
    status.textContent =
      "device labels not yet granted. Click any 'audio settings' page on a site to grant — vcclient-mic will then appear here.";
    status.classList.add("warn");
    return;
  }

  const found = inputs.find((d) => /vcclient[-_ ]?mic/i.test(d.label));
  if (found) {
    status.textContent = `vcclient-mic detected: ${found.label}`;
    status.classList.add("ok");
  } else {
    status.textContent =
      "vcclient-mic not detected. Is the engine running? Run `woys pw status`.";
    status.classList.add("warn");
  }
}

document.addEventListener("DOMContentLoaded", detectVcclientMic);
