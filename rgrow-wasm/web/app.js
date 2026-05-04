// rgrow web frontend.
//
// One module, no bundler. Pulls the wasm module from ./pkg, drives a
// `Sim` per requestAnimationFrame, and blits the rendered RGBA bytes
// onto a <canvas> via putImageData.
//
// Frame data round-trip: each frame, `sim.renderAlloc` returns a
// Uint8ClampedArray (one memcpy out of wasm memory). We wrap it in an
// ImageData and putImageData onto the canvas. A true zero-copy path
// (handing wasm a typed-array view of its own memory) is feasible but
// requires careful re-acquisition on memory growth; deferred to v1.1.

import init, { Sim } from "./pkg/rgrow_wasm.js";

const $ = (id) => document.getElementById(id);

const canvas = $("sim");
const ctx = canvas.getContext("2d");
const statsEl = $("stats");
const pauseBtn = $("pause");
const stepBtn = $("step");
const resetBtn = $("reset");
const eventsPerStepInput = $("events-per-step");
const stepBudgetMsInput = $("step-budget-ms");
const timescaleInput = $("timescale");
const maxEventsPerSecInput = $("max-events-per-sec");
const scaleInput = $("scale");
const scaleValueEl = $("scale-value");
const showMismatchesInput = $("show-mismatches");
const exampleSelect = $("example-select");
const fileInput = $("file-input");
const pasteInput = $("paste-input");
const pasteLoadBtn = $("paste-load");
const parameterList = $("parameter-list");
const importfileBlock = $("importfile-block");
const importfileMessage = $("importfile-message");
const importfileNameEl = $("importfile-name");
const importfileInput = $("importfile-input");
const importfileOffsetI = $("importfile-offset-i");
const importfileOffsetJ = $("importfile-offset-j");
const importfileStatus = $("importfile-status");

let wasm = null;     // wasm module exports (after init)
let sim = null;      // current Sim instance
let lastTilesetText = null; // for reload
let lastTilesetKind = null; // "yaml" | "json"

let paused = false;
let lastFrameTimestamp = 0;
let smoothedFps = 0;

// Per-real-second event counter for the max-events/sec limiter. Mirrors
// the desktop GUI: count events in a 1-second window, cap the next
// evolve's for_events to whatever budget is left, reset on the second
// boundary.
let eventsThisSecond = 0;
let secondStart = 0;

function fmt(n, digits = 4) {
  if (typeof n === "bigint") n = Number(n);
  if (!Number.isFinite(n)) return String(n);
  if (Math.abs(n) >= 1e4 || (Math.abs(n) < 1e-3 && n !== 0)) {
    return n.toExponential(digits);
  }
  return n.toFixed(digits);
}

function setControlsEnabled(enabled) {
  pauseBtn.disabled = !enabled;
  stepBtn.disabled = !enabled;
  resetBtn.disabled = !enabled;
}

function detectKind(text) {
  const t = text.trimStart();
  if (t.startsWith("{")) return "json";
  // Xgrow files start with this header line. The YAML parser would
  // otherwise accept it as a string and then fail in less obvious ways.
  if (/^tile edges(\s|=)/.test(t) || /^num tile types\s*=/.test(t)) {
    return "xgrow";
  }
  return "yaml";
}

async function loadTilesetText(text, kind) {
  const k = kind || detectKind(text);
  if (sim) {
    sim.free?.();
  }
  try {
    sim =
      k === "json" ? Sim.fromJson(text)
      : k === "xgrow" ? Sim.fromXgrow(text)
      : Sim.fromYaml(text);
  } catch (e) {
    statsEl.textContent = `Error loading tileset: ${e.message || e}`;
    sim = null;
    setControlsEnabled(false);
    return;
  }
  lastTilesetText = text;
  lastTilesetKind = k;
  setControlsEnabled(true);
  paused = true;
  pauseBtn.textContent = "Resume";

  resizeCanvasFor(sim);
  rebuildParameterControls();
  if (k === "xgrow") {
    showImportfilePromptFor(text);
  } else {
    hideImportfilePrompt();
  }
  // Kick the RAF loop if it isn't running.
  if (!rafScheduled) {
    rafScheduled = true;
    requestAnimationFrame(frame);
  }
}

// Match `importfile=foo.seed` either as a real arg or inside the
// recipe-style `% xgrow ... importfile=...` comments many .tiles files
// carry. Capture the bare filename (no path / wrapping quotes).
function findImportfileReference(text) {
  const m = text.match(/importfile\s*=\s*([^\s&|;)<>"'`%]+)/i);
  if (!m) return null;
  // Strip any leading directory components — the user picks the file
  // themselves, so only the basename is meaningful.
  const raw = m[1];
  const base = raw.split(/[\\/]/).pop();
  return base || raw;
}

function showImportfilePromptFor(text) {
  const name = findImportfileReference(text);
  if (!name) {
    hideImportfilePrompt();
    return;
  }
  importfileBlock.hidden = false;
  importfileNameEl.textContent = name;
  importfileMessage.textContent =
    "This xgrow tileset references an importfile (a saved-flake `.seed`). " +
    "Pick the file below to load it as the initial canvas state.";
  importfileStatus.textContent = "";
  importfileInput.value = "";
}

function hideImportfilePrompt() {
  importfileBlock.hidden = true;
  importfileNameEl.textContent = "";
  importfileMessage.textContent = "";
  importfileStatus.textContent = "";
  importfileInput.value = "";
}

function resizeCanvasFor(s) {
  const scale = Number(scaleInput.value);
  const fs = s.frameSize(scale);
  canvas.width = fs.width;
  canvas.height = fs.height;
}

let rafScheduled = false;

function readPositiveNumber(input) {
  const s = input.value.trim();
  if (s === "") return null;
  const v = Number(s);
  if (!Number.isFinite(v) || v <= 0) return null;
  return v;
}

function frame(timestamp) {
  rafScheduled = false;
  if (!sim) return;

  const now = performance.now();
  const realDt = lastFrameTimestamp ? (now - lastFrameTimestamp) / 1000 : 0;

  if (!secondStart || now - secondStart >= 1000) {
    eventsThisSecond = 0;
    secondStart = now;
  }

  const budget = Math.max(1, Number(stepBudgetMsInput.value));
  const timescale = readPositiveNumber(timescaleInput);
  const maxEps = readPositiveNumber(maxEventsPerSecInput);

  if (!paused) {
    let forEvents = null;
    let forTime = timescale != null && realDt > 0 ? realDt * timescale : null;
    let forWallMs = budget;
    let skipEvolve = false;

    if (maxEps != null) {
      const remaining = maxEps - eventsThisSecond;
      if (remaining <= 0) {
        skipEvolve = true;
      } else {
        forEvents = remaining;
      }
    }

    if (!skipEvolve) {
      try {
        const r = sim.stepWithBounds(forEvents, forTime, forWallMs);
        if (r && typeof r.events_this_step !== "undefined") {
          const n = typeof r.events_this_step === "bigint"
            ? Number(r.events_this_step)
            : r.events_this_step;
          eventsThisSecond += n;
        }
      } catch (e) {
        statsEl.textContent = `Step error: ${e.message || e}`;
        paused = true;
        pauseBtn.textContent = "Resume";
      }
    }
  }

  const scale = Number(scaleInput.value);
  const fs = sim.frameSize(scale);
  if (fs.width !== canvas.width || fs.height !== canvas.height) {
    canvas.width = fs.width;
    canvas.height = fs.height;
  }

  let bytes;
  try {
    bytes = sim.renderAlloc(scale, showMismatchesInput.checked);
  } catch (e) {
    statsEl.textContent = `Render error: ${e.message || e}`;
    return;
  }
  // `bytes` is a Uint8ClampedArray of length width*height*4. Wrap once
  // and putImageData. Constructing ImageData from a typed array is
  // cheap; the typed array itself is the data buffer.
  const img = new ImageData(bytes, canvas.width, canvas.height);
  ctx.putImageData(img, 0, 0);

  // FPS smoothing.
  if (lastFrameTimestamp) {
    const inst = 1 / Math.max(realDt, 1e-6);
    smoothedFps = smoothedFps ? smoothedFps * 0.9 + inst * 0.1 : inst;
  }
  lastFrameTimestamp = now;

  statsEl.textContent =
    `model: ${sim.modelName()}    ` +
    `t = ${fmt(sim.time())}    events = ${sim.totalEvents().toLocaleString()}    ` +
    `tiles = ${sim.nTiles()}    mismatches = ${sim.mismatches()}    ` +
    `energy = ${fmt(sim.energy())}    fps ≈ ${smoothedFps.toFixed(1)}`;

  rafScheduled = true;
  requestAnimationFrame(frame);
}

function rebuildParameterControls() {
  parameterList.innerHTML = "";
  if (!sim) return;
  let params;
  try {
    params = sim.parameters();
  } catch (e) {
    parameterList.textContent = `(parameter list unavailable: ${e.message || e})`;
    return;
  }
  if (!params || !params.length) {
    parameterList.innerHTML = "<em>This model exposes no tunable parameters.</em>";
    return;
  }
  for (const p of params) {
    const row = document.createElement("div");
    row.className = "parameter-row";

    const label = document.createElement("span");
    label.className = "param-name";
    label.textContent = p.units ? `${p.name} (${p.units})` : p.name;

    const input = document.createElement("input");
    input.type = "number";
    input.step = String(p.default_increment ?? 0.1);
    input.value = String(p.current_value ?? 0);
    if (p.min_value != null) input.min = String(p.min_value);
    if (p.max_value != null) input.max = String(p.max_value);

    const apply = () => {
      const v = Number(input.value);
      if (!Number.isFinite(v)) return;
      try {
        sim.setParameter(p.name, v);
      } catch (e) {
        console.warn("setParameter failed:", e);
      }
    };
    input.addEventListener("change", apply);

    const minus = document.createElement("button");
    minus.textContent = "−";
    minus.addEventListener("click", () => {
      input.value = String(Number(input.value) - Number(input.step || 0.1));
      apply();
    });

    const plus = document.createElement("button");
    plus.textContent = "+";
    plus.addEventListener("click", () => {
      input.value = String(Number(input.value) + Number(input.step || 0.1));
      apply();
    });

    row.append(label, input, minus, plus);
    parameterList.append(row);
  }
}

// ── Wiring ────────────────────────────────────────────────────────────

pauseBtn.addEventListener("click", () => {
  paused = !paused;
  pauseBtn.textContent = paused ? "Resume" : "Pause";
  if (!paused && !rafScheduled) {
    rafScheduled = true;
    requestAnimationFrame(frame);
  }
});

stepBtn.addEventListener("click", () => {
  if (!sim) return;
  paused = true;
  pauseBtn.textContent = "Resume";
  const events = Math.max(1, Number(eventsPerStepInput.value));
  try {
    sim.stepForEvents(BigInt(events));
  } catch (e) {
    statsEl.textContent = `Step error: ${e.message || e}`;
    return;
  }
  // Force one render.
  if (!rafScheduled) {
    rafScheduled = true;
    requestAnimationFrame(frame);
  }
});

resetBtn.addEventListener("click", () => {
  if (lastTilesetText) {
    loadTilesetText(lastTilesetText, lastTilesetKind);
  }
});

scaleInput.addEventListener("input", () => {
  scaleValueEl.textContent = scaleInput.value;
  if (sim) resizeCanvasFor(sim);
});

function kindFromName(name) {
  if (name.endsWith(".json")) return "json";
  if (name.endsWith(".tiles")) return "xgrow";
  if (name.endsWith(".yaml") || name.endsWith(".yml")) return "yaml";
  return null;
}

exampleSelect.addEventListener("change", async () => {
  const v = exampleSelect.value;
  if (!v) return;
  try {
    const r = await fetch(`./examples/${v}`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const text = await r.text();
    await loadTilesetText(text, kindFromName(v));
  } catch (e) {
    statsEl.textContent = `Failed to fetch example: ${e.message || e}`;
  }
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () =>
    loadTilesetText(String(reader.result), kindFromName(file.name));
  reader.readAsText(file);
});

pasteLoadBtn.addEventListener("click", () => {
  const text = pasteInput.value;
  if (!text.trim()) return;
  loadTilesetText(text);
});

function readOptionalInt(input) {
  const s = input.value.trim();
  if (s === "") return null;
  const v = Number(s);
  if (!Number.isFinite(v)) return null;
  return Math.trunc(v);
}

importfileInput.addEventListener("change", () => {
  const file = importfileInput.files?.[0];
  if (!file || !sim) return;
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const oi = readOptionalInt(importfileOffsetI);
      const oj = readOptionalInt(importfileOffsetJ);
      const placed = sim.loadXgrowSeed(
        String(reader.result),
        oi == null ? undefined : oi,
        oj == null ? undefined : oj,
      );
      importfileStatus.textContent = `Loaded ${placed}x${placed} flake from ${file.name}.`;
      // Force one frame so the imported state shows immediately.
      if (!rafScheduled) {
        rafScheduled = true;
        requestAnimationFrame(frame);
      }
    } catch (e) {
      importfileStatus.textContent = `Failed to load importfile: ${e.message || e}`;
    }
  };
  reader.readAsText(file);
});

(async () => {
  wasm = await init();
  statsEl.textContent = "Pick a built-in tileset, drop a YAML file, or paste a definition above.";
})();
