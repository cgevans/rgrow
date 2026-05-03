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
const scaleInput = $("scale");
const scaleValueEl = $("scale-value");
const showMismatchesInput = $("show-mismatches");
const exampleSelect = $("example-select");
const fileInput = $("file-input");
const pasteInput = $("paste-input");
const pasteLoadBtn = $("paste-load");
const parameterList = $("parameter-list");

let wasm = null;     // wasm module exports (after init)
let sim = null;      // current Sim instance
let lastTilesetText = null; // for reload
let lastTilesetKind = null; // "yaml" | "json"

let paused = false;
let lastFrameTimestamp = 0;
let smoothedFps = 0;

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
  return t.startsWith("{") ? "json" : "yaml";
}

async function loadTilesetText(text, kind) {
  const k = kind || detectKind(text);
  if (sim) {
    sim.free?.();
  }
  try {
    sim = k === "json" ? Sim.fromJson(text) : Sim.fromYaml(text);
  } catch (e) {
    statsEl.textContent = `Error loading tileset: ${e.message || e}`;
    sim = null;
    setControlsEnabled(false);
    return;
  }
  lastTilesetText = text;
  lastTilesetKind = k;
  setControlsEnabled(true);
  paused = false;
  pauseBtn.textContent = "Pause";

  resizeCanvasFor(sim);
  rebuildParameterControls();
  // Kick the RAF loop if it isn't running.
  if (!rafScheduled) {
    rafScheduled = true;
    requestAnimationFrame(frame);
  }
}

function resizeCanvasFor(s) {
  const scale = Number(scaleInput.value);
  const fs = s.frameSize(scale);
  canvas.width = fs.width;
  canvas.height = fs.height;
}

let rafScheduled = false;

function frame(timestamp) {
  rafScheduled = false;
  if (!sim) return;

  const budget = Math.max(1, Number(stepBudgetMsInput.value));
  if (!paused) {
    try {
      sim.stepForRealMs(budget);
    } catch (e) {
      statsEl.textContent = `Step error: ${e.message || e}`;
      paused = true;
      pauseBtn.textContent = "Resume";
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
  const now = performance.now();
  if (lastFrameTimestamp) {
    const dt = (now - lastFrameTimestamp) / 1000;
    const inst = 1 / dt;
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

exampleSelect.addEventListener("change", async () => {
  const v = exampleSelect.value;
  if (!v) return;
  try {
    const r = await fetch(`./examples/${v}`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const text = await r.text();
    await loadTilesetText(text, v.endsWith(".json") ? "json" : "yaml");
  } catch (e) {
    statsEl.textContent = `Failed to fetch example: ${e.message || e}`;
  }
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => loadTilesetText(String(reader.result));
  reader.readAsText(file);
});

pasteLoadBtn.addEventListener("click", () => {
  const text = pasteInput.value;
  if (!text.trim()) return;
  loadTilesetText(text);
});

(async () => {
  wasm = await init();
  statsEl.textContent = "Pick a built-in tileset, drop a YAML file, or paste a definition above.";
})();
