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
const showMismatchesInput = $("show-mismatches");
const showTileNamesInput = $("show-tile-names");
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
const tileInfoEl = $("tile-info");
const tilesetPanel = $("tileset-panel");
const tilesetTable = $("tileset-table");
const tilesetTableBody = tilesetTable.querySelector("tbody");
const tilesetCountEl = $("tileset-count");
const gluePanel = $("glue-panel");
const glueTable = $("glue-table");
const glueTableBody = glueTable.querySelector("tbody");
const glueCountEl = $("glue-count");
const blockerPanel = $("blocker-panel");
const blockerTable = $("blocker-table");
const blockerTableBody = blockerTable.querySelector("tbody");
const blockerCountEl = $("blocker-count");

const TILESET_SPRITE_PX = 22;
// Don't bother annotating tile names below this cell size — the text
// would be unreadable and the per-cell measureText calls aren't free.
const TILE_LABEL_MIN_SCALE = 8;
let editFeatures = {
  tile_concentration: false,
  tile_edge_glue: false,
  glue_interaction: false,
};
let currentGlueList = null;
// id -> displayed label string (empty if no name). Refilled lazily as we
// see new tile ids during rendering; reset when a new sim is loaded
// because the id space (and KBlock blocker variants) is per-model.
let tileLabelCache = new Map();

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
    tilesetPanel.hidden = true;
    gluePanel.hidden = true;
    blockerPanel.hidden = true;
    return;
  }
  lastTilesetText = text;
  lastTilesetKind = k;
  setControlsEnabled(true);
  paused = true;
  pauseBtn.textContent = "Resume";

  resizeCanvasFor(sim);
  tileLabelCache = new Map();
  refreshEditableFeatures();
  refreshGlueList();
  rebuildTileSetPanel();
  rebuildGlueInteractionsPanel();
  rebuildBlockerPanel();
  rebuildParameterControls();
  pinnedCell = null;
  pinnedTilesetId = null;
  renderTileInfo(null);
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

// Auto-picked cell size in canvas pixels. Recomputed from the canvas
// container's CSS width on tileset load and on window resize. The
// `image-rendering: pixelated` style means we want this to match (or be
// an integer multiple of) the on-screen cell size, otherwise the
// browser does fractional scaling and we lose crispness.
let currentScale = 8;

function computeAutoScale(s) {
  // canvasSize is in pre-scale subcells. For diamond canvases, multiplying
  // by `scale` gives `frame_pixels = subcells * scale * (subcells_per_tile/2)`
  // — i.e., subcells absorb half the scale. Probe `frameSize(2)` and divide
  // by 2 to recover the per-scale-unit pixel ratio without having to know
  // the canvas's subcell convention.
  const probe = s.frameSize(2);
  const cols = Math.max(1, Math.round(probe.width / 2));
  const rows = Math.max(1, Math.round(probe.height / 2));
  // Container's content-box width (excluding canvas's 1px border).
  const parent = canvas.parentElement;
  const availW = Math.max(64, (parent?.clientWidth ?? 800) - 2);
  // Bound the height too so a tube canvas with rows ≪ cols doesn't get
  // a giant scale that overflows the viewport vertically.
  const availH = Math.max(64, window.innerHeight - 200);
  const sW = Math.floor(availW / cols);
  const sH = Math.floor(availH / rows);
  return Math.max(2, Math.min(32, Math.min(sW, sH)));
}

function resizeCanvasFor(s) {
  currentScale = computeAutoScale(s);
  const fs = s.frameSize(currentScale);
  if (canvas.width !== fs.width || canvas.height !== fs.height) {
    canvas.width = fs.width;
    canvas.height = fs.height;
  }
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

  const scale = currentScale;
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

  drawTileLabels(scale);

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

  // Refresh the pinned tile readout so it reflects the current state of
  // the cell (which may have flipped tile types between frames).
  if (pinnedCell) renderTileInfo(pinnedCell, { pinned: true });

  rafScheduled = true;
  requestAnimationFrame(frame);
}

// Look up the displayed name for a tile id, caching wasm results. The
// cache is reset on each sim load (`tileLabelCache = new Map()`) so the
// id space and KBlock blocker-variant mappings stay in sync.
function getTileLabel(id) {
  if (id === 0) return "";
  let s = tileLabelCache.get(id);
  if (s === undefined) {
    try {
      s = sim.tileLabel(id) || "";
    } catch {
      s = "";
    }
    tileLabelCache.set(id, s);
  }
  return s;
}

// Overlay tile names on the canvas where the cell is large enough and
// the name actually fits. We measure each unique label once per frame
// and skip cells whose label would overflow ~90% of the cell width.
// Black text with a translucent white outline reads well over both
// dark and light tile colors without needing a per-tile contrast check.
function drawTileLabels(scale) {
  if (!sim || scale < TILE_LABEL_MIN_SCALE) return;
  if (!showTileNamesInput.checked) return;
  // `labelAnchors(scale)` returns a flat Float32Array of triples
  // (cx_px, cy_px, tile_id) covering every non-empty cell at its
  // canvas-aware pixel position. Tube canvases shear/stagger inside this
  // call so JS doesn't need to know about storage→physical layout.
  let anchors;
  try {
    anchors = sim.labelAnchors(scale);
  } catch {
    return;
  }
  if (!anchors || anchors.length === 0) return;

  // Tile bounding-box pixel size — diamond canvases use 2× scale per tile
  // because each tile occupies a 2×2 subcell block.
  const subcellsPerTile = (() => {
    try { return sim.subcellsPerTile() || 1; } catch { return 1; }
  })();
  const tilePx = scale * subcellsPerTile;

  const fontSize = Math.max(8, Math.min(16, Math.round(tilePx * 0.55)));
  const maxWidth = tilePx * 0.92;
  const widthCache = new Map();

  ctx.save();
  ctx.font = `${fontSize}px sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.lineJoin = "round";
  ctx.lineWidth = Math.max(2, Math.round(fontSize / 4));
  ctx.strokeStyle = "rgba(255, 255, 255, 0.85)";
  ctx.fillStyle = "rgba(0, 0, 0, 1)";

  for (let i = 0; i < anchors.length; i += 3) {
    const cx = anchors[i];
    const cy = anchors[i + 1];
    const id = anchors[i + 2] | 0;
    const label = getTileLabel(id);
    if (!label) continue;
    let w = widthCache.get(label);
    if (w === undefined) {
      w = ctx.measureText(label).width;
      widthCache.set(label, w);
    }
    if (w > maxWidth) continue;
    ctx.strokeText(label, cx, cy);
    ctx.fillText(label, cx, cy);
  }
  ctx.restore();
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

// ── Tileset panel ─────────────────────────────────────────────────────
//
// One row per non-empty tile id, with a sprite (rendered via the painter
// over the wasm boundary so duples / multi-color tiles look right), id,
// name, optional concentration, and per-side glue names. Hover and click
// reuse the `#tile-info` bar — clicking a row pins it the same way
// clicking a canvas cell does. `pinnedCell` and `pinnedTilesetId` are
// mutually exclusive.

let pinnedTilesetId = null;
let currentTileSetData = null;

function rebuildTileSetPanel() {
  tilesetTableBody.replaceChildren();
  currentTileSetData = null;
  if (!sim) {
    tilesetPanel.hidden = true;
    return;
  }
  let tiles;
  try {
    tiles = sim.tileSet();
  } catch (e) {
    tilesetPanel.hidden = true;
    console.warn("tileSet() failed:", e);
    return;
  }
  if (!Array.isArray(tiles) || tiles.length === 0) {
    tilesetPanel.hidden = true;
    return;
  }
  currentTileSetData = tiles;
  tilesetPanel.hidden = false;
  tilesetCountEl.textContent = `${tiles.length} tile${tiles.length === 1 ? "" : "s"}`;

  const anyConc = tiles.some((t) => t.concentration != null);
  const anyFreeConc = tiles.some((t) => t.free_concentration != null);
  const anyGlue = tiles.some((t) =>
    Array.isArray(t.edge_glues) && t.edge_glues.some((g) => g != null && g !== "")
  );
  tilesetTable.classList.toggle("no-conc", !anyConc);
  tilesetTable.classList.toggle("no-free-conc", !anyFreeConc);
  tilesetTable.classList.toggle("no-glues", !anyGlue);

  for (const t of tiles) {
    const tr = document.createElement("tr");
    tr.dataset.tileId = String(t.id);

    const tdSprite = document.createElement("td");
    const sc = document.createElement("canvas");
    sc.width = TILESET_SPRITE_PX;
    sc.height = TILESET_SPRITE_PX;
    try {
      const bytes = sim.tilePixels(t.id, TILESET_SPRITE_PX);
      sc.getContext("2d").putImageData(
        new ImageData(bytes, TILESET_SPRITE_PX, TILESET_SPRITE_PX),
        0,
        0,
      );
    } catch {
      sc.style.background = rgbaCss(t.color);
    }
    tdSprite.append(sc);

    const tdId = document.createElement("td");
    tdId.className = "col-id";
    tdId.textContent = `#${t.id}`;

    const tdName = document.createElement("td");
    tdName.className = "col-name";
    tdName.textContent = t.name && t.name.length ? t.name : "(unnamed)";

    const tdConc = document.createElement("td");
    tdConc.className = "col-conc";
    if (editFeatures.tile_concentration && t.concentration != null) {
      const input = makeNumberInput(t.concentration, 3, (v, revert) => {
        applyEdit(
          () => {
            sim.setTileConcentration(t.id, v);
            // Editing a tile's total concentration shifts the
            // equilibrium free-blocker concentrations, which in turn
            // shifts every tile's "free" concentration. Refresh both
            // dependent panels so what's displayed stays consistent
            // with what the simulator is using.
            refreshDerivedConcentrationDisplays();
          },
          revert,
          `Failed to set tile #${t.id} concentration`,
        );
      });
      tdConc.append(wirePropagateButton(input, {
        digits: 3,
        kind: "tiles",
        label: "Conc.",
        errorPrefix: "Failed to bulk-set tile concentration",
        listMatches: (sourceValue) => {
          const key = formatForInput(sourceValue, 3);
          let tiles;
          try { tiles = sim.tileSet(); } catch { return []; }
          if (!Array.isArray(tiles)) return [];
          return tiles.filter(
            (x) => x.concentration != null && formatForInput(x.concentration, 3) === key,
          );
        },
        applySetter: (m, v) => sim.setTileConcentration(m.id, v),
        afterApply: () => {
          rebuildTileSetPanel();
          refreshDerivedConcentrationDisplays();
        },
      }));
    } else {
      tdConc.textContent = t.concentration != null ? fmt(t.concentration, 3) : "";
    }

    const tdFreeConc = document.createElement("td");
    tdFreeConc.className = "col-free-conc";
    tdFreeConc.dataset.tileId = String(t.id);
    tdFreeConc.textContent =
      t.free_concentration != null ? fmt(t.free_concentration, 3) : "";

    const glueCells = (t.edge_glue_ids || [null, null, null, null]).map(
      (gid, sideIdx) => {
        const td = document.createElement("td");
        td.className = "col-glue" + (gid != null ? "" : " empty");
        if (editFeatures.tile_edge_glue) {
          const inp = makeGlueInput(gid, (newId, revert) => {
            applyEdit(
              () => sim.setTileEdgeGlue(t.id, sideIdx, newId),
              revert,
              `Failed to set tile #${t.id} side ${"NESW"[sideIdx]} glue`,
            );
          });
          td.append(inp);
        } else {
          const name = t.edge_glues?.[sideIdx];
          td.textContent = name && name.length ? name : gid != null ? `#${gid}` : "—";
        }
        return td;
      },
    );

    tr.append(tdSprite, tdId, tdName, tdConc, tdFreeConc, ...glueCells);

    tr.addEventListener("mouseenter", () => {
      if (pinnedCell || pinnedTilesetId != null) return;
      renderTileInfoForTile(t, { pinned: false });
    });
    tr.addEventListener("mouseleave", () => {
      if (pinnedCell || pinnedTilesetId != null) return;
      renderTileInfo(null);
    });
    tr.addEventListener("click", (e) => {
      // Don't toggle the row pin when the user is interacting with an
      // editable form control inside the row.
      if (e.target.closest("input, select, button, textarea, label")) return;
      if (pinnedTilesetId === t.id) {
        pinnedTilesetId = null;
        tr.classList.remove("pinned");
        renderTileInfoForTile(t, { pinned: false });
      } else {
        // A canvas pin and a tileset pin are mutually exclusive — clear
        // any canvas pin so the visual state stays coherent.
        pinnedCell = null;
        clearAllPinnedRows();
        pinnedTilesetId = t.id;
        tr.classList.add("pinned");
        renderTileInfoForTile(t, { pinned: true });
      }
    });

    tilesetTableBody.append(tr);
  }
}

function clearAllPinnedRows() {
  for (const tr of tilesetTableBody.querySelectorAll("tr.pinned")) {
    tr.classList.remove("pinned");
  }
}

// ── Editing helpers ───────────────────────────────────────────────────

function refreshEditableFeatures() {
  if (!sim) {
    editFeatures = {
      tile_concentration: false,
      tile_edge_glue: false,
      glue_interaction: false,
    };
    return;
  }
  try {
    const f = sim.editableFeatures();
    editFeatures = {
      tile_concentration: !!f.tile_concentration,
      tile_edge_glue: !!f.tile_edge_glue,
      glue_interaction: !!f.glue_interaction,
    };
  } catch (e) {
    console.warn("editableFeatures() failed:", e);
    editFeatures = {
      tile_concentration: false,
      tile_edge_glue: false,
      glue_interaction: false,
    };
  }
}

// Common change handler: read the input, parse, run the wasm setter, and
// revert on error. The caller passes the setter as a thunk so we don't
// need a dedicated wrapper per field.
function applyEdit(setterThunk, revert, errorPrefix) {
  if (!sim) return;
  try {
    setterThunk();
  } catch (e) {
    statsEl.textContent = `${errorPrefix}: ${e.message || e}`;
    revert?.();
  }
}

function makeNumberInput(initialValue, digits, onCommit, opts = {}) {
  const input = document.createElement("input");
  input.type = "number";
  input.step = "any";
  if (!opts.allowNegative) input.min = "0";
  input.value = formatForInput(initialValue, digits);
  let lastCommitted = input.value;
  // The browser fires `change` on Enter or blur — exactly what we want
  // (no per-keystroke applies).
  input.addEventListener("change", () => {
    const v = Number(input.value);
    if (!Number.isFinite(v) || (!opts.allowNegative && v < 0)) {
      input.value = lastCommitted;
      return;
    }
    onCommit(v, () => {
      input.value = lastCommitted;
    });
    lastCommitted = input.value;
  });
  // Avoid the row click handler hijacking focus interactions.
  input.addEventListener("click", (e) => e.stopPropagation());
  return input;
}

function formatForInput(n, digits) {
  if (n == null || !Number.isFinite(n)) return "";
  // Use a plain-number representation by default; fall back to exponential
  // for tiny values where toFixed loses information.
  if (Math.abs(n) < 1e-3 && n !== 0) return n.toExponential(digits);
  return Number(n.toFixed(digits)).toString();
}

// ── Bulk-edit ("propagate") popover ───────────────────────────────────
//
// One shared popover floats over the page (position: fixed). Opens from
// a small "propagate" button next to each editable numeric input;
// pushes the row's value out to every other row whose current value
// rounds to the same display string. Matching on the displayed-precision
// string (rather than raw float equality) matches what the user sees
// and avoids 0.1+0.2-style surprises.

let propagatePopover = null;
let propagatePopoverAnchor = null;
// Detacher for the per-open click/keydown handlers — invoked from
// every dismissal path so stale closures from a previous open don't
// stack on top of new ones.
let propagatePopoverCleanup = null;

function ensurePropagatePopover() {
  if (propagatePopover) return propagatePopover;
  const div = document.createElement("div");
  div.id = "propagate-popover";
  div.hidden = true;
  const summary = document.createElement("div");
  summary.className = "propagate-summary";
  const label = document.createElement("label");
  const labelText = document.createElement("span");
  labelText.textContent = "New value:";
  const input = document.createElement("input");
  input.type = "number";
  input.step = "any";
  label.append(labelText, input);
  const buttons = document.createElement("div");
  buttons.className = "propagate-buttons";
  const cancelBtn = document.createElement("button");
  cancelBtn.type = "button";
  cancelBtn.className = "cancel";
  cancelBtn.textContent = "Cancel";
  const applyBtn = document.createElement("button");
  applyBtn.type = "button";
  applyBtn.className = "apply";
  applyBtn.textContent = "Apply";
  buttons.append(cancelBtn, applyBtn);
  div.append(summary, label, buttons);
  // Stop clicks inside from being treated as "outside" by the
  // document-level dismissal listener.
  div.addEventListener("mousedown", (e) => e.stopPropagation());
  document.body.append(div);
  propagatePopover = div;
  return div;
}

function positionPropagatePopover(anchor) {
  const pop = ensurePropagatePopover();
  const rect = anchor.getBoundingClientRect();
  const popW = pop.offsetWidth || 260;
  let left = rect.right - popW;
  if (left < 8) left = 8;
  let top = rect.bottom + 4;
  const popH = pop.offsetHeight || 140;
  if (top + popH > window.innerHeight - 8) {
    top = rect.top - 4 - popH;
    if (top < 8) top = 8;
  }
  pop.style.left = `${left}px`;
  pop.style.top = `${top}px`;
}

function hidePropagatePopover() {
  if (propagatePopoverCleanup) {
    propagatePopoverCleanup();
    propagatePopoverCleanup = null;
  }
  if (propagatePopover) propagatePopover.hidden = true;
  propagatePopoverAnchor = null;
}

// Close on scroll/resize/outside-click/Escape — same lifecycle as the
// glue combo dropdown above.
window.addEventListener("scroll", hidePropagatePopover, true);
window.addEventListener("resize", hidePropagatePopover);
window.addEventListener("mousedown", (e) => {
  if (!propagatePopover || propagatePopover.hidden) return;
  if (propagatePopover.contains(e.target)) return;
  if (propagatePopoverAnchor && propagatePopoverAnchor.contains(e.target)) return;
  hidePropagatePopover();
});
window.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && propagatePopover && !propagatePopover.hidden) {
    hidePropagatePopover();
  }
});

function openPropagatePopover(anchor, opts) {
  // Detach handlers from any previous open before attaching new ones —
  // outside-click / Escape dismissal doesn't go through the
  // Apply/Cancel path, so cleanup must run here too.
  if (propagatePopoverCleanup) {
    propagatePopoverCleanup();
    propagatePopoverCleanup = null;
  }
  const pop = ensurePropagatePopover();
  propagatePopoverAnchor = anchor;
  const summary = pop.querySelector(".propagate-summary");
  const input = pop.querySelector("input[type='number']");
  const cancelBtn = pop.querySelector("button.cancel");
  const applyBtn = pop.querySelector("button.apply");

  const sourceKey = formatForInput(opts.sourceValue, opts.digits);
  const matches = opts.listMatches(opts.sourceValue) || [];
  // matches includes the source row itself; "other rows" excludes it.
  const otherRows = Math.max(0, matches.length - 1);

  summary.replaceChildren();
  const head = document.createElement("div");
  const headLeft = document.createTextNode(`Set all ${opts.kind} where `);
  const headBold = document.createElement("b");
  headBold.textContent = `${opts.label} = ${sourceKey}`;
  head.append(headLeft, headBold);
  const sub = document.createElement("div");
  sub.className = "muted";
  sub.textContent = otherRows === 0
    ? "(no other rows match this value)"
    : `(${otherRows} other row${otherRows === 1 ? "" : "s"}; ${matches.length} total)`;
  summary.append(head, sub);

  input.value = formatForInput(opts.sourceValue, opts.digits);
  input.min = opts.allowNegative ? "" : "0";

  // Detach prior handlers (re-opening from a different anchor reuses
  // the same DOM nodes).
  const cleanup = () => {
    cancelBtn.removeEventListener("click", onCancel);
    applyBtn.removeEventListener("click", onApply);
    input.removeEventListener("keydown", onKey);
  };
  const onCancel = () => {
    cleanup();
    hidePropagatePopover();
  };
  const onApply = () => {
    const v = Number(input.value);
    if (!Number.isFinite(v) || (!opts.allowNegative && v < 0)) return;
    let aborted = false;
    for (const m of matches) {
      if (aborted) break;
      applyEdit(
        () => opts.applySetter(m, v),
        () => { aborted = true; },
        opts.errorPrefix,
      );
    }
    if (opts.afterApply) opts.afterApply();
    cleanup();
    hidePropagatePopover();
  };
  const onKey = (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      onApply();
    }
  };
  cancelBtn.addEventListener("click", onCancel);
  applyBtn.addEventListener("click", onApply);
  input.addEventListener("keydown", onKey);
  propagatePopoverCleanup = cleanup;

  applyBtn.disabled = otherRows === 0;

  pop.hidden = false;
  positionPropagatePopover(anchor);
  input.focus();
  input.select();
}

// Wrap a numeric `<input>` with a small "↪" button that opens the
// bulk-edit popover. Returns a flex span containing both. The input is
// still the focused-by-default element; the row's edit handler is
// untouched, so single-row editing behaves exactly as before.
function wirePropagateButton(input, opts) {
  const wrap = document.createElement("span");
  wrap.className = "cell-with-propagate";
  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "propagate-btn";
  btn.title = `Set all matching ${opts.kind} to a new value`;
  btn.textContent = "↪";
  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    const v = Number(input.value);
    if (!Number.isFinite(v)) return;
    openPropagatePopover(btn, {
      sourceValue: v,
      digits: opts.digits,
      kind: opts.kind,
      label: opts.label,
      allowNegative: !!opts.allowNegative,
      listMatches: opts.listMatches,
      applySetter: opts.applySetter,
      errorPrefix: opts.errorPrefix,
      afterApply: opts.afterApply,
    });
  });
  wrap.append(input, btn);
  return wrap;
}

// Display label for a glue id — its name if it has one, else `#<id>`.
// Used as the <input> value so the user-visible string is also what the
// browser autocomplete matches against.
function glueDisplayLabel(id) {
  if (id == null) return "";
  if (Array.isArray(currentGlueList)) {
    const g = currentGlueList.find((x) => x.id === id);
    if (g && g.name && g.name.length) return g.name;
  }
  return `#${id}`;
}

// Resolve a typed string to a glue id (or null for empty). Accepts the
// glue's name, or an explicit `#<id>` form. Returns `undefined` on no
// match so the caller can distinguish "no glue" from "invalid".
function resolveGlueInput(raw) {
  const s = raw.trim();
  if (s === "") return null;
  if (Array.isArray(currentGlueList)) {
    const byName = currentGlueList.find((g) => g.name === s);
    if (byName) return byName.id;
  }
  const m = s.match(/^#(\d+)$/);
  if (m) {
    const id = Number(m[1]);
    if (Array.isArray(currentGlueList) && currentGlueList.some((g) => g.id === id)) {
      return id;
    }
  }
  return undefined;
}

// One shared dropdown floats over the page (position: fixed) so it
// isn't clipped by the table's scroll container. Only one combobox can
// be open at a time, which is fine — we anchor it to whichever input
// has focus.
let glueDropdown = null;
let glueDropdownInput = null;
let glueDropdownActiveIndex = -1;

function ensureGlueDropdown() {
  if (glueDropdown) return glueDropdown;
  const ul = document.createElement("ul");
  ul.className = "glue-combo-list";
  ul.hidden = true;
  document.body.append(ul);
  glueDropdown = ul;
  return ul;
}

function positionGlueDropdown(input) {
  const ul = ensureGlueDropdown();
  const rect = input.getBoundingClientRect();
  ul.style.left = `${rect.left}px`;
  ul.style.top = `${rect.bottom + 2}px`;
  ul.style.minWidth = `${rect.width}px`;
}

function hideGlueDropdown() {
  if (glueDropdown) glueDropdown.hidden = true;
  glueDropdownInput = null;
  glueDropdownActiveIndex = -1;
}

function isGlueDropdownOpenFor(input) {
  return (
    glueDropdown &&
    !glueDropdown.hidden &&
    glueDropdownInput === input
  );
}

function updateGlueDropdownActive() {
  if (!glueDropdown) return;
  const items = glueDropdown.children;
  for (let i = 0; i < items.length; i++) {
    items[i].classList.toggle("active", i === glueDropdownActiveIndex);
  }
  const active = items[glueDropdownActiveIndex];
  if (active) active.scrollIntoView({ block: "nearest" });
}

function showGlueDropdown(input, filter, onPick) {
  const ul = ensureGlueDropdown();
  glueDropdownInput = input;

  const opts = [{ id: null, label: "—" }];
  if (Array.isArray(currentGlueList)) {
    for (const g of currentGlueList) {
      opts.push({
        id: g.id,
        label: g.name && g.name.length ? g.name : `#${g.id}`,
      });
    }
  }
  const f = filter.trim().toLowerCase();
  const matches = f === ""
    ? opts
    : opts.filter((o) => o.label.toLowerCase().includes(f));

  ul.replaceChildren();
  if (matches.length === 0) {
    hideGlueDropdown();
    return;
  }
  for (const opt of matches) {
    const li = document.createElement("li");
    li.textContent = opt.label;
    li.dataset.glueId = opt.id == null ? "" : String(opt.id);
    // mousedown so we beat the input's `blur` handler — preventDefault
    // keeps focus on the input.
    li.addEventListener("mousedown", (e) => {
      e.preventDefault();
      onPick(opt.id);
      hideGlueDropdown();
    });
    ul.append(li);
  }
  glueDropdownActiveIndex = 0;
  updateGlueDropdownActive();
  positionGlueDropdown(input);
  ul.hidden = false;
}

// Reposition / close on viewport changes. Use capture so nested
// scrollers (the tileset table wrap) trigger us too.
window.addEventListener("scroll", hideGlueDropdown, true);
window.addEventListener("resize", hideGlueDropdown);

function makeGlueInput(currentId, onCommit) {
  const input = document.createElement("input");
  input.type = "text";
  input.autocomplete = "off";
  input.spellcheck = false;
  input.placeholder = "—";
  input.value = glueDisplayLabel(currentId);
  let lastCommitted = input.value;

  const commitText = () => {
    const resolved = resolveGlueInput(input.value);
    if (resolved === undefined) {
      input.value = lastCommitted;
      return;
    }
    const newLabel = glueDisplayLabel(resolved);
    if (newLabel === lastCommitted) {
      // Already at this value (e.g. typed the same name again). Just
      // normalize the display and skip the wasm round-trip.
      input.value = newLabel;
      return;
    }
    let reverted = false;
    onCommit(resolved, () => {
      input.value = lastCommitted;
      reverted = true;
    });
    if (!reverted) {
      input.value = newLabel;
      lastCommitted = newLabel;
    }
  };

  const pickById = (id) => {
    input.value = id == null ? "" : glueDisplayLabel(id);
    commitText();
  };

  input.addEventListener("focus", () =>
    showGlueDropdown(input, "", pickById),
  );
  input.addEventListener("click", (e) => {
    e.stopPropagation();
    showGlueDropdown(input, input.value, pickById);
  });
  input.addEventListener("input", () =>
    showGlueDropdown(input, input.value, pickById),
  );
  input.addEventListener("blur", () => {
    // The list's `mousedown` runs before blur and calls preventDefault,
    // so a click on a list item won't reach this path. A blur from
    // tabbing / clicking outside should commit whatever's in the box.
    setTimeout(() => {
      if (glueDropdownInput === input) hideGlueDropdown();
    }, 0);
    commitText();
  });
  input.addEventListener("keydown", (e) => {
    const ul = glueDropdown;
    const open = isGlueDropdownOpenFor(input);
    if (e.key === "ArrowDown") {
      e.preventDefault();
      if (!open) {
        showGlueDropdown(input, input.value, pickById);
      } else if (ul.children.length > 0) {
        glueDropdownActiveIndex = Math.min(
          glueDropdownActiveIndex + 1,
          ul.children.length - 1,
        );
        updateGlueDropdownActive();
      }
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      if (open) {
        glueDropdownActiveIndex = Math.max(glueDropdownActiveIndex - 1, 0);
        updateGlueDropdownActive();
      }
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (open && glueDropdownActiveIndex >= 0) {
        const li = ul.children[glueDropdownActiveIndex];
        const idStr = li.dataset.glueId;
        pickById(idStr === "" ? null : Number(idStr));
      } else {
        commitText();
      }
      hideGlueDropdown();
    } else if (e.key === "Escape") {
      input.value = lastCommitted;
      hideGlueDropdown();
      input.blur();
    }
  });

  return input;
}

// ── Glue list (for per-side dropdowns) and interactions panel ─────────

// Refresh the cached glue list used by the per-side glue combobox
// inputs in the tileset table. The combobox renders directly from
// `currentGlueList` each time it opens, so there's nothing more to
// rebuild here.
function refreshGlueList() {
  currentGlueList = null;
  if (sim) {
    try {
      const glues = sim.glueList();
      currentGlueList = Array.isArray(glues) ? glues : null;
    } catch (e) {
      console.warn("glueList() failed:", e);
    }
  }
}

function rebuildGlueInteractionsPanel() {
  glueTableBody.replaceChildren();
  if (!sim) {
    gluePanel.hidden = true;
    return;
  }
  let schema;
  try {
    schema = sim.interactionSchema() || {};
  } catch (e) {
    gluePanel.hidden = true;
    console.warn("interactionSchema() failed:", e);
    return;
  }
  let interactions;
  try {
    interactions = sim.glueInteractions();
  } catch (e) {
    gluePanel.hidden = true;
    console.warn("glueInteractions() failed:", e);
    return;
  }
  if (!Array.isArray(interactions) || interactions.length === 0) {
    gluePanel.hidden = true;
    return;
  }
  gluePanel.hidden = false;
  glueCountEl.textContent =
    `${interactions.length} pair${interactions.length === 1 ? "" : "s"}`;

  const thDg = $("glue-th-dg");
  const thDs = $("glue-th-ds");
  thDg.textContent = schema.label_dg || "Strength";
  if (schema.has_ds) {
    thDs.textContent = schema.label_ds || "ΔS";
    glueTable.classList.remove("no-ds");
  } else {
    glueTable.classList.add("no-ds");
  }

  for (const iax of interactions) {
    const tr = document.createElement("tr");

    const tdPair = document.createElement("td");
    tdPair.className = "col-pair";
    const aName = iax.a_name && iax.a_name.length ? iax.a_name : `#${iax.a}`;
    const bName = iax.b_name && iax.b_name.length ? iax.b_name : `#${iax.b}`;
    if (iax.matching) {
      tdPair.append(document.createTextNode(aName));
      const tag = document.createElement("span");
      tag.className = "self-tag";
      tag.textContent = "(self)";
      tdPair.append(tag);
    } else {
      tdPair.append(document.createTextNode(`${aName} — ${bName}`));
    }
    const idHint = document.createElement("span");
    idHint.className = "glue-id";
    idHint.textContent = iax.matching
      ? `(#${iax.a})`
      : `(#${iax.a}–#${iax.b})`;
    tdPair.append(idHint);

    const tdDg = document.createElement("td");
    if (editFeatures.glue_interaction) {
      const input = makeNumberInput(
        iax.dg,
        4,
        (v, revert) => {
          applyEdit(
            () => {
              sim.setGlueInteraction(iax.a, iax.b, v, schema.has_ds ? iax.ds : undefined);
              // For KBlock, changing a glue ΔG (especially the
              // self-pair, which is the blocker–glue binding energy)
              // shifts free-blocker and free-tile concentrations.
              refreshDerivedConcentrationDisplays();
            },
            revert,
            `Failed to set ${schema.label_dg} for pair (#${iax.a}, #${iax.b})`,
          );
          iax.dg = v;
        },
        { allowNegative: true },
      );
      tdDg.append(wirePropagateButton(input, {
        digits: 4,
        kind: "glue pairs",
        label: schema.label_dg || "Strength",
        allowNegative: true,
        errorPrefix: `Failed to bulk-set ${schema.label_dg || "strength"}`,
        listMatches: (sourceValue) => {
          const key = formatForInput(sourceValue, 4);
          let list;
          try { list = sim.glueInteractions(); } catch { return []; }
          if (!Array.isArray(list)) return [];
          return list.filter((x) => formatForInput(x.dg, 4) === key);
        },
        applySetter: (m, v) =>
          sim.setGlueInteraction(m.a, m.b, v, schema.has_ds ? m.ds : undefined),
        afterApply: () => {
          rebuildGlueInteractionsPanel();
          refreshDerivedConcentrationDisplays();
        },
      }));
    } else {
      tdDg.textContent = fmt(iax.dg, 4);
    }

    const tdDs = document.createElement("td");
    tdDs.className = "col-ds";
    if (schema.has_ds) {
      if (editFeatures.glue_interaction) {
        const input = makeNumberInput(
          iax.ds ?? 0,
          6,
          (v, revert) => {
            applyEdit(
              () => sim.setGlueInteraction(iax.a, iax.b, iax.dg, v),
              revert,
              `Failed to set ${schema.label_ds || "ΔS"} for pair (#${iax.a}, #${iax.b})`,
            );
            iax.ds = v;
          },
          { allowNegative: true },
        );
        tdDs.append(wirePropagateButton(input, {
          digits: 6,
          kind: "glue pairs",
          label: schema.label_ds || "ΔS",
          allowNegative: true,
          errorPrefix: `Failed to bulk-set ${schema.label_ds || "ΔS"}`,
          listMatches: (sourceValue) => {
            const key = formatForInput(sourceValue, 6);
            let list;
            try { list = sim.glueInteractions(); } catch { return []; }
            if (!Array.isArray(list)) return [];
            return list.filter(
              (x) => x.ds != null && formatForInput(x.ds, 6) === key,
            );
          },
          applySetter: (m, v) => sim.setGlueInteraction(m.a, m.b, m.dg, v),
          afterApply: () => {
            rebuildGlueInteractionsPanel();
            refreshDerivedConcentrationDisplays();
          },
        }));
      } else {
        tdDs.textContent = iax.ds != null ? fmt(iax.ds, 6) : "";
      }
    }

    tr.append(tdPair, tdDg, tdDs);
    glueTableBody.append(tr);
  }
}

// ── Blocker panel (KBlock) ────────────────────────────────────────────
//
// One row per glue that has a blocker definition (any glue with a name).
// Total blocker concentration is editable; free concentration is
// computed by the model and read back after each edit. Hidden for
// non-KBlock models, which return an empty list from `blockerList()`.

function rebuildBlockerPanel() {
  blockerTableBody.replaceChildren();
  if (!sim) {
    blockerPanel.hidden = true;
    return;
  }
  let blockers;
  try {
    blockers = sim.blockerList();
  } catch (e) {
    blockerPanel.hidden = true;
    console.warn("blockerList() failed:", e);
    return;
  }
  if (!Array.isArray(blockers) || blockers.length === 0) {
    blockerPanel.hidden = true;
    return;
  }
  blockerPanel.hidden = false;
  blockerCountEl.textContent = `${blockers.length} glue${blockers.length === 1 ? "" : "s"}`;

  for (const b of blockers) {
    const tr = document.createElement("tr");

    const tdGlue = document.createElement("td");
    tdGlue.className = "col-glue-name";
    const glueName = b.glue_name && b.glue_name.length ? b.glue_name : `#${b.glue_id}`;
    tdGlue.append(document.createTextNode(glueName));
    const idHint = document.createElement("span");
    idHint.className = "glue-id";
    idHint.textContent = `(#${b.glue_id})`;
    tdGlue.append(idHint);

    const tdConc = document.createElement("td");
    tdConc.className = "col-conc";
    const input = makeNumberInput(b.concentration, 3, (v, revert) => {
      applyEdit(
        () => {
          sim.setBlockerConcentration(b.glue_id, v);
          // Changing one blocker shifts every tile's free concentration
          // (and other glues' free-blocker concentrations through the
          // shared tile-glue usage), so refresh both panels.
          refreshDerivedConcentrationDisplays();
        },
        revert,
        `Failed to set blocker concentration for glue #${b.glue_id}`,
      );
    });
    tdConc.append(wirePropagateButton(input, {
      digits: 3,
      kind: "blockers",
      label: "Conc.",
      errorPrefix: "Failed to bulk-set blocker concentration",
      listMatches: (sourceValue) => {
        const key = formatForInput(sourceValue, 3);
        let list;
        try { list = sim.blockerList(); } catch { return []; }
        if (!Array.isArray(list)) return [];
        return list.filter(
          (x) => x.concentration != null && formatForInput(x.concentration, 3) === key,
        );
      },
      applySetter: (m, v) => sim.setBlockerConcentration(m.glue_id, v),
      afterApply: () => {
        rebuildBlockerPanel();
        refreshDerivedConcentrationDisplays();
      },
    }));

    const tdFree = document.createElement("td");
    tdFree.className = "col-free-conc";
    tdFree.dataset.glueId = String(b.glue_id);
    tdFree.textContent = fmt(b.free_concentration, 3);

    tr.append(tdGlue, tdConc, tdFree);
    blockerTableBody.append(tr);
  }
}

// Re-read free-concentration values from the wasm side and patch the
// existing tileset / blocker rows in place. Avoids a full panel rebuild
// (which would re-render every input and steal focus from the user
// mid-edit). Called after any edit that perturbs the blocker
// equilibrium — tile concentration, blocker concentration, or glue ΔG.
function refreshDerivedConcentrationDisplays() {
  if (!sim) return;
  try {
    const tiles = sim.tileSet();
    if (Array.isArray(tiles)) {
      for (const t of tiles) {
        if (t.free_concentration == null) continue;
        const cell = tilesetTableBody.querySelector(
          `td.col-free-conc[data-tile-id="${t.id}"]`,
        );
        if (cell) cell.textContent = fmt(t.free_concentration, 3);
      }
    }
  } catch (e) {
    console.warn("tileSet refresh failed:", e);
  }
  try {
    const blockers = sim.blockerList();
    if (Array.isArray(blockers)) {
      for (const b of blockers) {
        const cell = blockerTableBody.querySelector(
          `td.col-free-conc[data-glue-id="${b.glue_id}"]`,
        );
        if (cell) cell.textContent = fmt(b.free_concentration, 3);
      }
    }
  } catch (e) {
    console.warn("blockerList refresh failed:", e);
  }
}

function renderTileInfoForTile(tile, opts = {}) {
  const pinned = !!opts.pinned;
  tileInfoEl.classList.remove("empty");
  tileInfoEl.classList.toggle("pinned", pinned);

  const swatch = document.createElement("span");
  swatch.className = "swatch";
  swatch.style.background = rgbaCss(tile.color);

  const text = document.createElement("span");
  const name = tile.name && tile.name.length ? tile.name : "(unnamed)";
  let s = `tile #${tile.id} ${name}`;
  if (tile.concentration != null) s += `   c=${fmt(tile.concentration, 3)}`;
  text.textContent = s;

  tileInfoEl.replaceChildren(swatch, text);

  const hint = document.createElement("span");
  hint.className = "pin-hint";
  hint.textContent = pinned ? "click to unpin" : "click to pin";
  tileInfoEl.append(hint);
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

// Auto-fit the cell size when the viewport changes. Debounce so a
// drag-resize doesn't thrash the canvas (each `canvas.width` write
// clears the buffer; the next frame will repaint, but at 60 Hz a wide
// drag can briefly flash empty).
let resizeTimer = null;
window.addEventListener("resize", () => {
  if (!sim) return;
  if (resizeTimer != null) clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    resizeTimer = null;
    resizeCanvasFor(sim);
    if (!rafScheduled) {
      rafScheduled = true;
      requestAnimationFrame(frame);
    }
  }, 80);
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

// ── Tile info on hover / click ────────────────────────────────────────
//
// `pinned` holds either null (track the cursor live) or a {x, y} cell.
// While pinned, mousemove updates are suppressed; clicking the same
// cell toggles pinning off, clicking a different cell re-pins there.
let pinnedCell = null;

// Returns the canvas-pixel coordinates under the pointer, or null if the
// pointer is outside the canvas. Use with `sim.cellInfoAtPixel(...)` to
// get the storage cell — for tube canvases the storage coords are not a
// simple `floor(px/scale)` of the pixel position.
function canvasToPixel(event) {
  if (!sim) return null;
  const rect = canvas.getBoundingClientRect();
  if (rect.width === 0 || rect.height === 0) return null;
  const cssX = event.clientX - rect.left;
  const cssY = event.clientY - rect.top;
  const px = (cssX / rect.width) * canvas.width;
  const py = (cssY / rect.height) * canvas.height;
  if (px < 0 || py < 0 || px >= canvas.width || py >= canvas.height) {
    return null;
  }
  return { px: Math.floor(px), py: Math.floor(py) };
}

function rgbaCss(c) {
  // c is a 4-byte typed array or plain array; alpha 0 = transparent
  // empty cell, render as a dim hatch instead of vanishing.
  if (!c || c.length < 4) return "transparent";
  return `rgba(${c[0]}, ${c[1]}, ${c[2]}, ${(c[3] / 255).toFixed(3)})`;
}

// `cell` is either { px, py } (pixel coords from a hover/click event) or
// { x, y } (storage coords for a pinned cell that we want to re-display
// each frame as its tile may flip).
function renderTileInfo(cell, opts = {}) {
  const pinned = !!opts.pinned;
  if (!sim || !cell) {
    tileInfoEl.classList.add("empty");
    tileInfoEl.classList.remove("pinned");
    tileInfoEl.textContent = sim
      ? "Hover the canvas to inspect a tile."
      : "Load a tileset to inspect tiles.";
    return;
  }
  let info;
  try {
    if (cell.px !== undefined) {
      info = sim.cellInfoAtPixel(cell.px, cell.py, currentScale);
    } else {
      info = sim.cellInfo(cell.x, cell.y);
    }
  } catch (e) {
    tileInfoEl.classList.add("empty");
    tileInfoEl.textContent = `cellInfo error: ${e.message || e}`;
    return;
  }
  if (!info) {
    tileInfoEl.classList.add("empty");
    tileInfoEl.textContent = "(out of bounds)";
    return;
  }
  tileInfoEl.classList.remove("empty");
  tileInfoEl.classList.toggle("pinned", pinned);

  const swatch = document.createElement("span");
  swatch.className = "swatch";
  swatch.style.background = info.tile === 0 ? "transparent" : rgbaCss(info.color);
  if (info.tile === 0) {
    swatch.style.background =
      "repeating-linear-gradient(45deg, #2a2a30 0 4px, #1a1a1d 4px 8px)";
  }

  const text = document.createElement("span");
  const name = info.name && info.name.length ? info.name : "(unnamed)";
  text.textContent =
    info.tile === 0
      ? `(${info.x}, ${info.y}) — empty`
      : `(${info.x}, ${info.y}) — tile #${info.tile} ${name}`;

  tileInfoEl.replaceChildren(swatch, text);

  const hint = document.createElement("span");
  hint.className = "pin-hint";
  hint.textContent = pinned ? "click to unpin" : "click to pin";
  tileInfoEl.append(hint);
}

canvas.addEventListener("mousemove", (e) => {
  if (pinnedCell) return;
  const px = canvasToPixel(e);
  renderTileInfo(px, { pinned: false });
});

canvas.addEventListener("mouseleave", () => {
  if (pinnedCell) return;
  // If a tileset row is pinned, restore that view rather than clearing.
  if (pinnedTilesetId != null) {
    const tile = currentTileSetData?.find((t) => t.id === pinnedTilesetId);
    if (tile) {
      renderTileInfoForTile(tile, { pinned: true });
      return;
    }
  }
  renderTileInfo(null);
});

canvas.addEventListener("click", (e) => {
  const px = canvasToPixel(e);
  if (!px) return;
  let info;
  try {
    info = sim.cellInfoAtPixel(px.px, px.py, currentScale);
  } catch {
    return;
  }
  if (!info) return; // empty triangle — ignore click
  const cell = { x: info.x, y: info.y };
  if (pinnedCell && pinnedCell.x === cell.x && pinnedCell.y === cell.y) {
    pinnedCell = null;
    renderTileInfo(cell, { pinned: false });
  } else {
    // Clear any tileset-row pin so only one pinned thing is highlighted.
    pinnedTilesetId = null;
    clearAllPinnedRows();
    pinnedCell = cell;
    renderTileInfo(cell, { pinned: true });
  }
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
