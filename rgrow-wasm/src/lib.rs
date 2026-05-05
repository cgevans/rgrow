//! WebAssembly bindings for the `rgrow` tile-assembly simulator.
//!
//! Exposes a `Sim` JS class that owns a `(SystemEnum, StateEnum)` pair
//! and lets the page step the simulation, render frames into a JS-owned
//! `Uint8ClampedArray`, and read/write parameters.
//!
//! The desktop GUI is split across two processes connected by a Unix
//! socket plus shared memory. In the browser we collapse that into one
//! wasm module: stepping and rendering happen on the main thread, JS
//! draws the bytes onto a `<canvas>`.

use std::{any::Any, collections::HashMap};

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

use rgrow::base::Tile;
use rgrow::canvas::Canvas;
use rgrow::models::sdc2d::{SDC2DParams, SDC2DSquare, SDC2DStrand};
use rgrow::models::sdc_common::{GsOrSeq, RefOrPair};
use rgrow::painter::render_frame_dyn;
use rgrow::state::{StateEnum, StateStatus};
use rgrow::system::{
    DynSystem, EvolveBounds, EvolveOutcome, ParameterInfo, SystemEnum, TileBondInfo,
};
use rgrow::tileset::{CanvasType, TileSet, TrackingConfig};

/// Result of a step call. JS reads these fields after each tick.
#[derive(Serialize)]
pub struct StepResult {
    pub outcome: String,
    pub n_tiles: u32,
    pub time: f64,
    pub total_events: u64,
    pub events_this_step: u64,
    pub mismatches: u32,
    pub energy: f64,
}

/// Frame size in pixels. Returned by `frame_size`.
#[derive(Serialize)]
pub struct FrameSize {
    pub width: u32,
    pub height: u32,
}

/// Tile-grid size in cells. Returned by `canvas_size`.
#[derive(Serialize)]
pub struct CanvasSize {
    pub width: u32,
    pub height: u32,
}

/// Per-cell info for hover/click tooltips on the web UI.
#[derive(Serialize)]
pub struct CellInfo {
    pub x: u32,
    pub y: u32,
    pub tile: u32,
    pub name: String,
    pub color: [u8; 4],
}

/// Per-tile info for the tileset panel below the canvas.
///
/// `concentration` and `stoic` are `None` for models that do not expose
/// per-tile concentration data (currently anything other than KTAM).
/// `edge_glues` is `[N, E, S, W]`; an entry is `None` when the tile has
/// no glue on that side or the model has no notion of one.
#[derive(Serialize)]
pub struct TileInfo {
    pub id: u32,
    pub name: String,
    pub color: [u8; 4],
    pub concentration: Option<f64>,
    pub stoic: Option<f64>,
    pub edge_glues: [Option<String>; 4],
}

#[wasm_bindgen]
pub struct Sim {
    sys: SystemEnum,
    state: StateEnum,
}

fn outcome_to_string(o: &EvolveOutcome) -> String {
    format!("{:?}", o)
}

fn js_err<E: std::fmt::Display>(e: E) -> JsError {
    JsError::new(&e.to_string())
}

#[wasm_bindgen]
impl Sim {
    /// Construct a simulation from a YAML tileset string.
    #[wasm_bindgen(js_name = fromYaml)]
    pub fn from_yaml(yaml: &str) -> Result<Sim, JsError> {
        console_error_panic_hook::set_once();
        let tileset = TileSet::from_yaml(yaml).map_err(js_err)?;
        let (sys, state) = tileset.create_system_and_state().map_err(js_err)?;
        Ok(Sim { sys, state })
    }

    /// Construct a simulation from a JSON tileset string.
    #[wasm_bindgen(js_name = fromJson)]
    pub fn from_json(json: &str) -> Result<Sim, JsError> {
        console_error_panic_hook::set_once();
        match TileSet::from_json(json) {
            Ok(tileset) => {
                let (sys, state) = tileset.create_system_and_state().map_err(js_err)?;
                Ok(Sim { sys, state })
            }
            Err(tileset_err) => WebExample::from_json(json)
                .map_err(|example_err| {
                    JsError::new(&format!(
                        "JSON is neither a tileset nor a web example. tileset: {tileset_err}; web example: {example_err}"
                    ))
                })?
                .into_sim()
                .map_err(js_err),
        }
    }

    /// Construct a simulation from an Xgrow-format tileset string.
    #[wasm_bindgen(js_name = fromXgrow)]
    pub fn from_xgrow(text: &str) -> Result<Sim, JsError> {
        console_error_panic_hook::set_once();
        let tileset = rgrow::parser_xgrow::parse_xgrow_string(text).map_err(js_err)?;
        let (sys, state) = tileset.create_system_and_state().map_err(js_err)?;
        Ok(Sim { sys, state })
    }

    // ── stepping ────────────────────────────────────────────────────────

    /// Run for at most `events` events. Returns step stats (JsValue).
    #[wasm_bindgen(js_name = stepForEvents)]
    pub fn step_for_events(&mut self, events: u64) -> Result<JsValue, JsError> {
        self.run_bounds(EvolveBounds {
            for_events: Some(events),
            ..Default::default()
        })
    }

    /// Run for at most `dt` simulation seconds.
    #[wasm_bindgen(js_name = stepForSimTime)]
    pub fn step_for_sim_time(&mut self, dt: f64) -> Result<JsValue, JsError> {
        self.run_bounds(EvolveBounds {
            for_time: Some(dt),
            ..Default::default()
        })
    }

    /// Run for at most `ms` real-time milliseconds. Uses
    /// `web_time::Instant` (which calls `performance.now()` on wasm) for
    /// the wall-time check inside `evolve`.
    #[wasm_bindgen(js_name = stepForRealMs)]
    pub fn step_for_real_ms(&mut self, ms: f64) -> Result<JsValue, JsError> {
        let dur = std::time::Duration::from_micros((ms * 1000.0) as u64);
        self.run_bounds(EvolveBounds {
            for_wall_time: Some(dur),
            ..Default::default()
        })
    }

    /// Run with a combination of bounds. Each argument may be `undefined`
    /// to disable that bound. Used by the web UI to combine a per-frame
    /// wall-time budget with optional timescale (sim-time) and
    /// events-per-second caps.
    ///
    /// `for_events` is taken as `f64` for JS ergonomics — the integer
    /// budgets used by the UI fit comfortably in f64's 53-bit mantissa.
    #[wasm_bindgen(js_name = stepWithBounds)]
    pub fn step_with_bounds(
        &mut self,
        for_events: Option<f64>,
        for_time: Option<f64>,
        for_wall_ms: Option<f64>,
    ) -> Result<JsValue, JsError> {
        let bounds = EvolveBounds {
            for_events: for_events.map(|e| e.max(0.0) as u64),
            for_time,
            for_wall_time: for_wall_ms
                .map(|ms| std::time::Duration::from_micros((ms.max(0.0) * 1000.0) as u64)),
            ..Default::default()
        };
        self.run_bounds(bounds)
    }

    fn run_bounds(&mut self, bounds: EvolveBounds) -> Result<JsValue, JsError> {
        let events_before = self.state.total_events();
        let outcome = self.sys.evolve(&mut self.state, bounds).map_err(js_err)?;
        let events_this_step = self.state.total_events() - events_before;
        let result = StepResult {
            outcome: outcome_to_string(&outcome),
            n_tiles: self.state.n_tiles(),
            time: self.state.time().into(),
            total_events: self.state.total_events(),
            events_this_step,
            mismatches: self.sys.calc_mismatches(&self.state) as u32,
            energy: self.state.energy(),
        };
        serde_wasm_bindgen::to_value(&result).map_err(js_err)
    }

    // ── rendering ───────────────────────────────────────────────────────

    /// Render the current state and return the RGBA bytes as a fresh
    /// `Uint8ClampedArray`.
    ///
    /// wasm-bindgen converts the returned `Vec<u8>` into a JS typed
    /// array; this involves one memcpy per frame from wasm linear memory
    /// into a JS-owned `ArrayBuffer`. At the canvas sizes typical of
    /// rgrow demos this is well under a millisecond and is dominated by
    /// the simulation step. A zero-copy variant taking a JS-supplied
    /// pointer is feasible (see `__wbindgen_malloc`) but adds enough
    /// JS-side complexity that we defer it.
    #[wasm_bindgen(js_name = renderAlloc)]
    pub fn render_alloc(
        &mut self,
        scale: usize,
        show_mismatches: bool,
    ) -> js_sys::Uint8ClampedArray {
        let (w, h) = self.state.draw_size();
        let frame_w = (w * scale as u32) as usize;
        let frame_h = (h * scale as u32) as usize;
        let mut frame = vec![0u8; frame_w * frame_h * 4];
        let _stats = render_frame_dyn(&self.sys, &self.state, scale, show_mismatches, &mut frame);
        js_sys::Uint8ClampedArray::from(&frame[..])
    }

    /// Pixel size of a frame at the given scale. Use to size the
    /// `<canvas>` and the JS-side `Uint8ClampedArray`.
    #[wasm_bindgen(js_name = frameSize)]
    pub fn frame_size(&self, scale: usize) -> Result<JsValue, JsError> {
        let (w, h) = self.state.draw_size();
        let s = FrameSize {
            width: w * scale as u32,
            height: h * scale as u32,
        };
        serde_wasm_bindgen::to_value(&s).map_err(js_err)
    }

    /// Tile-grid (canvas) size in cells, before scaling.
    #[wasm_bindgen(js_name = canvasSize)]
    pub fn canvas_size(&self) -> Result<JsValue, JsError> {
        let (w, h) = self.state.draw_size();
        let s = CanvasSize {
            width: w,
            height: h,
        };
        serde_wasm_bindgen::to_value(&s).map_err(js_err)
    }

    // ── stats accessors ─────────────────────────────────────────────────

    /// Number of tiles currently on the canvas.
    #[wasm_bindgen(js_name = nTiles)]
    pub fn n_tiles(&self) -> u32 {
        self.state.n_tiles()
    }

    /// Current simulation time, in seconds.
    pub fn time(&self) -> f64 {
        self.state.time().into()
    }

    /// Cumulative number of events that have occurred.
    #[wasm_bindgen(js_name = totalEvents)]
    pub fn total_events(&self) -> u64 {
        self.state.total_events()
    }

    /// Count of (boundary) mismatches in the current state.
    pub fn mismatches(&self) -> u32 {
        self.sys.calc_mismatches(&self.state) as u32
    }

    /// Current bonded energy.
    pub fn energy(&self) -> f64 {
        self.state.energy()
    }

    /// Short human-readable model name (e.g. `"kTAM"`, `"KBlock"`).
    #[wasm_bindgen(js_name = modelName)]
    pub fn model_name(&self) -> String {
        let info = self.sys.system_info();
        // Mirrors `System::extract_model_name`, copied here because that
        // method is on the `System` trait (per-variant) rather than
        // `DynSystem` (on `SystemEnum`).
        if info.starts_with("kTAM") {
            "kTAM".to_string()
        } else if info.starts_with("aTAM") {
            "aTAM".to_string()
        } else if info.starts_with("Old kTAM") || info.starts_with("OldkTAM") {
            "Old kTAM".to_string()
        } else if info.starts_with("SDC") || info.contains("SDC") {
            "SDC".to_string()
        } else if info.starts_with("KBlock") {
            "KBlock".to_string()
        } else {
            "Unknown".to_string()
        }
    }

    // ── parameter control ───────────────────────────────────────────────

    /// Returns the model's tunable parameters (name, units, min, max,
    /// current value, default increment) as a JSON-shaped array.
    pub fn parameters(&self) -> Result<JsValue, JsError> {
        let mut params: Vec<ParameterInfo> = self.sys.list_parameters();
        // Refresh `current_value` for parameters that report a live f64.
        for p in &mut params {
            if let Ok(v) = self.sys.get_param(&p.name) {
                if let Some(f) = v.downcast_ref::<f64>() {
                    p.current_value = *f;
                }
            }
        }
        serde_wasm_bindgen::to_value(&params).map_err(js_err)
    }

    /// Set a parameter by name. The corresponding state-update
    /// (recomputing rates etc.) is applied automatically.
    #[wasm_bindgen(js_name = setParameter)]
    pub fn set_parameter(&mut self, name: &str, value: f64) -> Result<(), JsError> {
        let needed: rgrow::system::NeededUpdate = self
            .sys
            .set_param(name, Box::new(value) as Box<dyn Any>)
            .map_err(js_err)?;
        self.sys.update_state(&mut self.state, &needed);
        Ok(())
    }

    // ── tile colors / direct edits ──────────────────────────────────────

    /// RGBA tile colors, indexed by tile id (flat `Uint8Array`).
    #[wasm_bindgen(js_name = tileColors)]
    pub fn tile_colors(&self) -> js_sys::Uint8Array {
        let cs: &Vec<[u8; 4]> = self.sys.tile_colors();
        let mut flat = Vec::with_capacity(cs.len() * 4);
        for c in cs {
            flat.extend_from_slice(c);
        }
        js_sys::Uint8Array::from(&flat[..])
    }

    /// Tile names indexed by tile id (parallel to `tileColors`). Empty
    /// strings are returned for ids the model does not name.
    #[wasm_bindgen(js_name = tileNames)]
    pub fn tile_names(&self) -> Vec<String> {
        self.sys.tile_names().to_vec()
    }

    /// Information about the cell at grid `(x, y)`: which tile is there,
    /// its name (if any), and its color. Returns `null` for out-of-bounds
    /// coordinates so the JS side can no-op gracefully when the mouse
    /// drifts outside the canvas during a hover update.
    #[wasm_bindgen(js_name = cellInfo)]
    pub fn cell_info(&self, x: u32, y: u32) -> Result<JsValue, JsError> {
        let (w, h) = self.state.draw_size();
        if x >= w || y >= h {
            return Ok(JsValue::NULL);
        }
        let tile = self.state.raw_array()[[y as usize, x as usize]] as u32;
        let names = self.sys.tile_names();
        let colors = self.sys.tile_colors();
        let name = names.get(tile as usize).cloned().unwrap_or_default();
        let color = colors.get(tile as usize).copied().unwrap_or([0, 0, 0, 0]);
        let info = CellInfo {
            x,
            y,
            tile,
            name,
            color,
        };
        serde_wasm_bindgen::to_value(&info).map_err(js_err)
    }

    /// Full tileset inventory: one `TileInfo` per non-empty tile id (id 0
    /// is the empty tile and is omitted). Concentration and stoichiometry
    /// are populated only for models that expose them; glue names are
    /// populated for models that implement `tile_edge_glues` (KTAM,
    /// SDC2DSquare).
    #[wasm_bindgen(js_name = tileSet)]
    pub fn tile_set(&self) -> Result<JsValue, JsError> {
        let names = self.sys.tile_names();
        let colors = self.sys.tile_colors();
        let n = names.len().min(colors.len());
        // Concentrations are only available on models that implement
        // `SystemInfo`. `bond_names` is `todo!()` on ATAM, so we only call
        // it when we know it's safe (KTAM, SDC2DSquare).
        let (concs, stoics, has_bond_names) = match &self.sys {
            SystemEnum::KTAM(k) => {
                use rgrow::system::SystemInfo;
                (Some(k.tile_concs()), None, true)
            }
            SystemEnum::SDC2DSquare(_) => (None, None, true),
            _ => (None, None, false),
        };
        let bond_names: &[String] = if has_bond_names {
            self.sys.bond_names()
        } else {
            &[]
        };
        let mut out: Vec<TileInfo> = Vec::with_capacity(n.saturating_sub(1));
        for id in 1..n {
            let edges = self.sys.tile_edge_glues(id as Tile);
            let edge_glues = edges.map(|g| {
                g.and_then(|gid| bond_names.get(gid).cloned())
                    .filter(|s| !s.is_empty())
            });
            out.push(TileInfo {
                id: id as u32,
                name: names.get(id).cloned().unwrap_or_default(),
                color: colors.get(id).copied().unwrap_or([0, 0, 0, 0]),
                concentration: concs.as_ref().and_then(|v| v.get(id).copied()),
                stoic: stoics.as_ref().and_then(|v: &Vec<f64>| v.get(id).copied()),
                edge_glues,
            });
        }
        serde_wasm_bindgen::to_value(&out).map_err(js_err)
    }

    /// Render a `size`×`size` RGBA sprite for tile id `id` (using the
    /// model's `tile_pixels` so compound tiles render with their proper
    /// per-side colors). Returned bytes are suitable for
    /// `new ImageData(bytes, size, size)` in JS.
    #[wasm_bindgen(js_name = tilePixels)]
    pub fn tile_pixels(&self, id: u32, size: u32) -> js_sys::Uint8ClampedArray {
        let sprite = self.sys.tile_pixels(id as Tile, size as usize);
        js_sys::Uint8ClampedArray::from(&sprite.pixels[..])
    }

    /// Place a specific tile id at grid `(x, y)`. Out-of-bounds returns
    /// an error.
    #[wasm_bindgen(js_name = setPoint)]
    pub fn set_point(&mut self, x: u32, y: u32, tile: u32) -> Result<(), JsError> {
        let (w, h) = self.state.draw_size();
        if x >= w || y >= h {
            return Err(JsError::new("set_point: coordinates out of bounds"));
        }
        // The Canvas trait exposes raw_array_mut; we update the cell
        // directly. (Rate / RateStore is updated lazily on next event.)
        let mut arr = self.state.raw_array_mut();
        arr[[y as usize, x as usize]] = tile as Tile;
        Ok(())
    }

    /// Load an Xgrow `importfile` (a `.seed` file produced by xgrow's
    /// state dump) into the current canvas. The flake is centered on
    /// the canvas, mirroring xgrow's `(size - flake_size) / 2`
    /// translation, with optional `offset_i`/`offset_j` extra
    /// translation. After applying, rates are recomputed for the whole
    /// canvas. Returns the placed flake's edge length in cells.
    #[wasm_bindgen(js_name = loadXgrowSeed)]
    pub fn load_xgrow_seed(
        &mut self,
        text: &str,
        offset_i: Option<i32>,
        offset_j: Option<i32>,
    ) -> Result<u32, JsError> {
        let grid = parse_xgrow_seed(text).map_err(js_err)?;
        let flake_size = grid.len();
        if flake_size == 0 {
            return Err(JsError::new("xgrow seed: empty flake"));
        }
        let (w, h) = self.state.draw_size();
        let canvas_w = w as i32;
        let canvas_h = h as i32;
        let translate_i = (canvas_h - flake_size as i32) / 2 + offset_i.unwrap_or(0);
        let translate_j = (canvas_w - flake_size as i32) / 2 + offset_j.unwrap_or(0);
        {
            let mut arr = self.state.raw_array_mut();
            for (i, row) in grid.iter().enumerate() {
                for (j, &t) in row.iter().enumerate() {
                    let y = translate_i + i as i32;
                    let x = translate_j + j as i32;
                    if y < 0 || y >= canvas_h || x < 0 || x >= canvas_w {
                        continue;
                    }
                    arr[[y as usize, x as usize]] = t;
                }
            }
        }
        self.sys
            .update_state(&mut self.state, &rgrow::system::NeededUpdate::All);
        Ok(flake_size as u32)
    }
}

/// Parse the tile-grid out of an Xgrow saved-flake (`.seed`) file.
///
/// Format (per xgrow.c, `read_flake_file`):
///
/// ```text
/// flake{N}={ ...
/// [ ... stats ... ],...
/// [ ... per-glue values ... ],...
/// [ t11 t12 ... t1n ; ... ; tn1 ... tnn ] };
/// ```
///
/// We skip the first two `],...`-terminated bracketed sections, then
/// read the third bracketed section as a square grid. Rows are
/// separated by `;`.
fn parse_xgrow_seed(text: &str) -> Result<Vec<Vec<Tile>>, String> {
    // Strip xgrow's `...` line-continuation markers — they're noise.
    let cleaned = text.replace("...", " ");
    // Skip everything up to the third `[`.
    let mut rest = cleaned.as_str();
    for n in 0..3 {
        let start = rest
            .find('[')
            .ok_or_else(|| format!("xgrow seed: missing `[` (#{})", n + 1))?;
        rest = &rest[start + 1..];
        if n < 2 {
            // Skip to the closing `]` of this section.
            let end = rest
                .find(']')
                .ok_or_else(|| format!("xgrow seed: missing `]` (#{})", n + 1))?;
            rest = &rest[end + 1..];
        }
    }
    // `rest` is now the grid contents up to the closing `]`.
    let end = rest
        .find(']')
        .ok_or_else(|| "xgrow seed: missing closing `]` for grid".to_string())?;
    let grid_text = &rest[..end];

    let mut grid: Vec<Vec<Tile>> = Vec::new();
    for raw_row in grid_text.split(';') {
        let row: Vec<Tile> = raw_row
            .split_whitespace()
            .map(|tok| {
                tok.parse::<u64>()
                    .map(|v| v as Tile)
                    .map_err(|e| format!("xgrow seed: bad tile id `{tok}`: {e}"))
            })
            .collect::<Result<_, _>>()?;
        if !row.is_empty() {
            grid.push(row);
        }
    }
    if grid.is_empty() {
        return Err("xgrow seed: no rows parsed".into());
    }
    let w = grid[0].len();
    if !grid.iter().all(|r| r.len() == w) {
        return Err(format!(
            "xgrow seed: rows are not all the same length (expected {w})"
        ));
    }
    Ok(grid)
}

#[derive(Deserialize)]
#[serde(tag = "model")]
enum WebExample {
    #[serde(rename = "sdc2d-square")]
    Sdc2dSquare(WebSdc2dSquare),
}

impl WebExample {
    fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    fn into_sim(self) -> Result<Sim, rgrow::base::GrowError> {
        match self {
            WebExample::Sdc2dSquare(example) => example.into_sim(),
        }
    }
}

#[derive(Deserialize)]
struct WebSdc2dSquare {
    strands: Vec<WebSdc2dStrand>,
    scaffold: Vec<Vec<Option<String>>>,
    scaffold_concentration: f64,
    glue_dg37_ds: Vec<WebGlueEnergy>,
    k_f: f64,
    temperature: f64,
    seed: Vec<(usize, usize, String)>,
    #[serde(default = "default_sdc2d_canvas_type")]
    canvas_type: WebCanvasType,
}

impl WebSdc2dSquare {
    fn into_sim(self) -> Result<Sim, rgrow::base::GrowError> {
        let sys = SDC2DSquare::from_params(SDC2DParams {
            strands: self.strands.into_iter().map(Into::into).collect(),
            scaffold: self.scaffold,
            scaffold_concentration: self.scaffold_concentration,
            glue_dg37_ds: self
                .glue_dg37_ds
                .into_iter()
                .map(|g| (RefOrPair::Ref(g.name), GsOrSeq::GS((g.dg37, g.ds))))
                .collect::<HashMap<_, _>>(),
            k_f: self.k_f,
            temperature: self.temperature,
            seed: self.seed,
        });
        let n_tile_types = sys.n_strands();
        let mut state = StateEnum::empty(
            (sys.nrows(), sys.ncols()),
            self.canvas_type.into(),
            &TrackingConfig::None,
            n_tile_types,
        )?;
        let sys = SystemEnum::SDC2DSquare(sys);
        sys.setup_state(&mut state)?;
        sys.update_state(&mut state, &rgrow::system::NeededUpdate::All);
        Ok(Sim { sys, state })
    }
}

#[derive(Deserialize)]
struct WebSdc2dStrand {
    name: Option<String>,
    color: Option<String>,
    concentration: f64,
    north_glue: Option<String>,
    east_glue: Option<String>,
    south_glue: Option<String>,
    west_glue: Option<String>,
    bottom_glue: Option<String>,
}

impl From<WebSdc2dStrand> for SDC2DStrand {
    fn from(strand: WebSdc2dStrand) -> Self {
        SDC2DStrand {
            name: strand.name,
            color: strand.color,
            concentration: strand.concentration,
            north_glue: strand.north_glue,
            east_glue: strand.east_glue,
            south_glue: strand.south_glue,
            west_glue: strand.west_glue,
            bottom_glue: strand.bottom_glue,
        }
    }
}

#[derive(Deserialize)]
struct WebGlueEnergy {
    name: String,
    dg37: f64,
    #[serde(default)]
    ds: f64,
}

#[derive(Default, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WebCanvasType {
    Square,
    #[default]
    SquareCompact,
}

fn default_sdc2d_canvas_type() -> WebCanvasType {
    WebCanvasType::SquareCompact
}

impl From<WebCanvasType> for CanvasType {
    fn from(value: WebCanvasType) -> Self {
        match value {
            WebCanvasType::Square => CanvasType::Square,
            WebCanvasType::SquareCompact => CanvasType::SquareCompact,
        }
    }
}
