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
use rgrow::canvas::{Canvas, PointSafe2};
use rgrow::models::kblock::{
    GlueIdentifier, KBlock, KBlockParams, KBlockTile, StrenOrSeq, TileIdentifier,
};
use rgrow::models::sdc1d::{SDCParams, SDCStrand, SingleOrMultiScaffold, SDC};
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
/// no glue on that side or the model has no notion of one. `edge_glue_ids`
/// carries the same information as numeric ids, which the editing UI
/// uses to drive per-side glue dropdowns without having to round-trip
/// names back through `bond_names`.
#[derive(Serialize)]
pub struct TileInfo {
    pub id: u32,
    pub name: String,
    pub color: [u8; 4],
    pub concentration: Option<f64>,
    pub stoic: Option<f64>,
    pub edge_glues: [Option<String>; 4],
    pub edge_glue_ids: [Option<u32>; 4],
}

/// Per-glue info for the per-side glue dropdown in the tileset table.
#[derive(Serialize)]
pub struct GlueInfo {
    pub id: u32,
    pub name: String,
}

/// One non-zero entry in the model's pair interaction matrix. `dg` is the
/// model's primary number — KTAM: dimensionless strength; SDC2D / KBlock:
/// ΔG in kcal/mol. `ds` is only populated for SDC2D (ΔS in kcal/(mol·K)).
/// `matching` flags KTAM's special case where `(g, g)` reads from
/// `glue_strengths[g]` instead of `glue_links[(g, g)]`.
#[derive(Serialize)]
pub struct GlueInteraction {
    pub a: u32,
    pub a_name: String,
    pub b: u32,
    pub b_name: String,
    pub matching: bool,
    pub dg: f64,
    pub ds: Option<f64>,
}

/// Describes the per-pair editing schema for the loaded model — what the
/// numbers mean, how to label them, and whether the second column exists.
#[derive(Serialize, Default)]
pub struct InteractionSchema {
    pub label_dg: String,
    pub has_ds: bool,
    pub label_ds: Option<String>,
}

/// Capability flags for the editing UI. JS reads this once after a sim
/// loads and decides which cells become editable. Avoids matching on
/// `modelName()` strings in the JS layer.
#[derive(Serialize, Default)]
pub struct EditableFeatures {
    pub tile_concentration: bool,
    pub tile_edge_glue: bool,
    pub glue_interaction: bool,
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
    ///
    /// Tries the `WebExample` schema first (its `model` tag namespaces
    /// browser-only tileset shapes like `sdc2d-square` and `sdc1d` so they
    /// don't collide with the generic `TileSet` model field). Falls back to
    /// `TileSet::from_json` for everything else.
    #[wasm_bindgen(js_name = fromJson)]
    pub fn from_json(json: &str) -> Result<Sim, JsError> {
        console_error_panic_hook::set_once();
        match WebExample::from_json(json) {
            Ok(example) => example.into_sim().map_err(js_err),
            Err(example_err) => match TileSet::from_json(json) {
                Ok(tileset) => {
                    let (sys, state) = tileset.create_system_and_state().map_err(js_err)?;
                    Ok(Sim { sys, state })
                }
                Err(tileset_err) => Err(JsError::new(&format!(
                    "JSON is neither a web example nor a tileset. web example: {example_err}; tileset: {tileset_err}"
                ))),
            },
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
        let (frame_w, frame_h) = self.state.frame_size_px(scale as u32);
        let mut frame = vec![0u8; (frame_w as usize) * (frame_h as usize) * 4];
        let _stats = render_frame_dyn(&self.sys, &self.state, scale, show_mismatches, &mut frame);
        js_sys::Uint8ClampedArray::from(&frame[..])
    }

    /// Pixel size of a frame at the given scale. Use to size the
    /// `<canvas>` and the JS-side `Uint8ClampedArray`.
    #[wasm_bindgen(js_name = frameSize)]
    pub fn frame_size(&self, scale: usize) -> Result<JsValue, JsError> {
        let (w, h) = self.state.frame_size_px(scale as u32);
        let s = FrameSize {
            width: w,
            height: h,
        };
        serde_wasm_bindgen::to_value(&s).map_err(js_err)
    }

    /// Pre-scale frame extent in subcells. For square canvases this is
    /// `(ncols, nrows)`. For tube canvases this is the physical extent of
    /// the rendered frame (sheared/staggered) in subcell units; multiply by
    /// the canvas's per-subcell pixel size to get the pixel frame.
    #[wasm_bindgen(js_name = canvasSize)]
    pub fn canvas_size(&self) -> Result<JsValue, JsError> {
        let (w, h) = self.state.frame_size_subcells();
        let s = CanvasSize {
            width: w,
            height: h,
        };
        serde_wasm_bindgen::to_value(&s).map_err(js_err)
    }

    /// Subcells per scale-unit for the loaded canvas. `1` for square /
    /// diagonal-tube canvases, `2` for the zigzag-tube (diamond) canvas
    /// where each tile occupies a 2×2 subcell block. JS uses this to drive
    /// cell-size computations that need the pixel size of a single tile,
    /// not a single subcell.
    #[wasm_bindgen(js_name = subcellsPerTile)]
    pub fn subcells_per_tile(&self) -> u32 {
        use rgrow::canvas::TileShape;
        match self.state.tile_shape() {
            TileShape::Square => 1,
            TileShape::Diamond => 2,
        }
    }

    /// Storage-grid shape `(rows, cols)` of the underlying tile array.
    /// Useful for callers that want to enumerate raw storage independent
    /// of how the canvas displays it (e.g. the tube canvases).
    #[wasm_bindgen(js_name = tileStorageShape)]
    pub fn tile_storage_shape(&self) -> Result<JsValue, JsError> {
        let arr = self.state.raw_array();
        let s = CanvasSize {
            width: arr.ncols() as u32,
            height: arr.nrows() as u32,
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

    /// Flat tile-id grid for the current canvas, row-major over the
    /// underlying `raw_array` (length `rows * cols`). Use
    /// `tileGridShape()` to recover the shape — for tube canvases the
    /// raw array is smaller than `canvasSize()` and the painter only
    /// fills the corresponding region.
    #[wasm_bindgen(js_name = tileGrid)]
    pub fn tile_grid(&self) -> js_sys::Uint32Array {
        let arr = self.state.raw_array();
        let nrows = arr.nrows();
        let ncols = arr.ncols();
        let mut flat = Vec::with_capacity(nrows * ncols);
        for y in 0..nrows {
            for x in 0..ncols {
                flat.push(arr[[y, x]]);
            }
        }
        js_sys::Uint32Array::from(&flat[..])
    }

    /// Shape of `tileGrid()` as `{width: cols, height: rows}`.
    #[wasm_bindgen(js_name = tileGridShape)]
    pub fn tile_grid_shape(&self) -> Result<JsValue, JsError> {
        let arr = self.state.raw_array();
        let s = CanvasSize {
            width: arr.ncols() as u32,
            height: arr.nrows() as u32,
        };
        serde_wasm_bindgen::to_value(&s).map_err(js_err)
    }

    /// Display label for tile id `id`. Goes through `sys.tile_name` so
    /// KBlock's blocker-state variants resolve to their underlying tile
    /// name. Empty string for the empty tile (id 0) and for ids the
    /// model does not name.
    #[wasm_bindgen(js_name = tileLabel)]
    pub fn tile_label(&self, id: u32) -> String {
        if id == 0 {
            return String::new();
        }
        match &self.sys {
            // KBlock's `tile_name` maps the raw id through `tile_index`
            // (id >> 4), so it's safe for any id the canvas can hold.
            SystemEnum::KBlock(_) => self.sys.tile_name(id as Tile).to_string(),
            _ => {
                let names = self.sys.tile_names();
                names.get(id as usize).cloned().unwrap_or_default()
            }
        }
    }

    /// Information about the cell at pixel position `(px, py)` rendered at
    /// `scale`: which tile is there, its name (if any), and its color.
    /// Returns `null` if the pixel falls outside any tile (e.g. the empty
    /// triangles of a sheared tube canvas, or beyond the frame). The
    /// returned `x` / `y` are storage-grid coordinates.
    #[wasm_bindgen(js_name = cellInfoAtPixel)]
    pub fn cell_info_at_pixel(&self, px: u32, py: u32, scale: u32) -> Result<JsValue, JsError> {
        let Some(p) = self.state.pixel_to_storage(px, py, scale) else {
            return Ok(JsValue::NULL);
        };
        let (row, col) = p.0;
        Ok(self.cell_info_inner(col as u32, row as u32))
    }

    /// Information about the cell at storage grid `(x=col, y=row)`. Storage
    /// coordinates index `raw_array` directly. For tube canvases, prefer
    /// `cellInfoAtPixel` to avoid having to do storage-coord conversion in
    /// JS. Returns `null` for out-of-bounds.
    #[wasm_bindgen(js_name = cellInfo)]
    pub fn cell_info(&self, x: u32, y: u32) -> Result<JsValue, JsError> {
        let arr = self.state.raw_array();
        if (x as usize) >= arr.ncols() || (y as usize) >= arr.nrows() {
            return Ok(JsValue::NULL);
        }
        Ok(self.cell_info_inner(x, y))
    }

    fn cell_info_inner(&self, col: u32, row: u32) -> JsValue {
        let arr = self.state.raw_array();
        let tile = arr[[row as usize, col as usize]];
        let names = self.sys.tile_names();
        let colors = self.sys.tile_colors();
        let name = names.get(tile as usize).cloned().unwrap_or_default();
        let color = colors.get(tile as usize).copied().unwrap_or([0, 0, 0, 0]);
        let info = CellInfo {
            x: col,
            y: row,
            tile,
            name,
            color,
        };
        serde_wasm_bindgen::to_value(&info).unwrap_or(JsValue::NULL)
    }

    /// Per-cell label-anchor info: for every non-empty cell, returns the
    /// pixel center of that cell at `scale`, plus its tile id. Used by JS
    /// to overlay tile-name labels on the canvas without re-implementing
    /// the canvas's storage→pixel transform. Returned as a flat
    /// `Float32Array` of triples `(cx_px, cy_px, tile_id)`.
    #[wasm_bindgen(js_name = labelAnchors)]
    pub fn label_anchors(&self, scale: u32) -> js_sys::Float32Array {
        let arr = self.state.raw_array();
        let tile_size = self.state.tile_size_px(scale) as f32;
        let half = tile_size * 0.5;
        let mut out: Vec<f32> = Vec::with_capacity(arr.len() * 3);
        for ((row, col), &tileid) in arr.indexed_iter() {
            if tileid == 0 {
                continue;
            }
            let p = PointSafe2((row, col));
            let (ox, oy) = self.state.tile_origin_px(p, scale);
            out.push(ox as f32 + half);
            out.push(oy as f32 + half);
            out.push(tileid as f32);
        }
        js_sys::Float32Array::from(&out[..])
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
            let edge_glue_ids = edges.map(|g| g.map(|gid| gid as u32));
            out.push(TileInfo {
                id: id as u32,
                name: names.get(id).cloned().unwrap_or_default(),
                color: colors.get(id).copied().unwrap_or([0, 0, 0, 0]),
                concentration: concs.as_ref().and_then(|v| v.get(id).copied()),
                stoic: stoics.as_ref().and_then(|v: &Vec<f64>| v.get(id).copied()),
                edge_glues,
                edge_glue_ids,
            });
        }
        serde_wasm_bindgen::to_value(&out).map_err(js_err)
    }

    /// All glues defined by the loaded tileset (id 0 is omitted). Used to
    /// populate per-side glue dropdowns in the tileset table.
    #[wasm_bindgen(js_name = glueList)]
    pub fn glue_list(&self) -> Result<JsValue, JsError> {
        // ATAM's `bond_names` is `todo!()`, so guard.
        let names: &[String] = match &self.sys {
            SystemEnum::ATAM(_) => &[],
            _ => self.sys.bond_names(),
        };
        let mut out: Vec<GlueInfo> = Vec::with_capacity(names.len().saturating_sub(1));
        for (id, name) in names.iter().enumerate().skip(1) {
            if name.is_empty() {
                // Skip placeholder slots — these are usually internal padding.
                continue;
            }
            out.push(GlueInfo {
                id: id as u32,
                name: name.clone(),
            });
        }
        serde_wasm_bindgen::to_value(&out).map_err(js_err)
    }

    /// Per-pair interaction labels and arity for the loaded model. JS uses
    /// this to lay out the glue-interactions panel.
    #[wasm_bindgen(js_name = interactionSchema)]
    pub fn interaction_schema(&self) -> Result<JsValue, JsError> {
        let s = match &self.sys {
            SystemEnum::KTAM(_) => InteractionSchema {
                label_dg: "Strength".to_string(),
                has_ds: false,
                label_ds: None,
            },
            SystemEnum::SDC2DSquare(_) => InteractionSchema {
                label_dg: "ΔG (kcal/mol)".to_string(),
                has_ds: true,
                label_ds: Some("ΔS (kcal/(mol·K))".to_string()),
            },
            SystemEnum::KBlock(_) => InteractionSchema {
                label_dg: "ΔG (kcal/mol)".to_string(),
                has_ds: false,
                label_ds: None,
            },
            _ => InteractionSchema::default(),
        };
        serde_wasm_bindgen::to_value(&s).map_err(js_err)
    }

    /// Non-zero glue-glue interactions. For KTAM, includes self-pairs
    /// `(g, g)` whose value lives in `glue_strengths` (flagged with
    /// `matching: true`). Pairs with `a < b` come from the link matrix.
    #[wasm_bindgen(js_name = glueInteractions)]
    pub fn glue_interactions(&self) -> Result<JsValue, JsError> {
        let mut out: Vec<GlueInteraction> = Vec::new();
        let names: &[String] = match &self.sys {
            SystemEnum::ATAM(_) => &[],
            _ => self.sys.bond_names(),
        };
        let name_at = |id: usize| -> String { names.get(id).cloned().unwrap_or_default() };
        match &self.sys {
            SystemEnum::KTAM(k) => {
                let n = k.glue_strengths.len();
                for g in 1..n {
                    let v = k.glue_strengths[g];
                    if v != 0.0 {
                        out.push(GlueInteraction {
                            a: g as u32,
                            a_name: name_at(g),
                            b: g as u32,
                            b_name: name_at(g),
                            matching: true,
                            dg: v,
                            ds: None,
                        });
                    }
                }
                let m = k.glue_links.nrows().min(k.glue_links.ncols());
                for a in 1..m {
                    for b in (a + 1)..m {
                        let v = k.glue_links[(a, b)];
                        if v != 0.0 {
                            out.push(GlueInteraction {
                                a: a as u32,
                                a_name: name_at(a),
                                b: b as u32,
                                b_name: name_at(b),
                                matching: false,
                                dg: v,
                                ds: None,
                            });
                        }
                    }
                }
            }
            SystemEnum::SDC2DSquare(s) => {
                let m = s
                    .delta_g_matrix
                    .nrows()
                    .min(s.delta_g_matrix.ncols())
                    .min(s.entropy_matrix.nrows())
                    .min(s.entropy_matrix.ncols());
                for a in 1..m {
                    for b in a..m {
                        let dg: f64 = s.delta_g_matrix[(a, b)].into();
                        let ds: f64 = s.entropy_matrix[(a, b)].into();
                        if dg != 0.0 || ds != 0.0 {
                            out.push(GlueInteraction {
                                a: a as u32,
                                a_name: name_at(a),
                                b: b as u32,
                                b_name: name_at(b),
                                matching: a == b,
                                dg,
                                ds: Some(ds),
                            });
                        }
                    }
                }
            }
            SystemEnum::KBlock(k) => {
                let links = k.glue_links();
                let m = links.nrows().min(links.ncols());
                for a in 1..m {
                    for b in a..m {
                        let dg: f64 = links[(a, b)].into();
                        if dg != 0.0 {
                            out.push(GlueInteraction {
                                a: a as u32,
                                a_name: name_at(a),
                                b: b as u32,
                                b_name: name_at(b),
                                matching: a == b,
                                dg,
                                ds: None,
                            });
                        }
                    }
                }
            }
            _ => {}
        }
        serde_wasm_bindgen::to_value(&out).map_err(js_err)
    }

    /// Editing capabilities of the loaded model.
    #[wasm_bindgen(js_name = editableFeatures)]
    pub fn editable_features(&self) -> Result<JsValue, JsError> {
        let f = match &self.sys {
            SystemEnum::KTAM(_) => EditableFeatures {
                tile_concentration: true,
                tile_edge_glue: true,
                glue_interaction: true,
            },
            SystemEnum::SDC2DSquare(_) => EditableFeatures {
                tile_concentration: true,
                tile_edge_glue: true,
                glue_interaction: true,
            },
            SystemEnum::KBlock(_) => EditableFeatures {
                tile_concentration: false,
                tile_edge_glue: false,
                glue_interaction: true,
            },
            _ => EditableFeatures::default(),
        };
        serde_wasm_bindgen::to_value(&f).map_err(js_err)
    }

    /// Set tile `id`'s concentration. KTAM and SDC2DSquare only.
    #[wasm_bindgen(js_name = setTileConcentration)]
    pub fn set_tile_concentration(&mut self, id: u32, value: f64) -> Result<(), JsError> {
        if !value.is_finite() || value < 0.0 {
            return Err(JsError::new(
                "setTileConcentration: value must be a non-negative finite number",
            ));
        }
        let idx = id as usize;
        match &mut self.sys {
            SystemEnum::KTAM(k) => {
                if idx == 0 || idx >= k.tile_concs.len() {
                    return Err(JsError::new("setTileConcentration: tile id out of range"));
                }
                k.tile_concs[idx] = value;
                k.update_system();
            }
            SystemEnum::SDC2DSquare(s) => {
                if idx == 0 || idx >= s.strand_concentration.len() {
                    return Err(JsError::new("setTileConcentration: strand id out of range"));
                }
                s.strand_concentration[idx] = rgrow::units::Molar::from(value);
                s.update_system();
            }
            _ => {
                return Err(JsError::new(
                    "setTileConcentration: not supported for this model",
                ));
            }
        }
        self.sys
            .update_state(&mut self.state, &rgrow::system::NeededUpdate::NonZero);
        Ok(())
    }

    /// Set tile `id`'s glue on `side` (0=N, 1=E, 2=S, 3=W). `glue_id` of
    /// `None` (or `undefined` from JS) clears the glue. KTAM only.
    #[wasm_bindgen(js_name = setTileEdgeGlue)]
    pub fn set_tile_edge_glue(
        &mut self,
        id: u32,
        side: u32,
        glue_id: Option<u32>,
    ) -> Result<(), JsError> {
        if side >= 4 {
            return Err(JsError::new("setTileEdgeGlue: side must be 0..=3"));
        }
        let g = glue_id.unwrap_or(0) as usize;
        let idx = id as usize;
        match &mut self.sys {
            SystemEnum::KTAM(k) => {
                if idx == 0 || idx >= k.tile_edges.nrows() {
                    return Err(JsError::new("setTileEdgeGlue: tile id out of range"));
                }
                if g >= k.glue_strengths.len() {
                    return Err(JsError::new("setTileEdgeGlue: glue id out of range"));
                }
                k.tile_edges[(idx, side as usize)] = g;
                k.update_system();
            }
            SystemEnum::SDC2DSquare(s) => {
                // strand_glues columns are NORTH/EAST/SOUTH/WEST (0..=3)
                // — same as the JS-side `side` convention. The bottom
                // (scaffold) glue lives in column 4 and isn't editable
                // from the per-side cells.
                if idx == 0 || idx >= s.strand_glues.nrows() {
                    return Err(JsError::new("setTileEdgeGlue: strand id out of range"));
                }
                if g >= s.glue_names.len() {
                    return Err(JsError::new("setTileEdgeGlue: glue id out of range"));
                }
                s.strand_glues[(idx, side as usize)] = g;
                s.update_system();
            }
            _ => {
                return Err(JsError::new(
                    "setTileEdgeGlue: not supported for this model",
                ));
            }
        }
        self.sys
            .update_state(&mut self.state, &rgrow::system::NeededUpdate::NonZero);
        Ok(())
    }

    /// Set the interaction value(s) for the glue pair `(a, b)`. The matrix
    /// is symmetric, so `(b, a)` is mirrored automatically. KTAM ignores
    /// `ds`; SDC2D requires it; KBlock ignores it.
    #[wasm_bindgen(js_name = setGlueInteraction)]
    pub fn set_glue_interaction(
        &mut self,
        a: u32,
        b: u32,
        dg: f64,
        ds: Option<f64>,
    ) -> Result<(), JsError> {
        if !dg.is_finite() {
            return Err(JsError::new("setGlueInteraction: dg must be finite"));
        }
        if let Some(d) = ds {
            if !d.is_finite() {
                return Err(JsError::new("setGlueInteraction: ds must be finite"));
            }
        }
        let ai = a as usize;
        let bi = b as usize;
        if ai == 0 || bi == 0 {
            return Err(JsError::new(
                "setGlueInteraction: glue id 0 is the null glue",
            ));
        }
        match &mut self.sys {
            SystemEnum::KTAM(k) => {
                let n = k.glue_strengths.len();
                if ai >= n || bi >= n {
                    return Err(JsError::new("setGlueInteraction: glue id out of range"));
                }
                if ai == bi {
                    k.glue_strengths[ai] = dg;
                } else {
                    k.glue_links[(ai, bi)] = dg;
                    k.glue_links[(bi, ai)] = dg;
                }
                k.update_system();
            }
            SystemEnum::SDC2DSquare(s) => {
                let nrows = s.delta_g_matrix.nrows();
                let ncols = s.delta_g_matrix.ncols();
                if ai >= nrows || bi >= ncols || ai >= s.entropy_matrix.nrows() {
                    return Err(JsError::new("setGlueInteraction: glue id out of range"));
                }
                let dg_val = rgrow::units::KcalPerMol::from(dg);
                s.delta_g_matrix[(ai, bi)] = dg_val;
                s.delta_g_matrix[(bi, ai)] = dg_val;
                if let Some(d) = ds {
                    let ds_val = rgrow::units::KcalPerMolKelvin::from(d);
                    s.entropy_matrix[(ai, bi)] = ds_val;
                    s.entropy_matrix[(bi, ai)] = ds_val;
                }
                s.update_system();
            }
            SystemEnum::KBlock(k) => {
                let links = k.glue_links();
                if ai >= links.nrows() || bi >= links.ncols() {
                    return Err(JsError::new("setGlueInteraction: glue id out of range"));
                }
                k.set_glue_link(ai, bi, rgrow::units::KcalPerMol::from(dg));
                k.update();
            }
            _ => {
                return Err(JsError::new(
                    "setGlueInteraction: not supported for this model",
                ));
            }
        }
        self.sys
            .update_state(&mut self.state, &rgrow::system::NeededUpdate::NonZero);
        Ok(())
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

    /// Place a specific tile id at storage grid `(x=col, y=row)`.
    /// Out-of-bounds returns an error. For tube canvases, prefer
    /// `setPointAtPixel` to avoid having to do storage-coord conversion in
    /// JS.
    #[wasm_bindgen(js_name = setPoint)]
    pub fn set_point(&mut self, x: u32, y: u32, tile: u32) -> Result<(), JsError> {
        let arr_shape = {
            let arr = self.state.raw_array();
            (arr.nrows(), arr.ncols())
        };
        if (x as usize) >= arr_shape.1 || (y as usize) >= arr_shape.0 {
            return Err(JsError::new("set_point: coordinates out of bounds"));
        }
        let mut arr = self.state.raw_array_mut();
        arr[[y as usize, x as usize]] = tile as Tile;
        Ok(())
    }

    /// Place a specific tile id at the storage cell whose rendered area
    /// contains pixel `(px, py)` at the given `scale`. Returns an error
    /// when the pixel falls outside any tile (e.g. an empty triangle of
    /// a sheared tube canvas).
    #[wasm_bindgen(js_name = setPointAtPixel)]
    pub fn set_point_at_pixel(
        &mut self,
        px: u32,
        py: u32,
        scale: u32,
        tile: u32,
    ) -> Result<(), JsError> {
        let p = self
            .state
            .pixel_to_storage(px, py, scale)
            .ok_or_else(|| JsError::new("set_point_at_pixel: pixel is outside any tile"))?;
        let (row, col) = p.0;
        let mut arr = self.state.raw_array_mut();
        arr[[row, col]] = tile as Tile;
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
    #[serde(rename = "sdc1d")]
    Sdc1d(WebSdc1d),
    #[serde(rename = "kblock")]
    KBlock(WebKBlock),
}

impl WebExample {
    fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    fn into_sim(self) -> Result<Sim, rgrow::base::GrowError> {
        match self {
            WebExample::Sdc2dSquare(example) => example.into_sim(),
            WebExample::Sdc1d(example) => example.into_sim(),
            WebExample::KBlock(example) => example.into_sim(),
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

#[derive(Deserialize)]
struct WebSdc1d {
    strands: Vec<WebSdc1dStrand>,
    /// Single 1D scaffold, repeated `n_scaffolds` times along the i-axis.
    scaffold: Vec<Option<String>>,
    scaffold_concentration: f64,
    glue_dg_s: Vec<WebGlueEnergy>,
    k_f: f64,
    #[serde(default = "default_k_n")]
    k_n: f64,
    #[serde(default = "default_k_c")]
    k_c: f64,
    temperature: f64,
    #[serde(default = "default_n_scaffolds")]
    n_scaffolds: usize,
    #[serde(default)]
    junction_penalty_dg: Option<f64>,
    #[serde(default)]
    junction_penalty_ds: Option<f64>,
    #[serde(default = "default_sdc1d_canvas_type")]
    canvas_type: WebCanvasType,
}

#[derive(Deserialize)]
struct WebSdc1dStrand {
    name: Option<String>,
    color: Option<String>,
    concentration: f64,
    left_glue: Option<String>,
    btm_glue: Option<String>,
    right_glue: Option<String>,
}

impl From<WebSdc1dStrand> for SDCStrand {
    fn from(strand: WebSdc1dStrand) -> Self {
        SDCStrand {
            name: strand.name,
            color: strand.color,
            concentration: strand.concentration,
            btm_glue: strand.btm_glue,
            left_glue: strand.left_glue,
            right_glue: strand.right_glue,
        }
    }
}

fn default_k_n() -> f64 {
    1e5
}

fn default_k_c() -> f64 {
    1e4
}

fn default_n_scaffolds() -> usize {
    32
}

fn default_sdc1d_canvas_type() -> WebCanvasType {
    WebCanvasType::SquareCompact
}

impl WebSdc1d {
    fn into_sim(self) -> Result<Sim, rgrow::base::GrowError> {
        let scaffold_len = self.scaffold.len();
        if scaffold_len == 0 {
            return Err(rgrow::base::GrowError::NotSupported(
                "sdc1d: scaffold must have at least one position".to_string(),
            ));
        }
        let scaffold = SingleOrMultiScaffold::Single(self.scaffold);
        let glue_dg_s = self
            .glue_dg_s
            .into_iter()
            .map(|g| (RefOrPair::Ref(g.name), GsOrSeq::GS((g.dg37, g.ds))))
            .collect::<HashMap<_, _>>();
        let params = SDCParams {
            strands: self.strands.into_iter().map(Into::into).collect(),
            scaffold,
            scaffold_concentration: self.scaffold_concentration,
            glue_dg_s,
            k_f: self.k_f,
            k_n: self.k_n,
            k_c: self.k_c,
            temperature: self.temperature,
            junction_penalty_dg: self.junction_penalty_dg.map(rgrow::units::KcalPerMol::from),
            junction_penalty_ds: self
                .junction_penalty_ds
                .map(rgrow::units::KcalPerMolKelvin::from),
            quencher_name: None,
            quencher_concentration: 0.0,
            reporter_name: None,
            fluorophore_concentration: 0.0,
        };
        let sys = SDC::from_params(params);
        let n_tile_types = sys.strand_names.len();
        let mut state = StateEnum::empty(
            (self.n_scaffolds, scaffold_len),
            self.canvas_type.into(),
            &TrackingConfig::None,
            n_tile_types,
        )?;
        let sys = SystemEnum::SDC(sys);
        sys.setup_state(&mut state)?;
        sys.update_state(&mut state, &rgrow::system::NeededUpdate::All);
        Ok(Sim { sys, state })
    }
}

#[derive(Deserialize)]
struct WebKBlock {
    tiles: Vec<WebKBlockTile>,
    /// Map glue name → blocker concentration (M).
    blocker_conc: HashMap<String, f64>,
    /// Seed entries as `[row, col, "tile_name"]`.
    seed: Vec<(usize, usize, String)>,
    /// Glue name → either a DNA sequence (string, parsed for ΔG) or a
    /// pre-computed ΔG in kcal/mol (number).
    binding_strength: HashMap<String, WebStrenOrSeq>,
    #[serde(default = "default_kblock_ds_lat")]
    ds_lat: f64,
    #[serde(default = "default_kblock_kf")]
    kf: f64,
    #[serde(default = "default_kblock_temp")]
    temp: f64,
    #[serde(default = "default_kblock_no_pba")]
    no_partially_blocked_attachments: bool,
    #[serde(default)]
    blocker_energy_adj: f64,
    /// Canvas shape `(rows, cols)`. Defaults to a 12-helix tube.
    #[serde(default = "default_kblock_canvas_size")]
    canvas_size: (usize, usize),
    #[serde(default = "default_kblock_canvas_type")]
    canvas_type: WebKBlockCanvasType,
}

#[derive(Deserialize)]
struct WebKBlockTile {
    name: String,
    concentration: f64,
    glues: [String; 4],
    /// Color as `"#RRGGBB"`, an array `[r, g, b, a]`, or omitted for a
    /// random color.
    #[serde(default)]
    color: Option<WebColor>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum WebColor {
    Name(String),
    Rgba([u8; 4]),
}

#[derive(Deserialize)]
#[serde(untagged)]
enum WebStrenOrSeq {
    Sequence(String),
    DG(f64),
}

#[derive(Default, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum WebKBlockCanvasType {
    #[default]
    Tube,
    TubeDiagonals,
}

impl From<WebKBlockCanvasType> for CanvasType {
    fn from(value: WebKBlockCanvasType) -> Self {
        match value {
            WebKBlockCanvasType::Tube => CanvasType::Tube,
            WebKBlockCanvasType::TubeDiagonals => CanvasType::TubeDiagonals,
        }
    }
}

fn default_kblock_ds_lat() -> f64 {
    -14.12 / 1000.0
}

fn default_kblock_kf() -> f64 {
    1.0e6
}

fn default_kblock_temp() -> f64 {
    40.0
}

fn default_kblock_no_pba() -> bool {
    true
}

fn default_kblock_canvas_size() -> (usize, usize) {
    (12, 256)
}

fn default_kblock_canvas_type() -> WebKBlockCanvasType {
    WebKBlockCanvasType::Tube
}

impl WebKBlock {
    fn into_sim(self) -> Result<Sim, rgrow::base::GrowError> {
        let canvas_size = self.canvas_size;
        let canvas_type: CanvasType = self.canvas_type.into();
        let tiles: Vec<KBlockTile> = self
            .tiles
            .into_iter()
            .map(|t| {
                let color = match t.color {
                    Some(WebColor::Name(s)) => {
                        rgrow::colors::get_color(&s).unwrap_or([128, 128, 128, 255])
                    }
                    Some(WebColor::Rgba(c)) => c,
                    None => {
                        rgrow::colors::get_color_or_random(None).unwrap_or([128, 128, 128, 255])
                    }
                };
                KBlockTile {
                    name: t.name,
                    concentration: t.concentration,
                    glues: t.glues,
                    color,
                }
            })
            .collect();
        let blocker_conc: HashMap<GlueIdentifier, rgrow::units::Molar> = self
            .blocker_conc
            .into_iter()
            .map(|(k, v)| (GlueIdentifier::Name(k), rgrow::units::Molar::from(v)))
            .collect();
        let seed: HashMap<(usize, usize), TileIdentifier> = self
            .seed
            .into_iter()
            .map(|(r, c, name)| ((r, c), TileIdentifier::Name(name)))
            .collect();
        let binding_strength: HashMap<String, StrenOrSeq> = self
            .binding_strength
            .into_iter()
            .map(|(k, v)| {
                let s = match v {
                    WebStrenOrSeq::Sequence(seq) => StrenOrSeq::Sequence(seq),
                    WebStrenOrSeq::DG(dg) => StrenOrSeq::DG(rgrow::units::KcalPerMol::from(dg)),
                };
                (k, s)
            })
            .collect();
        let params = KBlockParams {
            tiles,
            blocker_conc,
            seed,
            binding_strength,
            ds_lat: rgrow::units::KcalPerMolKelvin::from(self.ds_lat),
            kf: rgrow::units::PerMolarSecond::from(self.kf),
            temp: rgrow::units::Celsius::from(self.temp),
            no_partially_blocked_attachments: self.no_partially_blocked_attachments,
            blocker_energy_adj: rgrow::units::KcalPerMol::from(self.blocker_energy_adj),
        };
        let sys = KBlock::from(params);
        let n_tile_types = sys.tile_names.len() * 16; // KBlock encodes blocker state in tile id
        let mut state = StateEnum::empty(
            canvas_size,
            canvas_type,
            &TrackingConfig::None,
            n_tile_types,
        )?;
        let sys = SystemEnum::KBlock(sys);
        sys.setup_state(&mut state)?;
        sys.update_state(&mut state, &rgrow::system::NeededUpdate::All);
        Ok(Sim { sys, state })
    }
}
