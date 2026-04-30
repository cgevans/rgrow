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

use std::any::Any;

use serde::Serialize;
use wasm_bindgen::prelude::*;

use rgrow::base::Tile;
use rgrow::canvas::Canvas;
use rgrow::painter::render_frame_dyn;
use rgrow::state::{StateEnum, StateStatus};
use rgrow::system::{
    DynSystem, EvolveBounds, EvolveOutcome, ParameterInfo, SystemEnum, TileBondInfo,
};
use rgrow::tileset::TileSet;

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
        let tileset = TileSet::from_json(json).map_err(js_err)?;
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
}
