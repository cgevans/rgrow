use serde::{Deserialize, Serialize};

use crate::system::{BlockerData, EditableFeatures, GlueInteractionData, InteractionSchema};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterInfo {
    pub name: String,
    pub units: String,
    pub default_increment: f64,
    pub min_value: Option<f64>,
    pub max_value: Option<f64>,
    pub description: Option<String>,
    pub current_value: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IpcMessage {
    Init(InitMessage),
    Update(UpdateNotification),
    Resize(ResizeMessage),
    /// Full editable-model state, sent once after init and re-sent after
    /// every edit so the GUI can rebuild its panels.
    ModelSnapshot(ModelSnapshot),
    Ready,
    Close,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitMessage {
    pub width: u32,
    pub height: u32,
    pub tile_colors: Vec<[u8; 4]>,
    pub block: Option<usize>,
    pub shm_path: String,
    pub shm_size: usize,
    /// Shared-memory region carrying the per-frame tile-id grid (`u32`
    /// row-major over `raw_array`), pushed only while inspection is on.
    pub grid_shm_path: String,
    pub grid_shm_size: usize,
    pub start_paused: bool,
    pub model_name: String,
    pub has_temperature: bool,
    pub initial_temperature: Option<f64>,
    pub parameters: Vec<ParameterInfo>,
    pub initial_timescale: Option<f64>,
    pub initial_max_events_per_sec: Option<u64>,
}

/// Notification that new frame data is available in shared memory.
/// The actual pixel data is in the shared memory region.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateNotification {
    pub frame_width: u32,
    pub frame_height: u32,
    pub time: f64,
    pub total_events: u64,
    pub n_tiles: u32,
    pub mismatches: u32,
    pub energy: f64,
    pub scale: usize,
    pub data_len: usize,
    /// Whether the grid shm holds a fresh tile-id grid this frame (only
    /// when inspection is enabled, to avoid the per-frame copy otherwise).
    pub grid_included: bool,
    /// Storage-grid shape of `raw_array` (rows, cols) and the byte length of
    /// the `u32` grid in the grid shm.
    pub grid_rows: u32,
    pub grid_cols: u32,
    pub grid_data_len: usize,
    /// Pixel size of one tile cell at `scale` (`State::tile_size_px`).
    pub subcell_px: u32,
    /// `true` if the canvas renders diamonds (zigzag tube), `false` for the
    /// square-shaped canvases.
    pub tile_shape_diamond: bool,
    /// Rendered frame extent in subcells (`State::frame_size_subcells`).
    pub frame_subcells_w: u32,
    pub frame_subcells_h: u32,
    /// True only for plain-square canvases where storage `(row, col)` maps
    /// linearly to pixels `(col*scale, row*scale)`. False for the sheared
    /// (diagonal tube) and staggered (zigzag/diamond tube) canvases, where
    /// the GUI's linear overlay mapping would be wrong, so the GUI disables
    /// positioned labels / hover there (click-to-place still works via the
    /// sim's canvas-aware `pixel_to_storage`).
    pub overlay_linear: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResizeMessage {
    pub width: u32,
    pub height: u32,
}

/// One tile row for the tileset panel. `tri_colors` are the NSEW triangle
/// colors used to draw the reference sprite; `color` is the flat fallback.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TileSnapshot {
    pub id: u32,
    pub name: String,
    pub color: [u8; 4],
    pub tri_colors: [[u8; 4]; 4],
    pub concentration: Option<f64>,
    pub free_concentration: Option<f64>,
    pub edge_glues: [Option<String>; 4],
    pub edge_glue_ids: [Option<u32>; 4],
}

/// One glue for the per-side glue dropdowns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlueSnapshot {
    pub id: u32,
    pub name: String,
}

/// Full editable-model state pushed to the GUI. Built from `EditableSystem`
/// + `TileBondInfo` trait calls on the concrete model in the sim loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSnapshot {
    pub model_name: String,
    pub editable: EditableFeatures,
    pub schema: InteractionSchema,
    pub tiles: Vec<TileSnapshot>,
    pub glues: Vec<GlueSnapshot>,
    pub interactions: Vec<GlueInteractionData>,
    pub blockers: Vec<BlockerData>,
    /// Right-shift to turn a raw grid value into a base/display tile id
    /// (`base = raw >> tile_id_shift`). `0` for models that store base ids
    /// directly; `4` for KBlock.
    pub tile_id_shift: u32,
}

/// Control messages sent from GUI to simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControlMessage {
    Pause,
    Resume,
    Step {
        events: u64,
    },
    SetMaxEventsPerSec(Option<u64>), // None = unlimited
    SetTimescale(Option<f64>),       // None = unlimited, else sim_time/real_time
    SetTemperature(f64),             // Deprecated, use SetParameter instead
    SetParameter {
        name: String,
        value: f64,
    },
    SetShowMismatches(bool),
    /// Start/stop pushing the per-frame tile-id grid (for hover/overlay).
    SetInspection(bool),
    /// Ask the sim to re-send a `ModelSnapshot`.
    RequestSnapshot,
    SetTileConcentration {
        id: u32,
        value: f64,
    },
    SetTileEdgeGlue {
        id: u32,
        side: u32,
        glue_id: Option<u32>,
    },
    SetGlueInteraction {
        a: u32,
        b: u32,
        dg: f64,
        ds: Option<f64>,
    },
    SetBlockerConcentration {
        glue_id: u32,
        value: f64,
    },
    /// Place `tile` at the storage cell whose rendered area contains pixel
    /// `(px, py)` at `scale`. The sim does the (canvas-aware) inverse map.
    SetPointAtPixel {
        px: u32,
        py: u32,
        scale: u32,
        tile: u32,
    },
    /// Place `tile` at storage grid `(x=col, y=row)`.
    SetPoint {
        x: u32,
        y: u32,
        tile: u32,
    },
    /// Load an xgrow saved-flake into the canvas, centered with optional
    /// extra offsets.
    LoadXgrowSeed {
        text: String,
        offset_i: Option<i32>,
        offset_j: Option<i32>,
    },
}
