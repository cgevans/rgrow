use serde::{Deserialize, Serialize};

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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResizeMessage {
    pub width: u32,
    pub height: u32,
}

/// Control messages sent from GUI to simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControlMessage {
    Pause,
    Resume,
    Step { events: u64 },
    SetMaxEventsPerSec(Option<u64>), // None = unlimited
    SetTimescale(Option<f64>),       // None = unlimited, else sim_time/real_time
    SetTemperature(f64),             // Deprecated, use SetParameter instead
    SetParameter { name: String, value: f64 },
}
