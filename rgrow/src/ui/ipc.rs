use serde::{Deserialize, Serialize};

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
    pub scale: usize,
    pub data_len: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResizeMessage {
    pub width: u32,
    pub height: u32,
}
