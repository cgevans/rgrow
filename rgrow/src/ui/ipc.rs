use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IpcMessage {
    Init(InitMessage),
    Update(UpdateMessage),
    Resize(ResizeMessage),
    Close,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitMessage {
    pub width: u32,
    pub height: u32,
    pub tile_colors: Vec<[u8; 4]>,
    pub block: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateMessage {
    pub frame_data: Vec<u8>,
    pub time: f64,
    pub total_events: u64,
    pub n_tiles: u32,
    pub mismatches: u32,
    pub scale: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResizeMessage {
    pub width: u32,
    pub height: u32,
}
