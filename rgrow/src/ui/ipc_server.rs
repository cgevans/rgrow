#[cfg(feature = "ui")]
use crate::ui::ipc::IpcMessage;
use std::io::Write;
use std::os::unix::net::UnixStream;
use std::path::Path;

pub struct IpcClient {
    stream: UnixStream,
}

impl IpcClient {
    pub fn connect<P: AsRef<Path>>(socket_path: P) -> Result<Self, std::io::Error> {
        let stream = UnixStream::connect(socket_path)?;
        Ok(IpcClient { stream })
    }

    pub fn send(&mut self, message: &IpcMessage) -> Result<(), Box<dyn std::error::Error>> {
        let serialized = bincode::serialize(message)?;
        let len = serialized.len() as u64;
        self.stream.write_all(&len.to_le_bytes())?;
        self.stream.write_all(&serialized)?;
        self.stream.flush()?;
        Ok(())
    }
}
