#[cfg(feature = "ui")]
use crate::ui::ipc::{InitMessage, IpcMessage, UpdateNotification};
use memmap2::MmapMut;
use std::fs::OpenOptions;
use std::io::Write;
use std::os::unix::net::UnixStream;
use std::path::Path;

pub struct IpcClient {
    stream: UnixStream,
    shm: Option<MmapMut>,
    shm_path: String,
}

impl IpcClient {
    pub fn connect<P: AsRef<Path>>(socket_path: P) -> Result<Self, std::io::Error> {
        let stream = UnixStream::connect(socket_path)?;
        Ok(IpcClient {
            stream,
            shm: None,
            shm_path: String::new(),
        })
    }

    pub fn setup_shm(&mut self, shm_path: &str, size: usize) -> Result<(), std::io::Error> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(shm_path)?;
        file.set_len(size as u64)?;
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        self.shm = Some(mmap);
        self.shm_path = shm_path.to_string();
        Ok(())
    }

    pub fn send_init(&mut self, init: &InitMessage) -> Result<(), Box<dyn std::error::Error>> {
        self.setup_shm(&init.shm_path, init.shm_size)?;
        let msg = IpcMessage::Init(init.clone());
        let serialized = bincode::serialize(&msg)?;
        let len = serialized.len() as u64;
        self.stream.write_all(&len.to_le_bytes())?;
        self.stream.write_all(&serialized)?;
        self.stream.flush()?;
        Ok(())
    }

    pub fn send_frame(
        &mut self,
        frame_data: &[u8],
        notification: UpdateNotification,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(ref mut shm) = self.shm {
            shm[..frame_data.len()].copy_from_slice(frame_data);
        } else {
            return Err("Shared memory not initialized".into());
        }

        let msg = IpcMessage::Update(notification);
        let serialized = bincode::serialize(&msg)?;
        let len = serialized.len() as u64;
        self.stream.write_all(&len.to_le_bytes())?;
        self.stream.write_all(&serialized)?;
        self.stream.flush()?;
        Ok(())
    }

    pub fn send_close(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let msg = IpcMessage::Close;
        let serialized = bincode::serialize(&msg)?;
        let len = serialized.len() as u64;
        self.stream.write_all(&len.to_le_bytes())?;
        self.stream.write_all(&serialized)?;
        self.stream.flush()?;
        Ok(())
    }
}

impl Drop for IpcClient {
    fn drop(&mut self) {
        if !self.shm_path.is_empty() {
            let _ = std::fs::remove_file(&self.shm_path);
        }
    }
}

pub struct ShmReader {
    mmap: MmapMut,
}

impl ShmReader {
    pub fn open<P: AsRef<Path>>(path: P, size: usize) -> Result<Self, std::io::Error> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        if file.metadata()?.len() < size as u64 {
            file.set_len(size as u64)?;
        }
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        Ok(ShmReader { mmap })
    }

    pub fn read(&self, len: usize) -> &[u8] {
        &self.mmap[..len]
    }
}
