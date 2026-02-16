use crate::ui::ipc::{ControlMessage, InitMessage, IpcMessage, UpdateNotification};
use memmap2::MmapMut;
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::path::Path;
use std::time::Duration;

#[cfg(unix)]
use std::os::unix::net::UnixStream;

#[cfg(windows)]
use named_pipe::PipeClient;

pub struct IpcClient {
    #[cfg(unix)]
    stream: UnixStream,
    #[cfg(windows)]
    stream: PipeClient,
    shm: Option<MmapMut>,
    shm_path: String,
}

impl IpcClient {
    pub fn connect<P: AsRef<Path>>(socket_path: P) -> Result<Self, std::io::Error> {
        #[cfg(unix)]
        {
            let stream = UnixStream::connect(socket_path)?;
            Ok(IpcClient {
                stream,
                shm: None,
                shm_path: String::new(),
            })
        }
        #[cfg(windows)]
        {
            let pipe_name = socket_path.as_ref().to_string_lossy();
            let pipe_name = if pipe_name.starts_with(r"\\.\pipe\") {
                pipe_name.to_string()
            } else {
                format!(
                    r"\\.\pipe\{}",
                    pipe_name.replace('/', "_").replace('\\', "_")
                )
            };
            let stream = PipeClient::connect(pipe_name.as_str())?;
            Ok(IpcClient {
                stream,
                shm: None,
                shm_path: String::new(),
            })
        }
    }

    #[cfg(unix)]
    pub fn wait_for_ready(&mut self, timeout: Duration) -> Result<(), Box<dyn std::error::Error>> {
        use std::time::Instant;

        let start = Instant::now();
        self.stream.set_nonblocking(true)?;

        let mut len_bytes = [0u8; 8];
        let mut len_read = 0;

        while len_read < 8 {
            if start.elapsed() > timeout {
                self.stream.set_nonblocking(false)?;
                return Err("Timeout waiting for GUI ready signal".into());
            }
            match self.stream.read(&mut len_bytes[len_read..]) {
                Ok(0) => {
                    self.stream.set_nonblocking(false)?;
                    return Err("Connection closed while waiting for ready".into());
                }
                Ok(n) => len_read += n,
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(50));
                }
                Err(e) => {
                    self.stream.set_nonblocking(false)?;
                    return Err(e.into());
                }
            }
        }

        let len = u64::from_le_bytes(len_bytes) as usize;
        let mut buffer = vec![0u8; len];
        let mut data_read = 0;

        while data_read < len {
            if start.elapsed() > timeout {
                self.stream.set_nonblocking(false)?;
                return Err("Timeout waiting for GUI ready signal".into());
            }
            match self.stream.read(&mut buffer[data_read..]) {
                Ok(0) => {
                    self.stream.set_nonblocking(false)?;
                    return Err("Connection closed while waiting for ready".into());
                }
                Ok(n) => data_read += n,
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(Duration::from_millis(50));
                }
                Err(e) => {
                    self.stream.set_nonblocking(false)?;
                    return Err(e.into());
                }
            }
        }

        self.stream.set_nonblocking(false)?;

        let msg: IpcMessage = bincode::deserialize(&buffer)?;
        match msg {
            IpcMessage::Ready => Ok(()),
            _ => Err("Expected Ready message".into()),
        }
    }

    #[cfg(windows)]
    pub fn wait_for_ready(&mut self, _timeout: Duration) -> Result<(), Box<dyn std::error::Error>> {
        // On Windows, just do a blocking read - the GUI should send Ready quickly
        let mut len_bytes = [0u8; 8];
        self.stream.read_exact(&mut len_bytes)?;

        let len = u64::from_le_bytes(len_bytes) as usize;
        let mut buffer = vec![0u8; len];
        self.stream.read_exact(&mut buffer)?;

        let msg: IpcMessage = bincode::deserialize(&buffer)?;
        match msg {
            IpcMessage::Ready => Ok(()),
            _ => Err("Expected Ready message".into()),
        }
    }

    pub fn setup_shm(&mut self, shm_path: &str, size: usize) -> Result<(), std::io::Error> {
        #[cfg(unix)]
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(shm_path)?;

        #[cfg(windows)]
        let file = {
            use std::os::windows::fs::OpenOptionsExt;
            OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .share_mode(0x7) // FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE
                .open(shm_path)?
        };

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

    #[cfg(unix)]
    pub fn try_recv_control(&mut self) -> Option<ControlMessage> {
        self.stream.set_nonblocking(true).ok()?;

        let mut len_bytes = [0u8; 8];
        match self.stream.read_exact(&mut len_bytes) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                let _ = self.stream.set_nonblocking(false);
                return None;
            }
            Err(_) => {
                let _ = self.stream.set_nonblocking(false);
                return None;
            }
        }

        let len = u64::from_le_bytes(len_bytes) as usize;
        let mut buffer = vec![0u8; len];

        let _ = self.stream.set_nonblocking(false);
        if self.stream.read_exact(&mut buffer).is_err() {
            return None;
        }

        bincode::deserialize(&buffer).ok()
    }

    #[cfg(windows)]
    pub fn try_recv_control(&mut self) -> Option<ControlMessage> {
        // Windows named pipes don't support non-blocking reads easily.
        // For now, skip control message processing on Windows.
        // TODO: implement proper async control message handling for Windows
        None
    }
}

impl Drop for IpcClient {
    fn drop(&mut self) {
        if !self.shm_path.is_empty() {
            let _ = std::fs::remove_file(&self.shm_path);
        }
    }
}
