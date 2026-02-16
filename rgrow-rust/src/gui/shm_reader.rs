use memmap2::MmapMut;
use std::fs::OpenOptions;
use std::path::Path;

pub struct ShmReader {
    mmap: MmapMut,
}

impl ShmReader {
    pub fn open<P: AsRef<Path>>(path: P, size: usize) -> Result<Self, std::io::Error> {
        #[cfg(unix)]
        let file = OpenOptions::new().read(true).write(true).open(path)?;

        #[cfg(windows)]
        let file = {
            use std::os::windows::fs::OpenOptionsExt;
            OpenOptions::new()
                .read(true)
                .write(true)
                .share_mode(0x7) // FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE
                .open(path)?
        };

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
