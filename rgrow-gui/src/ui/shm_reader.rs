use memmap2::MmapMut;
use std::fs::OpenOptions;
use std::path::Path;

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
