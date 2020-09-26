use thiserror;

pub type Point = (usize, usize);
pub type NumTiles = u32;
pub type NumEvents = u64;
pub type Tile = u32;
pub type Rate = f64;
pub type Energy = f64;
pub type Glue = u32;
pub type CanvasLength = usize;

#[derive(thiserror::Error, Debug)]
pub enum GrowError {
    #[error("can't create canvas from array of size ({0}, {1})")]
    WrongCanvasSize(usize, usize)
}

pub type GrowResult<T> = Result<T, GrowError>;