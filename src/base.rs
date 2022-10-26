use thiserror;

use crate::tileset::ParserError;

pub type Point = (usize, usize);
pub type NumTiles = u32;
pub type NumEvents = u64;
pub type Tile = usize;
pub type Energy = f64;
pub type Glue = usize;
pub type CanvasLength = usize;

#[derive(thiserror::Error, Debug)]
pub enum GrowError {
    #[error("can't create canvas from array of size ({0}, {1})")]
    WrongCanvasSize(usize, usize),
    #[error("FFS is meaningless for the aTAM.")]
    FFSCannotRunATAM,
}

#[derive(thiserror::Error, Debug)]
pub enum RgrowError {
    #[error(transparent)]
    Parser(#[from] ParserError),
    #[error(transparent)]
    Grow(#[from] GrowError),
    #[cfg(feature = "ui")]
    #[error(transparent)]
    Pixel(#[from] pixels::Error),
}

pub type GrowResult<T> = Result<T, GrowError>;

pub type Rate = f64;
