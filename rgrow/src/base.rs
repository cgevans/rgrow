use thiserror;

use crate::tileset::ParserError;
use thiserror::Error;

pub type Point = (usize, usize);
pub type NumTiles = u32;
pub type NumEvents = u64;
pub type Tile = u32;
pub type Energy = f64;
pub type Glue = usize;
pub type CanvasLength = usize;

#[derive(Error, Debug)]
pub enum GrowError {
    #[error("can't create canvas from array of size ({0}, {1})")]
    WrongCanvasSize(usize, usize),
    #[error("FFS is meaningless for the aTAM.")]
    FFSCannotRunATAM,
    #[error("Point ({0}, {1}) is out of bounds.")]
    OutOfBounds(usize, usize),
    #[error("{0}")]
    NotImplemented(String),
    #[error(transparent)]
    PoisonError(#[from] std::sync::PoisonError<()>),
    #[error("There is no state with key {0}")]
    NoState(usize),
    #[error("There is no modifiable parameter with name {0}")]
    NoParameter(String),
    #[error("Parameter type is wrong for {0}")]
    WrongParameterType(String),
}

#[cfg(feature = "python")]
impl From<GrowError> for pyo3::PyErr {
    fn from(err: GrowError) -> Self {
        pyo3::exceptions::PyException::new_err(err.to_string())
    }
}

#[derive(Error, Debug)]
pub enum RgrowError {
    #[error(transparent)]
    Parser(#[from] ParserError),
    #[error(transparent)]
    Grow(#[from] GrowError),
    #[error(transparent)]
    ModelError(#[from] ModelError),
    #[cfg(feature = "ui")]
    #[error(transparent)]
    Pixel(#[from] pixels::Error),
    #[error(transparent)]
    IO(#[from] std::io::Error),
}

#[cfg(feature = "python")]
impl From<RgrowError> for pyo3::PyErr {
    fn from(err: RgrowError) -> Self {
        pyo3::exceptions::PyException::new_err(err.to_string())
    }
}

#[derive(Error, Debug)]
pub enum ModelError {
    #[error("Model does not support duples.")]
    DuplesNotSupported,
}

pub type GrowResult<T> = Result<T, GrowError>;

pub type Rate = f64;

use fnv::{FnvHashMap, FnvHashSet};
pub(crate) type HashSetType<T> = FnvHashSet<T>;
pub(crate) type HashMapType<K, V> = FnvHashMap<K, V>;
