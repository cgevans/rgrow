use std::fmt::{Display, Formatter};

use std::any::Any;

#[cfg(feature = "python")]
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyArrayMethods};
use serde::{Deserialize, Serialize};
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
#[error("{0}")]
pub struct StringConvError(pub(crate) String);

#[cfg(feature = "python")]
impl From<StringConvError> for pyo3::PyErr {
    fn from(value: StringConvError) -> Self {
        pyo3::exceptions::PyValueError::new_err(value.0)
    }
}

#[cfg(feature = "python")]
use pyo3::{FromPyObject, IntoPy, PyAny, PyErr, PyObject, PyResult, Python};

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
    #[error("No UI is available.")]
    NoUI
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

#[derive(Clone, Serialize, Deserialize, Eq, PartialEq)]
#[serde(untagged)]
#[cfg_attr(feature = "python", derive(FromPyObject))]
pub enum Ident {
    Num(usize),
    Name(String),
}

#[cfg(feature = "python")]
impl IntoPy<PyObject> for Ident {
    fn into_py(self, py: Python) -> PyObject {
        match self {
            Ident::Num(num) => num.into_py(py),
            Ident::Name(name) => name.into_py(py),
        }
    }
}

impl From<u32> for Ident {
    fn from(value: u32) -> Self {
        Self::Num(value as usize)
    }
}

impl Display for Ident {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Name(s) => write!(f, "\"{s}\""),
            Self::Num(n) => write!(f, "{n}"),
        }
    }
}

impl core::fmt::Debug for Ident {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Name(s) => write!(f, "\"{s}\""),
            Self::Num(n) => write!(f, "{n}"),
        }
    }
}

pub type GlueIdent = Ident;
pub type TileIdent = Ident;


pub struct RustAny(pub Box<dyn Any>);

#[cfg(feature = "python")]
impl FromPyObject<'_> for RustAny {
    fn extract(obj: &PyAny) -> PyResult<Self> {
        if let Ok(val) = obj.extract::<u64>() {
            Ok(RustAny(Box::new(val)))
        } else if let Ok(val) = obj.extract::<f64>() {
            Ok(RustAny(Box::new(val)))
        } else if let Ok(val) = obj.extract::<i64>() {
            Ok(RustAny(Box::new(val)))
        } else if let Ok(val) = obj.extract::<bool>() {
            Ok(RustAny(Box::new(val)))
        } else if let Ok(val) = obj.extract::<String>() {
            Ok(RustAny(Box::new(val)))
        } else if let Ok(val) = obj.extract::<PyReadonlyArray1<f64>>() {
            Ok(RustAny(Box::new(val.to_owned_array())))
        } else if let Ok(val) = obj.extract::<PyReadonlyArray2<f64>>() {
            Ok(RustAny(Box::new(val.to_owned_array())))
        } else if let Ok(val) = obj.extract::<(u64, u64)>() {
            Ok(RustAny(Box::new(val)))
        } else if let Ok(val) = obj.extract::<(usize, usize, Ident)>() {
            Ok(RustAny(Box::new(val)))
        } else if let Ok(val) = obj.extract::<Vec<(usize, usize, Ident)>>() {
            Ok(RustAny(Box::new(val)))
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "Cannot convert value {:?}",
                obj
            )))
        }
    }
}

#[cfg(feature = "python")]
impl IntoPy<PyObject> for RustAny {
    fn into_py(self, py: Python<'_>) -> PyObject {
        if let Some(val) = self.0.downcast_ref::<f64>() {
            val.into_py(py)
        } else if let Some(val) = self.0.downcast_ref::<u64>() {
            val.into_py(py)
        } else if let Some(val) = self.0.downcast_ref::<i64>() {
            val.into_py(py)
        } else if let Some(val) = self.0.downcast_ref::<bool>() {
            val.into_py(py)
        } else if let Some(val) = self.0.downcast_ref::<String>() {
            val.into_py(py)
        } else if let Some(val) = self.0.downcast_ref::<ndarray::Array1<f64>>() {
            PyArray1::from_array_bound(py, val).into_py(py)
        } else if let Some(val) = self.0.downcast_ref::<ndarray::Array2<f64>>() {
            PyArray2::from_array_bound(py, val).into_py(py)
        } else if let Some(val) = self.0.downcast_ref::<ndarray::Array1<u64>>() {
            PyArray1::from_array_bound(py, val).into_py(py)
        } else if let Some(val) = self.0.downcast_ref::<ndarray::Array2<u64>>() {
            PyArray2::from_array_bound(py, val).into_py(py)
        } else if let Some(val) = self.0.downcast_ref::<(u64, u64)>() {
            (val.0, val.1).into_py(py)
        } else if let Some(val) = self.0.downcast_ref::<(usize, usize, Ident)>() {
            (val.0, val.1, val.2.clone()).into_py(py)
        } else {
            panic!("Cannot convert Any to PyAny");
        }
    }
}
