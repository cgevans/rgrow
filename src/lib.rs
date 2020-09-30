//! The rgrow (FIXME: we need a better name!) tileset simulator.  This generally uses the algorithms of the venerable Xgrow.


extern crate ndarray;
extern crate phf;

pub mod parser;

//#[cfg(feature = "parser_xgrow")]
pub mod parser_xgrow;

pub mod colors;

#[cfg(feature = "ui")]
pub mod ui;

pub mod system;
pub mod canvas;
pub mod base;
pub mod state;
pub mod ffs;
pub mod fission;

pub mod ratestore;