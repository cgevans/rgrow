//#![feature(associated_type_bounds)]
//#![feature(map_first_last)]

//! The rgrow (FIXME: we need a better name!) tileset simulator.  This generally uses the algorithms of the venerable Xgrow.

extern crate ndarray;
extern crate phf;

pub mod utils;

pub mod units;

pub mod tileset;

pub mod parser_xgrow;

pub mod colors;

pub mod base;
pub mod canvas;
pub mod ffs;
pub mod state;
pub mod system;

pub mod ratestore;

pub mod models;

pub mod ui;

#[cfg(feature = "python")]
pub mod python;

#[cfg(feature = "python")]
pub mod pytileset;

// pub mod cffi;

// pub mod newsystem;
