//#![feature(associated_type_bounds)]
//#![feature(map_first_last)]

//! The rgrow (FIXME: we need a better name!) tileset simulator.  This generally uses the algorithms of the venerable Xgrow.

extern crate ndarray;
extern crate phf;

pub mod tileset;

pub mod parser_xgrow;

pub mod colors;

#[cfg(feature = "ui")]
pub mod ui;

pub mod base;
pub mod canvas;
pub mod ffs;
pub mod state;
pub mod system;

pub mod ratestore;

pub mod simulation;

pub mod models;

// pub mod arcsim;

pub mod cffi;
