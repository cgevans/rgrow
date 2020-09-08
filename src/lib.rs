//! The rgrow (FIXME: we need a better name!) tileset simulator.

extern crate ndarray;


#[macro_use]
extern crate lazy_static;

pub mod parser;
mod system;
mod canvas;
mod base;
mod state;
pub mod ffstest;
pub mod ffs;
mod fission;

pub use system::*;
pub use canvas::*;
pub use base::*;
pub use state::*;
