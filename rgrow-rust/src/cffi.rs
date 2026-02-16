#![allow(clippy::missing_safety_doc)]

use std::{
    ffi::{c_char, c_void},
    time::Duration,
};

use crate::{
    base::{NumEvents, NumTiles},
    system::{self},
    tileset::TileSet,
};

#[repr(C)]
pub enum COption<T> {
    None,
    Some(T),
}

impl<T> From<COption<T>> for Option<T> {
    fn from(value: COption<T>) -> Self {
        match value {
            COption::Some(x) => Some(x),
            COption::None => None,
        }
    }
}

impl<T> From<Option<T>> for COption<T> {
    fn from(value: Option<T>) -> Self {
        match value {
            Some(v) => COption::Some(v),
            None => COption::None,
        }
    }
}

#[repr(C)]
pub struct CArrayView2<T> {
    pub data: *const T,
    pub nrows: u64,
    pub ncols: u64,
}

#[repr(C)]
pub struct EvolveBounds {
    /// Stop if this number of events has taken place during this evolve call.
    pub for_events: COption<NumEvents>,
    /// Stop if this number of events has been reached in total for the state.
    pub total_events: COption<NumEvents>,
    /// Stop if this amount of (simulated) time has passed during this evolve call.
    pub for_time: COption<f64>,
    /// Stop if this amount of (simulated) time has passed in total for the state.
    pub total_time: COption<f64>,
    /// Stop if the number of tiles is equal to or less than this number.
    pub size_min: COption<NumTiles>,
    /// Stop if the number of tiles is equal to or greater than this number.
    pub size_max: COption<NumTiles>,
    /// Stop after this amount of (real) time has passed.
    pub for_wall_time: COption<f64>,
}

impl From<system::EvolveBounds> for EvolveBounds {
    fn from(value: system::EvolveBounds) -> Self {
        Self {
            for_events: value.for_events.into(),
            total_events: value.total_events.into(),
            for_time: value.for_time.into(),
            total_time: value.total_time.into(),
            size_min: value.size_min.into(),
            size_max: value.size_max.into(),
            for_wall_time: value.for_wall_time.map(|d| d.as_secs_f64()).into(),
        }
    }
}

impl From<EvolveBounds> for system::EvolveBounds {
    fn from(value: EvolveBounds) -> Self {
        Self {
            for_events: value.for_events.into(),
            total_events: value.total_events.into(),
            for_time: value.for_time.into(),
            total_time: value.total_time.into(),
            size_min: value.size_min.into(),
            size_max: value.size_max.into(),
            for_wall_time: Option::from(value.for_wall_time).map(Duration::from_secs_f64),
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn create_tileset_from_file(s: *const c_char) -> *mut TileSet {
    let ts = TileSet::from_file(unsafe { std::ffi::CStr::from_ptr(s) }.to_str().unwrap()).unwrap();
    Box::into_raw(Box::new(ts))
}

#[no_mangle]
pub unsafe extern "C" fn create_tileset_from_json(s: *const c_char) -> *mut TileSet {
    let ts = TileSet::from_json(unsafe { std::ffi::CStr::from_ptr(s) }.to_str().unwrap()).unwrap();
    Box::into_raw(Box::new(ts))
}

#[no_mangle]
pub unsafe extern "C" fn create_system_from_tileset(t: *const TileSet) -> *mut c_void {
    let ts = &*t;
    let sim = ts.create_dynsystem().unwrap();
    Box::into_raw(sim.0) as *mut c_void
}

#[no_mangle]
pub unsafe extern "C" fn create_state_from_tileset(ts: *const TileSet) -> *mut c_void {
    let ts = &*ts;
    let state = ts.create_state_empty().unwrap();
    Box::into_raw(state) as *mut c_void
}

#[no_mangle]
pub unsafe extern "C" fn drop_tileset(ts: *mut TileSet) {
    let ts = Box::from_raw(ts);
    drop(ts);
}
