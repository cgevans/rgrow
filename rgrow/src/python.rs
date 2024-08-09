use std::fs::File;
use std::ops::DerefMut;
use std::time::Duration;

use crate::base::{NumEvents, NumTiles, RgrowError, RustAny, Tile};
use crate::canvas::{Canvas, PointSafeHere};
use crate::ffs::{FFSRunConfig, FFSRunResult, FFSStateRef};
use crate::models::atam::ATAM;
use crate::models::ktam::KTAM;
use crate::models::oldktam::OldKTAM;
use crate::models::sdc1d::{SDCParams, SDC};
use crate::ratestore::RateStore;
use crate::state::{StateEnum, StateStatus, TrackerData};
use crate::system::{
    DimerInfo, DynSystem, EvolveBounds, EvolveOutcome, NeededUpdate, SystemWithDimers, TileBondInfo,
};
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};

/// A State object.
#[cfg_attr(feature = "python", pyclass(name = "State", module = "rgrow"))]
#[repr(transparent)]
pub struct PyState(pub(crate) StateEnum);

/// A single 'assembly', or 'state', containing a canvas with tiles at locations.
/// Generally does not store concentration or temperature information, but does store time simulated.
#[cfg(feature = "python")]
#[pymethods]
impl PyState {
    #[new]
    #[pyo3(signature = (shape, kind="Square", tracking="None", n_tile_types=None))]
    pub fn empty(
        shape: (usize, usize),
        kind: &str,
        tracking: &str,
        n_tile_types: Option<usize>,
    ) -> PyResult<Self> {
        Ok(PyState(StateEnum::empty(
            shape,
            kind.try_into()?,
            tracking.try_into()?,
            n_tile_types.unwrap_or(1),
        )?))
    }

    /// Return a cloned copy of an array with the total possible next event rate for each point in the canvas.
    /// This is the deepest level of the quadtree for tree-based states.
    ///
    /// Returns
    /// -------
    /// NDArray[np.uint]
    pub fn rate_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<crate::base::Rate>> {
        self.0.rate_array().to_pyarray_bound(py)
    }

    #[getter]
    /// float: the total rate of possible next events for the state.
    pub fn total_rate(&self) -> crate::base::Rate {
        RateStore::total_rate(&self.0)
    }

    #[getter]
    /// NDArray[np.uint]: a direct, mutable view of the state's canvas.  This is potentially unsafe.
    pub fn canvas_view<'py>(
        this: Bound<'py, Self>,
        _py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray2<crate::base::Tile>>> {
        let t = this.borrow();
        let ra = t.0.raw_array();

        unsafe { Ok(PyArray2::borrow_from_array_bound(&ra, this.into_any())) }
    }

    /// Return a copy of the state's canvas.  This is safe, but can't be modified and is slower than `canvas_view`.
    ///
    /// Returns
    /// -------
    /// NDArray[np.uint]
    ///     A cloned copy of the state's canvas, in raw form.
    pub fn canvas_copy<'py>(
        this: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray2<crate::base::Tile>>> {
        let t = this.borrow();
        let ra = t.0.raw_array();

        Ok(PyArray2::from_array_bound(py, &ra))
    }

    /// Return the total possible next event rate at a specific canvas point.
    ///
    /// Parameters
    /// ----------
    /// point: tuple[int, int]
    ///     The canvas point.
    ///
    /// Returns
    /// -------
    /// f64
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     if `point` is out of bounds for the canvas.
    pub fn rate_at_point(&self, point: (usize, usize)) -> PyResult<f64> {
        if self.0.inbounds(point) {
            Ok(self.0.rate_at_point(PointSafeHere(point)))
        } else {
            Err(PyValueError::new_err(format!(
                "Point {:?} is out of bounds.",
                point
            )))
        }
    }

    /// Return a copy of the tracker's tracking data.
    ///
    /// Returns
    /// -------
    /// Any
    pub fn tracking_copy(this: &Bound<Self>) -> PyResult<RustAny> {
        let t = this.borrow();
        let ra = t.0.get_tracker_data();

        Ok(ra)
    }

    /// int: the number of tiles in the state.
    #[getter]
    pub fn n_tiles(&self) -> NumTiles {
        self.0.n_tiles()
    }

    /// int: the number of tiles in the state (deprecated, use `n_tiles` instead).
    #[getter]
    pub fn ntiles(&self) -> NumTiles {
        self.0.n_tiles()
    }

    /// int: the total number of events that have occurred in the state.
    #[getter]
    pub fn total_events(&self) -> NumEvents {
        self.0.total_events()
    }

    /// float: the total time the state has simulated, in seconds.
    #[getter]
    pub fn time(&self) -> f64 {
        self.0.time()
    }

    pub fn __repr__(&self) -> String {
        format!(
            "State(n_tiles={}, time={} s, events={}, size=({}, {}), total_rate={})",
            self.n_tiles(),
            self.0.time(),
            self.total_events(),
            self.0.ncols(),
            self.0.nrows(),
            self.0.total_rate()
        )
    }

    pub fn print_debug(&self) {
        println!("{:?}", self.0);
    }

    /// Write the state to a JSON file.  This is inefficient, and is likely
    /// useful primarily for debugging.
    pub fn write_json(&self, filename: &str) -> Result<(), RgrowError> {
        serde_json::to_writer(File::create(filename)?, &self.0).unwrap();
        Ok(())
    }

    #[staticmethod]
    pub fn read_json(filename: &str) -> Result<Self, RgrowError> {
        Ok(PyState(
            serde_json::from_reader(File::open(filename)?).unwrap(),
        ))
    }
}

#[cfg(feature = "python")]
#[derive(FromPyObject)]
pub enum PyStateOrStates<'py> {
    #[pyo3(transparent)]
    State(Bound<'py, PyState>),
    #[pyo3(transparent)]
    States(Vec<Bound<'py, PyState>>),
}

#[cfg(feature = "python")]
#[derive(FromPyObject)]

pub enum PyStateOrRef<'py> {
    State(Bound<'py, PyState>),
    Ref(Bound<'py, FFSStateRef>),
}

impl From<FFSStateRef> for PyState {
    fn from(state: FFSStateRef) -> Self {
        state.clone_state()
    }
}

macro_rules! create_py_system {
    ($name: ident) => {
        #[cfg(feature = "python")]
        #[pymethods]
        impl $name {


            #[allow(clippy::too_many_arguments)]
            #[pyo3(
                                                        name = "evolve",
                                                        signature = (state,
                                                                    for_events=None,
                                                                    total_events=None,
                                                                    for_time=None,
                                                                    total_time=None,
                                                                    size_min=None,
                                                                    size_max=None,
                                                                    for_wall_time=None,
                                                                    require_strong_bound=true,
                                                                    show_window=false,
                                                                    parallel=true)
                                                    )]
            /// Evolve a state (or states), with some bounds on the simulation.
            ///
            /// If evolving multiple states, the bounds are applied per-state.
            ///
            /// Parameters
            /// ----------
            /// state : State or Sequence[State]
            ///   The state or states to evolve.
            /// for_events : int, optional
            ///   Stop evolving each state after this many events.
            /// total_events : int, optional
            ///   Stop evelving each state when the state's total number of events (including
            ///   previous events) reaches this.
            /// for_time : float, optional
            ///   Stop evolving each state after this many seconds of simulated time.
            /// total_time : float, optional
            ///   Stop evolving each state when the state's total time (including previous steps)
            ///   reaches this.
            /// size_min : int, optional
            ///   Stop evolving each state when the state's number of tiles is less than or equal to this.
            /// size_max : int, optional
            ///   Stop evolving each state when the state's number of tiles is greater than or equal to this.
            /// for_wall_time : float, optional
            ///   Stop evolving each state after this many seconds of wall time.
            /// require_strong_bound : bool
            ///   Require that the stopping conditions are strong, i.e., they are guaranteed to be eventually
            ///   satisfied under normal conditions.
            /// show_window : bool
            ///   Show a graphical UI window while evolving (requires ui feature, and a single state).
            /// parallel : bool
            ///   Use multiple threads.
            ///
            /// Returns
            /// -------
            /// EvolveOutcome or List[EvolveOutcome]
            ///  The outcome (stopping condition) of the evolution.  If evolving a single state, returns a single outcome.
            pub fn py_evolve<'py>(
                &mut self,
                state: PyStateOrStates<'py>,
                for_events: Option<u64>,
                total_events: Option<u64>,
                for_time: Option<f64>,
                total_time: Option<f64>,
                size_min: Option<u32>,
                size_max: Option<u32>,
                for_wall_time: Option<f64>,
                require_strong_bound: bool,
                show_window: bool,
                parallel: bool,
                py: Python<'py>,
            ) -> PyResult<PyObject> {
                let bounds = EvolveBounds {
                    for_events,
                    for_time,
                    total_events,
                    total_time,
                    size_min,
                    size_max,
                    for_wall_time: for_wall_time.map(Duration::from_secs_f64),
                };

                if require_strong_bound & !bounds.is_strongly_bounded() {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "No strong bounds specified.",
                    ));
                }

                if !bounds.is_weakly_bounded() {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "No weak bounds specified.",
                    ));
                }

                match state {
                    PyStateOrStates::State(pystate) => {
                        let state = &mut pystate.borrow_mut().0;
                        if show_window {
                            Ok(py
                                .allow_threads(|| {
                                    DynSystem::evolve_in_window(self, state, None, bounds)
                                })?
                                .into_py(py))
                        } else {
                            Ok(py
                                .allow_threads(|| DynSystem::evolve(self, state, bounds))?
                                .into_py(py))
                        }
                    }
                    PyStateOrStates::States(pystates) => {
                        if show_window {
                            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                                "Cannot show window with multiple states.",
                            ));
                        }
                        let mut refs = pystates
                            .into_iter()
                            .map(|x| x.borrow_mut())
                            .collect::<Vec<_>>();
                        let mut states = refs.iter_mut().map(|x| x.deref_mut()).collect::<Vec<_>>();
                        let out = if parallel {
                            py.allow_threads(|| {
                                states
                                    .par_iter_mut()
                                    .map(|state| DynSystem::evolve(self, &mut state.0, bounds))
                                    .collect::<Vec<_>>()
                            })
                        } else {
                            states
                                .iter_mut()
                                .map(|state| DynSystem::evolve(self, &mut state.0, bounds))
                                .collect::<Vec<_>>()
                        };
                        let o: Result<Vec<EvolveOutcome>, PyErr> = out
                            .into_iter()
                            .map(|x| {
                                x.map_err(|y| {
                                    pyo3::exceptions::PyValueError::new_err(y.to_string())
                                })
                            })
                            .collect();
                        o.map(|x| x.into_py(py))
                    }
                }
            }

            /// Calculate the number of mismatches in a state.
            ///
            /// Parameters
            /// ----------
            /// state : State or FFSStateRef
            ///   The state to calculate mismatches for.
            ///
            /// Returns
            /// -------
            /// int
            ///  The number of mismatches.
            ///
            /// See also
            /// --------
            /// calc_mismatch_locations
            ///   Calculate the location and direction of mismatches, not jus the number.
            fn calc_mismatches(&self, state: PyStateOrRef) -> usize {
                match state {
                    PyStateOrRef::State(s) => DynSystem::calc_mismatches(self, &s.borrow().0),
                    PyStateOrRef::Ref(s) => {
                        DynSystem::calc_mismatches(self, &s.borrow().clone_state().0)
                    }
                }
            }

            /// Calculate information about the dimers the system is able to form.
            ///
            /// Returns
            /// -------
            /// List[DimerInfo]
            fn calc_dimers(&self) -> Vec<DimerInfo> {
                SystemWithDimers::calc_dimers(self)
            }

            /// Calculate the locations of mismatches in the state.
            ///
            /// This returns a copy of the canvas, with the values set to 0 if there is no mismatch
            /// in the location, and > 0, in a model defined way, if there is at least one mismatch.
            /// Most models use v = 8*N + 4*E + 2*S + W, where N, E, S, W are the four directions.
            /// Thus, a tile with mismatches to the E and W would have v = 4+2 = 6.
            ///
            /// Parameters
            /// ----------
            /// state : State or FFSStateRef
            ///    The state to calculate mismatches for.
            ///
            /// Returns
            /// -------
            /// ndarray
            ///   An array of the same shape as the state's canvas, with the values set as described above.
            fn calc_mismatch_locations<'py>(
                &mut self,
                state: PyStateOrRef,
                py: Python<'py>,
            ) -> PyResult<Bound<'py, PyArray2<usize>>> {
                let ra = match state {
                    PyStateOrRef::State(s) => {
                        DynSystem::calc_mismatch_locations(self, &s.borrow().0)
                    }
                    PyStateOrRef::Ref(s) => {
                        DynSystem::calc_mismatch_locations(self, &s.borrow().clone_state().0)
                    }
                };
                Ok(PyArray2::from_array_bound(py, &ra))
            }

            /// Set a system parameter.
            ///
            /// Parameters
            /// ----------
            /// param_name : str
            ///     The name of the parameter to set.
            /// value : Any
            ///     The value to set the parameter to.
            ///
            /// Returns
            /// -------
            /// NeededUpdate
            ///     The type of state update needed.  This can be passed to
            ///    `update_state` to update the state.
            fn set_param(&mut self, param_name: &str, value: RustAny) -> PyResult<NeededUpdate> {
                Ok(DynSystem::set_param(self, param_name, value.0)?)
            }

            /// Names of tiles, by tile number.
            #[getter]
            fn tile_names(&self) -> Vec<String> {
                TileBondInfo::tile_names(self)
                    .iter()
                    .map(|x| x.to_string())
                    .collect()
            }

            /// Given a tile name, return the tile number.
            ///
            /// Parameters
            /// ----------
            /// tile_name : str
            ///   The name of the tile.
            ///
            /// Returns
            /// -------
            /// int
            ///  The tile number.
            fn tile_number_from_name(&self, tile_name: &str) -> Option<Tile> {
                TileBondInfo::tile_names(self)
                    .iter()
                    .position(|x| *x == tile_name)
                    .map(|x| x as Tile)
            }

            /// Given a tile number, return the color of the tile.
            ///
            /// Parameters
            /// ----------
            /// tile_number : int
            ///  The tile number.
            ///
            /// Returns
            /// -------
            /// list[int]
            ///   The color of the tile, as a list of 4 integers (RGBA).
            fn tile_color(&self, tile_number: Tile) -> [u8; 4] {
                TileBondInfo::tile_color(self, tile_number)
            }

            #[getter]
            fn tile_colors<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u8>> {
                let colors = TileBondInfo::tile_colors(self);
                let mut arr = Array2::zeros((colors.len(), 4));
                for (i, c) in colors.iter().enumerate() {
                    arr[[i, 0]] = c[0];
                    arr[[i, 1]] = c[1];
                    arr[[i, 2]] = c[2];
                    arr[[i, 3]] = c[3];
                }
                arr.into_pyarray_bound(py)
            }

            fn get_param(&mut self, param_name: &str) -> PyResult<RustAny> {
                Ok(RustAny(DynSystem::get_param(self, param_name)?))
            }

            #[pyo3(signature = (state, needed = &NeededUpdate::All))]
            fn update_all(&self, state: &mut PyState, needed: &NeededUpdate) {
                DynSystem::update_state(self, &mut state.0, needed)
            }

            /// Recalculate a state's rates.
            ///
            /// This is usually needed when a parameter of the system has
            /// been changed.
            ///
            /// Parameters
            /// ----------
            /// state : State
            ///   The state to update.
            /// needed : NeededUpdate, optional
            ///   The type of update needed.  If not provided, all locations
            ///   will be recalculated.
            #[pyo3(signature = (state, needed = &NeededUpdate::All))]
            fn update_state(&self, state: &mut PyState, needed: &NeededUpdate) {
                DynSystem::update_state(self, &mut state.0, needed)
            }

            /// Run FFS.
            ///
            /// Parameters
            /// ----------
            /// config : FFSRunConfig
            ///  The configuration for the FFS run.
            /// **kwargs
            ///   FFSRunConfig parameters as keyword arguments.
            ///
            /// Returns
            /// -------
            /// FFSRunResult
            ///  The result of the FFS run.
            #[pyo3(name = "run_ffs", signature = (config = FFSRunConfig::default(), **kwargs))]
            fn py_run_ffs(
                &mut self,
                config: FFSRunConfig,
                kwargs: Option<Bound<PyDict>>,
                py: Python<'_>,
            ) -> PyResult<FFSRunResult> {
                let mut c = config;

                if let Some(dict) = kwargs {
                    for (k, v) in dict.iter() {
                        c._py_set(&k.extract::<String>()?, v)?;
                    }
                }

                let res = py.allow_threads(|| DynSystem::run_ffs(self, &c));
                match res {
                    Ok(res) => Ok(res),
                    Err(err) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        err.to_string(),
                    )),
                }
            }

            fn __repr__(&self) -> String {
                format!("System({})", DynSystem::system_info(self))
            }

            pub fn print_debug(&self) {
                println!("{:?}", self);
            }

            /// Write the system to a JSON file.
            ///
            /// Parameters
            /// ----------
            /// filename : str
            ///     The name of the file to write to.
            pub fn write_json(&self, filename: &str) -> Result<(), RgrowError> {
                serde_json::to_writer(File::create(filename)?, self).unwrap();
                Ok(())
            }

            /// Read a system from a JSON file.
            ///
            /// Parameters
            /// ----------
            /// filename : str
            ///     The name of the file to read from.
            ///
            /// Returns
            /// -------
            /// Self
            #[staticmethod]
            pub fn read_json(filename: &str) -> Result<Self, RgrowError> {
                Ok(serde_json::from_reader(File::open(filename)?).unwrap())
            }
        }
    };
}

create_py_system!(KTAM);
create_py_system!(ATAM);
create_py_system!(OldKTAM);
create_py_system!(SDC);
