use std::collections::HashMap;
use std::fs::File;
use std::ops::DerefMut;
use std::time::Duration;

use crate::base::{NumEvents, NumTiles, RgrowError, RustAny, Tile};
use crate::canvas::{Canvas, PointSafe2, PointSafeHere};
use crate::ffs::{FFSRunConfig, FFSRunResult, FFSStateRef};
use crate::models::atam::ATAM;
use crate::models::kblock::KBlock;
use crate::models::ktam::KTAM;
use crate::models::oldktam::OldKTAM;
use crate::models::sdc1d::SDC;
use crate::ratestore::RateStore;
use crate::state::{StateEnum, StateStatus, TileCounts, TrackerData};
use crate::system::{
    DimerInfo, DynSystem, EvolveBounds, EvolveOutcome, NeededUpdate, System, TileBondInfo,
};
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::IntoPyObjectExt;
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

    #[staticmethod]
    #[pyo3(signature = (array, kind="Square", tracking="None", n_tile_types=None))]
    pub fn from_array(
        array: PyReadonlyArray2<crate::base::Tile>,
        kind: &str,
        tracking: &str,
        n_tile_types: Option<usize>,
    ) -> PyResult<Self> {
        Ok(PyState(StateEnum::from_array(
            array.as_array(),
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
    pub fn rate_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.0.rate_array().mapv(|x| x.into()).to_pyarray(py)
    }

    #[getter]
    /// float: the total rate of possible next events for the state.
    pub fn total_rate(&self) -> f64 {
        RateStore::total_rate(&self.0).into()
    }

    #[getter]
    /// NDArray[np.uint]: a direct, mutable view of the state's canvas.  This is potentially unsafe.
    pub fn canvas_view<'py>(
        this: Bound<'py, Self>,
        _py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray2<crate::base::Tile>>> {
        let t = this.borrow();
        let ra = t.0.raw_array();

        unsafe { Ok(PyArray2::borrow_from_array(&ra, this.into_any())) }
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

        Ok(PyArray2::from_array(py, &ra))
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
            Ok(self.0.rate_at_point(PointSafeHere(point)).into())
        } else {
            Err(PyValueError::new_err(format!(
                "Point {point:?} is out of bounds."
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
        self.0.time().into()
    }

    #[getter]
    pub fn tile_counts<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        self.0.tile_counts().to_pyarray(py)
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

    /// Create a copy of the state.
    ///
    /// This creates a complete clone of the state, including all canvas data,
    /// tracking information, and simulation state (time, events, etc.).
    ///
    /// Returns
    /// -------
    /// State
    ///     A new State object that is a copy of this state.
    ///
    /// Examples
    /// --------
    /// >>> original_state = State((10, 10))
    /// >>> copied_state = original_state.copy()
    /// >>> # The copied state is independent of the original
    /// >>> assert copied_state.time == original_state.time
    /// >>> assert copied_state.total_events == original_state.total_events
    pub fn copy(&self) -> Self {
        PyState(self.0.clone())
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
            ) -> PyResult<Py<PyAny>> {
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
                            py
                                .detach(|| {
                                    System::evolve_in_window(self, state, None, bounds)
                                })?
                                .into_py_any(py)
                        } else {
                            py
                                .detach(|| System::evolve(self, state, bounds))?
                                .into_py_any(py)
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
                        let out = py.detach(|| {
                            if parallel {
                                states
                                    .par_iter_mut()
                                    .map(|state| System::evolve(self, &mut state.0, bounds))
                                    .collect::<Vec<_>>()
                        } else {
                            states
                                .iter_mut()
                                .map(|state| System::evolve(self, &mut state.0, bounds))
                                .collect::<Vec<_>>()
                        }});
                        let o: Result<Vec<EvolveOutcome>, PyErr> = out
                            .into_iter()
                            .map(|x| {
                                x.map_err(|y| {
                                    pyo3::exceptions::PyValueError::new_err(y.to_string())
                                })
                            })
                            .collect();
                        o.map(|x| x.into_py_any(py).unwrap())
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
                    PyStateOrRef::State(s) => System::calc_mismatches(self, &s.borrow().0),
                    PyStateOrRef::Ref(s) => {
                        System::calc_mismatches(self, &s.borrow().clone_state().0)
                    }
                }
            }

            /// Calculate information about the dimers the system is able to form.
            ///
            /// Returns
            /// -------
            /// List[DimerInfo]
            ///
            /// Raises
            /// ------
            /// ValueError
            ///     If the system doesn't support dimer calculation
            fn calc_dimers(&self) -> PyResult<Vec<DimerInfo>> {
                System::calc_dimers(self).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
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
                        System::calc_mismatch_locations(self, &s.borrow().0)
                    }
                    PyStateOrRef::Ref(s) => {
                        System::calc_mismatch_locations(self, &s.borrow().clone_state().0)
                    }
                };
                Ok(PyArray2::from_array(py, &ra))
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
                Ok(System::set_param(self, param_name, value.0)?)
            }

            /// Names of tiles, by tile number.
            #[getter]
            fn tile_names(&self) -> Vec<String> {
                TileBondInfo::tile_names(self)
                    .iter()
                    .map(|x| x.to_string())
                    .collect()
            }

            #[getter]
            fn bond_names(&self) -> Vec<String> {
                TileBondInfo::bond_names(self)
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
                arr.into_pyarray(py)
            }

            fn get_param(&mut self, param_name: &str) -> PyResult<RustAny> {
                Ok(RustAny(System::get_param(self, param_name)?))
            }

            #[pyo3(signature = (state, needed = &NeededUpdate::All))]
            fn update_all(&self, state: &mut PyState, needed: &NeededUpdate) {
                System::update_state(self, &mut state.0, needed)
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
                System::update_state(self, &mut state.0, needed)
            }

            #[pyo3(name = "setup_state")]
            fn py_setup_state(&self, state: &mut PyState) -> PyResult<()> {
                self.setup_state(&mut state.0).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
                Ok(())
            }

            /// Calculate the committer function for a state: the probability that when a simulation
            /// is started from that state, the assembly will grow to a larger size (cutoff_size)
            /// rather than melting to zero tiles.
            ///
            /// Parameters
            /// ----------
            /// state : State
            ///     The state to analyze
            /// cutoff_size : int
            ///     Size threshold for commitment
            /// num_trials : int
            ///     Number of trials to run
            /// max_time : float, optional
            ///     Maximum simulation time per trial
            /// max_events : int, optional
            ///     Maximum events per trial
            ///
            /// Returns
            /// -------
            /// float
            ///     Probability of reaching cutoff_size (between 0.0 and 1.0)
            #[pyo3(name = "calc_committer", signature = (state, cutoff_size, num_trials, max_time=None, max_events=None))]
            fn py_calc_committer(
                &mut self,
                state: &PyState,
                cutoff_size: NumTiles,
                num_trials: usize,
                max_time: Option<f64>,
                max_events: Option<NumEvents>,
                py: Python<'_>,
            ) -> PyResult<f64> {

                let state = &state.0;
                
                let out = py.detach(|| {
                    self.calc_committer(
                        &state,
                        cutoff_size,
                        max_time,
                        max_events,
                        num_trials,
                    )});
                out.map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
            }

            /// Calculate the committer function for a state using adaptive sampling: the probability
            /// that when a simulation is started from that state, the assembly will grow to a larger
            /// size (cutoff_size) rather than melting to zero tiles. Automatically determines the
            /// number of trials needed to achieve a specified confidence interval margin.
            ///
            /// Parameters
            /// ----------
            /// state : State
            ///     The state to analyze
            /// cutoff_size : int
            ///     Size threshold for commitment
            /// conf_interval_margin : float
            ///     Confidence interval margin (e.g., 0.05 for 5%)
            /// max_time : float, optional
            ///     Maximum simulation time per trial
            /// max_events : int, optional
            ///     Maximum events per trial
            ///
            /// Returns
            /// -------
            /// tuple[float, int]
            ///     Tuple of (probability of reaching cutoff_size, number of trials run)
            #[pyo3(name = "calc_committer_adaptive", signature = (state, cutoff_size, conf_interval_margin, max_time=None, max_events=None))]
            fn py_calc_committer_adaptive(
                &self,
                state: &PyState,
                cutoff_size: NumTiles,
                conf_interval_margin: f64,
                max_time: Option<f64>,
                max_events: Option<NumEvents>,
                py: Python<'_>,
            ) -> PyResult<(f64, usize)> {
                py.detach(|| {
                    self.calc_committer_adaptive(
                        &state.0,
                        cutoff_size,
                        max_time,
                        max_events,
                        conf_interval_margin,
                    )
                })
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
            }

            /// Calculate the committer function for multiple states using adaptive sampling.
            ///
            /// Parameters
            /// ----------
            /// states : List[State]
            ///     The states to analyze
            /// cutoff_size : int
            ///     Size threshold for commitment
            /// conf_interval_margin : float
            ///     Confidence interval margin (e.g., 0.05 for 5%)
            /// max_time : float, optional
            ///     Maximum simulation time per trial
            /// max_events : int, optional
            ///     Maximum events per trial
            ///
            /// Returns
            /// -------
            /// tuple[NDArray[float64], NDArray[usize]]
            ///     Tuple of (committer probabilities, number of trials for each state)
            #[pyo3(name = "calc_committers_adaptive", signature = (states, cutoff_size, conf_interval_margin, max_time=None, max_events=None))]
            fn py_calc_committers_adaptive<'py>(
                &self,
                states: Vec<Bound<'py, PyState>>,
                cutoff_size: NumTiles,
                conf_interval_margin: f64,
                max_time: Option<f64>,
                max_events: Option<NumEvents>,
                py: Python<'py>,
            ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<usize>>)> {

                let refs = states.iter().map(|x| x.borrow()).collect::<Vec<_>>();
                let states = refs.iter().map(|x| &x.0).collect::<Vec<_>>();
                let (committers, trials) = py.detach(|| {
                    self.calc_committers_adaptive(&states, cutoff_size, max_time, max_events, conf_interval_margin)
                }).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

                Ok((committers.into_pyarray(py), trials.into_pyarray(py)))
            }

            /// Calculate forward probability for a given state.
            ///
            /// This function calculates the probability that a state will grow by at least
            /// `forward_step` tiles before shrinking to size 0. Unlike calc_committer which
            /// uses a fixed cutoff size, this uses a dynamic cutoff based on the current
            /// state size plus the forward_step parameter.
            ///
            /// Parameters
            /// ----------
            /// state : State
            ///     The initial state to analyze
            /// forward_step : int, optional
            ///     Number of tiles to grow beyond current size (default: 1)
            /// num_trials : int
            ///     Number of simulation trials to run
            /// max_time : float, optional
            ///     Maximum simulation time per trial
            /// max_events : int, optional
            ///     Maximum number of events per trial
            ///
            /// Returns
            /// -------
            /// float
            ///     Probability of reaching forward_step additional tiles (between 0.0 and 1.0)
            #[pyo3(name = "calc_forward_probability", signature = (state, num_trials, forward_step=1, max_time=None, max_events=None))]
            fn py_calc_forward_probability(
                &mut self,
                state: &PyState,
                num_trials: usize,
                forward_step: NumTiles,
                max_time: Option<f64>,
                max_events: Option<NumEvents>,
            ) -> PyResult<f64> {
                let result = self.calc_forward_probability(&state.0, forward_step, max_time, max_events, num_trials);
                match result {
                    Ok(probability) => Ok(probability),
                    Err(e) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string())),
                }
            }

            /// Calculate forward probability adaptively for a given state.
            ///
            /// Uses adaptive sampling to determine the number of trials needed based on a
            /// confidence interval margin. Runs until the confidence interval is narrow enough.
            ///
            /// Parameters
            /// ----------
            /// state : State
            ///     The initial state to analyze
            /// forward_step : int, optional
            ///     Number of tiles to grow beyond current size (default: 1)
            /// conf_interval_margin : float
            ///     Desired confidence interval margin (e.g., 0.05 for 5%)
            /// max_time : float, optional
            ///     Maximum simulation time per trial
            /// max_events : int, optional
            ///     Maximum number of events per trial
            ///
            /// Returns
            /// -------
            /// tuple[float, int]
            ///     Tuple of (forward probability, number of trials run)
            #[pyo3(name = "calc_forward_probability_adaptive", signature = (state, conf_interval_margin, forward_step=1, max_time=None, max_events=None))]
            fn py_calc_forward_probability_adaptive(
                &self,
                state: &PyState,
                conf_interval_margin: f64,
                forward_step: NumTiles,
                max_time: Option<f64>,
                max_events: Option<NumEvents>,
                py: Python<'_>,
            ) -> PyResult<(f64, usize)> {
                let (probability, trials) = py.detach(|| {
                    self.calc_forward_probability_adaptive(&state.0, forward_step, max_time, max_events, conf_interval_margin)
                }).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

                Ok((probability, trials))
            }

            /// Calculate forward probabilities adaptively for multiple states.
            ///
            /// Uses adaptive sampling for each state in parallel to determine forward
            /// probabilities with specified confidence intervals.
            ///
            /// Parameters
            /// ----------
            /// states : list[State]
            ///     List of initial states to analyze
            /// forward_step : int, optional
            ///     Number of tiles to grow beyond current size for each state (default: 1)
            /// conf_interval_margin : float
            ///     Desired confidence interval margin (e.g., 0.05 for 5%)
            /// max_time : float, optional
            ///     Maximum simulation time per trial
            /// max_events : int, optional
            ///     Maximum number of events per trial
            ///
            /// Returns
            /// -------
            /// tuple[NDArray[float64], NDArray[usize]]
            ///     Tuple of (forward probabilities, number of trials for each state)
            #[pyo3(name = "calc_forward_probabilities_adaptive", signature = (states, conf_interval_margin, forward_step=1, max_time=None, max_events=None))]
            fn py_calc_forward_probabilities_adaptive<'py>(
                &self,
                states: Vec<Bound<'py, PyState>>,
                conf_interval_margin: f64,
                forward_step: NumTiles,
                max_time: Option<f64>,
                max_events: Option<NumEvents>,
                py: Python<'py>,
            ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<usize>>)> {

                let refs = states.iter().map(|x| x.borrow()).collect::<Vec<_>>();
                let states = refs.iter().map(|x| &x.0).collect::<Vec<_>>();
                let (probabilities, trials) = py.detach(|| {
                    self.calc_forward_probabilities_adaptive(&states, forward_step, max_time, max_events, conf_interval_margin)
                }).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

                Ok((probabilities.into_pyarray(py), trials.into_pyarray(py)))
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

                let res = py.detach(|| self.run_ffs(&c));
                match res {
                    Ok(res) => Ok(res),
                    Err(err) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        err.to_string(),
                    )),
                }
            }

            fn __repr__(&self) -> String {
                format!("System({})", System::system_info(self))
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

            /// Place a tile at a point in the given state.
            ///
            /// Parameters
            /// ----------
            /// state : PyState
            ///     The state to modify.
            /// point : tuple of int
            ///     The coordinates at which to place the tile (i, j).
            /// tile : int
            ///     The tile number to place.
            ///
            /// Returns
            /// -------
            /// float
            ///     The energy change from placing the tile.
            #[pyo3(name = "place_tile")]
            pub fn py_place_tile(
                &self,
                state: &mut PyState,
                point: (usize, usize),
                tile: u32,
            ) -> Result<f64, RgrowError> {
                let pt = PointSafe2(point);
                let energy_change = self.place_tile(&mut state.0, pt, tile.into())?;
                Ok(energy_change)
            }
        }
    };
}

create_py_system!(KTAM);
create_py_system!(ATAM);
create_py_system!(OldKTAM);
create_py_system!(SDC);
create_py_system!(KBlock);

#[pymethods]
impl KBlock {
    #[getter]
    fn get_seed(&self) -> HashMap<(usize, usize), u32> {
        self.seed
            .clone()
            .into_iter()
            .map(|(k, v)| (k.0, v.into()))
            .collect()
    }

    #[setter]
    fn set_seed(&mut self, seed: HashMap<(usize, usize), u32>) {
        self.seed = seed
            .into_iter()
            .map(|(k, v)| {
                (
                    PointSafe2(k),
                    crate::models::kblock::TileType(v as usize).unblocked(),
                )
            })
            .collect();
    }

    #[getter]
    fn get_glue_links<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.glue_links.mapv(|x| x.into()).to_pyarray(py)
    }

    #[setter]
    fn set_glue_links(&mut self, glue_links: &Bound<PyArray2<f64>>) {
        self.glue_links = glue_links.to_owned_array().mapv(|x| x.into());
        self.update();
    }

    fn py_get_tile_raw_glues(&self, tile: u32) -> Vec<usize> {
        self.get_tile_raw_glues(tile.into())
    }

    fn py_get_tile_uncovered_glues(&self, tile: u32) -> Vec<usize> {
        self.get_tile_unblocked_glues(tile.into())
    }

    #[getter]
    fn get_cover_concentrations(&self) -> Vec<f64> {
        self.blocker_concentrations
            .clone()
            .into_iter()
            .map(|x| x.into())
            .collect()
    }

    #[setter]
    fn set_cover_concentrations(&mut self, cover_concentrations: Vec<f64>) {
        self.blocker_concentrations = cover_concentrations.into_iter().map(|x| x.into()).collect();
        self.update();
    }
}
