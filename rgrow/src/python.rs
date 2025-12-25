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
use crate::system::{CriticalStateConfig, CriticalStateResult};
use crate::system::{
    DimerInfo, DynSystem, EvolveBounds, EvolveOutcome, NeededUpdate, System, TileBondInfo,
};
use crate::units::Second;
use ndarray::Array2;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray2, ToPyArray,
};
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

    /// Serialize state for pickling.
    fn __getstate__(&self) -> PyResult<Vec<u8>> {
        bincode::serialize(&self.0).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Failed to serialize state: {e}"
            ))
        })
    }

    /// Deserialize state from pickle data.
    fn __setstate__(&mut self, state: Vec<u8>) -> PyResult<()> {
        self.0 = bincode::deserialize(&state).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Failed to deserialize state: {e}"
            ))
        })?;
        Ok(())
    }

    /// Return arguments for __new__ during unpickling.
    fn __getnewargs__(&self) -> ((usize, usize),) {
        ((1, 1),)
    }

    /// Replay the events from a MovieTracker up to a given event ID.
    ///
    /// This reconstructs the state by replaying all events from the MovieTracker.
    /// The state must have been created with Movie tracking enabled.
    ///
    /// Parameters
    /// ----------
    /// up_to_event : int, optional
    ///     The event ID up to which to replay (inclusive). If not provided,
    ///     all events are replayed.
    ///
    /// Returns
    /// -------
    /// State
    ///     A new State with the events replayed. The returned state has no
    ///     tracker and no rates calculated.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the state does not have a MovieTracker.
    ///
    /// Examples
    /// --------
    /// >>> # Create a state with movie tracking and evolve it
    /// >>> state = ts.create_state(tracking="Movie")
    /// >>> sys.evolve(state, for_events=100)
    /// >>> # Replay to get state at event 50
    /// >>> replayed = state.replay(up_to_event=50)
    #[pyo3(signature = (up_to_event=None))]
    pub fn replay(&self, up_to_event: Option<u64>) -> PyResult<Self> {
        self.0
            .replay(up_to_event)
            .map(PyState)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    /// Replay events in-place on this state from external event data.
    ///
    /// This modifies the state's canvas by applying the events from the provided
    /// coordinate and tile arrays. Unlike `replay()`, this method takes external
    /// event data rather than using a MovieTracker.
    ///
    /// Parameters
    /// ----------
    /// coords : list[tuple[int, int]]
    ///     List of (row, col) coordinates for each event.
    /// new_tiles : list[int]
    ///     List of tile values for each event.
    /// event_ids : list[int]
    ///     List of event IDs for each event.
    /// up_to_event_id : int
    ///     The event ID up to which to replay (inclusive).
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If there is an error during replay.
    ///
    /// Examples
    /// --------
    /// >>> state = State((10, 10))
    /// >>> coords = [(1, 1), (2, 2)]
    /// >>> new_tiles = [1, 2]
    /// >>> event_ids = [0, 1]
    /// >>> state.replay_inplace(coords, new_tiles, event_ids, 1)
    pub fn replay_inplace(
        &mut self,
        coords: Vec<(usize, usize)>,
        new_tiles: Vec<Tile>,
        event_ids: Vec<u64>,
        up_to_event_id: u64,
        n_tiles: Option<Vec<u32>>,
        total_time: Option<Vec<f64>>,
        energy: Option<Vec<f64>>,
    ) -> PyResult<()> {
        let total_time_seconds: Option<Vec<Second>> =
            total_time.map(|v| v.into_iter().map(Second::new).collect());
        self.0
            .replay_inplace(
                &coords,
                &new_tiles,
                &event_ids,
                up_to_event_id,
                n_tiles.as_ref().map(|v| v.as_slice()),
                total_time_seconds.as_ref().map(|v| v.as_slice()),
                energy.as_ref().map(|v| v.as_slice()),
            )
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
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

#[cfg(feature = "python")]
#[derive(FromPyObject)]
pub enum PyStateOrCanvasRef<'py> {
    State(Bound<'py, PyState>),
    Ref(Bound<'py, FFSStateRef>),
    Array(Bound<'py, PyArray2<Tile>>),
}

impl From<FFSStateRef> for PyState {
    fn from(state: FFSStateRef) -> Self {
        state.clone_state()
    }
}

macro_rules! create_py_system {
    ($name: ident) => {
        create_py_system!($name, |tile: u32| tile as usize);
    };
    ($name: ident, $tile_index_fn: expr) => {
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
                                                                    start_window_paused=true,
                                                                    parallel=true,
                                                                    initial_timescale=None,
                                                                    initial_max_events_per_sec=None)
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
            ///   Show a graphical UI window while evolving (requires rgrow-gui to be installed, and a single state).
            /// start_window_paused : bool
            ///   If show_window is True, start the GUI window in a paused state. Defaults to True.
            /// parallel : bool
            ///   Use multiple threads.
            /// initial_timescale : float, optional
            ///   If show_window is True, set the initial timescale (sim_time/real_time) in the GUI. None means unlimited.
            /// initial_max_events_per_sec : int, optional
            ///   If show_window is True, set the initial max events per second limit in the GUI. None means unlimited.
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
                start_window_paused: bool,
                parallel: bool,
                initial_timescale: Option<f64>,
                initial_max_events_per_sec: Option<u64>,
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

                if require_strong_bound && !show_window && !bounds.is_strongly_bounded() {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "No strong bounds specified.",
                    ));
                }

                if !show_window && !bounds.is_weakly_bounded() {
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
                                    System::evolve_in_window(self, state, None, start_window_paused, bounds, initial_timescale, initial_max_events_per_sec)
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

            /// Returns the current canvas for state as an array of tile names.
            /// 'empty' indicates empty locations.
            ///
            /// Parameters
            /// ----------
            /// state : State or FFSStateRef
            ///   The state to return.
            ///
            /// Returns
            /// -------
            /// NDArray[str]
            ///   The current canvas for the state, as an array of tile names.
            fn name_canvas<'py>(&self, state: PyStateOrRef<'py>, py: Python<'py>) -> PyResult<Py<PyArray2<Py<PyAny>>>> {
                let tile_names = TileBondInfo::tile_names(self);
                let canvas = match &state {
                    PyStateOrRef::State(s) => s.borrow().0.raw_array().to_owned(),
                    PyStateOrRef::Ref(s) => s.borrow().clone_state().0.raw_array().to_owned(),
                };
                let tile_index_fn = $tile_index_fn;
                let name_array = canvas.mapv(|tile| {
                    let tile_index: usize = tile_index_fn(tile);
                    tile_names[tile_index].clone().into_pyobject(py).unwrap().unbind().into()
                });
                Ok(name_array.into_pyarray(py).unbind())
            }

            /// Returns the current canvas for state as an array of tile colors.
            ///
            /// Parameters
            /// ----------
            /// state : State, FFSStateRef, or NDArray
            ///   The state or canvas array to colorize.
            ///
            /// Returns
            /// -------
            /// NDArray[uint8]
            ///   The current canvas for the state, as an array of RGBA colors with shape (rows, cols, 4).
            fn color_canvas<'py>(
                &self,
                state: PyStateOrCanvasRef<'py>,
                py: Python<'py>,
            ) -> PyResult<Bound<'py, PyArray3<u8>>> {
                let colors = TileBondInfo::tile_colors(self);
                let canvas = match &state {
                    PyStateOrCanvasRef::State(s) => s.borrow().0.raw_array().to_owned(),
                    PyStateOrCanvasRef::Ref(s) => s.borrow().clone_state().0.raw_array().to_owned(),
                    PyStateOrCanvasRef::Array(arr) => arr.readonly().as_array().to_owned(),
                };
                let tile_index_fn = $tile_index_fn;
                let (rows, cols) = canvas.dim();
                let mut color_array = ndarray::Array3::<u8>::zeros((rows, cols, 4));
                for ((i, j), &tile) in canvas.indexed_iter() {
                    let tile_index: usize = tile_index_fn(tile);
                    let c = colors[tile_index];
                    color_array[[i, j, 0]] = c[0];
                    color_array[[i, j, 1]] = c[1];
                    color_array[[i, j, 2]] = c[2];
                    color_array[[i, j, 3]] = c[3];
                }
                Ok(color_array.into_pyarray(py))
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

            /// Determine whether the committer probability for a state is above or below a threshold
            /// with a specified confidence level using adaptive sampling.
            ///
            /// This function uses adaptive sampling to determine with the desired confidence whether
            /// the true committer probability is above or below the given threshold. It continues
            /// sampling until the confidence interval is narrow enough to make a definitive determination.
            ///
            /// Parameters
            /// ----------
            /// state : State
            ///     The state to analyze
            /// cutoff_size : int
            ///     Size threshold for commitment
            /// threshold : float
            ///     The probability threshold to compare against (e.g., 0.5)
            /// confidence_level : float
            ///     Confidence level for the threshold test (e.g., 0.95 for 95% confidence)
            /// max_time : float, optional
            ///     Maximum simulation time per trial
            /// max_events : int, optional
            ///     Maximum events per trial
            /// max_trials : int, optional
            ///     Maximum number of trials to run (default: 100000)
            /// return_on_max_trials : bool, optional
            ///     If True, return results even when max_trials is exceeded (default: False)
            /// ci_confidence_level : float, optional
            ///     Confidence level for the returned confidence interval (default: None, no CI returned)
            ///     Can be different from confidence_level (e.g., test at 95%, show 99% CI)
            ///
            /// Returns
            /// -------
            /// tuple[bool, float, tuple[float, float] | None, int, bool]
            ///     Tuple of (is_above_threshold, probability_estimate, confidence_interval, num_trials, exceeded_max_trials) where:
            ///     - is_above_threshold: True if probability is above threshold with given confidence
            ///     - probability_estimate: The estimated probability
            ///     - confidence_interval: Tuple of (lower_bound, upper_bound) or None if ci_confidence_level not provided
            ///     - num_trials: Number of trials performed
            ///     - exceeded_max_trials: True if max_trials was exceeded (warning flag)
            #[allow(clippy::too_many_arguments)]
            #[pyo3(name = "calc_committer_threshold_test", signature = (state, cutoff_size, threshold, confidence_level, max_time=None, max_events=None, max_trials=None, return_on_max_trials=false))]
            fn py_calc_committer_threshold_test(
                &mut self,
                state: &mut PyState,
                cutoff_size: NumTiles,
                threshold: f64,
                confidence_level: f64,
                max_time: Option<f64>,
                max_events: Option<NumEvents>,
                max_trials: Option<usize>,
                return_on_max_trials: bool,
                py: Python<'_>,
            ) -> PyResult<(bool, f64, usize, bool)> {
                py.detach(|| {
                    self.calc_committer_threshold_test(
                        &state.0,
                        cutoff_size,
                        threshold,
                        confidence_level,
                        max_time,
                        max_events,
                        max_trials,
                        return_on_max_trials,
                    )
                })
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
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

            // /// Find the first state in a trajectory above the critical threshold.
            // ///
            // /// Iterates through the trajectory (after filtering redundant events),
            // /// reconstructing the state at each point and testing if the committer
            // /// probability is above the threshold with the specified confidence.
            // ///
            // /// Parameters
            // /// ----------
            // /// trajectory : pl.DataFrame
            // ///     DataFrame with columns: row, col, new_tile, energy
            // /// config : CriticalStateConfig, optional
            // ///     Configuration for the search (uses defaults if not provided)
            // ///
            // /// Returns
            // /// -------
            // /// CriticalStateResult | None
            // ///     The first critical state found, or None if no state is above threshold.
            #[pyo3(name = "find_first_critical_state", signature = (end_state, config=CriticalStateConfig::default()))]
            pub fn py_find_first_critical_state(
                &mut self,
                end_state: &PyState,
                config: CriticalStateConfig,
                py: Python<'_>,
            ) -> PyResult<Option<CriticalStateResult>> {
                py.detach(|| {
                    self.find_first_critical_state(&end_state.0, &config)
                })
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
            }

            // TODO: Uncomment when find_last_critical_state is implemented on System trait
            // /// Find the last state not above threshold, return the next state.
            // ///
            // /// Iterates backwards through the trajectory to find the last state that is
            // /// NOT above the critical threshold, then returns the next state (which should
            // /// be above threshold). This is useful for finding the "critical nucleus".
            // ///
            // /// Parameters
            // /// ----------
            // /// trajectory : pl.DataFrame
            // ///     DataFrame with columns: row, col, new_tile, energy
            // /// config : CriticalStateConfig, optional
            // ///     Configuration for the search (uses defaults if not provided)
            // ///
            // /// Returns
            // /// -------
            // /// CriticalStateResult | None
            // ///     The first state above threshold (following the last subcritical state),
            // ///     or None if no transition is found.
            #[pyo3(name = "find_last_critical_state", signature = (end_state, config=CriticalStateConfig::default()))]
            pub fn py_find_last_critical_state(
                &mut self,
                end_state: &PyState,
                config: CriticalStateConfig,
                py: Python<'_>,
            ) -> PyResult<Option<CriticalStateResult>> {
                py.detach(|| {
                    self.find_last_critical_state(&end_state.0, &config)
                })
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
            }
        }
    };
}

create_py_system!(KTAM);
create_py_system!(ATAM);
create_py_system!(OldKTAM);
create_py_system!(SDC);
create_py_system!(KBlock, |tile: u32| (tile >> 4) as usize);

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
