use std::ops::DerefMut;
use std::time::Duration;

use crate::base::{NumEvents, NumTiles, RustAny, Tile};
use crate::canvas::Canvas;
use crate::ffs::{BoxedFFSResult, FFSRunConfig};
use crate::ratestore::RateStore;
use crate::state::{StateEnum, StateStatus, TrackerData};
use crate::system::{
    DynSystem, EvolveBounds, EvolveOutcome, NeededUpdate, SystemEnum, TileBondInfo,
};
use crate::tileset::CanvasType;
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};

#[cfg_attr(feature = "python", pyclass(name = "State"))]
#[repr(transparent)]
pub struct PyState(pub(crate) StateEnum);

/// A single 'assembly', or 'state', containing a canvas with tiles at locations.
/// Generally does not store concentration or temperature information, but does store time simulated.
#[cfg(feature = "python")]
#[pymethods]
impl PyState {
    #[getter]
    /// A direct, mutable view of the state's canvas.  This is potentially unsafe.
    pub fn canvas_view<'py>(
        this: Bound<'py, Self>,
        _py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray2<crate::base::Tile>>> {
        let t = this.borrow();
        let ra = t.0.raw_array();

        unsafe { Ok(PyArray2::borrow_from_array_bound(&ra, this.into_any())) }
    }

    /// A copy of the state's canvas.  This is safe, but can't be modified and is slower than `canvas_view`.
    pub fn canvas_copy<'py>(
        this: &Bound<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray2<crate::base::Tile>>> {
        let t = this.borrow();
        let ra = t.0.raw_array();

        Ok(PyArray2::from_array_bound(py, &ra))
    }

    pub fn tracking_copy(
        this: &Bound<Self>,
    ) -> PyResult<RustAny> {
        let t = this.borrow();
        let ra = t.0.get_tracker_data();

        Ok(ra)
    }

    /// The number of tiles in the state.
    #[getter]
    pub fn n_tiles(&self) -> NumTiles {
        self.0.n_tiles()
    }

    /// The number of tiles in the state (deprecated, use `n_tiles` instead).
    #[getter]
    pub fn ntiles(&self) -> NumTiles {
        self.0.n_tiles()
    }

    /// The total number of events that have occurred in the state.
    #[getter]
    pub fn total_events(&self) -> NumEvents {
        self.0.total_events()
    }

    /// The total time the state has simulated, in seconds.
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
}

#[cfg(feature = "python")]
#[derive(FromPyObject)]
pub enum PyStateOrStates<'py> {
    #[pyo3(transparent)]
    State(Bound<'py, PyState>),
    #[pyo3(transparent)]
    States(Vec<Bound<'py, PyState>>),
}

#[repr(transparent)]
#[cfg_attr(
    feature = "python",
    pyclass(module = "rgrow", name = "System", subclass)
)]
pub struct PySystem(pub SystemEnum);

#[cfg(feature = "python")]
#[pymethods]
impl PySystem {
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
                        .allow_threads(|| self.0.evolve_in_window(state, None, bounds))?
                        .into_py(py))
                } else {
                    Ok(py
                        .allow_threads(|| self.0.evolve(state, bounds))?
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
                            .map(|state| self.0.evolve(&mut state.0, bounds))
                            .collect::<Vec<_>>()
                    })
                } else {
                    states
                        .iter_mut()
                        .map(|state| self.0.evolve(&mut state.0, bounds))
                        .collect::<Vec<_>>()
                };
                let o: Result<Vec<EvolveOutcome>, PyErr> = out
                    .into_iter()
                    .map(|x| x.map_err(|y| pyo3::exceptions::PyValueError::new_err(y.to_string())))
                    .collect();
                o.map(|x| x.into_py(py))
            }
        }
    }

    fn calc_mismatches(&self, state: &PyState) -> usize {
        self.0.calc_mismatches(&state.0)
    }

    fn calc_mismatch_locations<'py>(
        this: &Bound<'py, Self>,
        state: &PyState,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray2<usize>>> {
        let t = this.borrow();
        let ra = t.0.calc_mismatch_locations(&state.0);
        Ok(PyArray2::from_array_bound(py, &ra))
    }

    fn set_param(&mut self, param_name: &str, value: RustAny) -> PyResult<NeededUpdate> {
        Ok(self.0.set_param(param_name, value.0)?)
    }

    /// Names of tiles, per number.
    // #[getter]
    // fn tile_names(&self, py: Python<'_>) -> PyArray1<PyFixedUnicode<MAX_NAME_LENGTH>> {
    //     PyArray1::from_vec(py, self.0.tile_names()).into()
    // }

    #[getter]
    fn tile_names(&self) -> Vec<String> {
        self.0.tile_names().iter().map(|x| x.to_string()).collect()
    }

    fn tile_number(&self, tile_name: &str) -> Option<Tile> {
        self.0
            .tile_names()
            .iter()
            .position(|x| *x == tile_name)
            .map(|x| x as Tile)
    }

    fn tile_color(&self, tile_number: Tile) -> [u8; 4] {
        self.0.tile_color(tile_number)
    }

    #[getter]
    fn tile_colors<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u8>> {
        let colors = self.0.tile_colors();
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
        Ok(RustAny(self.0.get_param(param_name)?))
    }

    fn update_all(&self, state: &mut PyState, needed: &NeededUpdate) {
        self.0.update_all(&mut state.0, needed)
    }

    #[pyo3(name = "run_ffs", signature = (config = FFSRunConfig::default(), canvas_type = None, **kwargs))]
    fn py_run_ffs(
        &mut self,
        config: FFSRunConfig,
        canvas_type: Option<CanvasType>,
        kwargs: Option<Bound<PyDict>>,
        py: Python<'_>,
    ) -> PyResult<BoxedFFSResult> {
        let mut c = config;

        if let Some(dict) = kwargs {
            for (k, v) in dict.iter() {
                c._py_set(&k.extract::<String>()?, v)?;
            }
        }

        let res = py.allow_threads(|| self.0.run_ffs(&c, canvas_type));
        match res {
            Ok(res) => Ok(res),
            Err(err) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                err.to_string(),
            )),
        }
    }

    fn __repr__(&self) -> String {
        format!("System({})", self.0.system_info())
    }
}
