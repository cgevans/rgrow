use std::time::Duration;

use numpy::PyArray2;
use numpy::ToPyArray;
use pyo3::{prelude::*, types::PyType};

use rgrow::ffs;
use rgrow::simulation;
use rgrow::system::EvolveBounds;
use rgrow::tileset;

#[derive(FromPyObject)]
enum Ident {
    Num(usize),
    Name(String),
}

impl IntoPy<PyObject> for Ident {
    fn into_py(self, py: Python) -> PyObject {
        match self {
            Ident::Num(num) => num.into_py(py),
            Ident::Name(name) => name.into_py(py),
        }
    }
}

impl From<Ident> for tileset::GlueIdent {
    fn from(ident: Ident) -> Self {
        match ident {
            Ident::Num(num) => tileset::GlueIdent::Num(num),
            Ident::Name(name) => tileset::GlueIdent::Name(name),
        }
    }
}

impl From<Ident> for tileset::TileIdent {
    fn from(ident: Ident) -> Self {
        match ident {
            Ident::Num(num) => tileset::TileIdent::Num(num),
            Ident::Name(name) => tileset::TileIdent::Name(name),
        }
    }
}

#[pyclass]
pub struct Tile(tileset::Tile);

#[pymethods]
impl Tile {
    #[new]
    fn new(
        edges: Vec<Ident>,
        name: Option<String>,
        stoic: Option<f64>,
        color: Option<String>,
    ) -> Self {
        let edges: Vec<tileset::GlueIdent> = edges.into_iter().map(|e| e.into()).collect();
        let tile = tileset::Tile {
            name,
            edges,
            stoic,
            color,
        };
        Tile(tile)
    }

    #[getter]
    fn get_name(&self) -> Option<&str> {
        self.0.name.as_deref()
    }

    #[setter]
    fn set_name(&mut self, name: Option<String>) {
        self.0.name = name;
    }

    #[getter]
    fn get_stoic(&self) -> Option<f64> {
        self.0.stoic
    }

    #[setter]
    fn set_stoic(&mut self, stoic: Option<f64>) {
        self.0.stoic = stoic;
    }

    #[getter]
    fn get_color(&self) -> Option<&str> {
        self.0.color.as_deref()
    }

    #[setter]
    fn set_color(&mut self, color: Option<String>) {
        self.0.color = color;
    }

    #[getter]
    fn get_edges(&self) -> Vec<Ident> {
        self.0
            .edges
            .iter()
            .map(|e| match e {
                tileset::GlueIdent::Name(name) => Ident::Name(name.clone()),
                tileset::GlueIdent::Num(num) => Ident::Num(*num),
            })
            .collect()
    }

    #[setter]
    fn set_edges(&mut self, edges: Vec<Ident>) {
        self.0.edges = edges.into_iter().map(|e| e.into()).collect();
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

#[pyclass]
pub struct TileSet(tileset::TileSet);

#[pymethods]
impl TileSet {
    #[classmethod]
    fn from_json(_cls: &PyType, data: &str) -> PyResult<Self> {
        let tileset = tileset::TileSet::from_json(data);
        match tileset {
            Ok(tileset) => Ok(TileSet(tileset)),
            Err(err) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                err.to_string(),
            )),
        }
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }

    fn to_simulation(&self) -> PyResult<Simulation> {
        let sim = self.0.into_simulation();
        match sim {
            Ok(sim) => Ok(Simulation(sim)),
            Err(err) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                err.to_string(),
            )),
        }
    }

    #[cfg(feature = "ui")]
    fn run_window(&self) -> PyResult<Simulation> {
        let f = self.0.clone();
        let s = rgrow::ui::run_window(&f);

        let s2 =
            s.map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))?;
        Ok(Simulation(s2))
    }

    #[allow(clippy::too_many_arguments)]
    #[args(
        varpermean2 = "1e-4",
        min_configs = "1_000",
        max_size = "200",
        cutoff_probability = "0.99",
        cutoff_surfaces = "4",
        min_cutoff_size = "30",
        surface_size_step = "1",
        surface_init_size = "3",
        max_init_events = "10_000",
        max_subseq_events = "1_000_000",
        keep_surface_configs = "false"
    )]
    fn run_ffs(
        &self,
        varpermean2: f64,
        min_configs: usize,
        max_size: u32,
        cutoff_probability: f64,
        cutoff_surfaces: usize,
        min_cutoff_size: u32,
        max_init_events: u64,
        max_subseq_events: u64,
        surface_init_size: u32,
        surface_size_step: u32,
        keep_surface_configs: bool,
        min_nuc_rate: Option<f64>,
        py: Python<'_>,
    ) -> PyResult<FFSResult> {
        let res = py.allow_threads(|| {
            self.0.run_ffs(
                varpermean2,
                min_configs,
                max_size,
                cutoff_probability,
                cutoff_surfaces,
                min_cutoff_size,
                max_init_events,
                max_subseq_events,
                surface_init_size,
                surface_size_step,
                keep_surface_configs,
                min_nuc_rate,
            )
        });
        match res {
            Ok(res) => Ok(FFSResult(res)),
            Err(err) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                err.to_string(),
            )),
        }
    }
}

#[pyclass]
pub struct Simulation(Box<dyn simulation::Simulation>);

#[pymethods]
impl Simulation {
    /// Evolves an individual state within bounds.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        text_signature = "($self, state_index, for_events, for_time, size_min, size_max, for_wall_time)"
    )]
    fn evolve(
        &mut self,
        state_index: Option<usize>,
        for_events: Option<u64>,
        for_time: Option<f64>,
        size_min: Option<u32>,
        size_max: Option<u32>,
        for_wall_time: Option<f64>,
        py: Python<'_>,
    ) -> PyResult<()> {
        // If all bounds are none, raise an error.
        if state_index.is_none()
            && for_events.is_none()
            && for_time.is_none()
            && size_min.is_none()
            && size_max.is_none()
            && for_wall_time.is_none()
        {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "No bounds specified.",
            ));
        }

        let bounds = EvolveBounds {
            events: for_events,
            time: for_time,
            size_min,
            size_max,
            wall_time: for_wall_time.map(Duration::from_secs_f64),
        };

        py.allow_threads(|| {
            self.0
                .evolve(state_index.unwrap_or(0), bounds)
                .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))
                .map(|_x| ())
        })
    }

    #[cfg(feature = "use_rayon")]
    fn evolve_all(
        &mut self,
        for_events: Option<u64>,
        for_time: Option<f64>,
        size_min: Option<u32>,
        size_max: Option<u32>,
        for_wall_time: Option<f64>,
        py: Python<'_>,
    ) -> PyResult<()> {
        // If all bounds are none, raise an error.
        if for_events.is_none()
            && for_time.is_none()
            && size_min.is_none()
            && size_max.is_none()
            && for_wall_time.is_none()
        {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "No bounds specified.",
            ));
        }

        let bounds = EvolveBounds {
            events: for_events,
            time: for_time,
            size_min,
            size_max,
            wall_time: for_wall_time.map(Duration::from_secs_f64),
        };

        py.allow_threads(|| self.0.evolve_all(bounds)); // FIXME: handle errors

        Ok(())
    }

    /// Returns the current canvas for state_index (default 0), as an array copy.
    #[pyo3(text_signature = "($self, state_index)")]
    fn canvas<'py>(&self, state_index: Option<usize>, py: Python<'py>) -> &'py PyArray2<usize> {
        self.0
            .state_ref(state_index.unwrap_or(0))
            .raw_array()
            .to_pyarray(py)
    }

    fn state_ntiles(&self, state_index: Option<usize>) -> u32 {
        self.0.state_ref(state_index.unwrap_or(0)).ntiles()
    }

    fn state_time(&self, state_index: Option<usize>) -> f64 {
        self.0.state_ref(state_index.unwrap_or(0)).time()
    }

    fn state_total_events(&self, state_index: Option<usize>) -> u64 {
        self.0.state_ref(state_index.unwrap_or(0)).total_events()
    }

    /// Add a new state to the simulation.
    #[pyo3(text_signature = "($self, shape)")]
    fn add_state(&mut self, shape: (usize, usize)) -> PyResult<usize> {
        self.0
            .add_state(shape)
            .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))
    }

    #[pyo3(text_signature = "($self, n, shape")]
    fn add_n_states(&mut self, n: usize, shape: (usize, usize)) -> PyResult<Vec<usize>> {
        self.0
            .add_n_states(n, shape)
            .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))
    }
}

#[pyclass]
pub struct FFSResult(Box<dyn ffs::FFSResult>);

#[pymethods]
impl FFSResult {
    #[getter]
    fn get_nucleation_rate(&self) -> f64 {
        self.0.nucleation_rate()
    }

    #[getter]
    fn get_forward_vec(&self) -> Vec<f64> {
        self.0.forward_vec().clone()
    }

    fn __str__(&self) -> String {
        format!(
            "{:1.4e} {:?}",
            self.0.nucleation_rate(),
            self.0.forward_vec()
        )
    }

    fn __repr__(&self) -> String {
        self.__str__()
    }
}
