use std::any::Any;
use std::collections::HashMap;
use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;
use std::sync::RwLock;
use std::sync::RwLockReadGuard;
use std::sync::RwLockWriteGuard;
use std::time::Duration;

use numpy::PyArray2;
use numpy::ToPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyDict;
use pyo3::{prelude::*, types::PyType};

use rgrow::ffs;
use rgrow::ffs::FFSRunConfig;
use rgrow::simulation;
use rgrow::system::EvolveBounds;
use rgrow::system::EvolveOutcome;
use rgrow::tileset;
use rgrow::tileset::Ident;
use rgrow::tileset::TileShape;



#[derive(FromPyObject, Clone)]
struct Bond(Ident, f64);

impl From<tileset::Tile> for Tile {
    fn from(tile: tileset::Tile) -> Self {
        Tile(tile)
    }
}

#[pyclass]
#[repr(transparent)]
pub struct Tile(tileset::Tile);

#[pymethods]
impl Tile {
    #[new]
    fn new(
        edges: Vec<Ident>,
        name: Option<String>,
        stoic: Option<f64>,
        color: Option<String>,
        _shape: Option<String>,
    ) -> Self {
        let edges: Vec<tileset::GlueIdent> = edges.into_iter().collect();
        let tile = tileset::Tile {
            name,
            edges,
            stoic,
            color,
            shape: Some(TileShape::Single),
        };
        Self(tile)
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
    /// The glues on the edges of the tile, in clockwise order starting from the North,
    /// or the North-facing edge furthest to the West if not a single tile.
    ///
    /// Glues should be either strings, integers (starting at 1), or None or 0 to
    /// refer to a null glue.
    fn get_edges(&self) -> Vec<Ident> {
        self.0.edges.to_vec()
    }

    #[setter]
    fn set_edges(&mut self, edges: Vec<Ident>) {
        self.0.edges = edges.into_iter().collect();
    }

    fn __repr__(&self) -> String {
        self.0.to_string()
    }
}

/// A class representing a tile set.
///
/// Parameters
/// ----------
///
/// tiles : list[Tile]
///    The tiles in the tile set.
/// bonds : list[tuple[str | int, float]]
///   The bonds in the tile set. Each bond is a tuple of the name of the glue and the strength of the bond.
/// glues : list[tuple[str | int, str | int, float]]
///   Specific glue-glue interactions.
/// options : dict
///   Options for the tile set.
#[pyclass]
#[repr(transparent)]
pub struct TileSet(Arc<RwLock<tileset::TileSet>>);

impl TileSet {
    fn read(&self) -> PyResult<std::sync::RwLockReadGuard<'_, tileset::TileSet>> {
        let x = self
            .0
            .read()
            .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))?;
        Ok(x)
    }

    fn write(&self) -> PyResult<std::sync::RwLockWriteGuard<'_, tileset::TileSet>> {
        let x = self
            .0
            .write()
            .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))?;
        Ok(x)
    }
}

#[pymethods]
impl TileSet {
    #[new]
    #[pyo3(signature = (tiles, bonds=Vec::default(), glues=Vec::default(), options=HashMap::default()))]
    fn new(
        tiles: Vec<&PyCell<Tile>>,
        bonds: Vec<(Ident, f64)>,
        glues: Vec<(Ident, Ident, f64)>,
        options: HashMap<String, RustAny>,
    ) -> PyResult<TileSet> {
        let opts_any: HashMap<String, Box<dyn Any>> =
            options.into_iter().map(|(k, v)| (k, v.0)).collect();
        let args = opts_any.into();
        let tileset = tileset::TileSet {
            tiles: tiles.into_iter().map(|x| x.borrow().0.clone()).collect(),
            bonds: bonds
                .iter()
                .map(|x| tileset::Bond {
                    name: x.0.clone(),
                    strength: x.1,
                })
                .collect(),
            glues: glues
                .iter()
                .map(|x| (x.0.clone(), x.1.clone(), x.2))
                .collect(),
            options: args,
            cover_strands: None,
        };
        Ok(Self(Arc::new(RwLock::new(tileset))))
    }

    /// Parses a JSON string into a TileSet.
    #[classmethod]
    fn from_json(_cls: &PyType, data: &str) -> PyResult<Self> {
        let tileset = tileset::TileSet::from_json(data);
        match tileset {
            Ok(tileset) => Ok(TileSet(Arc::new(RwLock::new(tileset)))),
            Err(err) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                err.to_string(),
            )),
        }
    }

    /// Creates a TileSet from a dict by exporting to json, then parsing the json.
    /// FIXME: implement this without the json trip.
    #[classmethod]
    fn from_dict(_cls: &PyType, data: PyObject) -> PyResult<Self> {
        let json: String = Python::with_gil(|py| {
            let json = PyModule::import(py, "json")?;
            json.call_method1("dumps", (data,))?.extract::<String>()
        })?;

        let tileset = tileset::TileSet::from_json(&json);
        match tileset {
            Ok(tileset) => Ok(TileSet(Arc::new(RwLock::new(tileset)))),
            Err(err) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                err.to_string(),
            )),
        }
    }

    /// Parses a file (JSON, YAML, etc) into a TileSet
    #[classmethod]
    fn from_file(_cls: &PyType, path: &str) -> PyResult<Self> {
        let ts = tileset::TileSet::from_file(path)
            .map_err(|err| PyErr::new::<PyValueError, _>(err.to_string()))?;
        Ok(TileSet(Arc::new(RwLock::new(ts))))
    }

    /// Creates a :any:`Simulation` from the TileSet.
    fn to_simulation(&self) -> PyResult<Simulation> {
        let sim = self.write()?.into_simulation();
        match sim {
            Ok(sim) => Ok(Simulation(RwLock::new(sim))),
            Err(err) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                err.to_string(),
            )),
        }
    }

    /// Creates a simulation, and runs it in a UI.  Returns the :any:`Simulation` when
    /// finished.
    #[cfg(feature = "ui")]
    fn run_window(&self) -> PyResult<Simulation> {
        let f = self.read()?;
        let s = rgrow::ui::run_window(&f);

        let s2 =
            s.map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))?;
        Ok(Simulation(RwLock::new(s2)))
    }

    #[getter]
    fn get_tiles(&self) -> PyResult<Vec<Tile>> {
        Ok(self
            .read()?
            .tiles
            .iter()
            .map(|t| t.clone().into())
            .collect())
    }

    /// Runs FFS.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (config = FFSRunConfig::default(), **kwargs))]
    fn run_ffs(
        &self,
        config: FFSRunConfig,
        kwargs: Option<&PyDict>,
        py: Python<'_>,
    ) -> PyResult<FFSResult> {
        let mut c = config;

        if let Some(dict) = kwargs {
            for (k, v) in dict.iter() {
                c._py_set(&k.extract::<String>()?, v, py)?;
            }
        }

        let res = py.allow_threads(|| self.read().unwrap().run_ffs(&c));
        match res {
            Ok(res) => Ok(FFSResult(res.into())),
            Err(err) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                err.to_string(),
            )),
        }
    }

    fn __repr__(&self) -> String {
        self.to_string()
    }
}

impl Display for TileSet {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let ts = self.read().unwrap();
        writeln!(f, "TileSet(")?;
        writeln!(f, "    tiles=[")?;
        for tile in &ts.tiles {
            writeln!(f, "        {},", tile)?;
        }
        writeln!(f, "    ],")?;
        if !&ts.bonds.is_empty() {
            writeln!(f, "    bonds=[")?;
            for bond in &ts.bonds {
                writeln!(f, "        ({}, {}),", bond.name, bond.strength)?;
            }
            writeln!(f, "    ],")?;
        };
        if !&ts.glues.is_empty() {
            writeln!(f, "    glues=[")?;
            for (a, b, s) in &ts.glues {
                writeln!(f, "        ({}, {}, {}),", a, b, s)?;
            }
            writeln!(f, "    ],")?;
        };
        writeln!(f, "    options=[")?;
        write!(f, "{}", indent::indent_all_by(8, ts.options.to_string()))?;
        write!(f, "    ]\n)\n")?;
        Ok(())
    }
}

/// A combination of a System, and a list of States, which can be added to.
///
/// This is not generally created directly, but is instead usually created
/// from a :any:`TileSet`, using :any:`TileSet.to_simulation()`.
#[pyclass]
pub struct Simulation(RwLock<Box<dyn simulation::Simulation>>);

#[pyclass]
pub struct TileColorView(Arc<RwLock<dyn simulation::Simulation>>);

impl Simulation {
    fn ensure_state(&mut self, state_index: Option<usize>) -> PyResult<usize> {
        let sim = self.read()?;

        match state_index {
            Some(x) => {
                if sim.n_states() <= x {
                    Err(PyValueError::new_err(format!(
                        "State index {x} is out of bounds."
                    )))
                } else {
                    Ok(x)
                }
            }
            None => {
                if sim.n_states() > 0 {
                    return Ok(0);
                }
                drop(sim);
                let mut sim = self.0.write().unwrap();
                sim.add_state()
                    .map_err(|x| PyValueError::new_err(x.to_string()))
            }
        }
    }
    fn check_state(&self, state_index: Option<usize>) -> PyResult<usize> {
        let state_index = state_index.unwrap_or(0);
        let sim = self.read()?;

        if sim.n_states() <= state_index {
            Err(PyValueError::new_err(format!(
                "State index {state_index} is out of bounds."
            )))
        } else {
            Ok(state_index)
        }
    }
    fn read(
        &self,
    ) -> Result<RwLockReadGuard<'_, Box<dyn rgrow::simulation::Simulation>>, pyo3::PyErr> {
        self.0
            .try_read()
            .map_err(|_| PyValueError::new_err("lock failure"))
    }
    fn write(
        &self,
    ) -> Result<RwLockWriteGuard<'_, Box<dyn rgrow::simulation::Simulation>>, pyo3::PyErr> {
        self.0
            .try_write()
            .map_err(|_| PyValueError::new_err("lock failure"))
    }
}

#[pymethods]
impl Simulation {
    /// Evolve a particular state, with index `state_index`,
    /// subject to some bounds.  Runs state 0 by default.
    ///
    /// By default, this requires a strong bound (the simulation
    /// will eventually end, eg, not a size or other potentially
    /// unreachable bound). Releases the GIL during the simulation.    
    ///
    /// Parameters
    /// ----------
    /// state_index : int, optional
    ///    The index of the state to evolve.  Defaults to 0, and creates sufficient states
    ///    if they do not already exist.
    /// for_events : int, optional
    ///    Evolve until this many events have occurred.  Defaults to no limit. (Strong bound)
    /// total_events : int, optional
    ///    Evolve until this many events have occurred in total.  Defaults to no limit. (Strong bound)
    /// for_time : float, optional
    ///    Evolve until this much (physical) time has passed.  Defaults to no limit. (Strong bound)
    /// total_time : float, optional
    ///    Evolve until this much (physical) time has passed since the state creation.  
    ///    Defaults to no limit. (Strong bound)
    /// size_min : int, optional
    ///    Evolve until the system has this many, or fewer, tiles. Defaults to no limit. (Weak bound)
    /// size_max : int, optional
    ///    Evolve until the system has this many, or more, tiles. Defaults to no limit. (Weak bound)
    /// for_wall_time : float, optional
    ///    Evolve until this much (wall) time has passed.  Defaults to no limit. (Strong bound)
    /// require_strong_bound : bool, optional
    ///    If True (default), a ValueError will be raised unless at least one strong bound has been
    ///    set, ensuring that the simulation will eventually end.  If False, ensure only that some
    ///    weak bound has been set, which may result in an infinite simulation.
    ///
    /// Returns
    /// -------
    ///
    /// EvolveOutcome
    ///   The stopping condition that caused the simulation to end.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (state_index=None,
                    for_events=None,
                    total_events=None,
                    for_time=None,
                    total_time=None,
                    size_min=None,
                    size_max=None,
                    for_wall_time=None,
                    require_strong_bound=true)
    )]
    fn evolve(
        &mut self,
        state_index: Option<usize>,
        for_events: Option<u64>,
        total_events: Option<u64>,
        for_time: Option<f64>,
        total_time: Option<f64>,
        size_min: Option<u32>,
        size_max: Option<u32>,
        for_wall_time: Option<f64>,
        require_strong_bound: bool,
        py: Python<'_>,
    ) -> PyResult<EvolveOutcome> {
        let state_index = self.ensure_state(state_index)?;

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

        py.allow_threads(|| {
            self.write()?
                .evolve(state_index, bounds)
                .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(
        signature = (for_events=None,
                    total_events=None,
                    for_time=None,
                    total_time=None,
                    size_min=None,
                    size_max=None,
                    for_wall_time=None,
                    require_strong_bound=true)
    )]
    #[cfg(feature = "use_rayon")]
    /// Evolve *all* states, stopping each as they reach the
    /// boundary conditions.  Runs multithreaded using available
    /// cores.
    ///
    /// By default, this requires a strong bound (the simulation
    /// will eventually end, eg, not a size or other potentially
    /// unreachable bound). Releases the GIL during the simulation.
    /// Bounds are applied for each state individually.    
    ///
    /// Parameters
    /// ----------
    /// state_index : int, optional
    ///    The index of the state to evolve.  Defaults to 0, and creates sufficient states
    ///    if they do not already exist.
    /// for_events : int, optional
    ///    Evolve until this many events have occurred.  Defaults to no limit. (Strong bound)
    /// total_events : int, optional
    ///    Evolve until this many events have occurred in total.  Defaults to no limit. (Strong bound)
    /// for_time : float, optional
    ///    Evolve until this much (physical) time has passed.  Defaults to no limit. (Strong bound)
    /// total_time : float, optional
    ///    Evolve until this much (physical) time has passed since the state creation.  
    ///    Defaults to no limit. (Strong bound)
    /// size_min : int, optional
    ///    Evolve until the system has this many, or fewer, tiles. Defaults to no limit. (Weak bound)
    /// size_max : int, optional
    ///    Evolve until the system has this many, or more, tiles. Defaults to no limit. (Weak bound)
    /// for_wall_time : float, optional
    ///    Evolve until this much (wall) time has passed.  Defaults to no limit. (Strong bound)
    /// require_strong_bound : bool, optional
    ///    If True (default), a ValueError will be raised unless at least one strong bound has been
    ///    set, ensuring that the simulation will eventually end.  If False, ensure only that some
    ///    weak bound has been set, which may result in an infinite simulation.
    ///
    /// Returns
    /// -------
    ///
    /// list[EvolveOutcome]
    ///   The stopping condition that caused each simulation to end.
    fn evolve_all(
        &mut self,
        for_events: Option<u64>,
        total_events: Option<u64>,
        for_time: Option<f64>,
        total_time: Option<f64>,
        size_min: Option<u32>,
        size_max: Option<u32>,
        for_wall_time: Option<f64>,
        require_strong_bound: Option<bool>,
        py: Python<'_>,
    ) -> PyResult<Vec<EvolveOutcome>> {
        let bounds = EvolveBounds {
            for_events,
            for_time,
            total_events,
            total_time,
            size_min,
            size_max,
            for_wall_time: for_wall_time.map(Duration::from_secs_f64),
        };

        let require_strong_bound = require_strong_bound.unwrap_or(false);

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

        let res = py.allow_threads(|| self.write().unwrap().evolve_all(bounds));

        res.into_iter()
            .map(|x| x.map_err(|y| PyValueError::new_err(y.to_string())))
            .collect()
    }

    #[pyo3(
        signature = (state_indices,
                    for_events=None,
                    total_events=None,
                    for_time=None,
                    total_time=None,
                    size_min=None,
                    size_max=None,
                    for_wall_time=None,
                    require_strong_bound=true)
    )]
    fn evolve_some(
        &mut self,
        state_indices: Vec<usize>,
        for_events: Option<u64>,
        total_events: Option<u64>,
        for_time: Option<f64>,
        total_time: Option<f64>,
        size_min: Option<u32>,
        size_max: Option<u32>,
        for_wall_time: Option<f64>,
        require_strong_bound: Option<bool>,
        py: Python<'_>,
    ) -> PyResult<Vec<EvolveOutcome>> {
        #![allow(clippy::too_many_arguments)]
        let bounds = EvolveBounds {
            for_events,
            for_time,
            total_events,
            total_time,
            size_min,
            size_max,
            for_wall_time: for_wall_time.map(Duration::from_secs_f64),
        };

        let require_strong_bound = require_strong_bound.unwrap_or(false);

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

        let res = py.allow_threads(|| {
            self.write()
                .unwrap()
                .evolve_some(&state_indices[..], bounds)
        });

        res.into_iter()
            .map(|x| x.map_err(|y| PyValueError::new_err(y.to_string())))
            .collect()
    }

    /// Returns the current canvas for state_index (default 0), as an array copy.
    ///
    /// Parameters
    /// ----------
    /// state_index : int, optional
    ///   The index of the state to return.  Defaults to 0.
    ///
    /// Returns
    /// -------
    ///
    /// numpy.ndarray[int]
    ///  The current canvas for the state, copied.
    fn canvas_copy<'py>(
        &self,
        state_index: Option<usize>,
        py: Python<'py>,
    ) -> PyResult<&'py PyArray2<rgrow::base::Tile>> {
        let state_index = self.check_state(state_index)?;
        Ok(self
            .read()?
            .state_ref(state_index)
            .raw_array()
            .to_pyarray(py))
    }

    /// Returns the current canvas for state_index (default 0), as a
    /// *direct* view of the state array.  This array will update as
    /// the simulation evolves.  It should not be modified, as modifications
    /// will not result in rate and other necessary updates.
    ///
    /// Using this may cause memory safety problems: it is 'unsafe'-labelled in Rust.
    /// Unless the state is deleted, the array should remain valid so long as the
    /// underlying Simulation has not been garbage-collected.
    ///
    /// Parameters
    /// ----------
    /// state_index : int, optional
    ///   The index of the state to return.  Defaults to 0.
    ///
    /// Returns
    /// -------
    ///
    /// numpy.ndarray[int]
    ///  The current canvas for the state.
    fn canvas_view<'py>(
        this: &'py PyCell<Self>,
        state_index: Option<usize>,
        _py: Python<'py>,
    ) -> PyResult<&'py PyArray2<rgrow::base::Tile>> {
        let sim = this.borrow();
        let state_index = sim.check_state(state_index)?;
        let sim = sim.read()?;

        let ra = sim.state_ref(state_index).raw_array();

        unsafe { Ok(PyArray2::borrow_from_array(&ra, this)) }
    }

    /// Returns the number of tiles in the state.
    fn state_ntiles(&self, state_index: Option<usize>) -> PyResult<u32> {
        let state_index = self.check_state(state_index)?;
        Ok(self.read()?.state_ref(state_index).ntiles())
    }

    /// Returns the amount of time simulated (in seconds) for the state.
    fn state_time(&self, state_index: Option<usize>) -> PyResult<f64> {
        let state_index = self.check_state(state_index)?;
        Ok(self.read()?.state_ref(state_index).time())
    }

    /// Returns the number of events simulated for the state.
    fn state_events(&self, state_index: Option<usize>) -> PyResult<u64> {
        let state_index = self.check_state(state_index)?;
        Ok(self.read()?.state_ref(state_index).total_events())
    }

    /// Add a new state to the simulation.
    fn add_state(&mut self) -> PyResult<usize> {
        self.write()?
            .add_state()
            .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))
    }

    /// Adds n new states to the simulation.
    fn add_n_states(&mut self, n: usize) -> PyResult<Vec<usize>> {
        self.write()?
            .add_n_states(n)
            .map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))
    }

    #[getter]
    fn get_tile_concs(&self) -> PyResult<Vec<f64>> {
        Ok(self.read()?.tile_concs())
    }

    #[getter]
    fn get_tile_stoics(&self) -> PyResult<Vec<f64>> {
        Ok(self.read()?.tile_stoics())
    }

    #[getter]
    fn get_tile_names(&self) -> PyResult<Vec<String>> {
        Ok(self.read()?.tile_names())
    }

    fn set_system_param(&self, name: &str, value: RustAny) {
        self.write()
            .unwrap()
            .set_system_param(name, value.0)
            .unwrap();
    }

    fn get_system_param(&self, name: &str, py: Python) -> PyResult<Py<PyAny>> {
        self.read()
            .unwrap()
            .get_system_param(name)
            .map(|x| RustAny(x).into_py(py))
            .map_err(|x| PyValueError::new_err(x.to_string()))
    }

    fn n_mismatches(&self, state_index: Option<usize>) -> PyResult<u64> {
        let state_index = self.check_state(state_index)?;
        Ok(self.read()?.n_mismatches(state_index) as u64)
    }

    fn mismatch_array<'p>(
        &self,
        state_index: Option<usize>,
        py: Python<'p>,
    ) -> PyResult<&'p PyArray2<usize>> {
        let state_index = self.check_state(state_index)?;
        let sim = self.read()?;
        let ra = sim.mismatch_array(state_index);
        Ok(PyArray2::from_owned_array(py, ra))
    }
}

struct RustAny(Box<dyn Any>);

impl FromPyObject<'_> for RustAny {
    fn extract(obj: &PyAny) -> PyResult<Self> {
        if let Ok(val) = obj.extract::<u64>() {
            Ok(RustAny(Box::new(val)))
        } else if let Ok(val) = obj.extract::<f64>() {
            Ok(RustAny(Box::new(val)))
        } else if let Ok(val) = obj.extract::<i64>() {
            Ok(RustAny(Box::new(val)))
        } else if let Ok(val) = obj.extract::<bool>() {
            Ok(RustAny(Box::new(val)))
        } else if let Ok(val) = obj.extract::<String>() {
            Ok(RustAny(Box::new(val)))
        } else if let Ok(val) = obj.extract::<(u64, u64)>() {
            Ok(RustAny(Box::new(val)))
        } else if let Ok(val) = obj.extract::<(usize, usize, Ident)>() {
            Ok(RustAny(Box::new(val)))
        } else if let Ok(val) = obj.extract::<Vec<(usize, usize, Ident)>>() {
            Ok(RustAny(Box::new(val)))
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "Cannot convert value {:?}",
                obj
            )))
        }
    }
}

impl IntoPy<PyObject> for RustAny {
    fn into_py(self, py: Python<'_>) -> PyObject {
        if let Some(val) = self.0.downcast_ref::<f64>() {
            val.into_py(py)
        } else if let Some(val) = self.0.downcast_ref::<u64>() {
            val.into_py(py)
        } else if let Some(val) = self.0.downcast_ref::<i64>() {
            val.into_py(py)
        } else if let Some(val) = self.0.downcast_ref::<bool>() {
            val.into_py(py)
        } else if let Some(val) = self.0.downcast_ref::<String>() {
            val.into_py(py)
        } else {
            panic!("Cannot convert Any to PyAny");
        }
    }
}

#[pyclass]
pub struct FFSResult(pub(crate) Arc<Box<dyn ffs::FFSResult>>);

#[pymethods]
impl FFSResult {
    /// Nucleation rate, in M/s.  Calculated from the forward probability vector,
    /// and dimerization rate.
    #[getter]
    fn get_nucleation_rate(&self) -> f64 {
        self.0.nucleation_rate()
    }

    #[getter]
    fn get_forward_vec(&self) -> Vec<f64> {
        self.0.forward_vec().clone()
    }

    #[getter]
    fn get_dimerization_rate(&self) -> f64 {
        self.0.dimerization_rate()
    }

    #[getter]
    fn get_surfaces(&self) -> Vec<FFSLevel> {
        self.0
            .surfaces()
            .iter()
            .enumerate()
            .map(|(i, _)| FFSLevel {
                res: self.0.clone(),
                level: i,
            })
            .collect()
    }

    fn __str__(&self) -> String {
        format!(
            "FFSResult({:1.4e} M/s, {:?})",
            self.0.nucleation_rate(),
            self.0.forward_vec()
        )
    }

    fn __repr__(&self) -> String {
        self.__str__()
    }

    #[getter]
    fn previous_indices(&self) -> Vec<Vec<usize>> {
        self.get_surfaces()
            .iter()
            .map(|x| x.get_previous_indices())
            .collect()
    }
}

#[pyclass]
pub struct FFSLevel {
    res: Arc<Box<dyn ffs::FFSResult>>,
    level: usize,
}

#[pymethods]
impl FFSLevel {
    #[getter]
    fn get_configs<'py>(&self, py: Python<'py>) -> Vec<&'py PyArray2<rgrow::base::Tile>> {
        self.res.surfaces()[self.level]
            .configs()
            .iter()
            .map(|x| x.to_pyarray(py))
            .collect()
    }

    #[getter]
    fn get_previous_indices(&self) -> Vec<usize> {
        self.res.surfaces()[self.level].previous_list()
    }
}
