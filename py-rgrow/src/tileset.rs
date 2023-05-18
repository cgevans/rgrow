use std::any::Any;
use std::collections::HashMap;
use std::fmt;
use std::fmt::Display;
use std::fmt::Formatter;

use std::sync::Arc;
use std::sync::RwLock;

use numpy::PyArray2;
use numpy::ToPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyDict;

use pyo3::{prelude::*, types::PyType};

use rgrow::base::GlueIdent;
use rgrow::base::Ident;
use rgrow::base::RustAny;
use rgrow::ffs;
use rgrow::ffs::FFSRunConfig;
use rgrow::simulation;
use rgrow::state::BoxedState;
use rgrow::system::BoxedSystem;

use rgrow::tileset;
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
        let edges: Vec<GlueIdent> = edges.into_iter().collect();
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

    fn create_system(&self) -> PyResult<BoxedSystem> {
        let sys = self.read()?.create_dynsystem()?;
        Ok(sys)
    }

    fn create_state(&self) -> PyResult<BoxedState> {
        let sys = self.read()?.create_dynsystem()?;
        let mut state = self.read()?.create_state()?;
        sys.setup_state(&mut *state)?;
        Ok(state.into())
    }

    /// Creates a simulation, and runs it in a UI.  Returns the :any:`Simulation` when
    /// finished.
    #[cfg(feature = "ui")]
    fn run_window(&self) -> PyResult<BoxedState> {
        let f = self.read()?;
        let s = f.run_window();

        let st =
            s.map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))?;
        Ok(st.into())
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

#[pyclass]
pub struct TileColorView(Arc<RwLock<dyn simulation::Simulation>>);

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
