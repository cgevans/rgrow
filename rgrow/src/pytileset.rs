use std::ops::Deref;

use numpy::{PyArray2, PyArrayMethods};
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyDict, PyType},
};

use crate::{
    base::GlueIdent,
    ffs::{FFSRunConfig, FFSRunResult},
    models::{atam::ATAM, ktam::KTAM, oldktam::OldKTAM},
    python::PyState,
    system::SystemEnum,
    tileset::{self, Bond, Tile, TileSet},
};

#[pymethods]
#[cfg(feature = "python")]
impl TileSet {
    #[new]
    #[pyo3(signature = (tiles, bonds=Vec::default(), glues=Vec::default(), **kwargs))]
    fn new(
        tiles: Vec<Tile>,
        bonds: Vec<Bond>,
        glues: Vec<(GlueIdent, GlueIdent, f64)>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<TileSet> {
        let mut tileset = TileSet {
            tiles,
            bonds,
            glues: glues
                .iter()
                .map(|x| (x.0.clone(), x.1.clone(), x.2))
                .collect(),
            ..Default::default()
        };
        if let Some(x) = kwargs {
            for (k, v) in x.iter() {
                let key = k.extract::<String>()?;
                match key.as_str() {
                    "Gse" | "gse" => tileset.gse = Some(v.extract()?),
                    "Gmc" | "gmc" => tileset.gmc = Some(v.extract()?),
                    "alpha" => tileset.alpha = Some(v.extract()?),
                    "threshold" => tileset.threshold = Some(v.extract()?),
                    "seed" => tileset.seed = Some(v.extract()?),
                    "size" => tileset.size = Some(v.extract()?),
                    "tau" => tileset.tau = Some(v.extract()?),
                    "smax" => tileset.smax = Some(v.extract()?),
                    "update_rate" => tileset.update_rate = Some(v.extract()?),
                    "k_f" | "kf" => tileset.kf = Some(v.extract()?),
                    "fission" => tileset.fission = Some(v.extract::<&str>()?.try_into()?),
                    "block" => tileset.block = Some(v.extract()?),
                    "chunk_handling" => {
                        tileset.chunk_handling = Some(v.extract::<&str>()?.try_into()?)
                    }
                    "chunk_size" => tileset.chunk_size = Some(v.extract::<&str>()?.try_into()?),
                    "canvas_type" => tileset.canvas_type = Some(v.extract::<&str>()?.try_into()?),
                    "tracking" => tileset.tracking = Some(v.extract::<&str>()?.try_into()?),
                    "hdoubletiles" => tileset.hdoubletiles = Some(v.extract()?),
                    "vdoubletiles" => tileset.vdoubletiles = Some(v.extract()?),
                    "model" => tileset.model = Some(v.extract::<&str>()?.try_into()?),
                    v => Python::attach(|py| {
                        let user_warning = py.get_type::<pyo3::exceptions::PyUserWarning>();
                        PyErr::warn(
                            py,
                            &user_warning,
                            std::ffi::CString::new(format!("Ignoring unknown key {v}."))
                                .unwrap()
                                .as_c_str(),
                            0,
                        )
                        .unwrap();
                    }),
                }
            }
        }
        Ok(tileset)
    }

    /// Parses a JSON string into a TileSet.
    #[pyo3(name = "from_json")]
    #[classmethod]
    fn py_from_json(_cls: &Bound<'_, PyType>, data: &str) -> PyResult<Self> {
        let tileset = tileset::TileSet::from_json(data);
        match tileset {
            Ok(tileset) => Ok(tileset),
            Err(err) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                err.to_string(),
            )),
        }
    }

    /// Creates a TileSet from a dict by exporting to json, then parsing the json.
    /// FIXME: implement this without the json trip.
    #[pyo3(name = "from_dict")]
    #[classmethod]
    fn py_from_dict(_cls: &Bound<'_, PyType>, data: &Bound<'_, PyAny>) -> PyResult<Self> {
        let json: String = Python::attach(|py| {
            let json = PyModule::import(py, "json")?;
            json.call_method1("dumps", (data,))?.extract::<String>()
        })?;

        let tileset = tileset::TileSet::from_json(&json);
        match tileset {
            Ok(tileset) => Ok(tileset),
            Err(err) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                err.to_string(),
            )),
        }
    }

    /// Parses a file (JSON, YAML, etc) into a TileSet
    #[pyo3(name = "from_file")]
    #[classmethod]
    fn py_from_file(_cls: &Bound<'_, PyType>, path: &str) -> PyResult<Self> {
        let ts = tileset::TileSet::from_file(path)
            .map_err(|err| PyErr::new::<PyValueError, _>(err.to_string()))?;
        Ok(ts)
    }

    #[pyo3(name = "create_system")]
    fn py_create_system(&self) -> PyResult<SystemEnum> {
        let sys = self.create_system()?;
        Ok(sys)
    }

    #[pyo3(name = "create_state", signature = (system=None))]
    fn py_create_state(&self, system: Option<&Bound<'_, PyAny>>) -> PyResult<PyState> {
        match system {
            None => Ok(PyState(
                self.create_state_with_system(&self.create_system()?)?,
            )),
            Some(x) => {
                if let Ok(sys) = x.downcast::<KTAM>() {
                    Ok(PyState(
                        self.create_state_with_system(sys.borrow().deref())?,
                    ))
                } else if let Ok(sys) = x.downcast::<ATAM>() {
                    Ok(PyState(
                        self.create_state_with_system(sys.borrow().deref())?,
                    ))
                } else if let Ok(sys) = x.downcast::<OldKTAM>() {
                    Ok(PyState(
                        self.create_state_with_system(sys.borrow().deref())?,
                    ))
                } else {
                    Err(PyErr::new::<PyValueError, _>("Invalid system type."))
                }
            }
        }
    }

    #[pyo3(name = "create_system_and_state")]
    fn py_create_system_and_state(&self) -> PyResult<(SystemEnum, PyState)> {
        let (sys, state) = self.create_system_and_state()?;
        Ok((sys, PyState(state)))
    }

    #[pyo3(name = "create_state_from_canvas")]
    fn py_create_state_from_canvas(&self, canvas: &Bound<'_, PyArray2<u32>>) -> PyResult<PyState> {
        let state = self.create_state_from_canvas(canvas.to_owned_array())?;
        Ok(PyState(state))
    }

    /// Creates a simulation, and runs it in a UI.  Returns the :any:`Simulation` when
    /// finished.
    #[pyo3(name = "run_window")]
    fn py_run_window(&self) -> PyResult<PyState> {
        use pyo3::PyErr;

        use crate::python::PyState;

        let s = self.run_window();

        let st =
            s.map_err(|err| PyErr::new::<pyo3::exceptions::PyValueError, _>(err.to_string()))?;
        Ok(PyState(st))
    }

    /// Runs FFS.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(name = "run_ffs", signature = (config = FFSRunConfig::default(), **kwargs))]
    fn py_run_ffs<'py>(
        &self,
        config: FFSRunConfig,
        kwargs: Option<&Bound<'py, PyDict>>,
        py: Python<'py>,
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
        self.to_string()
    }
}
