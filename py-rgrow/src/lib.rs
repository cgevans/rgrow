use pyo3::pymodule;

#[pymodule(gil_used = false)]
mod rgrow {
    use pyo3::prelude::*;
    use std::path::PathBuf;

    #[pymodule_export]
    use rgrow::tileset::TileSet;

    #[pymodule_export]
    use rgrow::python::PyState;

    #[pymodule_export]
    use rgrow::ffs::FFSLevelRef;
    #[pymodule_export]
    use rgrow::ffs::FFSRunResult;
    #[pymodule_export]
    use rgrow::ffs::FFSRunResultDF;
    #[pymodule_export]
    use rgrow::ffs::FFSStateRef;

    #[pymodule_export]
    use rgrow::ffs::FFSRunConfig;
    #[pymodule_export]
    use rgrow::system::CriticalStateConfig;
    #[pymodule_export]
    use rgrow::system::CriticalStateResult;
    #[pymodule_export]
    use rgrow::system::EvolveBounds;
    #[pymodule_export]
    use rgrow::system::EvolveOutcome;

    #[pymodule_export]
    use rgrow::colors::get_color;
    #[pymodule_export]
    use rgrow::models::atam::ATAM;
    #[pymodule_export]
    use rgrow::models::kblock::KBlock;
    #[pymodule_export]
    use rgrow::models::ktam::KTAM;
    #[pymodule_export]
    use rgrow::models::oldktam::OldKTAM;
    #[pymodule_export]
    use rgrow::models::sdc1d::AnnealProtocol;
    #[pymodule_export]
    use rgrow::models::sdc1d::SDC;
    #[pymodule_export]
    use rgrow::system::DimerInfo;
    #[pymodule_export]
    use rgrow::system::NeededUpdate;
    #[pymodule_export]
    use rgrow::utils::loop_penalty;
    #[pymodule_export]
    use rgrow::utils::string_dna_dg_ds;

    #[pymodule_export]
    use rgrow::colors::get_color_or_random;

    /// Get the package directory where rgrow is installed.
    /// This is useful for finding resources like the rgrow-gui binary.
    #[pyfunction]
    fn get_package_dir() -> PyResult<String> {
        Python::attach(|py| {
            let importlib = py.import("importlib.util")?;
            let spec = importlib.call_method1("find_spec", ("rgrow",))?;
            let origin = spec.getattr("origin")?;

            if origin.is_none() {
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "Could not find rgrow package location",
                ));
            }

            let origin_str = origin.extract::<String>()?;
            let path = PathBuf::from(origin_str);

            // Get the directory containing the .so file
            if let Some(parent) = path.parent() {
                Ok(parent.to_string_lossy().to_string())
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "Could not determine package directory",
                ))
            }
        })
    }
}
