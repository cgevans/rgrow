use pyo3::prelude::*;

#[pymodule]
#[pyo3(name = "rgrow")]
fn pyrgrow(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<rgrow::tileset::TileSet>()?;
    m.add_class::<rgrow::tileset::TileShape>()?;

    m.add_class::<rgrow::python::PySystem>()?;
    m.add_class::<rgrow::python::PyState>()?;

    m.add_class::<rgrow::ffs::FFSRunResult>()?;
    m.add_class::<rgrow::ffs::FFSLevelRef>()?;
    m.add_class::<rgrow::ffs::FFSStateRef>()?;

    m.add_class::<rgrow::ffs::FFSRunConfig>()?;
    m.add_class::<rgrow::system::EvolveBounds>()?;
    m.add_class::<rgrow::system::EvolveOutcome>()?;

    m.add_function(wrap_pyfunction!(rgrow::utils::string_dna_dg_ds, m)?)?;

    Ok(())
}
