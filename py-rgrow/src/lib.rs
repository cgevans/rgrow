mod tileset;
use pyo3::prelude::*;

#[pymodule]
#[pyo3(name = "rgrow")]
fn pyrgrow(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<tileset::TileSet>()?;
    m.add_class::<tileset::Tile>()?;
    m.add_class::<tileset::Simulation>()?;

    m.add_class::<tileset::FFSResult>()?;
    m.add_class::<tileset::FFSLevel>()?;

    m.add_class::<rgrow::ffs::FFSRunConfig>()?;
    m.add_class::<rgrow::system::EvolveBounds>()?;
    m.add_class::<rgrow::system::EvolveOutcome>()?;

    Ok(())
}
