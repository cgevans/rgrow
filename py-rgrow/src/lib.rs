use pyo3::types::PyType;
use numpy::IntoPyArray;
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rgrow::{State2DQT, StaticKTAM, StaticATAM, 
    StateEvolve, StateCreate, StateStatus, Energy, StateStep, Tile};


#[pymodule]
fn rgrow<'py>(_py: Python<'py>, m: &PyModule) -> PyResult<()> {

    #[pyclass]
    #[derive(Clone)]
    struct PyStaticKTAM {
        sys: StaticKTAM
    }

    #[pyclass]
    #[derive(Clone)]
    struct PyStaticATAM {
        sys: StaticATAM
    }

    #[pyclass]
    #[derive(Clone)]
    struct PyStateKTAM {
        state: State2DQT<StaticKTAM>
    }

    #[pyclass]
    #[derive(Clone)]
    struct PyStateATAM {
        state: State2DQT<StaticATAM>
    }

    #[pymethods]
    impl PyStaticKTAM {
        #[new]
        fn new(tile_concs: PyReadonlyArray1<f64>, tile_edges: PyReadonlyArray2<Tile>,
            glue_strengths: PyReadonlyArray1<Energy>, gse: Energy) -> PyResult<Self> {
            let sys = StaticKTAM::new(tile_concs.to_owned_array(),
                        tile_edges.to_owned_array(),
                        glue_strengths.to_owned_array(), gse);

            Ok(PyStaticKTAM { sys })
        }

        #[classmethod]
        fn from_raw(_cls: &PyType, tile_rates: PyReadonlyArray1<f64>, energy_ns: PyReadonlyArray2<Energy>,
            energy_we: PyReadonlyArray2<Energy>) -> Self {
                Self { sys: StaticKTAM::from_raw(
                    tile_rates.to_owned_array(),
                    energy_ns.to_owned_array(),
                    energy_we.to_owned_array())
                 }
            }
    }

    #[pymethods]
    impl PyStaticATAM {
        #[new]
        fn new(tile_concs: PyReadonlyArray1<f64>, tile_edges: PyReadonlyArray2<Tile>,
            glue_strengths: PyReadonlyArray1<Energy>, tau: Energy) -> PyResult<Self> {
            let sys = StaticATAM::new(tile_concs.to_owned_array(),
                        tile_edges.to_owned_array(),
                        glue_strengths.to_owned_array(), tau);

            Ok(PyStaticATAM { sys })
        }
    }

    #[pymethods]
    impl PyStateKTAM {
        #[new]
        fn new(canvas: PyReadonlyArray2<usize>, system: &mut PyStaticKTAM) -> PyResult<Self> {
            let state = State2DQT::<StaticKTAM>::create(&canvas.to_owned_array(), &mut system.sys);

            Ok(Self { state })
        }

        #[classmethod]
        fn create_we_pair(_cls: &PyType, system: &mut PyStaticKTAM, w: usize, e: usize, size: usize) -> PyResult<Self> {
            let state = State2DQT::<StaticKTAM>::create_we_pair(&mut system.sys, w, e, size);

            Ok(Self { state })
        }

        #[classmethod]
        fn create_ns_pair(_cls: &PyType, system: &mut PyStaticKTAM, n: usize, s: usize, size: usize) -> PyResult<Self> {
            let state = State2DQT::<StaticKTAM>::create_ns_pair(&mut system.sys, n, s, size);

            Ok(Self { state })
        }

        fn set_point(&mut self, system: &mut PyStaticKTAM, py: usize, px: usize, t: usize) -> PyResult<()> {
            self.state.set_point(&mut system.sys, (px, py), t);

            Ok(())
        }

        fn take_step(&mut self, system: &mut PyStaticKTAM) -> PyResult<()> {
            self.state.take_step(&mut system.sys);

            Ok(())
        }

        fn evolve_in_size_range(&mut self, system: &mut PyStaticKTAM, minsize: usize, maxsize: usize, maxevents: u64) -> PyResult<()> {
            self.state.evolve_in_size_range(&mut system.sys, minsize, maxsize, maxevents);

            Ok(())
        }

        fn to_array(&self, py: Python) -> Py<PyArray2<usize>> {
            self.state.canvas.canvas.to_owned().into_pyarray(py).into()
        }


        fn copy(&self) -> PyResult<Self> {
            Ok(Self { state: self.state.clone() })
        }

        fn ntiles(&self) -> usize {
            self.state.ntiles()
        }

        fn rates(&self, level: usize, py: Python) -> Py<PyArray2<f64>> {
            self.state.rates[level].to_owned().into_pyarray(py).into()
        }
    }

    #[pymethods]
    impl PyStateATAM {
        #[new]
        fn new(canvas: PyReadonlyArray2<usize>, system: &mut PyStaticATAM) -> PyResult<Self> {
            let state = State2DQT::<StaticATAM>::create(&canvas.to_owned_array(), &mut system.sys);

            Ok(Self { state })
        }

        fn take_step(&mut self, system: &mut PyStaticATAM) -> PyResult<()> {
            self.state.take_step(&mut system.sys);

            Ok(())
        }

        fn evolve_in_size_range(&mut self, system: &mut PyStaticATAM, minsize: usize, maxsize: usize, maxevents: u64) -> PyResult<()> {
            self.state.evolve_in_size_range(&mut system.sys, minsize, maxsize, maxevents);

            Ok(())
        }

        fn to_array(&self, py: Python) -> Py<PyArray2<usize>> {
            self.state.canvas.canvas.to_owned().into_pyarray(py).into()
        }

        fn ntiles(&self) -> usize {
            self.state.ntiles()
        }

        fn rates(&self, level: usize, py: Python) -> Py<PyArray2<f64>> {
            self.state.rates[level].to_owned().into_pyarray(py).into()
        }
    }
    m.add_class::<PyStaticATAM>()?;
    m.add_class::<PyStaticKTAM>()?;
    m.add_class::<PyStateKTAM>()?;
    m.add_class::<PyStateATAM>()?;

    Ok(())
}
