use pyo3::types::PyType;
use numpy::{ToPyArray, IntoPyArray};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rgrow::{State2DQT, StaticKTAM, StaticATAM, 
    StateEvolve, StateCreate, StateStatus, Energy, StateStep, Tile, NullStateTracker, TileSubsetTracker,
StateUpdateSingle, StateTracked, Seed};


#[pymodule]
fn rgrow<'py>(_py: Python<'py>, m: &PyModule) -> PyResult<()> {

    #[pyclass]
    #[derive(Clone, Debug)]
    struct PyStaticKTAM {
        sys: StaticKTAM
    }

    #[pyclass]
    #[derive(Clone, Debug)]
    struct PyStaticATAM {
        sys: StaticATAM
    }

    #[pyclass]
    #[derive(Clone, Debug)]
    struct PyStateKTAM {
        state: State2DQT<StaticKTAM, NullStateTracker>
    }

    #[pyclass]
    #[derive(Clone, Debug)]
    struct PyStateKTAMSubTrack {    
        state: State2DQT<StaticKTAM, TileSubsetTracker>
    }

    #[pyclass]
    #[derive(Clone, Debug)]
    struct PyStateATAM {
        state: State2DQT<StaticATAM, NullStateTracker>
    }

    #[pymethods]
    impl PyStaticKTAM {
        #[new]
        fn new(tile_concs: PyReadonlyArray1<f64>, tile_edges: PyReadonlyArray2<Tile>,
            glue_strengths: PyReadonlyArray1<Energy>, gse: Energy, gmc: Energy,
            alpha: Option<f64>, k_f: Option<f64>) -> PyResult<Self> {
            let sys = StaticKTAM::from_ktam(tile_concs.to_owned_array(),
                        tile_edges.to_owned_array(),
                        glue_strengths.to_owned_array(), gse, gmc, 
                    alpha, k_f, Some(Seed::None()));

            Ok(PyStaticKTAM { sys })
        }

        /// PyStaticKTAM.from_raw(tile_rates, energy_ns, energy_we)
        #[classmethod]
        fn from_raw(_cls: &PyType, tile_rates: PyReadonlyArray1<f64>, energy_ns: PyReadonlyArray2<Energy>,
            energy_we: PyReadonlyArray2<Energy>) -> Self {
                Self { sys: StaticKTAM::from_raw(
                    tile_rates.to_owned_array(),
                    energy_ns.to_owned_array(),
                    energy_we.to_owned_array(), 1e6, 0.
                ),
                 }
            }

        #[getter]
        fn tile_rates<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
            self.sys.tile_adj_concs.to_pyarray(py)
        }

        #[getter]
        fn energy_ns<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
            self.sys.energy_ns.to_pyarray(py)
        }

        #[getter]
        fn energy_we<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
            self.sys.energy_we.to_pyarray(py)
        }
    }

    #[pymethods]
    impl PyStaticATAM {
        #[new]
        fn new(tile_concs: PyReadonlyArray1<f64>, tile_edges: PyReadonlyArray2<Tile>,
            glue_strengths: PyReadonlyArray1<Energy>, tau: Energy) -> PyResult<Self> {
            let sys = StaticATAM::new(tile_concs.to_owned_array(),
                        tile_edges.to_owned_array(),
                        glue_strengths.to_owned_array(), tau, Some(Seed::None()));

            Ok(PyStaticATAM { sys })
        }
    }

    #[pymethods]
    impl PyStateKTAM {
        #[new]
        fn new(canvas: PyReadonlyArray2<usize>, system: &PyStaticKTAM) -> PyResult<Self> {
            let state = State2DQT::from_canvas(&system.sys, canvas.to_owned_array());

            Ok(Self { state })
        }

        #[classmethod]
        fn create_we_pair(_cls: &PyType, system: &PyStaticKTAM, w: usize, e: usize, size: usize) -> PyResult<Self> {
            let state = State2DQT::create_we_pair(&system.sys, w, e, size);

            Ok(Self { state })
        }

        #[classmethod]
        fn create_ns_pair(_cls: &PyType, system: &PyStaticKTAM, n: usize, s: usize, size: usize) -> PyResult<Self> {
            let state = State2DQT::create_ns_pair(&system.sys, n, s, size);

            Ok(Self { state })
        }

        fn set_point(&mut self, system: &PyStaticKTAM, py: usize, px: usize, t: usize) -> PyResult<()> {
            self.state.set_point(&system.sys, (px, py), t);

            Ok(())
        }

        fn take_step(&mut self, system: &PyStaticKTAM) -> PyResult<()> {
            self.state.take_step(&system.sys);

            Ok(())
        }

        fn evolve_in_size_range(&mut self, system: &PyStaticKTAM, minsize: usize, maxsize: usize, maxevents: u64) -> PyResult<()> {
            self.state.evolve_in_size_range(&system.sys, minsize, maxsize, maxevents);

            Ok(())
        }

        fn to_array<'py>(&self, py: Python<'py>) -> &'py PyArray2<usize> {
            self.state.canvas.canvas.to_pyarray(py)
        }


        fn copy(&self) -> PyResult<Self> {
            Ok(Self { state: self.state.clone() })
        }

        fn ntiles(&self) -> usize {
            self.state.ntiles()
        }

        fn rates<'py>(&self, level: usize, py: Python<'py>) -> &'py PyArray2<f64> {
            self.state.rates[level].to_pyarray(py)
        }
    }

    #[pymethods]
    impl PyStateKTAMSubTrack {
        #[new]
        fn new(canvas: PyReadonlyArray2<usize>, system: &PyStaticKTAM, subs: Vec<usize>) -> PyResult<Self> {
            let mut state = State2DQT::from_canvas(&system.sys, canvas.to_owned_array());
            state.set_tracker(TileSubsetTracker::new(subs));

            Ok(Self { state })
        }

        #[classmethod]
        fn create_we_pair(_cls: &PyType, system: &PyStaticKTAM, w: usize, e: usize, size: usize, subs: Vec<usize>) -> PyResult<Self> {
            let state = State2DQT::create_we_pair_with_tracker(&system.sys, w, e, size, TileSubsetTracker::new(subs));

            Ok(Self { state })
        }

        #[classmethod]
        fn create_ns_pair(_cls: &PyType, system: &PyStaticKTAM, n: usize, s: usize, size: usize, subs: Vec<usize>) -> PyResult<Self> {
            let state = State2DQT::create_ns_pair_with_tracker(&system.sys, n, s, size, TileSubsetTracker::new(subs));

            Ok(Self { state })
        }

        fn set_point(&mut self, system: &PyStaticKTAM, py: usize, px: usize, t: usize) -> PyResult<()> {
            self.state.set_point(&system.sys, (px, py), t);

            Ok(())
        }

        fn num_tracked(&self) -> usize {
            self.state.tracker.num_in_subset
        }

        fn take_step(&mut self, system: &PyStaticKTAM) -> PyResult<()> {
            self.state.take_step(&system.sys);

            Ok(())
        }

        /// Evolve a system until the number of total tiles is >= maxsize or <= minsize.
        /// Panics if the number of events done in the function (not total events) goes above maxevents.
        fn evolve_in_size_range(&mut self, system: &PyStaticKTAM, minsize: usize, maxsize: usize, maxevents: u64) -> PyResult<()> {
            self.state.evolve_in_size_range(&system.sys, minsize, maxsize, maxevents);

            Ok(())
        }

        /// Evolve a system until the number of subset-specified tiles is >= maxsize or <= minsize.
        /// Panics if the number of events done in the function (not total events) goes above maxevents.
        fn evolve_in_subset_range(&mut self, system: &PyStaticKTAM, minsize: usize, maxsize: usize, maxevents: u64) -> PyResult<()> {
            let condition = |s:&State2DQT<_, TileSubsetTracker>, events| {
                if events > maxevents { panic!("Too many events!") };
                (s.tracker.num_in_subset <= minsize) | (s.tracker.num_in_subset >= maxsize)
            };


            self.state.evolve_until_condition(&system.sys, &condition);

            Ok(())
        }

        /// Evolve a system until the number of subset-specified tiles is >= maxsize or total_tiles is <= minsize.
        /// Panics if the number of events done in the function (not total events) goes above maxevents.
        fn evolve_in_subset_max_or_total_min(&mut self, system: &PyStaticKTAM, minsize: usize, maxsize: usize, maxevents: u64) -> PyResult<()> {
            let condition = |s:&State2DQT<_, TileSubsetTracker>, events| {
                if events > maxevents { panic!("Too many events!") };
                println!("{:?} {:?}", s.tracker.num_in_subset, s.ntiles());
                (s.ntiles() <= minsize) | (s.tracker.num_in_subset >= maxsize)
            };


            self.state.evolve_until_condition(&system.sys, &condition);

            Ok(())
        }

        #[text_signature="(system, minsize, maxsize, maxevents)"]
        ///
        /// Evolve a system until the number of subset-specified tiles is >= maxsize or total_tiles is <= minsize.
        /// Panics if the number of events done in the function (not total events) goes above maxevents.
        fn evolve_in_subset_max_or_total_min_nofail(&mut self, system: &PyStaticKTAM, minsize: usize, maxsize: usize, maxevents: u64) -> PyResult<()> {
            let condition = |s:&State2DQT<_, TileSubsetTracker>, events| {
                println!("{:?} {:?}", s.tracker.num_in_subset, s.ntiles());
                (events > maxevents) |
                (s.ntiles() <= minsize) | (s.tracker.num_in_subset >= maxsize)
            };


            self.state.evolve_until_condition(&system.sys, &condition);

            Ok(())
        }

        // fn quick_clear(&mut self, system: &PyStaticKTAM) -> PyResult<()> {
        //     Ok(())
        // }

        fn to_array<'py>(&self, py: Python<'py>) -> &'py PyArray2<usize> {
            self.state.canvas.canvas.to_pyarray(py)
        }


        fn copy(&self) -> PyResult<Self> {
            Ok(Self { state: self.state.clone() })
        }

        fn ntiles(&self) -> usize {
            self.state.ntiles()
        }

        fn rates<'py>(&self, level: usize, py: Python<'py>) -> &'py PyArray2<f64> {
            self.state.rates[level].to_pyarray(py)
        }
    }

    #[pymethods]
    impl PyStateATAM {
        #[new]
        fn new(canvas: PyReadonlyArray2<usize>, system: &mut PyStaticATAM) -> PyResult<Self> {
            let state = State2DQT::from_canvas(&system.sys, canvas.to_owned_array());
            Ok(Self { state })
        }

        fn take_step(&mut self, system: &mut PyStaticATAM) -> PyResult<()> {
            self.state.take_step(&system.sys);

            Ok(())
        }

        fn evolve_in_size_range(&mut self, system: &mut PyStaticATAM, minsize: usize, maxsize: usize, maxevents: u64) -> PyResult<()> {
            self.state.evolve_in_size_range(&system.sys, minsize, maxsize, maxevents);

            Ok(())
        }

        fn to_array<'py>(&self, py: Python<'py>) -> &'py PyArray2<usize> {
            self.state.canvas.canvas.to_pyarray(py)
        }

        fn ntiles(&self) -> usize {
            self.state.ntiles()
        }

        fn rates<'py>(&self, level: usize, py: Python<'py>) -> &'py PyArray2<f64> {
            self.state.rates[level].to_pyarray(py)
        }
    }
    m.add_class::<PyStaticATAM>()?;
    m.add_class::<PyStaticKTAM>()?;
    m.add_class::<PyStateKTAM>()?;
    m.add_class::<PyStateKTAMSubTrack>()?;
    m.add_class::<PyStateATAM>()?;

    Ok(())
}
