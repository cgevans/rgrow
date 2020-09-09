use numpy::{IntoPyArray, ToPyArray};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::types::PyType;
use pyo3::{prelude::*, wrap_pyfunction};
use rgrow::{
    ffs, Energy, NullStateTracker, Seed, State2DQT, StateCreate, StateEvolve, StateStatus,
    StateStep, StateTracked, StateUpdateSingle, StaticATAM, StaticKTAM, Tile, TileSubsetTracker, NumTiles
};

#[pymodule]
fn rgrow<'py>(_py: Python<'py>, m: &PyModule) -> PyResult<()> {
    #[pyclass]
    #[derive(Clone, Debug)]
    struct PyStaticKTAM {
        sys: StaticKTAM,
    }

    #[pyclass]
    #[derive(Clone, Debug)]
    struct PyStaticATAM {
        sys: StaticATAM,
    }

    #[pyclass]
    #[derive(Clone, Debug)]
    struct PyStateKTAM {
        state: State2DQT<StaticKTAM, NullStateTracker>,
    }

    #[pyclass]
    #[derive(Clone, Debug)]
    struct PyStateKTAMSubTrack {
        state: State2DQT<StaticKTAM, TileSubsetTracker>,
    }

    #[pyclass]
    #[derive(Clone, Debug)]
    struct PyStateATAM {
        state: State2DQT<StaticATAM, NullStateTracker>,
    }

    #[pymethods]
    impl PyStaticKTAM {
        #[new]
        fn new(
            tile_concs: PyReadonlyArray1<f64>,
            tile_edges: PyReadonlyArray2<Tile>,
            glue_strengths: PyReadonlyArray1<Energy>,
            gse: Energy,
            gmc: Energy,
            alpha: Option<f64>,
            k_f: Option<f64>,
        ) -> PyResult<Self> {
            let sys = StaticKTAM::from_ktam(
                tile_concs.to_owned_array(),
                tile_edges.to_owned_array(),
                glue_strengths.to_owned_array(),
                gse,
                gmc,
                alpha,
                k_f,
                Some(Seed::None()),
            );

            Ok(PyStaticKTAM { sys })
        }

        /// PyStaticKTAM.from_raw(tile_rates, energy_ns, energy_we, k_f, alpha)
        #[classmethod]
        fn from_raw(
            _cls: &PyType,
            tile_rates: PyReadonlyArray1<f64>,
            energy_ns: PyReadonlyArray2<Energy>,
            energy_we: PyReadonlyArray2<Energy>,
            k_f: f64,
            alpha: f64,
        ) -> Self {
            Self {
                sys: StaticKTAM::from_raw(
                    tile_rates.to_owned_array(),
                    energy_ns.to_owned_array(),
                    energy_we.to_owned_array(),
                    k_f,
                    alpha,
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
        fn new(
            tile_concs: PyReadonlyArray1<f64>,
            tile_edges: PyReadonlyArray2<Tile>,
            glue_strengths: PyReadonlyArray1<Energy>,
            tau: Energy,
        ) -> PyResult<Self> {
            let sys = StaticATAM::new(
                tile_concs.to_owned_array(),
                tile_edges.to_owned_array(),
                glue_strengths.to_owned_array(),
                tau,
                Some(Seed::None()),
            );

            Ok(PyStaticATAM { sys })
        }
    }

    #[pymethods]
    impl PyStateKTAM {
        #[new]
        fn new(canvas: PyReadonlyArray2<Tile>, system: &PyStaticKTAM) -> PyResult<Self> {
            let state = State2DQT::from_canvas(&system.sys, canvas.to_owned_array());

            Ok(Self { state })
        }

        #[classmethod]
        fn create_we_pair(
            _cls: &PyType,
            system: &PyStaticKTAM,
            w: Tile,
            e: Tile,
            size: usize,
        ) -> PyResult<Self> {
            let state = State2DQT::create_we_pair(&system.sys, w, e, size);

            Ok(Self { state })
        }

        #[classmethod]
        fn create_ns_pair(
            _cls: &PyType,
            system: &PyStaticKTAM,
            n: Tile,
            s: Tile,
            size: usize,
        ) -> PyResult<Self> {
            let state = State2DQT::create_ns_pair(&system.sys, n, s, size);

            Ok(Self { state })
        }

        fn set_point(
            &mut self,
            system: &PyStaticKTAM,
            py: usize,
            px: usize,
            t: Tile,
        ) -> PyResult<()> {
            self.state.set_point(&system.sys, (px, py), t);

            Ok(())
        }

        fn take_step(&mut self, system: &PyStaticKTAM) -> PyResult<()> {
            self.state.take_step(&system.sys);

            Ok(())
        }

        fn evolve_in_size_range(
            &mut self,
            system: &PyStaticKTAM,
            minsize: NumTiles,
            maxsize: NumTiles,
            maxevents: u64,
        ) -> PyResult<()> {
            self.state
                .evolve_in_size_range_events_max(&system.sys, minsize, maxsize, maxevents);

            Ok(())
        }

        fn to_array<'py>(&self, py: Python<'py>) -> &'py PyArray2<Tile> {
            self.state.canvas.canvas.to_pyarray(py)
        }

        fn copy(&self) -> PyResult<Self> {
            Ok(Self {
                state: self.state.clone(),
            })
        }

        fn ntiles(&self) -> NumTiles {
            self.state.ntiles()
        }

        fn rates<'py>(&self, level: usize, py: Python<'py>) -> &'py PyArray2<f64> {
            self.state.rates[level].to_pyarray(py)
        }
    }

    #[pymethods]
    impl PyStateKTAMSubTrack {
        #[new]
        fn new(
            canvas: PyReadonlyArray2<Tile>,
            system: &PyStaticKTAM,
            subs: Vec<Tile>,
        ) -> PyResult<Self> {
            let mut state = State2DQT::from_canvas(&system.sys, canvas.to_owned_array());
            state.set_tracker(TileSubsetTracker::new(subs));

            Ok(Self { state })
        }

        #[classmethod]
        fn create_we_pair(
            _cls: &PyType,
            system: &PyStaticKTAM,
            w: Tile,
            e: Tile,
            size: usize,
            subs: Vec<Tile>,
        ) -> PyResult<Self> {
            let state = State2DQT::create_we_pair_with_tracker(
                &system.sys,
                w,
                e,
                size,
                TileSubsetTracker::new(subs),
            );

            Ok(Self { state })
        }

        #[classmethod]
        fn create_ns_pair(
            _cls: &PyType,
            system: &PyStaticKTAM,
            n: Tile,
            s: Tile,
            size: usize,
            subs: Vec<Tile>,
        ) -> PyResult<Self> {
            let state = State2DQT::create_ns_pair_with_tracker(
                &system.sys,
                n,
                s,
                size,
                TileSubsetTracker::new(subs),
            );

            Ok(Self { state })
        }

        fn set_point(
            &mut self,
            system: &PyStaticKTAM,
            py: usize,
            px: usize,
            t: Tile,
        ) -> PyResult<()> {
            self.state.set_point(&system.sys, (px, py), t);

            Ok(())
        }

        fn num_tracked(&self) -> NumTiles {
            self.state.tracker.num_in_subset
        }

        fn take_step(&mut self, system: &PyStaticKTAM) -> PyResult<()> {
            self.state.take_step(&system.sys);

            Ok(())
        }

        /// Evolve a system until the number of total tiles is >= maxsize or <= minsize.
        /// Panics if the number of events done in the function (not total events) goes above maxevents.
        fn evolve_in_size_range(
            &mut self,
            system: &PyStaticKTAM,
            minsize:  NumTiles,
            maxsize:  NumTiles,
            maxevents: u64,
        ) -> PyResult<()> {
            self.state
                .evolve_in_size_range_events_max(&system.sys, minsize, maxsize, maxevents);

            Ok(())
        }

        /// Evolve a system until the number of subset-specified tiles is >= maxsize or <= minsize.
        /// Panics if the number of events done in the function (not total events) goes above maxevents.
        fn evolve_in_subset_range(
            &mut self,
            system: &PyStaticKTAM,
            minsize: NumTiles,
            maxsize:  NumTiles,
            maxevents: u64,
        ) -> PyResult<()> {
            let condition = |s: &State2DQT<_, TileSubsetTracker>, events| {
                if events > maxevents {
                    panic!("Too many events!")
                };
                (s.tracker.num_in_subset <= minsize) | (s.tracker.num_in_subset >= maxsize)
            };

            self.state.evolve_until_condition(&system.sys, &condition);

            Ok(())
        }

        /// Evolve a system until the number of subset-specified tiles is >= maxsize or total_tiles is <= minsize.
        /// Panics if the number of events done in the function (not total events) goes above maxevents.
        fn evolve_in_subset_max_or_total_min(
            &mut self,
            system: &PyStaticKTAM,
            minsize: NumTiles,
            maxsize: NumTiles,
            maxevents: u64,
        ) -> PyResult<()> {
            let condition = |s: &State2DQT<_, TileSubsetTracker>, events| {
                if events > maxevents {
                    panic!("Too many events!")
                };
                println!("{:?} {:?}", s.tracker.num_in_subset, s.ntiles());
                (s.ntiles() <= minsize) | (s.tracker.num_in_subset >= maxsize)
            };

            self.state.evolve_until_condition(&system.sys, &condition);

            Ok(())
        }

        #[text_signature = "(system, minsize, maxsize, maxevents)"]
        ///
        /// Evolve a system until the number of subset-specified tiles is >= maxsize or total_tiles is <= minsize.
        /// Panics if the number of events done in the function (not total events) goes above maxevents.
        fn evolve_in_subset_max_or_total_min_nofail(
            &mut self,
            system: &PyStaticKTAM,
            minsize: NumTiles,
            maxsize: NumTiles,
            maxevents: u64,
        ) -> PyResult<()> {
            let condition = |s: &State2DQT<_, TileSubsetTracker>, events| {
                println!("{:?} {:?}", s.tracker.num_in_subset, s.ntiles());
                (events > maxevents)
                    | (s.ntiles() <= minsize)
                    | (s.tracker.num_in_subset >= maxsize)
            };

            self.state.evolve_until_condition(&system.sys, &condition);

            Ok(())
        }

        // fn quick_clear(&mut self, system: &PyStaticKTAM) -> PyResult<()> {
        //     Ok(())
        // }

        fn to_array<'py>(&self, py: Python<'py>) -> &'py PyArray2<Tile> {
            self.state.canvas.canvas.to_pyarray(py)
        }

        fn copy(&self) -> PyResult<Self> {
            Ok(Self {
                state: self.state.clone(),
            })
        }

        fn ntiles(&self) -> NumTiles {
            self.state.ntiles()
        }

        fn rates<'py>(&self, level: usize, py: Python<'py>) -> &'py PyArray2<f64> {
            self.state.rates[level].to_pyarray(py)
        }
    }

    #[pymethods]
    impl PyStateATAM {
        #[new]
        fn new(canvas: PyReadonlyArray2<Tile>, system: &mut PyStaticATAM) -> PyResult<Self> {
            let state = State2DQT::from_canvas(&system.sys, canvas.to_owned_array());
            Ok(Self { state })
        }

        fn take_step(&mut self, system: &mut PyStaticATAM) -> PyResult<()> {
            self.state.take_step(&system.sys).unwrap();
            Ok(())
        }

        fn evolve_in_size_range(
            &mut self,
            system: &mut PyStaticATAM,
            minsize: NumTiles,
            maxsize: NumTiles,
            maxevents: u64,
        ) -> PyResult<()> {
            self.state
                .evolve_in_size_range_events_max(&system.sys, minsize, maxsize, maxevents);

            Ok(())
        }

        fn to_array<'py>(&self, py: Python<'py>) -> &'py PyArray2<Tile> {
            self.state.canvas.canvas.to_pyarray(py)
        }

        fn ntiles(&self) -> NumTiles {
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

    #[pyfunction]
    #[text_signature = "(system, num_states, target_size, canvas_size, max_init_events, max_subseq_events)"]
    /// Runs Forward Flux Sampling, and returns a tuple of
    /// (nucleation_rate, dimerization_rate, forward_probs[2->3, 3->4, etc])
    fn ffs_run(
        system: &PyStaticKTAM,
        num_states: usize,
        target_size: NumTiles,
        canvas_size: usize,
        max_init_events: u64,
        max_subseq_events: u64,
        start_size: NumTiles,
        size_step: NumTiles,
    ) -> (f64, f64, Vec<f64>) {
        let fr = ffs::FFSRun::create(
            &system.sys,
            num_states,
            target_size,
            canvas_size,
            max_init_events,
            max_subseq_events,
            start_size,
            size_step,
        );

        (fr.nucleation_rate(), fr.dimerization_rate, fr.forward_vec())
    }

    #[pyfunction]
    #[text_signature = "(system, num_states, target_size, canvas_size, max_init_events, max_subseq_events)"]
    /// Runs Forward Flux Sampling, and returns a tuple of
    /// (nucleation_rate, dimerization_rate, forward_probs[2->3, 3->4, etc])
    fn ffs_run_full<'py>(
        system: &PyStaticKTAM,
        num_states: usize,
        target_size: NumTiles,
        canvas_size: usize,
        max_init_events: u64,
        max_subseq_events: u64,
        start_size: NumTiles,
        size_step: NumTiles,
        py: Python<'py>,
    ) -> (
        f64,
        f64,
        Vec<f64>,
        Vec<Vec<usize>>,
        Vec<Vec<&'py PyArray2<Tile>>>,
    ) {
        let fr = ffs::FFSRun::create(
            &system.sys,
            num_states,
            target_size,
            canvas_size,
            max_init_events,
            max_subseq_events,
            start_size,
            size_step,
        );

        let prevlist: Vec<_> = fr
            .level_list
            .iter()
            .map(|level| level.previous_list.clone())
            .collect();

        let assemblies = fr
            .level_list
            .iter()
            .map(|level| {
                level
                    .state_list
                    .iter()
                    .map(|state| state.canvas.canvas.to_pyarray(py))
                    .collect()
            })
            .collect();

        (
            fr.nucleation_rate(),
            fr.dimerization_rate,
            fr.forward_vec(),
            prevlist,
            assemblies,
        )
    }

    m.add_wrapped(wrap_pyfunction!(ffs_run))?;
    m.add_wrapped(wrap_pyfunction!(ffs_run_full))?;

    Ok(())
}
