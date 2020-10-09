#![feature(associated_type_bounds)]

use numpy::{ToPyArray};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use ndarray::Array2;
use pyo3::exceptions::ValueError;
use pyo3::types::PyType;
use pyo3::{prelude::*, wrap_pyfunction};
use rgrow::base;
use rgrow::canvas;
use rgrow::canvas::Canvas;
use rgrow::ffs;
use rand::{SeedableRng, rngs::SmallRng};
use rgrow::system::StepOutcome;

use rgrow::state;
use rgrow::system;
use rgrow::system::{System, SystemWithStateCreate};
use rgrow::state::{StateCreate, StateStatus};
use rgrow::system::FissionHandling;

use std::fmt::Debug;

#[pymodule]
fn rgrow<'py>(_py: Python<'py>, m: &PyModule) -> PyResult<()> {
    //? A (somewhat rudimentary and very unstable) Python interface to Rgrow.
    //?
    //? As static dispatch doesn't work with PyO3, this currently has separate types for
    //? combinations of different systems and states/canvases.

    /// Static (no changes to concentrations, bond strengths, parameters, etc) kTAM system.
    /// Currently implements fission, but not Xgrow's chunk fission, and not double tiles.
    /// Can be used by multiple simulations.
    /// Currently, Python can't handle seeds.
    ///
    /// Parameters:
    ///
    /// tile_concs: an array of f64s
    /// tile_edges: an array of u32s
    /// glue_strengths: an array of f64s
    /// gse, gmc: f64s
    /// alpha (optional, default 0.0)
    /// k_f (optional, default 1e6)
    /// fission (optional): one of "off", "just-detach", "on", "keep-largest", "keep-weighted"
    /// tile_names (optional): list of strings
    #[pyclass]
    #[derive(Debug)]
    #[text_signature = "(tile_concs, tile_edges, glue_strengths, gse, gmc, alpha, k_f, fission, tile_names)"]
    struct StaticKTAM {
        inner:
            system::StaticKTAM<state::QuadTreeState<canvas::CanvasSquare, state::NullStateTracker>>
    }

    #[pyclass]
    #[derive(Debug)]
    #[text_signature = "(tile_concs, tile_edges, glue_strengths, gse, gmc, alpha, k_f, fission, tile_names)"]
    struct StaticKTAMCover {
        inner:
            system::StaticKTAMCover<state::QuadTreeState<canvas::CanvasSquare, state::NullStateTracker>>
    }

    #[pyclass]
    #[derive(Debug)]
    #[text_signature = "(tile_concs, tile_edges, glue_strengths, gse, gmc, alpha, k_f, fission, tile_names)"]
    struct StaticKTAMPeriodic {
        inner: system::StaticKTAM<state::QuadTreeState<canvas::CanvasPeriodic, state::NullStateTracker>>
    }

    #[pymethods]
    impl StaticKTAMPeriodic {
        #[new]
        fn new(
            tile_concs: PyReadonlyArray1<f64>,
            tile_edges: PyReadonlyArray2<base::Tile>,
            glue_strengths: PyReadonlyArray1<base::Energy>,
            gse: base::Energy,
            gmc: base::Energy,
            alpha: Option<f64>,
            k_f: Option<f64>,
            fission: Option<&str>,
            tile_names: Option<Vec<String>>,
        ) -> PyResult<Self> {
            let fission_handling = match fission {
                Some(fs) => Some(match fs {
                    "off" => FissionHandling::NoFission,
                    "just-detach" => FissionHandling::JustDetach,
                    "on" => FissionHandling::KeepSeeded,
                    "keep-largest" => FissionHandling::KeepLargest,
                    "keep-weighted" => FissionHandling::KeepWeighted,
                    _ => return Err(ValueError::py_err("Invalid fission handling option")),
                }),
                None => None,
            };

            let inner = system::StaticKTAM::from_ktam(
                tile_concs.to_owned_array(),
                tile_edges.to_owned_array(),
                glue_strengths.to_owned_array(),
                gse,
                gmc,
                alpha,
                k_f,
                Some(system::Seed::None()),
                fission_handling,
                None,
                None,
                tile_names,
                None,
            );

            Ok(Self { inner })
        }

        #[classmethod]
        fn from_json(_cls: &PyType, json_data: &str) -> PyResult<Self> {
            let tileset = match rgrow::parser::TileSet::from_json(json_data) {
                Ok(t) => t,
                Err(e) => {
                    return Err(ValueError::py_err(format!(
                        "Couldn't parse tileset json: {:?}",
                        e
                    )))
                }
            };
            Ok(Self {
                inner: tileset.into_static_seeded_ktam_p(),
            })
        }

        /// StaticKTAM.from_raw(tile_rates, energy_ns, energy_we, k_f, alpha, fission)
        #[text_signature = "(tile_rates, energy_ns, energy_we, k_f, alpha, fission)"]
        #[classmethod]
        fn from_raw(
            _cls: &PyType,
            tile_rates: PyReadonlyArray1<f64>,
            energy_ns: PyReadonlyArray2<base::Energy>,
            energy_we: PyReadonlyArray2<base::Energy>,
            k_f: f64,
            alpha: f64,
            fission: Option<&str>,
        ) -> PyResult<Self> {
            let fission_handling = match fission {
                Some(fs) => Some(match fs {
                    "off" => FissionHandling::NoFission,
                    "just-detach" => FissionHandling::JustDetach,
                    "on" => FissionHandling::KeepSeeded,
                    "keep-largest" => FissionHandling::KeepLargest,
                    "keep-weighted" => FissionHandling::KeepWeighted,
                    _ => return Err(ValueError::py_err("Invalid fission handling option")),
                }),
                None => None,
            };

            Ok(Self {
                inner: system::StaticKTAM::from_raw(
                    tile_rates.to_owned_array(),
                    energy_ns.to_owned_array(),
                    energy_we.to_owned_array(),
                    k_f,
                    alpha,
                    fission_handling,
                ),
            })
        }

        #[getter]
        fn tile_rates<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
            self.inner.tile_adj_concs.to_pyarray(py)
        }

        #[getter]
        fn energy_ns<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
            self.inner.energy_ns.to_pyarray(py)
        }

        #[getter]
        fn energy_we<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
            self.inner.energy_we.to_pyarray(py)
        }
    }

    #[pymethods]
    impl StaticKTAMCover {
        #[classmethod]
        fn from_json(_cls: &PyType, json_data: &str) -> PyResult<Self> {
            let mut tileset = match rgrow::parser::TileSet::from_json(json_data) {
                Ok(t) => t,
                Err(e) => {
                    return Err(ValueError::py_err(format!(
                        "Couldn't parse tileset json: {:?}",
                        e
                    )))
                }
            };
            Ok(Self {
                inner: tileset.into_static_ktam_cover(),
            })
        }

        fn debug(&self) -> String {
            format!("{:?}", self.inner)
        }

        fn new_state(&mut self, size: usize) -> PyResult<StateKTAM> {
            let mut state = state::QuadTreeState::<_, state::NullStateTracker>::create_raw(
                Array2::zeros((size, size)),
            ).unwrap();
        
            let sl = self.inner.seed_locs();

            for (p, t) in sl {
                // FIXME: for large seeds,
                // this could be faster by doing raw writes, then update_entire_state
                // but we would need to distinguish sizing.
                // Or maybe there is fancier way with a set?
                self.inner.set_point(&mut state, p.0, t);
            }

            Ok(StateKTAM { inner: state })
        }

        #[getter]
        fn tile_rates<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
            self.inner.inner.tile_adj_concs.to_pyarray(py)
        }

        #[getter]
        fn energy_ns<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
            self.inner.inner.energy_ns.to_pyarray(py)
        }

        #[getter]
        fn energy_we<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
            self.inner.inner.energy_we.to_pyarray(py)
        }

        fn calc_mismatch_locations<'py>(&self, state: &StateKTAM, py: Python<'py>) -> &'py PyArray2<usize> {
            self.inner.calc_mismatch_locations(&state.inner).to_pyarray(py)
        }

        fn calc_mismatches(&self, state: &StateKTAM) -> u32 {
            self.inner.calc_mismatches(&state.inner)
        }

        fn evolve_in_size_range_events_max(&mut self, state: &mut StateKTAM, minsize: u32, maxsize: u32, maxevents: u64) {
            let mut rng = SmallRng::from_entropy();
            self.inner.evolve_in_size_range_events_max(&mut state.inner, minsize, maxsize, maxevents, &mut rng);
        }
    }

    

    #[pymethods]
    impl StaticKTAM {
        #[new]
        fn new(
            tile_concs: PyReadonlyArray1<f64>,
            tile_edges: PyReadonlyArray2<base::Tile>,
            glue_strengths: PyReadonlyArray1<base::Energy>,
            gse: base::Energy,
            gmc: base::Energy,
            alpha: Option<f64>,
            k_f: Option<f64>,
            fission: Option<&str>,
            tile_names: Option<Vec<String>>,
        ) -> PyResult<Self> {
            let fission_handling = match fission {
                Some(fs) => Some(match fs {
                    "off" => FissionHandling::NoFission,
                    "just-detach" => FissionHandling::JustDetach,
                    "on" => FissionHandling::KeepSeeded,
                    "keep-largest" => FissionHandling::KeepLargest,
                    "keep-weighted" => FissionHandling::KeepWeighted,
                    _ => return Err(ValueError::py_err("Invalid fission handling option")),
                }),
                None => None,
            };

            let inner = system::StaticKTAM::from_ktam(
                tile_concs.to_owned_array(),
                tile_edges.to_owned_array(),
                glue_strengths.to_owned_array(),
                gse,
                gmc,
                alpha,
                k_f,
                Some(system::Seed::None()),
                fission_handling,
                None,
                None,
                tile_names,
                None,
            );

            Ok(Self { inner })
        }

        #[classmethod]
        fn from_json(_cls: &PyType, json_data: &str) -> PyResult<Self> {
            let tileset = match rgrow::parser::TileSet::from_json(json_data) {
                Ok(t) => t,
                Err(e) => {
                    return Err(ValueError::py_err(format!(
                        "Couldn't parse tileset json: {:?}",
                        e
                    )))
                }
            };
            Ok(Self {
                inner: tileset.into_static_seeded_ktam(),
            })
        }

        /// StaticKTAM.from_raw(tile_rates, energy_ns, energy_we, k_f, alpha, fission)
        #[text_signature = "(tile_rates, energy_ns, energy_we, k_f, alpha, fission)"]
        #[classmethod]
        fn from_raw(
            _cls: &PyType,
            tile_rates: PyReadonlyArray1<f64>,
            energy_ns: PyReadonlyArray2<base::Energy>,
            energy_we: PyReadonlyArray2<base::Energy>,
            k_f: Option<f64>,
            alpha: Option<f64>,
            fission: Option<&str>,
        ) -> PyResult<Self> {
            let fission_handling = match fission {
                Some(fs) => Some(match fs {
                    "off" => FissionHandling::NoFission,
                    "just-detach" => FissionHandling::JustDetach,
                    "on" => FissionHandling::KeepSeeded,
                    "keep-largest" => FissionHandling::KeepLargest,
                    "keep-weighted" => FissionHandling::KeepWeighted,
                    _ => return Err(ValueError::py_err("Invalid fission handling option")),
                }),
                None => None,
            };

            Ok(Self {
                inner: system::StaticKTAM::from_raw(
                    tile_rates.to_owned_array(),
                    energy_ns.to_owned_array(),
                    energy_we.to_owned_array(),
                    k_f.unwrap_or(1e6),
                    alpha.unwrap_or(0.),
                    fission_handling,
                ),
            })
        }

        #[getter]
        fn tile_rates<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
            self.inner.tile_adj_concs.to_pyarray(py)
        }


        fn debug(&self) -> String {
            format!("{:?}", self.inner)
        }

        #[getter]
        fn energy_ns<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
            self.inner.energy_ns.to_pyarray(py)
        }

        #[getter]
        fn energy_we<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
            self.inner.energy_we.to_pyarray(py)
        }

        fn calc_mismatch_locations<'py>(&self, state: &StateKTAM, py: Python<'py>) -> &'py PyArray2<usize> {
            self.inner.calc_mismatch_locations(&state.inner).to_pyarray(py)
        }

        fn calc_mismatches(&self, state: &StateKTAM) -> u32 {
            self.inner.calc_mismatches(&state.inner)
        }

        fn evolve_in_size_range_events_max(&mut self, state: &mut StateKTAM, minsize: u32, maxsize: u32, maxevents: u64) {
            let mut rng = SmallRng::from_entropy();
            self.inner.evolve_in_size_range_events_max(&mut state.inner, minsize, maxsize, maxevents, &mut rng);
        }
    }


    /// A simulation state for a static kTAM simulation, using a square canvas.
    /// Takes an initial canvas (ndarray of u32s, must be square with width 2^L)
    /// and a StaticKTAM system instance.
    #[pyclass]
    #[derive(Clone, Debug)]
    #[text_signature = "(canvas, system)"]
    struct StateKTAM {
        inner: state::QuadTreeState<canvas::CanvasSquare, state::NullStateTracker>,
    }

    #[pymethods]
    impl StateKTAM {
        #[new]
        fn new(size: usize, system: &mut StaticKTAM) -> PyResult<Self> {
            let mut state = state::QuadTreeState::<_, state::NullStateTracker>::create_raw(
                Array2::zeros((size, size)),
            ).unwrap();
        
            let sl = system.inner.seed_locs();

            for (p, t) in sl {
                // FIXME: for large seeds,
                // this could be faster by doing raw writes, then update_entire_state
                // but we would need to distinguish sizing.
                // Or maybe there is fancier way with a set?
                system.inner.set_point(&mut state, p.0, t);
            }

            Ok(Self { inner: state })
        }

        fn debug(&self) -> String {
            format!("{:?}", self.inner)
        }

        #[classmethod]
        /// Creates a simulation state with the West-East dimer of tile numbers w, e, centered in
        /// a canvas of size size (must be 2^L).
        #[text_signature = "(system, w, e, size)"]
        fn create_we_pair(
            _cls: &PyType,
            system: &mut StaticKTAM,
            w: base::Tile,
            e: base::Tile,
            size: usize,
        ) -> PyResult<Self> {
            let inner = system.inner.create_we_pair(w, e, size).unwrap();

            Ok(Self { inner })
        }

        #[classmethod]
        #[text_signature = "(system, n, s, size)"]
        /// Creates a simulation state with the North-South dimer of tile numbers n, s, centered in
        /// a canvas of size size (must be 2^L).
        fn create_ns_pair(
            _cls: &PyType,
            system: &mut StaticKTAM,
            n: base::Tile,
            s: base::Tile,
            size: usize,
        ) -> PyResult<Self> {
            let inner = system.inner.create_ns_pair(n, s, size).unwrap();

            Ok(Self { inner })
       }

        #[text_signature = "(system, point_y, point_x, tile)"]
        /// Sets the point (py, px) to a particular tile (or empty, with 0).
        fn set_point(
            &mut self,
            system: &mut StaticKTAM,
            py: usize,
            px: usize,
            t: base::Tile,
        ) -> PyResult<()> {
            system.inner.set_point(&mut self.inner, (py, px), t);

            Ok(())
        }

        /// Tries to take a single step.  May fail if the canvas is empty, or there is no step possible.
        fn take_step(&mut self, system: &StaticKTAM) -> PyResult<()> {
            let mut rng = SmallRng::from_entropy();
            match system.inner.state_step(&mut self.inner, &mut rng, 1e100) {
                StepOutcome::HadEventAt(_) | StepOutcome::DeadEventAt(_) => {Ok(())},
                StepOutcome::NoEventIn(_) => {Err(ValueError::py_err("No event"))},
                StepOutcome::ZeroRate => {Err(ValueError::py_err("Zero rate"))}
            }
        }

        fn evolve_in_size_range(
            &mut self,
            system: &mut StaticKTAM,
            minsize: base::NumTiles,
            maxsize: base::NumTiles,
            maxevents: u64,
        ) -> PyResult<()> {
            let mut rng = SmallRng::from_entropy();
            system.inner
                .evolve_in_size_range_events_max(&mut self.inner, minsize, maxsize, maxevents, &mut rng);

            Ok(())
        }

        fn to_array<'py>(&self, py: Python<'py>) -> &'py PyArray2<base::Tile> {
            self.inner.canvas.raw_array().to_pyarray(py)
        }

        fn copy(&self) -> PyResult<Self> {
            Ok(Self {
                inner: self.inner.clone(),
            })
        }

        fn ntiles(&self) -> base::NumTiles {
            self.inner.ntiles()
        }

        fn time(&self) -> f64 {
            self.inner.time()
        }

        fn events(&self) -> u64 {
            self.inner.total_events()
        }

        fn rates<'py>(&self, level: usize, py: Python<'py>) -> &'py PyArray2<f64> {
            self.inner.rates.0[level].to_pyarray(py)
        }

    }
    m.add_class::<StaticKTAM>()?;
    m.add_class::<StaticKTAMCover>()?;
    m.add_class::<StaticKTAMPeriodic>()?;
    m.add_class::<StateKTAM>()?;

    #[pyfunction]
    #[text_signature = "(system, num_states, target_size, canvas_size, max_init_events, max_subseq_events, start_size, size_step)"]
    /// Runs Forward Flux Sampling, and returns a tuple of
    /// (nucleation_rate, dimerization_rate, forward_probs[2->3, 3->4, etc])
    fn ffs_run(
        system: &StaticKTAM,
        num_states: usize,
        target_size: base::NumTiles,
        canvas_size: usize,
        max_init_events: u64,
        max_subseq_events: u64,
        start_size: base::NumTiles,
        size_step: base::NumTiles,
    ) -> (f64, f64, Vec<f64>) {
        let fr = ffs::FFSRun::create(
            system.inner.to_owned(),
            num_states,
            target_size,
            canvas_size,
            max_init_events,
            max_subseq_events,
            start_size,
            size_step,
        );

        let ret = (fr.nucleation_rate(), fr.dimerization_rate, fr.forward_vec().clone());

        drop(fr);

        ret
    }

    #[pyfunction]
    #[text_signature = "(system, num_states, target_size, canvas_size, max_init_events, max_subseq_events, start_size, size_step)"]
    /// Runs Forward Flux Sampling, and returns a tuple of
    /// (nucleation_rate, dimerization_rate, forward_probs[2->3, 3->4, etc])
    fn ffs_run_full<'py>(
        system: &StaticKTAM,
        num_states: usize,
        target_size: base::NumTiles,
        canvas_size: usize,
        max_init_events: u64,
        max_subseq_events: u64,
        start_size: base::NumTiles,
        size_step: base::NumTiles,
        py: Python<'py>,
    ) -> (
        f64,
        f64,
        Vec<f64>,
        Vec<Vec<usize>>,
        Vec<Vec<&'py PyArray2<base::Tile>>>,
    ) {
        let fr = ffs::FFSRun::create(
            system.inner.to_owned(),
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
                    .map(|state| state.canvas.raw_array().to_pyarray(py))
                    .collect()
            })
            .collect();

        let ret = (
            fr.nucleation_rate(),
            fr.dimerization_rate,
            fr.forward_vec().clone(),
            prevlist,
            assemblies,
        );

        drop(fr);

        ret
    }

    #[pyfunction]
    #[text_signature = "(system, num_states, target_size, canvas_size, max_init_events, max_subseq_events, start_size, size_step)"]
    /// Runs Forward Flux Sampling, and returns a tuple of
    /// (nucleation_rate, dimerization_rate, forward_probs[2->3, 3->4, etc])
    fn ffs_run_final<'py>(
        system: &StaticKTAM,
        num_states: usize,
        target_size: base::NumTiles,
        canvas_size: usize,
        max_init_events: u64,
        max_subseq_events: u64,
        start_size: base::NumTiles,
        size_step: base::NumTiles,
        py: Python<'py>,
    ) -> (f64, f64, Vec<f64>, Vec<&'py PyArray2<base::Tile>>) {
        let fr = ffs::FFSRun::create_without_history(
            system.inner.to_owned(),
            num_states,
            target_size,
            canvas_size,
            max_init_events,
            max_subseq_events,
            start_size,
            size_step,
        );

        let assemblies = fr
            .level_list
            .last()
            .unwrap()
            .state_list
            .iter()
            .map(|state| state.canvas.raw_array().to_pyarray(py))
            .collect();

        let ret = (
            fr.nucleation_rate(),
            fr.dimerization_rate,
            fr.forward_vec().clone(),
            assemblies,
        );

        drop(fr);

        ret
    }

    #[pyfunction]
    #[text_signature = "(system, num_states, target_size, canvas_size, max_init_events, max_subseq_events, start_size, size_step)"]
    /// Runs Forward Flux Sampling, and returns a tuple of
    /// (nucleation_rate, dimerization_rate, forward_probs[2->3, 3->4, etc])
    fn ffs_run_final_cover<'py>(
        system: &StaticKTAMCover,
        num_states: usize,
        target_size: base::NumTiles,
        canvas_size: usize,
        max_init_events: u64,
        max_subseq_events: u64,
        start_size: base::NumTiles,
        size_step: base::NumTiles,
        py: Python<'py>,
    ) -> (f64, f64, Vec<f64>, Vec<&'py PyArray2<base::Tile>>) {
        let fr = ffs::FFSRun::create_without_history(
            system.inner.to_owned(),
            num_states,
            target_size,
            canvas_size,
            max_init_events,
            max_subseq_events,
            start_size,
            size_step,
        );

        let assemblies = fr
            .level_list
            .last()
            .unwrap()
            .state_list
            .iter()
            .map(|state| state.canvas.raw_array().to_pyarray(py))
            .collect();

        let ret = (
            fr.nucleation_rate(),
            fr.dimerization_rate,
            fr.forward_vec().clone(),
            assemblies,
        );

        drop(fr);

        ret
    }

    #[pyfunction]
    #[text_signature = "(system, num_states, target_size, canvas_size, max_init_events, max_subseq_events, start_size, size_step)"]
    /// Runs Forward Flux Sampling, and returns a tuple of
    /// (nucleation_rate, dimerization_rate, forward_probs[2->3, 3->4, etc])
    fn ffs_run_final_p<'py>(
        system: &StaticKTAMPeriodic,
        num_states: usize,
        target_size: base::NumTiles,
        canvas_size: usize,
        max_init_events: u64,
        max_subseq_events: u64,
        start_size: base::NumTiles,
        size_step: base::NumTiles,
        py: Python<'py>,
    ) -> (f64, f64, Vec<f64>, Vec<&'py PyArray2<base::Tile>>) {
        let fr = ffs::FFSRun::create_without_history(
            system.inner.to_owned(),
            num_states,
            target_size,
            canvas_size,
            max_init_events,
            max_subseq_events,
            start_size,
            size_step,
        );

        let assemblies = fr
            .level_list
            .last()
            .unwrap()
            .state_list
            .iter()
            .map(|state| state.canvas.raw_array().to_pyarray(py))
            .collect();

        let ret = (
            fr.nucleation_rate(),
            fr.dimerization_rate,
            fr.forward_vec().clone(),
            assemblies,
        );

        drop(fr);

        ret
    }

    m.add_wrapped(wrap_pyfunction!(ffs_run))?;
    m.add_wrapped(wrap_pyfunction!(ffs_run_full))?;
    m.add_wrapped(wrap_pyfunction!(ffs_run_final))?;
    m.add_wrapped(wrap_pyfunction!(ffs_run_final_p))?;
    m.add_wrapped(wrap_pyfunction!(ffs_run_final_cover))?;

    Ok(())
}
