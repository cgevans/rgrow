#![feature(associated_type_bounds)]

use numpy::ToPyArray;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::ValueError;
use pyo3::types::PyType;
use pyo3::{prelude::*, wrap_pyfunction};
use rgrow as rg;
use rgrow::canvas::{Canvas, CanvasCreate};
use rgrow::ffs;
use rgrow::state::{StateCreate, StateEvolve, StateStatus, StateStep, StateUpdateSingle};
use rgrow::system::FissionHandling;

#[pymodule]
fn rgrow<'py>(_py: Python<'py>, m: &PyModule) -> PyResult<()> {
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
        inner: rg::StaticKTAM<rg::CanvasSquare>,
    }

    #[pyclass]
    #[derive(Debug)]
    #[text_signature = "(tile_concs, tile_edges, glue_strengths, gse, gmc, alpha, k_f, fission, tile_names)"]
    struct StaticKTAMPeriodic {
        inner: rg::StaticKTAM<rg::CanvasPeriodic>,
    }

    /// Static (no changes to concentrations, bond strengths, parameters, etc) aTAM system.
    #[pyclass]
    #[derive(Clone, Debug)]
    struct StaticATAM {
        inner: rg::StaticATAM,
    }

    /// A simulation state for a static aTAM simulation, using a square canvas.
    #[pyclass]
    #[derive(Clone, Debug)]
    struct StateATAM {
        inner: rg::QuadTreeState<rg::CanvasSquare, rg::StaticATAM, rg::NullStateTracker>,
    }

    #[pymethods]
    trait System {
        #[new]
        fn new(
            tile_concs: PyReadonlyArray1<f64>,
            tile_edges: PyReadonlyArray2<rg::Tile>,
            glue_strengths: PyReadonlyArray1<rg::Energy>,
            gse: rg::Energy,
            gmc: rg::Energy,
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

            let inner = rg::StaticKTAM::from_ktam(
                tile_concs.to_owned_array(),
                tile_edges.to_owned_array(),
                glue_strengths.to_owned_array(),
                gse,
                gmc,
                alpha,
                k_f,
                Some(rg::Seed::None()),
                fission_handling,
                None,
                None,
                tile_names,
                None,
            );

            Ok(Self {
                inner: Box::new(inner),
            })
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
            Ok(StaticKTAM {
                inner: tileset.into_static_seeded_ktam(),
            })
        }

        /// StaticKTAM.from_raw(tile_rates, energy_ns, energy_we, k_f, alpha, fission)
        #[text_signature = "(tile_rates, energy_ns, energy_we, k_f, alpha, fission)"]
        #[classmethod]
        fn from_raw(
            _cls: &PyType,
            tile_rates: PyReadonlyArray1<f64>,
            energy_ns: PyReadonlyArray2<rg::Energy>,
            energy_we: PyReadonlyArray2<rg::Energy>,
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
                inner: rg::StaticKTAM::from_raw(
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
    impl StaticATAM {
        #[new]
        fn new(
            tile_concs: PyReadonlyArray1<f64>,
            tile_edges: PyReadonlyArray2<rg::Tile>,
            glue_strengths: PyReadonlyArray1<rg::Energy>,
            tau: rg::Energy,
        ) -> PyResult<Self> {
            let inner = rg::StaticATAM::new(
                tile_concs.to_owned_array(),
                tile_edges.to_owned_array(),
                glue_strengths.to_owned_array(),
                tau,
                Some(rg::Seed::None()),
            );

            Ok(StaticATAM { inner })
        }
    }

    /// A simulation state for a static kTAM simulation, using a square canvas.
    /// Takes an initial canvas (ndarray of u32s, must be square with width 2^L)
    /// and a StaticKTAM system instance.
    #[pyclass]
    #[derive(Clone, Debug)]
    #[text_signature = "(canvas, system)"]
    struct StateKTAM {
        inner: rg::QuadTreeState<
            rg::CanvasSquare,
            rg::StaticKTAM<rg::CanvasSquare>,
            rg::NullStateTracker,
        >,
    }

    #[pymethods]
    impl StateKTAM {
        #[new]
        fn new(canvas: PyReadonlyArray2<rg::Tile>, system: &StaticKTAM) -> PyResult<Self> {
            let inner = rg::QuadTreeState::from_canvas(&system.inner, canvas.to_owned_array());

            Ok(Self { inner })
        }

        #[classmethod]
        /// Creates a simulation state with the West-East dimer of tile numbers w, e, centered in
        /// a canvas of size size (must be 2^L).
        #[text_signature = "(system, w, e, size)"]
        fn create_we_pair(
            _cls: &PyType,
            system: &StaticKTAM,
            w: rg::Tile,
            e: rg::Tile,
            size: usize,
        ) -> PyResult<Self> {
            let inner = rg::QuadTreeState::create_we_pair(&system.inner, w, e, size);

            Ok(Self { inner })
        }

        #[classmethod]
        #[text_signature = "(system, n, s, size)"]
        /// Creates a simulation state with the North-South dimer of tile numbers n, s, centered in
        /// a canvas of size size (must be 2^L).
        fn create_ns_pair(
            _cls: &PyType,
            system: &StaticKTAM,
            n: rg::Tile,
            s: rg::Tile,
            size: usize,
        ) -> PyResult<Self> {
            let inner = rg::QuadTreeState::create_ns_pair(&system.inner, n, s, size);

            Ok(Self { inner })
        }

        #[text_signature = "(system, point_y, point_x, tile)"]
        /// Sets the point (py, px) to a particular tile (or empty, with 0).
        fn set_point(
            &mut self,
            system: &StaticKTAM,
            py: usize,
            px: usize,
            t: rg::Tile,
        ) -> PyResult<()> {
            self.inner.set_point(&system.inner, (px, py), t);

            Ok(())
        }

        /// Tries to take a single step.  May fail if the canvas is empty, or there is no step possible.
        fn take_step(&mut self, system: &StaticKTAM) -> PyResult<()> {
            match self.inner.take_step(&system.inner) {
                Ok(_) => Ok(()),
                Err(_) => Err(ValueError::py_err("Step-taking failed")),
            }
        }

        fn evolve_in_size_range(
            &mut self,
            system: &StaticKTAM,
            minsize: rg::NumTiles,
            maxsize: rg::NumTiles,
            maxevents: u64,
        ) -> PyResult<()> {
            self.inner
                .evolve_in_size_range_events_max(&system.inner, minsize, maxsize, maxevents);

            Ok(())
        }

        fn to_array<'py>(&self, py: Python<'py>) -> &'py PyArray2<rg::Tile> {
            self.inner.canvas.raw_array().to_pyarray(py)
        }

        fn copy(&self) -> PyResult<Self> {
            Ok(Self {
                inner: self.inner.clone(),
            })
        }

        fn ntiles(&self) -> rg::NumTiles {
            self.inner.ntiles()
        }

        fn rates<'py>(&self, level: usize, py: Python<'py>) -> &'py PyArray2<f64> {
            self.inner.rates[level].to_pyarray(py)
        }
    }

    #[pymethods]
    impl StateATAM {
        #[new]
        fn new(canvas: PyReadonlyArray2<rg::Tile>, system: &mut StaticATAM) -> PyResult<Self> {
            let inner = rg::QuadTreeState::from_canvas(&system.inner, canvas.to_owned_array());
            Ok(Self { inner })
        }

        fn take_step(&mut self, system: &mut StaticATAM) -> PyResult<()> {
            self.inner.take_step(&system.inner).unwrap();
            Ok(())
        }

        fn evolve_in_size_range(
            &mut self,
            system: &mut StaticATAM,
            minsize: rg::NumTiles,
            maxsize: rg::NumTiles,
            maxevents: u64,
        ) -> PyResult<()> {
            self.inner
                .evolve_in_size_range_events_max(&system.inner, minsize, maxsize, maxevents);

            Ok(())
        }

        fn to_array<'py>(&self, py: Python<'py>) -> &'py PyArray2<rg::Tile> {
            self.inner.canvas.raw_array().to_pyarray(py)
        }

        fn ntiles(&self) -> rg::NumTiles {
            self.inner.ntiles()
        }

        fn rates<'py>(&self, level: usize, py: Python<'py>) -> &'py PyArray2<f64> {
            self.inner.rates[level].to_pyarray(py)
        }
    }
    m.add_class::<StaticATAM>()?;
    m.add_class::<StaticKTAM>()?;
    m.add_class::<System>()?;
    m.add_class::<StateKTAM>()?;
    m.add_class::<StateATAM>()?;

    #[pyfunction]
    #[text_signature = "(system, num_states, target_size, canvas_size, max_init_events, max_subseq_events, start_size, size_step)"]
    /// Runs Forward Flux Sampling, and returns a tuple of
    /// (nucleation_rate, dimerization_rate, forward_probs[2->3, 3->4, etc])
    fn ffs_run(
        system: &StaticKTAM,
        num_states: usize,
        target_size: rg::NumTiles,
        canvas_size: usize,
        max_init_events: u64,
        max_subseq_events: u64,
        start_size: rg::NumTiles,
        size_step: rg::NumTiles,
    ) -> (f64, f64, Vec<f64>) {
        let fr = ffs::FFSRun::create(
            &system.inner,
            num_states,
            target_size,
            canvas_size,
            max_init_events,
            max_subseq_events,
            start_size,
            size_step,
        );

        let ret = (fr.nucleation_rate(), fr.dimerization_rate, fr.forward_vec());

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
        target_size: rg::NumTiles,
        canvas_size: usize,
        max_init_events: u64,
        max_subseq_events: u64,
        start_size: rg::NumTiles,
        size_step: rg::NumTiles,
        py: Python<'py>,
    ) -> (
        f64,
        f64,
        Vec<f64>,
        Vec<Vec<usize>>,
        Vec<Vec<&'py PyArray2<rg::Tile>>>,
    ) {
        let fr = ffs::FFSRun::create(
            &system.inner,
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
            fr.forward_vec(),
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
        target_size: rg::NumTiles,
        canvas_size: usize,
        max_init_events: u64,
        max_subseq_events: u64,
        start_size: rg::NumTiles,
        size_step: rg::NumTiles,
        py: Python<'py>,
    ) -> (
        f64,
        f64,
        Vec<f64>,
        Vec<Vec<usize>>,
        Vec<&'py PyArray2<rg::Tile>>,
    ) {
        let fr = ffs::FFSRun::create(
            &system.inner,
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
            .last()
            .unwrap()
            .state_list
            .iter()
            .map(|state| state.canvas.raw_array().to_pyarray(py))
            .collect();

        let ret = (
            fr.nucleation_rate(),
            fr.dimerization_rate,
            fr.forward_vec(),
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
    fn ffs_run_final_p<'py>(
        system: &StaticKTAM,
        num_states: usize,
        target_size: rg::NumTiles,
        canvas_size: usize,
        max_init_events: u64,
        max_subseq_events: u64,
        start_size: rg::NumTiles,
        size_step: rg::NumTiles,
        py: Python<'py>,
    ) -> (
        f64,
        f64,
        Vec<f64>,
        Vec<Vec<usize>>,
        Vec<&'py PyArray2<rg::Tile>>,
    ) {
        let fr = ffs::FFSRun::create(
            &system.inner,
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
            .last()
            .unwrap()
            .state_list
            .iter()
            .map(|state| state.canvas.raw_array().to_pyarray(py))
            .collect();

        let ret = (
            fr.nucleation_rate(),
            fr.dimerization_rate,
            fr.forward_vec(),
            prevlist,
            assemblies,
        );

        drop(fr);

        ret
    }

    m.add_wrapped(wrap_pyfunction!(ffs_run))?;
    m.add_wrapped(wrap_pyfunction!(ffs_run_full))?;
    m.add_wrapped(wrap_pyfunction!(ffs_run_final))?;

    Ok(())
}
