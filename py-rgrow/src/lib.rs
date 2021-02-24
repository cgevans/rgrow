use ndarray::Array2;
use numpy::ToPyArray;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::types::PyType;
use pyo3::{prelude::*, wrap_pyfunction};
use rand::{rngs::SmallRng, SeedableRng};
use rgrow::base;
use rgrow::canvas;
use rgrow::canvas::Canvas;
use rgrow::ffs;
use rgrow::system::StepOutcome;

use rgrow::state;
use rgrow::state::{StateCreate, StateStatus};
use rgrow::system;
use rgrow::system::{ChunkHandling, ChunkSize, FissionHandling};
use rgrow::system::{System, SystemWithStateCreate};

use core::f64;
use std::fmt::Debug;

/// A (somewhat rudimentary and very unstable) Python interface to Rgrow.
///
/// As static dispatch doesn't work with PyO3, this currently has separate types for
/// combinations of different systems and states/canvases.
///
/// The most important classes are:
///
/// - StaticKTAM (and the StaticKTAMPeriodic variant), which specify the kinetic model to use and its parameters.
/// - StateKTAM, which stores the state of a single assembly.
#[pymodule]
fn rgrow<'py>(_py: Python<'py>, m: &PyModule) -> PyResult<()> {
    /// Static (no changes to concentrations, bond strengths, parameters, etc) kTAM system.
    /// Currently implements fission, and dimer chunk detachment.
    ///
    /// Can be used for multiple simulations / states.
    ///
    /// Currently, Python can't handle seeds.
    ///
    /// Parameters:
    ///     tile_stoics (f64 N 1D array): tile stoichiometry (for N-1 tiles).  1.0 means tile concentration :math:`e^{-G_{mc}+\alpha}`.
    ///                                   "Tile 0" must correspond with an empty space, and should have stoic 0.
    ///     tile_edges (u32 Nx4 array): tile edges for each tile.  First row should be [0, 0, 0, 0].
    ///     glue_strengths (f64 G 1D array): glue strengths for each glue. 0 should have strength 0 and corresponds with a null glue.
    ///                                      Strength 1.0 corresponds to a glue of strength :math:`G_{se}`.
    ///     gse, gmc (f64): :math:`G_{mc}` and :math:`G_{se}`.
    ///     alpha (optional, float): Non-edge-dependent attachment strength adjustment (default 0.0).
    ///     k_f (optional, float): Non-adjusted forward rate constant, not accounting for :math:`\alpha` (default 1e6).
    ///     fission (optional): one of "off", "just-detach", "on", "keep-largest", "keep-weighted" (default "off").
    ///     chunk_handling (optional, str): one of "off"/"none" or "detach" (default "off")
    ///     chunk_size (optional, str): currently, must be "dimer" if chunk_handling is set to "detach".  Can also be set to "off"/"none" (the default).
    ///     tile_names (optional, list[str]): list of tile names (default None).
    #[pyclass]
    #[derive(Debug)]
    #[text_signature = "(tile_stoics, tile_edges, glue_strengths, gse, gmc, alpha, k_f, fission, chunk_handling, chunk_size, tile_names)"]
    struct StaticKTAM {
        inner:
            system::StaticKTAM<state::QuadTreeState<canvas::CanvasSquare, state::NullStateTracker>>,
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
            chunk_handling: Option<&str>,
            chunk_size: Option<&str>,
            tile_names: Option<Vec<String>>,
        ) -> PyResult<Self> {
            let fission_handling = match fission {
                Some(fs) => Some(match fs {
                    "off" => FissionHandling::NoFission,
                    "just-detach" => FissionHandling::JustDetach,
                    "on" => FissionHandling::KeepSeeded,
                    "keep-largest" => FissionHandling::KeepLargest,
                    "keep-weighted" => FissionHandling::KeepWeighted,
                    _ => return Err(PyValueError::new_err("Invalid fission handling option")),
                }),
                None => None,
            };

            let chunk_handling = Some(match chunk_handling {
                Some(ch) => match ch {
                    "off" | "none" => ChunkHandling::None,
                    "detach" => ChunkHandling::Detach,
                    _ => return Err(PyValueError::new_err("Invalid chunk handling option")),
                },
                None => ChunkHandling::None,
            });

            let chunk_size = match chunk_size {
                Some(cs) => match cs {
                    "dimer" => Some(ChunkSize::Dimer),
                    _ => return Err(PyValueError::new_err("Invalid chunk size option")),
                },
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
                chunk_handling,
                chunk_size,
                tile_names,
                None,
            );

            Ok(Self { inner })
        }

        #[classmethod]
        /// Creates a StaticKTAM instance from a JSON string.  Ignores canvas choice in JSON.
        #[text_signature = "(self, json_data)"]
        fn from_json(_cls: &PyType, json_data: &str) -> PyResult<Self> {
            let tileset = match rgrow::parser::TileSet::from_json(json_data) {
                Ok(t) => t,
                Err(e) => {
                    return Err(PyValueError::new_err(format!(
                        "Couldn't parse tileset json: {:?}",
                        e
                    )))
                }
            };
            Ok(Self {
                inner: tileset.into_static_seeded_ktam(),
            })
        }

        /// Generates a StaticKTAM instance from "raw" inputs, similar to what the model uses internally.
        ///
        /// Parameters:
        ///     tile_adj_rates (float array, shape N): the "adjusted unitless attachment rate" for each tile (N-1 tiles, 0 is empty).  This corresponds to
        ///                     :math:`e^{-G_{mc}}` for a tile with :math:`G_{mc}`, ie, it does not account for :math:`k_f` or
        ///                     :math:`\alpha`.
        ///     energy_ns (float array, shape NxN): in position [i,j], the bond strength that results from tile i being north of tile j.
        ///     energy_ws (float array, shape NxN): same, now with i the west tile, and j the east tile.
        ///     k_f (float, optional): :math:`k_f`, default 1e6.
        ///     alpha (float, optional): :math:`\alpha`, default 0.0.
        ///     fission (optional): one of "off", "just-detach", "on", "keep-largest", "keep-weighted" (default "off").
        ///     chunk_handling (optional, str): one of "off"/"none" or "detach" (default "off")
        ///     chunk_size (optional, str): currently, must be "dimer" if chunk_handling is set to "detach".  Can also be set to "off"/"none" (the default).
        #[text_signature = "(tile_adj_rates, energy_ns, energy_we, k_f, alpha, fission)"]
        #[classmethod]
        fn from_raw(
            _cls: &PyType,
            tile_adj_rates: PyReadonlyArray1<f64>,
            energy_ns: PyReadonlyArray2<base::Energy>,
            energy_we: PyReadonlyArray2<base::Energy>,
            k_f: Option<f64>,
            alpha: Option<f64>,
            fission: Option<&str>,
            chunk_handling: Option<&str>,
            chunk_size: Option<&str>,
        ) -> PyResult<Self> {
            let fission_handling = match fission {
                Some(fs) => Some(match fs {
                    "off" => FissionHandling::NoFission,
                    "just-detach" => FissionHandling::JustDetach,
                    "on" => FissionHandling::KeepSeeded,
                    "keep-largest" => FissionHandling::KeepLargest,
                    "keep-weighted" => FissionHandling::KeepWeighted,
                    _ => return Err(PyValueError::new_err("Invalid fission handling option")),
                }),
                None => None,
            };

            let chunk_handling = Some(match chunk_handling {
                Some(ch) => match ch {
                    "off" | "none" => ChunkHandling::None,
                    "detach" => ChunkHandling::Detach,
                    _ => return Err(PyValueError::new_err("Invalid chunk handling option")),
                },
                None => ChunkHandling::None,
            });

            let chunk_size = match chunk_size {
                Some(cs) => match cs {
                    "dimer" => Some(ChunkSize::Dimer),
                    _ => return Err(PyValueError::new_err("Invalid chunk size option")),
                },
                None => None,
            };

            Ok(Self {
                inner: system::StaticKTAM::from_raw(
                    tile_adj_rates.to_owned_array(),
                    energy_ns.to_owned_array(),
                    energy_we.to_owned_array(),
                    k_f.unwrap_or(1e6),
                    alpha.unwrap_or(0.),
                    fission_handling,
                    chunk_handling,
                    chunk_size,
                ),
            })
        }

        #[getter]
        fn tile_rates<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
            self.inner.tile_adj_concs.to_pyarray(py)
        }

        /// Debug info for model.
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

        fn calc_mismatch_locations<'py>(
            &self,
            state: &StateKTAM,
            py: Python<'py>,
        ) -> &'py PyArray2<usize> {
            self.inner
                .calc_mismatch_locations(&state.inner)
                .to_pyarray(py)
        }

        fn calc_mismatches(&self, state: &StateKTAM) -> u32 {
            self.inner.calc_mismatches(&state.inner)
        }

        /// StaticKTAM.evolve_in_size_range_events_max(self, state, minsize, maxsize, maxevents)
        ///
        /// A System-centric evolve method.  Evolves the provided state until it has either <= minsize or >= maxsize tiles, or
        /// the evolution has performed maxevents steps.
        ///
        /// Parameters:
        ///     state (StateKTAM)
        ///     minsize (int)
        ///     maxsize (int)
        ///     maxevents (int)
        #[text_signature = "(self, state, minsize, maxsize, maxevents)"]
        fn evolve_in_size_range_events_max(
            &mut self,
            state: &mut StateKTAM,
            minsize: u32,
            maxsize: u32,
            maxevents: u64,
        ) {
            let mut rng = SmallRng::from_entropy();
            self.inner.evolve_in_size_range_events_max(
                &mut state.inner,
                minsize,
                maxsize,
                maxevents,
                &mut rng,
            );
        }
    }

    /// Static (no changes to concentrations, bond strengths, parameters, etc) kTAM system, with periodic boundaries.
    /// Currently implements fission, and dimer chunk detachment.
    ///
    /// Can be used for multiple simulations / states.
    ///
    /// Currently, Python can't handle seeds.
    ///
    /// Parameters:
    ///     tile_stoics (f64 N 1D array): tile stoichiometry (for N-1 tiles).  1.0 means tile concentration :math:`e^{-G_{mc}+\alpha}`.
    ///                                   "Tile 0" must correspond with an empty space, and should have stoic 0.
    ///     tile_edges (u32 Nx4 array): tile edges for each tile.  First row should be [0, 0, 0, 0].
    ///     glue_strengths (f64 G 1D array): glue strengths for each glue. 0 should have strength 0 and corresponds with a null glue.
    ///                                      Strength 1.0 corresponds to a glue of strength :math:`G_{se}`.
    ///     gse, gmc (f64): :math:`G_{mc}` and :math:`G_{se}`.
    ///     alpha (optional, float): Non-edge-dependent attachment strength adjustment (default 0.0).
    ///     k_f (optional, float): Non-adjusted forward rate constant, not accounting for :math:`\alpha` (default 1e6).
    ///     fission (optional, str): one of "off", "just-detach", "on", "keep-largest", "keep-weighted" (default "off").
    ///     chunk_handling (optional, str): one of "off"/"none" or "detach" (default "off")
    ///     chunk_size (optional, str): currently, must be "dimer" if chunk_handling is set to "detach".  Can also be set to "off"/"none" (the default).
    ///     tile_names (optional, list[str]): list of tile names (default None).    #[pyclass]
    #[pyclass]
    #[derive(Debug)]
    #[text_signature = "(tile_concs, tile_edges, glue_strengths, gse, gmc, alpha, k_f, fission, chunk_handling, chunk_size, tile_names)"]
    struct StaticKTAMPeriodic {
        inner: system::StaticKTAM<
            state::QuadTreeState<canvas::CanvasPeriodic, state::NullStateTracker>,
        >,
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
            chunk_handling: Option<&str>,
            chunk_size: Option<&str>,
            tile_names: Option<Vec<String>>,
        ) -> PyResult<Self> {
            let fission_handling = match fission {
                Some(fs) => Some(match fs {
                    "off" => FissionHandling::NoFission,
                    "just-detach" => FissionHandling::JustDetach,
                    "on" => FissionHandling::KeepSeeded,
                    "keep-largest" => FissionHandling::KeepLargest,
                    "keep-weighted" => FissionHandling::KeepWeighted,
                    _ => return Err(PyValueError::new_err("Invalid fission handling option")),
                }),
                None => None,
            };

            let chunk_handling = Some(match chunk_handling {
                Some(ch) => match ch {
                    "off" | "none" => ChunkHandling::None,
                    "detach" => ChunkHandling::Detach,
                    _ => return Err(PyValueError::new_err("Invalid chunk handling option")),
                },
                None => ChunkHandling::None,
            });

            let chunk_size = match chunk_size {
                Some(cs) => match cs {
                    "dimer" => Some(ChunkSize::Dimer),
                    _ => return Err(PyValueError::new_err("Invalid chunk size option")),
                },
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
                chunk_handling,
                chunk_size,
                tile_names,
                None,
            );

            Ok(Self { inner })
        }

        #[classmethod]
        /// Creates a StaticKTAMPeriodic instance from a JSON string.  Ignores canvas choice in JSON.
        #[text_signature = "(self, json_data)"]
        fn from_json(_cls: &PyType, json_data: &str) -> PyResult<Self> {
            let tileset = match rgrow::parser::TileSet::from_json(json_data) {
                Ok(t) => t,
                Err(e) => {
                    return Err(PyValueError::new_err(format!(
                        "Couldn't parse tileset json: {:?}",
                        e
                    )))
                }
            };
            Ok(Self {
                inner: tileset.into_static_seeded_ktam_p(),
            })
        }

        /// Generates a StaticKTAMPeriodic instance from "raw" inputs, similar to what the model uses internally.
        ///
        /// Parameters:
        ///     tile_adj_rates (float array, shape N): the "adjusted unitless attachment rate" for each tile (N-1 tiles, 0 is empty).  This corresponds to
        ///                     :math:`e^{-G_{mc}}` for a tile with :math:`G_{mc}`, ie, it does not account for :math:`k_f` or
        ///                     :math:`\alpha`.
        ///     energy_ns (float array, shape NxN): in position [i,j], the bond strength that results from tile i being north of tile j.
        ///     energy_ws (float array, shape NxN): same, now with i the west tile, and j the east tile.
        ///     k_f (float, optional): :math:`k_f`, default 1e6.
        ///     alpha (float, optional): :math:`\alpha`, default 0.0.
        ///     fission (optional): one of "off", "just-detach", "on", "keep-largest", "keep-weighted" (default "off").
        ///     chunk_handling (optional, str): one of "off"/"none" or "detach" (default "off")
        ///     chunk_size (optional, str): currently, must be "dimer" if chunk_handling is set to "detach".  Can also be set to "off"/"none" (the default).
        #[text_signature = "(tile_rates, energy_ns, energy_we, k_f, alpha, fission, chunk_handling, chunk_size)"]
        #[classmethod]
        fn from_raw(
            _cls: &PyType,
            tile_rates: PyReadonlyArray1<f64>,
            energy_ns: PyReadonlyArray2<base::Energy>,
            energy_we: PyReadonlyArray2<base::Energy>,
            k_f: f64,
            alpha: f64,
            fission: Option<&str>,
            chunk_handling: Option<&str>,
            chunk_size: Option<&str>,
        ) -> PyResult<Self> {
            let fission_handling = match fission {
                Some(fs) => Some(match fs {
                    "off" => FissionHandling::NoFission,
                    "just-detach" => FissionHandling::JustDetach,
                    "on" => FissionHandling::KeepSeeded,
                    "keep-largest" => FissionHandling::KeepLargest,
                    "keep-weighted" => FissionHandling::KeepWeighted,
                    _ => return Err(PyValueError::new_err("Invalid fission handling option")),
                }),
                None => None,
            };

            let chunk_handling = Some(match chunk_handling {
                Some(ch) => match ch {
                    "off" | "none" => ChunkHandling::None,
                    "detach" => ChunkHandling::Detach,
                    _ => return Err(PyValueError::new_err("Invalid chunk handling option")),
                },
                None => ChunkHandling::None,
            });

            let chunk_size = match chunk_size {
                Some(cs) => match cs {
                    "dimer" => Some(ChunkSize::Dimer),
                    _ => return Err(PyValueError::new_err("Invalid chunk size option")),
                },
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
                    chunk_handling,
                    chunk_size,
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

    /// Static KTAM with cover strands.  Currently very unstable.  Needs to be generated from json.
    #[pyclass]
    #[derive(Debug)]
    #[text_signature = "(tile_concs, tile_edges, glue_strengths, gse, gmc, alpha, k_f, fission, tile_names)"]
    struct StaticKTAMCover {
        inner: system::StaticKTAMCover<
            state::QuadTreeState<canvas::CanvasSquare, state::NullStateTracker>,
        >,
    }

    #[pymethods]
    impl StaticKTAMCover {
        #[classmethod]
        fn from_json(_cls: &PyType, json_data: &str) -> PyResult<Self> {
            let mut tileset = match rgrow::parser::TileSet::from_json(json_data) {
                Ok(t) => t,
                Err(e) => {
                    return Err(PyValueError::new_err(format!(
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
            )
            .unwrap();

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

        fn calc_mismatch_locations<'py>(
            &self,
            state: &StateKTAM,
            py: Python<'py>,
        ) -> &'py PyArray2<usize> {
            self.inner
                .calc_mismatch_locations(&state.inner)
                .to_pyarray(py)
        }

        fn calc_mismatches(&self, state: &StateKTAM) -> u32 {
            self.inner.calc_mismatches(&state.inner)
        }

        fn evolve_in_size_range_events_max(
            &mut self,
            state: &mut StateKTAM,
            minsize: u32,
            maxsize: u32,
            maxevents: u64,
        ) {
            let mut rng = SmallRng::from_entropy();
            self.inner.evolve_in_size_range_events_max(
                &mut state.inner,
                minsize,
                maxsize,
                maxevents,
                &mut rng,
            );
        }
    }

    /// A simulation state for a static kTAM simulation, using a square canvas.
    ///
    /// Parameters:
    ///     size (int): the size of the canvas (assumed to be square).  Should be a power of 2.
    ///     system (StaticKTAM): the system that the simulation will use.
    #[pyclass]
    #[derive(Clone, Debug)]
    #[text_signature = "(size, system)"]
    struct StateKTAM {
        inner: state::QuadTreeState<canvas::CanvasSquare, state::NullStateTracker>,
    }

    #[pymethods]
    impl StateKTAM {
        #[new]
        fn new(size: usize, system: &mut StaticKTAM) -> PyResult<Self> {
            let mut state = state::QuadTreeState::<_, state::NullStateTracker>::create_raw(
                Array2::zeros((size, size)),
            )
            .unwrap();

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

        /// Prints debug information.
        fn debug(&self) -> String {
            format!("{:?}", self.inner)
        }

        #[classmethod]
        /// Creates a simulation state with the West-East dimer of tile numbers w, e, centered in
        /// a canvas of size size (must be 2^L).
        #[text_signature = "(self, system, w, e, size)"]
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
        #[text_signature = "(self, system, n, s, size)"]
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

        #[text_signature = "(self, system, point_y, point_x, tile)"]
        /// Sets the point (py, px) to a particular tile (or empty, with 0), using StaticKTAM `system`.
        /// Updates rates after setting.
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

        /// Tries to take a single step.  May fail if the canvas is empty, or there is no step possible, in which case
        /// it will raise a PyValueError.
        #[text_signature = "(self, system)"]
        fn take_step(&mut self, system: &StaticKTAM) -> PyResult<()> {
            let mut rng = SmallRng::from_entropy();
            match system.inner.state_step(&mut self.inner, &mut rng, 1e100) {
                StepOutcome::HadEventAt(_) | StepOutcome::DeadEventAt(_) => Ok(()),
                StepOutcome::NoEventIn(_) => Err(PyValueError::new_err("No event")),
                StepOutcome::ZeroRate => Err(PyValueError::new_err("Zero rate")),
            }
        }

        /// Provided with a StaticKTAM system, evolve the state until it reaches `minsize` or `maxsize` number of tiles,
        /// or until `maxevents` events have taken place during the evolution.  This is present for backward compatibility.
        #[text_signature = "(self, system, minsize, maxsize, maxevents)"]
        fn evolve_in_size_range(
            &mut self,
            system: &mut StaticKTAM,
            minsize: base::NumTiles,
            maxsize: base::NumTiles,
            maxevents: u64,
        ) -> PyResult<()> {
            let mut rng = SmallRng::from_entropy();
            system.inner.evolve_in_size_range_events_max(
                &mut self.inner,
                minsize,
                maxsize,
                maxevents,
                &mut rng,
            );

            Ok(())
        }

        /// Returns the canvas as a numpy array.  Note that this creates an array copy.
        fn to_array<'py>(&self, py: Python<'py>) -> &'py PyArray2<base::Tile> {
            self.inner.canvas.raw_array().to_pyarray(py)
        }

        /// Returns a copy of the state.
        fn copy(&self) -> PyResult<Self> {
            Ok(Self {
                inner: self.inner.clone(),
            })
        }

        /// The number of tiles in the state (stored, not calculated).
        #[getter]
        fn ntiles(&self) -> base::NumTiles {
            self.inner.ntiles()
        }

        /// The current time since initiation (in seconds) of the state.
        #[getter]
        fn time(&self) -> f64 {
            self.inner.time()
        }

        /// The total number of events that have taken place in the state.
        #[getter]
        fn events(&self) -> u64 {
            self.inner.total_events()
        }

        #[text_signature = "(self, level_number)"]
        /// Returns rate array for level `level_number` (0 is full, each subsequent is shape (previous/2, previous/2)
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
    /// Runs Forward Flux Sampling on a StaticKTAM system using number of tiles as a measure, and returns
    /// a tuple of (nucleation_rate, dimerization_rate, forward_probs).
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

        let ret = (
            fr.nucleation_rate(),
            fr.dimerization_rate,
            fr.forward_vec().clone(),
        );

        drop(fr);

        ret
    }

    #[pyfunction]
    #[text_signature = "(system, num_states, target_size, canvas_size, max_init_events, max_subseq_events, start_size, size_step)"]
    /// Runs Forward Flux Sampling, and returns a tuple of using number of tiles as a measure, and returns
    /// (nucleation_rate, dimerization_rate, forward_probs, final configs, configs for each level).
    /// Note that this consumes a *large* amount of memory, and other functions that do not return every level's configurations may be better
    /// if you don't need all of this information.
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
    /// Runs Forward Flux Sampling, and returns a tuple of using number of tiles as a measure, and returns
    /// (nucleation_rate, dimerization_rate, forward_probs, final configs, configs for each level).
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
    /// Runs Forward Flux Sampling for StaticKTAMCover, and returns a tuple of using number of tiles as a measure, and returns
    /// (nucleation_rate, dimerization_rate, forward_probs, final configs).
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
    /// Runs Forward Flux Sampling for StaticKTAMPeriodic, and returns a tuple of using number of tiles as a measure, and returns
    /// (nucleation_rate, dimerization_rate, forward_probs, final configs).
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

    #[pyfunction]
    #[text_signature = "(system, varpermean2, min_states, target_size, cutoff_prob, cutoff_number, canvas_size, max_init_events, max_subseq_events, start_size, size_step)"]
    /// Runs Forward Flux Sampling for StaticKTAMPeriodic, and returns a tuple of using number of tiles as a measure, and returns
    /// (nucleation_rate, dimerization_rate, forward_probs, final configs).
    fn ffs_run_final_p_cvar_cut<'py>(
        system: &StaticKTAMPeriodic,
        varpermean2: f64,
        min_states: usize,
        target_size: base::NumTiles,
        cutoff_prob: f64,
        cutoff_number: usize,
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
        Vec<&'py PyArray2<base::Tile>>,
        Vec<usize>,
        Vec<usize>,
        Vec<u32>,
    ) {
        let fr = ffs::FFSRun::create_with_constant_variance_and_size_cutoff(
            system.inner.to_owned(),
            varpermean2,
            min_states,
            target_size,
            cutoff_prob,
            cutoff_number,
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
            fr.level_list.iter().map(|x| x.num_states).collect(),
            fr.level_list.iter().map(|x| x.num_trials).collect(),
            fr.level_list.iter().map(|x| x.target_size).collect(),
        );

        drop(fr);

        ret
    }

    m.add_wrapped(wrap_pyfunction!(ffs_run))?;
    m.add_wrapped(wrap_pyfunction!(ffs_run_full))?;
    m.add_wrapped(wrap_pyfunction!(ffs_run_final))?;
    m.add_wrapped(wrap_pyfunction!(ffs_run_final_p))?;
    m.add_wrapped(wrap_pyfunction!(ffs_run_final_p_cvar_cut))?;
    m.add_wrapped(wrap_pyfunction!(ffs_run_final_cover))?;

    Ok(())
}
