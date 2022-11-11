extern crate rgrow;

use ndarray::Array2;
use numpy::ToPyArray;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyBytes, PyType};
use pyo3::{prelude::*, wrap_pyfunction};
use rand::{rngs::SmallRng, SeedableRng};
use rgrow::base;
use rgrow::canvas::Canvas;
use rgrow::canvas::{self, PointSafe2};
use rgrow::ffs::{self, FFSResult};
use rgrow::system::{EvolveBounds, StepOutcome, TileBondInfo};
use rgrow::tileset::FromTileSet;

use rgrow::models::ktam;
use rgrow::models::oldktam;

use rgrow::state;
use rgrow::state::{StateCreate, StateStatus};
use rgrow::system;
use rgrow::system::{ChunkHandling, ChunkSize, FissionHandling};
use rgrow::system::{System, SystemWithStateCreate};

use core::f64;
use std::collections::HashMap;
use std::fmt::Debug;
use std::time::Duration;

mod tileset;

use bincode::{deserialize, serialize};
use serde::{Deserialize, Serialize};

#[derive(FromPyObject)]
pub enum ParsedSeed {
    Single(usize, usize, base::Tile),
    Multi(Vec<(usize, usize, base::Tile)>),
}

/// A (somewhat rudimentary and very unstable) Python interface to Rgrow.
///
/// As static dispatch doesn't work with PyO3, this currently has separate types for
/// combinations of different systems and states/canvases.
///
/// The most important classes are:
///
/// - OldKTAM (and the StaticKTAMPeriodic variant), which specify the kinetic model to use and its parameters.
/// - StateKTAM, which stores the state of a single assembly.
#[pymodule]
#[pyo3(name = "rgrow")]
fn pyrgrow<'py>(_py: Python<'py>, m: &PyModule) -> PyResult<()> {
    #[pyclass(module = "rgrow.rgrow_old")]
    #[derive(Debug)]
    pub struct StaticKTAMOrder {
        inner: oldktam::OldKTAM<state::QuadTreeState<canvas::CanvasSquare, state::OrderTracker>>,
    }

    #[pymethods]
    impl StaticKTAMOrder {
        #[new]
        fn new(
            tile_concs: PyReadonlyArray1<f64>,
            tile_edges: PyReadonlyArray2<base::Glue>,
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

            let inner = oldktam::OldKTAM::from_ktam(
                tile_concs.to_owned_array(),
                tile_edges.to_owned_array(),
                glue_strengths.to_owned_array(),
                gse,
                gmc,
                alpha,
                k_f,
                Some(oldktam::Seed::None()),
                fission_handling,
                chunk_handling,
                chunk_size,
                tile_names,
                None,
            );

            Ok(Self { inner })
        }

        fn new_state(&mut self, shape: (usize, usize)) -> PyResult<StateKTAMOrder> {
            Ok(StateKTAMOrder {
                inner: self.inner.new_state(shape).unwrap(),
            })
        }

        #[classmethod]
        /// Creates a OldKTAM instance from a JSON string.  Ignores canvas choice in JSON.
        fn from_json(_cls: &PyType, json_data: &str) -> PyResult<Self> {
            let tileset = match rgrow::tileset::TileSet::from_json(json_data) {
                Ok(t) => t,
                Err(e) => {
                    return Err(PyValueError::new_err(format!(
                        "Couldn't parse tileset json: {:?}",
                        e
                    )))
                }
            };
            Ok(Self {
                inner: oldktam::OldKTAM::from_tileset(&tileset),
            })
        }
    }

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
    #[pyclass(module = "rgrow")]
    #[derive(Debug)]
    // #[text_signature = "(tile_stoics, tile_edges, glue_strengths, gse, gmc, alpha, k_f, fission, chunk_handling, chunk_size, tile_names)"]
    struct OldKTAM {
        inner:
            oldktam::OldKTAM<state::QuadTreeState<canvas::CanvasSquare, state::NullStateTracker>>,
    }

    #[pymethods]
    impl OldKTAM {
        #[new]
        fn new(
            tile_concs: PyReadonlyArray1<f64>,
            tile_edges: PyReadonlyArray2<base::Glue>,
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

            let inner = oldktam::OldKTAM::from_ktam(
                tile_concs.to_owned_array(),
                tile_edges.to_owned_array(),
                glue_strengths.to_owned_array(),
                gse,
                gmc,
                alpha,
                k_f,
                Some(oldktam::Seed::None()),
                fission_handling,
                chunk_handling,
                chunk_size,
                tile_names,
                None,
            );

            Ok(Self { inner })
        }

        fn new_state(&self, shape: (usize, usize)) -> PyResult<StateKTAM> {
            Ok(StateKTAM {
                inner: self.inner.new_state(shape).unwrap(),
            })
        }

        #[classmethod]
        /// Creates a OldKTAM instance from a JSON string.  Ignores canvas choice in JSON.
        // #[text_signature = "(self, json_data)"]
        fn from_json(_cls: &PyType, json_data: &str) -> PyResult<Self> {
            let tileset = match rgrow::tileset::TileSet::from_json(json_data) {
                Ok(t) => t,
                Err(e) => {
                    return Err(PyValueError::new_err(format!(
                        "Couldn't parse tileset json: {:?}",
                        e
                    )))
                }
            };
            Ok(Self {
                inner: oldktam::OldKTAM::from_tileset(&tileset),
            })
        }

        /// Generates a OldKTAM instance from "raw" inputs, similar to what the model uses internally.
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
        // #[text_signature = "(tile_adj_rates, energy_ns, energy_we, k_f, alpha, fission)"]
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
                inner: oldktam::OldKTAM::from_raw(
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
        fn tile_adj_rates<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
            self.inner.tile_adj_concs.to_pyarray(py)
        }

        #[getter]
        fn alpha(&self) -> f64 {
            self.inner.alpha
        }

        #[getter]
        fn g_mc(&self) -> Option<f64> {
            self.inner.g_mc
        }

        #[getter]
        fn g_se(&self) -> Option<f64> {
            self.inner.g_se
        }

        #[getter]
        fn tile_names(&self) -> Vec<String> {
            self.inner.tile_names()
        }

        #[getter]
        fn tile_colors(&self) -> Vec<[u8; 4]> {
            self.inner.tile_colors().to_owned()
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

        /// OldKTAM.evolve_in_size_range_events_max(self, state, minsize, maxsize, maxevents)
        ///
        /// A System-centric evolve method.  Evolves the provided state until it has either <= minsize or >= maxsize tiles, or
        /// the evolution has performed maxevents steps.
        ///
        /// Parameters:
        ///     state (StateKTAM)
        ///     minsize (int)
        ///     maxsize (int)
        ///     maxevents (int)
        // #[text_signature = "(self, state, minsize, maxsize, maxevents)"]
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
    ///     tile_names (optional, list[str]): list of tile names (default None).    
    #[pyclass(module = "rgrow")]
    #[derive(Debug, Serialize, Deserialize)]
    // #[text_signature = "()"]
    struct StaticKTAMPeriodic {
        inner: Option<
            oldktam::OldKTAM<state::QuadTreeState<canvas::CanvasPeriodic, state::NullStateTracker>>,
        >,
    }

    #[pymethods]
    impl StaticKTAMPeriodic {
        #[new]
        fn new() -> PyResult<Self> {
            Ok(Self { inner: None })
        }

        #[classmethod]
        fn from_params(
            _cls: &PyType,
            tile_concs: PyReadonlyArray1<f64>,
            tile_edges: PyReadonlyArray2<base::Glue>,
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

            let inner = Some(oldktam::OldKTAM::from_ktam(
                tile_concs.to_owned_array(),
                tile_edges.to_owned_array(),
                glue_strengths.to_owned_array(),
                gse,
                gmc,
                alpha,
                k_f,
                Some(oldktam::Seed::None()),
                fission_handling,
                chunk_handling,
                chunk_size,
                tile_names,
                None,
            ));

            Ok(Self { inner })
        }

        pub fn __getstate__(&self, py: Python) -> PyResult<PyObject> {
            Ok(PyBytes::new(py, &serialize(&self.inner).unwrap()).to_object(py))
        }

        pub fn __setstate__(&mut self, py: Python, state: PyObject) -> PyResult<()> {
            match state.extract::<&PyBytes>(py) {
                Ok(s) => {
                    self.inner = deserialize(s.as_bytes()).unwrap();
                    Ok(())
                }
                Err(e) => Err(e),
            }
        }

        #[classmethod]
        /// Creates a StaticKTAMPeriodic instance from a JSON string.  Ignores canvas choice in JSON.
        // #[text_signature = "(self, json_data)"]
        fn from_json(_cls: &PyType, json_data: &str) -> PyResult<Self> {
            let tileset = match rgrow::tileset::TileSet::from_json(json_data) {
                Ok(t) => t,
                Err(e) => {
                    return Err(PyValueError::new_err(format!(
                        "Couldn't parse tileset json: {:?}",
                        e
                    )))
                }
            };
            Ok(Self {
                inner: Some(oldktam::OldKTAM::from_tileset(&tileset)),
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
        // #[text_signature = "(tile_rates, energy_ns, energy_we, k_f, alpha, fission, chunk_handling, chunk_size)"]
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
                inner: Some(oldktam::OldKTAM::from_raw(
                    tile_rates.to_owned_array(),
                    energy_ns.to_owned_array(),
                    energy_we.to_owned_array(),
                    k_f,
                    alpha,
                    fission_handling,
                    chunk_handling,
                    chunk_size,
                )),
            })
        }

        #[getter]
        fn tile_rates<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
            self.inner.as_ref().unwrap().tile_adj_concs.to_pyarray(py)
        }

        #[getter]
        fn alpha(&self) -> f64 {
            self.inner.as_ref().unwrap().alpha
        }

        #[getter]
        fn g_mc(&self) -> Option<f64> {
            self.inner.as_ref().unwrap().g_mc
        }

        #[getter]
        fn g_se(&self) -> Option<f64> {
            self.inner.as_ref().unwrap().g_se
        }

        // #[getter]
        // fn tile_names(&self) -> Vec<String> {
        //     self.inner.tile_names()
        // }

        // #[getter]
        // fn tile_colors(&self) -> Vec<[u8; 4]> {
        //     self.inner.tile_colors()
        // }

        #[getter]
        fn energy_ns<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
            self.inner.as_ref().unwrap().energy_ns.to_pyarray(py)
        }

        #[getter]
        fn energy_we<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
            self.inner.as_ref().unwrap().energy_we.to_pyarray(py)
        }
    }

    // /// Static KTAM with cover strands.  Currently very unstable.  Needs to be generated from json.
    // #[pyclass(module = "rgrow")]
    // #[derive(Debug)]
    // // #[text_signature = "(tile_concs, tile_edges, glue_strengths, gse, gmc, alpha, k_f, fission, tile_names)"]
    // struct StaticKTAMCover {
    //     inner: system::StaticKTAMCover<
    //         state::QuadTreeState<canvas::CanvasSquare, state::NullStateTracker>,
    //     >,
    // }

    // #[pymethods]
    // impl StaticKTAMCover {
    //     #[classmethod]
    //     fn from_json(_cls: &PyType, json_data: &str) -> PyResult<Self> {
    //         let mut tileset = match rgrow::parser::TileSet::from_json(json_data) {
    //             Ok(t) => t,
    //             Err(e) => {
    //                 return Err(PyValueError::new_err(format!(
    //                     "Couldn't parse tileset json: {:?}",
    //                     e
    //                 )))
    //             }
    //         };
    //         Ok(Self {
    //             inner: tileset.into_static_ktam_cover(),
    //         })
    //     }

    //     fn debug(&self) -> String {
    //         format!("{:?}", self.inner)
    //     }

    //     fn new_state(&mut self, size: usize) -> PyResult<StateKTAM> {
    //         let mut state = state::QuadTreeState::<_, state::NullStateTracker>::create_raw(
    //             Array2::zeros((size, size)),
    //         )
    //         .unwrap();

    //         let sl = self.inner.seed_locs();

    //         for (p, t) in sl {
    //             // FIXME: for large seeds,
    //             // this could be faster by doing raw writes, then update_entire_state
    //             // but we would need to distinguish sizing.
    //             // Or maybe there is fancier way with a set?
    //             self.inner.set_point(&mut state, p.0, t);
    //         }

    //         Ok(StateKTAM { inner: state })
    //     }

    //     #[getter]
    //     fn tile_rates<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
    //         self.inner.inner.tile_adj_concs.to_pyarray(py)
    //     }

    //     #[getter]
    //     fn energy_ns<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
    //         self.inner.inner.energy_ns.to_pyarray(py)
    //     }

    //     #[getter]
    //     fn energy_we<'py>(&self, py: Python<'py>) -> &'py PyArray2<f64> {
    //         self.inner.inner.energy_we.to_pyarray(py)
    //     }

    //     fn calc_mismatch_locations<'py>(
    //         &self,
    //         state: &StateKTAM,
    //         py: Python<'py>,
    //     ) -> &'py PyArray2<usize> {
    //         self.inner
    //             .calc_mismatch_locations(&state.inner)
    //             .to_pyarray(py)
    //     }

    //     fn calc_mismatches(&self, state: &StateKTAM) -> u32 {
    //         self.inner.calc_mismatches(&state.inner)
    //     }

    //     #[getter]
    //     fn tile_names(&self) -> Vec<String> {
    //         self.inner.tile_names()
    //     }

    //     #[getter]
    //     fn tile_colors(&self) -> Vec<[u8; 4]> {
    //         self.inner.tile_colors()
    //     }

    //     fn evolve_in_size_range_events_max(
    //         &mut self,
    //         state: &mut StateKTAM,
    //         minsize: u32,
    //         maxsize: u32,
    //         maxevents: u64,
    //     ) {
    //         let mut rng = SmallRng::from_entropy();
    //         self.inner.evolve_in_size_range_events_max(
    //             &mut state.inner,
    //             minsize,
    //             maxsize,
    //             maxevents,
    //             &mut rng,
    //         );
    //     }
    // }

    /// A simulation state for a static kTAM simulation, using a square canvas.
    ///
    /// Parameters:
    ///     size (int): the size of the canvas (assumed to be square).  Should be a power of 2.
    ///     system (OldKTAM): the system that the simulation will use.
    #[pyclass(module = "rgrow")]
    #[derive(Clone, Debug)]
    // #[text_signature = "(size, system)"]
    struct StateKTAM {
        inner: state::QuadTreeState<canvas::CanvasSquare, state::NullStateTracker>,
    }

    /// A simulation state for a static kTAM simulation, using a square periodic canvas.
    ///
    /// Parameters:
    ///     size (int): the size of the canvas (assumed to be square).  Should be a power of 2.
    ///     system (OldKTAM): the system that the simulation will use.
    #[pyclass(module = "rgrow")]
    #[derive(Clone, Debug)]
    // #[text_signature = "(size, system)"]
    struct StateKTAMPeriodic {
        inner: state::QuadTreeState<canvas::CanvasPeriodic, state::NullStateTracker>,
    }

    #[pyclass(module = "rgrow")]
    #[derive(Clone, Debug)]
    // #[text_signature = "(size, system)"]
    struct StateKTAMOrder {
        inner: state::QuadTreeState<canvas::CanvasSquare, state::OrderTracker>,
    }

    #[pymethods]
    impl StateKTAM {
        #[new]
        fn new(size: usize) -> PyResult<Self> {
            let state = state::QuadTreeState::<_, state::NullStateTracker>::create_raw(
                Array2::zeros((size, size)),
            )
            .unwrap();

            Ok(Self { inner: state })
        }

        /// Prints debug information.
        fn debug(&self) -> String {
            format!("{:?}", self.inner)
        }

        #[classmethod]
        /// Creates a simulation state with the West-East dimer of tile numbers w, e, centered in
        /// a canvas of size size (must be 2^L).
        // #[text_signature = "(self, system, w, e, size)"]
        fn create_we_pair(
            _cls: &PyType,
            system: &mut OldKTAM,
            w: base::Tile,
            e: base::Tile,
            size: usize,
        ) -> PyResult<Self> {
            let inner = system.inner.create_we_pair(w, e, size).unwrap();

            Ok(Self { inner })
        }

        #[classmethod]
        // #[text_signature = "(self, canvas)"]
        fn create_raw<'py>(
            _cls: &PyType,
            canvas: &'py PyArray2<base::Tile>,
            _py: Python<'py>,
        ) -> PyResult<Self> {
            let inner = state::QuadTreeState::<_, state::NullStateTracker>::create_raw(
                canvas.to_owned_array(),
            )
            .unwrap();

            Ok(Self { inner })
        }

        #[classmethod]
        // #[text_signature = "(self, system, n, s, size)"]
        /// Creates a simulation state with the North-South dimer of tile numbers n, s, centered in
        /// a canvas of size size (must be 2^L).
        fn create_ns_pair(
            _cls: &PyType,
            system: &mut OldKTAM,
            n: base::Tile,
            s: base::Tile,
            size: usize,
        ) -> PyResult<Self> {
            let inner = system.inner.create_ns_pair(n, s, size).unwrap();

            Ok(Self { inner })
        }

        // #[text_signature = "(self, system, point_y, point_x, tile)"]
        /// Sets the point (py, px) to a particular tile (or empty, with 0), using OldKTAM `system`.
        /// Updates rates after setting.
        fn set_point(
            &mut self,
            system: &mut OldKTAM,
            py: usize,
            px: usize,
            t: base::Tile,
        ) -> PyResult<()> {
            system.inner.set_point(&mut self.inner, (py, px), t);

            Ok(())
        }

        /// Tries to take a single step.  May fail if the canvas is empty, or there is no step possible, in which case
        /// it will raise a PyValueError.
        // #[text_signature = "(self, system)"]
        fn take_step(&mut self, system: &OldKTAM) -> PyResult<()> {
            let mut rng = SmallRng::from_entropy();
            match system.inner.state_step(&mut self.inner, &mut rng, 1e100) {
                StepOutcome::HadEventAt(_) | StepOutcome::DeadEventAt(_) => Ok(()),
                StepOutcome::NoEventIn(_) => Err(PyValueError::new_err("No event")),
                StepOutcome::ZeroRate => Err(PyValueError::new_err("Zero rate")),
            }
        }

        /// Provided with a OldKTAM system, evolve the state until it reaches `minsize` or `maxsize` number of tiles,
        /// or until `maxevents` events have taken place during the evolution.  This is present for backward compatibility.
        // #[text_signature = "(self, system, minsize, maxsize, maxevents)"]
        fn evolve_in_size_range(
            &mut self,
            system: &mut OldKTAM,
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

        // #[text_signature = "(self, level_number)"]
        /// Returns rate array for level `level_number` (0 is full, each subsequent is shape (previous/2, previous/2)
        fn rates<'py>(&self, level: usize, py: Python<'py>) -> &'py PyArray2<f64> {
            self.inner.rates.0[level].to_pyarray(py)
        }
    }

    #[pymethods]
    impl StateKTAMOrder {
        #[new]
        fn new(size: usize, system: &mut StaticKTAMOrder) -> PyResult<Self> {
            let mut state = state::QuadTreeState::<_, state::OrderTracker>::create_raw(
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

        fn insert_seed(&mut self, system: &mut StaticKTAMOrder) -> PyResult<()> {
            system.inner.insert_seed(&mut self.inner);
            Ok(())
        }
        #[classmethod]
        /// Creates a simulation state with the West-East dimer of tile numbers w, e, centered in
        /// a canvas of size size (must be 2^L).
        // #[text_signature = "(self, system, w, e, size)"]
        fn create_we_pair(
            _cls: &PyType,
            system: &mut StaticKTAMOrder,
            w: base::Tile,
            e: base::Tile,
            size: usize,
        ) -> PyResult<Self> {
            let inner = system.inner.create_we_pair(w, e, size).unwrap();

            Ok(Self { inner })
        }

        #[classmethod]
        // #[text_signature = "(self, canvas)"]
        fn create_raw<'py>(
            _cls: &PyType,
            canvas: &'py PyArray2<base::Tile>,
            _py: Python<'py>,
        ) -> PyResult<Self> {
            let inner =
                state::QuadTreeState::<_, state::OrderTracker>::create_raw(canvas.to_owned_array())
                    .unwrap();

            Ok(Self { inner })
        }

        #[classmethod]
        // #[text_signature = "(self, system, n, s, size)"]
        /// Creates a simulation state with the North-South dimer of tile numbers n, s, centered in
        /// a canvas of size size (must be 2^L).
        fn create_ns_pair(
            _cls: &PyType,
            system: &mut StaticKTAMOrder,
            n: base::Tile,
            s: base::Tile,
            size: usize,
        ) -> PyResult<Self> {
            let inner = system.inner.create_ns_pair(n, s, size).unwrap();

            Ok(Self { inner })
        }

        // #[text_signature = "(self, system, point_y, point_x, tile)"]
        /// Sets the point (py, px) to a particular tile (or empty, with 0), using OldKTAM `system`.
        /// Updates rates after setting.
        fn set_point(
            &mut self,
            system: &mut StaticKTAMOrder,
            py: usize,
            px: usize,
            t: base::Tile,
        ) -> PyResult<()> {
            system.inner.set_point(&mut self.inner, (py, px), t);

            Ok(())
        }

        /// Tries to take a single step.  May fail if the canvas is empty, or there is no step possible, in which case
        /// it will raise a PyValueError.
        // #[text_signature = "(self, system)"]
        fn take_step(&mut self, system: &StaticKTAMOrder) -> PyResult<()> {
            let mut rng = SmallRng::from_entropy();
            match system.inner.state_step(&mut self.inner, &mut rng, 1e100) {
                StepOutcome::HadEventAt(_) | StepOutcome::DeadEventAt(_) => Ok(()),
                StepOutcome::NoEventIn(_) => Err(PyValueError::new_err("No event")),
                StepOutcome::ZeroRate => Err(PyValueError::new_err("Zero rate")),
            }
        }

        /// Provided with a OldKTAM system, evolve the state until it reaches `minsize` or `maxsize` number of tiles,
        /// or until `maxevents` events have taken place during the evolution.  This is present for backward compatibility.
        // #[text_signature = "(self, system, minsize, maxsize, maxevents)"]
        fn evolve_in_size_range(
            &mut self,
            system: &mut StaticKTAMOrder,
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

        fn order_array<'py>(&self, py: Python<'py>) -> &'py PyArray2<base::NumEvents> {
            self.inner.tracker.arr.to_pyarray(py)
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

        // #[text_signature = "(self, level_number)"]
        /// Returns rate array for level `level_number` (0 is full, each subsequent is shape (previous/2, previous/2)
        fn rates<'py>(&self, level: usize, py: Python<'py>) -> &'py PyArray2<f64> {
            self.inner.rates.0[level].to_pyarray(py)
        }
    }

    #[pymethods]
    impl StateKTAMPeriodic {
        #[new]
        fn new(size: usize) -> PyResult<Self> {
            let state = state::QuadTreeState::<_, state::NullStateTracker>::create_raw(
                Array2::zeros((size, size)),
            )
            .unwrap();

            Ok(Self { inner: state })
        }

        /// Prints debug information.
        fn debug(&self) -> String {
            format!("{:?}", self.inner)
        }

        #[classmethod]
        /// Creates a simulation state with the West-East dimer of tile numbers w, e, centered in
        /// a canvas of size size (must be 2^L).
        // #[text_signature = "(self, system, w, e, size)"]
        fn create_we_pair(
            _cls: &PyType,
            system: &mut StaticKTAMPeriodic,
            w: base::Tile,
            e: base::Tile,
            size: usize,
        ) -> PyResult<Self> {
            let inner = system
                .inner
                .as_mut()
                .unwrap()
                .create_we_pair(w, e, size)
                .unwrap();

            Ok(Self { inner })
        }

        #[classmethod]
        // #[text_signature = "(self, system, canvas)"]
        fn from_array<'py>(
            _cls: &PyType,
            canvas: &'py PyArray2<base::Tile>,
            _py: Python<'py>,
        ) -> PyResult<Self> {
            let mut inner = state::QuadTreeState::<_, state::NullStateTracker>::create_raw(
                canvas.to_owned_array(),
            )
            .unwrap();

            inner.recalc_ntiles();

            Ok(Self { inner })
        }

        #[classmethod]
        // #[text_signature = "(self, system, n, s, size)"]
        /// Creates a simulation state with the North-South dimer of tile numbers n, s, centered in
        /// a canvas of size size (must be 2^L).
        fn create_ns_pair(
            _cls: &PyType,
            system: &mut StaticKTAMPeriodic,
            n: base::Tile,
            s: base::Tile,
            size: usize,
        ) -> PyResult<Self> {
            let inner = system
                .inner
                .as_mut()
                .unwrap()
                .create_ns_pair(n, s, size)
                .unwrap();

            Ok(Self { inner })
        }

        // #[text_signature = "(self, system, point_y, point_x, tile)"]
        /// Sets the point (py, px) to a particular tile (or empty, with 0), using StaticKTAMPeriodic `system`.
        /// Updates rates after setting.
        fn set_point(
            &mut self,
            system: &mut StaticKTAMPeriodic,
            py: usize,
            px: usize,
            t: base::Tile,
        ) -> PyResult<()> {
            system
                .inner
                .as_mut()
                .unwrap()
                .set_point(&mut self.inner, (py, px), t);

            Ok(())
        }

        // /// Tries to take a single step.  May fail if the canvas is empty, or there is no step possible, in which case
        // /// it will raise a PyValueError.
        // #[text_signature = "(self, system)"]
        // fn take_step(&mut self, system: &StaticKTAMPeriodic) -> PyResult<()> {
        //     let mut rng = SmallRng::from_entropy();
        //     match system.inner.as_ref().unwrap().state_step(&mut self.inner, &mut rng, 1e100) {
        //         StepOutcome::HadEventAt(_) | StepOutcome::DeadEventAt(_) => Ok(()),
        //         StepOutcome::NoEventIn(_) => Err(PyValueError::new_err("No event")),
        //         StepOutcome::ZeroRate => Err(PyValueError::new_err("Zero rate")),
        //     }
        // }

        /// Provided with a StaticKTAMPeriodic system, evolve the state until it reaches `minsize` or `maxsize` number of tiles,
        /// or until `maxevents` events have taken place during the evolution.  This is present for backward compatibility.
        // #[text_signature = "(self, system, minsize, maxsize, maxevents)"]
        fn evolve_in_size_range(
            &mut self,
            system: &mut StaticKTAMPeriodic,
            minsize: base::NumTiles,
            maxsize: base::NumTiles,
            maxevents: u64,
        ) -> PyResult<()> {
            let mut rng = SmallRng::from_entropy();
            system
                .inner
                .as_mut()
                .unwrap()
                .evolve_in_size_range_events_max(
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

        // #[text_signature = "(self, level_number)"]
        /// Returns rate array for level `level_number` (0 is full, each subsequent is shape (previous/2, previous/2)
        fn rates<'py>(&self, level: usize, py: Python<'py>) -> &'py PyArray2<f64> {
            self.inner.rates.0[level].to_pyarray(py)
        }
    }

    #[pyclass(module = "rgrow")]
    #[derive(Debug, Serialize, Deserialize)]
    struct NewKTAMPeriodic {
        inner: ktam::KTAM<state::QuadTreeState<canvas::CanvasPeriodic, state::NullStateTracker>>,
    }

    #[pymethods]
    impl NewKTAMPeriodic {
        #[new]
        fn new(ntiles: usize, nglues: usize) -> Self {
            Self {
                inner: (ktam::KTAM::new_sized(ntiles, nglues)),
            }
        }

        fn new_state(&self, size: usize) -> PyResult<StateKTAMPeriodic> {
            let mut state = StateKTAMPeriodic::new(size)?;
            self.inner.setup_state(&mut state.inner);
            Ok(state)
        }

        #[getter]
        fn get_tile_names(&self) -> Vec<String> {
            self.inner.tile_names.clone()
        }

        #[setter]
        fn set_tile_names(&mut self, names: Vec<String>) {
            self.inner.tile_names = names;
        }

        #[getter]
        fn get_tile_concs<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
            self.inner.tile_concs.to_pyarray(py)
        }

        #[setter]
        fn set_tile_concs<'py>(&mut self, concs: &'py PyArray1<f64>) {
            concs
                .to_owned_array()
                .mapv(|x| x.into())
                .move_into(&mut self.inner.tile_concs);
            self.inner.update_system();
        }

        #[getter]
        fn get_tile_edges<'py>(&self, py: Python<'py>) -> &'py PyArray2<usize> {
            self.inner.tile_edges.to_pyarray(py)
        }

        #[setter]
        fn set_tile_edges<'py>(&mut self, edges: &'py PyArray2<usize>) {
            edges
                .to_owned_array()
                .mapv(|x| x.into())
                .move_into(&mut self.inner.tile_edges);
            self.inner.update_system();
        }

        #[getter]
        fn get_glue_strengths<'py>(&self, py: Python<'py>) -> &'py PyArray1<f64> {
            self.inner.glue_strengths.to_pyarray(py)
        }

        #[setter]
        fn set_glue_strengths<'py>(&mut self, strengths: &'py PyArray1<f64>) -> PyResult<()> {
            if strengths.shape() == self.inner.glue_strengths.shape() {
                strengths
                    .to_owned_array()
                    .mapv(|x| x.into())
                    .move_into(&mut self.inner.glue_strengths);
                self.inner.update_system();
                Ok(())
            } else {
                Err(PyValueError::new_err(
                    "Must specify same number of glue strengths as glues",
                ))
            }
        }

        #[getter]
        fn get_alpha(&self) -> f64 {
            self.inner.alpha.into()
        }

        #[setter]
        fn set_alpha(&mut self, alpha: f64) {
            self.inner.alpha = alpha.into();
            self.inner.update_system();
        }

        #[getter]
        fn get_g_se(&self) -> f64 {
            self.inner.g_se.into()
        }

        #[getter]
        fn get_g_mc(&self) -> f64 {
            // Return average of tile concs excluding first and zeros.
            let mut sum = 0.0;
            let mut count = 0;
            for i in 1..self.inner.tile_concs.len() {
                if self.inner.tile_concs[i] > 0.0 {
                    sum += self.inner.tile_concs[i];
                    count += 1;
                }
            }
            let mean_conc = sum / count as f64;
            -(mean_conc / 1.0e9).ln() + self.inner.alpha
        }

        #[setter]
        fn set_g_se(&mut self, g_se: f64) {
            self.inner.g_se = g_se.into();
            self.inner.update_system();
        }

        fn set_point(&mut self, state: &mut StateKTAMPeriodic, x: usize, y: usize, t: base::Tile) {
            self.inner.set_point(&mut state.inner, (x, y), t);
        }

        fn set_duples(&mut self, hduples: Vec<(usize, usize)>, vduples: Vec<(usize, usize)>) {
            self.inner.set_duples(hduples, vduples)
        }

        fn evolve(
            &self,
            state: &mut StateKTAMPeriodic,
            for_events: Option<u64>,
            for_time: Option<f64>,
            size_min: Option<u32>,
            size_max: Option<u32>,
            for_wall_time: Option<f64>,
            py: Python<'_>,
        ) {
            let mut rng = SmallRng::from_entropy();
            let bounds = EvolveBounds {
                events: for_events,
                time: for_time,
                size_min,
                size_max,
                wall_time: match for_wall_time {
                    Some(secs) => Some(Duration::from_secs_f64(secs)),
                    None => None,
                },
            };
            py.allow_threads(|| {
                self.inner
                    .evolve(&mut state.inner, &mut rng, bounds)
                    .unwrap();
            });
        }

        fn evolve_in_size_range_events_max(
            &mut self,
            state: &mut StateKTAMPeriodic,
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

        #[setter]
        fn set_fission(&mut self, fission: &str) {
            self.inner.fission_handling = match fission {
                "just-detach" | "surface" => system::FissionHandling::JustDetach,
                "on" | "keep-seeded" => system::FissionHandling::KeepSeeded,
                "keep-largest" => system::FissionHandling::KeepLargest,
                "keep-weighted" => system::FissionHandling::KeepWeighted,
                "off" | "no-fission" => system::FissionHandling::NoFission,
                _ => panic!("Invalid fission handling"),
            }
        }

        #[getter]
        fn get_fission(&self) -> String {
            match self.inner.fission_handling {
                system::FissionHandling::JustDetach => "just-detach".to_string(),
                system::FissionHandling::KeepSeeded => "keep-seeded".to_string(),
                system::FissionHandling::KeepLargest => "keep-largest".to_string(),
                system::FissionHandling::KeepWeighted => "keep-weighted".to_string(),
                system::FissionHandling::NoFission => "no-fission".to_string(),
            }
        }

        #[setter]
        fn set_seed(&mut self, seed: Option<ParsedSeed>) {
            let ns = match seed {
                Some(ParsedSeed::Single(y, x, v)) => ktam::Seed::SingleTile {
                    point: PointSafe2((y, x)),
                    tile: v,
                },
                None => ktam::Seed::None(),
                Some(ParsedSeed::Multi(vec)) => {
                    let mut hm = HashMap::default();
                    hm.extend(vec.iter().map(|(y, x, v)| (PointSafe2((*y, *x)), *v)));
                    ktam::Seed::MultiTile(hm)
                }
            };
            self.inner.seed = ns;
        }

        #[getter]
        fn get_seed(&self, py: Python<'_>) -> PyObject {
            match &self.inner.seed {
                ktam::Seed::SingleTile { point, tile } => {
                    (point.0 .0, point.0 .1, *tile).to_object(py)
                }
                ktam::Seed::None() => None::<bool>.to_object(py),
                ktam::Seed::MultiTile(hm) => {
                    let mut vec = Vec::new();
                    for (point, tile) in hm {
                        vec.push((point.0 .0, point.0 .1, *tile));
                    }
                    vec.to_object(py)
                }
            }
        }

        fn __repr__(&self) -> String {
            format!("{:?}", self.inner)
        }
    }

    m.add_class::<OldKTAM>()?;
    m.add_class::<StaticKTAMOrder>()?;
    // m.add_class::<StaticKTAMCover>()?;
    m.add_class::<StaticKTAMPeriodic>()?;
    m.add_class::<StateKTAM>()?;
    m.add_class::<StateKTAMOrder>()?;
    m.add_class::<StateKTAMPeriodic>()?;
    m.add_class::<NewKTAMPeriodic>()?;

    #[pyfunction]
    // #[text_signature = "(system, num_states, target_size, canvas_size, max_init_events, max_subseq_events, start_size, size_step)"]
    /// Runs Forward Flux Sampling on a OldKTAM system using number of tiles as a measure, and returns
    /// a tuple of (nucleation_rate, dimerization_rate, forward_probs).
    fn ffs_run(
        system: &OldKTAM,
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
    // #[text_signature = "(system, num_states, target_size, canvas_size, max_init_events, max_subseq_events, start_size, size_step)"]
    /// Runs Forward Flux Sampling, and returns a tuple of using number of tiles as a measure, and returns
    /// (nucleation_rate, dimerization_rate, forward_probs, final configs, configs for each level).
    /// Note that this consumes a *large* amount of memory, and other functions that do not return every level's configurations may be better
    /// if you don't need all of this information.
    fn ffs_run_full<'py>(
        system: &OldKTAM,
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
    // #[text_signature = "(system, num_states, target_size, canvas_size, max_init_events, max_subseq_events, start_size, size_step)"]
    /// Runs Forward Flux Sampling, and returns a tuple of using number of tiles as a measure, and returns
    /// (nucleation_rate, dimerization_rate, forward_probs, final configs, configs for each level).
    fn ffs_run_final<'py>(
        system: &OldKTAM,
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

    // #[pyfunction]
    // // #[text_signature = "(system, num_states, target_size, canvas_size, max_init_events, max_subseq_events, start_size, size_step)"]
    // /// Runs Forward Flux Sampling for StaticKTAMCover, and returns a tuple of using number of tiles as a measure, and returns
    // /// (nucleation_rate, dimerization_rate, forward_probs, final configs).
    // fn ffs_run_final_cover<'py>(
    //     system: &StaticKTAMCover,
    //     num_states: usize,
    //     target_size: base::NumTiles,
    //     canvas_size: usize,
    //     max_init_events: u64,
    //     max_subseq_events: u64,
    //     start_size: base::NumTiles,
    //     size_step: base::NumTiles,
    //     py: Python<'py>,
    // ) -> (f64, f64, Vec<f64>, Vec<&'py PyArray2<base::Tile>>) {
    //     let fr = ffs::FFSRun::create_without_history(
    //         system.inner.to_owned(),
    //         num_states,
    //         target_size,
    //         canvas_size,
    //         max_init_events,
    //         max_subseq_events,
    //         start_size,
    //         size_step,
    //     );

    //     let assemblies = fr
    //         .level_list
    //         .last()
    //         .unwrap()
    //         .state_list
    //         .iter()
    //         .map(|state| state.canvas.raw_array().to_pyarray(py))
    //         .collect();

    //     let ret = (
    //         fr.nucleation_rate(),
    //         fr.dimerization_rate,
    //         fr.forward_vec().clone(),
    //         assemblies,
    //     );

    //     drop(fr);

    //     ret
    // }

    #[pyfunction]
    // #[text_signature = "(system, num_states, target_size, canvas_size, max_init_events, max_subseq_events, start_size, size_step)"]
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
            system.inner.as_ref().unwrap().to_owned(),
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
    // #[text_signature = "(system, varpermean2, min_states, target_size, cutoff_prob, cutoff_number, min_cutoff_size, canvas_size, max_init_events, max_subseq_events, start_size, size_step, keep_states)"]
    /// Runs Forward Flux Sampling for StaticKTAMPeriodic, and returns a tuple of using number of tiles as a measure, and returns
    /// (nucleation_rate, dimerization_rate, forward_probs, configs, num states, num trials, size, prev_list).
    fn ffs_run_final_p_cvar_cut<'py>(
        system: &StaticKTAMPeriodic,
        varpermean2: f64,
        min_states: usize,
        target_size: base::NumTiles,
        cutoff_prob: f64,
        cutoff_number: usize,
        min_cutoff_size: base::NumTiles,
        canvas_size: (usize, usize),
        max_init_events: u64,
        max_subseq_events: u64,
        start_size: base::NumTiles,
        size_step: base::NumTiles,
        keep_states: bool,
        min_nuc_rate: Option<f64>,
        py: Python<'py>,
    ) -> (
        f64,
        f64,
        Vec<f64>,
        Vec<Vec<&'py PyArray2<base::Tile>>>,
        Vec<usize>,
        Vec<usize>,
        Vec<u32>,
        Vec<Vec<usize>>,
    ) {
        let syscopy = system.inner.as_ref().unwrap().to_owned();

        let fr = py.allow_threads(|| {
            ffs::FFSRun::create_with_constant_variance_and_size_cutoff(
                syscopy,
                varpermean2,
                min_states,
                target_size,
                cutoff_prob,
                cutoff_number,
                min_cutoff_size,
                canvas_size,
                max_init_events,
                max_subseq_events,
                start_size,
                size_step,
                keep_states,
                min_nuc_rate,
            )
        });

        let assemblies = if keep_states {
            fr.level_list
                .iter()
                .map(|level| {
                    level
                        .state_list
                        .iter()
                        .map(|state| state.canvas.raw_array().to_pyarray(py))
                        .collect::<Vec<_>>()
                })
                .collect()
        } else {
            vec![fr
                .level_list
                .last()
                .unwrap()
                .state_list
                .iter()
                .map(|state| state.canvas.raw_array().to_pyarray(py))
                .collect()]
        };

        let ret = (
            fr.nucleation_rate(),
            fr.dimerization_rate,
            fr.forward_vec().clone(),
            assemblies,
            fr.level_list.iter().map(|x| x.num_states).collect(),
            fr.level_list.iter().map(|x| x.num_trials).collect(),
            fr.level_list.iter().map(|x| x.target_size).collect(),
            fr.level_list
                .iter()
                .map(|x| x.previous_list.clone())
                .collect(),
        );

        drop(fr);

        ret
    }

    #[pyfunction]
    // #[text_signature = "(system, varpermean2, min_states, target_size, cutoff_prob, cutoff_number, min_cutoff_size, canvas_size, max_init_events, max_subseq_events, start_size, size_step, keep_states)"]
    /// Runs Forward Flux Sampling for StaticKTAMPeriodic, and returns a tuple of using number of tiles as a measure, and returns
    /// (nucleation_rate, dimerization_rate, forward_probs, configs, num states, num trials, size, prev_list).
    fn ffs_run_final_p_cvar_cut_new<'py>(
        system: &NewKTAMPeriodic,
        varpermean2: f64,
        min_states: usize,
        target_size: base::NumTiles,
        cutoff_prob: f64,
        cutoff_number: usize,
        min_cutoff_size: base::NumTiles,
        canvas_size: (usize, usize),
        max_init_events: u64,
        max_subseq_events: u64,
        start_size: base::NumTiles,
        size_step: base::NumTiles,
        keep_states: bool,
        min_nuc_rate: Option<f64>,
        py: Python<'py>,
    ) -> (
        f64,
        f64,
        Vec<f64>,
        Vec<Vec<&'py PyArray2<base::Tile>>>,
        Vec<usize>,
        Vec<usize>,
        Vec<u32>,
        Vec<Vec<usize>>,
    ) {
        let syscopy = system.inner.to_owned();

        let fr = py.allow_threads(|| {
            ffs::FFSRun::create_with_constant_variance_and_size_cutoff(
                syscopy,
                varpermean2,
                min_states,
                target_size,
                cutoff_prob,
                cutoff_number,
                min_cutoff_size,
                canvas_size,
                max_init_events,
                max_subseq_events,
                start_size,
                size_step,
                keep_states,
                min_nuc_rate,
            )
        });

        let assemblies = if keep_states {
            fr.level_list
                .iter()
                .map(|level| {
                    level
                        .state_list
                        .iter()
                        .map(|state| state.canvas.raw_array().to_pyarray(py))
                        .collect::<Vec<_>>()
                })
                .collect()
        } else {
            vec![fr
                .level_list
                .last()
                .unwrap()
                .state_list
                .iter()
                .map(|state| state.canvas.raw_array().to_pyarray(py))
                .collect()]
        };

        let ret = (
            fr.nucleation_rate(),
            fr.dimerization_rate,
            fr.forward_vec().clone(),
            assemblies,
            fr.level_list.iter().map(|x| x.num_states).collect(),
            fr.level_list.iter().map(|x| x.num_trials).collect(),
            fr.level_list.iter().map(|x| x.target_size).collect(),
            fr.level_list
                .iter()
                .map(|x| x.previous_list.clone())
                .collect(),
        );

        drop(fr);

        ret
    }

    m.add_wrapped(wrap_pyfunction!(ffs_run))?;
    m.add_wrapped(wrap_pyfunction!(ffs_run_full))?;
    m.add_wrapped(wrap_pyfunction!(ffs_run_final))?;
    m.add_wrapped(wrap_pyfunction!(ffs_run_final_p))?;
    m.add_wrapped(wrap_pyfunction!(ffs_run_final_p_cvar_cut))?;
    m.add_wrapped(wrap_pyfunction!(ffs_run_final_p_cvar_cut_new))?;
    // m.add_wrapped(wrap_pyfunction!(ffs_run_final_cover))?;

    m.add_class::<tileset::TileSet>()?;
    m.add_class::<tileset::Tile>()?;
    m.add_class::<tileset::FFSResult>()?;

    Ok(())
}
