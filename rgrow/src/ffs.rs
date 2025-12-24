#![allow(clippy::too_many_arguments)]

use std::fmt::{Display, Formatter};
#[cfg(feature = "python")]
use std::ops::Deref;
use std::sync::{Arc, Weak};

use crate::base::{GrowError, RgrowError, StringConvError, Tile};
use crate::canvas::{CanvasPeriodic, CanvasSquare, CanvasTube, CanvasTubeDiagonals, PointSafe2};
use crate::models::ktam::KTAM;
use crate::models::oldktam::OldKTAM;
use crate::state::{MovieTracker, NullStateTracker, QuadTreeState};
use crate::system::EvolveBounds;
use crate::tileset::{CanvasType, Model, TileSet, SIZE_DEFAULT};
use crate::units::{MolarPerSecond, PerSecond};

use canvas::Canvas;
use num_traits::{Float, Num, Zero};
use polars::prelude::*;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3_polars::error::PyPolarsErr;
#[cfg(feature = "python")]
use python::PyState;
#[cfg(feature = "python")]
use ratestore::RateStore;
use serde::{Deserialize, Serialize};

use super::*;

/// Configuration data retention mode for FFS simulations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
// #[cfg_attr(feature = "python", pyclass(module = "rgrow"))]
pub enum ConfigRetentionMode {
    /// No configuration data retained (minimal memory usage).
    None,
    /// Only dataframe-compatible data retained (balanced memory usage, default).
    #[default]
    DataFrameOnly,
    /// Full state objects retained (maximum memory usage, allows full state access).
    Full,
}

// #[cfg(feature = "python")]
// #[pymethods]
// impl ConfigRetentionMode {
//     /// Create ConfigRetentionMode.None
//     #[classattr]
//     const NONE: Self = Self::None;

//     /// Create ConfigRetentionMode.DataFrameOnly
//     #[classattr]
//     const DATAFRAME_ONLY: Self = Self::DataFrameOnly;

//     /// Create ConfigRetentionMode.Full
//     #[classattr]
//     const FULL: Self = Self::Full;

//     fn __str__(&self) -> &'static str {
//         match self {
//             Self::None => "None",
//             Self::DataFrameOnly => "DataFrameOnly",
//             Self::Full => "Full",
//         }
//     }

//     fn __repr__(&self) -> String {
//         format!("ConfigRetentionMode.{}", self.__str__())
//     }
// }

impl TryFrom<&str> for ConfigRetentionMode {
    type Error = StringConvError;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value.to_lowercase().as_str() {
            "none" => Ok(ConfigRetentionMode::None),
            "dataframeonly" | "dataframe_only" | "dataframe-only" => Ok(ConfigRetentionMode::DataFrameOnly),
            "full" => Ok(ConfigRetentionMode::Full),
            _ => Err(StringConvError(format!(
                "Unknown config retention mode {value}. Valid options are \"none\", \"dataframeonly\", \"full\"."
            ))),
        }
    }
}

impl Display for ConfigRetentionMode {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigRetentionMode::None => write!(f, "none"),
            ConfigRetentionMode::DataFrameOnly => write!(f, "dataframeonly"),
            ConfigRetentionMode::Full => write!(f, "full"),
        }
    }
}

#[cfg(feature = "python")]
impl FromPyObject<'_> for ConfigRetentionMode {
    fn extract_bound(ob: &Bound<'_, PyAny>) -> PyResult<Self> {
        let s: &str = ob.extract()?;
        ConfigRetentionMode::try_from(s)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }
}

#[cfg(feature = "python")]
impl<'py> IntoPyObject<'py> for ConfigRetentionMode {
    fn into_pyobject(
        self,
        py: Python<'py>,
    ) -> std::result::Result<pyo3::Bound<'py, pyo3::PyAny>, pyo3::PyErr> {
        match self {
            ConfigRetentionMode::None => pyo3::IntoPyObjectExt::into_bound_py_any("none", py),
            ConfigRetentionMode::DataFrameOnly => {
                pyo3::IntoPyObjectExt::into_bound_py_any("dataframeonly", py)
            }
            ConfigRetentionMode::Full => pyo3::IntoPyObjectExt::into_bound_py_any("full", py),
        }
    }
    type Target = pyo3::PyAny; // the Python type
    type Output = pyo3::Bound<'py, Self::Target>; // in most cases this will be `Bound`
    type Error = pyo3::PyErr;
}

/// Extracted dataframe data for a single configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigDataFrameRow {
    pub surface_index: u64,
    pub config_index: u64,
    pub size: NumTiles,
    pub time: f64,
    pub previous_config: u64,
    pub canvas: Vec<Tile>,
    pub min_i: u64,
    pub min_j: u64,
    pub shape_i: u64,
    pub shape_j: u64,
    pub energy: f64,
    pub num_trials: u64,
    pub num_successes: u64,
}

/// Dataframe data for an FFS level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FFSLevelDataFrame {
    pub config_rows: Vec<ConfigDataFrameRow>,
    pub previous_list: Vec<usize>,
    pub p_r: f64,
    pub num_states: usize,
    pub num_trials: usize,
    pub target_size: NumTiles,
}
//use ndarray::prelude::*;
//use ndarray::Zip;
use base::NumTiles;

use ndarray::{s, ArrayView2};
#[cfg(feature = "python")]
use numpy::{PyArray2, ToPyArray};
use rand::{distr::weighted::WeightedIndex, distr::Uniform, prelude::Distribution};
use rand::{prelude::SmallRng, SeedableRng};
use rand::{rng, Rng};

#[cfg(feature = "python")]
use pyo3::exceptions::PyTypeError;

#[cfg(feature = "python")]
use self::base::RustAny;

use self::state::OrderTracker;
use self::tileset::TrackingType;
#[cfg(feature = "python")]
use numpy::PyArray1;
#[cfg(feature = "python")]
use pyo3_polars::PyDataFrame;

use state::{
    ClonableState, LastAttachTimeTracker, PrintEventTracker, StateEnum, StateStatus,
    StateWithCreate,
};

use system::{DynSystem, Orientation, System, SystemEnum};
//use std::convert::{TryFrom, TryInto};

/// Configuration options for Forward Flux Sampling (FFS) simulations.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyclass(get_all, set_all, module = "rgrow"))]
pub struct FFSRunConfig {
    /// Use constant-variance, variable-configurations-per-surface method.
    ///
    /// When true, the number of configurations generated at each surface is determined
    /// dynamically to achieve a target variance of the forward probablity relative to the mean
    /// squared (var_per_mean2). When false, exactly max_configs configurations are generated at
    /// each surface.
    pub constant_variance: bool,

    /// Target variance per mean squared for the constant-variance method.
    ///
    /// Controls the statistical precision when constant_variance is true. Lower values
    /// require more configurations but provide better statistics. Typical values are
    /// 0.01 (1% variance) to 0.1 (10% variance). Only used when constant_variance is true.
    pub var_per_mean2: f64,

    /// Minimum number of configurations to generate at each surface level.
    ///
    /// Ensures a minimum sample size even when constant_variance is true and the
    /// target variance is achieved with fewer configurations.
    pub min_configs: usize,

    /// Maximum number of configurations to generate at each surface level.
    ///
    /// When constant_variance is false, exactly this many configurations are generated.
    /// When constant_variance is true, this serves as an upper limit to prevent
    /// excessive computation when success probabilities are very low.
    pub max_configs: usize,

    /// Enable early termination when success probabilities become very high.
    ///
    /// When true, FFS will terminate early if the success probability exceeds
    /// cutoff_probability for cutoff_number consecutive surfaces, provided the
    /// structure size is at least min_cutoff_size.
    pub early_cutoff: bool,

    /// Success probability threshold for early cutoff.
    ///
    /// If early_cutoff is true and the success probability exceeds this value
    /// for cutoff_number consecutive surfaces, FFS terminates early.
    pub cutoff_probability: f64,

    /// Number of consecutive high-probability surfaces required for early cutoff.
    ///
    /// FFS terminates early only after this many consecutive surfaces exceed
    /// cutoff_probability. Prevents premature termination due to statistical
    /// fluctuations. Only used when early_cutoff is true.
    pub cutoff_number: usize,

    /// Minimum structure size required before early cutoff can occur.
    ///
    /// Prevents early termination when structures are still small, even if success
    /// probabilities are high. Ensures the simulation reaches a meaningful size
    /// before terminating. Only used when early_cutoff is true.
    pub min_cutoff_size: NumTiles,

    /// Evolution bounds for the initial dimer-to-n-mer surface, to avoid
    /// infinite simulations.
    pub init_bound: EvolveBounds,

    /// Evolution bounds for subsequent surface-to-surface transitions, to avoid
    /// infinite simulations.
    pub subseq_bound: EvolveBounds,

    /// Initial cluster size for the first FFS surface.
    ///
    /// The size (number of tiles) that defines the first surface.  Must be >=2.
    pub start_size: NumTiles,

    /// Size increment between consecutive FFS surfaces.
    ///
    /// The number of tiles by which the target size increases between consecutive
    /// surfaces.
    pub size_step: NumTiles,

    /// Configuration data retention mode for each surface.
    ///
    /// Controls what data is retained during FFS simulation:
    /// - ConfigRetentionMode.None: No configuration data retained (minimal memory)
    /// - ConfigRetentionMode.DataFrameOnly: Only dataframe-compatible data retained (default)
    /// - ConfigRetentionMode.Full: Full state objects retained (maximum memory, allows full access)
    ///
    /// For backward compatibility, bool values are also accepted:
    /// - False -> ConfigRetentionMode.None
    /// - True -> ConfigRetentionMode.Full
    pub keep_configs: ConfigRetentionMode,

    /// Minimum nucleation rate threshold for early termination.
    ///
    /// If specified, FFS terminates early when the calculated nucleation rate
    /// falls below this threshold. Useful for avoiding excessive computation
    /// when nucleation rates become negligibly small. Units: M/s.
    pub min_nuc_rate: Option<MolarPerSecond>,

    /// Canvas dimensions (width, height) for the simulation.
    ///
    /// Defines the size of the 2D lattice on which tile assembly occurs.
    /// Must be large enough to accommodate the largest expected structures.
    pub canvas_size: (usize, usize),

    /// Type of boundary conditions for the simulation canvas.
    ///
    /// Determines how the edges of the canvas are handled:
    /// - Periodic: opposite edges are connected (torus topology)
    /// - Square: finite canvas with hard boundaries
    /// - Tube: periodic in one dimension, finite in the other
    pub canvas_type: CanvasType,

    /// Type of additional data tracking during simulation.
    ///
    /// Controls what extra information is recorded during evolution:
    /// - None: no additional tracking (fastest, default)
    /// - Order: track attachment order of tiles
    /// - LastAttachTime: track when the tile at each location last attached
    /// - PrintEvent: print events as they occur (debugging)
    /// - Movie: record all events
    pub tracking: TrackingType,

    /// Target structure size for FFS termination.
    pub target_size: NumTiles,

    /// Whether to store the FFS configuration in the result.
    ///
    /// When true, the complete FFSRunConfig is saved with the results.
    pub store_ffs_config: bool,

    /// Whether to store the tile system in the result.
    pub store_system: bool,
}

impl Default for FFSRunConfig {
    fn default() -> Self {
        Self {
            constant_variance: true,
            var_per_mean2: 0.01,
            min_configs: 1000,
            max_configs: 100000,
            early_cutoff: true,
            cutoff_probability: 0.99,
            cutoff_number: 4,
            min_cutoff_size: 30,
            init_bound: EvolveBounds::default().for_time(1e7),
            subseq_bound: EvolveBounds::default().for_time(1e7),
            start_size: 3,
            size_step: 1,
            keep_configs: ConfigRetentionMode::default(),
            min_nuc_rate: None,
            canvas_size: (32, 32),
            canvas_type: CanvasType::Periodic,
            tracking: TrackingType::None,
            target_size: 100,
            store_ffs_config: true,
            store_system: false,
        }
    }
}

#[cfg(feature = "python")]
impl FFSRunConfig {
    pub fn _py_set(&mut self, k: &str, v: Bound<'_, PyAny>) -> PyResult<()> {
        match k {
            "constant_variance" => self.constant_variance = v.extract()?,
            "var_per_mean2" => self.var_per_mean2 = v.extract()?,
            "min_configs" => self.min_configs = v.extract()?,
            "max_configs" => self.max_configs = v.extract()?,
            "early_cutoff" => self.early_cutoff = v.extract()?,
            "cutoff_probability" => self.cutoff_probability = v.extract()?,
            "cutoff_number" => self.cutoff_number = v.extract()?,
            "min_cutoff_size" => self.min_cutoff_size = v.extract()?,
            "init_bound" => self.init_bound = v.extract()?,
            "subseq_bound" => self.subseq_bound = v.extract()?,
            "start_size" => self.start_size = v.extract()?,
            "size_step" => self.size_step = v.extract()?,
            "keep_configs" => {
                // Handle backward compatibility: accept bool or ConfigRetentionMode
                if let Ok(bool_val) = v.extract::<bool>() {
                    self.keep_configs = if bool_val {
                        ConfigRetentionMode::Full
                    } else {
                        ConfigRetentionMode::None
                    };
                } else {
                    self.keep_configs = v.extract()?;
                }
            }
            "min_nuc_rate" => self.min_nuc_rate = v.extract()?,
            "canvas_size" => self.canvas_size = v.extract()?,
            "target_size" => self.target_size = v.extract()?,
            "store_ffs_config" => self.store_ffs_config = v.extract()?,
            "store_system" => self.store_system = v.extract()?,
            "canvas_type" => self.canvas_type = v.extract()?,
            "tracking" => {
                // Handle both string and TrackingType enum
                if let Ok(s) = v.extract::<&str>() {
                    self.tracking = TrackingType::try_from(s).map_err(|e| {
                        PyTypeError::new_err(format!("Invalid tracking type: {}", e.0))
                    })?;
                } else {
                    self.tracking = v.extract()?;
                }
            }
            _ => {
                return Err(PyTypeError::new_err(format!(
                    "Unknown FFSRunConfig setting: {k}"
                )))
            }
        };
        Ok(())
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl FFSRunConfig {
    #[new]
    #[pyo3(signature = (
        constant_variance=None,
        var_per_mean2=None,
        min_configs=None,
        max_configs=None,
        early_cutoff=None,
        cutoff_probability=None,
        cutoff_number=None,
        min_cutoff_size=None,
        init_bound=None,
        subseq_bound=None,
        start_size=None,
        size_step=None,
        keep_configs=None,
        min_nuc_rate=None,
        canvas_size=None,
        canvas_type=None,
        tracking=None,
        target_size=None,
        store_ffs_config=None,
        store_system=None,
    ))]
    fn new(
        constant_variance: Option<bool>,
        var_per_mean2: Option<f64>,
        min_configs: Option<usize>,
        max_configs: Option<usize>,
        early_cutoff: Option<bool>,
        cutoff_probability: Option<f64>,
        cutoff_number: Option<usize>,
        min_cutoff_size: Option<NumTiles>,
        init_bound: Option<EvolveBounds>,
        subseq_bound: Option<EvolveBounds>,
        start_size: Option<NumTiles>,
        size_step: Option<NumTiles>,
        keep_configs: Option<Bound<'_, PyAny>>, // bool (backward compatibility) or ConfigRetentionMode
        min_nuc_rate: Option<f64>,
        canvas_size: Option<(usize, usize)>,
        canvas_type: Option<CanvasType>,
        tracking: Option<Bound<'_, PyAny>>, // str or TrackingType
        target_size: Option<NumTiles>,
        store_ffs_config: Option<bool>,
        store_system: Option<bool>,
    ) -> PyResult<Self> {
        let mut rc = Self::default();

        if let Some(x) = constant_variance {
            rc.constant_variance = x;
        }

        if let Some(x) = var_per_mean2 {
            rc.var_per_mean2 = x;
        }

        if let Some(x) = min_configs {
            rc.min_configs = x;
        }
        if let Some(x) = max_configs {
            rc.max_configs = x;
        }
        if let Some(x) = early_cutoff {
            rc.early_cutoff = x;
        }
        if let Some(x) = cutoff_probability {
            rc.cutoff_probability = x;
        }
        if let Some(x) = cutoff_number {
            rc.cutoff_number = x;
        }
        if let Some(x) = min_cutoff_size {
            rc.min_cutoff_size = x;
        }
        if let Some(x) = init_bound {
            rc.init_bound = x;
        }
        if let Some(x) = subseq_bound {
            rc.subseq_bound = x;
        }
        if let Some(x) = start_size {
            rc.start_size = x;
        }
        if let Some(x) = size_step {
            rc.size_step = x;
        }
        if let Some(x) = keep_configs {
            // Handle backward compatibility: accept bool or ConfigRetentionMode
            if let Ok(bool_val) = x.extract::<bool>() {
                rc.keep_configs = if bool_val {
                    ConfigRetentionMode::Full
                } else {
                    ConfigRetentionMode::None
                };
            } else if let Ok(mode) = x.extract::<ConfigRetentionMode>() {
                rc.keep_configs = mode;
            } else {
                return Err(PyTypeError::new_err(
                    "keep_configs must be bool or ConfigRetentionMode",
                ));
            }
        }

        rc.min_nuc_rate = min_nuc_rate.map(MolarPerSecond::new);

        if let Some(x) = canvas_size {
            rc.canvas_size = x;
        }
        if let Some(x) = canvas_type {
            rc.canvas_type = x;
        }
        if let Some(x) = tracking {
            // Handle both string and TrackingType enum
            if let Ok(s) = x.extract::<&str>() {
                rc.tracking = TrackingType::try_from(s)
                    .map_err(|e| PyTypeError::new_err(format!("Invalid tracking type: {}", e.0)))?;
            } else if let Ok(t) = x.extract::<TrackingType>() {
                rc.tracking = t;
            } else {
                return Err(PyTypeError::new_err("tracking must be str or TrackingType"));
            }
        }
        if let Some(x) = target_size {
            rc.target_size = x;
        }
        if let Some(x) = store_ffs_config {
            rc.store_ffs_config = x;
        }
        if let Some(x) = store_system {
            rc.store_system = x;
        }
        Ok(rc)
    }
}

impl TileSet {
    pub fn run_ffs(&self, config: &FFSRunConfig) -> Result<FFSRunResult, RgrowError> {
        let model = self.model.unwrap_or(Model::KTAM);
        let config = {
            let mut c = config.clone();
            c.canvas_size = match self.size.unwrap_or(SIZE_DEFAULT) {
                tileset::Size::Single(x) => (x, x),
                tileset::Size::Pair(p) => p,
            };
            c.canvas_type = self.canvas_type.unwrap_or(CanvasType::Periodic);
            c.tracking = self.tracking.unwrap_or(TrackingType::None);
            c
        };

        match model {
            Model::KTAM => KTAM::try_from(self)?.run_ffs(&config),
            Model::ATAM => Err(RgrowError::FFSCannotRunModel("aTAM".into())),
            Model::SDC => Err(RgrowError::FFSCannotRunModel("SDC".into())),
            Model::OldKTAM => OldKTAM::try_from(self)?.run_ffs(&config),
        }
    }
}

fn _bounded_nonzero_region_of_array<'a, T: Num>(
    arr: &'a ArrayView2<T>,
) -> (ArrayView2<'a, T>, usize, usize, usize, usize) {
    let mut mini = arr.nrows();
    let mut minj = arr.ncols();
    let mut maxi = 0;
    let mut maxj = 0;

    for ((i, j), v) in arr.indexed_iter() {
        if !(*v).is_zero() {
            if i < mini {
                mini = i;
            }
            if i > maxi {
                maxi = i;
            }
            if j < minj {
                minj = j;
            }
            if j > maxj {
                maxj = j;
            }
        }
    }

    let subarr = arr.slice(s![mini..maxi + 1, minj..maxj + 1]);

    (subarr, mini, minj, maxi, maxj)
}

fn _bounded_nonnan_region_of_array<'a, T: Float>(
    arr: &'a ArrayView2<T>,
) -> (ArrayView2<'a, T>, usize, usize, usize, usize) {
    let mut mini = arr.nrows();
    let mut minj = arr.ncols();
    let mut maxi = 0;
    let mut maxj = 0;

    for ((i, j), v) in arr.indexed_iter() {
        if !(*v).is_nan() {
            if i < mini {
                mini = i;
            }
            if i > maxi {
                maxi = i;
            }
            if j < minj {
                minj = j;
            }
            if j > maxj {
                maxj = j;
            }
        }
    }

    let subarr = arr.slice(s![mini..maxi + 1, minj..maxj + 1]);

    (subarr, mini, minj, maxi, maxj)
}

fn variance_over_mean2(num_success: usize, num_trials: usize) -> f64 {
    let ns = num_success as f64;
    let nt = num_trials as f64;
    let p = ns / nt;
    (1. - p) / (ns)
}

/// Calculates 95% upper bound on probability of success given number of trials and number of successes.
fn max_prob(num_success: usize, num_trials: usize) -> f64 {
    let z = 1.96;
    let p = num_success as f64 / num_trials as f64;
    p + z * (p * (1. - p) / num_trials as f64).sqrt()
}

/// Extract dataframe data from a state.
fn extract_dataframe_data<St: ClonableState>(
    state: &St,
    surface_index: u64,
    config_index: u64,
    previous_config: u64,
    num_trials: u64,
    num_successes: u64,
) -> ConfigDataFrameRow {
    let ss = &state.raw_array();
    let (m, mini, minj, maxi, maxj) = _bounded_nonzero_region_of_array(ss);

    ConfigDataFrameRow {
        surface_index,
        config_index,
        size: state.n_tiles(),
        time: state.time().into(),
        previous_config,
        canvas: m.iter().copied().collect(),
        min_i: mini as u64,
        min_j: minj as u64,
        shape_i: (maxi - mini + 1) as u64,
        shape_j: (maxj - minj + 1) as u64,
        energy: state.energy(),
        num_trials,
        num_successes,
    }
}

/// Extract dataframe data from a StateEnum.
fn _extract_dataframe_data_from_state_enum(
    state: &StateEnum,
    surface_index: u64,
    config_index: u64,
    previous_config: u64,
    num_trials: u64,
    num_successes: u64,
) -> ConfigDataFrameRow {
    let ss = &state.raw_array();
    let (m, mini, minj, maxi, maxj) = _bounded_nonzero_region_of_array(ss);

    ConfigDataFrameRow {
        surface_index,
        config_index,
        size: state.n_tiles(),
        time: state.time().into(),
        previous_config,
        canvas: m.iter().copied().collect(),
        min_i: mini as u64,
        min_j: minj as u64,
        shape_i: (maxi - mini + 1) as u64,
        shape_j: (maxj - minj + 1) as u64,
        energy: state.energy(),
        num_trials,
        num_successes,
    }
}

pub struct FFSRun<St: ClonableState> {
    pub level_list: Vec<FFSLevel<St>>,
    pub dimerization_rate: MolarPerSecond,
    pub forward_prob: Vec<f64>,
}

impl<St: ClonableState + StateWithCreate<Params = (usize, usize)>> FFSRun<St> {
    pub fn create<Sy: System>(system: &mut Sy, config: &FFSRunConfig) -> Result<Self, GrowError> {
        let level_list = Vec::new();

        let dimerization_rate: MolarPerSecond = system
            .calc_dimers()?
            .iter()
            .fold(MolarPerSecond::zero(), |acc, d| acc + d.formation_rate);

        let mut ret = Self {
            level_list,
            dimerization_rate,
            forward_prob: Vec::new(),
        };

        let (first_level, mut dimer_level) = FFSLevel::nmers_from_dimers(system, config)?;

        ret.forward_prob.push(first_level.p_r);

        let mut current_size = first_level.target_size;

        match config.keep_configs {
            ConfigRetentionMode::None => {
                dimer_level.drop_states();
            }
            ConfigRetentionMode::DataFrameOnly => {
                dimer_level.drop_states_keep_dataframe();
            }
            ConfigRetentionMode::Full => {
                // Keep everything
            }
        }

        ret.level_list.push(dimer_level);
        ret.level_list.push(first_level);

        let mut above_cutoff: usize = 0;

        while current_size < config.target_size {
            let min_prob = config
                .min_nuc_rate
                .map(|min_nuc_rate| min_nuc_rate / ret.nucleation_rate());

            let surface_index = ret.level_list.len() as u64;
            let last = ret.level_list.last_mut().unwrap();
            let next = last.next_level(system, config, min_prob, surface_index)?;
            match config.keep_configs {
                ConfigRetentionMode::None => {
                    last.drop_states();
                }
                ConfigRetentionMode::DataFrameOnly => {
                    last.drop_states_keep_dataframe();
                }
                ConfigRetentionMode::Full => {
                    // Keep everything
                }
            }
            let pf = next.p_r;
            ret.forward_prob.push(pf);

            current_size = next.target_size;
            ret.level_list.push(next);

            if config.early_cutoff {
                if pf > config.cutoff_probability {
                    above_cutoff += 1;
                    if (above_cutoff > config.cutoff_number)
                        & (current_size >= config.min_cutoff_size)
                    {
                        break;
                    }
                } else {
                    above_cutoff = 0;
                }
            }

            if let Some(min_nuc_rate) = config.min_nuc_rate {
                if ret.nucleation_rate() < min_nuc_rate {
                    break;
                }
            }
        }

        Ok(ret)
    }
    pub fn dimer_conc(&self) -> f64 {
        self.level_list[0].p_r
    }
}

impl<St: ClonableState + StateWithCreate<Params = (usize, usize)>> FFSRun<St> {
    pub fn create_from_tileset<'a, Sy: System + TryFrom<&'a TileSet, Error = RgrowError>>(
        tileset: &'a TileSet,
        config: &FFSRunConfig,
    ) -> Result<Self, RgrowError> {
        let mut sys = Sy::try_from(tileset)?;
        let c = {
            let mut c = config.clone();
            c.canvas_size = match tileset.size.unwrap_or(SIZE_DEFAULT) {
                tileset::Size::Single(x) => (x, x),
                tileset::Size::Pair(p) => p,
            };
            c
        };

        Ok(Self::create(&mut sys, &c)?)
    }
}

pub struct FFSLevel<St: ClonableState> {
    pub state_list: Vec<St>,
    pub previous_list: Vec<usize>,
    pub p_r: f64,
    pub num_states: usize,
    pub num_trials: usize,
    pub last_surface_num_trials: Vec<usize>,
    pub last_surface_num_successes: Vec<usize>,
    pub target_size: NumTiles,
    pub dataframe_data: Option<Vec<ConfigDataFrameRow>>,
}

impl<St: ClonableState + StateWithCreate<Params = (usize, usize)>> FFSLevel<St> {
    pub fn drop_states(&mut self) -> &Self {
        drop(self.state_list.drain(..));
        drop(self.previous_list.drain(..));
        self
    }

    pub fn drop_states_keep_dataframe(&mut self) -> &Self {
        drop(self.state_list.drain(..));
        self
    }

    pub fn next_level<Sy: System>(
        &self,
        system: &mut Sy,
        config: &FFSRunConfig,
        min_prob: Option<f64>,
        surface_index: u64,
    ) -> Result<Self, GrowError> {
        let mut rng = rng();

        let mut state_list = Vec::new();
        let mut previous_list = Vec::new();
        let mut dataframe_data = if config.keep_configs == ConfigRetentionMode::DataFrameOnly {
            Some(Vec::new())
        } else {
            None
        };
        let mut i = 0usize;
        let target_size = self.target_size + config.size_step;

        // Track trials and successes per configuration from previous level
        let mut config_trials = vec![0; self.state_list.len()];
        let mut config_successes = vec![0; self.state_list.len()];

        let bounds = {
            let mut b = config.subseq_bound;
            b.size_max = Some(target_size);
            b.size_min = Some(0);
            b
        };

        let chooser = Uniform::new(0, self.state_list.len()).unwrap(); // FIXME: handle error

        let canvas_size = self.state_list[0].get_params();

        let cvar = if config.constant_variance {
            config.var_per_mean2
        } else {
            0.
        };

        while state_list.len() < config.max_configs {
            let mut state = St::empty(canvas_size)?;

            let mut i_old_state: usize = 0;

            while state.n_tiles() == 0 {
                if state.total_rate() != PerSecond::zero() {
                    panic!("Total rate is not zero! {state:?}");
                };
                i_old_state = chooser.sample(&mut rng);

                system.clone_state_into_empty_state(&self.state_list[i_old_state], &mut state);
                debug_assert_eq!(system.calc_n_tiles(&state), state.n_tiles());

                system.evolve(&mut state, bounds).unwrap();
                i += 1;
            }

            // Increment trial count for the selected configuration
            config_trials[i_old_state] += 1;

            if state.n_tiles() >= target_size {
                // >= hack for duples

                // Increment success count for the selected configuration
                config_successes[i_old_state] += 1;

                // Extract dataframe data if needed
                if let Some(ref mut df_data) = dataframe_data {
                    let df_row = extract_dataframe_data(
                        &state,
                        surface_index,
                        state_list.len() as u64,
                        i_old_state as u64,
                        0, // empty for now, will be filled in later
                        0, // empty for now, will be filled in later
                    );
                    df_data.push(df_row);
                }

                state_list.push(state);
                previous_list.push(i_old_state);
            } else {
                println!(
                    "Ran out of events: {} tiles, {} events, {} time, {} total rate.",
                    state.n_tiles(),
                    state.total_events(),
                    state.time(),
                    state.total_rate(),
                );
            }

            if (variance_over_mean2(state_list.len(), i) < cvar)
                & (state_list.len() >= config.min_configs)
            {
                break;
            }

            if let Some(min_prob) = min_prob {
                if max_prob(state_list.len(), i) < min_prob {
                    break;
                }
            }
        }
        let p_r = (state_list.len() as f64) / (i as f64);
        let num_states = state_list.len();

        Ok(Self {
            state_list,
            previous_list,
            p_r,
            target_size,
            num_states,
            num_trials: i,
            dataframe_data,
            last_surface_num_trials: config_trials,
            last_surface_num_successes: config_successes,
        })
    }

    pub fn nmers_from_dimers<Sy: System>(
        system: &mut Sy,
        config: &FFSRunConfig,
    ) -> Result<(Self, Self), GrowError> {
        let mut rng = SmallRng::from_os_rng();

        let dimers = system.calc_dimers()?;

        let mut state_list = Vec::with_capacity(config.min_configs);
        let mut previous_list = Vec::with_capacity(config.min_configs);
        let mut dataframe_data = if config.keep_configs == ConfigRetentionMode::DataFrameOnly {
            Some(Vec::new())
        } else {
            None
        };
        let mut dimer_dataframe_data = if config.keep_configs == ConfigRetentionMode::DataFrameOnly
        {
            Some(Vec::new())
        } else {
            None
        };
        let mut i = 0usize;

        let mut dimer_state_list = Vec::with_capacity(config.min_configs);

        let weights: Vec<_> = dimers.iter().map(|d| f64::from(d.formation_rate)).collect();
        let chooser = WeightedIndex::new(weights).unwrap();

        if config.canvas_size.0 < 4 || config.canvas_size.1 < 4 {
            panic!("Canvas size too small for dimers");
        }
        let mid = PointSafe2((config.canvas_size.0 / 2, config.canvas_size.1 / 2));

        let mut num_states = 0usize;

        let mut tile_list = Vec::with_capacity(config.min_configs);

        let mut other: PointSafe2;

        let cvar = if config.constant_variance {
            config.var_per_mean2
        } else {
            0.
        };

        let min_prob = if let Some(min_nuc_rate) = config.min_nuc_rate {
            let dimerization_rate = dimers
                .iter()
                .fold(MolarPerSecond::zero(), |acc, d| acc + d.formation_rate);
            min_nuc_rate / dimerization_rate
        } else {
            0.
        };

        let bounds = {
            let mut b = config.subseq_bound;
            b.size_max = Some(config.start_size);
            b.size_min = Some(0);
            b
        };

        while state_list.len() < config.max_configs {
            let mut state = St::empty(config.canvas_size)?;

            while state.n_tiles() == 0 {
                let i_old_state = chooser.sample(&mut rng);
                let dimer = &dimers[i_old_state];
                state.reset_tracking_assuming_empty_state();

                other = match dimer.orientation {
                    Orientation::NS => PointSafe2(state.move_sa_s(mid).0),
                    Orientation::WE => PointSafe2(state.move_sa_e(mid).0),
                };
                // Use place_tile to properly handle double tiles
                let energy_change = system.place_tile(&mut state, mid, dimer.t1)?
                    + system.place_tile(&mut state, other, dimer.t2)?;
                let cl = [(mid, dimer.t1), (other, dimer.t2)];
                state.record_event(
                    &system::Event::PolymerAttachment(cl.to_vec()),
                    PerSecond::zero(),
                    f64::NAN,
                    energy_change,
                    energy_change,
                    2,
                );

                debug_assert_eq!(system.calc_n_tiles(&state), state.n_tiles());

                system.evolve(&mut state, bounds).unwrap();
                i += 1;

                if state.n_tiles() >= config.start_size {
                    // FIXME: >= is a hack
                    // Create (retrospectively) a dimer state
                    let mut dimer_state = St::empty(config.canvas_size)?;

                    // Use place_tile to properly handle double tiles for dimer state too
                    let energy_change = system.place_tile(&mut dimer_state, mid, dimer.t1)?
                        + system.place_tile(&mut dimer_state, other, dimer.t2)?;
                    dimer_state.record_event(
                        &system::Event::PolymerAttachment(cl.to_vec()),
                        PerSecond::zero(),
                        f64::NAN,
                        energy_change,
                        energy_change,
                        2,
                    );

                    // Extract dataframe data if needed
                    if let Some(ref mut df_data) = dataframe_data {
                        let df_row = extract_dataframe_data(
                            &state,
                            1, // surface index 1 for first level
                            num_states as u64,
                            num_states as u64,
                            1, // Each successful configuration had 1 trial
                            1, // Each successful configuration had 1 success
                        );
                        df_data.push(df_row);
                    }

                    if let Some(ref mut dimer_df_data) = dimer_dataframe_data {
                        let dimer_df_row = extract_dataframe_data(
                            &dimer_state,
                            0, // surface index 0 for dimer level
                            num_states as u64,
                            if rng.random::<bool>() {
                                dimer.t1 as u64
                            } else {
                                dimer.t2 as u64
                            },
                            1, // Each dimer configuration had 1 trial
                            1, // Each dimer configuration had 1 success
                        );
                        dimer_df_data.push(dimer_df_row);
                    }

                    state_list.push(state);
                    dimer_state_list.push(dimer_state);

                    if rng.random::<bool>() {
                        tile_list.push(dimer.t1);
                    } else {
                        tile_list.push(dimer.t2);
                    }

                    previous_list.push(num_states);

                    num_states += 1;

                    break;
                } else {
                    if state.n_tiles() != 0 {
                        panic!("{}", state.panicinfo())
                    }
                    if state.total_rate() != PerSecond::zero() {
                        panic!("{}", state.panicinfo())
                    };
                }
            }

            if (variance_over_mean2(num_states, i) < cvar) & (num_states >= config.min_configs) {
                break;
            }

            if max_prob(num_states, i) < min_prob {
                break;
            }
        }

        let p_r = (num_states as f64) / (i as f64);
        let n_dimers = dimer_state_list.len();

        Ok((
            Self {
                state_list,
                previous_list,
                p_r,
                target_size: config.start_size,
                num_states,
                num_trials: i,
                dataframe_data,
                last_surface_num_trials: vec![1; n_dimers],
                last_surface_num_successes: vec![1; n_dimers],
            },
            Self {
                state_list: dimer_state_list,
                previous_list: tile_list.into_iter().map(|x| x as usize).collect(),
                p_r: 1.0,
                target_size: 2,
                num_states,
                num_trials: num_states,
                dataframe_data: dimer_dataframe_data,
                last_surface_num_trials: Vec::default(),
                last_surface_num_successes: Vec::default(),
            },
        ))
    }
}

// RESULTS CODE

#[cfg_attr(feature = "python", pyclass(module = "rgrow"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FFSRunResult {
    #[serde(skip)]
    pub level_list: Vec<Arc<FFSLevelResult>>,
    pub dimerization_rate: MolarPerSecond,
    pub forward_prob: Vec<f64>,
    pub ffs_config: Option<FFSRunConfig>,
    #[serde(skip)]
    pub system: Option<SystemEnum>,
}

#[cfg_attr(feature = "python", pyclass(module = "rgrow"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FFSRunResultDF {
    #[serde(skip)]
    pub surfaces_df: DataFrame,
    #[serde(skip)]
    pub configs_df: DataFrame,
    pub ffs_config: Option<FFSRunConfig>,
    pub system: Option<SystemEnum>,
    pub dimerization_rate: MolarPerSecond,
}

impl From<FFSRunResult> for FFSRunResultDF {
    fn from(value: FFSRunResult) -> Self {
        let surfaces_df = value.surfaces_dataframe().unwrap();
        let configs_df = value.configs_dataframe().unwrap();
        Self {
            surfaces_df,
            configs_df,
            ffs_config: value.ffs_config,
            system: value.system,
            dimerization_rate: value.dimerization_rate,
        }
    }
}

impl FFSRunResultDF {
    pub fn read_files(prefix: &str) -> Result<Self, PolarsError> {
        let file = std::fs::File::open(format!("{prefix}.surfaces.parquet"))?;
        let surfaces_df = ParquetReader::new(file).finish()?;
        let file = std::fs::File::open(format!("{prefix}.configurations.parquet"))?;
        let configs_df = ParquetReader::new(file).finish()?;
        let file = std::fs::File::open(format!("{prefix}.ffs_result.json"))?;
        let ffs_result: FFSRunResultDF = serde_json::from_reader(file).unwrap();
        Ok(Self {
            surfaces_df,
            configs_df,
            ffs_config: ffs_result.ffs_config,
            system: ffs_result.system,
            dimerization_rate: ffs_result.dimerization_rate,
        })
    }

    pub fn write_files(&mut self, prefix: &str) -> Result<(), PolarsError> {
        let file = std::fs::File::create(format!("{prefix}.surfaces.parquet"))?;
        ParquetWriter::new(file).finish(&mut self.surfaces_df)?;
        let file = std::fs::File::create(format!("{prefix}.configurations.parquet"))?;
        ParquetWriter::new(file).finish(&mut self.configs_df)?;
        let file = std::fs::File::create(format!("{prefix}.ffs_result.json"))?;
        serde_json::to_writer(file, &self).unwrap();
        Ok(())
    }

    pub fn forward_vec(&self) -> Vec<f64> {
        let mut it = self
            .surfaces_df
            .column("p_r")
            .unwrap()
            .f64()
            .unwrap()
            .into_iter();

        // first value is just 1.0 (dimers)
        it.next();

        it.map(|x| x.unwrap()).collect()
    }

    pub fn nucleation_rate(&self) -> MolarPerSecond {
        let ptot: f64 = self
            .surfaces_df
            .column("p_r")
            .unwrap()
            .product()
            .unwrap()
            .as_any_value()
            .try_extract()
            .unwrap();

        self.dimerization_rate * ptot
    }
}

impl<St: ClonableState> From<FFSRun<St>> for FFSRunResult
where
    FFSLevelResult: From<FFSLevel<St>>,
{
    fn from(value: FFSRun<St>) -> Self {
        Self {
            level_list: value
                .level_list
                .into_iter()
                .map(|x| Arc::new(x.into()))
                .collect(),
            dimerization_rate: value.dimerization_rate,
            forward_prob: value.forward_prob,
            ffs_config: None,
            system: None,
        }
    }
}

#[cfg_attr(feature = "python", pyclass(module = "rgrow"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FFSLevelResult {
    pub state_list: Vec<Arc<StateEnum>>,
    pub previous_list: Vec<usize>,
    pub p_r: f64,
    pub num_states: usize,
    pub num_trials: usize,
    pub target_size: NumTiles,
    pub dataframe_data: Option<Vec<ConfigDataFrameRow>>,
    pub last_surface_num_trials: Vec<usize>,
    pub last_surface_num_successes: Vec<usize>,
}

impl<St: ClonableState> From<FFSLevel<St>> for FFSLevelResult
where
    StateEnum: From<St>,
{
    fn from(value: FFSLevel<St>) -> Self {
        Self {
            state_list: value
                .state_list
                .into_iter()
                .map(|x| Arc::new(x.into()))
                .collect(),
            previous_list: value.previous_list,
            p_r: value.p_r,
            num_states: value.num_states,
            num_trials: value.num_trials,
            target_size: value.target_size,
            dataframe_data: value.dataframe_data,
            last_surface_num_trials: value.last_surface_num_trials,
            last_surface_num_successes: value.last_surface_num_successes,
        }
    }
}

pub trait FFSSurface: Send + Sync {
    fn get_config(&self, i: usize) -> ArrayView2<'_, Tile>;
    fn get_state(&self, i: usize) -> Weak<StateEnum>;
    fn states(&self) -> Vec<Weak<StateEnum>> {
        (0..self.num_stored_states())
            .map(|i| self.get_state(i))
            .collect()
    }
    fn configs(&self) -> Vec<ArrayView2<'_, Tile>> {
        (0..self.num_stored_states())
            .map(|i| self.get_config(i))
            .collect()
    }
    fn previous_list(&self) -> Vec<usize>;
    fn num_stored_states(&self) -> usize;
    fn num_configs(&self) -> usize;
    fn num_trials(&self) -> usize;
    fn target_size(&self) -> NumTiles;
    fn p_r(&self) -> f64;
}

impl<St: ClonableState> FFSRun<St> {
    fn nucleation_rate(&self) -> MolarPerSecond {
        self.dimerization_rate * self.forward_prob.iter().fold(1., |acc, level| acc * *level)
    }
}

impl FFSRunResult {
    pub fn nucleation_rate(&self) -> MolarPerSecond {
        self.dimerization_rate * self.forward_prob.iter().fold(1., |acc, level| acc * *level)
    }

    pub fn forward_vec(&self) -> &Vec<f64> {
        &self.forward_prob
    }

    pub fn surfaces(&self) -> Vec<Weak<FFSLevelResult>> {
        self.level_list.iter().map(Arc::downgrade).collect()
    }

    pub fn get_surface(&self, i: usize) -> Option<Weak<FFSLevelResult>> {
        self.level_list.get(i).map(Arc::downgrade)
    }

    pub fn dimerization_rate(&self) -> MolarPerSecond {
        self.dimerization_rate
    }

    fn surfaces_dataframe(&self) -> Result<DataFrame, PolarsError> {
        let surfaces = self.surfaces();

        let d = df!(
            "level" => 0..surfaces.len() as u64,
            "n_configs" => surfaces.iter().map(|x| (*x.upgrade().unwrap()).num_configs() as u64).collect::<Vec<u64>>(),
            "n_trials" => surfaces.iter().map(|x| (*x.upgrade().unwrap()).num_trials() as u64).collect::<Vec<u64>>(),
            "target_size" => surfaces.iter().map(|x| (*x.upgrade().unwrap()).target_size() as u64).collect::<Vec<u64>>(),
            "p_r" => surfaces.iter().map(|x| (*x.upgrade().unwrap()).p_r()).collect::<Vec<f64>>(),
        )
        .unwrap();

        Ok(d)
    }

    fn configs_dataframe(&self) -> Result<DataFrame, PolarsError> {
        let mut sizes = Vec::new();
        let mut times: Vec<f64> = Vec::new();
        let mut previndices = Vec::new();
        let mut canvases = Vec::new();
        let mut arr_mini = Vec::new();
        let mut arr_minj = Vec::new();
        let mut shape_i = Vec::new();
        let mut shape_j = Vec::new();
        let mut surfaceindex = Vec::new();
        let mut configindex = Vec::new();
        let mut energies = Vec::new();
        let mut num_trials = Vec::new();
        let mut num_successes = Vec::new();

        // Check if we have pre-compiled dataframe data
        let has_dataframe_data = self
            .surfaces()
            .iter()
            .any(|s| s.upgrade().unwrap().dataframe_data.is_some());

        if has_dataframe_data {
            // Use pre-compiled dataframe data
            for surface in self.surfaces().iter() {
                let surface_ref: Arc<FFSLevelResult> = surface.upgrade().unwrap();
                if let Some(ref df_data) = surface_ref.dataframe_data {
                    for row in df_data {
                        sizes.push(row.size);
                        times.push(row.time);
                        canvases.push(Series::from_vec("canvas".into(), row.canvas.clone()));
                        surfaceindex.push(row.surface_index);
                        configindex.push(row.config_index);
                        arr_mini.push(row.min_i);
                        arr_minj.push(row.min_j);
                        shape_i.push(row.shape_i);
                        shape_j.push(row.shape_j);
                        previndices.push(row.previous_config);
                        energies.push(row.energy);
                    }
                    num_trials.extend(
                        surface_ref
                            .last_surface_num_trials
                            .iter()
                            .map(|x| *x as u64),
                    );
                    num_successes.extend(
                        surface_ref
                            .last_surface_num_successes
                            .iter()
                            .map(|x| *x as u64),
                    );
                }
            }
        } else {
            // Fall back to extracting from states
            for (i, surface) in self.surfaces().iter().enumerate() {
                for (j, state) in surface.upgrade().unwrap().state_list.iter().enumerate() {
                    sizes.push(state.n_tiles());
                    times.push(state.time().into());
                    let ss = &state.raw_array();
                    let (m, mini, minj, maxi, maxj) = _bounded_nonzero_region_of_array(ss);
                    canvases.push(m.iter().collect::<Series>());
                    surfaceindex.push(i as u64);
                    configindex.push(j as u64);
                    arr_mini.push(mini as u64);
                    arr_minj.push(minj as u64);
                    shape_i.push((maxi - mini + 1) as u64);
                    shape_j.push((maxj - minj + 1) as u64);
                    energies.push(state.energy());
                }
                num_trials.extend(
                    surface
                        .upgrade()
                        .unwrap()
                        .last_surface_num_trials
                        .iter()
                        .map(|x| *x as u64),
                );
                num_successes.extend(
                    surface
                        .upgrade()
                        .unwrap()
                        .last_surface_num_successes
                        .iter()
                        .map(|x| *x as u64),
                );
                if !surface.upgrade().unwrap().state_list.is_empty() {
                    previndices.extend(
                        surface
                            .upgrade()
                            .unwrap()
                            .previous_list()
                            .iter()
                            .map(|x| *x as u64),
                    );
                }
            }
        }

        // We need to add 0 trials, 0 successes for the last surface
        let last_surface_num_configs = self
            .surfaces()
            .last()
            .unwrap()
            .upgrade()
            .unwrap()
            .num_configs();
        num_trials.extend(vec![0u64; last_surface_num_configs]);
        num_successes.extend(vec![0u64; last_surface_num_configs]);

        let df = df!(
            "surface_index" => surfaceindex,
            "config_index" => configindex,
            "size" => sizes,
            "time" => times,
            "previous_config" => previndices,
            "canvas" => canvases,
            "min_i" => arr_mini,
            "min_j" => arr_minj,
            "shape_i" => shape_i,
            "shape_j" => shape_j,
            "energy" => energies,
            "num_trials" => num_trials,
            "num_successes" => num_successes,
        )
        .unwrap();

        Ok(df)
    }

    pub fn write_files(&self, prefix: &str) -> Result<(), PolarsError> {
        let mut sdf = self.surfaces_dataframe()?;
        let mut cdf = self.configs_dataframe()?;

        let file = std::fs::File::create(format!("{prefix}.surfaces.parquet"))?;
        ParquetWriter::new(file).finish(&mut sdf)?;

        let file = std::fs::File::create(format!("{prefix}.configurations.parquet"))?;
        ParquetWriter::new(file).finish(&mut cdf)?;

        let file = std::fs::File::create(format!("{prefix}.ffs_result.json"))?;
        serde_json::to_writer_pretty(file, self).unwrap();

        Ok(())
    }

    pub fn run_from_system<Sy: System>(
        sys: &mut Sy,
        config: &FFSRunConfig,
    ) -> Result<FFSRunResult, RgrowError>
    where
        SystemEnum: From<Sy>,
    {
        let mut res: FFSRunResult = (match (config.canvas_type, config.tracking) {
            (CanvasType::Square, TrackingType::None) => {
                FFSRun::<QuadTreeState<CanvasSquare, NullStateTracker>>::create(sys, config)
                    .map(|x| x.into())
            }
            (CanvasType::Square, TrackingType::Order) => {
                FFSRun::<QuadTreeState<CanvasSquare, OrderTracker>>::create(sys, config)
                    .map(|x| x.into())
            }
            (CanvasType::Square, TrackingType::LastAttachTime) => {
                FFSRun::<QuadTreeState<CanvasSquare, LastAttachTimeTracker>>::create(sys, config)
                    .map(|x| x.into())
            }
            (CanvasType::Square, TrackingType::PrintEvent) => {
                FFSRun::<QuadTreeState<CanvasSquare, PrintEventTracker>>::create(sys, config)
                    .map(|x| x.into())
            }
            (CanvasType::Square, TrackingType::Movie) => {
                FFSRun::<QuadTreeState<CanvasSquare, MovieTracker>>::create(sys, config)
                    .map(|x| x.into())
            }
            (CanvasType::Periodic, TrackingType::None) => {
                FFSRun::<QuadTreeState<CanvasPeriodic, NullStateTracker>>::create(sys, config)
                    .map(|x| x.into())
            }
            (CanvasType::Periodic, TrackingType::Order) => {
                FFSRun::<QuadTreeState<CanvasPeriodic, OrderTracker>>::create(sys, config)
                    .map(|x| x.into())
            }
            (CanvasType::Periodic, TrackingType::LastAttachTime) => {
                FFSRun::<QuadTreeState<CanvasPeriodic, LastAttachTimeTracker>>::create(sys, config)
                    .map(|x| x.into())
            }
            (CanvasType::Periodic, TrackingType::PrintEvent) => {
                FFSRun::<QuadTreeState<CanvasPeriodic, PrintEventTracker>>::create(sys, config)
                    .map(|x| x.into())
            }
            (CanvasType::Periodic, TrackingType::Movie) => {
                FFSRun::<QuadTreeState<CanvasPeriodic, MovieTracker>>::create(sys, config)
                    .map(|x| x.into())
            }
            (CanvasType::Tube, TrackingType::None) => {
                FFSRun::<QuadTreeState<CanvasTube, NullStateTracker>>::create(sys, config)
                    .map(|x| x.into())
            }
            (CanvasType::Tube, TrackingType::Order) => {
                FFSRun::<QuadTreeState<CanvasTube, OrderTracker>>::create(sys, config)
                    .map(|x| x.into())
            }
            (CanvasType::Tube, TrackingType::LastAttachTime) => {
                FFSRun::<QuadTreeState<CanvasTube, LastAttachTimeTracker>>::create(sys, config)
                    .map(|x| x.into())
            }
            (CanvasType::Tube, TrackingType::PrintEvent) => {
                FFSRun::<QuadTreeState<CanvasTube, PrintEventTracker>>::create(sys, config)
                    .map(|x| x.into())
            }
            (CanvasType::Tube, TrackingType::Movie) => {
                FFSRun::<QuadTreeState<CanvasTube, MovieTracker>>::create(sys, config)
                    .map(|x| x.into())
            }
            (CanvasType::TubeDiagonals, TrackingType::None) => {
                FFSRun::<QuadTreeState<CanvasTubeDiagonals, NullStateTracker>>::create(sys, config)
                    .map(|x| x.into())
            }
            (CanvasType::TubeDiagonals, TrackingType::Order) => {
                FFSRun::<QuadTreeState<CanvasTubeDiagonals, OrderTracker>>::create(sys, config)
                    .map(|x| x.into())
            }
            (CanvasType::TubeDiagonals, TrackingType::LastAttachTime) => {
                FFSRun::<QuadTreeState<CanvasTubeDiagonals, LastAttachTimeTracker>>::create(
                    sys, config,
                )
                .map(|x| x.into())
            }
            (CanvasType::TubeDiagonals, TrackingType::PrintEvent) => {
                FFSRun::<QuadTreeState<CanvasTubeDiagonals, PrintEventTracker>>::create(sys, config)
                    .map(|x| x.into())
            }
            (CanvasType::TubeDiagonals, TrackingType::Movie) => {
                FFSRun::<QuadTreeState<CanvasTubeDiagonals, MovieTracker>>::create(sys, config)
                    .map(|x| x.into())
            }
        })?;

        if config.store_ffs_config {
            res.ffs_config = Some(config.clone());
        }
        if config.store_system {
            res.system = Some(sys.clone().into());
        }

        Ok(res)
    }
}

impl FFSLevelResult {
    pub fn get_config(&self, i: usize) -> ArrayView2<'_, Tile> {
        self.state_list[i].raw_array()
    }

    pub fn get_state(&self, i: usize) -> Option<Weak<StateEnum>> {
        self.state_list.get(i).map(Arc::downgrade)
    }

    pub fn num_configs(&self) -> usize {
        self.num_states
    }

    pub fn target_size(&self) -> NumTiles {
        self.target_size
    }

    pub fn num_trials(&self) -> usize {
        self.num_trials
    }

    pub fn previous_list(&self) -> Vec<usize> {
        self.previous_list.clone()
    }

    pub fn p_r(&self) -> f64 {
        self.p_r
    }

    pub fn num_stored_states(&self) -> usize {
        self.state_list.len()
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl FFSRunResult {
    /// float: Nucleation rate, in M/s.  Calculated from the forward probability vector,
    /// and dimerization rate.
    #[getter]
    fn get_nucleation_rate(&self) -> f64 {
        self.nucleation_rate().into()
    }

    /// list[float]: Forward probability vector.
    #[getter]
    fn get_forward_vec<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.forward_vec().to_pyarray(py)
    }

    /// float: Dimerization rate, in M/s.
    #[getter]
    fn get_dimerization_rate(&self) -> f64 {
        self.dimerization_rate().into()
    }

    /// list[FFSLevelRef]: list of surfaces.
    #[getter]
    fn get_surfaces(&self) -> Vec<FFSLevelRef> {
        self.level_list
            .iter()
            .map(|f| FFSLevelRef(Arc::downgrade(f)))
            .collect()
    }

    /// Returns
    /// -------
    /// pl.DataFrame
    #[pyo3(name = "surfaces_dataframe")]
    fn py_surfaces_dataframe(&self) -> PyResult<pyo3_polars::PyDataFrame> {
        self.surfaces_dataframe()
            .map(PyDataFrame)
            .map_err(|e| PyPolarsErr::from(e).into())
    }

    /// Returns
    /// -------
    /// pl.DataFrame
    #[pyo3(name = "configs_dataframe")]
    fn py_configs_dataframe(&self) -> PyResult<pyo3_polars::PyDataFrame> {
        self.configs_dataframe()
            .map(PyDataFrame)
            .map_err(|e| PyPolarsErr::from(e).into())
    }

    /// Returns
    /// -------
    /// pl.DataFrame
    #[pyo3(name = "surfaces_to_polars")]
    fn py_surfaces_to_polars(&self) -> PyResult<pyo3_polars::PyDataFrame> {
        self.surfaces_dataframe()
            .map(PyDataFrame)
            .map_err(|e| PyPolarsErr::from(e).into())
    }

    /// Returns
    /// -------
    /// pl.DataFrame
    #[pyo3(name = "states_to_polars")]
    fn py_states_to_polars(&self) -> PyResult<pyo3_polars::PyDataFrame> {
        self.configs_dataframe()
            .map(PyDataFrame)
            .map_err(|e| PyPolarsErr::from(e).into())
    }

    fn __str__(&self) -> String {
        format!(
            "FFSResult({:1.4e} M/s, {:?})",
            f64::from(self.nucleation_rate()),
            self.forward_vec()
        )
    }

    fn __repr__(&self) -> String {
        self.__str__()
    }

    #[getter]
    fn previous_indices(&self) -> Vec<Vec<usize>> {
        self.get_surfaces()
            .iter()
            .map(|x| x.get_previous_indices())
            .collect()
    }

    #[pyo3(name = "write_files")]
    fn py_write_files(&self, prefix: &str) -> PyResult<()> {
        self.write_files(prefix)
            .map_err(|e| PyPolarsErr::from(e).into())
    }

    fn into_resdf(this: Bound<FFSRunResult>) -> FFSRunResultDF {
        this.borrow_mut().clone().into()
    }

    // #[staticmethod]
    // fn read_json(filename: &str) -> PyResult<Self> {
    //     let f = std::fs::File::open(filename)?;
    //     let r: Self = serde_json::from_reader(f).unwrap();
    //     Ok(r)
    // }
}

#[cfg(feature = "python")]
#[pymethods]
impl FFSRunResultDF {
    /// float: Nucleation rate, in M/s.  Calculated from the forward probability vector,
    /// and dimerization rate.
    #[getter]
    fn get_nucleation_rate(&self) -> f64 {
        self.nucleation_rate().into()
    }

    /// list[float]: Forward probability vector.
    #[getter]
    fn get_forward_vec<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.forward_vec().to_pyarray(py)
    }

    /// float: Dimerization rate, in M/s.
    #[getter]
    fn get_dimerization_rate(&self) -> f64 {
        self.dimerization_rate.into()
    }

    // #[getter]
    // fn get_surfaces(&self) -> Vec<FFSLevelRef> {
    //     self.level_list
    //         .iter()
    //         .map(|f| FFSLevelRef(Arc::downgrade(f)))
    //         .collect()
    // }

    /// Get the surfaces as a Polars DataFrame.
    ///
    /// Returns
    /// -------
    /// pl.DataFrame
    #[pyo3(name = "surfaces_dataframe")]
    fn py_surfaces_dataframe(&self) -> PyResult<pyo3_polars::PyDataFrame> {
        Ok(PyDataFrame(self.surfaces_df.clone()))
    }

    /// Get the configurations as a Polars DataFrame.
    ///
    /// Returns
    /// -------
    /// pl.DataFrame
    #[pyo3(name = "configs_dataframe")]
    fn py_configs_dataframe(&self) -> PyResult<pyo3_polars::PyDataFrame> {
        Ok(PyDataFrame(self.configs_df.clone()))
    }

    fn __str__(&self) -> String {
        format!(
            "FFSResultDF({:1.4e} M/s, {:?})",
            f64::from(self.nucleation_rate()),
            self.forward_vec()
        )
    }

    fn __repr__(&self) -> String {
        self.__str__()
    }

    // #[getter]
    // fn previous_indices(&self) -> Vec<Vec<usize>> {
    //     self.get_surfaces()
    //         .iter()
    //         .map(|x| x.get_previous_indices())
    //         .collect()
    // }

    /// Write dataframes and result data to files.
    ///
    /// Parameters
    /// ----------
    /// prefix : str
    ///    Prefix for the filenames.  The files will be named
    ///    `{prefix}.surfaces.parquet`, `{prefix}.configurations.parquet`, and
    ///    `{prefix}.ffs_result.json`.
    #[pyo3(name = "write_files")]
    fn py_write_files(&mut self, prefix: &str) -> PyResult<()> {
        self.write_files(prefix)
            .map_err(|e| PyPolarsErr::from(e).into())
    }

    /// Read dataframes and result data from files.
    ///
    /// Returns
    /// -------
    /// Self
    #[pyo3(name = "read_files")]
    #[staticmethod]
    fn py_read_files(prefix: &str) -> PyResult<Self> {
        FFSRunResultDF::read_files(prefix).map_err(|e| PyPolarsErr::from(e).into())
    }

    // #[staticmethod]
    // fn read_json(filename: &str) -> PyResult<Self> {
    //     let f = std::fs::File::open(filename)?;
    //     let r: Self = serde_json::from_reader(f).unwrap();
    //     Ok(r)
    // }
}

#[cfg_attr(feature = "python", pyclass(module = "rgrow"))]
#[allow(dead_code)] // This is used in the python interface
pub struct FFSLevelRef(Weak<FFSLevelResult>);

#[cfg(feature = "python")]
#[pymethods]
impl FFSLevelRef {
    #[getter]
    fn get_configs<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyArray2<crate::base::Tile>>> {
        let level = self.0.upgrade().expect("FFSLevelResult has been dropped");
        level
            .state_list
            .iter()
            .map(|x| x.raw_array().to_pyarray(py))
            .collect()
    }

    #[getter]
    fn get_states(&self) -> Vec<FFSStateRef> {
        let level = self.0.upgrade().expect("FFSLevelResult has been dropped");
        level
            .state_list
            .iter()
            .map(|x| FFSStateRef(Arc::downgrade(x)))
            .collect()
    }

    #[getter]
    fn get_previous_indices(&self) -> Vec<usize> {
        let level = self.0.upgrade().expect("FFSLevelResult has been dropped");
        level.previous_list.clone()
    }

    // #[getter]
    // fn level(&self) -> usize {
    //     (*self.0).level
    // }

    fn get_state(&self, i: usize) -> FFSStateRef {
        let level = self.0.upgrade().expect("FFSLevelResult has been dropped");
        FFSStateRef(Arc::downgrade(&level.state_list[i]))
    }

    fn has_stored_states(&self) -> bool {
        let level = self.0.upgrade().expect("FFSLevelResult has been dropped");
        !level.state_list.is_empty()
    }

    fn __repr__(&self) -> String {
        let level = self.0.upgrade().expect("FFSLevelResult has been dropped");
        format!(
            "FFSLevelRef(n_configs={}, n_trials={}, target_size={}, p_r={}, has_stored_states={})",
            level.num_configs(),
            level.num_trials(),
            level.target_size(),
            level.p_r(),
            self.has_stored_states()
        )
    }
}

#[cfg_attr(feature = "python", pyclass(module = "rgrow"))]
#[allow(dead_code)] // This is used in the python interface
#[derive(Clone)]
pub struct FFSStateRef(Weak<StateEnum>);

#[cfg(feature = "python")]
#[pymethods]
impl FFSStateRef {
    /// float: the total time the state has simulated, in seconds.
    #[getter]
    pub fn time(&self) -> f64 {
        self.0
            .upgrade()
            .expect("StateEnum has been dropped")
            .time()
            .into()
    }

    /// int: the total number of events that have occurred in the state.
    #[getter]
    pub fn total_events(&self) -> base::NumEvents {
        self.0
            .upgrade()
            .expect("StateEnum has been dropped")
            .total_events()
    }

    /// int: the number of tiles in the state.
    #[getter]
    pub fn n_tiles(&self) -> NumTiles {
        self.0
            .upgrade()
            .expect("StateEnum has been dropped")
            .n_tiles()
    }

    /// Return a copy of the state behind the reference as a mutable `State` object.
    ///
    /// Returns
    /// -------
    /// State
    pub fn clone_state(&self) -> PyState {
        PyState(
            self.0
                .upgrade()
                .expect("StateEnum has been dropped")
                .as_ref()
                .clone(),
        )
    }

    #[getter]
    /// NDArray[np.uint]: a direct, mutable view of the state's canvas.  This is potentially unsafe.
    pub fn canvas_view<'py>(
        this: Bound<'py, Self>,
        _py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray2<crate::base::Tile>>> {
        let t = this.borrow();
        let state = t.0.upgrade().expect("StateEnum has been dropped");
        let ra = state.raw_array();

        unsafe { Ok(PyArray2::borrow_from_array(&ra, this.into_any())) }
    }

    /// Create a copy of the state's canvas.  This is safe, but can't be modified and is slower than
    /// `canvas_view`.
    ///
    /// Returns
    /// -------
    /// NDArray[np.uint]
    ///     a copy of the state's canvas.
    pub fn canvas_copy<'py>(
        this: &Bound<Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray2<crate::base::Tile>>> {
        let t = this.borrow();
        let state = t.0.upgrade().expect("StateEnum has been dropped");
        let ra = state.raw_array();

        Ok(PyArray2::from_array(py, &ra))
    }

    /// Return a copy of the tracker's tracking data.
    ///
    /// Returns
    /// -------
    /// Any
    pub fn tracking_copy(this: &Bound<Self>) -> PyResult<RustAny> {
        use crate::state::TrackerData;

        let t = this.borrow();
        let state = t.0.upgrade().expect("StateEnum has been dropped");
        let ra = state.get_tracker_data();

        Ok(ra)
    }

    pub fn __repr__(&self) -> String {
        let state = self.0.upgrade().expect("StateEnum has been dropped");
        format!(
            "FFSStateRef(n_tiles={}, time={} s, events={}, size=({}, {}), total_rate={})",
            state.n_tiles(),
            state.time(),
            state.total_events(),
            state.ncols(),
            state.nrows(),
            state.total_rate()
        )
    }

    /// Return a cloned copy of an array with the total possible next event rate for each point in the canvas.
    /// This is the deepest level of the quadtree for tree-based states.
    ///
    /// Returns
    /// -------
    /// NDArray[np.uint]
    pub fn rate_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let state = self.0.upgrade().expect("StateEnum has been dropped");
        state.rate_array().mapv(f64::from).to_pyarray(py)
    }

    /// float: the total rate of possible next events for the state.
    #[getter]
    pub fn total_rate(&self) -> f64 {
        let state = self.0.upgrade().expect("StateEnum has been dropped");
        RateStore::total_rate(state.deref()).into()
    }
}
