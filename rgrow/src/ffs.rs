#![allow(clippy::too_many_arguments)]

#[cfg(feature = "python")]
use std::ops::Deref;
use std::sync::{Arc, Weak};

use crate::base::{GrowError, RgrowError, Tile};
use crate::canvas::{CanvasPeriodic, CanvasSquare, CanvasTube, PointSafe2};
use crate::models::ktam::KTAM;
use crate::models::oldktam::OldKTAM;
use crate::state::{NullStateTracker, QuadTreeState};
use crate::system::{EvolveBounds, SystemWithDimers};
use crate::tileset::{CanvasType, FromTileSet, Model, TileSet, SIZE_DEFAULT};

use canvas::Canvas;
use num_traits::{Float, Num};
use polars::prelude::*;
#[cfg(feature = "python")]
use pyo3_polars::error::PyPolarsErr;
#[cfg(feature = "python")]
use python::PyState;
#[cfg(feature = "python")]
use ratestore::RateStore;
use serde::{Deserialize, Serialize};

use super::*;
//use ndarray::prelude::*;
//use ndarray::Zip;
use base::{NumTiles, Rate};

use ndarray::{s, Array2, ArrayView2};
#[cfg(feature = "python")]
use numpy::{PyArray2, ToPyArray};
use rand::{distributions::Uniform, distributions::WeightedIndex, prelude::Distribution};
use rand::{prelude::SmallRng, SeedableRng};
use rand::{thread_rng, Rng};

#[cfg(feature = "python")]
use pyo3::exceptions::PyTypeError;
#[cfg(feature = "python")]
use pyo3::prelude::*;

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
    StateWithCreate, TrackerData,
};

use system::{DynSystem, Orientation, System, SystemEnum};
//use std::convert::{TryFrom, TryInto};

/// Configuration options for FFS.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyclass(get_all, set_all, module = "rgrow"))]
pub struct FFSRunConfig {
    /// Use constant-variance, variable-configurations-per-surface method.
    /// If false, use max_configs for each surface.
    pub constant_variance: bool,
    /// Variance per mean^2 for constant-variance method.
    pub var_per_mean2: f64,
    /// Minimum number of configuratons to generate at each level.
    pub min_configs: usize,
    /// Maximum number of configurations to generate at each level.
    pub max_configs: usize,
    /// Use early cutoff for constant-variance method.
    pub early_cutoff: bool,
    pub cutoff_probability: f64,
    pub cutoff_number: usize,
    pub min_cutoff_size: NumTiles,
    pub init_bound: EvolveBounds,
    pub subseq_bound: EvolveBounds,
    pub start_size: NumTiles,
    pub size_step: NumTiles,
    pub keep_configs: bool,
    pub min_nuc_rate: Option<Rate>,
    pub canvas_size: (usize, usize),
    pub canvas_type: CanvasType,
    pub tracking: TrackingType,
    pub target_size: NumTiles,
    pub store_ffs_config: bool,
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
            keep_configs: false,
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
            "keep_configs" => self.keep_configs = v.extract()?,
            "min_nuc_rate" => self.min_nuc_rate = v.extract()?,
            "canvas_size" => self.canvas_size = v.extract()?,
            "target_size" => self.target_size = v.extract()?,
            "store_ffs_config" => self.store_ffs_config = v.extract()?,
            "store_system" => self.store_system = v.extract()?,
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
        keep_configs: Option<bool>,
        min_nuc_rate: Option<Rate>,
        canvas_size: Option<(usize, usize)>,
        target_size: Option<NumTiles>,
        store_ffs_config: Option<bool>,
        store_system: Option<bool>,
    ) -> Self {
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
            rc.keep_configs = x;
        }

        rc.min_nuc_rate = min_nuc_rate;

        if let Some(x) = canvas_size {
            rc.canvas_size = x;
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
        rc
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
            Model::KTAM => KTAM::from_tileset(self)?.run_ffs(&config),
            Model::ATAM => Err(RgrowError::FFSCannotRunModel("aTAM".into())),
            Model::SDC => Err(RgrowError::FFSCannotRunModel("SDC".into())),
            Model::OldKTAM => OldKTAM::from_tileset(self)?.run_ffs(&config),
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

pub struct FFSRun<St: ClonableState> {
    pub level_list: Vec<FFSLevel<St>>,
    pub dimerization_rate: f64,
    pub forward_prob: Vec<f64>,
}

impl<St: ClonableState + StateWithCreate<Params = (usize, usize)>> FFSRun<St> {
    pub fn create<Sy: SystemWithDimers + System>(
        system: &mut Sy,
        config: &FFSRunConfig,
    ) -> Result<Self, GrowError> {
        let level_list = Vec::new();

        let dimerization_rate = system
            .calc_dimers()
            .iter()
            .fold(0., |acc, d| acc + d.formation_rate);

        let mut ret = Self {
            level_list,
            dimerization_rate,
            forward_prob: Vec::new(),
        };

        let (first_level, mut dimer_level) = FFSLevel::nmers_from_dimers(system, config)?;

        ret.forward_prob.push(first_level.p_r);

        let mut current_size = first_level.target_size;

        if !config.keep_configs {
            dimer_level.drop_states();
        }

        ret.level_list.push(dimer_level);
        ret.level_list.push(first_level);

        let mut above_cutoff: usize = 0;

        while current_size < config.target_size {
            let last = ret.level_list.last_mut().unwrap();

            let next = last.next_level(system, config)?;
            if !config.keep_configs {
                last.drop_states();
            }
            let pf = next.p_r;
            ret.forward_prob.push(pf);
            // println!(
            //     "Done with target size {}: p_f {}, used {} trials for {} states.",
            //     last.target_size, pf, next.num_trials, next.num_states
            // );
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
    pub fn create_from_tileset<Sy: SystemWithDimers + System + FromTileSet>(
        tileset: &TileSet,
        config: &FFSRunConfig,
    ) -> Result<Self, RgrowError> {
        let mut sys = Sy::from_tileset(tileset)?;
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
    pub target_size: NumTiles,
}

impl<St: ClonableState + StateWithCreate<Params = (usize, usize)>> FFSLevel<St> {
    pub fn drop_states(&mut self) -> &Self {
        drop(self.state_list.drain(..));
        drop(self.previous_list.drain(..));
        self
    }

    pub fn next_level<Sy: SystemWithDimers + System>(
        &self,
        system: &mut Sy,
        config: &FFSRunConfig,
    ) -> Result<Self, GrowError> {
        let mut rng = thread_rng();

        let mut state_list = Vec::new();
        let mut previous_list = Vec::new();
        let mut i = 0usize;
        let target_size = self.target_size + config.size_step;

        let bounds = {
            let mut b = config.subseq_bound;
            b.size_max = Some(target_size);
            b.size_min = Some(0);
            b
        };

        let chooser = Uniform::new(0, self.state_list.len());

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
                if state.total_rate() != 0. {
                    panic!("Total rate is not zero! {state:?}");
                };
                i_old_state = chooser.sample(&mut rng);

                state.zeroed_copy_from_state_nonzero_rate(&self.state_list[i_old_state]);
                debug_assert_eq!(system.calc_n_tiles(&state), state.n_tiles());

                system.evolve(&mut state, bounds).unwrap();
                i += 1;
            }

            if state.n_tiles() >= target_size {
                // >= hack for duples
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
        })
    }

    pub fn nmers_from_dimers<Sy: SystemWithDimers + System>(
        system: &mut Sy,
        config: &FFSRunConfig,
    ) -> Result<(Self, Self), GrowError> {
        let mut rng = SmallRng::from_entropy();

        let dimers = system.calc_dimers();

        let mut state_list = Vec::with_capacity(config.min_configs);
        let mut previous_list = Vec::with_capacity(config.min_configs);
        let mut i = 0usize;

        let mut dimer_state_list = Vec::with_capacity(config.min_configs);

        let weights: Vec<_> = dimers.iter().map(|d| d.formation_rate).collect();
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
                let cl = [(mid, dimer.t1), (other, dimer.t2)];
                system.set_safe_points(&mut state, &cl);
                state.record_event(&system::Event::PolymerAttachment(cl.to_vec()));

                debug_assert_eq!(system.calc_n_tiles(&state), state.n_tiles());

                system.evolve(&mut state, bounds).unwrap();
                i += 1;

                if state.n_tiles() >= config.start_size {
                    // FIXME: >= is a hack
                    // Create (retrospectively) a dimer state
                    let mut dimer_state = St::empty(config.canvas_size)?;

                    system.set_safe_points(&mut dimer_state, &cl);
                    dimer_state.record_event(&system::Event::PolymerAttachment(cl.to_vec()));

                    state_list.push(state);

                    dimer_state_list.push(dimer_state);

                    if rng.gen::<bool>() {
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
                    if state.total_rate() != 0. {
                        panic!("{}", state.panicinfo())
                    };
                }
            }

            if (variance_over_mean2(num_states, i) < cvar) & (num_states >= config.min_configs) {
                break;
            }
        }

        let p_r = (num_states as f64) / (i as f64);

        Ok((
            Self {
                state_list,
                previous_list,
                p_r,
                target_size: config.start_size,
                num_states,
                num_trials: i,
            },
            Self {
                state_list: dimer_state_list,
                previous_list: tile_list.into_iter().map(|x| x as usize).collect(),
                p_r: 1.0,
                target_size: 2,
                num_states,
                num_trials: num_states,
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
    pub dimerization_rate: f64,
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
    pub dimerization_rate: f64,
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
        let file = std::fs::File::open(format!("{}.surfaces.parquet", prefix))?;
        let surfaces_df = ParquetReader::new(file).finish()?;
        let file = std::fs::File::open(format!("{}.configurations.parquet", prefix))?;
        let configs_df = ParquetReader::new(file).finish()?;
        let file = std::fs::File::open(format!("{}.ffs_result.json", prefix))?;
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
        let file = std::fs::File::create(format!("{}.surfaces.parquet", prefix))?;
        ParquetWriter::new(file).finish(&mut self.surfaces_df)?;
        let file = std::fs::File::create(format!("{}.configurations.parquet", prefix))?;
        ParquetWriter::new(file).finish(&mut self.configs_df)?;
        let file = std::fs::File::create(format!("{}.ffs_result.json", prefix))?;
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

    pub fn nucleation_rate(&self) -> Rate {
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
        }
    }
}

pub trait FFSSurface: Send + Sync {
    fn get_config(&self, i: usize) -> ArrayView2<Tile>;
    fn get_state(&self, i: usize) -> Arc<StateEnum>;
    fn states(&self) -> Vec<Arc<StateEnum>> {
        (0..self.num_stored_states())
            .map(|i| self.get_state(i))
            .collect()
    }
    fn configs(&self) -> Vec<ArrayView2<Tile>> {
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
    fn nucleation_rate(&self) -> Rate {
        self.dimerization_rate * self.forward_prob.iter().fold(1., |acc, level| acc * *level)
    }
}

impl FFSRunResult {
    pub fn nucleation_rate(&self) -> Rate {
        self.dimerization_rate * self.forward_prob.iter().fold(1., |acc, level| acc * *level)
    }

    pub fn forward_vec(&self) -> &Vec<f64> {
        &self.forward_prob
    }

    pub fn surfaces(&self) -> Vec<Weak<FFSLevelResult>> {
        self.level_list.iter().map(Arc::downgrade).collect()
    }

    pub fn get_surface(&self, i: usize) -> Option<Arc<FFSLevelResult>> {
        self.level_list.get(i).map(|x| (*x).clone())
    }

    pub fn dimerization_rate(&self) -> f64 {
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
        let mut times = Vec::new();
        let mut previndices = Vec::new();
        let mut canvases = Vec::new();
        let mut arr_mini = Vec::new();
        let mut arr_minj = Vec::new();
        let mut shape_i = Vec::new();
        let mut shape_j = Vec::new();
        let mut surfaceindex = Vec::new();
        let mut configindex = Vec::new();

        for (i, surface) in self.surfaces().iter().enumerate() {
            for (j, state) in surface.upgrade().unwrap().state_list.iter().enumerate() {
                sizes.push(state.n_tiles());
                times.push(state.time());
                let ss = &state.raw_array();
                let (m, mini, minj, maxi, maxj) = _bounded_nonzero_region_of_array(ss);
                canvases.push(m.iter().collect::<Series>());
                surfaceindex.push(i as u64);
                configindex.push(j as u64);
                arr_mini.push(mini as u64);
                arr_minj.push(minj as u64);
                shape_i.push((maxi - mini + 1) as u64);
                shape_j.push((maxj - minj + 1) as u64);
            }
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

        let mut df = df!(
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
        )
        .unwrap();

        let s = self
            .surfaces()
            .last()
            .unwrap()
            .upgrade()
            .unwrap()
            .get_state(0)
            .unwrap();

        let a = s.get_tracker_data();

        if a.0.downcast_ref::<Array2<u64>>().is_some() {
            let mut arrs = Vec::new();
            let mut minis = Vec::new();
            let mut minjs = Vec::new();
            let mut shapeis = Vec::new();
            let mut shapejs = Vec::new();

            for surface in self.surfaces().iter() {
                for state in surface.upgrade().unwrap().state_list.iter() {
                    if let Some(val) = state.get_tracker_data().0.downcast_ref::<Array2<u64>>() {
                        let v = &val.view();
                        let (m, mini, minj, maxi, maxj) = _bounded_nonzero_region_of_array(v);
                        arrs.push(m.iter().collect::<Series>());
                        minis.push(mini as u64);
                        minjs.push(minj as u64);
                        shapeis.push((maxi - mini + 1) as u64);
                        shapejs.push((maxj - minj + 1) as u64);
                    } else {
                        panic!()
                    }
                }
            }

            df = df
                .lazy()
                .with_columns([
                    Series::new("tracker", arrs).lit(),
                    Series::new("tracker_min_i", minis).lit(),
                    Series::new("tracker_min_j", minjs).lit(),
                    Series::new("tracker_shape_i", shapeis).lit(),
                    Series::new("tracker_shape_j", shapejs).lit(),
                ])
                .collect()
                .unwrap();
        } else if a.0.downcast_ref::<Array2<f64>>().is_some() {
            let mut arrs = Vec::new();
            let mut minis = Vec::new();
            let mut minjs = Vec::new();
            let mut shapeis = Vec::new();
            let mut shapejs = Vec::new();

            for surface in self.surfaces().iter() {
                for state in surface.upgrade().unwrap().state_list.iter() {
                    if let Some(val) = state.get_tracker_data().0.downcast_ref::<Array2<f64>>() {
                        let v = &val.view();
                        let (m, mini, minj, maxi, maxj) = _bounded_nonnan_region_of_array(v);
                        arrs.push(m.iter().collect::<Series>());
                        minis.push(mini as u64);
                        minjs.push(minj as u64);
                        shapeis.push((maxi - mini + 1) as u64);
                        shapejs.push((maxj - minj + 1) as u64);
                    } else {
                        panic!()
                    }
                }
            }

            df = df
                .lazy()
                .with_columns([
                    Series::new("tracker", arrs).lit(),
                    Series::new("tracker_min_i", minis).lit(),
                    Series::new("tracker_min_j", minjs).lit(),
                    Series::new("tracker_shape_i", shapeis).lit(),
                    Series::new("tracker_shape_j", shapejs).lit(),
                ])
                .collect()
                .unwrap();
        } else if a.0.downcast_ref::<()>().is_some() {
        } else {
            println!("Unknown tracker data type: skipping tracker data.")
        }

        Ok(df)
    }

    pub fn write_files(&self, prefix: &str) -> Result<(), PolarsError> {
        let mut sdf = self.surfaces_dataframe()?;
        let mut cdf = self.configs_dataframe()?;

        let file = std::fs::File::create(format!("{}.surfaces.parquet", prefix))?;
        ParquetWriter::new(file).finish(&mut sdf)?;

        let file = std::fs::File::create(format!("{}.configurations.parquet", prefix))?;
        ParquetWriter::new(file).finish(&mut cdf)?;

        let file = std::fs::File::create(format!("{}.ffs_result.json", prefix))?;
        serde_json::to_writer_pretty(file, self).unwrap();

        Ok(())
    }

    pub fn run_from_system<Sy: SystemWithDimers + System>(
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
    pub fn get_config(&self, i: usize) -> ArrayView2<Tile> {
        self.state_list[i].raw_array()
    }

    pub fn get_state(&self, i: usize) -> Option<Arc<StateEnum>> {
        self.state_list.get(i).map(|x| (*x).clone())
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
        self.nucleation_rate()
    }

    /// list[float]: Forward probability vector.
    #[getter]
    fn get_forward_vec<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.forward_vec().to_pyarray_bound(py)
    }

    /// float: Dimerization rate, in M/s.
    #[getter]
    fn get_dimerization_rate(&self) -> f64 {
        self.dimerization_rate()
    }

    /// list[FFSLevelRef]: list of surfaces.
    #[getter]
    fn get_surfaces(&self) -> Vec<FFSLevelRef> {
        self.surfaces()
            .iter()
            .map(|f| FFSLevelRef(f.upgrade().unwrap().clone()))
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
            self.nucleation_rate(),
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
        self.nucleation_rate()
    }

    /// list[float]: Forward probability vector.
    #[getter]
    fn get_forward_vec<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.forward_vec().to_pyarray_bound(py)
    }

    /// float: Dimerization rate, in M/s.
    #[getter]
    fn get_dimerization_rate(&self) -> f64 {
        self.dimerization_rate
    }

    // #[getter]
    // fn get_surfaces(&self) -> Vec<FFSLevelRef> {
    //     self.surfaces()
    //         .iter()
    //         .map(|f| FFSLevelRef(f.upgrade().unwrap().clone()))
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
            self.nucleation_rate(),
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
pub struct FFSLevelRef(Arc<FFSLevelResult>);

#[cfg(feature = "python")]
#[pymethods]
impl FFSLevelRef {
    #[getter]
    fn get_configs<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyArray2<crate::base::Tile>>> {
        self.0
            .state_list
            .iter()
            .map(|x| x.raw_array().to_pyarray_bound(py))
            .collect()
    }

    #[getter]
    fn get_states(&self) -> Vec<FFSStateRef> {
        self.0
            .state_list
            .iter()
            .map(|x| FFSStateRef(x.clone()))
            .collect()
    }

    #[getter]
    fn get_previous_indices(&self) -> Vec<usize> {
        self.0.previous_list.clone()
    }

    // #[getter]
    // fn level(&self) -> usize {
    //     (*self.0).level
    // }

    fn get_state(&self, i: usize) -> FFSStateRef {
        FFSStateRef(self.0.state_list[i].clone())
    }

    fn has_stored_states(&self) -> bool {
        !self.0.state_list.is_empty()
    }

    fn __repr__(&self) -> String {
        format!(
            "FFSLevelRef(n_configs={}, n_trials={}, target_size={}, p_r={}, has_stored_states={})",
            self.0.num_configs(),
            self.0.num_trials(),
            self.0.target_size(),
            self.0.p_r(),
            self.has_stored_states()
        )
    }
}

#[cfg_attr(feature = "python", pyclass(module = "rgrow"))]
#[allow(dead_code)] // This is used in the python interface
#[derive(Clone)]
pub struct FFSStateRef(Arc<StateEnum>);

#[cfg(feature = "python")]
#[pymethods]
impl FFSStateRef {
    /// float: the total time the state has simulated, in seconds.
    #[getter]
    pub fn time(&self) -> f64 {
        (*self.0).time()
    }

    /// int: the total number of events that have occurred in the state.
    #[getter]
    pub fn total_events(&self) -> base::NumEvents {
        (*self.0).total_events()
    }

    /// int: the number of tiles in the state.
    #[getter]
    pub fn n_tiles(&self) -> NumTiles {
        (*self.0).n_tiles()
    }

    /// Return a copy of the state behind the reference as a mutable `State` object.
    ///
    /// Returns
    /// -------
    /// State
    pub fn clone_state(&self) -> PyState {
        PyState((*self.0).clone())
    }

    #[getter]
    /// NDArray[np.uint]: a direct, mutable view of the state's canvas.  This is potentially unsafe.
    pub fn canvas_view<'py>(
        this: Bound<'py, Self>,
        _py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray2<crate::base::Tile>>> {
        let t = this.borrow();
        let ra = (*t.0).raw_array();

        unsafe { Ok(PyArray2::borrow_from_array_bound(&ra, this.into_any())) }
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
        let ra = (*t.0).raw_array();

        Ok(PyArray2::from_array_bound(py, &ra))
    }

    /// Return a copy of the tracker's tracking data.
    ///
    /// Returns
    /// -------
    /// Any
    pub fn tracking_copy(this: &Bound<Self>) -> PyResult<RustAny> {
        let t = this.borrow();
        let ra = (*t.0).get_tracker_data();

        Ok(ra)
    }

    pub fn __repr__(&self) -> String {
        format!(
            "FFSStateRef(n_tiles={}, time={} s, events={}, size=({}, {}), total_rate={})",
            self.0.n_tiles(),
            self.0.time(),
            self.0.total_events(),
            self.0.ncols(),
            self.0.nrows(),
            self.0.total_rate()
        )
    }

    /// Return a cloned copy of an array with the total possible next event rate for each point in the canvas.
    /// This is the deepest level of the quadtree for tree-based states.
    ///
    /// Returns
    /// -------
    /// NDArray[np.uint]
    pub fn rate_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<crate::base::Rate>> {
        self.0.rate_array().to_pyarray_bound(py)
    }

    /// float: the total rate of possible next events for the state.
    #[getter]
    pub fn total_rate(&self) -> crate::base::Rate {
        RateStore::total_rate(self.0.deref())
    }
}
