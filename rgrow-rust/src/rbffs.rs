//! An implementation of Rosenbluth-style Forward Flux Sampling
//!
//! Based on the explanation in R. J. Allen, C. Valeriani, and P. R. ten Wolde,
//! "Forward flux sampling for rare event simulations," J. Phys.: Condens. Matter,
//! vol. 21, no. 46, p. 463102, Oct. 2009, doi: 10.1088/0953-8984/21/46/463102.

use std::sync::Arc;

use num_traits::Zero;
use rand::distr::{weighted::WeightedIndex, Distribution};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[cfg(feature = "python")]
use ndarray::Array2;
#[cfg(feature = "python")]
use numpy::{PyArray1, PyArray2, ToPyArray};
#[cfg(feature = "python")]
use pyo3::exceptions::PyTypeError;
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
use crate::ffs::FFSStateRef;

use crate::{
    base::{GrowError, NumTiles, RgrowError},
    canvas::{
        CanvasPeriodic, CanvasSquare, CanvasSquareCompact, CanvasTube, CanvasTubeDiagonals,
        PointSafe2,
    },
    models::ktam::KTAM,
    models::oldktam::OldKTAM,
    state::{
        ClonableState, LastAttachTimeTracker, MovieTracker, NullStateTracker, OrderTracker,
        PrintEventTracker, QuadTreeState, StateEnum, StateWithCreate,
    },
    system::{self, DimerInfo, EvolveBounds, EvolveOutcome, Orientation, System, SystemEnum},
    tileset::{CanvasType, Model, Size, TileSet, TrackingType, SIZE_DEFAULT},
    units::{MolarPerSecond, PerSecond},
};

/// Configuration for Rosenbluth-style Forward Flux Sampling.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyclass(get_all, set_all, module = "rgrow.rgrow"))]
pub struct RBFFSRunConfig {
    /// Number of trials per surface per trajectory.
    pub n_trials: usize,
    /// Desired number of complete trajectories.
    pub n_trajectories: usize,
    /// Target assembly size to reach.
    pub target_size: NumTiles,
    /// Canvas dimensions (rows, cols).
    pub canvas_size: (usize, usize),
    /// Evolution bounds for each surface-to-surface transition.
    pub subseq_bound: EvolveBounds,
    /// Canvas boundary type.
    pub canvas_type: CanvasType,
    /// State tracking type.
    pub tracking: TrackingType,
    /// If true (default), store the state at every surface crossing.
    /// If false, store only the final state of each trajectory (saves memory with heavy trackers).
    pub keep_full_trajectories: bool,
    /// If true, store the system in the result so that `extend()` can run more trajectories.
    /// Default false to avoid overhead for heavy systems.
    pub store_system: bool,
    /// Size increment between consecutive FFS surfaces.
    /// Default 1 (surfaces at every tile count). Higher values mean fewer, coarser surfaces.
    pub size_step: NumTiles,
    /// If true, run trajectories in parallel using rayon.
    pub parallel: bool,
    /// Number of worker threads for parallel execution. None = use rayon default (all cores).
    pub num_workers: Option<usize>,
}

impl Default for RBFFSRunConfig {
    fn default() -> Self {
        Self {
            n_trials: 1000,
            n_trajectories: 1000,
            target_size: 100,
            canvas_size: (32, 32),
            subseq_bound: EvolveBounds::default().for_time(1e7),
            canvas_type: CanvasType::Periodic,
            tracking: TrackingType::None,
            keep_full_trajectories: true,
            store_system: false,
            size_step: 1,
            parallel: false,
            num_workers: None,
        }
    }
}

#[cfg(feature = "python")]
impl RBFFSRunConfig {
    pub fn _py_set(&mut self, k: &str, v: Bound<'_, PyAny>) -> PyResult<()> {
        match k {
            "n_trials" => self.n_trials = v.extract()?,
            "n_trajectories" => self.n_trajectories = v.extract()?,
            "target_size" => self.target_size = v.extract()?,
            "canvas_size" => self.canvas_size = v.extract()?,
            "subseq_bound" => self.subseq_bound = v.extract()?,
            "canvas_type" => self.canvas_type = v.extract()?,
            "tracking" => {
                if let Ok(s) = v.extract::<&str>() {
                    self.tracking = TrackingType::try_from(s).map_err(|e| {
                        PyTypeError::new_err(format!("Invalid tracking type: {}", e.0))
                    })?;
                } else {
                    self.tracking = v.extract()?;
                }
            }
            "keep_full_trajectories" => self.keep_full_trajectories = v.extract()?,
            "store_system" => self.store_system = v.extract()?,
            "size_step" => self.size_step = v.extract()?,
            "parallel" => self.parallel = v.extract()?,
            "num_workers" => self.num_workers = v.extract()?,
            _ => {
                return Err(PyTypeError::new_err(format!(
                    "Unknown RBFFSRunConfig setting: {k}"
                )))
            }
        };
        Ok(())
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl RBFFSRunConfig {
    #[new]
    #[pyo3(signature = (
        n_trials=None,
        n_trajectories=None,
        target_size=None,
        canvas_size=None,
        subseq_bound=None,
        canvas_type=None,
        tracking=None,
        keep_full_trajectories=None,
        store_system=None,
        size_step=None,
        parallel=None,
        num_workers=None,
    ))]
    fn new(
        n_trials: Option<usize>,
        n_trajectories: Option<usize>,
        target_size: Option<NumTiles>,
        canvas_size: Option<(usize, usize)>,
        subseq_bound: Option<EvolveBounds>,
        canvas_type: Option<CanvasType>,
        tracking: Option<Bound<'_, PyAny>>,
        keep_full_trajectories: Option<bool>,
        store_system: Option<bool>,
        size_step: Option<NumTiles>,
        parallel: Option<bool>,
        num_workers: Option<usize>,
    ) -> PyResult<Self> {
        let mut rc = Self::default();
        if let Some(x) = n_trials {
            rc.n_trials = x;
        }
        if let Some(x) = n_trajectories {
            rc.n_trajectories = x;
        }
        if let Some(x) = target_size {
            rc.target_size = x;
        }
        if let Some(x) = canvas_size {
            rc.canvas_size = x;
        }
        if let Some(x) = subseq_bound {
            rc.subseq_bound = x;
        }
        if let Some(x) = canvas_type {
            rc.canvas_type = x;
        }
        if let Some(x) = tracking {
            if let Ok(s) = x.extract::<&str>() {
                rc.tracking = TrackingType::try_from(s)
                    .map_err(|e| PyTypeError::new_err(format!("Invalid tracking type: {}", e.0)))?;
            } else if let Ok(t) = x.extract::<TrackingType>() {
                rc.tracking = t;
            } else {
                return Err(PyTypeError::new_err("tracking must be str or TrackingType"));
            }
        }
        if let Some(x) = keep_full_trajectories {
            rc.keep_full_trajectories = x;
        }
        if let Some(x) = store_system {
            rc.store_system = x;
        }
        if let Some(x) = size_step {
            rc.size_step = x;
        }
        if let Some(x) = parallel {
            rc.parallel = x;
        }
        if let Some(x) = num_workers {
            rc.num_workers = Some(x);
        }
        Ok(rc)
    }
}

/// Result of a Rosenbluth-style Forward Flux Sampling run.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyclass(module = "rgrow.rgrow"))]
pub struct RBFFSResult {
    /// Per-surface successes for ALL trajectories (complete + failed).
    /// Complete trajectories have len == n_surfs - 1.
    /// Failed trajectories are shorter; their last entry is 0.
    all_trajectory_successes: Vec<Vec<u64>>,
    config_trajectories: Vec<Vec<Arc<StateEnum>>>,
    n_trials: usize,
    n_surfs: usize,
    dimerization_rate: MolarPerSecond,
    /// Number of trajectories that failed (melted at some surface with 0 successes).
    n_failed_trajectories: usize,
    /// For each failed trajectory, the target size it was trying to reach when it failed.
    failed_at_size: Vec<NumTiles>,
    /// Optionally stored system for `extend()`. Skipped for serde (runtime-only state).
    #[serde(skip)]
    stored_system: Option<SystemEnum>,
    /// Config used for this run, needed by `extend()` for dispatch.
    config: RBFFSRunConfig,
}

impl RBFFSResult {
    /// The statistical weight of each complete trajectory.
    pub fn trajectory_weights(&self) -> Vec<f64> {
        self.all_trajectory_successes
            .iter()
            .filter(|s| s.len() >= self.n_surfs - 1)
            .map(|successes| {
                successes
                    .iter()
                    .map(|x| (*x as f64) / (self.n_trials as f64))
                    .product::<f64>()
            })
            .collect()
    }

    /// The forward probability at surface i of reaching surface i+1 rather than melting to 0.
    /// Includes data from all trajectories (complete + failed) that have data at surface i.
    fn forward_probability_i(&self, i: usize) -> f64 {
        forward_probability_i_from_data(&self.all_trajectory_successes, self.n_trials, i)
    }

    /// Vector of surface forward probabilities.
    pub fn forward_probabilities(&self) -> Vec<f64> {
        (0..self.n_surfs - 1)
            .map(|i| self.forward_probability_i(i))
            .collect()
    }

    /// Compute bootstrap confidence intervals by resampling trajectories.
    ///
    /// Deduplicates trajectories by their success vectors, precomputes
    /// per-group contributions, then resamples from the (much smaller)
    /// set of unique groups using a multinomial draw.
    pub fn bootstrap_ci(&self, n_bootstrap: usize, confidence_level: f64) -> RBFFSBootstrapResult {
        let n_traj = self.all_trajectory_successes.len();
        let n_fp = self.n_surfs - 1;
        let dimerization_rate = self.dimerization_rate;
        let inv_trials = 1.0 / (self.n_trials as f64);

        // Deduplicate trajectories by their success vector.
        // group_count[g] = how many trajectories share this pattern
        // group_weight[g * n_fp + i] = per-trajectory weight contribution at surface i
        // group_wsuccess[g * n_fp + i] = per-trajectory weighted-success at surface i
        let mut map: std::collections::HashMap<&Vec<u64>, usize> = std::collections::HashMap::new();
        let mut group_count: Vec<u64> = Vec::new();
        let mut group_weight: Vec<Vec<f64>> = Vec::new();
        let mut group_wsuccess: Vec<Vec<f64>> = Vec::new();

        for successes in &self.all_trajectory_successes {
            if let Some(&g) = map.get(successes) {
                group_count[g] += 1;
            } else {
                let g = group_count.len();
                map.insert(successes, g);
                group_count.push(1);

                let mut wc = vec![0.0_f64; n_fp];
                let mut ws = vec![0.0_f64; n_fp];
                let limit = successes.len().min(n_fp);
                let mut w = 1.0_f64;
                for i in 0..limit {
                    let p_i = (successes[i] as f64) * inv_trials;
                    wc[i] = w;
                    ws[i] = w * p_i;
                    w *= p_i;
                }
                group_weight.push(wc);
                group_wsuccess.push(ws);
            }
        }

        let n_groups = group_count.len();

        // Build CDF for weighted sampling (each group's probability = count / n_traj).
        let inv_n_traj = 1.0 / (n_traj as f64);
        let mut cdf = Vec::with_capacity(n_groups);
        let mut cum = 0.0_f64;
        for &c in &group_count {
            cum += (c as f64) * inv_n_traj;
            cdf.push(cum);
        }
        // Fix rounding: ensure last entry is exactly 1.0
        if let Some(last) = cdf.last_mut() {
            *last = 1.0;
        }

        let results: Vec<(f64, Vec<f64>)> = (0..n_bootstrap)
            .into_par_iter()
            .map(|_| {
                use rand::prelude::SmallRng;
                use rand::SeedableRng;
                let mut rng = SmallRng::from_os_rng();

                // Multinomial draw via sequential binomial decomposition.
                // For k groups with probabilities p_i summing to 1, draw:
                //   count_0 ~ Binomial(remaining, p_0 / remaining_prob)
                //   count_1 ~ Binomial(remaining - count_0, p_1 / remaining_prob)
                //   ...
                // This is O(n_groups) rather than O(n_traj).
                let mut counts = vec![0u32; n_groups];
                let mut remaining = n_traj as u32;
                let mut remaining_prob = 1.0_f64;
                for g in 0..n_groups {
                    if remaining == 0 {
                        break;
                    }
                    let p_g = (group_count[g] as f64) * inv_n_traj;
                    let p_cond = (p_g / remaining_prob).min(1.0);
                    let drawn = binomial_sample(&mut rng, remaining, p_cond);
                    counts[g] = drawn;
                    remaining -= drawn;
                    remaining_prob -= p_g;
                    if remaining_prob < 1e-15 {
                        remaining_prob = 1e-15;
                    }
                }

                let mut tot_w = vec![0.0_f64; n_fp];
                let mut tot_ws = vec![0.0_f64; n_fp];

                for (g, &c) in counts.iter().enumerate() {
                    if c == 0 {
                        continue;
                    }
                    let cf = c as f64;
                    let wc = &group_weight[g];
                    let wsc = &group_wsuccess[g];
                    for i in 0..n_fp {
                        tot_w[i] += cf * wc[i];
                        tot_ws[i] += cf * wsc[i];
                    }
                }

                let fwd_probs: Vec<f64> = tot_w
                    .iter()
                    .zip(tot_ws.iter())
                    .map(|(&tw, &ws)| if tw == 0.0 { 0.0 } else { ws / tw })
                    .collect();
                let nuc_rate = f64::from(dimerization_rate) * fwd_probs.iter().product::<f64>();
                (nuc_rate, fwd_probs)
            })
            .collect();

        let nucleation_rate_samples: Vec<f64> = results.iter().map(|(r, _)| *r).collect();
        let forward_probability_samples: Vec<Vec<f64>> =
            results.into_iter().map(|(_, fp)| fp).collect();

        RBFFSBootstrapResult {
            nucleation_rate_samples,
            forward_probability_samples,
            confidence_level,
            n_bootstrap,
        }
    }

    /// The nucleation rate: dimerization_rate * product of forward probabilities.
    pub fn nucleation_rate(&self) -> MolarPerSecond {
        self.dimerization_rate * self.forward_probabilities().iter().product::<f64>()
    }

    /// Resample `n` trajectories with probability proportional to their statistical
    /// weights, producing an evenly-weighted set.
    pub fn resample_trajectories(&self, n: usize) -> Vec<Vec<Arc<StateEnum>>> {
        if n == 0 {
            return Vec::new();
        }
        let weights = self.trajectory_weights();
        if weights.is_empty() {
            return Vec::new();
        }
        let Ok(chooser) = WeightedIndex::new(&weights) else {
            return Vec::new();
        };
        let mut rng = rand::rng();
        (0..n)
            .map(|_| self.config_trajectories[chooser.sample(&mut rng)].clone())
            .collect()
    }

    /// Select `n` unique trajectories via weighted sampling without replacement.
    ///
    /// Each trajectory is selected with probability proportional to its statistical
    /// weight, but once selected it cannot be chosen again. The result contains at
    /// most `n` trajectories (fewer if there aren't enough completed trajectories).
    /// Because higher-weight trajectories are more likely to be selected, the
    /// returned set has approximately even effective weight and can be treated as
    /// uniformly representative.
    pub fn select_unique_trajectories(&self, n: usize) -> Vec<Vec<Arc<StateEnum>>> {
        let weights = self.trajectory_weights();
        if weights.is_empty() || n == 0 {
            return Vec::new();
        }
        let n = n.min(weights.len());

        // Pool of (original_index, weight) pairs; shrinks as we pick.
        let mut available: Vec<(usize, f64)> = weights
            .into_iter()
            .enumerate()
            .filter(|(_, w)| *w > 0.0)
            .collect();
        let mut selected = Vec::with_capacity(n);
        let mut rng = rand::rng();

        for _ in 0..n {
            let w: Vec<f64> = available.iter().map(|(_, w)| *w).collect();
            let Ok(chooser) = WeightedIndex::new(&w) else {
                break;
            };
            let pick = chooser.sample(&mut rng);
            let (orig_idx, _) = available.swap_remove(pick);
            selected.push(self.config_trajectories[orig_idx].clone());
        }
        selected
    }

    pub fn trajectories(&self) -> &Vec<Vec<Arc<StateEnum>>> {
        &self.config_trajectories
    }

    /// Merge another batch of results into this one.
    fn merge(&mut self, other: RBFFSResult) {
        self.all_trajectory_successes
            .extend(other.all_trajectory_successes);
        self.config_trajectories.extend(other.config_trajectories);
        self.n_failed_trajectories += other.n_failed_trajectories;
        self.failed_at_size.extend(other.failed_at_size);
    }

    /// Run `n_trajectories` more trajectories and merge them into this result.
    /// Requires `store_system=true` in the config used to create this result.
    pub fn extend(&mut self, n_trajectories: usize) -> Result<(), RgrowError> {
        let sys = self.stored_system.as_mut().ok_or_else(|| {
            GrowError::NotImplemented("extend requires store_system=true in config".into())
        })?;
        let mut config = self.config.clone();
        config.n_trajectories = n_trajectories;
        let batch = match sys {
            SystemEnum::KTAM(k) => RBFFSResult::run_from_system(k, &config)?,
            SystemEnum::OldKTAM(o) => RBFFSResult::run_from_system(o, &config)?,
            _ => return Err(RgrowError::FFSCannotRunModel("unsupported model".into())),
        };
        self.merge(batch);
        Ok(())
    }

    /// Dispatch to the correct state type based on canvas_type and tracking.
    pub fn run_from_system<Sy: System>(
        sys: &mut Sy,
        config: &RBFFSRunConfig,
    ) -> Result<RBFFSResult, RgrowError>
    where
        SystemEnum: From<Sy>,
    {
        Ok(match (config.canvas_type, config.tracking) {
            (CanvasType::Square, TrackingType::None) => {
                run_rbffs::<Sy, QuadTreeState<CanvasSquare, NullStateTracker>>(sys, config)?
            }
            (CanvasType::Square, TrackingType::Order) => {
                run_rbffs::<Sy, QuadTreeState<CanvasSquare, OrderTracker>>(sys, config)?
            }
            (CanvasType::Square, TrackingType::LastAttachTime) => {
                run_rbffs::<Sy, QuadTreeState<CanvasSquare, LastAttachTimeTracker>>(sys, config)?
            }
            (CanvasType::Square, TrackingType::PrintEvent) => {
                run_rbffs::<Sy, QuadTreeState<CanvasSquare, PrintEventTracker>>(sys, config)?
            }
            (CanvasType::Square, TrackingType::Movie) => {
                run_rbffs::<Sy, QuadTreeState<CanvasSquare, MovieTracker>>(sys, config)?
            }

            (CanvasType::SquareCompact, TrackingType::None) => {
                run_rbffs::<Sy, QuadTreeState<CanvasSquareCompact, NullStateTracker>>(sys, config)?
            }
            (CanvasType::SquareCompact, TrackingType::Order) => {
                run_rbffs::<Sy, QuadTreeState<CanvasSquareCompact, OrderTracker>>(sys, config)?
            }
            (CanvasType::SquareCompact, TrackingType::LastAttachTime) => run_rbffs::<
                Sy,
                QuadTreeState<CanvasSquareCompact, LastAttachTimeTracker>,
            >(sys, config)?,
            (CanvasType::SquareCompact, TrackingType::PrintEvent) => {
                run_rbffs::<Sy, QuadTreeState<CanvasSquareCompact, PrintEventTracker>>(sys, config)?
            }
            (CanvasType::SquareCompact, TrackingType::Movie) => {
                run_rbffs::<Sy, QuadTreeState<CanvasSquareCompact, MovieTracker>>(sys, config)?
            }

            (CanvasType::Periodic, TrackingType::None) => {
                run_rbffs::<Sy, QuadTreeState<CanvasPeriodic, NullStateTracker>>(sys, config)?
            }
            (CanvasType::Periodic, TrackingType::Order) => {
                run_rbffs::<Sy, QuadTreeState<CanvasPeriodic, OrderTracker>>(sys, config)?
            }
            (CanvasType::Periodic, TrackingType::LastAttachTime) => {
                run_rbffs::<Sy, QuadTreeState<CanvasPeriodic, LastAttachTimeTracker>>(sys, config)?
            }
            (CanvasType::Periodic, TrackingType::PrintEvent) => {
                run_rbffs::<Sy, QuadTreeState<CanvasPeriodic, PrintEventTracker>>(sys, config)?
            }
            (CanvasType::Periodic, TrackingType::Movie) => {
                run_rbffs::<Sy, QuadTreeState<CanvasPeriodic, MovieTracker>>(sys, config)?
            }

            (CanvasType::Tube, TrackingType::None) => {
                run_rbffs::<Sy, QuadTreeState<CanvasTube, NullStateTracker>>(sys, config)?
            }
            (CanvasType::Tube, TrackingType::Order) => {
                run_rbffs::<Sy, QuadTreeState<CanvasTube, OrderTracker>>(sys, config)?
            }
            (CanvasType::Tube, TrackingType::LastAttachTime) => {
                run_rbffs::<Sy, QuadTreeState<CanvasTube, LastAttachTimeTracker>>(sys, config)?
            }
            (CanvasType::Tube, TrackingType::PrintEvent) => {
                run_rbffs::<Sy, QuadTreeState<CanvasTube, PrintEventTracker>>(sys, config)?
            }
            (CanvasType::Tube, TrackingType::Movie) => {
                run_rbffs::<Sy, QuadTreeState<CanvasTube, MovieTracker>>(sys, config)?
            }

            (CanvasType::TubeDiagonals, TrackingType::None) => {
                run_rbffs::<Sy, QuadTreeState<CanvasTubeDiagonals, NullStateTracker>>(sys, config)?
            }
            (CanvasType::TubeDiagonals, TrackingType::Order) => {
                run_rbffs::<Sy, QuadTreeState<CanvasTubeDiagonals, OrderTracker>>(sys, config)?
            }
            (CanvasType::TubeDiagonals, TrackingType::LastAttachTime) => run_rbffs::<
                Sy,
                QuadTreeState<CanvasTubeDiagonals, LastAttachTimeTracker>,
            >(sys, config)?,
            (CanvasType::TubeDiagonals, TrackingType::PrintEvent) => {
                run_rbffs::<Sy, QuadTreeState<CanvasTubeDiagonals, PrintEventTracker>>(sys, config)?
            }
            (CanvasType::TubeDiagonals, TrackingType::Movie) => {
                run_rbffs::<Sy, QuadTreeState<CanvasTubeDiagonals, MovieTracker>>(sys, config)?
            }
        })
    }
}

/// Sample from Binomial(n, p) using the BTPE algorithm for large np,
/// and direct inversion for small np.
fn binomial_sample<R: rand::Rng>(rng: &mut R, n: u32, p: f64) -> u32 {
    if n == 0 || p <= 0.0 {
        return 0;
    }
    if p >= 1.0 {
        return n;
    }
    let np = (n as f64) * p;
    if np < 20.0 {
        // Direct: iterate Bernoulli trials via geometric skipping.
        // For small np, draw from the CDF of the waiting time.
        let log_q = (1.0 - p).ln();
        if log_q == 0.0 {
            return 0;
        }
        let mut count = 0u32;
        let mut pos = 0.0_f64;
        loop {
            let u: f64 = rng.random();
            let skip = (u.ln() / log_q).floor();
            pos += skip + 1.0;
            if pos > n as f64 {
                break;
            }
            count += 1;
        }
        count
    } else {
        // Normal approximation with correction for large np.
        let q = 1.0 - p;
        let mean = np;
        let std = (np * q).sqrt();
        loop {
            let u1: f64 = rng.random();
            let u2: f64 = rng.random();
            // Box-Muller transform
            let z = (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos();
            let x = (mean + std * z + 0.5).floor();
            if x >= 0.0 && x <= n as f64 {
                return x as u32;
            }
        }
    }
}

/// Compute forward probability at surface `i` from a slice of trajectory success vectors.
fn forward_probability_i_from_data(trajectories: &[Vec<u64>], n_trials: usize, i: usize) -> f64 {
    let mut tot_weight = 0.0;
    let mut weighted_success = 0.0;

    for successes in trajectories {
        if successes.len() <= i {
            continue;
        }
        let w: f64 = if i == 0 {
            1.0
        } else {
            successes[0..i]
                .iter()
                .map(|x| (*x as f64) / (n_trials as f64))
                .product()
        };
        tot_weight += w;
        weighted_success += w * (successes[i] as f64) / (n_trials as f64);
    }

    if tot_weight == 0.0 {
        0.0
    } else {
        weighted_success / tot_weight
    }
}

/// Compute the p-th percentile from a sorted slice (nearest-rank method).
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = ((p / 100.0) * sorted.len() as f64).ceil() as usize;
    let idx = idx.clamp(1, sorted.len());
    sorted[idx - 1]
}

/// Compute a symmetric percentile confidence interval.
fn percentile_ci(samples: &mut [f64], confidence_level: f64) -> (f64, f64) {
    samples.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let alpha = (1.0 - confidence_level) / 2.0;
    let lo = percentile(samples, alpha * 100.0);
    let hi = percentile(samples, (1.0 - alpha) * 100.0);
    (lo, hi)
}

/// Bootstrap confidence interval results for RBFFS.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyclass(module = "rgrow.rgrow"))]
pub struct RBFFSBootstrapResult {
    nucleation_rate_samples: Vec<f64>,
    forward_probability_samples: Vec<Vec<f64>>, // [bootstrap_sample][surface]
    confidence_level: f64,
    n_bootstrap: usize,
}

impl RBFFSBootstrapResult {
    /// Confidence interval for the nucleation rate.
    pub fn nucleation_rate_ci(&self) -> (f64, f64) {
        let mut samples = self.nucleation_rate_samples.clone();
        percentile_ci(&mut samples, self.confidence_level)
    }

    /// Per-surface confidence intervals for forward probabilities.
    pub fn forward_probability_cis(&self) -> Vec<(f64, f64)> {
        if self.forward_probability_samples.is_empty() {
            return Vec::new();
        }
        let n_surfs = self.forward_probability_samples[0].len();
        (0..n_surfs)
            .map(|s| {
                let mut col: Vec<f64> = self
                    .forward_probability_samples
                    .iter()
                    .map(|fp| fp[s])
                    .collect();
                percentile_ci(&mut col, self.confidence_level)
            })
            .collect()
    }

    /// Median nucleation rate across bootstrap samples.
    pub fn nucleation_rate_median(&self) -> f64 {
        let mut samples = self.nucleation_rate_samples.clone();
        samples.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        percentile(&samples, 50.0)
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl RBFFSBootstrapResult {
    #[getter]
    fn get_nucleation_rate_ci(&self) -> (f64, f64) {
        self.nucleation_rate_ci()
    }

    #[getter]
    fn get_nucleation_rate_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.nucleation_rate_samples.to_pyarray(py)
    }

    #[getter]
    fn get_forward_probability_cis(&self) -> Vec<(f64, f64)> {
        self.forward_probability_cis()
    }

    #[getter]
    fn get_forward_probability_samples<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let n_bootstrap = self.forward_probability_samples.len();
        let n_surfs = if n_bootstrap > 0 {
            self.forward_probability_samples[0].len()
        } else {
            0
        };
        let mut arr = Array2::<f64>::zeros((n_bootstrap, n_surfs));
        for (i, row) in self.forward_probability_samples.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                arr[(i, j)] = val;
            }
        }
        arr.to_pyarray(py)
    }

    #[getter]
    fn get_confidence_level(&self) -> f64 {
        self.confidence_level
    }

    #[getter]
    fn get_nucleation_rate_median(&self) -> f64 {
        self.nucleation_rate_median()
    }

    fn __repr__(&self) -> String {
        let ci = self.nucleation_rate_ci();
        format!(
            "RBFFSBootstrapResult(n_bootstrap={}, confidence={:.0}%, nuc_rate_ci=({:.4e}, {:.4e}))",
            self.n_bootstrap,
            self.confidence_level * 100.0,
            ci.0,
            ci.1,
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl RBFFSResult {
    #[getter]
    fn get_forward_probabilities<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.forward_probabilities().to_pyarray(py)
    }

    #[getter]
    fn get_trajectory_weights<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.trajectory_weights().to_pyarray(py)
    }

    #[getter]
    fn get_n_trials(&self) -> usize {
        self.n_trials
    }

    #[getter]
    fn get_n_surfs(&self) -> usize {
        self.n_surfs
    }

    #[getter]
    fn get_dimerization_rate(&self) -> f64 {
        self.dimerization_rate.into()
    }

    #[getter]
    fn get_nucleation_rate(&self) -> f64 {
        self.nucleation_rate().into()
    }

    #[getter]
    fn get_n_trajectories(&self) -> usize {
        self.config_trajectories.len()
    }

    #[getter]
    fn get_n_failed_trajectories(&self) -> usize {
        self.n_failed_trajectories
    }

    #[getter]
    fn get_trajectories(&self) -> Vec<Vec<FFSStateRef>> {
        self.config_trajectories
            .iter()
            .map(|traj| {
                traj.iter()
                    .map(|s| FFSStateRef(Arc::downgrade(s)))
                    .collect()
            })
            .collect()
    }

    #[pyo3(name = "resample_trajectories")]
    fn py_resample_trajectories(&self, n: usize) -> Vec<Vec<FFSStateRef>> {
        self.resample_trajectories(n)
            .iter()
            .map(|traj| {
                traj.iter()
                    .map(|s| FFSStateRef(Arc::downgrade(s)))
                    .collect()
            })
            .collect()
    }

    #[pyo3(name = "select_unique_trajectories")]
    fn py_select_unique_trajectories(&self, n: usize) -> Vec<Vec<FFSStateRef>> {
        self.select_unique_trajectories(n)
            .iter()
            .map(|traj| {
                traj.iter()
                    .map(|s| FFSStateRef(Arc::downgrade(s)))
                    .collect()
            })
            .collect()
    }

    #[getter]
    fn get_failed_at_size<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        let v: Vec<u32> = self.failed_at_size.iter().map(|&x| x as u32).collect();
        v.to_pyarray(py)
    }

    #[pyo3(name = "extend")]
    fn py_extend(&mut self, n_trajectories: usize) -> PyResult<()> {
        self.extend(n_trajectories)
            .map_err(|e| PyTypeError::new_err(e.to_string()))
    }

    #[pyo3(name = "bootstrap_ci", signature = (n_bootstrap=10000, confidence_level=0.95))]
    fn py_bootstrap_ci(&self, n_bootstrap: usize, confidence_level: f64) -> RBFFSBootstrapResult {
        self.bootstrap_ci(n_bootstrap, confidence_level)
    }

    fn __repr__(&self) -> String {
        format!(
            "RBFFSResult(nuc_rate={:1.4e} M/s, n_trajectories={}, n_failed={}, n_surfs={}, fwd_probs={:?})",
            f64::from(self.nucleation_rate()),
            self.config_trajectories.len(),
            self.n_failed_trajectories,
            self.n_surfs,
            self.forward_probabilities()
        )
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Compute the sequence of surface sizes for RBFFS.
///
/// Surfaces start at the dimer (size 2) and step by `size_step` up to `target_size`.
/// The last surface is always exactly `target_size`, even if the final step is shorter.
fn compute_surface_sizes(target_size: NumTiles, size_step: NumTiles) -> Vec<NumTiles> {
    assert!(size_step >= 1, "size_step must be at least 1");
    assert!(target_size >= 3, "target_size must be at least 3");
    let mut sizes = Vec::new();
    let mut s = 2 + size_step;
    while s < target_size {
        sizes.push(s);
        s += size_step;
    }
    sizes.push(target_size);
    sizes
}

/// Outcome of a single RBFFS trajectory attempt.
struct TrajectoryOutcome<St> {
    successes: Vec<u64>,
    trajectory: Vec<St>,
    failed_at_size: Option<NumTiles>,
}

/// Run a single trajectory: sample a dimer, then grow surface-by-surface.
fn run_single_trajectory<
    Sy: System,
    St: ClonableState + StateWithCreate<Params = (usize, usize)>,
>(
    system: &Sy,
    dimers: &[DimerInfo],
    chooser: &WeightedIndex<f64>,
    config: &RBFFSRunConfig,
    surface_sizes: &[NumTiles],
    trial_state: &mut St,
) -> Result<TrajectoryOutcome<St>, GrowError> {
    let mut traj = Vec::<St>::new();
    let mut successes = Vec::new();

    let mut base_state = state_from_dimer::<Sy, St>(
        system,
        &dimers[chooser.sample(&mut rand::rng())],
        config.canvas_size,
    )?;

    for &next_size in surface_sizes {
        // Reservoir sampling: keep one uniformly random successful state.
        let mut kept_state: Option<St> = None;
        let mut n_success: u64 = 0;
        let bounds = {
            let mut b = config.subseq_bound;
            b.size_max = Some(next_size);
            b.size_min = Some(0);
            b
        };
        for _ in 0..config.n_trials {
            // Reuse trial_state: reset it and copy base_state into it.
            trial_state.reset_state();
            system.clone_state_into_empty_state(&base_state, trial_state);
            let outcome = system.evolve(trial_state, bounds)?;
            match outcome {
                EvolveOutcome::ReachedSizeMin | EvolveOutcome::ReachedTimeMax => continue,
                EvolveOutcome::ReachedSizeMax => {
                    n_success += 1;
                    // Reservoir sampling (k=1): keep each success with probability 1/n_success.
                    if rand::random_range(1..=n_success) == 1 {
                        kept_state = Some(trial_state.clone());
                    }
                }
                _ => {
                    panic!("Unexpected outcome: {:?}", outcome)
                }
            }
        }
        if n_success == 0 {
            successes.push(0);
            return Ok(TrajectoryOutcome {
                successes,
                trajectory: Vec::new(),
                failed_at_size: Some(next_size),
            });
        }
        successes.push(n_success);
        if config.keep_full_trajectories {
            traj.push(base_state);
        }
        base_state = kept_state.unwrap();
    }

    traj.push(base_state);
    Ok(TrajectoryOutcome {
        successes,
        trajectory: traj,
        failed_at_size: None,
    })
}

/// Collect trajectory outcomes into the result accumulators, returning number of newly completed.
fn collect_outcomes<St>(
    outcomes: Vec<TrajectoryOutcome<St>>,
    all_trajectory_successes: &mut Vec<Vec<u64>>,
    config_trajectories: &mut Vec<Vec<St>>,
    n_failed_trajectories: &mut usize,
    failed_at_size: &mut Vec<NumTiles>,
) -> usize {
    let mut n_complete = 0;
    for outcome in outcomes {
        all_trajectory_successes.push(outcome.successes);
        if let Some(size) = outcome.failed_at_size {
            *n_failed_trajectories += 1;
            failed_at_size.push(size);
        } else {
            config_trajectories.push(outcome.trajectory);
            n_complete += 1;
        }
    }
    n_complete
}

/// Build the final RBFFSResult from accumulators.
fn build_result<St>(
    all_trajectory_successes: Vec<Vec<u64>>,
    config_trajectories: Vec<Vec<St>>,
    config: &RBFFSRunConfig,
    n_surfs: usize,
    dimerization_rate: MolarPerSecond,
    n_failed_trajectories: usize,
    failed_at_size: Vec<NumTiles>,
) -> RBFFSResult
where
    StateEnum: From<St>,
{
    RBFFSResult {
        all_trajectory_successes,
        config_trajectories: config_trajectories
            .into_iter()
            .map(|traj| traj.into_iter().map(|s| Arc::new(s.into())).collect())
            .collect(),
        n_trials: config.n_trials,
        n_surfs,
        dimerization_rate,
        n_failed_trajectories,
        failed_at_size,
        stored_system: None,
        config: config.clone(),
    }
}

fn run_rbffs_sequential<Sy: System, St: ClonableState + StateWithCreate<Params = (usize, usize)>>(
    system: &Sy,
    config: &RBFFSRunConfig,
    dimers: &[DimerInfo],
    dimerization_rate: MolarPerSecond,
    chooser: &WeightedIndex<f64>,
    surface_sizes: &[NumTiles],
    n_surfs: usize,
) -> Result<RBFFSResult, GrowError>
where
    StateEnum: From<St>,
{
    let mut all_trajectory_successes = Vec::new();
    let mut config_trajectories = Vec::new();
    let mut n_failed_trajectories: usize = 0;
    let mut failed_at_size: Vec<NumTiles> = Vec::new();
    let mut trial_state = St::empty(config.canvas_size)?;
    let mut n_complete = 0;

    while n_complete < config.n_trajectories {
        let outcome = run_single_trajectory(
            system,
            dimers,
            chooser,
            config,
            surface_sizes,
            &mut trial_state,
        )?;
        if outcome.failed_at_size.is_none() {
            n_complete += 1;
        }
        collect_outcomes(
            vec![outcome],
            &mut all_trajectory_successes,
            &mut config_trajectories,
            &mut n_failed_trajectories,
            &mut failed_at_size,
        );
    }

    Ok(build_result(
        all_trajectory_successes,
        config_trajectories,
        config,
        n_surfs,
        dimerization_rate,
        n_failed_trajectories,
        failed_at_size,
    ))
}

fn run_rbffs_parallel<Sy: System, St: ClonableState + StateWithCreate<Params = (usize, usize)>>(
    system: &Sy,
    config: &RBFFSRunConfig,
    dimers: &[DimerInfo],
    dimerization_rate: MolarPerSecond,
    chooser: &WeightedIndex<f64>,
    surface_sizes: &[NumTiles],
    n_surfs: usize,
) -> Result<RBFFSResult, GrowError>
where
    StateEnum: From<St>,
{
    let run_batches = || -> Result<RBFFSResult, GrowError> {
        let mut all_trajectory_successes = Vec::new();
        let mut config_trajectories = Vec::new();
        let mut n_failed_trajectories: usize = 0;
        let mut failed_at_size: Vec<NumTiles> = Vec::new();
        let mut n_complete = 0;

        while n_complete < config.n_trajectories {
            let remaining = config.n_trajectories - n_complete;
            let batch_results: Result<Vec<TrajectoryOutcome<St>>, GrowError> = (0..remaining)
                .into_par_iter()
                .map_init(
                    || St::empty(config.canvas_size).unwrap(),
                    |trial_state, _| {
                        run_single_trajectory(
                            system,
                            dimers,
                            chooser,
                            config,
                            surface_sizes,
                            trial_state,
                        )
                    },
                )
                .collect();

            n_complete += collect_outcomes(
                batch_results?,
                &mut all_trajectory_successes,
                &mut config_trajectories,
                &mut n_failed_trajectories,
                &mut failed_at_size,
            );
        }

        Ok(build_result(
            all_trajectory_successes,
            config_trajectories,
            config,
            n_surfs,
            dimerization_rate,
            n_failed_trajectories,
            failed_at_size,
        ))
    };

    if let Some(n) = config.num_workers {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .map_err(|e| {
                GrowError::NotImplemented(format!("Failed to create thread pool: {}", e))
            })?;
        pool.install(run_batches)
    } else {
        run_batches()
    }
}

pub fn run_rbffs<Sy: System, St: ClonableState + StateWithCreate<Params = (usize, usize)>>(
    system: &mut Sy,
    config: &RBFFSRunConfig,
) -> Result<RBFFSResult, GrowError>
where
    StateEnum: From<St>,
{
    let dimers = system.calc_dimers()?;

    let dimerization_rate: MolarPerSecond = dimers
        .iter()
        .fold(MolarPerSecond::zero(), |acc, d| acc + d.formation_rate);

    let surface_sizes = compute_surface_sizes(config.target_size, config.size_step);
    // n_surfs = number of surfaces including the initial dimer surface
    let n_surfs = surface_sizes.len() + 1;

    let weights: Vec<_> = dimers.iter().map(|d| f64::from(d.formation_rate)).collect();
    let chooser = WeightedIndex::new(weights).unwrap();

    let system: &Sy = system;

    if config.parallel {
        run_rbffs_parallel::<Sy, St>(
            system,
            config,
            &dimers,
            dimerization_rate,
            &chooser,
            &surface_sizes,
            n_surfs,
        )
    } else {
        run_rbffs_sequential::<Sy, St>(
            system,
            config,
            &dimers,
            dimerization_rate,
            &chooser,
            &surface_sizes,
            n_surfs,
        )
    }
}

pub fn state_from_dimer<
    Sy: System,
    St: ClonableState + StateWithCreate<Params = (usize, usize)>,
>(
    system: &Sy,
    dimer: &DimerInfo,
    canvas_shape: (usize, usize),
) -> Result<St, GrowError> {
    let mut state = St::empty(canvas_shape)?;
    if canvas_shape.0 < 4 || canvas_shape.1 < 4 {
        panic!("Canvas size too small for dimers");
    }
    let mid = PointSafe2((canvas_shape.0 / 2, canvas_shape.1 / 2));
    let other = match dimer.orientation {
        Orientation::NS => PointSafe2(state.move_sa_s(mid).0),
        Orientation::WE => PointSafe2(state.move_sa_e(mid).0),
    };
    // Use place_tile to properly handle double tiles
    let energy_change = system.place_tile(&mut state, mid, dimer.t1, true)?
        + system.place_tile(&mut state, other, dimer.t2, true)?;
    let cl = [(mid, dimer.t1), (other, dimer.t2)];
    state.record_event(
        &system::Event::PolymerAttachment(cl.to_vec()),
        PerSecond::zero(),
        f64::NAN,
        energy_change,
        energy_change,
        2,
    );
    Ok(state)
}

impl TileSet {
    pub fn run_rbffs(&self, config: &RBFFSRunConfig) -> Result<RBFFSResult, RgrowError> {
        let model = self.model.unwrap_or(Model::KTAM);
        let config = {
            let mut c = config.clone();
            c.canvas_size = match self.size.unwrap_or(SIZE_DEFAULT) {
                Size::Single(x) => (x, x),
                Size::Pair(p) => p,
            };
            c.canvas_type = self.canvas_type.unwrap_or(CanvasType::Periodic);
            c.tracking = self.tracking.unwrap_or(TrackingType::None);
            c
        };

        match model {
            Model::KTAM => {
                let mut ktam = KTAM::try_from(self)?;
                let mut r = RBFFSResult::run_from_system(&mut ktam, &config)?;
                if config.store_system {
                    r.stored_system = Some(SystemEnum::from(ktam));
                }
                Ok(r)
            }
            Model::OldKTAM => {
                let mut oldktam = OldKTAM::try_from(self)?;
                let mut r = RBFFSResult::run_from_system(&mut oldktam, &config)?;
                if config.store_system {
                    r.stored_system = Some(SystemEnum::from(oldktam));
                }
                Ok(r)
            }
            Model::ATAM => Err(RgrowError::FFSCannotRunModel("aTAM".into())),
            Model::SDC => Err(RgrowError::FFSCannotRunModel("SDC".into())),
        }
    }
}
