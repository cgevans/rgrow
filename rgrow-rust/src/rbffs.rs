//! An implementation of Rosenbluth-style Forward Flux Sampling
//!
//! Based on the explanation in R. J. Allen, C. Valeriani, and P. R. ten Wolde,
//! "Forward flux sampling for rare event simulations," J. Phys.: Condens. Matter,
//! vol. 21, no. 46, p. 463102, Oct. 2009, doi: 10.1088/0953-8984/21/46/463102.

use num_traits::Zero;
use rand::distr::{weighted::WeightedIndex, Distribution};
use serde::{Deserialize, Serialize};

#[cfg(feature = "python")]
use numpy::{PyArray1, ToPyArray};
#[cfg(feature = "python")]
use pyo3::exceptions::PyTypeError;
#[cfg(feature = "python")]
use pyo3::prelude::*;

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
    ))]
    fn new(
        n_trials: Option<usize>,
        n_trajectories: Option<usize>,
        target_size: Option<NumTiles>,
        canvas_size: Option<(usize, usize)>,
        subseq_bound: Option<EvolveBounds>,
        canvas_type: Option<CanvasType>,
        tracking: Option<Bound<'_, PyAny>>,
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
    config_trajectories: Vec<Vec<StateEnum>>,
    n_trials: usize,
    n_surfs: usize,
    dimerization_rate: MolarPerSecond,
    /// Number of trajectories that failed (melted at some surface with 0 successes).
    n_failed_trajectories: usize,
    /// For each failed trajectory, the target size it was trying to reach when it failed.
    failed_at_size: Vec<NumTiles>,
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
        let mut tot_weight = 0.0;
        let mut weighted_success = 0.0;

        for successes in &self.all_trajectory_successes {
            if successes.len() <= i {
                continue; // no data at surface i
            }
            let w: f64 = if i == 0 {
                1.0
            } else {
                successes[0..i]
                    .iter()
                    .map(|x| (*x as f64) / (self.n_trials as f64))
                    .product()
            };
            tot_weight += w;
            weighted_success += w * (successes[i] as f64) / (self.n_trials as f64);
        }

        if tot_weight == 0.0 {
            0.0
        } else {
            weighted_success / tot_weight
        }
    }

    /// Vector of surface forward probabilities.
    pub fn forward_probabilities(&self) -> Vec<f64> {
        (0..self.n_surfs - 1)
            .map(|i| self.forward_probability_i(i))
            .collect()
    }

    /// The nucleation rate: dimerization_rate * product of forward probabilities.
    pub fn nucleation_rate(&self) -> MolarPerSecond {
        self.dimerization_rate * self.forward_probabilities().iter().product::<f64>()
    }

    pub fn trajectories(&self) -> &Vec<Vec<StateEnum>> {
        &self.config_trajectories
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
    fn get_failed_at_size<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        let v: Vec<u32> = self.failed_at_size.iter().map(|&x| x as u32).collect();
        v.to_pyarray(py)
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

    let mut n_surfs = 1;

    let weights: Vec<_> = dimers.iter().map(|d| f64::from(d.formation_rate)).collect();
    let chooser = WeightedIndex::new(weights).unwrap();

    let mut all_trajectory_successes = Vec::new();
    let mut config_trajectories = Vec::new();
    let mut n_failed_trajectories: usize = 0;
    let mut failed_at_size: Vec<NumTiles> = Vec::new();

    let mut n_traj = 0;
    let n_trials = config.n_trials;

    // Pre-allocate a reusable trial state to avoid per-trial allocation.
    let mut trial_state = St::empty(config.canvas_size)?;

    'outer: while n_traj < config.n_trajectories {
        if n_traj == 0 {
            n_surfs = 1;
        } // In case the first few trajectories fail
        let mut traj = Vec::<St>::new();
        let mut successes = Vec::new();

        let mut base_state = state_from_dimer::<Sy, St>(
            system,
            &dimers[chooser.sample(&mut rand::rng())],
            config.canvas_size,
        )?;
        let mut next_size = 3;

        while next_size <= config.target_size {
            if n_traj == 0 {
                n_surfs += 1;
            }
            // Reservoir sampling: keep one uniformly random successful state.
            let mut kept_state: Option<St> = None;
            let mut n_success: u64 = 0;
            let bounds = {
                let mut b = config.subseq_bound;
                b.size_max = Some(next_size);
                b.size_min = Some(0);
                b
            };
            for _ in 0..n_trials {
                // Reuse trial_state: reset it and copy base_state into it.
                trial_state.reset_state();
                system.clone_state_into_empty_state(&base_state, &mut trial_state);
                let outcome = system.evolve(&mut trial_state, bounds)?;
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
                all_trajectory_successes.push(successes);
                n_failed_trajectories += 1;
                failed_at_size.push(next_size);
                continue 'outer;
            }
            successes.push(n_success);
            traj.push(base_state);
            base_state = kept_state.unwrap();
            next_size += 1;
        }

        all_trajectory_successes.push(successes);
        config_trajectories.push(traj);
        n_traj += 1;
    }

    Ok(RBFFSResult {
        all_trajectory_successes,
        config_trajectories: config_trajectories
            .into_iter()
            .map(|traj| traj.into_iter().map(|s| s.into()).collect())
            .collect(),
        n_trials,
        n_surfs,
        dimerization_rate,
        n_failed_trajectories,
        failed_at_size,
    })
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
            Model::KTAM => RBFFSResult::run_from_system(&mut KTAM::try_from(self)?, &config),
            Model::OldKTAM => RBFFSResult::run_from_system(&mut OldKTAM::try_from(self)?, &config),
            Model::ATAM => Err(RgrowError::FFSCannotRunModel("aTAM".into())),
            Model::SDC => Err(RgrowError::FFSCannotRunModel("SDC".into())),
        }
    }
}
