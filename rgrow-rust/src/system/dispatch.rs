use enum_dispatch::enum_dispatch;
use ndarray::prelude::*;
use serde::{Deserialize, Serialize};

use std::any::Any;
use std::fmt::Debug;

use rayon::prelude::*;

use crate::base::{GrowError, NumEvents, NumTiles, RgrowError, Tile};
use crate::ffs::{FFSRunConfig, FFSRunResult};
use crate::models::atam::ATAM;
use crate::models::kblock::KBlock;
use crate::models::ktam::KTAM;
use crate::models::oldktam::OldKTAM;
use crate::models::sdc1d::SDC;
use crate::models::sdc1d_bindreplace::SDC1DBindReplace;
use crate::painter::{SpriteSquare, TileStyle};
use crate::state::StateEnum;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
use pyo3::IntoPyObjectExt;

use super::analysis;
use super::core::System;
use super::types::*;

#[enum_dispatch]
pub trait DynSystem: Sync + Send + TileBondInfo {
    /// Simulate a single state, until reaching specified stopping conditions.
    fn evolve(
        &self,
        state: &mut StateEnum,
        bounds: EvolveBounds,
    ) -> Result<EvolveOutcome, GrowError>;

    /// Evolve a list of states, in parallel.
    fn evolve_states(
        &mut self,
        states: &mut [&mut StateEnum],
        bounds: EvolveBounds,
    ) -> Vec<Result<EvolveOutcome, GrowError>>;

    fn setup_state(&self, state: &mut StateEnum) -> Result<(), GrowError>;

    fn evolve_in_window(
        &mut self,
        state: &mut StateEnum,
        block: Option<usize>,
        start_paused: bool,
        bounds: EvolveBounds,
        initial_timescale: Option<f64>,
        initial_max_events_per_sec: Option<u64>,
    ) -> Result<EvolveOutcome, RgrowError>;

    fn calc_mismatches(&self, state: &StateEnum) -> usize;
    fn calc_mismatch_locations(&self, state: &StateEnum) -> Array2<usize>;

    fn set_param(&mut self, name: &str, value: Box<dyn Any>) -> Result<NeededUpdate, GrowError>;
    fn get_param(&self, name: &str) -> Result<Box<dyn Any>, GrowError>;

    fn update_state(&self, state: &mut StateEnum, needed: &NeededUpdate);

    fn system_info(&self) -> String;

    fn run_ffs(&mut self, config: &FFSRunConfig) -> Result<FFSRunResult, RgrowError>;

    fn calc_committor(
        &mut self,
        initial_state: &StateEnum,
        cutoff_size: NumTiles,
        max_time: Option<f64>,
        max_events: Option<NumEvents>,
        num_trials: usize,
    ) -> Result<f64, GrowError>;

    fn calc_committor_adaptive(
        &self,
        initial_state: &StateEnum,
        cutoff_size: NumTiles,
        max_time: Option<f64>,
        max_events: Option<NumEvents>,
        conf_interval_margin: f64,
    ) -> Result<(f64, usize), GrowError>;

    fn calc_committors_adaptive(
        &self,
        initial_states: &[&StateEnum],
        cutoff_size: NumTiles,
        max_time: Option<f64>,
        max_events: Option<NumEvents>,
        conf_interval_margin: f64,
    ) -> Result<(Vec<f64>, Vec<usize>), GrowError>;

    fn calc_forward_probability(
        &mut self,
        initial_state: &StateEnum,
        forward_step: NumTiles,
        max_time: Option<f64>,
        max_events: Option<NumEvents>,
        num_trials: usize,
    ) -> Result<f64, GrowError>;

    fn calc_forward_probability_adaptive(
        &self,
        initial_state: &StateEnum,
        forward_step: NumTiles,
        max_time: Option<f64>,
        max_events: Option<NumEvents>,
        conf_interval_margin: f64,
    ) -> Result<(f64, usize), GrowError>;

    fn calc_forward_probabilities_adaptive(
        &self,
        initial_states: &[&StateEnum],
        forward_step: NumTiles,
        max_time: Option<f64>,
        max_events: Option<NumEvents>,
        conf_interval_margin: f64,
    ) -> Result<(Vec<f64>, Vec<usize>), GrowError>;

    /// Determine whether the committor probability for a state is above or below a threshold
    /// with a specified confidence level using adaptive sampling.
    ///
    /// This function uses adaptive sampling with Wilson Score confidence intervals to determine
    /// with the desired confidence whether the true committor probability is above or below the
    /// given threshold. It continues sampling until the confidence interval is narrow enough to
    /// make a definitive determination, or until the maximum number of trials is reached.
    ///
    /// The committor probability is the probability that when a simulation is started from the
    /// given state, the assembly will grow to reach `cutoff_size` rather than melting to zero tiles.
    ///
    /// # Arguments
    ///
    /// * `initial_state` - The state to analyze
    /// * `cutoff_size` - Size threshold for commitment (number of tiles)
    /// * `threshold` - The probability threshold to compare against (must be between 0.0 and 1.0)
    /// * `confidence_level` - Confidence level for the threshold test (e.g., 0.95 for 95% confidence)
    /// * `max_time` - Optional maximum simulation time per trial
    /// * `max_events` - Optional maximum number of events per trial
    /// * `max_trials` - Optional maximum number of trials to run (default: 100,000)
    /// * `return_on_max_trials` - If `true`, return results even when max_trials is exceeded;
    ///   if `false`, return an error when max_trials is exceeded without reaching confidence
    /// * `ci_confidence_level` - Optional confidence level for the returned confidence interval.
    ///   If `None`, no confidence interval is returned. Can be different from `confidence_level`
    ///   (e.g., test at 95% confidence but show 99% confidence interval)
    ///
    /// # Returns
    ///
    /// Returns `Ok((is_above_threshold, probability_estimate, confidence_interval, num_trials, exceeded_max_trials))`
    /// where:
    /// * `is_above_threshold` - `true` if probability is above threshold with given confidence
    /// * `probability_estimate` - The estimated committor probability (between 0.0 and 1.0)
    /// * `confidence_interval` - `Some((lower_bound, upper_bound))` if `ci_confidence_level` is provided,
    ///   `None` otherwise
    /// * `num_trials` - Number of trials performed
    /// * `exceeded_max_trials` - `true` if max_trials was exceeded (warning flag)
    ///
    /// # Errors
    ///
    /// Returns `Err(GrowError)` if:
    /// * `threshold` is not between 0.0 and 1.0
    /// * `confidence_level` is not between 0.0 and 1.0
    /// * `ci_confidence_level` is provided but not between 0.0 and 1.0
    /// * `max_trials` is exceeded and `return_on_max_trials` is `false`
    /// * Evolution simulation encounters an unsupported outcome
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use rgrow::system::DynSystem;
    /// # use rgrow::state::StateEnum;
    /// # fn example(system: &mut dyn DynSystem, state: &StateEnum) -> Result<(), Box<dyn std::error::Error>> {
    /// // Test at 95% confidence
    /// let (is_above, prob, trials, exceeded) = system.calc_committor_threshold_test(
    ///     state,
    ///     10,      // cutoff_size
    ///     0.5,     // threshold
    ///     0.95,    // z_level
    ///     None,    // max_time
    ///     None,    // max_events
    ///     None,    // max_trials (default: 100,000)
    ///     false,   // return_on_max_trials
    /// )?;
    ///
    /// println!("Probability {} threshold 0.5", if is_above { "above" } else { "below" });
    /// println!("Estimate: {:.4}, Trials: {}", prob, trials);
    ///
    /// // Test with max_trials limit
    /// let (is_above, prob, trials, exceeded) = system.calc_committor_threshold_test(
    ///     state,
    ///     10,         // cutoff_size
    ///     0.5,        // threshold
    ///     0.95,       // z_level
    ///     None,       // max_time
    ///     None,       // max_events
    ///     Some(1000), // max_trials
    ///     true,       // return_on_max_trials
    /// )?;
    ///
    /// if exceeded {
    ///     println!("WARNING: Max trials exceeded!");
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)]
    fn calc_committor_threshold_test(
        &mut self,
        initial_state: &StateEnum,
        cutoff_size: NumTiles,
        threshold: f64,
        z_level: f64,
        max_time: Option<f64>,
        max_events: Option<NumEvents>,
        max_trials: Option<usize>,
        return_on_max_trials: bool,
    ) -> Result<(bool, f64, usize, bool), GrowError>;

    // /// Find the first state in a trajectory that is above the critical threshold.
    fn find_first_critical_state(
        &mut self,
        end_state: &StateEnum,
        config: &CriticalStateConfig,
    ) -> Result<Option<CriticalStateResult>, GrowError>;

    // /// Find the last state not above threshold, return the next state (first above threshold).
    fn find_last_critical_state(
        &mut self,
        end_state: &StateEnum,
        config: &CriticalStateConfig,
    ) -> Result<Option<CriticalStateResult>, GrowError>;
}

impl<S: System> DynSystem for S
where
    SystemEnum: From<S>,
{
    fn evolve(
        &self,
        state: &mut StateEnum,
        bounds: EvolveBounds,
    ) -> Result<EvolveOutcome, GrowError> {
        self.evolve(state, bounds)
    }

    fn evolve_states(
        &mut self,
        states: &mut [&mut StateEnum],
        bounds: EvolveBounds,
    ) -> Vec<Result<EvolveOutcome, GrowError>> {
        states
            .par_iter_mut()
            .map(|state| self.evolve(*state, bounds))
            .collect()
    }

    fn setup_state(&self, state: &mut StateEnum) -> Result<(), GrowError> {
        self.configure_empty_state(state)
    }

    fn evolve_in_window(
        &mut self,
        state: &mut StateEnum,
        block: Option<usize>,
        start_paused: bool,
        bounds: EvolveBounds,
        initial_timescale: Option<f64>,
        initial_max_events_per_sec: Option<u64>,
    ) -> Result<EvolveOutcome, RgrowError> {
        self.evolve_in_window(
            state,
            block,
            start_paused,
            bounds,
            initial_timescale,
            initial_max_events_per_sec,
        )
    }

    fn calc_mismatches(&self, state: &StateEnum) -> usize {
        self.calc_mismatches(state)
    }

    fn calc_mismatch_locations(&self, state: &StateEnum) -> Array2<usize> {
        self.calc_mismatch_locations(state)
    }

    fn set_param(&mut self, name: &str, value: Box<dyn Any>) -> Result<NeededUpdate, GrowError> {
        self.set_param(name, value)
    }

    fn get_param(&self, name: &str) -> Result<Box<dyn Any>, GrowError> {
        self.get_param(name)
    }

    fn update_state(&self, state: &mut StateEnum, needed: &NeededUpdate) {
        self.update_state(state, needed)
    }

    fn run_ffs(&mut self, config: &FFSRunConfig) -> Result<FFSRunResult, RgrowError> {
        FFSRunResult::run_from_system(self, config)
    }

    fn system_info(&self) -> String {
        self.system_info()
    }

    fn calc_committor(
        &mut self,
        initial_state: &StateEnum,
        cutoff_size: NumTiles,
        max_time: Option<f64>,
        max_events: Option<NumEvents>,
        num_trials: usize,
    ) -> Result<f64, GrowError> {
        analysis::calc_committor(self, initial_state, cutoff_size, max_time, max_events, num_trials)
    }

    fn calc_committor_adaptive(
        &self,
        initial_state: &StateEnum,
        cutoff_size: NumTiles,
        max_time: Option<f64>,
        max_events: Option<NumEvents>,
        conf_interval_margin: f64,
    ) -> Result<(f64, usize), GrowError> {
        analysis::calc_committor_adaptive(
            self,
            initial_state,
            cutoff_size,
            max_time,
            max_events,
            conf_interval_margin,
        )
    }

    fn calc_committors_adaptive(
        &self,
        initial_states: &[&StateEnum],
        cutoff_size: NumTiles,
        max_time: Option<f64>,
        max_events: Option<NumEvents>,
        conf_interval_margin: f64,
    ) -> Result<(Vec<f64>, Vec<usize>), GrowError> {
        analysis::calc_committors_adaptive(
            self,
            initial_states,
            cutoff_size,
            max_time,
            max_events,
            conf_interval_margin,
        )
    }

    fn calc_forward_probability(
        &mut self,
        initial_state: &StateEnum,
        forward_step: NumTiles,
        max_time: Option<f64>,
        max_events: Option<NumEvents>,
        num_trials: usize,
    ) -> Result<f64, GrowError> {
        analysis::calc_forward_probability(
            self,
            initial_state,
            forward_step,
            max_time,
            max_events,
            num_trials,
        )
    }

    fn calc_forward_probability_adaptive(
        &self,
        initial_state: &StateEnum,
        forward_step: NumTiles,
        max_time: Option<f64>,
        max_events: Option<NumEvents>,
        conf_interval_margin: f64,
    ) -> Result<(f64, usize), GrowError> {
        analysis::calc_forward_probability_adaptive(
            self,
            initial_state,
            forward_step,
            max_time,
            max_events,
            conf_interval_margin,
        )
    }

    fn calc_forward_probabilities_adaptive(
        &self,
        initial_states: &[&StateEnum],
        forward_step: NumTiles,
        max_time: Option<f64>,
        max_events: Option<NumEvents>,
        conf_interval_margin: f64,
    ) -> Result<(Vec<f64>, Vec<usize>), GrowError> {
        analysis::calc_forward_probabilities_adaptive(
            self,
            initial_states,
            forward_step,
            max_time,
            max_events,
            conf_interval_margin,
        )
    }

    fn calc_committor_threshold_test(
        &mut self,
        initial_state: &StateEnum,
        cutoff_size: NumTiles,
        threshold: f64,
        z_level: f64,
        max_time: Option<f64>,
        max_events: Option<NumEvents>,
        max_trials: Option<usize>,
        return_on_max_trials: bool,
    ) -> Result<(bool, f64, usize, bool), GrowError> {
        analysis::calc_committor_threshold_test(
            self,
            initial_state,
            cutoff_size,
            threshold,
            z_level,
            max_time,
            max_events,
            max_trials,
            return_on_max_trials,
        )
    }

    fn find_first_critical_state(
        &mut self,
        end_state: &StateEnum,
        config: &CriticalStateConfig,
    ) -> Result<Option<CriticalStateResult>, GrowError> {
        analysis::find_first_critical_state(self, end_state, config)
    }

    fn find_last_critical_state(
        &mut self,
        end_state: &StateEnum,
        config: &CriticalStateConfig,
    ) -> Result<Option<CriticalStateResult>, GrowError> {
        analysis::find_last_critical_state(self, end_state, config)
    }
}

#[enum_dispatch(DynSystem, TileBondInfo)]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", derive(FromPyObject))]
pub enum SystemEnum {
    KTAM,
    OldKTAM,
    ATAM,
    SDC, // StaticKTAMCover
    SDC1DBindReplace,
    KBlock,
}

#[cfg(feature = "python")]
impl<'py> IntoPyObject<'py> for SystemEnum {
    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            SystemEnum::KTAM(ktam) => ktam.into_bound_py_any(py),
            SystemEnum::OldKTAM(oldktam) => oldktam.into_bound_py_any(py),
            SystemEnum::ATAM(atam) => atam.into_bound_py_any(py),
            SystemEnum::SDC(sdc) => sdc.into_bound_py_any(py),
            SystemEnum::KBlock(kblock) => kblock.into_bound_py_any(py),
            SystemEnum::SDC1DBindReplace(sdc1dbr) => sdc1dbr.into_bound_py_any(py),
        }
    }

    type Target = pyo3::PyAny; // the Python type
    type Output = pyo3::Bound<'py, Self::Target>; // in most cases this will be `Bound`
    type Error = pyo3::PyErr;
}

#[enum_dispatch]
pub trait TileBondInfo {
    fn tile_color(&self, tile_number: Tile) -> [u8; 4] {
        self.tile_colors()[tile_number as usize]
    }
    fn tile_name(&self, tile_number: Tile) -> &str {
        &self.tile_names()[tile_number as usize]
    }
    fn bond_name(&self, bond_number: usize) -> &str {
        &self.bond_names()[bond_number]
    }

    fn tile_colors(&self) -> &Vec<[u8; 4]>;
    fn tile_names(&self) -> &[String];
    fn bond_names(&self) -> &[String];

    /// By default, we will make a tile be just a solid clor, but a system may override this to
    /// customize how a tile looks.
    fn tile_style(&self, tile_number: Tile) -> TileStyle {
        let color = self.tile_color(tile_number);
        let tri_colors = [color, color, color, color];
        TileStyle { tri_colors }
    }

    /// Turn the tile into a sprite
    fn tile_pixels(&self, tile_number: Tile, size: usize) -> SpriteSquare {
        self.tile_style(tile_number).as_sprite(size)
    }

    /// Return a bitmask of sides that have blockers attached.
    /// Bit layout: 0b_WESN (bit 0 = North, bit 1 = East, bit 2 = South, bit 3 = West).
    /// Default: no blockers.
    fn tile_blocker_mask(&self, _tile_number: Tile) -> u8 {
        0
    }
}

pub trait SystemInfo {
    fn tile_concs(&self) -> Vec<f64>;
    fn tile_stoics(&self) -> Vec<f64>;
}
