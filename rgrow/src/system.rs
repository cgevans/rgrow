use bpci::Interval;
use bpci::WilsonScore;
use enum_dispatch::enum_dispatch;
use ndarray::prelude::*;
use num_traits::Zero;
use rand::rng;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::base::RgrowError;
use crate::base::StringConvError;
use crate::ffs::FFSRunConfig;
use crate::ffs::FFSRunResult;
use crate::models::atam::ATAM;
use crate::models::kblock::KBlock;
use crate::models::ktam::KTAM;
use crate::models::oldktam::OldKTAM;
use crate::models::sdc1d::SDC;
use crate::state::State;
use crate::state::StateEnum;
use crate::state::StateStatus;
use crate::units::Molar;
use crate::units::MolarPerSecond;
use crate::units::Second;
use crate::units::{PerSecond, Rate};

use crate::{
    base::GrowError, base::NumEvents, base::NumTiles, canvas::PointSafeHere, state::StateWithCreate,
};

use super::base::{Point, Tile};
use crate::canvas::PointSafe2;

use std::any::Any;
use std::fmt::Debug;

use std::time::Duration;

use rayon::prelude::*;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
use pyo3::{types::PyModule, IntoPyObjectExt};

#[derive(Clone, Debug)]
pub enum Event {
    None,
    MonomerAttachment(PointSafe2, Tile),
    MonomerDetachment(PointSafe2),
    MonomerChange(PointSafe2, Tile),
    PolymerAttachment(Vec<(PointSafe2, Tile)>),
    PolymerDetachment(Vec<PointSafe2>),
    PolymerChange(Vec<(PointSafe2, Tile)>),
}

#[derive(Debug)]
pub enum StepOutcome {
    HadEventAt(Second),
    NoEventIn(Second),
    DeadEventAt(Second),
    ZeroRate,
}

#[derive(Debug, PartialEq, Eq)]
#[cfg_attr(feature = "python", pyclass(eq, eq_int, module = "rgrow"))]
pub enum NeededUpdate {
    None,
    NonZero,
    All,
}

#[derive(Debug, Copy, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyclass(module = "rgrow"))]
pub struct EvolveBounds {
    /// Stop if this number of events has taken place during this evolve call.
    pub for_events: Option<NumEvents>,
    /// Stop if this number of events has been reached in total for the state.
    pub total_events: Option<NumEvents>,
    /// Stop if this amount of (simulated) time has passed during this evolve call.
    pub for_time: Option<f64>,
    /// Stop if this amount of (simulated) time has passed in total for the state.
    pub total_time: Option<f64>,
    /// Stop if the number of tiles is equal to or less than this number.
    pub size_min: Option<NumTiles>,
    /// Stop if the number of tiles is equal to or greater than this number.
    pub size_max: Option<NumTiles>,
    /// Stop after this amount of (real) time has passed.
    pub for_wall_time: Option<Duration>,
}

pub use rgrow_ipc::ParameterInfo;

#[cfg(feature = "python")]
#[pymethods]
impl EvolveBounds {
    #[new]
    #[pyo3(signature = (for_events=None, for_time=None, size_min=None, size_max=None, for_wall_time=None))]
    pub fn new(
        for_events: Option<NumEvents>,
        for_time: Option<f64>,
        size_min: Option<NumTiles>,
        size_max: Option<NumTiles>,
        for_wall_time: Option<f64>,
    ) -> Self {
        Self {
            for_events,
            for_time,
            size_min,
            size_max,
            for_wall_time: for_wall_time.map(Duration::from_secs_f64),
            ..Default::default()
        }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "EvolveBounds(events={}, time={}, size_min={}, size_max={}, wall_time={})",
            self.for_events
                .map_or("None".to_string(), |v| format!("{v:?}")),
            self.for_time
                .map_or("None".to_string(), |v| format!("{v:?}")),
            self.size_min
                .map_or("None".to_string(), |v| format!("{v:?}")),
            self.size_max
                .map_or("None".to_string(), |v| format!("{v:?}")),
            self.for_wall_time
                .map_or("None".to_string(), |v| format!("{v:?}"))
        )
    }
}

#[cfg_attr(feature = "python", pymethods)]
impl EvolveBounds {
    /// Will the EvolveBounds actually bound anything, or is it just null, such that the simulation will continue
    /// until a ZeroRate or an error?  Note that this includes weak bounds (size minimum and maximum) that may
    /// never be reached.
    pub fn is_weakly_bounded(&self) -> bool {
        self.for_events.is_some()
            || self.total_events.is_some()
            || self.for_time.is_some()
            || self.total_time.is_some()
            || self.size_min.is_some()
            || self.size_max.is_some()
            || self.for_wall_time.is_some()
    }
    pub fn is_strongly_bounded(&self) -> bool {
        self.for_events.is_some()
            || self.total_events.is_some()
            || self.for_time.is_some()
            || self.total_time.is_some()
            || self.for_wall_time.is_some()
    }
}

impl EvolveBounds {
    pub fn for_time(mut self, time: f64) -> Self {
        self.for_time = Some(time);
        self
    }

    pub fn for_events(mut self, events: NumEvents) -> Self {
        self.for_events = Some(events);
        self
    }
}

#[cfg_attr(feature = "python", pyclass(eq, eq_int, module = "rgrow"))]
#[derive(Debug, Clone, PartialEq, Eq)]
#[repr(C)]
pub enum EvolveOutcome {
    ReachedEventsMax,
    ReachedTimeMax,
    ReachedWallTimeMax,
    ReachedSizeMin,
    ReachedSizeMax,
    ReachedZeroRate,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "python", pyclass(eq, eq_int, module = "rgrow"))]
pub enum Orientation {
    NS,
    WE,
}
#[derive(Clone, Debug)]
#[cfg_attr(feature = "python", pyclass(get_all, set_all, module = "rgrow"))]
pub struct DimerInfo {
    pub t1: Tile,
    pub t2: Tile,
    pub orientation: Orientation,
    pub formation_rate: MolarPerSecond,
    pub equilibrium_conc: Molar,
}

#[cfg(feature = "python")]
#[pymethods]
impl DimerInfo {
    pub fn __repr__(&self) -> String {
        format!("{self:?}")
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "python", pyclass(eq, eq_int, module = "rgrow"))]
pub enum ChunkHandling {
    #[serde(alias = "none")]
    None,
    #[serde(alias = "detach")]
    Detach,
    #[serde(alias = "equilibrium")]
    Equilibrium,
}

impl TryFrom<&str> for ChunkHandling {
    type Error = StringConvError;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        match s {
            "none" => Ok(Self::None),
            "detach" => Ok(Self::Detach),
            "equilibrium" => Ok(Self::Equilibrium),
            _ => Err(StringConvError(format!("Unknown chunk handling: {s}. Valid values are \"none\", \"detach\", \"equilibrium\"."))),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "python", pyclass(eq, eq_int, module = "rgrow"))]
pub enum ChunkSize {
    #[serde(alias = "single")]
    Single,
    #[serde(alias = "dimer")]
    Dimer,
}

impl TryFrom<&str> for ChunkSize {
    type Error = StringConvError;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        match s {
            "single" => Ok(Self::Single),
            "dimer" => Ok(Self::Dimer),
            _ => Err(StringConvError(format!(
                "Unknown chunk size: {s}. Valid values are \"single\" and \"dimer\"."
            ))),
        }
    }
}

/// Result of a critical state search.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "python", pyclass(module = "rgrow"))]
pub struct CriticalStateResult {
    /// The critical state found.
    pub state: StateEnum,
    /// Energy at the critical state.
    pub energy: f64,
    /// Index in the trajectory where the critical state was found.
    pub trajectory_index: usize,
    /// Whether the state is above threshold.
    pub is_above_threshold: bool,
    /// Estimated committer probability.
    pub probability: f64,
    /// Number of trials used in the calculation.
    pub num_trials: usize,
    /// Whether max trials was exceeded.
    pub max_trials_exceeded: bool,
}

#[cfg(feature = "python")]
#[pymethods]
impl CriticalStateResult {
    #[getter]
    fn get_state(&self) -> crate::python::PyState {
        crate::python::PyState(self.state.clone())
    }

    #[getter]
    fn get_energy(&self) -> f64 {
        self.energy
    }

    #[getter]
    fn get_trajectory_index(&self) -> usize {
        self.trajectory_index
    }

    #[getter]
    fn get_is_above_threshold(&self) -> bool {
        self.is_above_threshold
    }

    #[getter]
    fn get_probability(&self) -> f64 {
        self.probability
    }

    #[getter]
    fn get_num_trials(&self) -> usize {
        self.num_trials
    }

    #[getter]
    fn get_max_trials_exceeded(&self) -> bool {
        self.max_trials_exceeded
    }

    fn __repr__(&self) -> String {
        format!(
            "CriticalStateResult(energy={:.4}, trajectory_index={}, is_above={}, prob={:.4}, trials={})",
            self.energy, self.trajectory_index, self.is_above_threshold, self.probability, self.num_trials
        )
    }
}

// ============================================================================
// Critical State Finding
// ============================================================================

/// Configuration for critical state search algorithms.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyclass(get_all, set_all, module = "rgrow"))]
pub struct CriticalStateConfig {
    /// Cutoff size for committer calculation (tiles above which growth is considered successful).
    pub cutoff_size: NumTiles,
    /// Probability threshold for determining if state is "critical" (above/below this).
    pub threshold: f64,
    /// Confidence level for the threshold test.
    pub confidence_level: f64,
    /// Maximum number of trials for committer calculation.
    pub max_trials: usize,
    /// Confidence level for the confidence interval (if requested).
    pub ci_confidence_level: f64,
    /// Canvas size for state reconstruction.
    pub canvas_size: (usize, usize),
    /// Canvas type for state reconstruction.
    pub canvas_type: crate::tileset::CanvasType,
}

impl Default for CriticalStateConfig {
    fn default() -> Self {
        Self {
            cutoff_size: 100,
            threshold: 0.5,
            confidence_level: 0.98,
            max_trials: 100000,
            ci_confidence_level: 0.95,
            canvas_size: (32, 32),
            canvas_type: crate::tileset::CanvasType::Periodic,
        }
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl CriticalStateConfig {
    #[new]
    #[pyo3(signature = (
        cutoff_size=None,
        threshold=None,
        confidence_level=None,
        max_trials=None,
        ci_confidence_level=None,
        canvas_size=None,
        canvas_type=None,
    ))]
    fn new(
        cutoff_size: Option<NumTiles>,
        threshold: Option<f64>,
        confidence_level: Option<f64>,
        max_trials: Option<usize>,
        ci_confidence_level: Option<f64>,
        canvas_size: Option<(usize, usize)>,
        canvas_type: Option<crate::tileset::CanvasType>,
    ) -> Self {
        let mut config = Self::default();
        if let Some(x) = cutoff_size {
            config.cutoff_size = x;
        }
        if let Some(x) = threshold {
            config.threshold = x;
        }
        if let Some(x) = confidence_level {
            config.confidence_level = x;
        }
        if let Some(x) = max_trials {
            config.max_trials = x;
        }
        if let Some(x) = ci_confidence_level {
            config.ci_confidence_level = x;
        }
        if let Some(x) = canvas_size {
            config.canvas_size = x;
        }
        if let Some(x) = canvas_type {
            config.canvas_type = x;
        }
        config
    }

    fn __repr__(&self) -> String {
        format!(
            "CriticalStateConfig(cutoff_size={}, threshold={}, confidence_level={}, max_trials={}, ci_confidence_level={}, canvas_size={:?}, canvas_type={:?})",
            self.cutoff_size, self.threshold, self.confidence_level, self.max_trials, self.ci_confidence_level, self.canvas_size, self.canvas_type
        )
    }
}

pub trait System: Debug + Sync + Send + TileBondInfo + Clone {
    fn new_state<St: StateWithCreate + State>(&self, params: St::Params) -> Result<St, GrowError> {
        let mut new_state = St::empty(params)?;
        self.configure_empty_state(&mut new_state)?;
        Ok(new_state)
    }

    fn system_info(&self) -> String;

    fn calc_n_tiles<St: State>(&self, state: &St) -> NumTiles {
        state.calc_n_tiles()
    }

    fn take_single_step<St: State>(&self, state: &mut St, max_time_step: Second) -> StepOutcome {
        let total_rate = state.total_rate();
        let time_step = -f64::ln(rng().random()) / total_rate;
        if time_step > max_time_step {
            state.add_time(max_time_step);
            return StepOutcome::NoEventIn(max_time_step);
        }
        let (point, remainder) = state.choose_point(); // todo: resultify
        let (event, chosen_event_rate) = self.choose_event_at_point(
            state,
            PointSafe2(point),
            PerSecond::from_per_second(remainder),
        ); // FIXME
        if let Event::None = event {
            state.add_time(time_step);
            return StepOutcome::DeadEventAt(time_step);
        }

        let energy_change = self.perform_event(state, &event);
        self.update_after_event(state, &event);
        state.add_time(time_step);
        state.add_events(1);
        state.record_event(
            &event,
            total_rate,
            chosen_event_rate,
            energy_change,
            state.energy(),
            state.n_tiles(),
        );
        StepOutcome::HadEventAt(time_step)
    }

    fn evolve<St: State>(
        &self,
        state: &mut St,
        bounds: EvolveBounds,
    ) -> Result<EvolveOutcome, GrowError> {
        let mut events = 0;

        if bounds.total_events.is_some() {
            return Err(GrowError::NotImplemented(
                "Total events bound is not implemented".to_string(),
            ));
        }

        let mut rtime = match bounds.for_time {
            Some(t) => Second::new(t),
            None => Second::new(f64::INFINITY),
        };
        if let Some(t) = bounds.total_time {
            rtime = rtime.min(Second::new(t) - state.time());
        }

        // If we have a for_wall_time, get an instant to compare to
        let start_time = bounds.for_wall_time.map(|_| std::time::Instant::now());

        loop {
            if bounds.size_min.is_some_and(|ms| state.n_tiles() <= ms) {
                return Ok(EvolveOutcome::ReachedSizeMin);
            } else if bounds.size_max.is_some_and(|ms| state.n_tiles() >= ms) {
                return Ok(EvolveOutcome::ReachedSizeMax);
            } else if rtime <= Second::new(0.) {
                return Ok(EvolveOutcome::ReachedTimeMax);
            } else if bounds
                .for_wall_time
                .is_some_and(|t| start_time.unwrap().elapsed() >= t)
            {
                return Ok(EvolveOutcome::ReachedWallTimeMax);
            } else if bounds.for_events.is_some_and(|e| events >= e) {
                return Ok(EvolveOutcome::ReachedEventsMax);
            } else if state.total_rate().is_zero() {
                return Ok(EvolveOutcome::ReachedZeroRate);
            }
            let out = self.take_single_step(state, rtime);
            match out {
                StepOutcome::HadEventAt(t) => {
                    events += 1;
                    rtime -= t;
                }
                StepOutcome::NoEventIn(_) => return Ok(EvolveOutcome::ReachedTimeMax),
                StepOutcome::DeadEventAt(t) => {
                    rtime -= t;
                }
                StepOutcome::ZeroRate => {
                    return Ok(EvolveOutcome::ReachedZeroRate);
                }
            }
        }
    }

    fn evolve_states<St: State>(
        &mut self,
        states: &mut [St],
        bounds: EvolveBounds,
    ) -> Vec<Result<EvolveOutcome, GrowError>> {
        states
            .par_iter_mut()
            .map(|state| self.evolve(state, bounds))
            .collect()
    }

    fn set_point<St: State>(
        &self,
        state: &mut St,
        point: Point,
        tile: Tile,
    ) -> Result<&Self, GrowError> {
        if !state.inbounds(point) {
            Err(GrowError::OutOfBounds(point.0, point.1))
        } else {
            Ok(self.set_safe_point(state, PointSafe2(point), tile))
        }
    }

    fn set_safe_point<St: State>(&self, state: &mut St, point: PointSafe2, tile: Tile) -> &Self {
        let event = Event::MonomerChange(point, tile);

        self.perform_event(state, &event);
        self.update_after_event(state, &event);

        self
    }

    fn set_points<St: State>(&self, state: &mut St, changelist: &[(Point, Tile)]) -> &Self {
        for (point, _) in changelist {
            assert!(state.inbounds(*point))
        }
        let event = Event::PolymerChange(
            changelist
                .iter()
                .map(|(p, t)| (PointSafe2(*p), *t))
                .collect(),
        );
        self.perform_event(state, &event);
        self.update_after_event(state, &event);
        self
    }

    fn set_safe_points<St: State>(
        &self,
        state: &mut St,
        changelist: &[(PointSafe2, Tile)],
    ) -> &Self {
        // for (point, _) in changelist {
        //     assert!(state.inbounds(*point))
        // }
        let event = Event::PolymerChange(changelist.to_vec());
        self.perform_event(state, &event);
        self.update_after_event(state, &event);
        self
    }

    /// Place a tile at a particular location, handling double tiles appropriately for kTAM.
    /// For kTAM, placing a "real" tile (left/top part of double tile) will also place the
    /// corresponding "fake" tile (right/bottom part). Attempting to place a "fake" tile
    /// directly will place the corresponding "real" tile instead.
    ///
    /// Returns energy change caused by placement, or NaN if energy is not calculated.
    fn place_tile<St: State>(
        &self,
        state: &mut St,
        point: PointSafe2,
        tile: Tile,
    ) -> Result<f64, GrowError> {
        // Default implementation: just place the tile directly
        self.set_safe_point(state, point, tile);
        Ok(f64::NAN)
    }

    fn configure_empty_state<St: State>(&self, state: &mut St) -> Result<(), GrowError> {
        for (p, t) in self.seed_locs() {
            self.set_point(state, p.0, t)?;
        }
        state.record_event(
            &Event::PolymerAttachment(self.seed_locs()),
            PerSecond::zero(),
            0.,
            0.,
            0.,
            self.seed_locs().len() as u32,
        );
        Ok(())
    }

    /// Perform a particular event/change to a state.  Do not update the state's time/etc,
    /// or rates, which should be done in update_after_event and take_single_step.
    fn perform_event<St: State>(&self, state: &mut St, event: &Event) -> f64 {
        match event {
            Event::None => panic!("Being asked to perform null event."),
            Event::MonomerAttachment(point, tile) | Event::MonomerChange(point, tile) => {
                state.set_sa(point, tile);
            }
            Event::MonomerDetachment(point) => {
                state.set_sa(point, &0);
            }
            Event::PolymerAttachment(changelist) | Event::PolymerChange(changelist) => {
                for (point, tile) in changelist {
                    state.set_sa(point, tile);
                }
            }
            Event::PolymerDetachment(changelist) => {
                for point in changelist {
                    state.set_sa(point, &0);
                }
            }
        };
        f64::NAN // FIXME: should return the energy change
    }

    fn update_after_event<St: State>(&self, state: &mut St, event: &Event);

    /// Returns the total event rate at a given point.  These should correspond with the events chosen by `choose_event_at_point`.
    fn event_rate_at_point<St: State>(&self, state: &St, p: PointSafeHere) -> PerSecond;

    /// Given a point, and an accumulated random rate choice `acc` (which should be less than the total rate at the point),
    /// return the event that should take place, and the rate of that particular event.
    fn choose_event_at_point<St: State>(
        &self,
        state: &St,
        p: PointSafe2,
        acc: PerSecond,
    ) -> (Event, f64);

    /// Returns a vector of (point, tile number) tuples for the seed tiles, useful for populating an initial state.
    fn seed_locs(&self) -> Vec<(PointSafe2, Tile)>;

    /// Returns an array of mismatch locations.  At each point, mismatches are designated by 8*N+4*E+2*S+1*W.
    fn calc_mismatch_locations<St: State>(&self, state: &St) -> Array2<usize>;

    fn calc_mismatches<St: State>(&self, state: &St) -> usize {
        let mut arr = self.calc_mismatch_locations(state);
        arr.map_inplace(|x| *x = (*x & 0b01) + ((*x & 0b10) / 2));
        arr.sum()
    }

    fn update_points<St: State>(&self, state: &mut St, points: &[PointSafeHere]) {
        let p = points
            .iter()
            .map(|p| (*p, self.event_rate_at_point(state, *p)))
            .collect::<Vec<_>>();

        state.update_multiple(&p);
    }

    fn update_state<St: State>(&self, state: &mut St, needed: &NeededUpdate) {
        let ncols = state.ncols();
        let nrows = state.nrows();

        let all_points = match needed {
            NeededUpdate::None => todo!(),
            NeededUpdate::NonZero => (0..nrows)
                .flat_map(|r| (0..ncols).map(move |c| PointSafeHere((r, c))))
                .filter(|p| state.rate_at_point(*p) > PerSecond::zero())
                .collect::<Vec<_>>(),
            NeededUpdate::All => (0..nrows)
                .flat_map(|r| (0..ncols).map(move |c| PointSafeHere((r, c))))
                .collect::<Vec<_>>(),
        };

        self.update_points(state, &all_points);

        if *needed == NeededUpdate::All {
            state.set_n_tiles(state.calc_n_tiles());
        };
    }

    fn set_param(&mut self, _name: &str, _value: Box<dyn Any>) -> Result<NeededUpdate, GrowError> {
        todo!();
    }

    fn get_param(&self, _name: &str) -> Result<Box<dyn Any>, GrowError> {
        todo!();
    }

    fn list_parameters(&self) -> Vec<ParameterInfo> {
        Vec::new()
    }

    fn extract_model_name(info: &str) -> String {
        if info.starts_with("kTAM") {
            "kTAM".to_string()
        } else if info.starts_with("aTAM") {
            "aTAM".to_string()
        } else if info.starts_with("Old kTAM") || info.starts_with("OldkTAM") {
            "Old kTAM".to_string()
        } else if info.starts_with("SDC") || info.contains("SDC") {
            "SDC".to_string()
        } else if info.starts_with("KBlock") {
            "KBlock".to_string()
        } else {
            "Unknown".to_string()
        }
    }
    fn evolve_in_window<St: State>(
        &mut self,
        state: &mut St,
        block: Option<usize>,
        start_paused: bool,
        mut bounds: EvolveBounds,
        initial_timescale: Option<f64>,
        initial_max_events_per_sec: Option<u64>,
    ) -> Result<EvolveOutcome, RgrowError> {
        use crate::ui::ipc::{ControlMessage, InitMessage, UpdateNotification};
        use crate::ui::ipc_server::IpcClient;
        use std::process::{Command, Stdio};
        use std::time::{Duration, Instant};

        let debug_perf = std::env::var("RGROW_DEBUG_PERF").is_ok();

        let (width, height) = state.draw_size();
        let tile_colors_vec = self.tile_colors().clone();

        let scale = block.unwrap_or(8);

        let socket_path =
            std::env::temp_dir().join(format!("rgrow-gui-{}.sock", std::process::id()));
        let socket_path_str = socket_path.to_string_lossy().to_string();

        // Try to find rgrow-gui binary in multiple locations
        let gui_exe = find_gui_binary().ok_or_else(|| {
            RgrowError::IO(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!(
                    "rgrow-gui binary (version {}) not found. The GUI functionality requires the rgrow-gui package to be installed.\n\nFor Python installations, ensure rgrow-gui is installed:\n  pip install rgrow-gui\n\nFor Rust installations, ensure rgrow-gui is built and available on PATH:\n  cargo build --package rgrow-gui\n\nNote: GUI support may be optional in future versions.",
                    env!("CARGO_PKG_VERSION")
                )
            ))
        })?;

        let mut gui_process = Command::new(&gui_exe)
            .arg(&socket_path_str)
            .stdout(Stdio::null())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| {
                RgrowError::IO(std::io::Error::other(format!(
                    "Failed to spawn GUI process: {}. Make sure rgrow-gui is built.",
                    e
                )))
            })?;

        std::thread::sleep(Duration::from_millis(100));

        let mut ipc_client = IpcClient::connect(&socket_path).map_err(|e| {
            RgrowError::IO(std::io::Error::other(format!(
                "Failed to connect to GUI: {}",
                e
            )))
        })?;

        let shm_size = (width * height * scale as u32 * scale as u32 * 4) as usize;
        #[cfg(all(unix, not(target_os = "macos")))]
        let shm_path = format!("/dev/shm/rgrow-frame-{}", std::process::id());
        #[cfg(any(windows, target_os = "macos"))]
        let shm_path = std::env::temp_dir()
            .join(format!("rgrow-frame-{}", std::process::id()))
            .to_string_lossy()
            .to_string();

        let has_temperature = self.get_param("temperature").is_ok();
        let model_name = Self::extract_model_name(&self.system_info());
        let initial_temperature = if has_temperature {
            self.get_param("temperature")
                .ok()
                .and_then(|v| v.downcast_ref::<f64>().copied())
        } else {
            None
        };

        let mut parameters = self.list_parameters();
        for param in &mut parameters {
            if let Ok(value) = self.get_param(&param.name) {
                if let Some(f64_value) = value.downcast_ref::<f64>() {
                    param.current_value = *f64_value;
                }
            }
        }

        let init_msg = InitMessage {
            width,
            height,
            tile_colors: tile_colors_vec.clone(),
            block,
            shm_path: shm_path.clone(),
            shm_size,
            start_paused,
            model_name,
            has_temperature,
            initial_temperature,
            parameters,
            initial_timescale,
            initial_max_events_per_sec,
        };

        ipc_client.send_init(&init_msg).map_err(|e| {
            RgrowError::IO(std::io::Error::other(format!(
                "Failed to send init message: {}",
                e
            )))
        })?;

        // Wait for GUI to signal it's ready (up to 10 seconds)
        ipc_client
            .wait_for_ready(Duration::from_secs(10))
            .map_err(|e| {
                RgrowError::IO(std::io::Error::other(format!(
                    "GUI failed to become ready: {}",
                    e
                )))
            })?;

        // Control state
        let mut paused = start_paused;
        let mut remaining_step_events: Option<u64> = None;
        let mut max_events_per_sec: Option<u64> = initial_max_events_per_sec;
        let mut timescale: Option<f64> = initial_timescale;

        let mut evres: EvolveOutcome = EvolveOutcome::ReachedZeroRate;
        let mut frame_buffer = vec![0u8; shm_size];
        let mut last_frame_time = Instant::now();
        let mut events_this_second: u64 = 0;
        let mut second_start = Instant::now();

        loop {
            // Process control messages
            while let Some(ctrl) = ipc_client.try_recv_control() {
                if debug_perf {
                    eprintln!("[Sim] Received control message: {:?}", ctrl);
                }
                match ctrl {
                    ControlMessage::Pause => {
                        paused = true;
                        remaining_step_events = None;
                    }
                    ControlMessage::Resume => {
                        paused = false;
                        remaining_step_events = None;
                    }
                    ControlMessage::Step { events } => {
                        paused = false;
                        remaining_step_events = Some(events);
                    }
                    ControlMessage::SetMaxEventsPerSec(max) => {
                        max_events_per_sec = max;
                    }
                    ControlMessage::SetTimescale(ts) => {
                        timescale = ts;
                    }
                    ControlMessage::SetTemperature(temp) => {
                        if let Ok(needed) = self.set_param("temperature", Box::new(temp)) {
                            self.update_state(state, &needed);
                        }
                    }
                    ControlMessage::SetParameter { name, value } => {
                        if let Ok(needed) = self.set_param(&name, Box::new(value)) {
                            self.update_state(state, &needed);
                        }
                    }
                }
            }

            // Reset events counter each second
            if second_start.elapsed() >= Duration::from_secs(1) {
                events_this_second = 0;
                second_start = Instant::now();
            }

            // Determine if we should run simulation this frame
            let should_run = !paused || remaining_step_events.is_some();

            if should_run {
                // Calculate bounds based on speed settings
                let events_before = state.total_events();

                if let Some(ts) = timescale {
                    // Timescale mode: run for (real_elapsed * timescale) simulation time
                    let real_elapsed = last_frame_time.elapsed().as_secs_f64();
                    let target_sim_time = real_elapsed * ts;
                    bounds.for_time = Some(target_sim_time);
                    bounds.for_wall_time = None;
                    bounds.for_events = remaining_step_events;
                } else if let Some(ref mut step_events) = remaining_step_events {
                    // Step mode: run for specified events
                    bounds.for_events = Some(*step_events);
                    bounds.for_wall_time = Some(Duration::from_millis(16));
                    bounds.for_time = None;
                } else {
                    // Normal mode
                    bounds.for_wall_time = Some(Duration::from_millis(16));
                    bounds.for_events = None;
                    bounds.for_time = None;
                }

                // Check events per second limit
                if let Some(max_eps) = max_events_per_sec {
                    if events_this_second >= max_eps {
                        // Already hit limit this second, skip evolution
                        std::thread::sleep(Duration::from_millis(10));
                    } else {
                        let remaining_allowed = max_eps - events_this_second;
                        if let Some(ref mut be) = bounds.for_events {
                            *be = (*be).min(remaining_allowed);
                        } else {
                            bounds.for_events = Some(remaining_allowed);
                        }
                        evres = self.evolve(state, bounds)?;
                    }
                } else {
                    evres = self.evolve(state, bounds)?;
                }

                let events_this_frame = state.total_events() - events_before;
                events_this_second += events_this_frame;

                // Update step counter
                if let Some(ref mut step_events) = remaining_step_events {
                    if events_this_frame >= *step_events {
                        remaining_step_events = None;
                        paused = true;
                    } else {
                        *step_events -= events_this_frame;
                    }
                }
            }

            last_frame_time = Instant::now();

            // Draw frame
            let edge_size = scale / 10;
            let tile_size = scale - 2 * edge_size;
            let frame_width = (width * scale as u32) as usize;
            let frame_height = (height * scale as u32) as usize;
            frame_buffer.resize(frame_width * frame_height * 4, 0);

            let pixel_frame = &mut frame_buffer[..];

            if scale != 1 {
                if edge_size == 0 {
                    state.draw_scaled(pixel_frame, &tile_colors_vec, tile_size, edge_size);
                } else {
                    state.draw_scaled_with_mm(
                        pixel_frame,
                        &tile_colors_vec,
                        self.calc_mismatch_locations(state),
                        tile_size,
                        edge_size,
                    );
                }
            } else {
                state.draw(pixel_frame, &tile_colors_vec);
            }

            let notification = UpdateNotification {
                frame_width: frame_width as u32,
                frame_height: frame_height as u32,
                time: state.time().into(),
                total_events: state.total_events(),
                n_tiles: state.n_tiles(),
                mismatches: self.calc_mismatches(state) as u32,
                energy: state.energy(),
                scale,
                data_len: pixel_frame.len(),
            };

            let t_send = Instant::now();
            if ipc_client.send_frame(pixel_frame, notification).is_err() {
                break;
            }
            let t_send_elapsed = t_send.elapsed();

            if debug_perf {
                eprintln!(
                    "[IPC-send] shm write + notify: {:?}, size: {} bytes",
                    t_send_elapsed,
                    frame_buffer.len()
                );
            }

            std::thread::sleep(Duration::from_millis(16));

            // Only break on terminal conditions if not paused
            // Continue running for: wall time limit, time limit, events limit, zero rate
            // These are all normal "frame complete" conditions
            if !paused && remaining_step_events.is_none() {
                match evres {
                    EvolveOutcome::ReachedWallTimeMax => {}
                    EvolveOutcome::ReachedTimeMax => {}
                    EvolveOutcome::ReachedEventsMax => {}
                    EvolveOutcome::ReachedZeroRate => {}
                    _ => {
                        break;
                    }
                }
            }
        }

        let _ = ipc_client.send_close();
        let _ = gui_process.wait();
        let _ = std::fs::remove_file(&socket_path);

        Ok(evres)
    }

    /// Returns information on dimers that the system can form.
    fn calc_dimers(&self) -> Result<Vec<DimerInfo>, GrowError> {
        Err(GrowError::NotSupported(
            "Dimer calculation not supported by this system".to_string(),
        ))
    }

    fn clone_state<St: StateWithCreate>(&self, initial_state: &St) -> St {
        // Default here is to just clone the state
        initial_state.clone()
    }

    fn clone_state_into_state<St: StateWithCreate>(&self, initial_state: &St, target: &mut St) {
        target.clone_from(initial_state);
    }

    fn clone_state_into_empty_state<St: StateWithCreate>(
        &self,
        initial_state: &St,
        target: &mut St,
    ) {
        self.clone_state_into_state(initial_state, target);
    }
}

#[cfg_attr(test, allow(dead_code))]
#[cfg_attr(not(test), allow(dead_code))]
pub fn find_gui_binary() -> Option<std::path::PathBuf> {
    use std::process::Command;

    const EXPECTED_VERSION: &str = env!("CARGO_PKG_VERSION");

    // Helper to check version
    let check_version = |path: &std::path::Path| -> bool {
        if let Ok(output) = Command::new(path).arg("--version").output() {
            if let Ok(version_str) = String::from_utf8(output.stdout) {
                // Extract version from output (format: "rgrow-gui 0.20.0" or similar)
                let version = version_str.split_whitespace().last().unwrap_or("");
                return version == EXPECTED_VERSION;
            }
        }
        false
    };

    // 1. Check package directory (for Python installations where binary might be bundled)
    #[cfg(feature = "python")]
    {
        if let Ok(package_dir) = Python::attach(|py| -> PyResult<String> {
            let importlib = PyModule::import(py, "importlib.util")?;
            let spec = importlib.call_method1("find_spec", ("rgrow",))?;
            let origin = spec.getattr("origin")?;

            if origin.is_none() {
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "Could not find rgrow package",
                ));
            }

            let origin_str = origin.extract::<String>()?;
            let path = std::path::PathBuf::from(origin_str);

            if let Some(parent) = path.parent() {
                Ok(parent.to_string_lossy().to_string())
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "Could not determine package directory",
                ))
            }
        }) {
            let gui_exe = std::path::PathBuf::from(&package_dir).join("rgrow-gui");
            #[cfg(windows)]
            let gui_exe = std::path::PathBuf::from(&package_dir).join("rgrow-gui.exe");

            if gui_exe.exists() && check_version(&gui_exe) {
                return Some(gui_exe);
            }
        }
    }

    // 2. Check environment variable (set by Python if available)
    if let Ok(package_dir) = std::env::var("RGROW_PACKAGE_DIR") {
        let gui_exe = std::path::PathBuf::from(&package_dir).join("rgrow-gui");
        #[cfg(windows)]
        let gui_exe = std::path::PathBuf::from(&package_dir).join("rgrow-gui.exe");

        if gui_exe.exists() && check_version(&gui_exe) {
            return Some(gui_exe);
        }
    }

    // 3. Check PATH for rgrow-gui
    if let Ok(path) = which::which("rgrow-gui") {
        if check_version(&path) {
            return Some(path);
        } else {
            eprintln!(
                "Warning: Found rgrow-gui but version mismatch. Please update rgrow-gui to match rgrow version {}",
                env!("CARGO_PKG_VERSION")
            );
        }
    }

    // 4. Check in the same directory as the current executable (for Rust-only installations)
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            let gui_exe = exe_dir.join("rgrow-gui");
            #[cfg(windows)]
            let gui_exe = exe_dir.join("rgrow-gui.exe");

            if gui_exe.exists() && check_version(&gui_exe) {
                return Some(gui_exe);
            }
        }
    }

    None
}

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

    fn calc_committer(
        &mut self,
        initial_state: &StateEnum,
        cutoff_size: NumTiles,
        max_time: Option<f64>,
        max_events: Option<NumEvents>,
        num_trials: usize,
    ) -> Result<f64, GrowError>;

    fn calc_committer_adaptive(
        &self,
        initial_state: &StateEnum,
        cutoff_size: NumTiles,
        max_time: Option<f64>,
        max_events: Option<NumEvents>,
        conf_interval_margin: f64,
    ) -> Result<(f64, usize), GrowError>;

    fn calc_committers_adaptive(
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

    /// Determine whether the committer probability for a state is above or below a threshold
    /// with a specified confidence level using adaptive sampling.
    ///
    /// This function uses adaptive sampling with Wilson Score confidence intervals to determine
    /// with the desired confidence whether the true committer probability is above or below the
    /// given threshold. It continues sampling until the confidence interval is narrow enough to
    /// make a definitive determination, or until the maximum number of trials is reached.
    ///
    /// The committer probability is the probability that when a simulation is started from the
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
    /// * `probability_estimate` - The estimated committer probability (between 0.0 and 1.0)
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
    /// let (is_above, prob, trials, exceeded) = system.calc_committer_threshold_test(
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
    /// let (is_above, prob, trials, exceeded) = system.calc_committer_threshold_test(
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
    fn calc_committer_threshold_test(
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

    fn calc_committer(
        &mut self,
        initial_state: &StateEnum,
        cutoff_size: NumTiles,
        max_time: Option<f64>,
        max_events: Option<NumEvents>,
        num_trials: usize,
    ) -> Result<f64, GrowError> {
        if num_trials == 0 {
            return Err(GrowError::NotSupported(
                "Number of trials must be greater than 0".to_string(),
            ));
        }

        let mut successes = 0;

        let mut trial_states = (0..num_trials)
            .map(|_| initial_state.clone())
            .collect::<Vec<_>>();

        let bounds = EvolveBounds {
            size_min: Some(0),
            size_max: Some(cutoff_size),
            for_time: max_time,
            for_events: max_events,
            ..Default::default()
        };

        let outcomes = self.evolve_states(&mut trial_states, bounds);

        for outcome in outcomes.iter() {
            let outcome = outcome
                .as_ref()
                .map_err(|e| GrowError::NotSupported(e.to_string()))?;
            match outcome {
                EvolveOutcome::ReachedSizeMax => successes += 1,
                EvolveOutcome::ReachedSizeMin => {}
                _ => {
                    return Err(GrowError::NotSupported(
                        "Evolve outcome not supported".to_string(),
                    )); // FIXME: this should make more sense
                }
            }
        }

        Ok(successes as f64 / num_trials as f64)
    }

    fn calc_committer_adaptive(
        &self,
        initial_state: &StateEnum,
        cutoff_size: NumTiles,
        max_time: Option<f64>,
        max_events: Option<NumEvents>,
        conf_interval_margin: f64,
    ) -> Result<(f64, usize), GrowError> {
        use bpci::{NSuccessesSample, WilsonScore};

        let mut successes = 0u32;
        let mut num_trials = 0u32;

        let mut trial_state = initial_state.clone();

        let bounds = EvolveBounds {
            size_min: Some(0),
            size_max: Some(cutoff_size),
            for_time: max_time,
            for_events: max_events,
            ..Default::default()
        };

        while (NSuccessesSample::new(num_trials, successes)
            .unwrap()
            .wilson_score(1.960)
            .margin
            > conf_interval_margin)
            || num_trials < 1
        {
            let outcome = self.evolve(&mut trial_state, bounds)?;
            match outcome {
                EvolveOutcome::ReachedSizeMax => {
                    successes += 1;
                    num_trials += 1;
                    initial_state.clone_into(&mut trial_state);
                }
                EvolveOutcome::ReachedSizeMin => {
                    num_trials += 1;
                    initial_state.clone_into(&mut trial_state);
                }
                _ => {
                    return Err(GrowError::NotSupported(
                        "Evolve outcome not supported".to_string(),
                    )); // FIXME: this should make more sense
                }
            }
        }

        Ok((successes as f64 / num_trials as f64, num_trials as usize))
    }

    fn calc_committers_adaptive(
        &self,
        initial_states: &[&StateEnum],
        cutoff_size: NumTiles,
        max_time: Option<f64>,
        max_events: Option<NumEvents>,
        conf_interval_margin: f64,
    ) -> Result<(Vec<f64>, Vec<usize>), GrowError> {
        let results = initial_states
            .par_iter()
            .map(|initial_state| {
                self.calc_committer_adaptive(
                    initial_state,
                    cutoff_size,
                    max_time,
                    max_events,
                    conf_interval_margin,
                )
            })
            .collect::<Vec<_>>();

        let results: Vec<(f64, usize)> = results.into_iter().map(|r| r.unwrap()).collect();

        let committers: Vec<f64> = results.iter().map(|(c, _)| *c).collect();
        let trials: Vec<usize> = results.iter().map(|(_, t)| *t).collect();

        Ok((committers, trials))
    }

    fn calc_forward_probability(
        &mut self,
        initial_state: &StateEnum,
        forward_step: NumTiles,
        max_time: Option<f64>,
        max_events: Option<NumEvents>,
        num_trials: usize,
    ) -> Result<f64, GrowError> {
        if num_trials == 0 {
            return Err(GrowError::NotSupported(
                "Number of trials must be greater than 0".to_string(),
            ));
        }

        let initial_size = initial_state.n_tiles();
        let cutoff_size = initial_size + forward_step;

        let mut successes = 0;

        let mut trial_states = (0..num_trials)
            .map(|_| initial_state.clone())
            .collect::<Vec<_>>();

        let bounds = EvolveBounds {
            size_min: Some(0),
            size_max: Some(cutoff_size),
            for_time: max_time,
            for_events: max_events,
            ..Default::default()
        };

        let outcomes = self.evolve_states(&mut trial_states, bounds);

        for outcome in outcomes.iter() {
            let outcome = outcome
                .as_ref()
                .map_err(|e| GrowError::NotSupported(e.to_string()))?;
            match outcome {
                EvolveOutcome::ReachedSizeMax => successes += 1,
                EvolveOutcome::ReachedSizeMin => {}
                _ => {
                    return Err(GrowError::NotSupported(
                        "Evolve outcome not supported".to_string(),
                    ));
                }
            }
        }

        Ok(successes as f64 / num_trials as f64)
    }

    fn calc_forward_probability_adaptive(
        &self,
        initial_state: &StateEnum,
        forward_step: NumTiles,
        max_time: Option<f64>,
        max_events: Option<NumEvents>,
        conf_interval_margin: f64,
    ) -> Result<(f64, usize), GrowError> {
        use bpci::{NSuccessesSample, WilsonScore};

        let initial_size = initial_state.n_tiles();
        let cutoff_size = initial_size + forward_step;

        let mut successes = 0u32;
        let mut num_trials = 0u32;

        let mut trial_state = initial_state.clone();

        let bounds = EvolveBounds {
            size_min: Some(0),
            size_max: Some(cutoff_size),
            for_time: max_time,
            for_events: max_events,
            ..Default::default()
        };

        while (NSuccessesSample::new(num_trials, successes)
            .unwrap()
            .wilson_score(1.960)
            .margin
            > conf_interval_margin)
            || num_trials < 1
        {
            let outcome = self.evolve(&mut trial_state, bounds)?;
            match outcome {
                EvolveOutcome::ReachedSizeMax => {
                    successes += 1;
                    num_trials += 1;
                    initial_state.clone_into(&mut trial_state);
                }
                EvolveOutcome::ReachedSizeMin => {
                    num_trials += 1;
                    initial_state.clone_into(&mut trial_state);
                }
                _ => {
                    return Err(GrowError::NotSupported(
                        "Evolve outcome not supported".to_string(),
                    ));
                }
            }
        }

        Ok((successes as f64 / num_trials as f64, num_trials as usize))
    }

    fn calc_forward_probabilities_adaptive(
        &self,
        initial_states: &[&StateEnum],
        forward_step: NumTiles,
        max_time: Option<f64>,
        max_events: Option<NumEvents>,
        conf_interval_margin: f64,
    ) -> Result<(Vec<f64>, Vec<usize>), GrowError> {
        let results = initial_states
            .par_iter()
            .map(|initial_state| {
                self.calc_forward_probability_adaptive(
                    initial_state,
                    forward_step,
                    max_time,
                    max_events,
                    conf_interval_margin,
                )
            })
            .collect::<Vec<_>>();

        let results: Vec<(f64, usize)> = results.into_iter().map(|r| r.unwrap()).collect();

        let probabilities: Vec<f64> = results.iter().map(|(p, _)| *p).collect();
        let trials: Vec<usize> = results.iter().map(|(_, t)| *t).collect();

        Ok((probabilities, trials))
    }

    /// Implementation of committer threshold test using adaptive sampling.
    ///
    /// See trait documentation for detailed parameter and return value descriptions.
    /// This implementation uses Wilson Score confidence intervals for robust statistical inference.
    fn calc_committer_threshold_test(
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
        use bpci::NSuccessesSample;

        let n_par = rayon::current_num_threads();

        if !(0.0..=1.0).contains(&threshold) {
            return Err(GrowError::NotSupported(
                "Threshold must be between 0.0 and 1.0".to_string(),
            ));
        }

        let max_trials = max_trials.unwrap_or(100000);
        let mut successes = 0u32;
        let mut num_trials = 0u32;

        let mut trial_states = Vec::new();
        for _ in 0..n_par {
            trial_states.push(initial_state.clone());
        }

        let bounds = EvolveBounds {
            size_min: Some(0),
            size_max: Some(cutoff_size),
            for_time: max_time,
            for_events: max_events,
            ..Default::default()
        };

        // Continue sampling until we can determine with confidence whether
        // the probability is above or below the threshold
        loop {
            let outcomes = self.evolve_states(&mut trial_states, bounds);
            for outcome in outcomes {
                match outcome? {
                    EvolveOutcome::ReachedSizeMax => {
                        successes += 1;
                        num_trials += 1;
                    }
                    EvolveOutcome::ReachedSizeMin => {
                        num_trials += 1;
                    }
                    x => {
                        return Err(GrowError::NotSupported(format!(
                            "Evolve outcome not supported: {:?}",
                            x
                        )));
                    }
                }
            }

            // Need at least a few trials before we can make any statistical determination
            if num_trials < 10 {
                continue;
            }

            // Calculate Wilson score confidence interval for the threshold test
            // This uses the test confidence level to determine if we can make a decision
            let sample = NSuccessesSample::new(num_trials, successes).unwrap();
            let test_wilson = sample.wilson_score_with_cc(z_level);

            let test_lower_bound = test_wilson.lower();
            let test_upper_bound = test_wilson.upper();

            // Check if the test confidence interval excludes the threshold (definitive determination)
            if test_upper_bound < threshold {
                // We're confident the probability is below the threshold
                let probability_estimate = successes as f64 / num_trials as f64;
                return Ok((false, probability_estimate, num_trials as usize, false));
            } else if test_lower_bound > threshold {
                // We're confident the probability is above the threshold
                let probability_estimate = successes as f64 / num_trials as f64;
                return Ok((true, probability_estimate, num_trials as usize, false));
            }

            // Check if we've exceeded the maximum number of trials without reaching a decision
            if num_trials >= max_trials as u32 {
                let probability_estimate = successes as f64 / num_trials as f64;

                if return_on_max_trials {
                    // Return current best estimate with warning flag set
                    // Use simple point estimate comparison since we couldn't reach statistical confidence
                    let is_above_threshold = probability_estimate > threshold;
                    return Ok((
                        is_above_threshold,
                        probability_estimate,
                        num_trials as usize,
                        true,
                    ));
                } else {
                    // Raise error when max trials exceeded and user doesn't want fallback result
                    return Err(GrowError::NotSupported(format!(
                        "Maximum number of trials ({}) exceeded without reaching confidence",
                        max_trials
                    )));
                }
            }
            for state in trial_states.iter_mut() {
                initial_state.clone_into(state);
            }
        }
    }

    fn find_first_critical_state(
        &mut self,
        end_state: &StateEnum,
        config: &CriticalStateConfig,
    ) -> Result<Option<CriticalStateResult>, GrowError> {
        let _tracker = if let Some(tracker) = end_state.get_movie_tracker() {
            tracker
        } else {
            return Err(GrowError::NotSupported(
                "State does not have a movie tracker".to_string(),
            ));
        };

        let filtered_indices = end_state.filtered_movie_indices()?;

        if filtered_indices.is_empty() {
            return Ok(None);
        }

        for i in 0..filtered_indices.len() {
            let mut state = end_state.replay(Some(filtered_indices[i] as u64))?;
            self.update_state(&mut state, &NeededUpdate::All);

            let (is_above, prob, trials, exceeded) = self.calc_committer_threshold_test(
                &state,
                config.cutoff_size,
                config.threshold,
                config.confidence_level,
                None, // max_time
                None, // max_events
                Some(config.max_trials),
                true, // return_on_max_trials
            )?;

            if is_above {
                let orig_idx = filtered_indices[i];
                let energy = state.energy();

                return Ok(Some(CriticalStateResult {
                    state,
                    energy,
                    trajectory_index: orig_idx,
                    is_above_threshold: is_above,
                    probability: prob,
                    num_trials: trials,
                    max_trials_exceeded: exceeded,
                }));
            }
        }

        Ok(None)
    }

    fn find_last_critical_state(
        &mut self,
        end_state: &StateEnum,
        config: &CriticalStateConfig,
    ) -> Result<Option<CriticalStateResult>, GrowError> {
        let _tracker = if let Some(tracker) = end_state.get_movie_tracker() {
            tracker
        } else {
            return Err(GrowError::NotSupported(
                "State does not have a movie tracker".to_string(),
            ));
        };

        let filtered_indices = end_state.filtered_movie_indices()?;

        if filtered_indices.is_empty() {
            return Ok(None);
        }

        for i in (0..filtered_indices.len()).rev() {
            let mut state = end_state.replay(Some(filtered_indices[i] as u64))?;
            self.update_state(&mut state, &NeededUpdate::All);

            let (is_above, prob, trials, exceeded) = self.calc_committer_threshold_test(
                &state,
                config.cutoff_size,
                config.threshold,
                config.confidence_level,
                None, // max_time
                None, // max_events
                Some(config.max_trials),
                true, // return_on_max_trials
            )?;

            if !is_above {
                let orig_idx = filtered_indices[i];
                let energy = state.energy();

                return Ok(Some(CriticalStateResult {
                    state,
                    energy,
                    trajectory_index: orig_idx,
                    is_above_threshold: is_above,
                    probability: prob,
                    num_trials: trials,
                    max_trials_exceeded: exceeded,
                }));
            }
        }

        Ok(None)
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
        }
    }

    type Target = pyo3::PyAny; // the Python type
    type Output = pyo3::Bound<'py, Self::Target>; // in most cases this will be `Bound`
    type Error = pyo3::PyErr;
}

#[enum_dispatch]
pub trait TileBondInfo {
    fn tile_color(&self, tile_number: Tile) -> [u8; 4];
    fn tile_name(&self, tile_number: Tile) -> &str;
    fn bond_name(&self, bond_number: usize) -> &str;

    fn tile_colors(&self) -> &Vec<[u8; 4]>;
    fn tile_names(&self) -> Vec<&str>;
    fn bond_names(&self) -> Vec<&str>;
}

pub trait SystemInfo {
    fn tile_concs(&self) -> Vec<f64>;
    fn tile_stoics(&self) -> Vec<f64>;
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "python", pyclass(eq, eq_int, module = "rgrow"))]
pub enum FissionHandling {
    #[serde(alias = "off", alias = "no-fission")]
    NoFission,
    #[serde(alias = "just-detach", alias = "surface")]
    JustDetach,
    #[serde(alias = "on", alias = "keep-seeded")]
    KeepSeeded,
    #[serde(alias = "keep-largest")]
    KeepLargest,
    #[serde(alias = "keep-weighted")]
    KeepWeighted,
}

impl TryFrom<&str> for FissionHandling {
    type Error = StringConvError;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        match s {
            "off" | "no-fission" => Ok(FissionHandling::NoFission),
            "just-detach" | "surface" => Ok(FissionHandling::JustDetach),
            "on" | "keep-seeded" => Ok(FissionHandling::KeepSeeded),
            "keep-largest" => Ok(FissionHandling::KeepLargest),
            "keep-weighted" => Ok(FissionHandling::KeepWeighted),
            _ => Err(StringConvError(format!("Unknown fission handling mode: {s}. Valid values are: no-fission, just-detach, keep-seeded, keep-largest, keep-weighted"))),
        }
    }
}
