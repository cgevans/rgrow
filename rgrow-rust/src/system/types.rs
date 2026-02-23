use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::base::{NumEvents, NumTiles, StringConvError, Tile};
use crate::canvas::PointSafe2;
use crate::state::StateEnum;
use crate::units::{Molar, MolarPerSecond, Second};

pub use crate::ui::ipc::ParameterInfo;

#[cfg(feature = "python")]
use pyo3::prelude::*;

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
#[cfg_attr(feature = "python", pyclass(eq, eq_int, module = "rgrow.rgrow"))]
pub enum NeededUpdate {
    None,
    NonZero,
    All,
}

#[derive(Debug, Copy, Clone, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyclass(module = "rgrow.rgrow"))]
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

#[cfg_attr(feature = "python", pyclass(eq, eq_int, module = "rgrow.rgrow"))]
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
#[cfg_attr(feature = "python", pyclass(eq, eq_int, module = "rgrow.rgrow"))]
pub enum Orientation {
    NS,
    WE,
}
#[derive(Clone, Debug)]
#[cfg_attr(feature = "python", pyclass(get_all, set_all, module = "rgrow.rgrow"))]
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
#[cfg_attr(feature = "python", pyclass(eq, eq_int, module = "rgrow.rgrow"))]
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
#[cfg_attr(feature = "python", pyclass(eq, eq_int, module = "rgrow.rgrow"))]
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
#[cfg_attr(feature = "python", pyclass(module = "rgrow.rgrow"))]
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

/// Configuration for critical state search algorithms.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyclass(get_all, set_all, module = "rgrow.rgrow"))]
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

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "python", pyclass(eq, eq_int, module = "rgrow.rgrow"))]
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
