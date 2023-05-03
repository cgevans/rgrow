use ndarray::prelude::*;
use rand::thread_rng;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::base::CanvasLength;
use crate::state::State;
use crate::{
    base::GrowError, base::NumEvents, base::NumTiles, canvas::PointSafeHere, state::StateCreate,
};

use super::base::{Point, Rate, Tile};
use crate::canvas::PointSafe2;

use std::any::Any;
use std::fmt::Debug;
use std::time::Duration;

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
    HadEventAt(f64),
    NoEventIn(f64),
    DeadEventAt(f64),
    ZeroRate,
}

#[derive(Debug)]
pub enum NeededUpdate {
    None,
    NonZero,
    All,
}

#[derive(Debug, Copy, Clone, Default)]
#[cfg_attr(feature = "python", pyclass)]
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

#[cfg_attr(feature = "python", pyclass)]
#[derive(Debug, Clone)]
#[repr(C)]
pub enum EvolveOutcome {
    ReachedEventsMax,
    ReachedTimeMax,
    ReachedWallTimeMax,
    ReachedSizeMin,
    ReachedSizeMax,
    ReachedZeroRate,
}

#[derive(Clone, Debug)]
pub enum Orientation {
    NS,
    WE,
}
#[derive(Clone, Debug)]
pub struct DimerInfo {
    pub t1: Tile,
    pub t2: Tile,
    pub orientation: Orientation,
    pub formation_rate: Rate,
    pub equilibrium_conc: f64,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub enum ChunkHandling {
    #[serde(alias = "none")]
    None,
    #[serde(alias = "detach")]
    Detach,
    #[serde(alias = "equilibrium")]
    Equilibrium,
}

impl From<&str> for ChunkHandling {
    fn from(s: &str) -> Self {
        match s {
            "none" => Self::None,
            "detach" => Self::Detach,
            "equilibrium" => Self::Equilibrium,
            _ => panic!("Unknown chunk handling: {}", s),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub enum ChunkSize {
    #[serde(alias = "single")]
    Single,
    #[serde(alias = "dimer")]
    Dimer,
}

impl From<&str> for ChunkSize {
    fn from(s: &str) -> Self {
        match s {
            "single" => Self::Single,
            "dimer" => Self::Dimer,
            _ => panic!("Unknown chunk size: {}", s),
        }
    }
}

pub trait System: Debug + Sync + Send {
    fn new_state<St: StateCreate + State>(
        &self,
        shape: (CanvasLength, CanvasLength),
    ) -> Result<St, GrowError> {
        let mut new_state = St::empty(shape)?;
        self.insert_seed(&mut new_state)?;
        Ok(new_state)
    }

    fn create_we_pair<St: StateCreate + State>(
        &mut self,
        w: Tile,
        e: Tile,
        size: usize,
    ) -> Result<St, GrowError> {
        assert!(size > 8);
        let mut ret = St::empty((size, size))?;
        let mid = size / 2;
        // self.insert_seed(&mut ret);
        self.set_point(&mut ret, (mid, mid), w)?;
        self.set_point(&mut ret, (mid, mid + 1), e)?;
        Ok(ret)
    }

    fn create_ns_pair<St: StateCreate + State>(
        &mut self,
        n: Tile,
        s: Tile,
        size: usize,
    ) -> Result<St, GrowError> {
        assert!(size > 8);
        let mut ret = St::empty((size, size))?;
        let mid = size / 2;
        // self.insert_seed(&mut ret);
        self.set_point(&mut ret, (mid, mid), n)?;
        self.set_point(&mut ret, (mid + 1, mid), s)?;
        Ok(ret)
    }

    fn calc_ntiles<St: State>(&self, state: &St) -> NumTiles {
        state.calc_ntiles()
    }

    fn state_step<St: State>(&self, state: &mut St, max_time_step: f64) -> StepOutcome {
        let time_step = -f64::ln(thread_rng().gen()) / state.total_rate();
        if time_step > max_time_step {
            state.add_time(max_time_step);
            return StepOutcome::NoEventIn(max_time_step);
        }
        let (point, remainder) = state.choose_point(); // todo: resultify
        let event = self.choose_event_at_point(state, PointSafe2(point), remainder); // FIXME
        if let Event::None = event {
            return StepOutcome::DeadEventAt(time_step);
        }

        self.perform_event(state, &event);
        self.update_after_event(state, &event);
        state.add_time(time_step);
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
            Some(t) => t,
            None => f64::INFINITY,
        };
        if let Some(t) = bounds.total_time {
            rtime = rtime.min(t - state.time());
        }

        // If we have a for_wall_time, get an instant to compare to
        let start_time = bounds.for_wall_time.map(|_| std::time::Instant::now());

        loop {
            if bounds.size_min.is_some_and(|ms| state.ntiles() <= ms) {
                return Ok(EvolveOutcome::ReachedSizeMin);
            } else if bounds.size_max.is_some_and(|ms| state.ntiles() >= ms) {
                return Ok(EvolveOutcome::ReachedSizeMax);
            } else if rtime <= 0. {
                return Ok(EvolveOutcome::ReachedTimeMax);
            } else if bounds
                .for_wall_time
                .is_some_and(|t| start_time.unwrap().elapsed() >= t)
            {
                return Ok(EvolveOutcome::ReachedWallTimeMax);
            } else if bounds.for_events.is_some_and(|e| events >= e) {
                return Ok(EvolveOutcome::ReachedEventsMax);
            } else if state.total_rate() == 0. {
                return Ok(EvolveOutcome::ReachedZeroRate);
            }
            let out = self.state_step(state, rtime);
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

        self.perform_event(state, &event)
            .update_after_event(state, &event);

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
        self.perform_event(state, &event)
            .update_after_event(state, &event);
        self
    }

    fn insert_seed<St: State>(&self, state: &mut St) -> Result<(), GrowError> {
        for (p, t) in self.seed_locs() {
            self.set_point(state, p.0, t)?;
        }
        Ok(())
    }

    fn perform_event<St: State>(&self, state: &mut St, event: &Event) -> &Self {
        //state.record_event(&event);
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
        state.add_events(1);
        self
    }

    fn update_after_event<St: State>(&self, state: &mut St, event: &Event);

    /// Returns the total event rate at a given point.  These should correspond with the events chosen by `choose_event_at_point`.
    fn event_rate_at_point<St: State>(&self, state: &St, p: PointSafeHere) -> Rate;

    /// Given a point, and an accumulated random rate choice `acc` (which should be less than the total rate at the point),
    /// return the event that should take place.
    fn choose_event_at_point<St: State>(&self, state: &St, p: PointSafe2, acc: Rate) -> Event;

    /// Returns a vector of (point, tile number) tuples for the seed tiles, useful for populating an initial state.
    fn seed_locs(&self) -> Vec<(PointSafe2, Tile)>;

    /// Returns an array of mismatch locations.  At each point, mismatches are designated by 8*N+4*E+2*S+1*W.
    fn calc_mismatch_locations<St: State>(&self, state: &St) -> Array2<usize>;

    fn calc_mismatches<St: State>(&self, state: &St) -> NumTiles {
        let mut arr = self.calc_mismatch_locations(state);
        arr.map_inplace(|x| *x = (*x & 0b01) + ((*x & 0b10) / 2));
        arr.sum() as NumTiles
    }

    fn update_points<St: State>(&self, state: &mut St, points: &[PointSafeHere]) {
        let p = points
            .iter()
            .map(|p| (*p, self.event_rate_at_point(state, *p)))
            .collect::<Vec<_>>();

        state.update_multiple(&p);
    }

    fn update_all<St: State>(&self, state: &mut St, needed: &NeededUpdate) {
        let ncols = state.ncols();
        let nrows = state.nrows();

        let all_points = match needed {
            NeededUpdate::None => todo!(),
            NeededUpdate::NonZero => (0..nrows)
                .flat_map(|r| (0..ncols).map(move |c| PointSafeHere((r, c))))
                .filter(|p| state.rate_at_point(*p) > 0.)
                .collect::<Vec<_>>(),
            NeededUpdate::All => (0..nrows)
                .flat_map(|r| (0..ncols).map(move |c| PointSafeHere((r, c))))
                .collect::<Vec<_>>(),
        };

        self.update_points(state, &all_points);
    }

    fn set_param(&mut self, _name: &str, _value: Box<dyn Any>) -> Result<NeededUpdate, GrowError> {
        todo!();
    }

    fn get_param(&self, _name: &str) -> Result<Box<dyn Any>, GrowError> {
        todo!();
    }
}

pub trait SystemWithDimers: System {
    /// Returns information on dimers that the system can form, similarly useful for starting out a state.
    fn calc_dimers(&self) -> Vec<DimerInfo>;
}

pub trait TileBondInfo {
    fn tile_color(&self, tile_number: Tile) -> [u8; 4];
    fn tile_name(&self, tile_number: Tile) -> &str;
    fn bond_name(&self, bond_number: usize) -> &str;

    fn tile_colors(&self) -> &Vec<[u8; 4]>;
    fn tile_names(&self) -> Vec<String>;
    fn bond_names(&self) -> Vec<String>;
}

pub trait SystemInfo {
    fn tile_concs(&self) -> Vec<f64>;
    fn tile_stoics(&self) -> Vec<f64>;
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
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

impl From<&str> for FissionHandling {
    fn from(s: &str) -> Self {
        match s {
            "off" | "no-fission" => FissionHandling::NoFission,
            "just-detach" | "surface" => FissionHandling::JustDetach,
            "on" | "keep-seeded" => FissionHandling::KeepSeeded,
            "keep-largest" => FissionHandling::KeepLargest,
            "keep-weighted" => FissionHandling::KeepWeighted,
            _ => panic!("Unknown fission handling mode: {}", s),
        }
    }
}
