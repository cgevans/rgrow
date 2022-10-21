use ndarray::prelude::*;
use rand::{prelude::SmallRng, Rng};
use serde::{Deserialize, Serialize};

use crate::{
    base::GrowError, base::NumEvents, base::NumTiles, canvas::PointSafeHere, state::State,
    state::StateCreate,
};

use super::base::{Point, Rate, Tile};
use crate::canvas::PointSafe2;

use std::fmt::Debug;
use std::time::Duration;

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

#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub enum ChunkSize {
    #[serde(alias = "single")]
    Single,
    #[serde(alias = "dimer")]
    Dimer,
}

pub trait SystemWithStateCreate<S: State + StateCreate>: System<S> {
    fn new_state(&self, shape: (usize, usize)) -> Result<S, GrowError> {
        let mut new_state = S::empty(shape)?;
        self.insert_seed(&mut new_state);
        Ok(new_state)
    }

    fn create_we_pair(&mut self, w: Tile, e: Tile, size: usize) -> Result<S, GrowError> {
        assert!(size > 8);
        let mut ret = S::empty((size, size))?;
        let mid = size / 2;
        self.insert_seed(&mut ret);
        self.set_point(&mut ret, (mid, mid), w);
        self.set_point(&mut ret, (mid, mid + 1), e);
        Ok(ret)
    }

    fn create_ns_pair(&mut self, n: Tile, s: Tile, size: usize) -> Result<S, GrowError> {
        assert!(size > 8);
        let mut ret = S::empty((size, size))?;
        let mid = size / 2;
        self.insert_seed(&mut ret);
        self.set_point(&mut ret, (mid, mid), n);
        self.set_point(&mut ret, (mid + 1, mid), s);
        Ok(ret)
    }
}

impl<Sy: System<S>, S: State + StateCreate> SystemWithStateCreate<S> for Sy {}

pub trait System<S: State>: Debug {
    fn state_step(
        &self,
        mut state: &mut S,
        mut rng: &mut SmallRng,
        max_time_step: f64,
    ) -> StepOutcome {
        let time_step = -f64::ln(rng.gen()) / state.total_rate();
        if time_step > max_time_step {
            state.add_time(max_time_step);
            return StepOutcome::NoEventIn(max_time_step);
        }
        let (point, remainder) = state.choose_point(&mut rng); // todo: resultify
        let event = self.choose_event_at_point(&mut state, PointSafe2(point), remainder); // FIXME
        if let Event::None = event {
            return StepOutcome::DeadEventAt(time_step);
        }

        self.perform_event(&mut state, &event);
        self.update_after_event(&mut state, &event);
        state.add_time(time_step);
        StepOutcome::HadEventAt(time_step)
    }

    fn evolve(
        &self,
        state: &mut S,
        rng: &mut SmallRng,
        for_events: Option<NumEvents>,
        for_time: Option<f64>,
        min_size: Option<NumTiles>,
        max_size: Option<NumTiles>,
        for_wall_time: Option<Duration>,
    ) {
        let mut events = 0;
        let mut rtime = match for_time {
            Some(t) => t,
            None => f64::INFINITY,
        };

        // If we have a for_wall_time, get an instant to compare to
        let start_time = match for_wall_time {
            Some(_) => Some(std::time::Instant::now()),
            None => None,
        };

        while (for_events.is_none() || events < for_events.unwrap())
            && (min_size.is_none() || state.ntiles() > min_size.unwrap())
            && (max_size.is_none() || state.ntiles() < max_size.unwrap())
            && (match for_wall_time {
                Some(t) => start_time.unwrap().elapsed() < t,
                None => true,
            })
        {
            let out = self.state_step(state, rng, rtime);
            match out {
                StepOutcome::HadEventAt(t) => {
                    events += 1;
                    rtime -= t;
                }
                StepOutcome::NoEventIn(t) => {
                    rtime -= t;
                    break;
                }
                StepOutcome::DeadEventAt(t) => {
                    rtime -= t;
                }
                StepOutcome::ZeroRate => {
                    println!("Zero rate");
                    break;
                }
            }
        }
    }

    fn evolve_in_size_range_events_max(
        &mut self,
        state: &mut S,
        minsize: NumTiles,
        maxsize: NumTiles,
        maxevents: NumEvents,
        rng: &mut SmallRng,
    ) {
        let mut events: NumEvents = 0;

        while (events < maxevents) & (state.ntiles() < maxsize) & (state.ntiles() > minsize) {
            match self.state_step(state, rng, 1e100) {
                StepOutcome::HadEventAt(_) => {
                    events += 1;
                }
                StepOutcome::NoEventIn(_) => {
                    println!("Timeout {:?}", state);
                }
                StepOutcome::DeadEventAt(_) => {
                    println!("Dead");
                }
                StepOutcome::ZeroRate => {
                    panic!()
                }
            }
        }
    }

    fn set_point(&self, state: &mut S, point: Point, tile: Tile) {
        assert!(state.inbounds(point));

        let point = PointSafe2(point);

        state.set_sa(&point, &tile);

        let event = Event::MonomerAttachment(point, tile);

        self.update_after_event(state, &event);
    }

    fn insert_seed(&self, state: &mut S) {
        for (p, t) in self.seed_locs() {
            self.set_point(state, p.0, t);
        }
    }

    fn perform_event(&self, state: &mut S, event: &Event) {
        //state.record_event(&event);
        match event {
            Event::None => panic!("Being asked to perform null event."),
            Event::MonomerAttachment(point, tile) | Event::MonomerChange(point, tile) => {
                state.set_sa(point, tile);
            }
            Event::MonomerDetachment(point) => {
                state.set_sa(point, &0usize);
            }
            Event::PolymerAttachment(changelist) | Event::PolymerChange(changelist) => {
                for (point, tile) in changelist {
                    state.set_sa(point, tile);
                }
            }
            Event::PolymerDetachment(changelist) => {
                for point in changelist {
                    state.set_sa(point, &0usize);
                }
            }
        }
    }

    fn update_after_event(&self, state: &mut S, event: &Event);

    /// Returns the total event rate at a given point.  These should correspond with the events chosen by `choose_event_at_point`.
    fn event_rate_at_point(&self, state: &S, p: PointSafeHere) -> Rate;

    /// Given a point, and an accumulated random rate choice `acc` (which should be less than the total rate at the point),
    /// return the event that should take place.
    fn choose_event_at_point(&self, state: &S, p: PointSafe2, acc: Rate) -> Event;

    /// Returns a vector of (point, tile number) tuples for the seed tiles, useful for populating an initial state.
    fn seed_locs(&self) -> Vec<(PointSafe2, Tile)>;

    /// Returns information on dimers that the system can form, similarly useful for starting out a state.
    fn calc_dimers(&self) -> Vec<DimerInfo>;

    fn calc_mismatch_locations(&self, state: &S) -> Array2<usize>;

    fn calc_mismatches(&self, state: &S) -> NumTiles {
        let arr = self.calc_mismatch_locations(state);
        arr.sum() as u32 / 2
    }

    fn update_points(&self, state: &mut S, points: &[PointSafeHere]) {
        let rates = points
            .iter()
            .map(|p| self.event_rate_at_point(state, *p))
            .collect::<Vec<_>>();

        state.update_multiple(&points, &rates);
    }
}

pub trait TileBondInfo {
    fn tile_color(&self, tile_number: Tile) -> [u8; 4];
    fn tile_name(&self, tile_number: Tile) -> &str;
    fn bond_name(&self, bond_number: usize) -> &str;

    fn tile_colors(&self) -> Vec<[u8; 4]>;
    fn tile_names(&self) -> Vec<String>;
    fn bond_names(&self) -> Vec<String>;
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
