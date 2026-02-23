use ndarray::prelude::*;
use num_traits::Zero;
use rand::rng;
use rand::Rng;

use std::any::Any;
use std::fmt::Debug;

use rayon::prelude::*;

use crate::base::{GrowError, NumTiles, Point, Tile};
use crate::canvas::{PointSafe2, PointSafeHere};
use crate::state::{State, StateWithCreate};
use crate::units::{PerSecond, Rate, Second};

use super::dispatch::TileBondInfo;
use super::gui::evolve_in_window_impl;
use super::types::*;

use crate::base::RgrowError;

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
    /// If `replace` is true, any existing tile at the target site is removed first.
    /// If `replace` is false, returns `GrowError::TilePlacementBlocked` if the site is occupied.
    ///
    /// This updates tile counts and rates but does not increment the event counter
    /// or record events in the state tracker. Callers that need event tracking
    /// should call `record_event` separately.
    ///
    /// Returns energy change caused by placement, or NaN if energy is not calculated.
    fn place_tile<St: State>(
        &self,
        state: &mut St,
        point: PointSafe2,
        tile: Tile,
        replace: bool,
    ) -> Result<f64, GrowError> {
        let existing = state.tile_at_point(point);
        if existing != 0 {
            if !replace {
                return Err(GrowError::TilePlacementBlocked {
                    row: point.0 .0,
                    col: point.0 .1,
                    tile,
                    existing_tile: existing,
                });
            }
            let ev = Event::MonomerDetachment(point);
            self.perform_event(state, &ev);
            self.update_after_event(state, &ev);
        }
        let ev = Event::MonomerAttachment(point, tile);
        let energy_change = self.perform_event(state, &ev);
        self.update_after_event(state, &ev);
        Ok(energy_change)
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
        bounds: EvolveBounds,
        initial_timescale: Option<f64>,
        initial_max_events_per_sec: Option<u64>,
    ) -> Result<EvolveOutcome, RgrowError> {
        evolve_in_window_impl(
            self,
            state,
            block,
            start_paused,
            bounds,
            initial_timescale,
            initial_max_events_per_sec,
        )
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
