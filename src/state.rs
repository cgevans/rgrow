use super::base::*;
use super::canvas::*;
use super::system::*;
use ndarray::prelude::*;
use rand::{SeedableRng, random};
use std::{convert::TryInto, fmt::Debug, iter::FromIterator, marker::PhantomData};

type HashSet<T> = fnv::FnvHashSet<T>;

pub trait StateEvolve<C: Canvas, S: System<C>>: StateStatus + StateStep<C, S> {
    fn evolve_until_condition(
        &mut self,
        system: &S,
        condition: &dyn Fn(&Self, NumEvents) -> bool,
    ) -> &mut Self {
        let mut events: NumEvents = 0;

        while !condition(&self, events) {
            self.take_step(system).unwrap();
            events += 1;
        }
        self
    }

    fn evolve_in_size_range_emax_cond(
        &mut self,
        system: &S,
        minsize: NumTiles,
        maxsize: NumTiles,
        maxevents: NumEvents,
    ) -> &mut Self {
        let condition = move |state: &Self, events| -> bool {
            (state.ntiles() <= minsize) | (state.ntiles() >= maxsize) | (events > maxevents)
        };

        self.evolve_until_condition(system, &condition)
    }

    fn evolve_in_size_range_events_max(
        &mut self,
        system: &S,
        minsize: NumTiles,
        maxsize: NumTiles,
        maxevents: NumEvents,
    ) -> &mut Self {
        let mut events: NumEvents = 0;

        while (events < maxevents) & (self.ntiles() < maxsize) & (self.ntiles() > minsize) {
            self.take_step(system).unwrap();
            events += 1;
        }
        self
    }

    fn evolve_steps(
        &mut self,
        system: &S,
        events: NumEvents
    ) -> Result<&mut Self, StateError> {
        for _ in 0..events {
            self.take_step(system)?;
            // FIXME: there is a problem here if "dead" events are the *only* events possible.
        }
        Ok(self)
    }
}


pub trait StateStep<C: Canvas, S: System<C>>: StateUpdateSingle<C, S> + StateStatus {
    fn take_step(&mut self, system: &S) -> Result<&Self, StateError> {
        let (p, acc) = self.choose_event_point()?;
        Ok(self.do_single_event_at_location(system, p, acc))
    }
}


pub trait StateUpdateSingle<C: Canvas, S: System<C>> {
    fn choose_event_point(&self) -> Result<(Point, Rate), StateError>;
    fn do_single_event_at_location(&mut self, system: &S, point: Point, acc: Rate) -> &mut Self;
    fn update_after_single_event(&mut self, system: &S, point: Point) -> &mut Self;
    fn update_entire_state(&mut self, system: &S) -> &mut Self;
    fn set_point(&mut self, sys: &S, p: Point, t: Tile) -> &mut Self;
}

#[derive(Debug, thiserror::Error)]
pub enum StateError {
    #[error("the canvas is empty")]
    EmptyCanvas,
    #[error("the total event rate is zero")]
    ZeroRate,
    #[error("an unknown error occured")]
    Unknown,
}

pub trait StateCreate<C: Canvas, S: System<C>>:
    StateUpdateSingle<C, S> + Sized + StateUpdateSingle<C, S>
{
    /// Given a canvas array as an initial configuration, create a state.
    fn create_raw(canvas: Array2<Tile>) -> Self;

    fn from_canvas(system: &S, canvas: Array2<Tile>) -> Self {
        let mut ret = Self::create_raw(canvas.to_owned());
        ret.insert_seed(system);
        ret.update_entire_state(system);
        ret
    }

    /// Creates an empty state of size `size` with no tiles and all zero rates.
    /// Does *not* insert seed tiles.
    fn empty(size: Point) -> Self {
        Self::create_raw(Array2::<Tile>::zeros(size))
    }

    fn default(size: Point, system: &mut S) -> Self {
        let mut ret = Self::empty(size);
        ret.insert_seed(system);
        ret
    }

    fn create_we_pair(sys: &S, w: Tile, e: Tile, size: usize) -> Self {
        assert!(size > 8);
        let mut ret = Self::empty((size, size));
        let mid = size / 2;
        ret.insert_seed(sys);
        ret.set_point(sys, (mid, mid), w)
            .set_point(sys, (mid, mid + 1), e);
        ret
    }

    fn create_ns_pair(sys: &S, n: Tile, s: Tile, size: usize) -> Self {
        assert!(size > 8);
        let mut ret = Self::empty((size, size));
        let mid = size / 2;
        ret.insert_seed(sys);
        ret.set_point(sys, (mid, mid), n)
            .set_point(sys, (mid + 1, mid), s);
        ret
    }

    fn insert_seed(&mut self, sys: &S) -> &mut Self {
        for (p, t) in sys.seed_locs() {
            // FIXME: for large seeds,
            // this could be faster by doing raw writes, then update_entire_state
            // but we would need to distinguish sizing.
            // Or maybe there is fancier way with a set?
            self.set_point(sys, p, t);
        }
        self
    }
}
pub trait StateStatus {
    fn ntiles(&self) -> NumTiles;
    fn total_events(&self) -> NumEvents;
    fn total_rate(&self) -> Rate;

    //fn time(&self) -> Time;

    //fn last_step_time(&self) -> Time;
}

use crate::ratestore::{RateStore, QuadTreeArray, CreateSizedRateStore};

#[derive(Debug)]
pub struct QuadTreeState<C: CanvasSquarable, S: System<C>, T: StateTracker> {
    pub rates: QuadTreeArray<Rate>,
    pub canvas: C,
    phantomsys: PhantomData<*const S>,
    ntiles: NumTiles,
    total_events: NumEvents,
    pub tracker: T,
}

unsafe impl<C: CanvasSquarable, S: System<C>, T: StateTracker> Send for QuadTreeState<C, S, T> {}

impl<C: CanvasSquarable + Clone, S: System<C>, T: StateTracker + Clone> Clone
    for QuadTreeState<C, S, T>
{
    fn clone(&self) -> Self {
        Self {
            rates: self.rates.clone(),
            canvas: self.canvas.clone(),
            phantomsys: self.phantomsys,
            ntiles: self.ntiles,
            total_events: self.total_events,
            tracker: self.tracker.clone(),
        }
    }
}

impl<C, S, T> StateEvolve<C, S> for QuadTreeState<C, S, T>
where
    C: CanvasSquarable,
    S: System<C>,
    T: StateTracker,
{
}

impl<C, S, T> StateStep<C, S> for QuadTreeState<C, S, T>
where
    C: CanvasSquarable,
    S: System<C>,
    T: StateTracker,
{
}

impl<C, S, T> StateCreate<C, S> for QuadTreeState<C, S, T>
where
    C: CanvasSquarable + CanvasCreate,
    S: System<C>,
    T: StateTracker,
{
    fn create_raw(canvas: Array2<Tile>) -> Self {
        assert!(canvas.nrows().is_power_of_two());
        assert!(canvas.ncols() == canvas.nrows());

        let rates = QuadTreeArray::new_with_size(canvas.nrows());
        let canvas = C::from_array(canvas).unwrap();
        let tracker = T::default(&canvas);

        QuadTreeState::<C, S, T> {
            rates,
            canvas,
            phantomsys: PhantomData,
            ntiles: 0,
            total_events: 0,
            tracker,
        }
    }
}

impl<C: CanvasSquarable, S, T> StateStatus for QuadTreeState<C, S, T>
where
    S: System<C>,
    T: StateTracker,
{
    #[inline(always)]
    fn ntiles(&self) -> NumTiles {
        self.ntiles
    }

    #[inline(always)]
    fn total_events(&self) -> NumEvents {
        self.total_events
    }

    #[inline(always)]
    fn total_rate(&self) -> Rate {
        self.rates.total_rate()
    }
}

impl<C: CanvasSquarable, S, T> StateUpdateSingle<C, S> for QuadTreeState<C, S, T>
where
    S: System<C>,
    T: StateTracker,
{
    fn choose_event_point(&self) -> Result<(Point, Rate), StateError> {
        let mut rng = rand::rngs::SmallRng::from_entropy();

        let ((y, x), threshold) = self.rates.choose_point(&mut rng);

        if (y, x) == (0, 0) {
            // This happens when we have no tiles, or a zero rate.
            if self.ntiles() == 0 {
                Err(StateError::EmptyCanvas)
            } else if self.total_rate() == 0. {
                Err(StateError::ZeroRate)
            } else if self.canvas.inbounds((0,0)) {
                Ok(((y, x), threshold))
            }
            else {
                Err(StateError::Unknown)
            }
        } else {
            Ok(((y, x), threshold))
        }
    }

    #[inline(always)]
    fn update_after_single_event(&mut self, system: &S, point: Point) -> &mut Self {
        match system.updates_around_point() {
            Updates::Plus => self.update_rates_ps(system, point),

            Updates::DimerChunk => {
                self.update_rates_ps(system, point);
                let pww = unsafe {
                    self.canvas
                        .u_move_point_w(self.canvas.u_move_point_w(point))
                };
                if self.canvas.inbounds(pww) {
                    self.update_rates_single(system, pww);
                }
                let pnn = unsafe {
                    self.canvas
                        .u_move_point_n(self.canvas.u_move_point_n(point))
                };
                if self.canvas.inbounds(pnn) {
                    self.update_rates_single(system, pnn);
                }
                let pnw = self.canvas.u_move_point_nw(point);
                let pse = self.canvas.u_move_point_sw(point);
                let pne = self.canvas.u_move_point_ne(point);
                self.update_rates_single(system, pnw)
                    .update_rates_single(system, pse)
                    .update_rates_single(system, pne)
            }
        }
    }

    #[inline(always)]
    fn do_single_event_at_location(&mut self, system: &S, p: Point, acc: Rate) -> &mut Self {
        let event = system.choose_event_at_point(&self.canvas, p, acc);

        match event {
            Event::None => {
                println!("dead event")
                // The event was probably cancelled: do nothing.
            }
            Event::SingleTileAttach(new_tile) => {
                self.ntiles += 1;
                self.total_events += 1;

                // Repeatedly checked!
                let loc = unsafe { self.canvas.uvm_p(p) };

                let old_tile: Tile = *loc;
                *loc = new_tile;

                self.update_after_single_event(system, p)
                    .record_single_event(p, old_tile, new_tile);
            }
            Event::SingleTileDetach => {
                self.ntiles -= 1;
                self.total_events += 1;

                // Repeatedly checked!
                let loc = unsafe { self.canvas.uvm_p(p) };

                let old_tile: Tile = *loc;
                *loc = 0;

                self.update_after_single_event(system, p)
                    .record_single_event(p, old_tile, 0);
            }
            Event::SingleTileChange(_) => todo!(),
            Event::MultiTileDetach(pointvec) => {
                self.total_events += 1;

                for point in pointvec {
                    let loc = unsafe { self.canvas.uvm_p(point) };
                    let old_tile: Tile = *loc;

                    self.ntiles -= 1;

                    *loc = 0;

                    self.update_after_single_event(system, point)
                        .record_single_event(point, old_tile, 0);
                }
            }
            Event::MultiTileAttach(pointvec) => {
                self.total_events += 1;

                for (point, tile) in pointvec {
                    let loc = unsafe { self.canvas.uvm_p(point) };
                    let old_tile: Tile = *loc;

                    self.ntiles += 1;

                    *loc = tile;

                    self.update_after_single_event(system, point)
                        .record_single_event(point, old_tile, tile);
                }
            }
        }

        self
    }

    fn update_entire_state(&mut self, system: &S) -> &mut Self {
        let size = self.canvas.square_size();
        for y in 0..size {
            for x in 0..size {
                if self.canvas.inbounds((y,x)) {
                    self.update_rates_single(system, (y, x));
                }
            }
        }
        self.ntiles = self.canvas.calc_ntiles();
        self
    }

    fn set_point(&mut self, sys: &S, p: Point, t: Tile) -> &mut Self {
        assert!(self.canvas.inbounds(p));
        let ot = unsafe { self.canvas.uv_p(p) };
        unsafe {
            *self.canvas.uvm_p(p) = t;
        }
        self.record_single_event(p, ot, t);
        self.update_after_single_event(sys, p);
        if (t == 0) & (ot != 0) {
            self.ntiles -= 1
        } else if (t != 0) & (ot == 0) {
            self.ntiles += 1
        };
        self
    }
}

pub trait StateTracked<C, S, T>
where
    C: Canvas,
    S: System<C>,
    T: StateTracker,
{
    fn set_tracker(&mut self, tracker: T) -> &mut Self;
    fn tracker(&mut self) -> &T;
    fn record_single_event(&mut self, _p: Point, old_tile: Tile, new_tile: Tile) -> &mut Self;
}

impl<C, S, T> StateTracked<C, S, T> for QuadTreeState<C, S, T>
where
    C: CanvasSquarable,
    S: System<C>,
    T: StateTracker,
{
    fn set_tracker(&mut self, tracker: T) -> &mut Self {
        self.tracker = tracker;
        self
    }

    fn tracker(&mut self) -> &T {
        &self.tracker
    }

    fn record_single_event(&mut self, p: Point, old_tile: Tile, new_tile: Tile) -> &mut Self {
        self.tracker.record_single_event(p, old_tile, new_tile);
        self
    }
}

impl<C: CanvasSquarable + CanvasCreate, S, T> QuadTreeState<C, S, T>
where
    S: System<C>,
    T: StateTracker,
{
    pub fn create_we_pair_with_tracker(
        sys: &S,
        w: Tile,
        e: Tile,
        size: CanvasLength,
        tracker: T,
    ) -> Self {
        assert!(size > 8);
        let mut ret = Self::empty((size, size));
        ret.tracker = tracker;
        let mid = size / 2;
        ret.set_point(sys, (mid, mid), w)
            .set_point(sys, (mid, mid + 1), e);
        ret
    }

    pub fn create_ns_pair_with_tracker(
        sys: &S,
        n: Tile,
        s: Tile,
        size: CanvasLength,
        tracker: T,
    ) -> Self {
        assert!(size > 8);
        let mut ret = Self::empty((size, size));
        ret.tracker = tracker;
        let mid = size / 2;
        ret.set_point(sys, (mid, mid), n)
            .set_point(sys, (mid + 1, mid), s);
        ret
    }
}

impl<C: CanvasSquarable, S, T> QuadTreeState<C, S, T>
where
    S: System<C>,
    T: StateTracker + Clone,
{
    fn update_rates_ps(&mut self, system: &S, p: Point) -> &mut Self {

        let points =  &[
            self.canvas.u_move_point_w(p),
            p,
            self.canvas.u_move_point_e(p),
            self.canvas.u_move_point_n(p),
            self.canvas.u_move_point_s(p),
        ];
        
        let rates = points.into_iter().map(|x| system.event_rate_at_point(&self.canvas, *x)).collect::<Vec<_>>();

        self.rates.update_multiple(points, &rates);
        
        self
    }

    #[allow(dead_code)]
    fn update_rates_single(&mut self, system: &S, mut p: Point) -> &mut Self {
        self.rates.update_point(p, system.event_rate_at_point(&self.canvas, p));
        self
    }

    /// Efficiently, but dangerously, copies a state into zeroed state, when certain conditions are satisfied:
    ///
    /// - The system must be fully unseeded kTAM: specifically, all locations with tiles must have a nonzero rate.
    /// - The assignee state is assumed to have all zero rates, and an all zero canvas.  This is not checked!
    ///
    /// This is fast when the number of tiles << the size of the canvas, eg, when putting in dimers.
    ///
    /// If on debug, conditions should be checked (TODO)
    pub fn zeroed_copy_from_state_nonzero_rate(&mut self, source: &Self) -> &mut Self {
        let max_level = self.rates.0.len()-1; // FIXME: should not go into RateStore

        self.copy_level_quad(source, max_level, (0, 0));

        
        // General housekeeping
        self.ntiles = source.ntiles;
        self.total_events = source.total_events;
        self.tracker = source.tracker.clone();

        if self.canvas.calc_ntiles() != self.ntiles {
            panic!("sink {:?} / source {:?}", self, source);
        }

        self
    }

    #[inline(never)]
    fn copy_level_quad(&mut self, source: &Self, level: usize, point: (usize, usize)) -> &mut Self { // FIXME: should not go into ratestore
        let (y, x) = point;

        if level > 0 {
            for (yy, xx) in &[(y, x), (y, x + 1), (y + 1, x), (y + 1, x + 1)] {
                let z = source.rates.0[level][(*yy, *xx)];
                if z > 0. {
                    self.rates.0[level][(*yy, *xx)] = z;
                    self.copy_level_quad(source, level - 1, (*yy * 2, *xx * 2));
                }
            }
        } else {
            for (yy, xx) in &[(y, x), (y, x + 1), (y + 1, x), (y + 1, x + 1)] {
                let z = source.rates.0[level][(*yy, *xx)];
                let t = unsafe { source.canvas.uv_p((*yy, *xx)) };
                if z > 0. {
                    self.rates.0[level][(*yy, *xx)] = z;
                    if t > 0 {
                        // Tile must have nonzero rate, so we only check if the rate is nonzero.
                        let v = unsafe { self.canvas.uvm_p((*yy, *xx)) };
                        *v = t;
                        drop(v);
                    }
                }
            }
        };
        self
    }
}


pub trait StateTracker: Clone + Debug {
    fn default(canvas: &dyn Canvas) -> Self;

    fn record_single_event(&mut self, p: Point, old_tile: Tile, new_tile: Tile) -> &mut Self;
}

#[derive(Copy, Clone, Debug)]
pub struct NullStateTracker();

impl StateTracker for NullStateTracker {
    fn default(_state: &dyn Canvas) -> Self {
        Self()
    }

    fn record_single_event(&mut self, _p: Point, _old_tile: Tile, _new_tile: Tile) -> &mut Self {
        self
    }
}

#[derive(Clone, Debug)]
pub struct TileSubsetTracker {
    pub num_in_subset: NumTiles,
    set: HashSet<Tile>,
}

impl StateTracker for TileSubsetTracker {
    fn default(_canvas: &dyn Canvas) -> Self {
        // Default is to track nothing.
        Self {
            num_in_subset: 0,
            set: HashSet::<Tile>::default(),
        }
    }

    fn record_single_event(&mut self, _p: Point, old_tile: Tile, new_tile: Tile) -> &mut Self {
        if (old_tile == 0) & self.set.contains(&new_tile) {
            self.num_in_subset += 1
        } else if (new_tile == 0) & self.set.contains(&old_tile) {
            self.num_in_subset -= 1
        } else {
            // FIXME: ignores tile swaps
        };
        self
    }
}

impl TileSubsetTracker {
    pub fn new(tiles: Vec<Tile>) -> Self {
        Self {
            num_in_subset: 0,
            set: HashSet::<Tile>::from_iter(tiles),
        }
    }
}

// #[derive(Clone, Debug)]
// pub struct OrderTracker {
//     pub orders: Box<dyn Canvas>,
//     cur_order: usize,
// }

// impl OrderTracker {
//     pub fn new(canvas: &dyn Canvas) -> Self {
//         OrderTracker {
//             orders: canvas.clone(),
//             cur_order: 1,
//         }
//     }
// }

// impl StateTracker for OrderTracker {
//     fn default(canvas: &dyn Canvas) -> Self {
//         OrderTracker {
//             orders: canvas.clone(),
//             cur_order: 1,
//         }
//     }

//     fn record_single_event(&mut self, p: Point, _old_tile: Tile, new_tile: Tile) -> &mut Self {
//         if new_tile == 0 {
//             unsafe {
//                 *self.orders.uget_mut(p) = 0;
//             }
//         } else {
//             unsafe {
//                 *self.orders.uget_mut(p) = self.cur_order;
//                 self.cur_order += 1;
//             }
//         };
//         self
//     }
// }
