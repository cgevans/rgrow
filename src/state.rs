use super::base::*;
use super::canvas::*;
use super::system::*;
use ndarray::prelude::*;
use rand::random;
use std::{convert::TryInto, iter::FromIterator, marker::PhantomData};
use thiserror::Error;

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
}

pub trait StateUpdateSingle<C: Canvas, S: System<C>> {
    fn choose_event_point(&self) -> Result<(Point, Rate), StateError>;
    fn do_single_event_at_location(&mut self, system: &S, point: Point, acc: Rate) -> &mut Self;
    fn update_after_single_event(&mut self, system: &S, point: Point) -> &mut Self;
    fn update_entire_state(&mut self, system: &S) -> &mut Self;
    fn set_point(&mut self, sys: &S, p: Point, t: Tile) -> &mut Self;
}

#[derive(Debug, Error)]
pub enum StateError {
    #[error("the canvas is empty")]
    EmptyCanvas,
    #[error("the total event rate is zero")]
    ZeroRate,
    #[error("an unknown error occured")]
    Unknown,
}

pub trait StateStep<C: Canvas, S: System<C>>: StateUpdateSingle<C, S> + StateStatus {
    fn take_step(&mut self, system: &S) -> Result<&Self, StateError> {
        let (p, acc) = self.choose_event_point()?;
        Ok(self.do_single_event_at_location(system, p, acc))
    }
}

pub trait StateCreate<C: Canvas, S: System<C>>:
    StateUpdateSingle<C, S> + Sized + StateUpdateSingle<C, S>
{
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

pub trait StateTracker: Clone {
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

#[derive(Debug)]
pub struct QuadTreeState<C: CanvasSquarable, S: System<C>, T: StateTracker> {
    pub rates: Vec<Array2<Rate>>,
    pub canvas: C,
    phantomsys: PhantomData<*const S>,
    ntiles: NumTiles,
    total_rate: Rate,
    total_events: NumEvents,
    pub tracker: T,
}

impl<C: CanvasSquarable + Clone, S: System<C>, T: StateTracker + Clone> Clone for QuadTreeState<C, S, T> {
    fn clone(&self) -> Self {
        Self {
            rates: self.rates.clone(),
            canvas: self.canvas.clone(),
            phantomsys: self.phantomsys,
            ntiles: self.ntiles,
            total_rate: self.total_rate,
            total_events: self.total_events,
            tracker: self.tracker.clone()
        }
    }
}

impl<C,S, T> StateEvolve<C, S> for QuadTreeState<C, S, T>
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

        let p: u32 = (1 + canvas.nrows().trailing_zeros()).try_into().unwrap();

        let mut rates = Vec::<Array2<Rate>>::new();

        for i in (1..p).rev() {
            rates.push(Array2::<Rate>::zeros((2usize.pow(i), 2usize.pow(i))))
        }

        let canvas = C::from_array(canvas);
        let tracker = T::default(&canvas);

        QuadTreeState::<C, S, T> {
            rates,
            canvas,
            phantomsys: PhantomData,
            ntiles: 0,
            total_rate: 0.,
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
        self.total_rate
    }
}

impl<C: CanvasSquarable, S, T> StateUpdateSingle<C, S> for QuadTreeState<C, S, T>
where
    S: System<C>,
    T: StateTracker,
{
    fn choose_event_point(&self) -> Result<(Point, Rate), StateError> {
        let mut threshold = self.total_rate * random::<Rate>();

        let mut x: usize = 0;
        let mut y: usize = 0;

        for r in self.rates.iter().rev() {
            y *= 2;
            x *= 2;
            let mut v = unsafe { *r.uget((y, x)) };
            if threshold - v <= 0. {
                continue;
            } else {
                threshold -= v;
                x += 1;
                v = unsafe { *r.uget((y, x)) };
            }
            if threshold - v <= 0. {
                continue;
            } else {
                threshold -= v;
                x -= 1;
                y += 1;
                v = unsafe { *r.uget((y, x)) };
            }
            if threshold - v <= 0. {
                continue;
            } else {
                threshold -= v;
                x += 1;
                v = unsafe { *r.uget((y, x)) };
            }
            if threshold - v <= 0. {
                continue;
            } else {
                panic!("Failure in quadtree position finding: remaining threshold {:?}, ratetree array {:?}.", threshold, r);
            }
        }

        if (y, x) == (0, 0) {
            // This happens when we have no tiles, or a zero rate.
            if self.ntiles() == 0 {
                Err(StateError::EmptyCanvas)
            } else if self.total_rate() == 0. {
                Err(StateError::ZeroRate)
            } else {
                Err(StateError::Unknown)
            }
        } else {Ok(((y, x), threshold))}
    }

    #[inline(always)]
    fn update_after_single_event(&mut self, system: &S, point: Point) -> &mut Self {
        match system.updates_around_point() {
            Updates::Plus => {             self.update_rates_ps(system, point)
            }
            Updates::DimerChunk => { 
                self.update_rates_ps(system, point);
                let pww = unsafe { self.canvas.u_move_point_w(self.canvas.u_move_point_w(point)) };
                if self.canvas.inbounds(pww) {self.update_rates_single(system, pww);}
                let pnn = unsafe { self.canvas.u_move_point_n(self.canvas.u_move_point_n(point)) };
                if self.canvas.inbounds(pnn) {self.update_rates_single(system, pnn);}
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
            Event::SingleTileChange(_) => { todo!() }
            Event::MultiTileDetach(pointvec) => {
                self.total_events += 1;

                for point in pointvec {
                    let loc = unsafe {self.canvas.uvm_p(point)};
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
                    let loc = unsafe {self.canvas.uvm_p(point)};
                    let old_tile: Tile = *loc;

                    self.ntiles += 1;

                    *loc = tile;

                    self.update_after_single_event(system, point).record_single_event(point, old_tile, tile);
                }
            }
        }

        self
    }

    fn update_entire_state(&mut self, system: &S) -> &mut Self {
        let size = self.canvas.size();
        for y in 1..size - 1 {
            for x in 1..size - 1 {
                self.update_rates_single_noprop(system, (y, x));
            }
        }
        self.ntiles = self.canvas.calc_ntiles();
        self.rebuild_ratetree()
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
        let mut rtiter = self.rates.iter_mut();

        // The base level
        let mut rt = rtiter.next().unwrap();
        let mut np: (usize, usize) = p.clone();

        for ps in &[
            self.canvas.u_move_point_w(p),
            p,
            self.canvas.u_move_point_e(p),
            self.canvas.u_move_point_n(p),
            self.canvas.u_move_point_s(p),
        ] {
            rt[*ps] = system.event_rate_at_point(&self.canvas, *ps);
        }

        let mut div: usize = 2;

        for rn in rtiter {
            np = (np.0 / 2, np.1 / 2);

            qt_update_level(rn, rt, np);

            // If on boundary of , update to N; if on
            if p.0 % div == 0 {
                qt_update_level(rn, rt, (np.0 - 1, np.1))
            } else if (p.0 + 1) % div == 0 {
                qt_update_level(rn, rt, (np.0 + 1, np.1))
            };

            if p.1 % div == 0 {
                qt_update_level(rn, rt, (np.0, np.1 - 1))
            } else if (p.1 + 1) % div == 0 {
                qt_update_level(rn, rt, (np.0, np.1 + 1))
            };

            div *= 2;
            rt = rn;
        }

        self.total_rate = rt.sum();

        return self;
    }

    #[allow(dead_code)]
    fn update_rates_single(&mut self, system: &S, p: Point) -> &mut Self {
        let mut rtiter = self.rates.iter_mut();
        let mut rt = rtiter.next().unwrap();
        let mut np: (usize, usize) = p.clone();

        rt[p] = system.event_rate_at_point(&self.canvas, p);

        for rn in rtiter {
            np = (np.0 / 2, np.1 / 2);
            qt_update_level(rn, rt, np);
            rt = rn;
        }

        self.total_rate = rt.sum();

        return self;
    }

    fn update_rates_single_noprop(&mut self, system: &S, p: Point) -> &mut Self {
        let rt = self.rates.iter_mut().next().unwrap();

        rt[p] = system.event_rate_at_point(&self.canvas, p);

        return self;
    }

    fn rebuild_ratetree(&mut self) -> &mut Self {
        let mut rtiter = self.rates.iter_mut();
        let mut rt = rtiter.next().unwrap();

        for rn in rtiter {
            for (p, v) in rn.indexed_iter_mut() {
                qt_update_level_val(v, rt, p)
            }

            rt = rn;
        }

        self.total_rate = rt.sum();

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
        self.copy_level_quad(source, 0, (0,0))
    }


    fn copy_level_quad(&mut self, source: &Self, level: usize, point: (usize, usize)) -> &mut Self {
        let (y, x) = point;

        if level < self.rates.len()-1 {
            for (yy, xx) in &[(y,x), (y,x+1), (y+1, x), (y, x+1)] {
                let z = source.rates[level][(*yy, *xx)];
                if z > 0. {
                    self.rates[level][(*yy, *xx)] = z;
                    self.copy_level_quad(source, level+1, (*yy * 2, *xx *2));
                }
            }
        } else {
            for (yy, xx) in &[(y,x), (y,x+1), (y+1, x), (y, x+1)] {
                let z = source.rates[level][(*yy, *xx)];
                let t = unsafe { source.canvas.uv_p((*yy, *xx)) };
                if z > 0. {
                    self.rates[level][(*yy, *xx)] = z;
                if t > 0 { // Tile must have nonzero rate, so we only check if the rate is nonzero.
                    let v = unsafe { self.canvas.uvm_p((*yy, *xx)) };
                    *v = t;
                    drop(v);
                }
                }
            }
        };

        // General housekeeping
        self.ntiles = source.ntiles;
        self.total_rate = source.total_rate;
        self.total_events = source.total_events;
        self.tracker = source.tracker.clone();

        self
    }

}

#[inline(always)]
fn qt_update_level(rn: &mut Array2<Rate>, rt: &Array2<Rate>, np: Point) {
    qt_update_level_val(unsafe { rn.uget_mut(np) }, rt, np);
}

#[inline(always)]
fn qt_update_level_val(rn: &mut f64, rt: &Array2<Rate>, np: Point) {
    let ip = (np.0 * 2, np.1 * 2);

    unsafe {
        *rn = *rt.uget(ip)
            + *rt.uget((ip.0, ip.1 + 1))
            + *rt.uget((ip.0 + 1, ip.1))
            + *rt.uget((ip.0 + 1, ip.1 + 1));
    }
}
