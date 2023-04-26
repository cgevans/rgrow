use super::base::*;
use crate::canvas::{Canvas, CanvasCreate, CanvasSquarable};
use crate::{
    canvas::PointSafe2,
    canvas::PointSafeHere,
    ratestore::{CreateSizedRateStore, QuadTreeSquareArray, RateStore},
    system,
};
use ndarray::prelude::*;

use std::fmt::Debug;

pub trait State: RateStoreP + Canvas + StateStatus + Sync + Send {
    fn panicinfo(&self) -> String;
}

pub trait StateStatus {
    fn ntiles(&self) -> NumTiles;
    fn total_events(&self) -> NumEvents;
    fn add_events(&mut self, n: NumEvents);
    fn reset_events(&mut self);
    fn add_time(&mut self, time: f64);
    fn time(&self) -> f64;
}

pub trait StateCreate: Sized {
    fn create_raw(canvas: Array2<Tile>) -> Result<Self, GrowError>;
    fn empty(shape: (usize, usize)) -> Result<Self, GrowError> {
        Self::create_raw(Array2::zeros(shape))
    }
}

#[derive(Debug, Clone)]
pub struct QuadTreeState<C: CanvasSquarable, T: StateTracker> {
    pub rates: QuadTreeSquareArray<Rate>,
    pub canvas: C,
    ntiles: NumTiles,
    total_events: NumEvents,
    time: f64,
    pub tracker: T,
}

impl<C: CanvasSquarable, T: StateTracker> QuadTreeState<C, T> {
    pub fn recalc_ntiles(&mut self) {
        self.ntiles = self.calc_ntiles();
    }
}

// Storage for event rates,
pub trait RateStoreP {
    fn choose_point(&self) -> (Point, Rate);
    fn update_point(&mut self, point: PointSafeHere, new_rate: Rate);
    fn update_multiple(&mut self, points: &[(PointSafeHere, Rate)]);
    fn rate_at_point(&self, point: PointSafeHere) -> Rate;
    fn total_rate(&self) -> Rate;
}
impl<C: CanvasSquarable + CanvasCreate, T: StateTracker> State for QuadTreeState<C, T> {
    fn panicinfo(&self) -> String {
        format!(
            "{:?} {:?} {}={}",
            self.rates,
            self.canvas.raw_array(),
            self.ntiles(),
            self.calc_ntiles()
        )
    }
}

impl<C: CanvasSquarable, T: StateTracker> RateStoreP for QuadTreeState<C, T> {
    fn choose_point(&self) -> (Point, Rate) {
        self.rates.choose_point()
    }

    fn rate_at_point(&self, point: PointSafeHere) -> Rate {
        self.rates.rate_at_point(point)
    }

    fn update_point(&mut self, point: PointSafeHere, new_rate: Rate) {
        self.rates.update_point(point.0, new_rate)
    }

    #[inline]
    fn update_multiple(&mut self, points: &[(PointSafeHere, Rate)]) {
        self.rates.update_multiple(points);
    }

    fn total_rate(&self) -> Rate {
        self.rates.total_rate()
    }
}

impl<C: CanvasSquarable, T: StateTracker> Canvas for QuadTreeState<C, T> {
    unsafe fn uv_pr(&self, p: Point) -> &Tile {
        self.canvas.uv_pr(p)
    }

    unsafe fn uvm_p(&mut self, p: Point) -> &mut Tile {
        self.canvas.uvm_p(p)
    }

    fn u_move_point_n(&self, p: Point) -> Point {
        self.canvas.u_move_point_n(p)
    }

    fn u_move_point_e(&self, p: Point) -> Point {
        self.canvas.u_move_point_e(p)
    }

    fn u_move_point_s(&self, p: Point) -> Point {
        self.canvas.u_move_point_s(p)
    }

    fn u_move_point_w(&self, p: Point) -> Point {
        self.canvas.u_move_point_w(p)
    }

    fn inbounds(&self, p: Point) -> bool {
        self.canvas.inbounds(p)
    }

    fn calc_ntiles(&self) -> NumTiles {
        self.canvas.calc_ntiles()
    }

    fn calc_ntiles_with_tilearray(&self, should_be_counted: &Array1<bool>) -> NumTiles {
        self.canvas.calc_ntiles_with_tilearray(should_be_counted)
    }

    fn raw_array(&self) -> ArrayView2<Tile> {
        self.canvas.raw_array()
    }

    fn nrows(&self) -> usize {
        self.canvas.nrows()
    }

    fn ncols(&self) -> usize {
        self.canvas.ncols()
    }

    fn set_sa(&mut self, p: &PointSafe2, t: &Tile) {
        let r = unsafe { self.uvm_p(p.0) };

        let old_tile = *r;

        *r = *t;

        if (old_tile == 0) & (*t > 0) {
            self.ntiles += 1
        }
        if (old_tile > 0) & (*t == 0) {
            self.ntiles -= 1
        }
    }

    fn set_sa_countabletilearray(
        &mut self,
        p: &PointSafe2,
        t: &Tile,
        should_be_counted: &Array1<bool>,
    ) {
        let r = unsafe { self.uvm_p(p.0) };

        let old_tile = *r;

        *r = *t;

        if should_be_counted[old_tile as usize] & !should_be_counted[*t as usize] {
            self.ntiles -= 1
        }
        if !should_be_counted[old_tile as usize] & should_be_counted[*t as usize] {
            self.ntiles += 1
        }
    }

    fn draw_size(&self) -> (u32, u32) {
        self.canvas.draw_size()
    }

    fn draw(&self, frame: &mut [u8], colors: &[[u8; 4]]) {
        self.canvas.draw(frame, colors)
    }
}

impl<C, T> StateCreate for QuadTreeState<C, T>
where
    C: CanvasSquarable + CanvasCreate,
    T: StateTracker,
{
    fn create_raw(canvas: Array2<Tile>) -> Result<Self, GrowError> {
        let (ys, xs) = canvas.dim();

        let rates = QuadTreeSquareArray::new_with_size(xs, ys);

        let canvas = C::from_array(canvas)?;
        let tracker = T::default(&canvas);

        Ok(QuadTreeState::<C, T> {
            rates,
            canvas,
            ntiles: 0,
            total_events: 0,
            time: 0.,
            tracker,
        })
    }
}

unsafe impl<C: CanvasSquarable, T: StateTracker> Send for QuadTreeState<C, T> {}

impl<C: CanvasSquarable, T: StateTracker> StateStatus for QuadTreeState<C, T> {
    #[inline(always)]
    fn ntiles(&self) -> NumTiles {
        self.ntiles
    }

    #[inline(always)]
    fn total_events(&self) -> NumEvents {
        self.total_events
    }

    fn add_time(&mut self, time: f64) {
        self.time += time;
    }

    fn time(&self) -> f64 {
        self.time
    }

    fn add_events(&mut self, n: NumEvents) {
        self.total_events += n;
    }

    fn reset_events(&mut self) {
        self.total_events = 0;
    }
}

pub trait DangerousStateClone {
    fn zeroed_copy_from_state_nonzero_rate(&mut self, source: &Self) -> &mut Self;
    fn copy_level_quad(&mut self, source: &Self, level: usize, point: (usize, usize)) -> &mut Self;
}

impl<C: Canvas + CanvasSquarable, T: StateTracker> DangerousStateClone for QuadTreeState<C, T> {
    /// Efficiently, but dangerously, copies a state into zeroed state, when certain conditions are satisfied:
    ///
    /// - The system must be fully unseeded kTAM: specifically, all locations with tiles must have a nonzero rate.
    /// - The assignee state is assumed to have all zero rates, and an all zero canvas.  This is not checked!
    ///
    /// This is fast when the number of tiles << the size of the canvas, eg, when putting in dimers.
    ///
    /// If on debug, conditions should be checked (TODO)
    fn zeroed_copy_from_state_nonzero_rate(&mut self, source: &Self) -> &mut Self {
        let max_level = self.rates.0.len() - 1; // FIXME: should not go into RateStore

        self.copy_level_quad(source, max_level, (0, 0));

        // General housekeeping
        self.ntiles = source.ntiles;
        self.total_events = source.total_events;
        self.tracker = source.tracker.clone();

        self.rates.1 = source.rates.1;

        self
    }

    fn copy_level_quad(&mut self, source: &Self, level: usize, point: (usize, usize)) -> &mut Self {
        // FIXME: should not go into ratestore
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
                    }
                }
            }
        };
        self
    }
}
pub trait StateTracked<T>
where
    T: StateTracker,
{
    fn set_tracker(&mut self, tracker: T) -> &mut Self;
    fn tracker(&mut self) -> &T;
    fn record_event(&mut self, event: &system::Event) -> &mut Self;
}

impl<C, T> StateTracked<T> for QuadTreeState<C, T>
where
    C: Canvas + CanvasSquarable,
    T: StateTracker,
{
    fn set_tracker(&mut self, tracker: T) -> &mut Self {
        self.tracker = tracker;
        self
    }

    fn tracker(&mut self) -> &T {
        &self.tracker
    }

    fn record_event(&mut self, event: &system::Event) -> &mut Self {
        self.tracker.record_single_event(event);
        self
    }
}
pub trait StateTracker: Clone + Debug + Sync + Send {
    fn default(canvas: &dyn Canvas) -> Self;

    fn record_single_event(&mut self, event: &system::Event) -> &mut Self;
}

#[derive(Copy, Clone, Debug)]
pub struct NullStateTracker;

impl StateTracker for NullStateTracker {
    fn default(_state: &dyn Canvas) -> Self {
        Self
    }

    fn record_single_event(&mut self, _event: &system::Event) -> &mut Self {
        self
    }
}

#[derive(Clone, Debug)]
pub struct OrderTracker {
    pub order: u64,
    pub arr: Array2<NumEvents>,
}

impl StateTracker for OrderTracker {
    fn default(canvas: &dyn Canvas) -> Self {
        OrderTracker {
            order: 0,
            arr: Array2::<NumEvents>::zeros((canvas.nrows(), canvas.ncols())),
        }
    }

    fn record_single_event(&mut self, event: &system::Event) -> &mut Self {
        match event {
            system::Event::None => self,
            system::Event::MonomerAttachment(p, _t) => {
                self.arr[p.0] = self.order;
                self.order += 1;
                self
            }
            system::Event::MonomerDetachment(p) => {
                self.arr[p.0] = 0;
                self
            }
            system::Event::MonomerChange(p, _t) => {
                self.arr[p.0] = self.order;
                self.order += 1;
                self
            }
            system::Event::PolymerChange(vec) => {
                for (p, _t) in vec {
                    self.arr[p.0] = self.order;
                }
                self.order += 1;
                self
            }
            system::Event::PolymerAttachment(vec) => {
                for (p, _t) in vec {
                    self.arr[p.0] = self.order;
                }
                self.order += 1;
                self
            }
            system::Event::PolymerDetachment(vec) => {
                for p in vec {
                    self.arr[p.0] = 0;
                }
                self
            }
        }
    }
}
