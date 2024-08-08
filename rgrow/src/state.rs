use super::base::*;
use crate::canvas::{Canvas, CanvasCreate, CanvasPeriodic, CanvasSquare, CanvasTube};
use crate::tileset::{CanvasType, TrackingType};
use crate::{
    canvas::PointSafe2,
    canvas::PointSafeHere,
    ratestore::{CreateSizedRateStore, QuadTreeSquareArray, RateStore},
    system,
};
use ndarray::prelude::*;

use enum_dispatch::enum_dispatch;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

#[enum_dispatch]
pub trait State: RateStore + Canvas + StateStatus + Sync + Send + TrackerData + TileCounts {
    fn panicinfo(&self) -> String;
}

#[enum_dispatch]
pub trait ClonableState: State {
    fn clone_as_stateenum(&self) -> StateEnum {
        panic!("Not implemented")
    }
}

impl ClonableState for QuadTreeState<CanvasSquare, OrderTracker> {
    fn clone_as_stateenum(&self) -> StateEnum {
        StateEnum::SquareOrderTracking(self.clone())
    }
}

impl ClonableState for QuadTreeState<CanvasSquare, NullStateTracker> {
    fn clone_as_stateenum(&self) -> StateEnum {
        StateEnum::SquareCanvasNullTracker(self.clone())
    }
}

impl ClonableState for QuadTreeState<CanvasPeriodic, OrderTracker> {
    fn clone_as_stateenum(&self) -> StateEnum {
        StateEnum::PeriodicOrderTracking(self.clone())
    }
}

impl ClonableState for QuadTreeState<CanvasPeriodic, NullStateTracker> {
    fn clone_as_stateenum(&self) -> StateEnum {
        StateEnum::PeriodicCanvasNoTracker(self.clone())
    }
}

impl ClonableState for QuadTreeState<CanvasTube, OrderTracker> {
    fn clone_as_stateenum(&self) -> StateEnum {
        StateEnum::TubeOrderTracking(self.clone())
    }
}

impl ClonableState for QuadTreeState<CanvasTube, NullStateTracker> {
    fn clone_as_stateenum(&self) -> StateEnum {
        StateEnum::TubeNoTracking(self.clone())
    }
}

impl ClonableState for QuadTreeState<CanvasSquare, LastAttachTimeTracker> {
    fn clone_as_stateenum(&self) -> StateEnum {
        StateEnum::SquareLastAttachTimeTracking(self.clone())
    }
}

impl ClonableState for QuadTreeState<CanvasPeriodic, LastAttachTimeTracker> {
    fn clone_as_stateenum(&self) -> StateEnum {
        StateEnum::PeriodicLastAttachTimeTracking(self.clone())
    }
}

impl ClonableState for QuadTreeState<CanvasTube, LastAttachTimeTracker> {
    fn clone_as_stateenum(&self) -> StateEnum {
        StateEnum::TubeLastAttachTimeTracking(self.clone())
    }
}

impl ClonableState for QuadTreeState<CanvasSquare, PrintEventTracker> {
    fn clone_as_stateenum(&self) -> StateEnum {
        StateEnum::SquarePrintEventTracking(self.clone())
    }
}

impl ClonableState for QuadTreeState<CanvasPeriodic, PrintEventTracker> {
    fn clone_as_stateenum(&self) -> StateEnum {
        StateEnum::PeriodicPrintEventTracking(self.clone())
    }
}

impl ClonableState for QuadTreeState<CanvasTube, PrintEventTracker> {
    fn clone_as_stateenum(&self) -> StateEnum {
        StateEnum::TubePrintEventTracking(self.clone())
    }
}

#[enum_dispatch(
    State,
    StateStatus,
    Canvas,
    RateStore,
    TrackerData,
    CloneAsStateEnum,
    TileCounts
)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StateEnum {
    SquareCanvasNullTracker(QuadTreeState<CanvasSquare, NullStateTracker>),
    PeriodicCanvasNoTracker(QuadTreeState<CanvasPeriodic, NullStateTracker>),
    TubeNoTracking(QuadTreeState<CanvasTube, NullStateTracker>),
    SquareOrderTracking(QuadTreeState<CanvasSquare, OrderTracker>),
    PeriodicOrderTracking(QuadTreeState<CanvasPeriodic, OrderTracker>),
    TubeOrderTracking(QuadTreeState<CanvasTube, OrderTracker>),
    SquareLastAttachTimeTracking(QuadTreeState<CanvasSquare, LastAttachTimeTracker>),
    PeriodicLastAttachTimeTracking(QuadTreeState<CanvasPeriodic, LastAttachTimeTracker>),
    TubeLastAttachTimeTracking(QuadTreeState<CanvasTube, LastAttachTimeTracker>),
    SquarePrintEventTracking(QuadTreeState<CanvasSquare, PrintEventTracker>),
    PeriodicPrintEventTracking(QuadTreeState<CanvasPeriodic, PrintEventTracker>),
    TubePrintEventTracking(QuadTreeState<CanvasTube, PrintEventTracker>),
}

impl StateEnum {
    pub fn empty(
        shape: (usize, usize),
        kind: CanvasType,
        tracking: TrackingType,
        n_tile_types: usize,
    ) -> Result<StateEnum, GrowError> {
        Ok(match kind {
            CanvasType::Square => match tracking {
                TrackingType::None => {
                    QuadTreeState::<CanvasSquare, NullStateTracker>::empty_with_types(
                        shape,
                        n_tile_types,
                    )?
                    .into()
                }
                TrackingType::Order => {
                    QuadTreeState::<CanvasSquare, OrderTracker>::empty_with_types(
                        shape,
                        n_tile_types,
                    )?
                    .into()
                }
                TrackingType::LastAttachTime => {
                    QuadTreeState::<CanvasSquare, LastAttachTimeTracker>::empty_with_types(
                        shape,
                        n_tile_types,
                    )?
                    .into()
                }
                TrackingType::PrintEvent => {
                    QuadTreeState::<CanvasSquare, PrintEventTracker>::empty_with_types(
                        shape,
                        n_tile_types,
                    )?
                    .into()
                }
            },
            CanvasType::Periodic => match tracking {
                TrackingType::None => {
                    QuadTreeState::<CanvasPeriodic, NullStateTracker>::empty_with_types(
                        shape,
                        n_tile_types,
                    )?
                    .into()
                }
                TrackingType::Order => {
                    QuadTreeState::<CanvasPeriodic, OrderTracker>::empty_with_types(
                        shape,
                        n_tile_types,
                    )?
                    .into()
                }
                TrackingType::LastAttachTime => {
                    QuadTreeState::<CanvasPeriodic, LastAttachTimeTracker>::empty_with_types(
                        shape,
                        n_tile_types,
                    )?
                    .into()
                }
                TrackingType::PrintEvent => {
                    QuadTreeState::<CanvasPeriodic, PrintEventTracker>::empty_with_types(
                        shape,
                        n_tile_types,
                    )?
                    .into()
                }
            },
            CanvasType::Tube => match tracking {
                TrackingType::None => {
                    QuadTreeState::<CanvasTube, NullStateTracker>::empty_with_types(
                        shape,
                        n_tile_types,
                    )?
                    .into()
                }
                TrackingType::Order => QuadTreeState::<CanvasTube, OrderTracker>::empty_with_types(
                    shape,
                    n_tile_types,
                )?
                .into(),
                TrackingType::LastAttachTime => {
                    QuadTreeState::<CanvasTube, LastAttachTimeTracker>::empty_with_types(
                        shape,
                        n_tile_types,
                    )?
                    .into()
                }
                TrackingType::PrintEvent => {
                    QuadTreeState::<CanvasTube, PrintEventTracker>::empty_with_types(
                        shape,
                        n_tile_types,
                    )?
                    .into()
                }
            },
        })
    }
}

#[enum_dispatch]
pub trait StateStatus {
    fn n_tiles(&self) -> NumTiles;
    fn total_events(&self) -> NumEvents;
    fn add_events(&mut self, n: NumEvents);
    fn reset_events(&mut self);
    fn add_time(&mut self, time: f64);
    fn time(&self) -> f64;
    fn record_event(&mut self, event: &system::Event);
    fn reset_tracking_assuming_empty_state(&mut self);
}

#[enum_dispatch]
pub trait TileCounts {
    fn tile_counts(&self) -> ArrayView1<NumTiles>;
    fn count_of_tile(&self, tile: Tile) -> NumTiles;

    /// Change the tile count based on the tile attaching
    fn update_attachment(&mut self, tile: Tile);
    /// Change the tile count based on the tile detaching
    fn update_detachment(&mut self, tile: Tile);
}

pub trait StateWithCreate: State + Sized {
    type Params;
    // fn new_raw(canvas: Self::RawCanvas) -> Result<Self, GrowError>;
    fn empty(params: Self::Params) -> Result<Self, GrowError>;
    fn empty_with_types(params: Self::Params, n_tile_types: usize) -> Result<Self, GrowError>;
    fn from_array(arr: Array2<Tile>) -> Result<Self, GrowError>;
    fn get_params(&self) -> Self::Params;
    fn zeroed_copy_from_state_nonzero_rate(&mut self, source: &Self) -> &mut Self;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuadTreeState<C: Canvas, T: StateTracker> {
    // #[serde(skip_serializing)]
    pub rates: QuadTreeSquareArray<Rate>,
    pub canvas: C,
    ntiles: NumTiles,
    total_events: NumEvents,
    time: f64,
    pub tracker: T,
    tile_counts: Array1<NumTiles>,
}

impl<C: Canvas, T: StateTracker> QuadTreeState<C, T> {
    pub fn recalc_ntiles(&mut self) {
        self.ntiles = self.calc_n_tiles();
    }
}

impl<C: Canvas, T: StateTracker> TileCounts for QuadTreeState<C, T> {
    fn tile_counts(&self) -> ArrayView1<NumTiles> {
        self.tile_counts.view()
    }

    fn count_of_tile(&self, tile: Tile) -> NumTiles {
        *self.tile_counts.get(tile as usize).expect(
            format!(
                "Count Of Tile out of bounds ({} not in arr of len {})",
                tile as usize,
                self.tile_counts.len()
            )
            .as_str(),
        )
    }

    fn update_attachment(&mut self, tile: Tile) {
        *self
            .tile_counts
            .get_mut(tile as usize)
            .expect("Out of bounds on attachment update") += 1;
    }

    fn update_detachment(&mut self, tile: Tile) {
        *self
            .tile_counts
            .get_mut(tile as usize)
            .expect("Out of bounds on detachment update") -= 1;
    }
}

impl<C: Canvas + CanvasCreate, T: StateTracker> State for QuadTreeState<C, T> {
    fn panicinfo(&self) -> String {
        format!(
            "{:?} {:?} {}={}",
            self.rates,
            self.canvas.raw_array(),
            self.n_tiles(),
            self.calc_n_tiles()
        )
    }
}

impl<C: Canvas, T: StateTracker> RateStore for QuadTreeState<C, T> {
    fn choose_point(&self) -> (Point, Rate) {
        self.rates.choose_point()
    }

    fn rate_at_point(&self, point: PointSafeHere) -> Rate {
        self.rates.rate_at_point(point)
    }

    fn update_point(&mut self, point: PointSafeHere, new_rate: Rate) {
        self.rates.update_point(point, new_rate)
    }

    #[inline]
    fn update_multiple(&mut self, points: &[(PointSafeHere, Rate)]) {
        self.rates.update_multiple(points);
    }

    fn total_rate(&self) -> Rate {
        self.rates.total_rate()
    }

    fn rate_array(&self) -> ArrayView2<Rate> {
        self.rates
            .0
            .first()
            .unwrap()
            .slice(s![..self.canvas.nrows(), ..self.canvas.ncols()])
    }
}

impl<C: Canvas, T: StateTracker> Canvas for QuadTreeState<C, T> {
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

    fn calc_n_tiles(&self) -> NumTiles {
        self.canvas.calc_n_tiles()
    }

    fn calc_n_tiles_with_tilearray(&self, should_be_counted: &Array1<bool>) -> NumTiles {
        self.canvas.calc_n_tiles_with_tilearray(should_be_counted)
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

    fn nrows_usable(&self) -> usize {
        self.canvas.nrows_usable()
    }

    fn ncols_usable(&self) -> usize {
        self.canvas.ncols_usable()
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

    fn draw_scaled_with_mm(
        &self,
        frame: &mut [u8],
        colors: &[[u8; 4]],
        mismatches: Array2<usize>,
        tile_size: usize,
        edge_size: usize,
    ) {
        self.canvas
            .draw_scaled_with_mm(frame, colors, mismatches, tile_size, edge_size)
    }

    fn draw_scaled(
        &self,
        frame: &mut [u8],
        colors: &[[u8; 4]],
        tile_size: usize,
        edge_size: usize,
    ) {
        self.canvas.draw_scaled(frame, colors, tile_size, edge_size)
    }
}

impl<C, T> StateWithCreate for QuadTreeState<C, T>
where
    C: Canvas + CanvasCreate<Params = (usize, usize)>,
    T: StateTracker,
{
    type Params = (usize, usize);

    fn empty(params: Self::Params) -> Result<Self, GrowError> {
        let rates: QuadTreeSquareArray<f64> =
            QuadTreeSquareArray::new_with_size(params.0, params.1);
        let canvas = C::new_sized(params)?;
        let tracker = T::default(&canvas);
        Ok(QuadTreeState::<C, T> {
            rates,
            canvas,
            ntiles: 0,
            total_events: 0,
            time: 0.,
            tracker,
            tile_counts: Array1::<NumTiles>::zeros(1),
        })
    }

    fn empty_with_types(params: Self::Params, n_tile_types: usize) -> Result<Self, GrowError> {
        let rates: QuadTreeSquareArray<f64> =
            QuadTreeSquareArray::new_with_size(params.0, params.1);
        let canvas = C::new_sized(params)?;
        let tracker = T::default(&canvas);
        Ok(QuadTreeState::<C, T> {
            rates,
            canvas,
            ntiles: 0,
            total_events: 0,
            time: 0.,
            tracker,
            tile_counts: Array1::<NumTiles>::zeros(n_tile_types),
        })
    }

    fn from_array(arr: Array2<Tile>) -> Result<Self, GrowError> {
        let shape = arr.shape();
        let rates: QuadTreeSquareArray<f64> =
            QuadTreeSquareArray::new_with_size(shape[0], shape[1]);
        let canvas = C::from_array(arr)?;
        let tracker = T::default(&canvas);
        Ok(QuadTreeState::<C, T> {
            rates,
            canvas,
            ntiles: 0,
            total_events: 0,
            time: 0.,
            tracker,
            tile_counts: Array1::<NumTiles>::zeros(1),
        })
    }

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
        self.time = source.time;
        self.tracker = source.tracker.clone();

        self.rates.1 = source.rates.1;

        self
    }

    fn get_params(&self) -> Self::Params {
        (self.canvas.nrows(), self.canvas.ncols())
    }
}

unsafe impl<C: Canvas, T: StateTracker> Send for QuadTreeState<C, T> {}

impl<C: Canvas, T: StateTracker> StateStatus for QuadTreeState<C, T> {
    #[inline(always)]
    fn n_tiles(&self) -> NumTiles {
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

    fn record_event(&mut self, event: &system::Event) {
        self.tracker.record_single_event(event, self.time);
    }

    fn reset_tracking_assuming_empty_state(&mut self) {
        self.tracker.reset_assuming_empty_state()
    }
}

impl<C: Canvas + Canvas, T: StateTracker> QuadTreeState<C, T> {
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

pub trait StateTrackSet<T>
where
    T: StateTracker,
{
    fn set_tracker(&mut self, tracker: T) -> &mut Self;
    fn tracker(&mut self) -> &T;
}

impl<C, T> StateTrackSet<T> for QuadTreeState<C, T>
where
    C: Canvas + Canvas,
    T: StateTracker,
{
    fn set_tracker(&mut self, tracker: T) -> &mut Self {
        self.tracker = tracker;
        self
    }

    fn tracker(&mut self) -> &T {
        &self.tracker
    }
}

#[enum_dispatch]

pub trait TrackerData {
    fn get_tracker_data(&self) -> RustAny;
}

impl<C: Canvas, T: StateTracker> TrackerData for QuadTreeState<C, T> {
    fn get_tracker_data(&self) -> RustAny {
        self.tracker.get_tracker_data()
    }
}

pub trait StateTracker: Clone + Debug + Sync + Send {
    fn default(canvas: &dyn Canvas) -> Self;

    fn record_single_event(&mut self, event: &system::Event, time: f64) -> &mut Self;

    fn get_tracker_data(&self) -> RustAny;

    fn reset(&mut self);

    fn reset_assuming_empty_state(&mut self) {
        self.reset()
    }
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct NullStateTracker;

impl StateTracker for NullStateTracker {
    fn default(_state: &dyn Canvas) -> Self {
        Self
    }

    fn record_single_event(&mut self, _event: &system::Event, _time: f64) -> &mut Self {
        self
    }

    fn reset(&mut self) {}

    fn get_tracker_data(&self) -> RustAny {
        RustAny(Box::new(()))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OrderTracker {
    pub order: u64,
    pub arr: Array2<NumEvents>,
}

impl StateTracker for OrderTracker {
    fn default(canvas: &dyn Canvas) -> Self {
        OrderTracker {
            order: 1,
            arr: Array2::<NumEvents>::zeros((canvas.nrows(), canvas.ncols())),
        }
    }

    fn reset(&mut self) {
        self.order = 1;
        self.arr.fill(0);
    }

    fn reset_assuming_empty_state(&mut self) {
        self.order = 1;
    }

    fn record_single_event(&mut self, event: &system::Event, _time: f64) -> &mut Self {
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

    fn get_tracker_data(&self) -> RustAny {
        RustAny(Box::new(self.arr.to_owned()))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LastAttachTimeTracker {
    pub arr: Array2<f64>,
}

impl StateTracker for LastAttachTimeTracker {
    fn default(canvas: &dyn Canvas) -> Self {
        LastAttachTimeTracker {
            arr: Array2::<f64>::from_elem((canvas.nrows(), canvas.ncols()), f64::NAN),
        }
    }

    fn reset(&mut self) {
        self.arr.fill(f64::NAN);
    }

    fn reset_assuming_empty_state(&mut self) {}

    fn record_single_event(&mut self, event: &system::Event, time: f64) -> &mut Self {
        match event {
            system::Event::None => self,
            system::Event::MonomerAttachment(p, _t) => {
                self.arr[p.0] = time;
                self
            }
            system::Event::MonomerDetachment(p) => {
                self.arr[p.0] = f64::NAN;
                self
            }
            system::Event::MonomerChange(p, _t) => {
                self.arr[p.0] = time;
                self
            }
            system::Event::PolymerChange(vec) => {
                for (p, _t) in vec {
                    self.arr[p.0] = time;
                }
                self
            }
            system::Event::PolymerAttachment(vec) => {
                for (p, _t) in vec {
                    self.arr[p.0] = time;
                }
                self
            }
            system::Event::PolymerDetachment(vec) => {
                for p in vec {
                    self.arr[p.0] = f64::NAN;
                }
                self
            }
        }
    }

    fn get_tracker_data(&self) -> RustAny {
        RustAny(Box::new(self.arr.to_owned()))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PrintEventTracker();

impl StateTracker for PrintEventTracker {
    fn default(_state: &dyn Canvas) -> Self {
        PrintEventTracker()
    }

    fn reset(&mut self) {
        // Default is to do nothing
    }

    fn record_single_event(&mut self, event: &system::Event, time: f64) -> &mut Self {
        println!("{}: {:?}", time, event);
        self
    }

    fn get_tracker_data(&self) -> RustAny {
        RustAny(Box::new(()))
    }
}
