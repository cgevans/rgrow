use super::base::*;
use crate::canvas::{
    Canvas, CanvasCreate, CanvasPeriodic, CanvasSquare, CanvasTube, CanvasTubeDiagonals,
};
use crate::tileset::{CanvasType, TrackingType};
use crate::units::{PerSecond, Second};
use crate::{
    canvas::PointSafe2,
    canvas::PointSafeHere,
    ratestore::{CreateSizedRateStore, QuadTreeSquareArray, RateStore},
    system,
};
use ndarray::prelude::*;

use enum_dispatch::enum_dispatch;
use num_traits::Zero;
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

#[enum_dispatch]
pub trait State: RateStore + Canvas + StateStatus + Sync + Send + TrackerData + TileCounts {
    fn panicinfo(&self) -> String;
}

#[enum_dispatch]
pub trait ClonableState: State {
    fn clone_as_stateenum(&self) -> StateEnum;
    fn clone_into_state(&self, target: &mut StateEnum);
}

macro_rules! impl_clonable_state {
    ($(($canvas:ty, $tracker:ty) => $variant:ident),*) => {
        $(
            impl ClonableState for QuadTreeState<$canvas, $tracker> {
                fn clone_as_stateenum(&self) -> StateEnum {
                    StateEnum::$variant(self.clone())
                }

                fn clone_into_state(&self, target: &mut StateEnum) {
                    match target {
                        StateEnum::$variant(target) => target.clone_from(self),
                        _ => panic!("Invalid target state enum variant"),
                    }
                }
            }
        )*
    };
}

impl_clonable_state! {
    (CanvasSquare, NullStateTracker) => SquareCanvasNullTracker,
    (CanvasPeriodic, NullStateTracker) => PeriodicCanvasNoTracker,
    (CanvasTube, NullStateTracker) => TubeNoTracking,
    (CanvasTubeDiagonals, NullStateTracker) => TubeDiagonalsNoTracking,

    (CanvasSquare, OrderTracker) => SquareOrderTracking,
    (CanvasPeriodic, OrderTracker) => PeriodicOrderTracking,
    (CanvasTube, OrderTracker) => TubeOrderTracking,
    (CanvasTubeDiagonals, OrderTracker) => TubeDiagonalsOrderTracking,

    (CanvasSquare, LastAttachTimeTracker) => SquareLastAttachTimeTracking,
    (CanvasPeriodic, LastAttachTimeTracker) => PeriodicLastAttachTimeTracking,
    (CanvasTube, LastAttachTimeTracker) => TubeLastAttachTimeTracking,
    (CanvasTubeDiagonals, LastAttachTimeTracker) => TubeDiagonalsLastAttachTimeTracking,

    (CanvasSquare, PrintEventTracker) => SquarePrintEventTracking,
    (CanvasPeriodic, PrintEventTracker) => PeriodicPrintEventTracking,
    (CanvasTube, PrintEventTracker) => TubePrintEventTracking,
    (CanvasTubeDiagonals, PrintEventTracker) => TubeDiagonalsPrintEventTracking,

    (CanvasSquare, MovieTracker) => SquareMovieTracking,
    (CanvasPeriodic, MovieTracker) => PeriodicMovieTracking,
    (CanvasTube, MovieTracker) => TubeMovieTracking,
    (CanvasTubeDiagonals, MovieTracker) => TubeDiagonalsMovieTracking
}

#[enum_dispatch(
    State,
    StateStatus,
    StateWithCreate,
    Canvas,
    RateStore,
    TrackerData,
    CloneAsStateEnum,
    TileCounts,
    ClonableState
)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StateEnum {
    SquareCanvasNullTracker(QuadTreeState<CanvasSquare, NullStateTracker>),
    PeriodicCanvasNoTracker(QuadTreeState<CanvasPeriodic, NullStateTracker>),
    TubeNoTracking(QuadTreeState<CanvasTube, NullStateTracker>),
    TubeDiagonalsNoTracking(QuadTreeState<CanvasTubeDiagonals, NullStateTracker>),
    SquareOrderTracking(QuadTreeState<CanvasSquare, OrderTracker>),
    PeriodicOrderTracking(QuadTreeState<CanvasPeriodic, OrderTracker>),
    TubeOrderTracking(QuadTreeState<CanvasTube, OrderTracker>),
    TubeDiagonalsOrderTracking(QuadTreeState<CanvasTubeDiagonals, OrderTracker>),
    SquareLastAttachTimeTracking(QuadTreeState<CanvasSquare, LastAttachTimeTracker>),
    PeriodicLastAttachTimeTracking(QuadTreeState<CanvasPeriodic, LastAttachTimeTracker>),
    TubeLastAttachTimeTracking(QuadTreeState<CanvasTube, LastAttachTimeTracker>),
    TubeDiagonalsLastAttachTimeTracking(QuadTreeState<CanvasTubeDiagonals, LastAttachTimeTracker>),
    SquarePrintEventTracking(QuadTreeState<CanvasSquare, PrintEventTracker>),
    PeriodicPrintEventTracking(QuadTreeState<CanvasPeriodic, PrintEventTracker>),
    TubePrintEventTracking(QuadTreeState<CanvasTube, PrintEventTracker>),
    TubeDiagonalsPrintEventTracking(QuadTreeState<CanvasTubeDiagonals, PrintEventTracker>),
    SquareMovieTracking(QuadTreeState<CanvasSquare, MovieTracker>),
    PeriodicMovieTracking(QuadTreeState<CanvasPeriodic, MovieTracker>),
    TubeMovieTracking(QuadTreeState<CanvasTube, MovieTracker>),
    TubeDiagonalsMovieTracking(QuadTreeState<CanvasTubeDiagonals, MovieTracker>),
}

impl StateEnum {
    pub fn from_array(
        array: ArrayView2<Tile>,
        kind: CanvasType,
        tracking: TrackingType,
        n_tile_types: usize,
    ) -> Result<StateEnum, GrowError> {
        let shape = array.shape();
        let mut state = StateEnum::empty((shape[0], shape[1]), kind, tracking, n_tile_types)?;
        let mut state_array = state.raw_array_mut();
        state_array.assign(&array);
        Ok(state)
    }

    pub fn empty(
        shape: (usize, usize),
        kind: CanvasType,
        tracking: TrackingType,
        n_tile_types: usize,
    ) -> Result<StateEnum, GrowError> {
        macro_rules! create_state {
            ($canvas:ty, $tracker:ty) => {
                QuadTreeState::<$canvas, $tracker>::empty_with_types(shape, n_tile_types)?.into()
            };
        }

        macro_rules! match_tracking {
            ($canvas:ty) => {
                match tracking {
                    TrackingType::None => create_state!($canvas, NullStateTracker),
                    TrackingType::Order => create_state!($canvas, OrderTracker),
                    TrackingType::LastAttachTime => create_state!($canvas, LastAttachTimeTracker),
                    TrackingType::PrintEvent => create_state!($canvas, PrintEventTracker),
                    TrackingType::Movie => create_state!($canvas, MovieTracker),
                }
            };
        }

        Ok(match kind {
            CanvasType::Square => match_tracking!(CanvasSquare),
            CanvasType::Periodic => match_tracking!(CanvasPeriodic),
            CanvasType::Tube => match_tracking!(CanvasTube),
            CanvasType::TubeDiagonals => match_tracking!(CanvasTubeDiagonals),
        })
    }

    pub fn get_movie_tracker(&self) -> Option<&MovieTracker> {
        match self {
            StateEnum::SquareMovieTracking(state) => Some(&state.tracker),
            StateEnum::PeriodicMovieTracking(state) => Some(&state.tracker),
            StateEnum::TubeMovieTracking(state) => Some(&state.tracker),
            StateEnum::TubeDiagonalsMovieTracking(state) => Some(&state.tracker),
            _ => None,
        }
    }

    pub fn clone_empty_no_tracker(&self) -> Result<StateEnum, GrowError> {
        match self {
            StateEnum::SquareMovieTracking(state) => Ok(StateEnum::SquareCanvasNullTracker(
                state.clone_empty_no_tracker()?,
            )),
            StateEnum::PeriodicMovieTracking(state) => Ok(StateEnum::PeriodicCanvasNoTracker(
                state.clone_empty_no_tracker()?,
            )),
            StateEnum::TubeMovieTracking(state) => {
                Ok(StateEnum::TubeNoTracking(state.clone_empty_no_tracker()?))
            }
            StateEnum::TubeDiagonalsMovieTracking(state) => Ok(StateEnum::TubeDiagonalsNoTracking(
                state.clone_empty_no_tracker()?,
            )),
            _ => Err(GrowError::NotSupported(
                "State does not have a movie tracker".to_string(),
            )),
        }
    }

    pub fn replay(&self, up_to_event: Option<u64>) -> Result<StateEnum, GrowError> {
        let movie_tracker = match self.get_movie_tracker() {
            Some(tracker) => tracker,
            None => {
                return Err(GrowError::NotSupported(
                    "State does not have a movie tracker".to_string(),
                ))
            }
        };
        let mut base_state = self.clone_empty_no_tracker()?;

        base_state.replay_inplace(
            &movie_tracker.coord,
            &movie_tracker.new_tile,
            &movie_tracker.event_id,
            up_to_event.unwrap_or(u64::MAX),
            Some(&movie_tracker.n_tiles),
            Some(&movie_tracker.time),
            Some(&movie_tracker.energy),
        )?;
        Ok(base_state)
    }

    pub fn replay_inplace(
        &mut self,
        coords: &[(usize, usize)],
        new_tiles: &[Tile],
        event_ids: &[u64],
        up_to_event_id: u64,
        n_tiles: Option<&[NumTiles]>,
        total_time: Option<&[Second]>,
        energy: Option<&[Energy]>,
    ) -> Result<(), GrowError> {
        let mut canvas = self.raw_array_mut();
        let mut last_idx = 0;
        for idx in 0..event_ids.len() {
            if event_ids[idx] > up_to_event_id {
                break;
            }
            canvas[(coords[idx].0, coords[idx].1)] = new_tiles[idx];
            last_idx = idx;
        }

        if let Some(n_tiles) = n_tiles {
            self.set_n_tiles(n_tiles[last_idx]);
        }
        if let Some(total_time) = total_time {
            self.add_time(total_time[last_idx]);
        }
        if let Some(energy) = energy {
            self.set_energy(energy[last_idx]);
        }
        self.add_events(event_ids[last_idx]);
        Ok(())
    }

    /// Filter trajectory to remove redundant/transient events.
    ///
    /// Returns indices of events to keep. An event is kept if:
    /// - The next event differs in (row, col, new_tile), AND
    /// - The previous event differs in (row, col) OR current new_tile != 0
    ///
    /// This removes transient attach/detach pairs that don't contribute to the final state.
    pub fn filtered_movie_indices(&self) -> Result<Vec<usize>, GrowError> {
        let tracker = if let Some(tracker) = self.get_movie_tracker() {
            tracker
        } else {
            return Err(GrowError::NotSupported(
                "State does not have a movie tracker".to_string(),
            ));
        };
        let n = tracker.coord.len();

        let mut keep = Vec::with_capacity(n);

        for i in 0..n {
            // Check condition 1: next event differs in (row, col, new_tile)
            // For the last element, there is no next, so we keep it if condition 2 is met
            let next_differs = if i + 1 < n {
                tracker.coord[i].0 != tracker.coord[i + 1].0
                    || tracker.coord[i].1 != tracker.coord[i + 1].1
                    || tracker.new_tile[i] != tracker.new_tile[i + 1]
            } else {
                true
            };

            // Check condition 2: previous event differs in (row, col) OR new_tile != 0
            // For the first element, there is no previous, so we check only new_tile
            let prev_condition = if i > 0 {
                tracker.coord[i].0 != tracker.coord[i - 1].0
                    || tracker.coord[i].1 != tracker.coord[i - 1].1
                    || tracker.new_tile[i] != 0
            } else {
                tracker.new_tile[i] != 0
            };

            if next_differs && prev_condition {
                keep.push(i);
            }
        }

        Ok(keep)
    }
}

#[enum_dispatch]
pub trait StateStatus {
    fn n_tiles(&self) -> NumTiles;
    fn total_events(&self) -> NumEvents;
    fn add_events(&mut self, n: NumEvents);
    fn reset_events(&mut self);
    fn add_time(&mut self, time: Second);
    fn set_n_tiles(&mut self, n: NumTiles);
    fn set_energy(&mut self, new_energy: Energy);
    fn change_energy(&mut self, change: Energy);
    fn energy(&self) -> Energy;
    fn time(&self) -> Second;
    fn record_event(
        &mut self,
        event: &system::Event,
        total_rate: PerSecond,
        chosen_event_rate: f64,
        energy_change: f64,
        energy: Energy,
        n_tiles: NumTiles,
    );
    fn reset_tracking_assuming_empty_state(&mut self);
}

#[enum_dispatch]
pub trait TileCounts {
    fn tile_counts(&self) -> ArrayView1<'_, NumTiles>;
    fn count_of_tile(&self, tile: Tile) -> NumTiles;

    /// Change the tile count based on the tile attaching
    fn update_attachment(&mut self, tile: Tile);
    /// Change the tile count based on the tile detaching
    fn update_detachment(&mut self, tile: Tile);
}

pub trait StateWithCreate: State + Sized + Clone {
    type Params;
    type C: Canvas;
    // fn new_raw(canvas: Self::RawCanvas) -> Result<Self, GrowError>;
    fn empty(params: Self::Params) -> Result<Self, GrowError>;
    fn empty_with_types(params: Self::Params, n_tile_types: usize) -> Result<Self, GrowError>;
    fn from_array(arr: Array2<Tile>) -> Result<Self, GrowError>;
    fn get_params(&self) -> Self::Params;
    fn zeroed_copy_from_state_nonzero_rate(&mut self, source: &Self);
    fn reset_state(&mut self);
    fn clone_empty(&self) -> Result<Self, GrowError>;
    fn clone_empty_no_tracker(&self)
        -> Result<QuadTreeState<Self::C, NullStateTracker>, GrowError>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuadTreeState<C: Canvas, T: StateTracker> {
    // #[serde(skip_serializing)]
    pub rates: QuadTreeSquareArray<PerSecond>,
    pub canvas: C,
    n_tiles: NumTiles,
    total_events: NumEvents,
    energy: Energy,
    time: Second,
    pub tracker: T,
    tile_counts: Array1<NumTiles>,
}

impl<C: Canvas, T: StateTracker> QuadTreeState<C, T> {
    pub fn recalc_ntiles(&mut self) {
        self.n_tiles = self.calc_n_tiles();
    }
}

impl<C: Canvas, T: StateTracker> TileCounts for QuadTreeState<C, T> {
    fn tile_counts(&self) -> ArrayView1<'_, NumTiles> {
        self.tile_counts.view()
    }

    fn count_of_tile(&self, tile: Tile) -> NumTiles {
        *self.tile_counts.get(tile as usize).unwrap_or_else(|| {
            panic!(
                "Count Of Tile out of bounds ({} not in arr of len {})",
                tile as usize,
                self.tile_counts.len()
            )
        })
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
    fn choose_point(&self) -> (Point, PerSecond) {
        self.rates.choose_point()
    }

    fn rate_at_point(&self, point: PointSafeHere) -> PerSecond {
        self.rates.rate_at_point(point)
    }

    fn update_point(&mut self, point: PointSafeHere, new_rate: PerSecond) {
        self.rates.update_point(point, new_rate)
    }

    #[inline]
    fn update_multiple(&mut self, points: &[(PointSafeHere, PerSecond)]) {
        self.rates.update_multiple(points);
    }

    fn total_rate(&self) -> PerSecond {
        self.rates.total_rate()
    }

    fn rate_array(&self) -> ArrayView2<'_, PerSecond> {
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

    fn raw_array(&self) -> ArrayView2<'_, Tile> {
        self.canvas.raw_array()
    }

    fn raw_array_mut(&mut self) -> ArrayViewMut2<'_, Tile> {
        self.canvas.raw_array_mut()
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
            self.n_tiles += 1
        }
        if (old_tile > 0) & (*t == 0) {
            self.n_tiles -= 1
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
            self.n_tiles -= 1
        }
        if !should_be_counted[old_tile as usize] & should_be_counted[*t as usize] {
            self.n_tiles += 1
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
    type C = C;

    fn empty(params: Self::Params) -> Result<Self, GrowError> {
        let rates: QuadTreeSquareArray<PerSecond> =
            QuadTreeSquareArray::new_with_size(params.0, params.1);
        let canvas = C::new_sized(params)?;
        let tracker = T::default(&canvas);
        Ok(QuadTreeState::<C, T> {
            rates,
            canvas,
            n_tiles: 0,
            total_events: 0,
            energy: 0.,
            time: Second::new(0.),
            tracker,
            tile_counts: Array1::<NumTiles>::zeros(1),
        })
    }

    fn empty_with_types(params: Self::Params, n_tile_types: usize) -> Result<Self, GrowError> {
        let rates: QuadTreeSquareArray<PerSecond> =
            QuadTreeSquareArray::new_with_size(params.0, params.1);
        let canvas = C::new_sized(params)?;
        let tracker = T::default(&canvas);
        Ok(QuadTreeState::<C, T> {
            rates,
            canvas,
            n_tiles: 0,
            total_events: 0,
            energy: 0.,
            time: Second::new(0.),
            tracker,
            tile_counts: Array1::<NumTiles>::zeros(n_tile_types),
        })
    }

    fn from_array(arr: Array2<Tile>) -> Result<Self, GrowError> {
        let shape = arr.shape();
        let rates: QuadTreeSquareArray<PerSecond> =
            QuadTreeSquareArray::new_with_size(shape[0], shape[1]);
        let canvas = C::from_array(arr)?;
        let tracker = T::default(&canvas);
        Ok(QuadTreeState::<C, T> {
            rates,
            canvas,
            n_tiles: 0,
            energy: 0.,
            total_events: 0,
            time: Second::new(0.),
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
    fn zeroed_copy_from_state_nonzero_rate(&mut self, source: &Self) {
        let max_level = self.rates.0.len() - 1; // FIXME: should not go into RateStore

        self.copy_level_quad(source, max_level, (0, 0));

        // General housekeeping
        self.n_tiles = source.n_tiles;
        self.energy = source.energy;
        self.total_events = source.total_events;
        self.time = source.time;
        self.tracker.clone_from(&source.tracker);
        self.rates.1 = source.rates.1;
        self.tile_counts.clone_from(&source.tile_counts);
    }

    fn get_params(&self) -> Self::Params {
        (self.canvas.nrows(), self.canvas.ncols())
    }

    fn reset_state(&mut self) {
        self.rates
            .0
            .iter_mut()
            .for_each(|r| r.fill(PerSecond::new(0.)));
        self.canvas.raw_array_mut().fill(0);
        self.n_tiles = 0;
        self.energy = 0.;
        self.total_events = 0;
        self.time = Second::new(0.);
        self.tracker.reset();
        self.tile_counts.fill(0);
    }

    fn clone_empty(&self) -> Result<Self, GrowError> {
        Self::empty_with_types(self.get_params(), self.tile_counts.len())
    }

    fn clone_empty_no_tracker(
        &self,
    ) -> Result<QuadTreeState<Self::C, NullStateTracker>, GrowError> {
        QuadTreeState::<Self::C, NullStateTracker>::empty_with_types(
            self.get_params(),
            self.tile_counts.len(),
        )
    }
}

unsafe impl<C: Canvas, T: StateTracker> Send for QuadTreeState<C, T> {}

impl<C: Canvas, T: StateTracker> StateStatus for QuadTreeState<C, T> {
    #[inline(always)]
    fn n_tiles(&self) -> NumTiles {
        self.n_tiles
    }

    #[inline(always)]
    fn total_events(&self) -> NumEvents {
        self.total_events
    }

    fn add_time(&mut self, time: Second) {
        self.time += time;
    }

    fn time(&self) -> Second {
        self.time
    }

    fn add_events(&mut self, n: NumEvents) {
        self.total_events += n;
    }

    fn reset_events(&mut self) {
        self.total_events = 0;
    }

    fn record_event(
        &mut self,
        event: &system::Event,
        total_rate: PerSecond,
        chosen_event_rate: f64,
        energy_change: f64,
        energy: Energy,
        n_tiles: NumTiles,
    ) {
        self.tracker.record_single_event(
            event,
            self.time,
            total_rate,
            chosen_event_rate,
            energy_change,
            energy,
            n_tiles,
        );
    }

    fn reset_tracking_assuming_empty_state(&mut self) {
        self.tracker.reset_assuming_empty_state()
    }

    #[inline(always)]
    fn set_n_tiles(&mut self, n: NumTiles) {
        self.n_tiles = n;
    }

    fn set_energy(&mut self, new_energy: Energy) {
        self.energy = new_energy;
    }

    fn change_energy(&mut self, change: Energy) {
        self.energy += change;
    }

    fn energy(&self) -> Energy {
        self.energy
    }
}

impl<C: Canvas + Canvas, T: StateTracker> QuadTreeState<C, T> {
    fn copy_level_quad(&mut self, source: &Self, level: usize, point: (usize, usize)) -> &mut Self {
        // FIXME: should not go into ratestore
        let (y, x) = point;

        if level > 0 {
            for (yy, xx) in &[(y, x), (y, x + 1), (y + 1, x), (y + 1, x + 1)] {
                let z = source.rates.0[level][(*yy, *xx)];
                if z > PerSecond::new(0.) {
                    self.rates.0[level][(*yy, *xx)] = z;
                    self.copy_level_quad(source, level - 1, (*yy * 2, *xx * 2));
                }
            }
        } else {
            for (yy, xx) in &[(y, x), (y, x + 1), (y + 1, x), (y + 1, x + 1)] {
                let z = source.rates.0[level][(*yy, *xx)];
                let t = unsafe { source.canvas.uv_p((*yy, *xx)) };
                if z > PerSecond::new(0.) {
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

    #[allow(clippy::too_many_arguments)]
    fn record_single_event(
        &mut self,
        event: &system::Event,
        time: Second,
        total_rate: PerSecond,
        chosen_event_rate: f64,
        energy_change: f64,
        energy: Energy,
        n_tiles: NumTiles,
    ) -> &mut Self;

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

    fn record_single_event(
        &mut self,
        _event: &system::Event,
        _time: Second,
        _total_rate: PerSecond,
        _chosen_event_rate: f64,
        _energy_change: f64,
        _energy: Energy,
        _n_tiles: NumTiles,
    ) -> &mut Self {
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

    fn record_single_event(
        &mut self,
        event: &system::Event,
        _time: Second,
        _total_rate: PerSecond,
        _chosen_event_rate: f64,
        _energy_change: f64,
        _energy: Energy,
        _n_tiles: NumTiles,
    ) -> &mut Self {
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
    pub arr: Array2<Second>,
}

impl StateTracker for LastAttachTimeTracker {
    fn default(canvas: &dyn Canvas) -> Self {
        LastAttachTimeTracker {
            arr: Array2::<Second>::from_elem(
                (canvas.nrows(), canvas.ncols()),
                Second::new(f64::NAN),
            ),
        }
    }

    fn reset(&mut self) {
        self.arr.fill(Second::new(f64::NAN));
    }

    fn reset_assuming_empty_state(&mut self) {}

    fn record_single_event(
        &mut self,
        event: &system::Event,
        time: Second,
        _total_rate: PerSecond,
        _chosen_event_rate: f64,
        _energy_change: f64,
        _energy: Energy,
        _n_tiles: NumTiles,
    ) -> &mut Self {
        match event {
            system::Event::None => self,
            system::Event::MonomerAttachment(p, _t) => {
                self.arr[p.0] = time;
                self
            }
            system::Event::MonomerDetachment(p) => {
                self.arr[p.0] = Second::new(f64::NAN);
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
                    self.arr[p.0] = Second::new(f64::NAN);
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

    fn record_single_event(
        &mut self,
        event: &system::Event,
        time: Second,
        _total_rate: PerSecond,
        _chosen_event_rate: f64,
        _energy_change: f64,
        _energy: Energy,
        _n_tiles: NumTiles,
    ) -> &mut Self {
        println!("{time}: {event:?}");
        self
    }

    fn get_tracker_data(&self) -> RustAny {
        RustAny(Box::new(()))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MovieTracker {
    pub event_id: Vec<NumEvents>,
    pub time: Vec<Second>,
    pub coord: Vec<Point>,
    pub new_tile: Vec<Tile>,
    pub total_rate_before: Vec<PerSecond>,
    pub chosen_event_rate: Vec<f64>,
    pub energy_change: Vec<f64>,
    pub energy: Vec<f64>,
    pub n_tiles: Vec<NumTiles>,
    current_event_id: NumEvents,
}

impl StateTracker for MovieTracker {
    fn default(state: &dyn Canvas) -> Self {
        let mut movie_tracker = MovieTracker {
            event_id: Vec::new(),
            time: Vec::new(),
            coord: Vec::new(),
            new_tile: Vec::new(),
            total_rate_before: Vec::new(),
            chosen_event_rate: Vec::new(),
            energy_change: Vec::new(),
            energy: Vec::new(),
            n_tiles: Vec::new(),
            current_event_id: 0,
        };
        let mut had_init = false;
        for i in 0..state.nrows() {
            for j in 0..state.ncols() {
                if state.inbounds((i, j)) {
                    let tile = state.tile_at_point(PointSafe2((i, j)));
                    if tile > 0 {
                        movie_tracker.event_id.push(0);
                        movie_tracker.time.push(Second::new(f64::NAN));
                        movie_tracker.coord.push((i, j));
                        movie_tracker.new_tile.push(tile);
                        movie_tracker.total_rate_before.push(PerSecond::zero());
                        movie_tracker.chosen_event_rate.push(f64::NAN);
                        movie_tracker.energy_change.push(0.);
                        movie_tracker.energy.push(0.);
                        movie_tracker
                            .n_tiles
                            .push(movie_tracker.n_tiles.last().unwrap_or(&0) + 1);
                        had_init = true;
                    }
                }
            }
        }
        if had_init {
            movie_tracker.current_event_id = 1;
        }
        movie_tracker
    }

    fn reset(&mut self) {
        self.event_id.clear();
        self.time.clear();
        self.coord.clear();
        self.new_tile.clear();
        self.total_rate_before.clear();
        self.chosen_event_rate.clear();
        self.energy_change.clear();
        self.energy.clear();
        self.n_tiles.clear();
        self.current_event_id = 0;
    }

    fn record_single_event(
        &mut self,
        event: &system::Event,
        time: Second,
        total_rate: PerSecond,
        chosen_event_rate: f64,
        energy_change: f64,
        energy: Energy,
        n_tiles: NumTiles,
    ) -> &mut Self {
        match event {
            system::Event::None => self,
            system::Event::MonomerAttachment(p, t) => {
                self.event_id.push(self.current_event_id);
                self.time.push(time);
                self.coord.push(p.0);
                self.new_tile.push(*t);
                self.total_rate_before.push(total_rate);
                self.energy_change.push(energy_change);
                self.energy.push(energy);
                self.n_tiles.push(n_tiles);
                self.chosen_event_rate.push(chosen_event_rate);
                self.current_event_id += 1;
                self
            }
            system::Event::MonomerDetachment(p) => {
                self.event_id.push(self.current_event_id);
                self.time.push(time);
                self.coord.push(p.0);
                self.new_tile.push(0);
                self.total_rate_before.push(total_rate);
                self.energy_change.push(energy_change);
                self.energy.push(energy);
                self.n_tiles.push(n_tiles);
                self.chosen_event_rate.push(chosen_event_rate);
                self.current_event_id += 1;
                self
            }
            system::Event::MonomerChange(p, t) => {
                self.event_id.push(self.current_event_id);
                self.time.push(time);
                self.coord.push(p.0);
                self.new_tile.push(*t);
                self.total_rate_before.push(total_rate);
                self.energy_change.push(energy_change);
                self.energy.push(energy);
                self.n_tiles.push(n_tiles);
                self.chosen_event_rate.push(chosen_event_rate);
                self.current_event_id += 1;
                self
            }
            system::Event::PolymerChange(vec) => {
                for (p, t) in vec {
                    self.event_id.push(self.current_event_id);
                    self.time.push(time);
                    self.coord.push(p.0);
                    self.new_tile.push(*t);
                    self.total_rate_before.push(total_rate);
                    self.energy_change.push(energy_change);
                    self.energy.push(energy);
                    self.n_tiles.push(n_tiles);
                    self.chosen_event_rate.push(chosen_event_rate);
                }
                self.current_event_id += 1;
                self
            }
            system::Event::PolymerAttachment(vec) => {
                for (p, t) in vec {
                    self.event_id.push(self.current_event_id);
                    self.time.push(time);
                    self.coord.push(p.0);
                    self.new_tile.push(*t);
                    self.total_rate_before.push(total_rate);
                    self.energy_change.push(energy_change);
                    self.energy.push(energy);
                    self.n_tiles.push(n_tiles);
                    self.chosen_event_rate.push(chosen_event_rate);
                }
                self.current_event_id += 1;
                self
            }
            system::Event::PolymerDetachment(vec) => {
                for p in vec {
                    self.event_id.push(self.current_event_id);
                    self.time.push(time);
                    self.coord.push(p.0);
                    self.new_tile.push(0);
                    self.total_rate_before.push(total_rate);
                    self.energy_change.push(energy_change);
                    self.energy.push(energy);
                    self.n_tiles.push(n_tiles);
                    self.chosen_event_rate.push(chosen_event_rate);
                }
                self.current_event_id += 1;
                self
            }
        }
    }

    fn get_tracker_data(&self) -> RustAny {
        // Convert time and rates to f64 for DataFrame
        let time_f64: Vec<f64> = self.time.iter().map(|&t| t.into()).collect();
        let rate_f64: Vec<f64> = self.total_rate_before.iter().map(|&r| r.into()).collect();
        // Extract row and column from coord tuples (convert to u64 for polars compatibility)
        let row: Vec<u64> = self.coord.iter().map(|c| c.0 as u64).collect();
        let col: Vec<u64> = self.coord.iter().map(|c| c.1 as u64).collect();

        // Create DataFrame
        let df = df! {
            "event_id" => &self.event_id,
            "time" => &time_f64,
            "row" => &row,
            "col" => &col,
            "new_tile" => &self.new_tile,
            "total_rate_before" => &rate_f64,
            "chosen_event_rate" => &self.chosen_event_rate,
            "energy" => &self.energy,
            "n_tiles" => &self.n_tiles,
            "energy_change" => &self.energy_change,
        }
        .expect("Failed to create DataFrame from MovieTracker data");

        RustAny(Box::new(df))
    }
}

// /// Reconstruct a state from a trajectory DataFrame up to a given index.
// ///
// /// # Arguments
// /// * `sys` - The system to use for updating the state
// /// * `base_state` - The base state to reconstruct from
// /// * `trajectory` - DataFrame with columns: row, col, new_tile
// /// * `up_to_event_id` - Event ID up to which to reconstruct (exclusive)
// ///
// /// # Returns
// /// A reconstructed StateEnum with rates updated
// pub fn reconstruct_state_from_trajectory_df(
//     sys: &SystemEnum,
//     base_state: &StateEnum,
//     trajectory: &DataFrame,
//     up_to_event_id: u64,
// ) -> Result<StateEnum, GrowError> {
//     let (rows, cols, new_tiles, event_ids, _energies) = extract_trajectory_data(trajectory)?;

//     reconstruct_state_from_trajectory(
//         sys,
//         base_state,
//         &rows,
//         &cols,
//         &new_tiles,
//         &event_ids,
//         up_to_event_id,
//     )
// }

// /// Extract trajectory data from a DataFrame.
// ///
// /// Returns (rows, cols, new_tiles, energies) vectors.
// fn extract_trajectory_data(
//     trajectory: &DataFrame,
// ) -> Result<(Vec<u64>, Vec<u64>, Vec<Tile>, Vec<u64>, Vec<f64>), GrowError> {
//     let rows = trajectory
//         .column("row")
//         .map_err(|e| GrowError::NotSupported(format!("Missing 'row' column: {e}")))?
//         .u64()
//         .map_err(|e| GrowError::NotSupported(format!("'row' column not u64: {e}")))?
//         .into_no_null_iter()
//         .collect::<Vec<_>>();

//     let cols = trajectory
//         .column("col")
//         .map_err(|e| GrowError::NotSupported(format!("Missing 'col' column: {e}")))?
//         .u64()
//         .map_err(|e| GrowError::NotSupported(format!("'col' column not u64: {e}")))?
//         .into_no_null_iter()
//         .collect::<Vec<_>>();

//     let new_tiles = trajectory
//         .column("new_tile")
//         .map_err(|e| GrowError::NotSupported(format!("Missing 'new_tile' column: {e}")))?
//         .u32()
//         .map_err(|e| GrowError::NotSupported(format!("'new_tile' column not u32: {e}")))?
//         .into_no_null_iter()
//         .collect::<Vec<_>>();

//     let event_ids = trajectory
//         .column("event_id")
//         .map_err(|e| GrowError::NotSupported(format!("Missing 'event_id' column: {e}")))?
//         .u64()
//         .map_err(|e| GrowError::NotSupported(format!("'event_id' column not u64: {e}")))?
//         .into_no_null_iter()
//         .collect::<Vec<_>>();

//     let energies = trajectory
//         .column("energy")
//         .map_err(|e| GrowError::NotSupported(format!("Missing 'energy' column: {e}")))?
//         .f64()
//         .map_err(|e| GrowError::NotSupported(format!("'energy' column not f64: {e}")))?
//         .into_no_null_iter()
//         .collect::<Vec<_>>();

//     Ok((rows, cols, new_tiles, event_ids, energies))
// }
