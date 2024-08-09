use enum_dispatch::enum_dispatch;
use ndarray::prelude::*;
use rand::thread_rng;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::base::RgrowError;
use crate::base::StringConvError;
use crate::ffs::FFSRunConfig;
use crate::ffs::FFSRunResult;
use crate::models::atam::ATAM;
use crate::models::ktam::KTAM;
use crate::models::oldktam::OldKTAM;
use crate::models::sdc1d::SDC;
use crate::state::State;
use crate::state::StateEnum;

use crate::{
    base::GrowError, base::NumEvents, base::NumTiles, canvas::PointSafeHere, state::StateWithCreate,
};

use super::base::{Point, Rate, Tile};
use crate::canvas::PointSafe2;

use std::any::Any;
use std::fmt::Debug;

use std::time::Duration;

#[cfg(feature = "ui")]
use fltk::{app, prelude::*, window::Window};

#[cfg(feature = "ui")]
use pixels::{Pixels, SurfaceTexture};

#[cfg(feature = "use_rayon")]
use rayon::prelude::*;

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
#[cfg_attr(feature = "python", pyclass(module = "rgrow"))]
pub enum NeededUpdate {
    None,
    NonZero,
    All,
}

#[cfg(feature = "ui")]
thread_local! {
    pub static APP: fltk::app::App = app::App::default()
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

#[cfg_attr(feature = "python", pyclass(module = "rgrow"))]
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
#[cfg_attr(feature = "python", pyclass(module = "rgrow"))]

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
    pub formation_rate: Rate,
    pub equilibrium_conc: f64,
}

#[cfg(feature = "python")]
#[pymethods]
impl DimerInfo {
    pub fn __repr__(&self) -> String {
        format!("{:?}", self)
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "python", pyclass(module = "rgrow"))]
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
            _ => Err(StringConvError(format!("Unknown chunk handling: {}. Valid values are \"none\", \"detach\", \"equilibrium\".", s))),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "python", pyclass(module = "rgrow"))]
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
                "Unknown chunk size: {}. Valid values are \"single\" and \"dimer\".",
                s
            ))),
        }
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

    fn take_single_step<St: State>(&self, state: &mut St, max_time_step: f64) -> StepOutcome {
        let time_step = -f64::ln(thread_rng().gen()) / state.total_rate();
        if time_step > max_time_step {
            state.add_time(max_time_step);
            return StepOutcome::NoEventIn(max_time_step);
        }
        let (point, remainder) = state.choose_point(); // todo: resultify
        let event = self.choose_event_at_point(state, PointSafe2(point), remainder); // FIXME
        if let Event::None = event {
            state.add_time(time_step);
            return StepOutcome::DeadEventAt(time_step);
        }

        self.perform_event(state, &event);
        self.update_after_event(state, &event);
        state.add_time(time_step);
        state.add_events(1);
        state.record_event(&event);
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
            if bounds.size_min.is_some_and(|ms| state.n_tiles() <= ms) {
                return Ok(EvolveOutcome::ReachedSizeMin);
            } else if bounds.size_max.is_some_and(|ms| state.n_tiles() >= ms) {
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

    #[cfg(feature = "use_rayon")]
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

    fn set_safe_points<St: State>(
        &self,
        state: &mut St,
        changelist: &[(PointSafe2, Tile)],
    ) -> &Self {
        // for (point, _) in changelist {
        //     assert!(state.inbounds(*point))
        // }
        let event = Event::PolymerChange(changelist.to_vec());
        self.perform_event(state, &event)
            .update_after_event(state, &event);
        self
    }

    fn configure_empty_state<St: State>(&self, state: &mut St) -> Result<(), GrowError> {
        for (p, t) in self.seed_locs() {
            self.set_point(state, p.0, t)?;
        }
        Ok(())
    }

    /// Perform a particular event/change to a state.  Do not update the state's time/etc,
    /// or rates, which should be done in update_after_event and take_single_step.
    fn perform_event<St: State>(&self, state: &mut St, event: &Event) -> &Self {
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

    #[cfg(not(feature = "ui"))]
    fn evolve_in_window<St: State>(
        &self,
        _state: &mut St,
        _block: Option<usize>,
        _bounds: EvolveBounds,
    ) -> Result<EvolveOutcome, RgrowError> {
        Err(RgrowError::NoUI)
    }

    #[cfg(feature = "ui")]
    fn evolve_in_window<St: State>(
        &self,
        state: &mut St,
        block: Option<usize>,
        mut bounds: EvolveBounds,
    ) -> Result<EvolveOutcome, RgrowError> {
        let (width, height) = state.draw_size();

        let mut scale = match block {
            Some(i) => i,
            None => {
                let (w, h) = app::screen_size();
                ((w - 50.) / (width as f64))
                    .min((h - 50.) / (height as f64))
                    .floor() as usize
            }
        };
        app::screen_size();

        // let sr = state.read().unwrap();
        let mut win = Window::default()
            .with_size(
                (scale * (width as usize)) as i32,
                ((scale * (height as usize)) + 30) as i32,
            )
            .with_label("rgrow");

        win.make_resizable(true);

        // add a frame with a label at the bottom of the window
        let mut frame = fltk::frame::Frame::default()
            .with_size(win.pixel_w(), 30)
            .with_pos(0, win.pixel_h() - 30)
            .with_label("Hello");
        win.end();
        win.show();

        let mut win_width = win.pixel_w() as u32;
        let mut win_height = win.pixel_h() as u32;

        let surface_texture = SurfaceTexture::new(win_width, win_height - 30, &win);

        let mut pixels = {
            Pixels::new(
                width * (scale as u32),
                height * (scale as u32),
                surface_texture,
            )?
        };

        bounds.for_wall_time = Some(Duration::from_millis(16));

        let mut evres: EvolveOutcome = EvolveOutcome::ReachedZeroRate;

        let tile_colors = self.tile_colors();

        while app::wait() {
            // Check if window was resized
            if win.w() != win_width as i32 || win.h() != win_height as i32 {
                win_width = win.pixel_w() as u32;
                win_height = win.pixel_h() as u32;
                pixels.resize_surface(win_width, win_height - 30).unwrap();
                if block.is_none() {
                    scale = (win_width / width).min((win_height - 30) / (height)) as usize;
                    if scale >= 10 {
                        scale = 10
                    } else {
                        scale = 1;
                    } // (scale - 10) % 10 + 10;
                    pixels
                        .resize_buffer(width * (scale as u32), height * (scale as u32))
                        .unwrap();
                }
                frame.set_pos(0, (win_height - 30) as i32);
                frame.set_size(win_width as i32, 30);
            }

            evres = self.evolve(state, bounds)?;
            let edge_size = scale / 10;
            let tile_size = scale - 2 * edge_size;
            let pixel_frame = pixels.frame_mut();

            if scale != 1 {
                if edge_size == 0 {
                    state.draw_scaled(pixel_frame, tile_colors, tile_size, edge_size);
                } else {
                    state.draw_scaled_with_mm(
                        pixel_frame,
                        tile_colors,
                        self.calc_mismatch_locations(state),
                        tile_size,
                        edge_size,
                    );
                }
            } else {
                state.draw(pixel_frame, tile_colors);
            }
            pixels.render()?;

            // Update text with the simulation time, events, and tiles
            frame.set_label(&format!(
                "Time: {:0.4e}\tEvents: {:0.4e}\tTiles: {}\t Mismatches: {}",
                state.time(),
                state.total_events(),
                state.n_tiles(),
                self.calc_mismatches(state) // FIXME: should not recalculate
            ));

            app::flush();
            app::awake();

            match evres {
                EvolveOutcome::ReachedWallTimeMax => {}
                EvolveOutcome::ReachedZeroRate => {}
                _ => {
                    break;
                }
            }
        }

        // Close window.
        win.hide();

        Ok(evres)
    }
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
    #[cfg(feature = "use_rayon")]
    fn evolve_states(
        &mut self,
        states: &mut [&mut StateEnum],
        bounds: EvolveBounds,
    ) -> Vec<Result<EvolveOutcome, GrowError>>;

    fn setup_state(&self, state: &mut StateEnum) -> Result<(), GrowError>;

    fn evolve_in_window(
        &self,
        state: &mut StateEnum,
        block: Option<usize>,
        bounds: EvolveBounds,
    ) -> Result<EvolveOutcome, RgrowError>;

    fn calc_mismatches(&self, state: &StateEnum) -> usize;
    fn calc_mismatch_locations(&self, state: &StateEnum) -> Array2<usize>;

    fn set_param(&mut self, name: &str, value: Box<dyn Any>) -> Result<NeededUpdate, GrowError>;
    fn get_param(&self, name: &str) -> Result<Box<dyn Any>, GrowError>;

    fn update_state(&self, state: &mut StateEnum, needed: &NeededUpdate);

    fn system_info(&self) -> String;

    fn run_ffs(&mut self, config: &FFSRunConfig) -> Result<FFSRunResult, RgrowError>;
}

impl<S: System + SystemWithDimers> DynSystem for S
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

    #[cfg(feature = "use_rayon")]
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
        &self,
        state: &mut StateEnum,
        block: Option<usize>,
        bounds: EvolveBounds,
    ) -> Result<EvolveOutcome, RgrowError> {
        self.evolve_in_window(state, block, bounds)
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
}

#[enum_dispatch(DynSystem, TileBondInfo, SystemWithDimers)]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", derive(FromPyObject))]
pub enum SystemEnum {
    KTAM,
    OldKTAM,
    ATAM,
    SDC, // StaticKTAMCover
}

#[cfg(feature = "python")]
impl IntoPy<PyObject> for SystemEnum {
    fn into_py(self, py: Python) -> PyObject {
        match self {
            SystemEnum::KTAM(ktam) => ktam.into_py(py),
            SystemEnum::OldKTAM(oldktam) => oldktam.into_py(py),
            SystemEnum::ATAM(atam) => atam.into_py(py),
            SystemEnum::SDC(sdc) => sdc.into_py(py),
        }
    }
}

#[enum_dispatch]
pub trait SystemWithDimers {
    /// Returns information on dimers that the system can form, similarly useful for starting out a state.
    fn calc_dimers(&self) -> Vec<DimerInfo>;
}

impl SystemWithDimers for SDC {
    fn calc_dimers(&self) -> Vec<DimerInfo> {
        panic!("Not implemented")
    }
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

#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
#[cfg_attr(feature = "python", pyclass(module = "rgrow"))]
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
            _ => Err(StringConvError(format!("Unknown fission handling mode: {}. Valid values are: no-fission, just-detach, keep-seeded, keep-largest, keep-weighted", s))),
        }
    }
}
