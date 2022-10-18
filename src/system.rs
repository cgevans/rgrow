use cached::{stores::SizedCache, Cached};
use fnv::FnvHashMap;
use fnv::FnvHashSet;
use ndarray::prelude::*;
use rand::{
    prelude::{Distribution, SmallRng},
    Rng,
};
use serde::{Deserialize, Serialize};

use crate::state::NullStateTracker;
use crate::state::StateTracked;
use crate::state::StateTracker;
use crate::{
    base::GrowError, base::NumEvents, base::NumTiles, canvas::PointSafeHere, state::State,
    state::StateCreate,
};

use super::base::{Energy, Glue, Point, Rate, Tile};
use crate::canvas::PointSafe2;

use super::fission;

use std::{fmt::Debug, marker::PhantomData, sync::RwLock};

type Cache = SizedCache<(Tile, Tile, Tile, Tile), f64>;

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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Seed {
    None(),
    SingleTile { point: Point, tile: Tile },
    MultiTile(FnvHashMap<Point, Tile>),
}

#[derive(Clone, Debug)]
pub struct StaticATAM {
    tile_rates: Array1<Rate>,
    strength_ns: Array2<Energy>,
    strength_we: Array2<Energy>,
    tau: Energy,
    seed: Seed,
}

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

#[derive(Debug)]
pub enum StepOutcome {
    HadEventAt(f64),
    NoEventIn(f64),
    DeadEventAt(f64),
    ZeroRate,
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
    ) {
        let mut events = 0;
        let mut time = 0.0;
        let mut rtime = match for_time {
            Some(t) => t,
            None => f64::INFINITY,
        };

        while (for_events.is_none() || events < for_events.unwrap())
            && (for_time.is_none() || time < for_time.unwrap())
            && (min_size.is_none() || state.ntiles() > min_size.unwrap())
            && (max_size.is_none() || state.ntiles() < max_size.unwrap())
        {
            let mut out = self.state_step(state, rng, rtime);
            match out {
                StepOutcome::HadEventAt(t) => {
                    events += 1;
                    time += t;
                    rtime -= t;
                }
                StepOutcome::NoEventIn(t) => {
                    time += t;
                    rtime -= t;
                    break;
                }
                StepOutcome::DeadEventAt(t) => {
                    time += t;
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

fn create_friend_data(
    energy_ns: &Array2<Energy>,
    energy_we: &Array2<Energy>,
) -> (
    Vec<FnvHashSet<Tile>>,
    Vec<FnvHashSet<Tile>>,
    Vec<FnvHashSet<Tile>>,
    Vec<FnvHashSet<Tile>>,
) {
    let mut friends_n = Vec::<FnvHashSet<Tile>>::new();
    let mut friends_e = Vec::<FnvHashSet<Tile>>::new();
    let mut friends_s = Vec::<FnvHashSet<Tile>>::new();
    let mut friends_w = Vec::<FnvHashSet<Tile>>::new();

    for _t1 in 0..energy_ns.nrows() {
        friends_n.push(FnvHashSet::default());
        friends_e.push(FnvHashSet::default());
        friends_s.push(FnvHashSet::default());
        friends_w.push(FnvHashSet::default());
    }

    for t1 in 0..energy_ns.nrows() {
        for t2 in 0..energy_ns.nrows() {
            if energy_ns[(t1, t2)] != 0. {
                friends_s[t1].insert(t2 as Tile);
                friends_n[t2].insert(t1 as Tile);
            }
            if energy_we[(t1, t2)] != 0. {
                friends_e[t1].insert(t2 as Tile);
                friends_w[t2].insert(t1 as Tile);
            }
        }
    }

    (friends_n, friends_e, friends_s, friends_w)
}

#[derive(Debug)]
struct ClonableCache(RwLock<Cache>);

impl Clone for ClonableCache {
    fn clone(&self) -> Self {
        Self(RwLock::new(self.0.read().unwrap().clone()))
    }
}

impl Default for ClonableCache {
    fn default() -> Self {
        Self(RwLock::new(Cache::with_size(10000)))
    }
}

#[derive(Debug, Clone)]
pub(crate) enum CoverType {
    NonCover,
    Cover,
    Composite,
}
#[derive(Debug, Clone)]
pub(crate) struct CoverAttach {
    pub(crate) like_tile: Tile,
    pub(crate) new_tile: Tile,
}

#[derive(Debug, Clone)]
pub(crate) struct CompositeDetach {
    pub(crate) like_tile: Tile,
    pub(crate) new_tile: Tile,
}

#[derive(Debug, Clone)]
pub struct StaticKTAMCover<S: State> {
    pub inner: StaticKTAM<S>,
    pub(crate) tile_is_cover: Vec<CoverType>,
    pub(crate) cover_attach_info: Vec<Vec<CoverAttach>>,
    pub(crate) composite_detach_info: Vec<Vec<CompositeDetach>>,
}

enum PossibleChoice {
    Remainder(Rate),
    Event(Event),
}

impl<S: State> System<S> for StaticKTAMCover<S> {
    fn update_after_event(&self, mut state: &mut S, event: &Event) {
        match event {
            Event::None => {
                panic!("Being asked to update after a dead event.")
            }
            Event::MonomerAttachment(p, _)
            | Event::MonomerDetachment(p)
            | Event::MonomerChange(p, _) => match self.inner.chunk_size {
                ChunkSize::Single => {
                    let points = [
                        state.move_sa_n(*p),
                        state.move_sa_w(*p),
                        PointSafeHere(p.0),
                        state.move_sa_e(*p),
                        state.move_sa_s(*p),
                    ];
                    self.update_points(&mut state, &points);
                }
                ChunkSize::Dimer => {
                    let mut points = Vec::with_capacity(10);
                    points.extend_from_slice(&[
                        state.move_sa_n(*p),
                        state.move_sa_w(*p),
                        PointSafeHere(p.0),
                        state.move_sa_e(*p),
                        state.move_sa_s(*p),
                        state.move_sa_nw(*p),
                        state.move_sa_ne(*p),
                        state.move_sa_sw(*p),
                    ]);

                    let w = state.move_sa_w(*p);
                    let n = state.move_sa_n(*p);

                    if state.inbounds(w.0) {
                        points.push(PointSafeHere(state.move_sh_w(w)));
                    }
                    if state.inbounds(n.0) {
                        points.push(PointSafeHere(state.move_sh_n(n)));
                    }

                    self.update_points(&mut state, &points);
                }
            },
            Event::PolymerDetachment(v) => {
                let mut points = Vec::new();
                for p in v {
                    points.extend(self.inner.points_to_update_around(state, p));
                }
                points.sort_unstable();
                points.dedup();
                self.update_points(&mut state, &points);
            }
            Event::PolymerAttachment(v) | Event::PolymerChange(v) => {
                let mut points = Vec::new();
                for (p, _) in v {
                    points.extend(self.inner.points_to_update_around(state, p));
                }
                points.sort_unstable();
                points.dedup();
                self.update_points(&mut state, &points);
            }
        }
    }

    fn event_rate_at_point(&self, state: &S, p: PointSafeHere) -> Rate {
        let t = state.v_sh(p) as usize;

        if !state.inbounds(p.0) {
            return 0.;
        }

        let sp = PointSafe2(p.0);

        match self.tile_is_cover[t] {
            CoverType::NonCover => self.inner.event_rate_at_point(state, p),
            CoverType::Cover => {
                self.inner.event_rate_at_point(state, p)
                    + self.cover_to_composite_rate(state, sp, t)
            }
            CoverType::Composite => self.composite_to_cover_rate(state, sp, t),
        }
    }

    fn choose_event_at_point(&self, state: &S, p: PointSafe2, acc: Rate) -> Event {
        let t = state.tile_at_point(p) as usize;

        match self.tile_is_cover[t] {
            CoverType::NonCover => self.inner.choose_event_at_point(state, p, acc),
            CoverType::Cover => match self.choose_cover_to_composite(state, p, t, acc) {
                PossibleChoice::Remainder(acc) => self.inner.choose_event_at_point(state, p, acc),
                PossibleChoice::Event(e) => e,
            },
            CoverType::Composite => match self.choose_composite_to_cover(state, p, t, acc) {
                PossibleChoice::Remainder(_) => {
                    panic!("Ran out of rate for composite.")
                }
                PossibleChoice::Event(e) => e,
            },
        }
    }

    fn seed_locs(&self) -> Vec<(PointSafe2, Tile)> {
        self.inner.seed_locs()
    }

    fn calc_dimers(&self) -> Vec<DimerInfo> {
        self.inner.calc_dimers()
    }

    fn calc_mismatch_locations(&self, state: &S) -> Array2<usize> {
        self.inner.calc_mismatch_locations(state)
    }

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
            state.add_time(time_step);
            return StepOutcome::DeadEventAt(time_step);
        }

        self.perform_event(&mut state, &event);
        self.update_after_event(&mut state, &event);
        state.add_time(time_step);
        StepOutcome::HadEventAt(time_step)
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

    fn perform_event(&self, state: &mut S, event: &Event) {
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

impl<S: State> StaticKTAMCover<S> {
    fn cover_to_composite_rate(&self, state: &S, p: PointSafe2, t: usize) -> Rate {
        let cc = &self.cover_attach_info[t as usize];

        let mut total_rate = 0.;
        for c in cc {
            if self
                .inner
                .bond_strength_of_tile_at_point(state, p, c.like_tile)
                > 0.
            {
                total_rate +=
                    self.inner.k_f_hat() * self.inner.tile_adj_concs[c.like_tile as usize];
            }
        }

        total_rate
    }
    fn choose_cover_to_composite(
        &self,
        state: &S,
        p: PointSafe2,
        t: usize,
        mut acc: Rate,
    ) -> PossibleChoice {
        let cc = &self.cover_attach_info[t as usize];

        for c in cc {
            if self
                .inner
                .bond_strength_of_tile_at_point(state, p, c.like_tile)
                > 0.
            {
                acc -= self.inner.k_f_hat() * self.inner.tile_adj_concs[c.like_tile as usize];
                if acc <= 0. {
                    return PossibleChoice::Event(Event::MonomerChange(p, c.new_tile));
                }
            }
        }

        PossibleChoice::Remainder(acc)
    }
    fn composite_to_cover_rate(&self, state: &S, p: PointSafe2, t: usize) -> Rate {
        let cc = &self.composite_detach_info[t as usize];

        let mut total_rate = 0.;
        for c in cc {
            total_rate += self.inner.k_f_hat()
                * f64::exp(
                    -self
                        .inner
                        .bond_strength_of_tile_at_point(state, p, c.like_tile),
                );
        }

        total_rate
    }
    fn choose_composite_to_cover(
        &self,
        state: &S,
        p: PointSafe2,
        t: usize,
        mut acc: Rate,
    ) -> PossibleChoice {
        let cc = &self.composite_detach_info[t as usize];

        for c in cc {
            acc -= self.inner.k_f_hat()
                * f64::exp(
                    -self
                        .inner
                        .bond_strength_of_tile_at_point(state, p, c.like_tile),
                );
            if acc <= 0. {
                return PossibleChoice::Event(Event::MonomerChange(p, c.new_tile));
            }
        }

        PossibleChoice::Remainder(acc)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StaticKTAM<C: State> {
    pub tile_adj_concs: Array1<Rate>,
    pub energy_ns: Array2<Energy>,
    pub energy_we: Array2<Energy>,
    friends_n: Vec<FnvHashSet<Tile>>,
    friends_e: Vec<FnvHashSet<Tile>>,
    friends_s: Vec<FnvHashSet<Tile>>,
    friends_w: Vec<FnvHashSet<Tile>>,
    #[serde(skip)]
    insertcache: ClonableCache,
    seed: Seed,
    pub k_f: f64,
    pub alpha: f64,
    pub g_se: Option<f64>,
    pub g_mc: Option<f64>,
    fission_handling: FissionHandling,
    chunk_handling: ChunkHandling,
    chunk_size: ChunkSize,
    tile_names: Vec<String>,
    tile_colors: Vec<[u8; 4]>,
    _canvas: PhantomData<*const C>,
}

unsafe impl<C: State> Send for StaticKTAM<C> {}

impl<C: State> TileBondInfo for StaticKTAMCover<C> {
    fn tile_color(&self, tile_number: Tile) -> [u8; 4] {
        self.inner.tile_colors[tile_number as usize]
    }

    fn tile_name(&self, tile_number: Tile) -> &str {
        self.inner.tile_names[tile_number as usize].as_str()
    }

    fn bond_name(&self, _bond_number: usize) -> &str {
        todo!()
    }

    fn tile_colors(&self) -> Vec<[u8; 4]> {
        self.inner.tile_colors()
    }

    fn tile_names(&self) -> Vec<String> {
        self.inner.tile_names()
    }

    fn bond_names(&self) -> Vec<String> {
        todo!()
    }
}

impl<C: State> TileBondInfo for StaticKTAM<C> {
    fn tile_color(&self, tile_number: Tile) -> [u8; 4] {
        self.tile_colors[tile_number as usize]
    }

    fn tile_name(&self, tile_number: Tile) -> &str {
        self.tile_names[tile_number as usize].as_str()
    }

    fn bond_name(&self, _bond_number: usize) -> &str {
        todo!()
    }

    fn tile_colors(&self) -> Vec<[u8; 4]> {
        self.tile_colors.clone()
    }

    fn tile_names(&self) -> Vec<String> {
        self.tile_names.clone()
    }

    fn bond_names(&self) -> Vec<String> {
        todo!()
    }
}

// // // BEGIN KTAM // // //

impl<C: State> StaticKTAM<C> {
    pub fn from_ktam(
        tile_stoics: Array1<f64>,
        tile_edges: Array2<Glue>,
        glue_strengths: Array1<f64>,
        g_se: f64,
        g_mc: f64,
        alpha: Option<f64>,
        k_f: Option<f64>,
        seed: Option<Seed>,
        fission_handling: Option<FissionHandling>,
        chunk_handling: Option<ChunkHandling>,
        chunk_size: Option<ChunkSize>,
        tile_names: Option<Vec<String>>,
        tile_colors: Option<Vec<[u8; 4]>>,
    ) -> Self {
        let ntiles = tile_stoics.len();
        assert!(ntiles == tile_edges.nrows());

        let mut energy_we: Array2<Energy> = Array2::zeros((ntiles, ntiles));
        let mut energy_ns: Array2<Energy> = Array2::zeros((ntiles, ntiles));

        for ti1 in 0..ntiles {
            for ti2 in 0..ntiles {
                let t1 = tile_edges.row(ti1);
                let t2 = tile_edges.row(ti2);
                if t1[2] == t2[0] {
                    energy_ns[(ti1, ti2)] = g_se * glue_strengths[t1[2] as usize];
                }
                if t1[1] == t2[3] {
                    energy_we[(ti1, ti2)] = g_se * glue_strengths[t1[1] as usize];
                }
            }
        }

        let tile_names_processed = match tile_names {
            Some(tn) => tn,
            None => (0..ntiles).into_iter().map(|x| x.to_string()).collect(),
        };

        let tile_colors_processed = match tile_colors {
            Some(tc) => tc,
            None => {
                let mut rng = rand::thread_rng();
                let ug = rand::distributions::Uniform::new(100u8, 254);
                (0..ntiles)
                    .into_iter()
                    .map(|_x| {
                        [
                            ug.sample(&mut rng),
                            ug.sample(&mut rng),
                            ug.sample(&mut rng),
                            0xffu8,
                        ]
                    })
                    .collect()
            }
        };

        let (friends_n, friends_e, friends_s, friends_w) =
            create_friend_data(&energy_ns, &energy_we);
        return StaticKTAM {
            tile_adj_concs: tile_stoics * f64::exp(-g_mc),
            energy_ns,
            energy_we,
            friends_n,
            friends_e,
            friends_s,
            friends_w,
            insertcache: ClonableCache(RwLock::new(Cache::with_size(10000))),
            seed: seed.unwrap_or(Seed::None()),
            alpha: alpha.unwrap_or(0.),
            g_mc: Some(g_mc),
            g_se: Some(g_se),
            k_f: k_f.unwrap_or(1e6),
            fission_handling: fission_handling.unwrap_or(FissionHandling::NoFission),
            tile_names: tile_names_processed,
            tile_colors: tile_colors_processed,
            chunk_handling: chunk_handling.unwrap_or(ChunkHandling::None),
            chunk_size: chunk_size.unwrap_or(ChunkSize::Single),
            _canvas: PhantomData,
        };
    }

    fn points_to_update_around(&self, state: &C, p: &PointSafe2) -> Vec<PointSafeHere> {
        match self.chunk_size {
            ChunkSize::Single => {
                let mut points = Vec::with_capacity(5);
                points.extend_from_slice(&[
                    state.move_sa_n(*p),
                    state.move_sa_w(*p),
                    PointSafeHere(p.0),
                    state.move_sa_e(*p),
                    state.move_sa_s(*p),
                ]);
                points
            }
            ChunkSize::Dimer => {
                let mut points = Vec::with_capacity(10);
                points.extend_from_slice(&[
                    state.move_sa_n(*p),
                    state.move_sa_w(*p),
                    PointSafeHere(p.0),
                    state.move_sa_e(*p),
                    state.move_sa_s(*p),
                    state.move_sa_nw(*p),
                    state.move_sa_ne(*p),
                    state.move_sa_sw(*p),
                ]);

                let w = state.move_sa_w(*p);
                let n = state.move_sa_n(*p);

                if state.inbounds(w.0) {
                    points.push(PointSafeHere(state.move_sh_w(w)));
                }
                if state.inbounds(n.0) {
                    points.push(PointSafeHere(state.move_sh_n(n)));
                }
                points
            }
        }
    }

    pub fn tile_concs(&self) -> Array1<f64> {
        self.tile_adj_concs.to_owned() * f64::exp(self.alpha)
    }

    fn k_f_hat(&self) -> f64 {
        self.k_f * f64::exp(self.alpha)
    }

    pub fn from_raw(
        tile_adj_concs: Array1<f64>,
        energy_ns: Array2<Energy>,
        energy_we: Array2<Energy>,
        k_f: f64,
        alpha: f64,
        fission_handling: Option<FissionHandling>,
        chunk_handling: Option<ChunkHandling>,
        chunk_size: Option<ChunkSize>,
    ) -> Self {
        let (friends_n, friends_e, friends_s, friends_w) =
            create_friend_data(&energy_ns, &energy_we);

        let ntiles = tile_adj_concs.len();

        let tile_names = (0..ntiles).into_iter().map(|x| x.to_string()).collect();

        let tile_colors = {
            let mut rng = rand::thread_rng();
            let ug = rand::distributions::Uniform::new(100u8, 254);
            (0..ntiles)
                .into_iter()
                .map(|_x| {
                    [
                        ug.sample(&mut rng),
                        ug.sample(&mut rng),
                        ug.sample(&mut rng),
                        0xff,
                    ]
                })
                .collect()
        };

        StaticKTAM {
            tile_adj_concs,
            energy_ns,
            energy_we,
            friends_n,
            friends_e,
            friends_s,
            friends_w,
            insertcache: ClonableCache(RwLock::new(Cache::with_size(10000))),
            seed: Seed::None(),
            alpha: alpha,
            g_mc: None,
            g_se: None,
            k_f: k_f,
            fission_handling: fission_handling.unwrap_or(FissionHandling::NoFission),
            tile_names,
            tile_colors,
            chunk_handling: chunk_handling.unwrap_or(ChunkHandling::None),
            chunk_size: chunk_size.unwrap_or(ChunkSize::Single),
            _canvas: PhantomData,
        }
    }

    /// Unsafe because does not check bounds of p: assumes inbounds (with border if applicable).
    /// This requires the tile to be specified because it is likely you've already accessed it.
    fn bond_strength_of_tile_at_point(&self, canvas: &C, p: PointSafe2, tile: Tile) -> Energy {
        let tn = { canvas.tile_to_n(p) };
        let tw = { canvas.tile_to_w(p) };
        let te = { canvas.tile_to_e(p) };
        let ts = { canvas.tile_to_s(p) };

        self.energy_ns[(tile as usize, ts as usize)]
            + self.energy_ns[(tn as usize, tile as usize)]
            + self.energy_we[(tile as usize, te as usize)]
            + self.energy_we[(tw as usize, tile as usize)]
    }

    fn is_seed(&self, p: Point) -> bool {
        match &self.seed {
            Seed::None() => false,
            Seed::SingleTile { point, tile: _ } => {
                if p == *point {
                    true
                } else {
                    false
                }
            }
            Seed::MultiTile(map) => {
                if map.contains_key(&p) {
                    true
                } else {
                    false
                }
            }
        }
    }

    // Dimer detachment rates are written manually.
    fn dimer_s_detach_rate(&self, canvas: &C, p: Point, t: Tile, ts: Energy) -> Rate {
        let p2 = canvas.u_move_point_s(p);
        if (!canvas.inbounds(p2)) | (unsafe { canvas.uv_p(p2) == 0 }) | self.is_seed(p2) {
            0.0
        } else {
            let t2 = unsafe { canvas.uv_p(p2) };
            {
                self.k_f_hat()
                    * Rate::exp(
                        -ts - self.bond_strength_of_tile_at_point(canvas, PointSafe2(p2), t2) // FIXME
                        + 2. * self.energy_ns[(t as usize, t2 as usize)],
                    )
            }
        }
    }

    // Dimer detachment rates are written manually.
    fn dimer_e_detach_rate(&self, canvas: &C, p: Point, t: Tile, ts: Energy) -> Rate {
        let p2 = canvas.u_move_point_e(p);
        if (!canvas.inbounds(p2)) | (unsafe { canvas.uv_p(p2) == 0 } | self.is_seed(p2)) {
            0.0
        } else {
            let t2 = unsafe { canvas.uv_p(p2) };
            {
                self.k_f_hat()
                    * Rate::exp(
                        -ts - self.bond_strength_of_tile_at_point(canvas, PointSafe2(p2), t2) // FIXME
                        + 2. * self.energy_we[(t as usize, t2 as usize)],
                    )
            }
        }
    }

    fn chunk_detach_rate(&self, canvas: &C, p: Point, t: Tile) -> Rate {
        match self.chunk_size {
            ChunkSize::Single => 0.0,
            ChunkSize::Dimer => {
                let ts = { self.bond_strength_of_tile_at_point(canvas, PointSafe2(p), t) }; // FIXME
                self.dimer_s_detach_rate(canvas, p, t, ts)
                    + self.dimer_e_detach_rate(canvas, p, t, ts)
            }
        }
    }

    fn choose_chunk_detachment(
        &self,
        canvas: &C,
        p: PointSafe2,
        tile: usize,
        acc: &mut Rate,
        now_empty: &mut Vec<PointSafe2>,
        possible_starts: &mut Vec<PointSafe2>,
    ) {
        match self.chunk_size {
            ChunkSize::Single => panic!(),
            ChunkSize::Dimer => {
                let ts = { self.bond_strength_of_tile_at_point(canvas, p, tile) };
                *acc -= self.dimer_s_detach_rate(canvas, p.0, tile, ts);
                if *acc <= 0. {
                    let p2 = PointSafe2(canvas.move_sa_s(p).0);
                    let t2 = { canvas.tile_at_point(p2) } as usize;
                    now_empty.push(p);
                    now_empty.push(p2);
                    // North tile adjacents
                    if self.energy_ns[({ canvas.tile_to_n(p) } as usize, tile)] > 0. {
                        possible_starts.push(PointSafe2(canvas.move_sa_n(p).0))
                    };
                    if self.energy_we[({ canvas.tile_to_w(p) } as usize, tile)] > 0. {
                        possible_starts.push(PointSafe2(canvas.move_sa_w(p).0))
                    };
                    if self.energy_we[(tile, { canvas.tile_to_e(p) } as usize)] > 0. {
                        possible_starts.push(PointSafe2(canvas.move_sa_e(p).0))
                    };
                    // South tile adjacents
                    if self.energy_ns[(t2, { canvas.tile_to_s(p2) } as usize)] > 0. {
                        possible_starts.push(PointSafe2(canvas.move_sa_s(p2).0))
                    };
                    if self.energy_we[({ canvas.tile_to_w(p2) } as usize, t2)] > 0. {
                        possible_starts.push(PointSafe2(canvas.move_sa_w(p2).0))
                    };
                    if self.energy_we[(t2, { canvas.tile_to_e(p2) } as usize)] > 0. {
                        possible_starts.push(PointSafe2(canvas.move_sa_e(p2).0))
                    };
                    return ();
                }
                *acc -= self.dimer_e_detach_rate(canvas, p.0, tile, ts);
                if *acc <= 0. {
                    let p2 = PointSafe2(canvas.move_sa_e(p).0);
                    let t2 = { canvas.tile_at_point(p2) } as usize;
                    now_empty.push(p);
                    now_empty.push(p2);
                    // West tile adjacents
                    if self.energy_we[({ canvas.tile_to_w(p) } as usize, tile)] > 0. {
                        possible_starts.push(PointSafe2(canvas.move_sa_w(p).0))
                    };
                    if self.energy_ns[({ canvas.tile_to_n(p) } as usize, tile)] > 0. {
                        possible_starts.push(PointSafe2(canvas.move_sa_n(p).0))
                    };
                    if self.energy_ns[(tile, { canvas.tile_to_s(p) } as usize)] > 0. {
                        possible_starts.push(PointSafe2(canvas.move_sa_s(p).0))
                    };
                    // East tile adjacents
                    if self.energy_we[(t2, { canvas.tile_to_e(p2) } as usize)] > 0. {
                        possible_starts.push(PointSafe2(canvas.move_sa_e(p2).0))
                    };
                    if self.energy_ns[({ canvas.tile_to_n(p2) } as usize, t2)] > 0. {
                        possible_starts.push(PointSafe2(canvas.move_sa_n(p2).0))
                    };
                    if self.energy_ns[(t2, { canvas.tile_to_s(p2) } as usize)] > 0. {
                        possible_starts.push(PointSafe2(canvas.move_sa_s(p2).0))
                    };
                    return ();
                }
                panic!("{:#?}", acc)
            }
        }
    }
}

impl<S> System<S> for StaticKTAM<S>
where
    S: State,
{
    fn event_rate_at_point(&self, canvas: &S, point: PointSafeHere) -> Rate {
        let p = if canvas.inbounds(point.0) {
            PointSafe2(point.0)
        } else {
            return 0.;
        };

        // Bound is previously checked.
        let tile = { canvas.tile_at_point(p) };

        if tile != 0 {
            // Deletion

            // Check seed
            if self.is_seed(p.0) {
                // FIXME
                return 0.0;
            }

            // Bound is previously checked
            let bound_energy = { self.bond_strength_of_tile_at_point(canvas, p, tile) };

            match self.chunk_handling {
                ChunkHandling::None => self.k_f_hat() * Rate::exp(-bound_energy),
                ChunkHandling::Detach | ChunkHandling::Equilibrium => {
                    self.k_f_hat() * Rate::exp(-bound_energy)
                        + self.chunk_detach_rate(canvas, p.0, tile) // FIXME
                }
            }
        } else {
            let tw = { canvas.tile_to_w(p) };
            let te = { canvas.tile_to_e(p) };
            let ts = { canvas.tile_to_s(p) };
            let tn = { canvas.tile_to_n(p) };

            // Short circuit if no adjacent tiles.
            if (tn == 0) & (tw == 0) & (te == 0) & (ts == 0) {
                return 0.0;
            }

            // Insertion
            let mut ic = self.insertcache.0.write().unwrap();

            match ic.cache_get(&(tn, te, ts, tw)) {
                Some(acc) => self.k_f_hat() * *acc,

                None => {
                    drop(ic);

                    let mut friends = FnvHashSet::<Tile>::default();

                    if tn != 0 {
                        friends.extend(&self.friends_s[tn as usize]);
                    }
                    if te != 0 {
                        friends.extend(&self.friends_w[te as usize]);
                    }
                    if ts != 0 {
                        friends.extend(&self.friends_n[ts as usize]);
                    }
                    if tw != 0 {
                        friends.extend(&self.friends_e[tw as usize]);
                    }

                    let mut acc = 0.;
                    for t in friends.drain() {
                        acc += self.tile_adj_concs[t as usize];
                    }

                    self.insertcache
                        .0
                        .write()
                        .unwrap()
                        .cache_set((tn, te, ts, tw), acc);

                    self.k_f_hat() * acc
                }
            }

            //     Zip::from(self.energy_ns.row(tn))
            //         .and(self.energy_we.column(te))
            //         .and(self.energy_ns.column(ts))
            //         .and(self.energy_we.row(tw))
            //         .and(&self.tile_rates)
            //         .fold(0., |acc, &n, &e, &s, &w, &r| {
            //             if (n != 0.) | (e != 0.) | (s != 0.) | (w != 0.) {
            //                 acc + r
            //             } else {
            //                 acc
            //             }
            //         })
        }
    }

    fn choose_event_at_point(&self, canvas: &S, p: PointSafe2, mut acc: Rate) -> Event {
        let tile = { canvas.tile_at_point(p) as usize };

        let tn = { canvas.tile_to_n(p) as usize };
        let tw = { canvas.tile_to_w(p) as usize };
        let te = { canvas.tile_to_e(p) as usize };
        let ts = { canvas.tile_to_s(p) as usize };

        if tile != 0 {
            acc -= {
                self.k_f_hat() * Rate::exp(-self.bond_strength_of_tile_at_point(canvas, p, tile))
            };

            let mut possible_starts = Vec::new();
            let mut now_empty = Vec::new();

            if acc <= 0. {
                // FIXME
                if self.energy_ns[(tn, tile)] > 0. {
                    possible_starts.push(PointSafe2(canvas.move_sa_n(p).0))
                };
                if self.energy_we[(tw, tile)] > 0. {
                    possible_starts.push(PointSafe2(canvas.move_sa_w(p).0))
                };
                if self.energy_ns[(tile, ts)] > 0. {
                    possible_starts.push(PointSafe2(canvas.move_sa_s(p).0))
                };
                if self.energy_we[(tile, te)] > 0. {
                    possible_starts.push(PointSafe2(canvas.move_sa_e(p).0))
                };

                now_empty.push(p);

                match self.determine_fission(canvas, &possible_starts, &now_empty) {
                    fission::FissionResult::NoFission => Event::MonomerDetachment(p),
                    fission::FissionResult::FissionGroups(g) => {
                        //println!("Fission handling {:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?}", p, tile, possible_starts, now_empty, tn, te, ts, tw, canvas.calc_ntiles(), g.map.len());
                        match self.fission_handling {
                            FissionHandling::NoFission => Event::None,
                            FissionHandling::JustDetach => Event::MonomerDetachment(p),
                            FissionHandling::KeepSeeded => {
                                let sl = self.seed_locs();
                                Event::PolymerDetachment(g.choose_deletions_seed_unattached(sl))
                            }
                            FissionHandling::KeepLargest => {
                                Event::PolymerDetachment(g.choose_deletions_keep_largest_group())
                            }
                            FissionHandling::KeepWeighted => {
                                Event::PolymerDetachment(g.choose_deletions_size_weighted())
                            }
                        }
                    }
                }
            } else {
                match self.chunk_handling {
                    ChunkHandling::None => {
                        panic!("Ran out of event possibilities at {:#?}, acc={:#?}", p, acc)
                    }
                    ChunkHandling::Detach | ChunkHandling::Equilibrium => {
                        self.choose_chunk_detachment(
                            canvas,
                            p,
                            tile,
                            &mut acc,
                            &mut now_empty,
                            &mut possible_starts,
                        );
                    }
                }

                match self.determine_fission(canvas, &possible_starts, &now_empty) {
                    fission::FissionResult::NoFission => Event::PolymerDetachment(now_empty),
                    fission::FissionResult::FissionGroups(g) => {
                        //println!("Fission handling {:?} {:?}", p, tile);
                        match self.fission_handling {
                            FissionHandling::NoFission => Event::None,
                            FissionHandling::JustDetach => Event::PolymerDetachment(now_empty),
                            FissionHandling::KeepSeeded => {
                                let sl = System::<S>::seed_locs(self);
                                Event::PolymerDetachment(g.choose_deletions_seed_unattached(sl))
                            }
                            FissionHandling::KeepLargest => {
                                Event::PolymerDetachment(g.choose_deletions_keep_largest_group())
                            }
                            FissionHandling::KeepWeighted => {
                                Event::PolymerDetachment(g.choose_deletions_size_weighted())
                            }
                        }
                    }
                }
            }
        } else {
            let mut friends = FnvHashSet::<Tile>::default();

            friends.extend(&self.friends_s[tn as usize]);
            friends.extend(&self.friends_w[te as usize]);
            friends.extend(&self.friends_e[tw as usize]);
            friends.extend(&self.friends_n[ts as usize]);

            for t in friends.drain() {
                acc -= self.k_f_hat() * self.tile_adj_concs[t as usize];
                if acc <= 0. {
                    return Event::MonomerAttachment(p, t);
                };
            }

            panic!();

            // // Insertion is hard!
            // let r = Zip::indexed(self.energy_ns.row(tn))
            //     .and(self.energy_we.column(te))
            //     .and(self.energy_ns.column(ts))
            //     .and(self.energy_we.row(tw))
            //     .and(&self.tile_rates)
            //     .fold_while((acc, 0), |(acc, _v), i, &n, &e, &s, &w, &r| {
            //         if (n != 0.) | (e != 0.) | (s != 0.) | (w != 0.) {
            //             if acc - r > 0. {
            //                 FoldWhile::Continue((acc - r, 0))
            //             } else {
            //                 FoldWhile::Done((acc - r, i))
            //             }
            //         } else {
            //             FoldWhile::Continue((acc, 0))
            //         }
            //     });

            // match r {
            //     FoldWhile::Done((_acc, i)) => i,

            //     FoldWhile::Continue((_acc, _i)) => panic!(),
            // }
        }
    }

    fn seed_locs(&self) -> Vec<(PointSafe2, Tile)> {
        let mut v = Vec::new();

        match &self.seed {
            Seed::None() => {}
            Seed::SingleTile { point, tile } => {
                v.push((PointSafe2(*point), *tile)); // FIXME
            }
            Seed::MultiTile(f) => {
                for (p, t) in f.into_iter() {
                    v.push((PointSafe2(*p), *t));
                }
            }
        };

        v
    }

    fn calc_dimers(&self) -> Vec<DimerInfo> {
        let mut dvec = Vec::new();

        for ((t1, t2), e) in self.energy_ns.indexed_iter() {
            if *e != 0. {
                let biconc =
                    f64::exp(2. * self.alpha) * self.tile_adj_concs[t1] * self.tile_adj_concs[t2];
                dvec.push(DimerInfo {
                    t1: t1 as Tile,
                    t2: t2 as Tile,
                    orientation: Orientation::NS,
                    formation_rate: self.k_f * biconc,
                    equilibrium_conc: biconc * f64::exp(*e - self.alpha),
                });
            }
        }

        for ((t1, t2), e) in self.energy_we.indexed_iter() {
            if *e != 0. {
                let biconc =
                    f64::exp(2. * self.alpha) * self.tile_adj_concs[t1] * self.tile_adj_concs[t2];
                dvec.push(DimerInfo {
                    t1: t1 as Tile,
                    t2: t2 as Tile,
                    orientation: Orientation::WE,
                    formation_rate: self.k_f * biconc,
                    equilibrium_conc: biconc * f64::exp(*e - self.alpha),
                });
            }
        }

        dvec
    }

    fn update_after_event(&self, mut state: &mut S, event: &Event) {
        match event {
            Event::None => {
                panic!("Being asked to update after a dead event.")
            }
            Event::MonomerAttachment(p, _)
            | Event::MonomerDetachment(p)
            | Event::MonomerChange(p, _) => match self.chunk_size {
                ChunkSize::Single => {
                    let points = [
                        state.move_sa_n(*p),
                        state.move_sa_w(*p),
                        PointSafeHere(p.0),
                        state.move_sa_e(*p),
                        state.move_sa_s(*p),
                    ];
                    self.update_points(&mut state, &points);
                }
                ChunkSize::Dimer => {
                    let mut points = Vec::with_capacity(10);
                    points.extend_from_slice(&[
                        state.move_sa_n(*p),
                        state.move_sa_w(*p),
                        PointSafeHere(p.0),
                        state.move_sa_e(*p),
                        state.move_sa_s(*p),
                        state.move_sa_nw(*p),
                        state.move_sa_ne(*p),
                        state.move_sa_sw(*p),
                    ]);

                    let w = state.move_sa_w(*p);
                    let n = state.move_sa_n(*p);

                    if state.inbounds(w.0) {
                        points.push(PointSafeHere(state.move_sh_w(w)));
                    }
                    if state.inbounds(n.0) {
                        points.push(PointSafeHere(state.move_sh_n(n)));
                    }

                    self.update_points(&mut state, &points);
                }
            },
            Event::PolymerDetachment(v) => {
                let mut points = Vec::new();
                for p in v {
                    points.extend(self.points_to_update_around(state, p));
                }
                points.sort_unstable();
                points.dedup();
                self.update_points(&mut state, &points);
            }
            Event::PolymerAttachment(v) | Event::PolymerChange(v) => {
                let mut points = Vec::new();
                for (p, _) in v {
                    points.extend(self.points_to_update_around(state, p));
                }
                points.sort_unstable();
                points.dedup();
                self.update_points(&mut state, &points);
            }
        }
    }

    fn calc_mismatch_locations(&self, state: &S) -> Array2<usize> {
        let threshold = 0.1;
        let mut arr = Array2::zeros(state.raw_array().raw_dim());

        for y in 1..(arr.nrows() - 1) {
            for x in 1..(arr.ncols() - 1) {
                let p = PointSafe2((y, x));
                let t = state.tile_at_point(p) as usize;

                if t == 0 {
                    arr[(y, x)] = 0;
                    continue;
                }

                let tn = state.tile_to_n(p) as usize;
                let te = state.tile_to_e(p) as usize;
                let ts = state.tile_to_s(p) as usize;
                let tw = state.tile_to_w(p) as usize;

                let nm = ((tn != 0) & (self.energy_ns[(tn, t)] < threshold)) as usize;
                let ne = ((te != 0) & (self.energy_we[(t, te)] < threshold)) as usize;
                let ns = ((ts != 0) & (self.energy_ns[(t, ts)] < threshold)) as usize;
                let nw = ((tw != 0) & (self.energy_we[(tw, t)] < threshold)) as usize;

                arr[(y, x)] = nm + ne + ns + nw;
            }
        }

        arr
    }
}
