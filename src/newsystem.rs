use crate::{
    base::Point,
    canvas::{Canvas, PointSafe2, PointSafeHere},
    state::{State, StateTracked, StateTracker},
    system::{DimerInfo, Event, Orientation, System, TileBondInfo},
};
use fnv::{FnvHashMap, FnvHashSet};
use ndarray::prelude::*;
use num_traits::Zero;
use serde::{Deserialize, Serialize};
use std::{
    marker::PhantomData,
    ops::{Add, AddAssign, Mul, Neg, SubAssign},
};
/// A concentration, in nM.  Note that this means u_0 is not 1.
// #[derive(PartialEq, Debug, Clone, Copy, Serialize, Deserialize)]
// pub struct Conc(f64);

// impl Add for Conc {
//     type Output = Self;

//     fn add(self, rhs: Self) -> Self::Output {
//         Self(self.0 + rhs.0)
//     }
// }

// impl Zero for Conc {
//     fn zero() -> Self {
//         Self(0.)
//     }

//     fn is_zero(&self) -> bool {
//         self.0.is_zero()
//     }
// }

// impl Into<f64> for Conc {
//     fn into(self) -> f64 {
//         self.0
//     }
// }

// impl From<f64> for Conc {
//     fn from(x: f64) -> Self {
//         Conc(x)
//     }
// }

type Conc = f64;

// #[derive(PartialEq, Debug, Clone, Copy, Serialize, Deserialize)]
// pub struct Glue(usize);

// impl Add for Glue {
//     type Output = Glue;

//     fn add(self, rhs: Self) -> Self::Output {
//         todo!()
//     }
// }

// impl Zero for Glue {
//     fn zero() -> Self {
//         Self(0)
//     }

//     fn is_zero(&self) -> bool {
//         self.0.is_zero()
//     }
// }

type Glue = usize;

//#[derive(PartialEq, Eq, Debug, Hash, Clone, Copy)]
type Tile = usize;

trait NonZero {
    fn nonzero(self) -> bool;
}

impl NonZero for Tile {
    fn nonzero(self) -> bool {
        self > 0
    }
}

const FAKE_EVENT_RATE: f64 = 1e-20;

// impl Index<Tile> for ArrayBase<OwnedRepr<Tile>, Dim<[usize; 1]>> {
//     type Output = Tile;

//     fn index(&self, index: Tile) -> &Self::Output {
//         &self[index.0]
//     }
// }

// #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
// pub struct Strength(f64);

// impl Add for Strength {
//     type Output = Self;

//     fn add(self, rhs: Self) -> Self::Output {
//         Self(self.0 + rhs.0)
//     }
// }

// impl Zero for Strength {
//     fn zero() -> Self {
//         Self(0.)
//     }

//     fn is_zero(&self) -> bool {
//         self.0.is_zero()
//     }
// }

type Strength = f64;

// #[derive(PartialEq, Debug, Clone, Copy, Serialize, Deserialize)]

// pub struct RatePerConc(f64);

// impl Mul<Conc> for RatePerConc {
//     type Output = Rate;

//     fn mul(self, rhs: Conc) -> Self::Output {
//         Rate(self.0 * rhs.0)
//     }
// }

type RatePerConc = f64;

/// Unitless energy.
// #[derive(PartialEq, Debug, Clone, Copy, Serialize, Deserialize)]

// pub struct Energy(f64);

// impl Add for Energy {
//     type Output = Self;

//     fn add(self, rhs: Self) -> Self::Output {
//         Energy(self.0 + rhs.0)
//     }
// }

// impl Into<f64> for Energy {
//     fn into(self) -> f64 {
//         self.0
//     }
// }

// impl From<f64> for Energy {
//     fn from(x: f64) -> Self {
//         Energy(x)
//     }
// }

// impl AddAssign for Energy {
//     fn add_assign(&mut self, rhs: Self) {
//         self.0 += rhs.0
//     }
// }

// impl Zero for Energy {
//     fn zero() -> Self {
//         Self(0.)
//     }

//     fn is_zero(&self) -> bool {
//         self.0.is_zero()
//     }
// }

// impl Neg for Energy {
//     type Output = Self;

//     fn neg(self) -> Self::Output {
//         Energy(-self.0)
//     }
// }

// impl Mul<Strength> for Energy {
//     type Output = Energy;

//     fn mul(self, rhs: Strength) -> Self::Output {
//         Energy(self.0 * rhs.0)
//     }
// }

type Energy = f64;

fn energy_exp_times_u0(x: f64) -> Conc {
    (1e9 * x.exp())
}

/// Rate in Hz
// #[derive(PartialEq, PartialOrd, Debug, Clone, Copy)]
// pub struct Rate(f64);

// impl AddAssign for Rate {
//     fn add_assign(&mut self, rhs: Self) {
//         self.0 += rhs.0
//     }
// }

// impl Add for Rate {
//     type Output = Rate;

//     fn add(self, rhs: Self) -> Self::Output {
//         Rate(self.0 + rhs.0)
//     }
// }

// impl SubAssign for Rate {
//     fn sub_assign(&mut self, rhs: Self) {
//         self.0 -= rhs.0
//     }
// }

type Rate = f64;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Seed {
    None(),
    SingleTile { point: PointSafe2, tile: Tile },
    MultiTile(FnvHashMap<PointSafe2, Tile>),
}

enum TileShape {
    Single,
    DupleToRight(Tile),
    DupleToBottom(Tile),
    DupleToLeft(Tile),
    DupleToTop(Tile),
}

// pub trait State {
//     fn tile_at_point(&self, p: Point) -> Tile;
//     fn tile_to_n(&self, p: Point) -> Tile;
//     fn tile_to_e(&self, p: Point) -> Tile;
//     fn tile_to_s(&self, p: Point) -> Tile;
//     fn tile_to_w(&self, p: Point) -> Tile;
//     fn tile_to_nn(&self, p: Point) -> Tile;
//     fn tile_to_ne(&self, p: Point) -> Tile;
//     fn tile_to_ee(&self, p: Point) -> Tile;
//     fn tile_to_se(&self, p: Point) -> Tile;
//     fn tile_to_ss(&self, p: Point) -> Tile;
//     fn tile_to_sw(&self, p: Point) -> Tile;
//     fn tile_to_ww(&self, p: Point) -> Tile;
//     fn tile_to_nw(&self, p: Point) -> Tile;
// }

// pub enum Event {
//     /// A function was asked for an event within some rate accumulator, but the function passed
//     /// on (the rate accumulutor wasn't used up), and so it is passing back the remainder.
//     Passed(Rate),
//     None,
//     MonomerAttachment(PointSafe2, Tile),
//     MonomerDetachment(PointSafe2)
// }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewKTAM<C: Canvas> {
    /// Tile names, as strings.  Only used for reference.
    pub tile_names: Vec<String>,
    /// Tile concentrations, actual (not modified by alpha/Gse/etc) in nM.
    pub tile_concs: Array1<Conc>,
    /// Glues (by number) on tile edges.
    pub tile_edges: Array2<Glue>,
    /// Strengths of self-binding glues (eg, glue 1 binding to glue 1),
    /// in abstract strength.
    pub glue_strengths: Array1<Strength>,
    /// Strengths of links between different glues (eg, glue 1 binding to
    /// glue 2).  Should be symmetric.  Will be added with glue_strengths.
    pub glue_links: Array2<Strength>,
    pub g_se: Energy,
    pub alpha: Energy,
    pub kf: RatePerConc,
    pub double_to_right: Array1<Tile>,
    pub double_to_bottom: Array1<Tile>,
    pub seed: Seed,
    // End of public stuff, now moving to calculated stuff.
    energy_ns: Array2<Energy>,
    energy_we: Array2<Energy>,

    /// Each "friends" hashset gives the potential tile attachments
    /// at point P if tile T is in that direction.  Eg, friends_e[T]
    /// is a set of tiles that might attach at point P if T is east of
    /// point P.  The ones other than NESW are only for duples.
    friends_n: Vec<FnvHashSet<Tile>>,
    friends_e: Vec<FnvHashSet<Tile>>,
    friends_s: Vec<FnvHashSet<Tile>>,
    friends_w: Vec<FnvHashSet<Tile>>,
    friends_ne: Vec<FnvHashSet<Tile>>,
    friends_ee: Vec<FnvHashSet<Tile>>,
    friends_se: Vec<FnvHashSet<Tile>>,
    friends_ss: Vec<FnvHashSet<Tile>>,
    friends_sw: Vec<FnvHashSet<Tile>>,

    has_duples: bool,
    double_to_left: Array1<Tile>,
    double_to_top: Array1<Tile>,

    /// We need to store the type of canvas we're using so we know
    /// how to move around.
    _canvas: PhantomData<*const C>,
}

unsafe impl<C: State> Send for NewKTAM<C> {}

impl<S: State + StateTracked<T>, T: StateTracker> System<S, T> for NewKTAM<S> {
    fn update_after_event(&self, state: &mut S, event: &Event) {
        match event {
            Event::None => todo!(),
            Event::MonomerAttachment(p, _)
            | Event::MonomerDetachment(p)
            | Event::MonomerChange(p, _) => {
                let points = [
                    state.move_sa_n(*p),
                    state.move_sa_w(*p),
                    PointSafeHere(p.0),
                    state.move_sa_e(*p),
                    state.move_sa_s(*p),
                    state.move_sa_nn(*p),
                    state.move_sa_ne(*p),
                    state.move_sa_ee(*p),
                    state.move_sa_se(*p),
                    state.move_sa_ss(*p),
                    state.move_sa_sw(*p),
                    state.move_sa_ww(*p),
                    state.move_sa_nw(*p),
                ];

                self.update_points(state, &points);
            }
            Event::PolymerAttachment(_) => todo!(),
            Event::PolymerDetachment(_) => todo!(),
            Event::PolymerChange(_) => todo!(),
        }
    }

    fn event_rate_at_point(&self, state: &S, p: crate::canvas::PointSafeHere) -> crate::base::Rate {
        if !state.inbounds(p.0) {
            return 0.;
        }
        let p = PointSafe2(p.0);
        let t = state.tile_at_point(p);
        if t.nonzero() {
            self.monomer_detachment_rate_at_point(state, p)
        } else {
            self.total_monomer_attachment_rate_at_point(state, p)
        }
    }

    fn choose_event_at_point(&self, state: &S, p: PointSafe2, acc: crate::base::Rate) -> Event {
        // println!("{:?}", acc);
        match self.choose_detachment_at_point(state, p, (acc)) {
            (true, _, event) => {
                // println!("{:?} {:?}", acc, event);
                event
            }
            (false, acc, _) => match self.choose_attachment_at_point(state, p, acc) {
                (true, _, event) => {
                    // println!("{:?} {:?}", acc, event);
                    event
                }
                (false, acc, _) => {
                    panic!(
                        "Rate: {:?}, {:?}, {:?}, {:?}",
                        acc,
                        p,
                        state,
                        state.raw_array()
                    );
                }
            },
        }
    }

    fn set_point(&mut self, state: &mut S, point: Point, tile: Tile) {
        assert!(state.inbounds(point));

        let point = PointSafe2(point);
        let oldt = state.tile_at_point(point);

        match self.tile_shape(oldt) {
            // Fixme: somewhat unsafe
            TileShape::Single => (),
            TileShape::DupleToRight(dt) => {
                debug_assert_eq!(dt, state.tile_to_e(point));
                state.set_sa(&PointSafe2(state.move_sa_e(point).0), &0usize)
            }
            TileShape::DupleToBottom(dt) => {
                debug_assert_eq!(dt, state.tile_to_s(point));
                state.set_sa(&PointSafe2(state.move_sa_s(point).0), &0usize)
            }
            TileShape::DupleToLeft(dt) => {
                debug_assert_eq!(dt, state.tile_to_w(point));
                state.set_sa(&PointSafe2(state.move_sa_w(point).0), &0usize)
            }
            TileShape::DupleToTop(dt) => {
                debug_assert_eq!(dt, state.tile_to_n(point));
                state.set_sa(&PointSafe2(state.move_sa_n(point).0), &0usize)
            }
        }

        state.set_sa(&point, &tile);

        match self.tile_shape(tile) {
            TileShape::Single => (),
            TileShape::DupleToRight(dt) => {
                debug_assert_eq!(state.tile_to_e(point), 0);
                state.set_sa(&PointSafe2(state.move_sa_e(point).0), &dt);
            }
            TileShape::DupleToBottom(dt) => {
                debug_assert_eq!(state.tile_to_s(point), 0);
                state.set_sa(&PointSafe2(state.move_sa_s(point).0), &dt);
            }
            TileShape::DupleToLeft(dt) => {
                debug_assert_eq!(state.tile_to_w(point), 0);
                state.set_sa(&PointSafe2(state.move_sa_w(point).0), &dt);
            }
            TileShape::DupleToTop(dt) => {
                debug_assert_eq!(state.tile_to_n(point), 0);
                state.set_sa(&PointSafe2(state.move_sa_n(point).0), &dt);
            }
        }

        let event = Event::MonomerAttachment(point, tile);

        self.update_after_event(state, &event);
    }

    fn perform_event(&self, state: &mut S, event: &Event) {
        match event {
            Event::None => panic!("Being asked to perform null event."),
            Event::MonomerAttachment(point, tile) | Event::MonomerChange(point, tile) => {
                state.set_sa(point, tile);
                match self.tile_shape(*tile) {
                    TileShape::Single => (),
                    TileShape::DupleToRight(dt) => {
                        debug_assert_eq!(state.tile_to_e(*point), 0);
                        state.set_sa(&PointSafe2(state.move_sa_e(*point).0), &dt);
                    }
                    TileShape::DupleToBottom(dt) => {
                        debug_assert_eq!(state.tile_to_s(*point), 0);
                        state.set_sa(&PointSafe2(state.move_sa_s(*point).0), &dt);
                    }
                    TileShape::DupleToLeft(dt) => {
                        debug_assert_eq!(state.tile_to_w(*point), 0);
                        state.set_sa(&PointSafe2(state.move_sa_w(*point).0), &dt);
                    }
                    TileShape::DupleToTop(dt) => {
                        debug_assert_eq!(state.tile_to_n(*point), 0);
                        state.set_sa(&PointSafe2(state.move_sa_n(*point).0), &dt);
                    }
                }
            }
            Event::MonomerDetachment(point) => {
                match self.tile_shape(state.tile_at_point(*point)) {
                    TileShape::Single => (),
                    TileShape::DupleToRight(dt) => {
                        debug_assert_eq!(state.tile_to_e(*point), dt);
                        state.set_sa(&PointSafe2(state.move_sa_e(*point).0), &0usize);
                    }
                    TileShape::DupleToBottom(dt) => {
                        debug_assert_eq!(state.tile_to_s(*point), dt);
                        state.set_sa(&PointSafe2(state.move_sa_s(*point).0), &0usize);
                    }
                    TileShape::DupleToLeft(dt) => {
                        debug_assert_eq!(state.tile_to_w(*point), dt);
                        state.set_sa(&PointSafe2(state.move_sa_w(*point).0), &0usize);
                    }
                    TileShape::DupleToTop(dt) => {
                        debug_assert_eq!(state.tile_to_n(*point), dt);
                        state.set_sa(&PointSafe2(state.move_sa_n(*point).0), &0usize);
                    }
                }
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

    fn seed_locs(&self) -> Vec<(PointSafe2, Tile)> {
        let mut v = Vec::new();

        match &self.seed {
            Seed::None() => {}
            Seed::SingleTile { point, tile } => {
                v.push((*point, *tile)); // FIXME
            }
            Seed::MultiTile(f) => {
                for (p, t) in f.into_iter() {
                    v.push((*p, *t));
                }
            }
        };
        v
    }

    fn calc_dimers(&self) -> Vec<DimerInfo> {
        // It is (reasonably) safe for us to use the same code that we used in the old StaticKTAM, despite duples being
        // here, because our EW/NS energies include the right/bottom tiles.  However, (FIXME), we need to think about
        // how this might actually double-count / double some rates: if, eg, a single tile can attach in two places to
        // a double tile, are we double-counting the rates?  Note also that this relies on
        let mut dvec = Vec::new();

        for ((t1, t2), e) in self.energy_ns.indexed_iter() {
            if *e != 0. {
                let biconc = self.tile_concs[t1] * self.tile_concs[t2];
                dvec.push(DimerInfo {
                    t1: t1 as Tile,
                    t2: t2 as Tile,
                    orientation: Orientation::NS,
                    formation_rate: self.kf * biconc / 1e9, // FIXME: 1e9 because we're using nM for concs
                    equilibrium_conc: biconc * f64::exp(*e - self.alpha),
                });
            }
        }

        for ((t1, t2), e) in self.energy_we.indexed_iter() {
            if *e != 0. {
                let biconc = f64::exp(2. * self.alpha) * self.tile_concs[t1] * self.tile_concs[t2];
                dvec.push(DimerInfo {
                    t1: t1 as Tile,
                    t2: t2 as Tile,
                    orientation: Orientation::WE,
                    formation_rate: self.kf * biconc / 1e9, // FIXME: 1e9 because we're using nM for concs
                    equilibrium_conc: biconc * f64::exp(*e - self.alpha),
                });
            }
        }

        dvec
    }

    fn calc_mismatch_locations(&self, state: &S) -> Array2<usize> {
        todo!()
    }
}

impl<C: State> TileBondInfo for NewKTAM<C> {
    fn tile_color(&self, tile_number: Tile) -> [u8; 4] {
        todo!();
        //self.tile_colors[tile_number as usize]
    }

    fn tile_name(&self, tile_number: Tile) -> &str {
        self.tile_names[tile_number].as_str()
    }

    fn bond_name(&self, _bond_number: usize) -> &str {
        todo!()
    }

    fn tile_colors(&self) -> Vec<[u8; 4]> {
        todo!();
        //self.tile_colors.clone()
    }

    fn tile_names(&self) -> Vec<String> {
        self.tile_names.clone()
    }

    fn bond_names(&self) -> Vec<String> {
        todo!()
    }
}

impl<S: Canvas> NewKTAM<S> {
    pub fn new_sized(ntiles: Tile, nglues: usize) -> Self {
        NewKTAM {
            tile_names: Vec::new(),
            tile_concs: Array1::zeros(ntiles + 1),
            tile_edges: Array2::zeros((ntiles + 1, 4)),
            glue_strengths: Array1::zeros(nglues + 1),
            glue_links: Array2::zeros((nglues + 1, nglues + 1)),
            g_se: (9.),
            alpha: (0.),
            kf: (1e-3),
            double_to_right: Array1::zeros(ntiles + 1),
            double_to_bottom: Array1::zeros(ntiles + 1),
            seed: Seed::None(),
            energy_ns: Array2::zeros((ntiles + 1, ntiles + 1)),
            energy_we: Array2::zeros((ntiles + 1, ntiles + 1)),
            friends_n: Vec::new(),
            friends_e: Vec::new(),
            friends_s: Vec::new(),
            friends_w: Vec::new(),
            friends_ne: Vec::new(),
            friends_ee: Vec::new(),
            friends_se: Vec::new(),
            friends_ss: Vec::new(),
            friends_sw: Vec::new(),
            has_duples: false,
            double_to_left: Array1::zeros(ntiles + 1),
            double_to_top: Array1::zeros(ntiles + 1),
            _canvas: PhantomData,
        }
    }

    pub fn update_system(&mut self) {
        let ntiles = self.tile_concs.len();

        for t1 in 0..ntiles {
            for t2 in 0..ntiles {
                let t1r = self.tile_edges.row(t1);
                let t2r = self.tile_edges.row(t2);
                self.energy_ns[(t1, t2)] = self.g_se * self.glue_links[(t1r[2], t2r[0])];
                if t1r[2] == t2r[0] {
                    self.energy_ns[(t1, t2)] = self.g_se * self.glue_strengths[t1r[2]]
                }
                self.energy_we[(t1, t2)] = self.g_se * self.glue_links[(t1r[1], t2r[3])];
                if t1r[1] == t2r[3] {
                    self.energy_we[(t1, t2)] = self.g_se * self.glue_strengths[t1r[1]]
                }
            }
        }

        if (self.double_to_right.sum() > 0) || (self.double_to_bottom.sum() > 0) {
            self.has_duples = true;
            for (t1, t2) in self.double_to_right.indexed_iter() {
                if (t1 > 0) & (t2 > &0) {
                    self.double_to_left[*t2] = t1;
                }
            }
            for (t1, t2) in self.double_to_bottom.indexed_iter() {
                if (t1 > 0) & (t2 > &0) {
                    self.double_to_top[*t2] = t1;
                }
            }
        } else {
            self.has_duples = false;
        }

        self.friends_n.drain(..);
        self.friends_e.drain(..);
        self.friends_s.drain(..);
        self.friends_w.drain(..);
        self.friends_ne.drain(..);
        self.friends_ee.drain(..);
        self.friends_se.drain(..);
        self.friends_ss.drain(..);
        self.friends_sw.drain(..);
        for _ in 0..ntiles {
            self.friends_n.push(FnvHashSet::default());
            self.friends_e.push(FnvHashSet::default());
            self.friends_s.push(FnvHashSet::default());
            self.friends_w.push(FnvHashSet::default());
            self.friends_ne.push(FnvHashSet::default());
            self.friends_ee.push(FnvHashSet::default());
            self.friends_se.push(FnvHashSet::default());
            self.friends_ss.push(FnvHashSet::default());
            self.friends_sw.push(FnvHashSet::default());
        }
        for t1 in 0..ntiles {
            for t2 in 0..ntiles {
                match self.tile_shape(t1) {
                    TileShape::Single => {
                        if self.energy_ns[(t2, t1)] != 0. {
                            self.friends_n[t2].insert(t1);
                        }
                        if self.energy_we[(t2, t1)] != 0. {
                            self.friends_w[t2].insert(t1);
                        }
                        if self.energy_ns[(t1, t2)] != 0. {
                            self.friends_s[t2].insert(t1);
                        }
                        if self.energy_we[(t1, t2)] != 0. {
                            self.friends_e[t2].insert(t1);
                        }
                    }
                    TileShape::DupleToRight(td) => {
                        if self.energy_ns[(t2, td)] != 0. {
                            self.friends_ne[t2].insert(t1);
                        }
                        if self.energy_ns[(td, t2)] != 0. {
                            self.friends_se[t2].insert(t1);
                        }
                        if self.energy_we[(td, t2)] != 0. {
                            self.friends_ee[t2].insert(t1);
                        }
                        if self.energy_ns[(t2, t1)] != 0. {
                            self.friends_n[t2].insert(t1);
                        }
                        if self.energy_we[(t2, t1)] != 0. {
                            self.friends_w[t2].insert(t1);
                        }
                        if self.energy_ns[(t1, t2)] != 0. {
                            self.friends_s[t2].insert(t1);
                        }
                        if self.energy_we[(t1, t2)] != 0. {
                            self.friends_e[t2].insert(t1);
                        }
                    }
                    TileShape::DupleToBottom(td) => {
                        if self.energy_we[(t2, td)] != 0. {
                            self.friends_sw[t2].insert(t1);
                        }
                        if self.energy_we[(td, t2)] != 0. {
                            self.friends_se[t2].insert(t1);
                        }
                        if self.energy_ns[(td, t2)] != 0. {
                            self.friends_ss[t2].insert(t1);
                        }
                        if self.energy_ns[(t2, t1)] != 0. {
                            self.friends_n[t2].insert(t1);
                        }
                        if self.energy_we[(t2, t1)] != 0. {
                            self.friends_w[t2].insert(t1);
                        }
                        if self.energy_ns[(t1, t2)] != 0. {
                            self.friends_s[t2].insert(t1);
                        }
                        if self.energy_we[(t1, t2)] != 0. {
                            self.friends_e[t2].insert(t1);
                        }
                    }
                    TileShape::DupleToLeft(_) => (),
                    TileShape::DupleToTop(_) => (),
                };
            }
        }
    }

    pub fn is_seed(&self, p: PointSafe2) -> bool {
        match &self.seed {
            Seed::None() => false,
            Seed::SingleTile {
                point: seed_point,
                tile: _,
            } => return p == *seed_point,
            Seed::MultiTile(seed_map) => return seed_map.contains_key(&p),
        }
    }

    pub fn monomer_detachment_rate_at_point(&self, state: &S, p: PointSafe2) -> Rate {
        // If the point is a seed, then there is no detachment rate.
        // ODD HACK: we set a very low detachment rate for seeds and duple bottom/right, to allow
        // rate-based copying.  We ignore these below.
        if self.is_seed(p) {
            return FAKE_EVENT_RATE;
        }

        let t = state.tile_at_point(p);
        if t == 0 {
            return 0.;
        }
        if (self.has_duples) && ((self.double_to_left[t] > 0) || (self.double_to_top[t] > 0)) {
            return FAKE_EVENT_RATE;
        }
        self.kf
            * energy_exp_times_u0(-self.bond_energy_of_tile_type_at_point(state, p, t) + self.alpha)
    }

    pub fn choose_detachment_at_point(
        &self,
        state: &S,
        p: PointSafe2,
        mut acc: Rate,
    ) -> (bool, Rate, Event) {
        acc -= self.monomer_detachment_rate_at_point(state, p);
        if acc <= 0. {
            // FIXME: may slow things down
            if self.is_seed(p)
                || ((self.has_duples)
                    && ((self.double_to_left[state.tile_at_point(p)] > 0)
                        || (self.double_to_top[state.tile_at_point(p)] > 0)))
            {
                (true, acc, Event::None)
            } else {
                (true, acc, Event::MonomerDetachment(p))
            }
        } else {
            (false, acc, Event::None)
        }
    }

    pub fn total_monomer_attachment_rate_at_point(&self, state: &S, p: PointSafe2) -> Rate {
        match self._find_monomer_attachment_possibilities_at_point(state, p, (0.), true) {
            (false, acc, _) => (-acc),
            _ => panic!(),
        }
    }

    pub fn choose_attachment_at_point(
        &self,
        state: &S,
        p: PointSafe2,
        acc: Rate,
    ) -> (bool, Rate, Event) {
        self.choose_monomer_attachment_at_point(state, p, acc)
    }

    pub fn choose_monomer_attachment_at_point(
        &self,
        state: &S,
        p: PointSafe2,
        acc: Rate,
    ) -> (bool, Rate, Event) {
        self._find_monomer_attachment_possibilities_at_point(state, p, acc, false)
    }

    fn _find_monomer_attachment_possibilities_at_point(
        &self,
        state: &S,
        p: PointSafe2,
        mut acc: Rate,
        just_calc: bool,
    ) -> (bool, Rate, Event) {
        let tn = state.tile_to_n(p);
        let tw = state.tile_to_w(p);
        let te = state.tile_to_e(p);
        let ts = state.tile_to_s(p);

        let mut friends = FnvHashSet::<Tile>::default();

        if tn.nonzero() {
            friends.extend(&self.friends_n[tn]);
        }
        if te.nonzero() {
            friends.extend(&self.friends_e[te]);
        }
        if ts.nonzero() {
            friends.extend(&self.friends_s[ts]);
        }
        if tw.nonzero() {
            friends.extend(&self.friends_w[tw]);
        }

        if self.has_duples {
            let tss = state.tile_to_ss(p);
            let tne = state.tile_to_ne(p);
            let tee = state.tile_to_ee(p);
            let tse = state.tile_to_se(p);

            if tss.nonzero() {
                friends.extend(&self.friends_ss[tss])
            }
            if tne.nonzero() {
                friends.extend(&self.friends_ne[tne])
            }
            if tee.nonzero() {
                friends.extend(&self.friends_ee[tee])
            }
            if tse.nonzero() {
                friends.extend(&self.friends_se[tse])
            }
        }

        for t in friends.drain() {
            // FIXME: this is likely rather slow, but it's better than giving very confusing rates (many
            // possible double-tile attachements at a point that aren't actually possible, because they are
            // blocked).
            match self.tile_shape(t) {
                TileShape::Single => (),
                TileShape::DupleToRight(dt) => {
                    if state.tile_to_e(p) != 0 {
                        continue;
                    }
                }
                TileShape::DupleToBottom(dt) => {
                    if state.tile_to_s(p) != 0 {
                        continue;
                    }
                }
                TileShape::DupleToLeft(dt) => {
                    if state.tile_to_w(p) != 0 {
                        continue;
                    }
                }
                TileShape::DupleToTop(dt) => {
                    if state.tile_to_n(p) != 0 {
                        continue;
                    }
                }
            }
            acc -= self.kf * self.tile_concs[t];
            if !just_calc & (acc <= (0.)) {
                return (true, acc, Event::MonomerAttachment(p, t));
            }
        }
        return (false, acc, Event::None);
    }

    pub fn bond_energy_of_tile_type_at_point(&self, state: &S, p: PointSafe2, t: Tile) -> Energy {
        let tn = state.tile_to_n(p);
        let tw = state.tile_to_w(p);
        let te = state.tile_to_e(p);
        let ts = state.tile_to_s(p);

        let mut energy = self.get_energy_ns(tn, t)
            + self.get_energy_ns(t, ts)
            + self.get_energy_we(tw, t)
            + self.get_energy_we(t, te);

        if !self.has_duples {
            return energy;
        }

        match self.tile_shape(t) {
            TileShape::Single => (),
            TileShape::DupleToRight(tright) => {
                debug_assert_eq!(tright, te);
                let tne = state.tile_to_ne(p);
                let tee = state.tile_to_ee(p);
                let tse = state.tile_to_se(p);
                energy += self.get_energy_ns(tne, tright)
                    + self.get_energy_we(tright, tee)
                    + self.get_energy_ns(tright, tse);
            }
            TileShape::DupleToBottom(tbottom) => {
                debug_assert_eq!(tbottom, ts);
                let tse = state.tile_to_se(p);
                let tss = state.tile_to_ss(p);
                let tsw = state.tile_to_sw(p);
                energy += self.get_energy_we(tbottom, tse)
                    + self.get_energy_ns(tbottom, tss)
                    + self.get_energy_we(tsw, tbottom);
            }
            // We should never want to calculate this for "accessory" parts of duples.
            TileShape::DupleToLeft(_) => panic!(),
            TileShape::DupleToTop(_) => panic!(),
        };

        return energy;
    }

    fn get_energy_ns(&self, tn: Tile, ts: Tile) -> Energy {
        self.energy_ns[(tn, ts)]
    }

    fn get_energy_we(&self, tw: Tile, te: Tile) -> Energy {
        self.energy_we[(tw, te)]
    }

    fn tile_shape(&self, t: Tile) -> TileShape {
        let dr = self.double_to_right[t];
        if dr.nonzero() {
            return TileShape::DupleToRight(dr);
        }
        let db = self.double_to_bottom[t];
        if db.nonzero() {
            return TileShape::DupleToBottom(db);
        }
        let dl = self.double_to_left[t];
        if dl.nonzero() {
            return TileShape::DupleToLeft(dl);
        }
        let dt = self.double_to_top[t];
        if dt.nonzero() {
            return TileShape::DupleToTop(dt);
        }
        return TileShape::Single;
    }
}
