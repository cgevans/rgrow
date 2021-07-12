use crate::{
    canvas::{Canvas, PointSafe2, PointSafeHere},
    state::{State, StateTracked, StateTracker},
    system::{DimerInfo, Event, System, TileBondInfo},
};
use fnv::{FnvHashMap, FnvHashSet};
use ndarray::prelude::*;
use num_traits::Zero;
use std::{
    marker::PhantomData,
    ops::{Add, AddAssign, Mul, Neg, SubAssign},
};
/// A concentration, in nM.  Note that this means u_0 is not 1.
#[derive(PartialEq, Debug, Clone, Copy)]
pub struct Conc(f64);

impl Add for Conc {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl Zero for Conc {
    fn zero() -> Self {
        Self(0.)
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

#[derive(PartialEq, Debug, Clone, Copy)]
pub struct Glue(usize);

impl Add for Glue {
    type Output = Glue;

    fn add(self, rhs: Self) -> Self::Output {
        todo!()
    }
}

impl Zero for Glue {
    fn zero() -> Self {
        Self(0)
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

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

// impl Index<Tile> for ArrayBase<OwnedRepr<Tile>, Dim<[usize; 1]>> {
//     type Output = Tile;

//     fn index(&self, index: Tile) -> &Self::Output {
//         &self[index.0]
//     }
// }

#[derive(Debug, Clone, Copy)]
pub struct Strength(f64);

impl Add for Strength {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl Zero for Strength {
    fn zero() -> Self {
        Self(0.)
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

#[derive(PartialEq, Debug, Clone, Copy)]

pub struct RatePerConc(f64);

impl Mul<Conc> for RatePerConc {
    type Output = Rate;

    fn mul(self, rhs: Conc) -> Self::Output {
        Rate(self.0 * rhs.0)
    }
}

/// Unitless energy.
#[derive(PartialEq, Debug, Clone, Copy)]

pub struct Energy(f64);

impl Add for Energy {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Energy(self.0 + rhs.0)
    }
}

impl Into<f64> for Energy {
    fn into(self) -> f64 {
        self.0
    }
}

impl From<f64> for Energy {
    fn from(x: f64) -> Self {
        Energy(x)
    }
}

impl AddAssign for Energy {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0
    }
}

impl Zero for Energy {
    fn zero() -> Self {
        Self(0.)
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl Neg for Energy {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Energy(-self.0)
    }
}

impl Mul<Strength> for Energy {
    type Output = Energy;

    fn mul(self, rhs: Strength) -> Self::Output {
        Energy(self.0 * rhs.0)
    }
}

impl Energy {
    fn exp_times_u0(self) -> Conc {
        Conc(1e9 * self.0.exp())
    }
}

/// Rate in Hz
#[derive(PartialEq, PartialOrd, Debug, Clone, Copy)]
pub struct Rate(f64);

impl AddAssign for Rate {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0
    }
}

impl Add for Rate {
    type Output = Rate;

    fn add(self, rhs: Self) -> Self::Output {
        Rate(self.0 + rhs.0)
    }
}

impl SubAssign for Rate {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0
    }
}

#[derive(Clone, Debug)]
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

#[derive(Debug, Clone)]
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
    pub g_mc: Energy,
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
            self.monomer_detachment_rate_at_point(state, p).0
        } else {
            self.total_monomer_attachment_rate_at_point(state, p).0
        }
    }

    fn choose_event_at_point(&self, state: &S, p: PointSafe2, acc: crate::base::Rate) -> Event {
        println!("{:?}", acc);
        match self.choose_detachment_at_point(state, p, Rate(acc)) {
            (true, _, event) => {
                println!("{:?} {:?}", acc, event);
                event
            }
            (false, acc, _) => match self.choose_attachment_at_point(state, p, acc) {
                (true, _, event) => {
                    println!("{:?} {:?}", acc, event);
                    event
                }
                (false, acc, _) => {
                    panic!();
                }
            },
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
        todo!();
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
            g_se: Energy(9.),
            alpha: Energy(0.),
            g_mc: Energy(18.),
            kf: RatePerConc(1e-3),
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
                self.energy_ns[(t1, t2)] = self.g_se * self.glue_links[(t1r[2].0, t2r[0].0)];
                if t1r[2] == t2r[0] {
                    self.energy_ns[(t1, t2)] = self.g_se * self.glue_strengths[t1r[2].0]
                }
                self.energy_we[(t1, t2)] = self.g_se * self.glue_links[(t1r[1].0, t2r[3].0)];
                if t1r[1] == t2r[3] {
                    self.energy_we[(t1, t2)] = self.g_se * self.glue_strengths[t1r[1].0]
                }
            }
        }

        if (self.double_to_right.sum() > 0) || (self.double_to_bottom.sum() > 0) {
            self.has_duples = true;
            for (t1, t2) in self.double_to_right.indexed_iter() {
                self.double_to_left[*t2] = t1;
            }
            for (t1, t2) in self.double_to_bottom.indexed_iter() {
                self.double_to_top[*t2] = t1;
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
        self.friends_se.drain(..);
        for _ in 0..ntiles {
            self.friends_n.push(FnvHashSet::default());
            self.friends_e.push(FnvHashSet::default());
            self.friends_s.push(FnvHashSet::default());
            self.friends_w.push(FnvHashSet::default());
            self.friends_ne.push(FnvHashSet::default());
            self.friends_ee.push(FnvHashSet::default());
            self.friends_se.push(FnvHashSet::default());
            self.friends_ss.push(FnvHashSet::default());
            self.friends_se.push(FnvHashSet::default());
        }
        for t1 in 0..ntiles {
            for t2 in 0..ntiles {
                if self.energy_ns[(t1, t2)] != Energy(0.) {
                    self.friends_s[t1].insert(t2);
                    self.friends_n[t2].insert(t1);
                }
                if self.energy_we[(t1, t2)] != Energy(0.) {
                    self.friends_e[t1].insert(t2);
                    self.friends_w[t2].insert(t1);
                }
                match self.tile_shape(t1) {
                    TileShape::Single => (),
                    TileShape::DupleToRight(td) => {
                        if self.energy_ns[(t2, td)] != Energy(0.) {
                            self.friends_ne[td].insert(t2);
                        }
                        if self.energy_ns[(td, t2)] != Energy(0.) {
                            self.friends_ne[td].insert(t2);
                        }
                        if self.energy_we[(td, t2)] != Energy(0.) {
                            self.friends_ee[td].insert(t2);
                        }
                    }
                    TileShape::DupleToBottom(td) => {
                        if self.energy_we[(t2, td)] != Energy(0.) {
                            self.friends_sw[td].insert(t2);
                        }
                        if self.energy_we[(td, t2)] != Energy(0.) {
                            self.friends_se[td].insert(t2);
                        }
                        if self.energy_ns[(td, t2)] != Energy(0.) {
                            self.friends_ss[td].insert(t2);
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
        if self.is_seed(p) {
            return Rate(0.);
        }

        let t = state.tile_at_point(p);
        if t == 0 {
            return Rate(0.);
        }
        if (self.has_duples) && ((self.double_to_left[t] > 0) || (self.double_to_top[t] > 0)) {
            return Rate(0.);
        }
        self.kf * (-self.bond_energy_of_tile_type_at_point(state, p, t) + self.alpha).exp_times_u0()
    }

    pub fn choose_detachment_at_point(
        &self,
        state: &S,
        p: PointSafe2,
        mut acc: Rate,
    ) -> (bool, Rate, Event) {
        acc -= self.monomer_detachment_rate_at_point(state, p);
        if acc <= Rate(0.) {
            (true, acc, Event::MonomerDetachment(p))
        } else {
            (false, acc, Event::None)
        }
    }

    pub fn total_monomer_attachment_rate_at_point(&self, state: &S, p: PointSafe2) -> Rate {
        match self._find_monomer_attachment_possibilities_at_point(state, p, Rate(0.), true) {
            (false, acc, _) => Rate(-acc.0),
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
            friends.extend(&self.friends_s[tn]);
        }
        if te.nonzero() {
            friends.extend(&self.friends_w[te]);
        }
        if ts.nonzero() {
            friends.extend(&self.friends_n[ts]);
        }
        if tw.nonzero() {
            friends.extend(&self.friends_e[tw]);
        }

        for t in friends.drain() {
            acc -= self.kf * self.tile_concs[t];
            if !just_calc & (acc <= Rate(0.)) {
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
