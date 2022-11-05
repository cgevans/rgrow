use crate::{
    base::RgrowError,
    canvas::{Canvas, PointSafe2, PointSafeHere},
    simulation::Simulation,
    state::{self, State, StateCreate},
    system::{Event, System, SystemInfo, TileBondInfo},
    tileset::{FromTileSet, ProcessedTileSet, SimFromTileSet, Size, TileSet},
};

use crate::base::{HashMapType, HashSetType};
use ndarray::prelude::*;
use rand::{prelude::Distribution, rngs::SmallRng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, marker::PhantomData};

type Conc = f64;
type Glue = usize;
type Tile = usize;
type Strength = f64;
type Energy = f64;
type Rate = f64;

trait NonZero {
    fn nonzero(self) -> bool;
}

impl NonZero for Tile {
    fn nonzero(self) -> bool {
        self > 0
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Seed {
    None(),
    SingleTile { point: PointSafe2, tile: Tile },
    MultiTile(HashMapType<PointSafe2, Tile>),
}

enum TileShape {
    Single,
    DupleToRight(Tile),
    DupleToBottom(Tile),
    DupleToLeft(Tile),
    DupleToTop(Tile),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ATAM<C: Canvas> {
    /// Tile names, as strings.  Only used for reference.
    pub tile_names: Vec<String>,
    /// Tile concentrations, actual (not modified by alpha/Gse/etc) in nM.
    pub tile_stoics: Array1<Conc>,
    /// Glues (by number) on tile edges.
    pub tile_edges: Array2<Glue>,
    pub threshold: f64,
    /// Strengths of self-binding glues (eg, glue 1 binding to glue 1),
    /// in abstract strength.
    pub glue_strengths: Array1<Strength>,
    /// Strengths of links between different glues (eg, glue 1 binding to
    /// glue 2).  Should be symmetric.  Will be added with glue_strengths.
    pub glue_links: Array2<Strength>,
    pub double_to_right: Array1<Tile>,
    pub double_to_bottom: Array1<Tile>,
    pub seed: Seed,
    pub tile_colors: Vec<[u8; 4]>,

    // End of public stuff, now moving to calculated stuff.
    pub(crate) energy_ns: Array2<Energy>,
    pub(crate) energy_we: Array2<Energy>,

    /// Each "friends" hashset gives the potential tile attachments
    /// at point P if tile T is in that direction.  Eg, friends_e[T]
    /// is a set of tiles that might attach at point P if T is east of
    /// point P.  The ones other than NESW are only for duples.
    friends_n: Vec<HashSetType<Tile>>,
    friends_e: Vec<HashSetType<Tile>>,
    friends_s: Vec<HashSetType<Tile>>,
    friends_w: Vec<HashSetType<Tile>>,
    friends_ne: Vec<HashSetType<Tile>>,
    friends_ee: Vec<HashSetType<Tile>>,
    friends_se: Vec<HashSetType<Tile>>,
    friends_ss: Vec<HashSetType<Tile>>,
    friends_sw: Vec<HashSetType<Tile>>,

    has_duples: bool,
    double_to_left: Array1<Tile>,
    double_to_top: Array1<Tile>,
    should_be_counted: Array1<bool>,

    /// We need to store the type of canvas we're using so we know
    /// how to move around.
    _canvas: PhantomData<C>,
}

impl<S: State> System<S> for ATAM<S> {
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
            Event::PolymerDetachment(v) => {
                let mut points = Vec::new();
                for p in v {
                    points.extend(self.points_to_update_around(state, p));
                }
                points.sort_unstable();
                points.dedup();
                self.update_points(state, &points);
            }
            Event::PolymerAttachment(_) => todo!(),
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
            0.
        } else {
            self.total_monomer_attachment_rate_at_point(state, p)
        }
    }

    fn choose_event_at_point(&self, state: &S, p: PointSafe2, acc: crate::base::Rate) -> Event {
        match self.choose_attachment_at_point(state, p, acc) {
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
        }
    }

    fn set_safe_point(&self, state: &mut S, point: PointSafe2, tile: Tile) -> &Self {
        let event = Event::MonomerAttachment(point, tile);

        self.perform_event(state, &event)
            .update_after_event(state, &event);
        self
    }

    fn perform_event(&self, state: &mut S, event: &Event) -> &Self {
        match event {
            Event::None => panic!("Being asked to perform null event."),
            Event::MonomerAttachment(point, tile) => {
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
            Event::MonomerChange(point, tile) => {
                let oldt = state.tile_at_point(*point);

                match self.tile_shape(oldt) {
                    // Fixme: somewhat unsafe
                    TileShape::Single => (),
                    TileShape::DupleToRight(dt) => {
                        debug_assert_eq!(dt, state.tile_to_e(*point));
                        state.set_sa(&PointSafe2(state.move_sa_e(*point).0), &0usize)
                    }
                    TileShape::DupleToBottom(dt) => {
                        debug_assert_eq!(dt, state.tile_to_s(*point));
                        state.set_sa(&PointSafe2(state.move_sa_s(*point).0), &0usize)
                    }
                    TileShape::DupleToLeft(dt) => {
                        debug_assert_eq!(dt, state.tile_to_w(*point));
                        state.set_sa(&PointSafe2(state.move_sa_w(*point).0), &0usize)
                    }
                    TileShape::DupleToTop(dt) => {
                        debug_assert_eq!(dt, state.tile_to_n(*point));
                        state.set_sa(&PointSafe2(state.move_sa_n(*point).0), &0usize)
                    }
                }

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
        };
        state.add_events(1);
        self
    }

    fn seed_locs(&self) -> Vec<(PointSafe2, Tile)> {
        let mut v = Vec::new();

        match &self.seed {
            Seed::None() => {}
            Seed::SingleTile { point, tile } => {
                v.push((*point, *tile)); // FIXME
            }
            Seed::MultiTile(f) => {
                for (p, t) in f.iter() {
                    v.push((*p, *t));
                }
            }
        };
        v
    }

    fn calc_mismatch_locations(&self, _state: &S) -> Array2<usize> {
        todo!()
    }
}

impl<S: State> ATAM<S> {
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
        TileShape::Single
    }

    pub fn total_monomer_attachment_rate_at_point(&self, state: &S, p: PointSafe2) -> Rate {
        match self._find_monomer_attachment_possibilities_at_point(state, p, 0., true) {
            (false, acc, _) => -acc,
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

        let mut friends = HashSetType::<Tile>::default();

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
                TileShape::DupleToRight(_) => {
                    if state.tile_to_e(p) != 0 {
                        continue;
                    }
                }
                TileShape::DupleToBottom(_) => {
                    if state.tile_to_s(p) != 0 {
                        continue;
                    }
                }
                TileShape::DupleToLeft(_) => {
                    if state.tile_to_w(p) != 0 {
                        continue;
                    }
                }
                TileShape::DupleToTop(_) => {
                    if state.tile_to_n(p) != 0 {
                        continue;
                    }
                }
            }
            if self.bond_energy_of_tile_type_at_point_hypothetical(state, p, t) < self.threshold {
                continue;
            }
            acc -= self.tile_stoics[t];
            if !just_calc & (acc <= (0.)) {
                return (true, acc, Event::MonomerAttachment(p, t));
            }
        }
        (false, acc, Event::None)
    }

    pub fn bond_energy_of_tile_type_at_point_hypothetical(
        &self,
        state: &S,
        p: PointSafe2,
        t: Tile,
    ) -> Energy {
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
                debug_assert!((tright == te) | (te == 0));
                let tne = state.tile_to_ne(p);
                let tee = state.tile_to_ee(p);
                let tse = state.tile_to_se(p);
                energy += self.get_energy_ns(tne, tright)
                    + self.get_energy_we(tright, tee)
                    + self.get_energy_ns(tright, tse);
            }
            TileShape::DupleToBottom(tbottom) => {
                debug_assert!((tbottom == ts) | (ts == 0));
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

        energy
    }

    fn points_to_update_around(&self, state: &S, p: &PointSafe2) -> Vec<PointSafeHere> {
        // match self.chunk_size {
        // ChunkSize::Single => {
        let mut points = Vec::with_capacity(13);
        points.extend_from_slice(&[
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
        ]);
        points
        //     }
        //     ChunkSize::Dimer => {
        //         let mut points = Vec::with_capacity(10);
        //         points.extend_from_slice(&[
        //             state.move_sa_n(*p),
        //             state.move_sa_w(*p),
        //             PointSafeHere(p.0),
        //             state.move_sa_e(*p),
        //             state.move_sa_s(*p),
        //             state.move_sa_nw(*p),
        //             state.move_sa_ne(*p),
        //             state.move_sa_sw(*p),
        //         ]);

        //         let w = state.move_sa_w(*p);
        //         let n = state.move_sa_n(*p);

        //         if state.inbounds(w.0) {
        //             points.push(PointSafeHere(state.move_sh_w(w)));
        //         }
        //         if state.inbounds(n.0) {
        //             points.push(PointSafeHere(state.move_sh_n(n)));
        //         }
        //         points
        //     }
        // }
    }

    pub fn new_sized(ntiles: Tile, nglues: usize) -> Self {
        Self {
            tile_names: Vec::new(),
            tile_stoics: Array1::zeros(ntiles + 1),
            tile_edges: Array2::zeros((ntiles + 1, 4)),
            glue_strengths: Array1::zeros(nglues + 1),
            glue_links: Array2::zeros((nglues + 1, nglues + 1)),
            double_to_right: Array1::zeros(ntiles + 1),
            double_to_bottom: Array1::zeros(ntiles + 1),
            seed: Seed::None(),
            tile_colors: Vec::new(),
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
            should_be_counted: Array1::default(ntiles + 1),
            _canvas: PhantomData,
            threshold: 2.,
        }
    }

    pub fn from_atam(
        tile_stoics: Array1<f64>,
        tile_edges: Array2<Glue>,
        glue_strengths: Array1<f64>,
        threshold: f64,
        seed: Option<Seed>,
        tile_names: Option<Vec<String>>,
        tile_colors: Option<Vec<[u8; 4]>>,
    ) -> Self {
        let ntiles = tile_stoics.len() as Tile;

        let mut atam = Self::new_sized(tile_stoics.len() as Tile - 1, glue_strengths.len() - 1);

        atam.tile_stoics = tile_stoics;
        atam.tile_edges = tile_edges;
        atam.glue_strengths = glue_strengths;
        atam.seed = seed.unwrap_or(atam.seed);
        //ktam.tile_colors = tile_colors.unwrap_or(ktam.tile_colors);
        atam.tile_names = tile_names.unwrap_or(atam.tile_names);

        atam.tile_colors = match tile_colors {
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

        atam.threshold = threshold;

        atam.update_system();

        atam
    }

    pub fn update_system(&mut self) {
        let ntiles = self.tile_stoics.len();

        for t1 in 0..ntiles {
            for t2 in 0..ntiles {
                let t1r = self.tile_edges.row(t1);
                let t2r = self.tile_edges.row(t2);
                self.energy_ns[(t1, t2)] = self.glue_links[(t1r[2], t2r[0])];
                if t1r[2] == t2r[0] {
                    self.energy_ns[(t1, t2)] = self.glue_strengths[t1r[2]]
                }
                self.energy_we[(t1, t2)] = self.glue_links[(t1r[1], t2r[3])];
                if t1r[1] == t2r[3] {
                    self.energy_we[(t1, t2)] = self.glue_strengths[t1r[1]]
                }
            }
            if (t1 > 0) && (self.tile_stoics[t1] > 0.) {
                self.should_be_counted[t1] = true;
            } else {
                self.should_be_counted[t1] = false;
            }
        }

        if (self.double_to_right.sum() > 0) || (self.double_to_bottom.sum() > 0) {
            self.has_duples = true;
            for (t1, t2) in self.double_to_right.indexed_iter() {
                if (t1 > 0) & (t2 > &0) {
                    self.double_to_left[*t2] = t1;
                    self.energy_we[(t1, *t2)] = 0.0;
                    self.should_be_counted[*t2] = false;
                }
            }
            for (t1, t2) in self.double_to_bottom.indexed_iter() {
                if (t1 > 0) & (t2 > &0) {
                    self.double_to_top[*t2] = t1;
                    self.energy_ns[(t1, *t2)] = 0.0;
                    self.should_be_counted[*t2] = false;
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
            self.friends_n.push(HashSetType::default());
            self.friends_e.push(HashSetType::default());
            self.friends_s.push(HashSetType::default());
            self.friends_w.push(HashSetType::default());
            self.friends_ne.push(HashSetType::default());
            self.friends_ee.push(HashSetType::default());
            self.friends_se.push(HashSetType::default());
            self.friends_ss.push(HashSetType::default());
            self.friends_sw.push(HashSetType::default());
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

    pub fn set_duples(&mut self, hduples: Vec<(usize, usize)>, vduples: Vec<(usize, usize)>) {
        // Reset double_to_right and double_to_bottom to zeros
        self.double_to_right.fill(0);
        self.double_to_bottom.fill(0);

        // For each hduple, set the first index to the second value
        for (i, j) in hduples {
            self.double_to_right[i] = j;
        }

        // For each vduples, set the first index to the second value
        for (i, j) in vduples {
            self.double_to_bottom[i] = j;
        }

        self.update_system();
    }
}

impl<St: State + StateCreate> SimFromTileSet for ATAM<St> {
    fn sim_from_tileset(tileset: &TileSet) -> Result<Box<dyn Simulation>, RgrowError> {
        let sys = Self::from_tileset(tileset)?;
        let size = match tileset.options.size {
            Size::Single(x) => (x, x),
            Size::Pair((x, y)) => (x, y),
        };
        // let state = sys.new_state(size)?;
        let sim = crate::simulation::ConcreteSimulation {
            system: sys,
            states: vec![],
            rng: SmallRng::from_entropy(),
            default_state_size: size,
        };
        Ok(Box::new(sim))
    }
}

impl<C: State> TileBondInfo for ATAM<C> {
    fn tile_color(&self, tile_number: Tile) -> [u8; 4] {
        self.tile_colors[tile_number]
    }

    fn tile_name(&self, tile_number: Tile) -> &str {
        self.tile_names[tile_number].as_str()
    }

    fn bond_name(&self, _bond_number: usize) -> &str {
        todo!()
    }

    fn tile_colors(&self) -> &Vec<[u8; 4]> {
        &self.tile_colors
    }

    fn tile_names(&self) -> Vec<String> {
        self.tile_names.clone()
    }

    fn bond_names(&self) -> Vec<String> {
        todo!()
    }
}

impl<St: state::State + state::StateCreate> FromTileSet for ATAM<St> {
    fn from_tileset(tileset: &TileSet) -> Result<Self, RgrowError> {
        let proc = ProcessedTileSet::from_tileset(tileset)?;
        let seed = if proc.seed.is_empty() {
            Seed::None()
        } else if proc.seed.len() == 1 {
            let (x, y, v) = proc.seed[0];
            Seed::SingleTile {
                point: PointSafe2((x, y)),
                tile: v,
            }
        } else {
            let mut hm = HashMap::default();
            hm.extend(proc.seed.iter().map(|(y, x, v)| (PointSafe2((*y, *x)), *v)));
            Seed::MultiTile(hm)
        };
        let gluelinks = tileset
            .glues
            .iter()
            .map(|(g1, g2, s)| (proc.gpmap(g1), proc.gpmap(g2), *s))
            .collect::<Vec<_>>();

        let mut newkt = Self::from_atam(
            proc.tile_stoics,
            proc.tile_edges,
            proc.glue_strengths,
            tileset.options.threshold,
            Some(seed),
            Some(proc.tile_names),
            Some(proc.tile_colors),
        );

        newkt.set_duples(proc.hdoubletiles, proc.vdoubletiles);

        for (g1, g2, s) in gluelinks {
            newkt.glue_links[(g2, g1)] = s;
            newkt.glue_links[(g1, g2)] = s;
        }

        newkt.update_system();

        Ok(newkt)
    }
}

impl<C: State> SystemInfo for ATAM<C> {
    fn tile_concs(&self) -> Vec<f64> {
        todo!()
    }

    fn tile_stoics(&self) -> Vec<f64> {
        self.tile_stoics.clone().into_raw_vec()
    }
}
