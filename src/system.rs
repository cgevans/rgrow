use cached::{stores::SizedCache, Cached};
use fnv::FnvHashMap;
use fnv::FnvHashSet;
use ndarray::prelude::*;
use ndarray::{FoldWhile, Zip};

use super::base::{CanvasLength, Energy, Glue, Point, Rate, Tile};
use super::canvas::Canvas;

use super::fission;

use std::sync::{Arc, RwLock};



//type Cache = FnvHashMap<(Tile, Tile, Tile, Tile), f64>;
type Cache = SizedCache<(Tile, Tile, Tile, Tile), f64>;

pub enum Orientation {
    NS,
    WE,
}
pub struct DimerInfo {
    pub t1: Tile,
    pub t2: Tile,
    pub orientation: Orientation,
    pub formation_rate: Rate,
    pub equilibrium_conc: f64,
}

pub trait System<C: Canvas> {
    /// Returns the total event rate at a given point.  These should correspond with the events chosen by `choose_event_at_point`.
    fn event_rate_at_point(&self, canvas: &C, p: Point) -> Rate;

    /// Given a point, and an accumulated random rate choice `acc` (which should be less than the total rate at the point),
    /// return the tile that point should change to.  TODO: we should support more general events than just a single tile change.
    fn choose_event_at_point(&self, canvas: &C, p: Point, acc: Rate) -> Tile;

    /// Returns a vector of (point, tile number) tuples for the seed tiles, useful for populating an initial state.
    fn seed_locs(&self) -> Vec<(Point, Tile)>;

    /// Returns information on dimers that the system can form, similarly useful for starting out a state.
    fn calc_dimers(&self) -> Vec<DimerInfo>;
}

#[derive(Clone, Debug)]
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
pub struct StaticKTAM {
    pub tile_adj_concs: Array1<Rate>,
    pub energy_ns: Array2<Energy>,
    pub energy_we: Array2<Energy>,
    friends_n: Vec<FnvHashSet<Tile>>,
    friends_e: Vec<FnvHashSet<Tile>>,
    friends_s: Vec<FnvHashSet<Tile>>,
    friends_w: Vec<FnvHashSet<Tile>>,
    insertcache: Arc<RwLock<Cache>>,
    seed: Seed,
    k_f: f64,
    alpha: f64,
    g_se: Option<f64>,
    g_mc: Option<f64>,
}

impl StaticATAM {
    pub fn new(
        tile_concs: Array1<f64>,
        tile_edges: Array2<Glue>,
        glue_strengths: Array1<f64>,
        tau: f64,
        seed: Option<Seed>,
    ) -> Self {
        let ntiles = tile_concs.len();
        assert!(ntiles == tile_edges.nrows());
        let mut strength_we: Array2<Energy> = Array2::zeros((ntiles, ntiles));
        let mut strength_ns: Array2<Energy> = Array2::zeros((ntiles, ntiles));
        for ti1 in 0..ntiles {
            for ti2 in 0..ntiles {
                let t1 = tile_edges.row(ti1);
                let t2 = tile_edges.row(ti2);
                if t1[2] == t2[0] {
                    strength_ns[(ti1, ti2)] = glue_strengths[t1[2] as usize];
                }
                if t1[1] == t2[3] {
                    strength_we[(ti1, ti2)] = glue_strengths[t1[1] as usize];
                }
            }
        }
        return StaticATAM {
            tile_rates: tile_concs,
            strength_ns,
            strength_we,
            tau,
            seed: seed.unwrap_or(Seed::None()),
        };
    }
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

impl StaticKTAM {
    pub fn from_ktam(
        tile_stoics: Array1<f64>,
        tile_edges: Array2<Glue>,
        glue_strengths: Array1<f64>,
        g_se: f64,
        g_mc: f64,
        alpha: Option<f64>,
        k_f: Option<f64>,
        seed: Option<Seed>,
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
            insertcache: Arc::new(RwLock::new(Cache::with_size(10000))),
            seed: seed.unwrap_or(Seed::None()),
            alpha: alpha.unwrap_or(0.),
            g_mc: Some(g_mc),
            g_se: Some(g_se),
            k_f: k_f.unwrap_or(1e6),
        };
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
    ) -> Self {
        let (friends_n, friends_e, friends_s, friends_w) =
            create_friend_data(&energy_ns, &energy_we);

        StaticKTAM {
            tile_adj_concs,
            energy_ns,
            energy_we,
            friends_n,
            friends_e,
            friends_s,
            friends_w,
            insertcache: Arc::new(RwLock::new(Cache::with_size(10000))),
            seed: Seed::None(),
            alpha: alpha,
            g_mc: None,
            g_se: None,
            k_f: k_f,
        }
    }
}

impl<C> System<C> for StaticATAM
where
    C: Canvas,
{
    #[inline]
    fn event_rate_at_point(&self, canvas: &C, p: Point) -> Rate {
        if !canvas.inbounds(p) {
            return 0.0;
        }

        // Bound is previously checked.

        let tw = unsafe { canvas.uv_w(p) };
        let tile = unsafe { canvas.uv_p(p) };
        let te = unsafe { canvas.uv_e(p) };
        let tn = unsafe { canvas.uv_n(p) };
        let ts = unsafe { canvas.uv_s(p) };

        if tile != 0 {
            // Deletion
            0.0
        } else {
            // Insertion

            Zip::from(self.strength_ns.row(tn as usize))
                .and(self.strength_we.column(te as usize))
                .and(self.strength_we.row(tw as usize))
                .and(self.strength_ns.column(ts as usize))
                .and(&self.tile_rates)
                .fold(0., |acc, &n, &e, &s, &w, &r| {
                    if n + e + s + w >= self.tau {
                        acc + r
                    } else {
                        acc
                    }
                })
        }
    }

    #[inline]
    fn choose_event_at_point(&self, canvas: &C, p: Point, acc: Rate) -> Tile {
        if !canvas.inbounds(p) {
            panic!("Out of bounds point in choose_event_at_point: {:?}", p);
        }

        // Bound is previously checked.
        let tile = unsafe { canvas.uv_p(p) };

        if tile != 0 {
            // Deletion is easy!
            panic!("We shouldn't be deleting in the aTAM!")
        } else {
            let tw = unsafe { canvas.uv_w(p) };
            let te = unsafe { canvas.uv_e(p) };
            let tn = unsafe { canvas.uv_n(p) };
            let ts = unsafe { canvas.uv_s(p) };

            // Insertion is hard!
            let r = Zip::indexed(self.strength_ns.row(tn as usize))
                .and(self.strength_we.column(te as usize))
                .and(self.strength_ns.column(ts as usize))
                .and(self.strength_we.row(tw as usize))
                .and(&self.tile_rates)
                .fold_while((acc, 0), |(acc, _v), i, &n, &e, &s, &w, &r| {
                    if n + e + s + w >= self.tau {
                        if acc - r > 0. {
                            FoldWhile::Continue((acc - r, 0))
                        } else {
                            FoldWhile::Done((acc - r, i))
                        }
                    } else {
                        FoldWhile::Continue((acc, 0))
                    }
                });

            match r {
                FoldWhile::Done((_acc, i)) => i as Tile,

                FoldWhile::Continue((_acc, _i)) => panic!(
                    "Reached end of insertion possibilities, but still have {:?} rate remaining.",
                    _acc
                ),
            }
        }
    }

    fn seed_locs(&self) -> Vec<(Point, Tile)> {
        let mut v = Vec::new();

        match &self.seed {
            Seed::None() => {}
            Seed::SingleTile { point, tile } => {
                v.push((*point, *tile));
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
        todo!()
    }
}

impl<C> System<C> for StaticKTAM
where
    C: Canvas,
{
    fn event_rate_at_point(&self, canvas: &C, p: Point) -> Rate {
        if !canvas.inbounds(p) {
            return 0.0;
        }

        // Bound is previously checked.

        let tn = unsafe { canvas.uv_n(p) };
        let tw = unsafe { canvas.uv_w(p) };
        let tile = unsafe { canvas.uv_p(p) };
        let te = unsafe { canvas.uv_e(p) };
        let ts = unsafe { canvas.uv_s(p) };

        if tile != 0 {
            // Deletion

            // Check seed
            match &self.seed {
                Seed::None() => {}
                Seed::SingleTile { point, tile: _ } => {
                    if p == *point {
                        return 0.0;
                    }
                }
                Seed::MultiTile(map) => {
                    if map.contains_key(&p) {
                        return 0.0;
                    }
                }
            }

            // Bound is previously checked
            let bound_energy: Energy = self.energy_ns[(tile as usize, ts as usize)]
                + self.energy_ns[(tn as usize, tile as usize)]
                + self.energy_we[(tile as usize, te as usize)]
                + self.energy_we[(tw as usize, tile as usize)];

            Rate::exp(-bound_energy)
        } else if (tn == 0) & (te == 0) & (tw == 0) & (ts == 0) {
            // Short circuit for no possibility of insertion (no adjacents)
            0.0
        } else {
            // Insertion

            let mut ic = self.insertcache.write().unwrap();

            match ic.cache_get(&(tn, te, ts, tw)) {
                Some(acc) => *acc,

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
                        .write()
                        .unwrap()
                        .cache_set((tn, te, ts, tw), acc);

                    acc
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

    fn choose_event_at_point(&self, canvas: &C, p: Point, mut acc: Rate) -> Tile {
        if !canvas.inbounds(p) {
            panic!("Out of bounds point in choose_event_at_point: {:?}", p);
        }

        // Bound is previously checked.
        let tile = unsafe { canvas.uv_p(p) };

        let tn = unsafe { canvas.uv_n(p) as usize };
        let tw = unsafe { canvas.uv_w(p) as usize };
        let te = unsafe { canvas.uv_e(p) as usize };
        let ts = unsafe { canvas.uv_s(p) as usize };

        if tile != 0 {
            // Deletion is easy! But first, check if we might have a fission problem.
            match self.determine_fission(canvas, &[canvas.move_point_n(p), canvas.move_point_e(p), canvas.move_point_s(p), canvas.move_point_w(p)], &[p]) {
                fission::FissionResult::NoFission => {0}
                // Disallow fission for now! FIXME FIXME
                fission::FissionResult::FissionGroups(m, g) => {println!("Actual fission, {:?}", g); 0 }
            }
            
        } else {
            let mut friends = FnvHashSet::<Tile>::default();

            friends.extend(&self.friends_s[tn as usize]);
            friends.extend(&self.friends_w[te as usize]);
            friends.extend(&self.friends_e[tw as usize]);
            friends.extend(&self.friends_n[ts as usize]);

            for t in friends.drain() {
                acc -= self.tile_adj_concs[t as usize];
                if acc <= 0. {
                    return t;
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

    fn seed_locs(&self) -> Vec<(Point, Tile)> {
        let mut v = Vec::new();

        match &self.seed {
            Seed::None() => {}
            Seed::SingleTile { point, tile } => {
                v.push((*point, *tile));
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
        let mut dvec = Vec::new();

        for ((t1, t2), e) in self.energy_ns.indexed_iter() {
            if *e != 0. {
                let biconc = self.tile_adj_concs[t1] * self.tile_adj_concs[t2];
                dvec.push(DimerInfo {
                    t1: t1 as Tile,
                    t2: t2 as Tile,
                    orientation: Orientation::NS,
                    formation_rate: self.k_f_hat() * biconc,
                    equilibrium_conc: biconc * f64::exp(-*e + 2. * self.alpha),
                });
            }
        }

        for ((t1, t2), e) in self.energy_we.indexed_iter() {
            if *e != 0. {
                let biconc = self.tile_adj_concs[t1] * self.tile_adj_concs[t2];
                dvec.push(DimerInfo {
                    t1: t1 as Tile,
                    t2: t2 as Tile,
                    orientation: Orientation::WE,
                    formation_rate: self.k_f_hat() * biconc,
                    equilibrium_conc: biconc * f64::exp(-*e + 2. * self.alpha),
                });
            }
        }

        dvec
    }
}
