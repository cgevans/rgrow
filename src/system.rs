use cached::{stores::SizedCache, Cached};
use fnv::FnvHashMap;
use fnv::FnvHashSet;
use ndarray::prelude::*;
use ndarray::{FoldWhile, Zip};
use rand::prelude::Distribution;
use serde::{Deserialize, Serialize};

use super::base::{Energy, Glue, Point, Rate, Tile};
use super::canvas::Canvas;

use super::fission;

use std::{
    marker::PhantomData,
    sync::{Arc, RwLock},
};


use dashmap::DashMap;

//type Cache = FnvHashMap<(Tile, Tile, Tile, Tile), f64>;
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
#[derive(Clone, Debug)]
pub enum Event {
    None,
    SingleTileAttach(Tile),
    SingleTileDetach,
    SingleTileChange(Tile),
    MultiTileAttach(Vec<(Point, Tile)>),
    MultiTileDetach(Vec<Point>),
}
#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub enum ChunkHandling {
    #[serde(alias="none")]
    None,
    #[serde(alias="detach")]
    Detach,
    #[serde(alias="equilibrium")]
    Equilibrium,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub enum ChunkSize {
    #[serde(alias="single")]
    Single,
    #[serde(alias="dimer")]
    Dimer,
}

pub enum Updates {
    Plus,
    DimerChunk
}

pub trait System<C: Canvas> {
    /// Returns the total event rate at a given point.  These should correspond with the events chosen by `choose_event_at_point`.
    fn event_rate_at_point(&self, canvas: &C, p: Point) -> Rate;

    /// Given a point, and an accumulated random rate choice `acc` (which should be less than the total rate at the point),
    /// return the event that should take place.
    fn choose_event_at_point(&self, canvas: &C, p: Point, acc: Rate) -> Event;

    /// Returns a vector of (point, tile number) tuples for the seed tiles, useful for populating an initial state.
    fn seed_locs(&self) -> Vec<(Point, Tile)>;

    /// Returns information on dimers that the system can form, similarly useful for starting out a state.
    fn calc_dimers(&self) -> Vec<DimerInfo>;

    fn updates_around_point(&self) -> Updates;
}

pub trait TileBondInfo {
    fn tile_color(&self, tile_number: Tile) -> [u8; 4];
    fn tile_name(&self, tile_number: Tile) -> &str;
    fn bond_name(&self, bond_number: usize) -> &str;
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
#[derive(Debug)]
pub struct StaticKTAM<C: Canvas> {
    pub tile_adj_concs: Array1<Rate>,
    pub energy_ns: Array2<Energy>,
    pub energy_we: Array2<Energy>,
    friends_n: Vec<FnvHashSet<Tile>>,
    friends_e: Vec<FnvHashSet<Tile>>,
    friends_s: Vec<FnvHashSet<Tile>>,
    friends_w: Vec<FnvHashSet<Tile>>,
    insertcache: RwLock<Cache>,
    seed: Seed,
    k_f: f64,
    alpha: f64,
    g_se: Option<f64>,
    g_mc: Option<f64>,
    fission_handling: FissionHandling,
    chunk_handling: ChunkHandling,
    chunk_size: ChunkSize,
    tile_names: Vec<String>,
    tile_colors: Vec<[u8; 4]>,
    _canvas: PhantomData<C>,
}

impl<'a, C: Canvas> TileBondInfo for StaticKTAM<C> {
    fn tile_color(&self, tile_number: Tile) -> [u8; 4] {
        self.tile_colors[tile_number as usize]
    }

    fn tile_name(&self, tile_number: Tile) -> &str {
        self.tile_names[tile_number as usize].as_str()
    }

    fn bond_name(&self, _bond_number: usize) -> &str {
        todo!()
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

impl<C: Canvas> StaticKTAM<C> {
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
            insertcache: RwLock::new(Cache::with_size(10000)),
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
            insertcache: RwLock::new(Cache::with_size(10000)),
            seed: Seed::None(),
            alpha: alpha,
            g_mc: None,
            g_se: None,
            k_f: k_f,
            fission_handling: fission_handling.unwrap_or(FissionHandling::NoFission),
            tile_names,
            tile_colors,
            chunk_handling: ChunkHandling::None,
            chunk_size: ChunkSize::Single,
            _canvas: PhantomData,
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
    fn choose_event_at_point(&self, canvas: &C, p: Point, acc: Rate) -> Event {
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
                FoldWhile::Done((_acc, i)) => Event::SingleTileAttach(i as Tile),

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

    fn updates_around_point(&self) -> Updates {
        todo!()
    }
}

impl<C: Canvas> StaticKTAM<C> {
    /// Unsafe because does not check bounds of p: assumes inbounds (with border if applicable).
    /// This requires the tile to be specified because it is likely you've already accessed it.
    unsafe fn u_bond_strength_of_tile_at_point(&self, canvas: &C, p: Point, tile: Tile) -> Energy {
        let tn = { canvas.uv_n(p) };
        let tw = { canvas.uv_w(p) };
        let te = { canvas.uv_e(p) };
        let ts = { canvas.uv_s(p) };

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
            unsafe {
                Rate::exp(
                    -ts - self.u_bond_strength_of_tile_at_point(canvas, p2, t2)
                        + 2.*self.energy_ns[(t as usize, t2 as usize)],
                )
            }
        }
    }

    // Dimer detachment rates are written manually.
    fn dimer_e_detach_rate(&self, canvas: &C, p: Point, t: Tile, ts: Energy) -> Rate {
        let p2 = canvas.u_move_point_e(p);
        if (!canvas.inbounds(p2)) | (unsafe { canvas.uv_p(p2) == 0 } | self.is_seed(p2) ) {
            0.0
        } else {
            let t2 = unsafe { canvas.uv_p(p2) };
            unsafe {
                Rate::exp(
                    -ts - self.u_bond_strength_of_tile_at_point(canvas, p2, t2)
                        + 2.*self.energy_we[(t as usize, t2 as usize)],
                )
            }
        }
    }

    fn chunk_detach_rate(&self, canvas: &C, p: Point, t: Tile) -> Rate {
        match self.chunk_size {
            ChunkSize::Single => 0.0,
            ChunkSize::Dimer => {
                let ts = unsafe { self.u_bond_strength_of_tile_at_point(canvas, p, t) };
                self.dimer_s_detach_rate(canvas, p, t, ts)
                    + self.dimer_e_detach_rate(canvas, p, t, ts)
            }
        }
    }

    fn choose_chunk_detachment(
        &self,
        canvas: &C,
        p: Point,
        tile: usize,
        acc: &mut Rate,
        now_empty: &mut Vec<Point>,
        possible_starts: &mut Vec<Point>,
    ) {
        match self.chunk_size {
            ChunkSize::Single => { panic!() }
            ChunkSize::Dimer => { 
                let ts = unsafe { self.u_bond_strength_of_tile_at_point(canvas, p, tile as u32) };
                *acc -= self.dimer_s_detach_rate(canvas, p, tile as u32, ts);
                if *acc <= 0. {
                    let p2 = {canvas.u_move_point_s(p)};
                    let t2 = unsafe {canvas.uv_p(p2)} as usize;
                    now_empty.push(p);
                    now_empty.push(p2);
                    // North tile adjacents
                    if self.energy_ns[(unsafe {canvas.uv_n(p)} as usize, tile)] > 0. {
                        possible_starts.push(canvas.u_move_point_n(p))
                    };
                    if self.energy_we[(unsafe {canvas.uv_w(p)} as usize, tile)] > 0. {
                        possible_starts.push(canvas.u_move_point_w(p))
                    };
                    if self.energy_we[(tile, unsafe {canvas.uv_e(p)} as usize)] > 0. {
                        possible_starts.push(canvas.u_move_point_e(p))
                    };
                    // South tile adjacents
                    if self.energy_ns[(t2, unsafe {canvas.uv_s(p2)} as usize)] > 0. {
                        possible_starts.push(canvas.u_move_point_s(p2))
                    };
                    if self.energy_we[(unsafe {canvas.uv_w(p2)} as usize, t2)] > 0. {
                        possible_starts.push(canvas.u_move_point_w(p2))
                    };
                    if self.energy_we[(t2, unsafe {canvas.uv_e(p2)} as usize)] > 0. {
                        possible_starts.push(canvas.u_move_point_e(p2))
                    };
                    return ()
                }
                *acc -= self.dimer_e_detach_rate(canvas, p, tile as u32, ts);
                if *acc <= 0. {
                    let p2 =  {canvas.u_move_point_e(p)};
                    let t2 = unsafe {canvas.uv_p(p2)} as usize;
                    now_empty.push(p);
                    now_empty.push(p2);
                    // West tile adjacents
                    if self.energy_we[(unsafe {canvas.uv_w(p)} as usize, tile)] > 0. {
                        possible_starts.push(canvas.u_move_point_w(p))
                    };
                    if self.energy_ns[(unsafe {canvas.uv_n(p)} as usize, tile)] > 0. {
                        possible_starts.push(canvas.u_move_point_n(p))
                    };
                    if self.energy_ns[(tile, unsafe {canvas.uv_s(p)} as usize)] > 0. {
                        possible_starts.push(canvas.u_move_point_s(p))
                    };
                    // East tile adjacents
                    if self.energy_we[(t2, unsafe {canvas.uv_e(p2)} as usize)] > 0. {
                        possible_starts.push(canvas.u_move_point_e(p2))
                    };
                    if self.energy_ns[(unsafe {canvas.uv_n(p2)} as usize, t2)] > 0. {
                        possible_starts.push(canvas.u_move_point_n(p2))
                    };
                    if self.energy_ns[(t2, unsafe {canvas.uv_s(p2)} as usize)] > 0. {
                        possible_starts.push(canvas.u_move_point_s(p2))
                    };
                    return ()
                }
                panic!("{:#?}", acc)
             }
        }
    }
}

impl<C> System<C> for StaticKTAM<C>
where
    C: Canvas,
{
    fn event_rate_at_point(&self, canvas: &C, p: Point) -> Rate {
        if !canvas.inbounds(p) {
            return 0.0;
        }

        // Bound is previously checked.

        let tile = unsafe { canvas.uv_p(p) };

        if tile != 0 {
            // Deletion

            // Check seed
            if self.is_seed(p) {
                return 0.0;
            }

            // Bound is previously checked
            let bound_energy = unsafe { self.u_bond_strength_of_tile_at_point(canvas, p, tile) };

            match self.chunk_handling {
                ChunkHandling::None => Rate::exp(-bound_energy),
                ChunkHandling::Detach | ChunkHandling::Equilibrium => {
                    Rate::exp(-bound_energy) + self.chunk_detach_rate(canvas, p, tile)
                }
            }
        } else {
            let tn = unsafe { canvas.uv_n(p) };
            let tw = unsafe { canvas.uv_w(p) };
            let te = unsafe { canvas.uv_e(p) };
            let ts = unsafe { canvas.uv_s(p) };

            // Short circuit if no adjacent tiles.
            if (tn == 0) & (tw == 0) & (te == 0) & (ts == 0) {
                return 0.0;
            }

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

    fn choose_event_at_point(&self, canvas: &C, p: Point, mut acc: Rate) -> Event {
        if !canvas.inbounds(p) {
            panic!("Out of bounds point in choose_event_at_point: {:?}", p);
        }

        // Bound is previously checked.
        let tile = unsafe { canvas.uv_p(p) as usize };

        let tn = unsafe { canvas.uv_n(p) as usize };
        let tw = unsafe { canvas.uv_w(p) as usize };
        let te = unsafe { canvas.uv_e(p) as usize };
        let ts = unsafe { canvas.uv_s(p) as usize };

        if tile != 0 {
            acc -= 
                unsafe {
                    Rate::exp(-self.u_bond_strength_of_tile_at_point(canvas, p, tile as u32))
                };

            let mut possible_starts = Vec::new();
            let mut now_empty = Vec::new();

            if acc <= 0. {
                if self.energy_ns[(tn, tile)] > 0. {
                    possible_starts.push(canvas.u_move_point_n(p))
                };
                if self.energy_we[(tw, tile)] > 0. {
                    possible_starts.push(canvas.u_move_point_w(p))
                };
                if self.energy_ns[(tile, ts)] > 0. {
                    possible_starts.push(canvas.u_move_point_s(p))
                };
                if self.energy_we[(tile, te)] > 0. {
                    possible_starts.push(canvas.u_move_point_e(p))
                };

                now_empty.push(p);

                match self.determine_fission(canvas, &possible_starts, &now_empty) {
                    fission::FissionResult::NoFission => Event::SingleTileDetach,
                    fission::FissionResult::FissionGroups(g) => {
                        //println!("Fission handling {:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?}", p, tile, possible_starts, now_empty, tn, te, ts, tw, canvas.calc_ntiles(), g.map.len());
                        match self.fission_handling {
                            FissionHandling::NoFission => Event::None,
                            FissionHandling::JustDetach => Event::SingleTileDetach,
                            FissionHandling::KeepSeeded => {
                                let sl = System::<C>::seed_locs(self);
                                Event::MultiTileDetach(g.choose_deletions_seed_unattached(sl))
                            }
                            FissionHandling::KeepLargest => {
                                Event::MultiTileDetach(g.choose_deletions_keep_largest_group())
                            }
                            FissionHandling::KeepWeighted => {
                                Event::MultiTileDetach(g.choose_deletions_size_weighted())
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
                    fission::FissionResult::NoFission => Event::MultiTileDetach(now_empty),
                    fission::FissionResult::FissionGroups(g) => {
                        //println!("Fission handling {:?} {:?}", p, tile);
                        match self.fission_handling {
                            FissionHandling::NoFission => Event::None,
                            FissionHandling::JustDetach => Event::MultiTileDetach(now_empty),
                            FissionHandling::KeepSeeded => {
                                let sl = System::<C>::seed_locs(self);
                                Event::MultiTileDetach(g.choose_deletions_seed_unattached(sl))
                            }
                            FissionHandling::KeepLargest => {
                                Event::MultiTileDetach(g.choose_deletions_keep_largest_group())
                            }
                            FissionHandling::KeepWeighted => {
                                Event::MultiTileDetach(g.choose_deletions_size_weighted())
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
                acc -= self.tile_adj_concs[t as usize];
                if acc <= 0. {
                    return Event::SingleTileAttach(t);
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

    fn updates_around_point(&self) -> Updates {
        match self.chunk_size {
            ChunkSize::Single => { Updates::Plus }
            ChunkSize::Dimer => { Updates::DimerChunk }
        }
    }
}
