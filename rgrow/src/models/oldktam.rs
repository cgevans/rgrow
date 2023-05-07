use std::{collections::HashMap, sync::RwLock};

use crate::base::{HashMapType, HashSetType};
use cached::{Cached, SizedCache};
use fnv::FnvHashMap;
use ndarray::{Array1, Array2};
use rand::prelude::Distribution;
use serde::{Deserialize, Serialize};

use crate::{
    base::{Energy, Glue, ModelError, Point, Rate, RgrowError, Tile},
    canvas::{PointSafe2, PointSafeHere},
    state::State,
    system::{
        ChunkHandling, ChunkSize, DimerInfo, Event, FissionHandling, Orientation, System,
        SystemInfo, SystemWithDimers, TileBondInfo,
    },
    tileset::{FromTileSet, ProcessedTileSet, TileSet},
};

type Cache = SizedCache<(Tile, Tile, Tile, Tile), f64>;

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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Seed {
    None(),
    SingleTile { point: Point, tile: Tile },
    MultiTile(HashMapType<Point, Tile>),
}

type TileHashSetVec = Vec<HashSetType<Tile>>;

fn create_friend_data(
    energy_ns: &Array2<Energy>,
    energy_we: &Array2<Energy>,
) -> (
    TileHashSetVec,
    TileHashSetVec,
    TileHashSetVec,
    TileHashSetVec,
) {
    let mut friends_n = Vec::<HashSetType<Tile>>::new();
    let mut friends_e = Vec::<HashSetType<Tile>>::new();
    let mut friends_s = Vec::<HashSetType<Tile>>::new();
    let mut friends_w = Vec::<HashSetType<Tile>>::new();

    for _t1 in 0..energy_ns.nrows() {
        friends_n.push(HashSetType::default());
        friends_e.push(HashSetType::default());
        friends_s.push(HashSetType::default());
        friends_w.push(HashSetType::default());
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OldKTAM {
    pub tile_adj_concs: Array1<Rate>,
    pub energy_ns: Array2<Energy>,
    pub energy_we: Array2<Energy>,
    friends_n: TileHashSetVec,
    friends_e: TileHashSetVec,
    friends_s: TileHashSetVec,
    friends_w: TileHashSetVec,
    #[serde(skip)]
    insertcache: ClonableCache,
    seed: Seed,
    pub k_f: f64,
    pub alpha: f64,
    pub g_se: Option<f64>,
    pub g_mc: Option<f64>,
    fission_handling: FissionHandling,
    chunk_handling: ChunkHandling,
    pub(crate) chunk_size: ChunkSize,
    pub(crate) tile_names: Vec<String>,
    pub(crate) tile_colors: Vec<[u8; 4]>,
}

impl TileBondInfo for OldKTAM {
    fn tile_color(&self, tile_number: Tile) -> [u8; 4] {
        self.tile_colors[tile_number as usize]
    }

    fn tile_name(&self, tile_number: Tile) -> &str {
        self.tile_names[tile_number as usize].as_str()
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

#[allow(clippy::too_many_arguments)]
impl OldKTAM {
    pub fn from_ktam(
        tile_stoics: Array1<f64>,
        tile_edges: Array2<Glue>,
        glue_strengths: Array1<f64>,
        glue_links: Vec<(Glue, Glue, f64)>,
        g_se: f64,
        g_mc: f64,
        alpha: Option<f64>,
        k_f: Option<f64>,
        seed: Option<Seed>,
        fission_handling: Option<FissionHandling>,
        chunk_handling: Option<ChunkHandling>,
        chunk_size: Option<ChunkSize>,
        tile_names: Vec<String>,
        tile_colors: Vec<[u8; 4]>,
    ) -> Self {
        let ntiles = tile_stoics.len();
        assert!(ntiles == tile_edges.nrows());

        let mut energy_we: Array2<Energy> = Array2::zeros((ntiles, ntiles));
        let mut energy_ns: Array2<Energy> = Array2::zeros((ntiles, ntiles));

        let mut gsmap = FnvHashMap::<(Glue, Glue), f64>::default();
        for (g1, g2, v) in glue_links {
            gsmap.insert((g1, g2), v);
            gsmap.insert((g2, g1), v);
        }

        for ti1 in 0..ntiles {
            for ti2 in 0..ntiles {
                let t1 = tile_edges.row(ti1);
                let t2 = tile_edges.row(ti2);
                if t1[2] == t2[0] {
                    energy_ns[(ti1, ti2)] = g_se * glue_strengths[t1[2]];
                }
                if t1[1] == t2[3] {
                    energy_we[(ti1, ti2)] = g_se * glue_strengths[t1[1]];
                }
                if let Some(v) = gsmap.get(&(t1[2], t2[0])) {
                    energy_ns[(ti1, ti2)] += g_se * (*v);
                }
                if let Some(v) = gsmap.get(&(t1[1], t2[3])) {
                    energy_we[(ti1, ti2)] += g_se * (*v);
                }
            }
        }

        let (friends_n, friends_e, friends_s, friends_w) =
            create_friend_data(&energy_ns, &energy_we);
        OldKTAM {
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
            tile_names,
            tile_colors,
            chunk_handling: chunk_handling.unwrap_or(ChunkHandling::None),
            chunk_size: chunk_size.unwrap_or(ChunkSize::Single),
        }
    }

    pub(crate) fn points_to_update_around<C: State>(
        &self,
        state: &C,
        p: &PointSafe2,
    ) -> Vec<PointSafeHere> {
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

    pub(crate) fn k_f_hat(&self) -> f64 {
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

        let tile_names = (0..ntiles).map(|x| x.to_string()).collect();

        let tile_colors = {
            let mut rng = rand::thread_rng();
            let ug = rand::distributions::Uniform::new(100u8, 254);
            (0..ntiles)
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

        OldKTAM {
            tile_adj_concs,
            energy_ns,
            energy_we,
            friends_n,
            friends_e,
            friends_s,
            friends_w,
            insertcache: ClonableCache(RwLock::new(Cache::with_size(10000))),
            seed: Seed::None(),
            alpha,
            g_mc: None,
            g_se: None,
            k_f,
            fission_handling: fission_handling.unwrap_or(FissionHandling::NoFission),
            tile_names,
            tile_colors,
            chunk_handling: chunk_handling.unwrap_or(ChunkHandling::None),
            chunk_size: chunk_size.unwrap_or(ChunkSize::Single),
        }
    }

    pub(crate) fn get_energy_ns(&self, t1: Tile, t2: Tile) -> f64 {
        self.energy_ns[(t1 as usize, t2 as usize)]
    }

    pub(crate) fn get_energy_we(&self, t1: Tile, t2: Tile) -> f64 {
        self.energy_we[(t1 as usize, t2 as usize)]
    }

    /// Unsafe because does not check bounds of p: assumes inbounds (with border if applicable).
    /// This requires the tile to be specified because it is likely you've already accessed it.
    pub(crate) fn bond_strength_of_tile_at_point<C: State>(
        &self,
        canvas: &C,
        p: PointSafe2,
        tile: Tile,
    ) -> Energy {
        let tn = { canvas.tile_to_n(p) };
        let tw = { canvas.tile_to_w(p) };
        let te = { canvas.tile_to_e(p) };
        let ts = { canvas.tile_to_s(p) };

        self.get_energy_ns(tile, ts)
            + self.get_energy_ns(tn, tile)
            + self.get_energy_we(tile, te)
            + self.get_energy_we(tw, tile)
    }

    fn is_seed(&self, p: Point) -> bool {
        match &self.seed {
            Seed::None() => false,
            Seed::SingleTile { point, tile: _ } => p == *point,
            Seed::MultiTile(map) => map.contains_key(&p),
        }
    }

    // Dimer detachment rates are written manually.
    fn dimer_s_detach_rate<C: State>(&self, canvas: &C, p: Point, t: Tile, ts: Energy) -> Rate {
        let p2 = canvas.u_move_point_s(p);
        if (!canvas.inbounds(p2)) | (unsafe { canvas.uv_p(p2) == 0 }) | self.is_seed(p2) {
            0.0
        } else {
            let t2 = unsafe { canvas.uv_p(p2) };
            {
                self.k_f_hat()
                    * Rate::exp(
                        -ts - self.bond_strength_of_tile_at_point(canvas, PointSafe2(p2), t2) // FIXME
                        + 2. * self.get_energy_ns(t, t2),
                    )
            }
        }
    }

    // Dimer detachment rates are written manually.
    fn dimer_e_detach_rate<C: State>(&self, canvas: &C, p: Point, t: Tile, ts: Energy) -> Rate {
        let p2 = canvas.u_move_point_e(p);
        if (!canvas.inbounds(p2)) | (unsafe { canvas.uv_p(p2) == 0 } | self.is_seed(p2)) {
            0.0
        } else {
            let t2 = unsafe { canvas.uv_p(p2) };
            {
                self.k_f_hat()
                    * Rate::exp(
                        -ts - self.bond_strength_of_tile_at_point(canvas, PointSafe2(p2), t2) // FIXME
                        + 2. * self.get_energy_we(t, t2),
                    )
            }
        }
    }

    fn chunk_detach_rate<C: State>(&self, canvas: &C, p: Point, t: Tile) -> Rate {
        match self.chunk_size {
            ChunkSize::Single => 0.0,
            ChunkSize::Dimer => {
                let ts = { self.bond_strength_of_tile_at_point(canvas, PointSafe2(p), t) }; // FIXME
                self.dimer_s_detach_rate(canvas, p, t, ts)
                    + self.dimer_e_detach_rate(canvas, p, t, ts)
            }
        }
    }

    fn choose_chunk_detachment<C: State>(
        &self,
        canvas: &C,
        p: PointSafe2,
        tile: Tile,
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
                    let t2 = { canvas.tile_at_point(p2) };
                    now_empty.push(p);
                    now_empty.push(p2);
                    // North tile adjacents
                    if self.get_energy_ns(canvas.tile_to_n(p), tile) > 0. {
                        possible_starts.push(PointSafe2(canvas.move_sa_n(p).0))
                    };
                    if self.get_energy_we(canvas.tile_to_w(p), tile) > 0. {
                        possible_starts.push(PointSafe2(canvas.move_sa_w(p).0))
                    };
                    if self.get_energy_we(tile, canvas.tile_to_e(p)) > 0. {
                        possible_starts.push(PointSafe2(canvas.move_sa_e(p).0))
                    };
                    // South tile adjacents
                    if self.get_energy_ns(t2, canvas.tile_to_s(p2)) > 0. {
                        possible_starts.push(PointSafe2(canvas.move_sa_s(p2).0))
                    };
                    if self.get_energy_we(canvas.tile_to_w(p2), t2) > 0. {
                        possible_starts.push(PointSafe2(canvas.move_sa_w(p2).0))
                    };
                    if self.get_energy_we(t2, canvas.tile_to_e(p2)) > 0. {
                        possible_starts.push(PointSafe2(canvas.move_sa_e(p2).0))
                    };
                    return;
                }
                *acc -= self.dimer_e_detach_rate(canvas, p.0, tile, ts);
                if *acc <= 0. {
                    let p2 = PointSafe2(canvas.move_sa_e(p).0);
                    let t2 = { canvas.tile_at_point(p2) };
                    now_empty.push(p);
                    now_empty.push(p2);
                    // West tile adjacents
                    if self.get_energy_we(canvas.tile_to_w(p), tile) > 0. {
                        possible_starts.push(PointSafe2(canvas.move_sa_w(p).0))
                    };
                    if self.get_energy_ns(canvas.tile_to_n(p), tile) > 0. {
                        possible_starts.push(PointSafe2(canvas.move_sa_n(p).0))
                    };
                    if self.get_energy_ns(tile, canvas.tile_to_s(p)) > 0. {
                        possible_starts.push(PointSafe2(canvas.move_sa_s(p).0))
                    };
                    // East tile adjacents
                    if self.get_energy_we(t2, canvas.tile_to_e(p2)) > 0. {
                        possible_starts.push(PointSafe2(canvas.move_sa_e(p2).0))
                    };
                    if self.get_energy_ns(canvas.tile_to_n(p2), t2) > 0. {
                        possible_starts.push(PointSafe2(canvas.move_sa_n(p2).0))
                    };
                    if self.get_energy_ns(t2, canvas.tile_to_s(p2)) > 0. {
                        possible_starts.push(PointSafe2(canvas.move_sa_s(p2).0))
                    };
                    return;
                }
                panic!("{acc:#?}")
            }
        }
    }
}

impl System for OldKTAM {
    fn event_rate_at_point<S: State>(&self, canvas: &S, point: PointSafeHere) -> Rate {
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

                    let mut friends = HashSetType::<Tile>::default();

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

    fn choose_event_at_point<S: State>(&self, canvas: &S, p: PointSafe2, mut acc: Rate) -> Event {
        let tile = { canvas.tile_at_point(p) };

        let tn = { canvas.tile_to_n(p) };
        let tw = { canvas.tile_to_w(p) };
        let te = { canvas.tile_to_e(p) };
        let ts = { canvas.tile_to_s(p) };

        if tile != 0 {
            acc -= {
                self.k_f_hat() * Rate::exp(-self.bond_strength_of_tile_at_point(canvas, p, tile))
            };

            let mut possible_starts = Vec::new();
            let mut now_empty = Vec::new();

            if acc <= 0. {
                // FIXME
                if self.get_energy_ns(tn, tile) > 0. {
                    possible_starts.push(PointSafe2(canvas.move_sa_n(p).0))
                };
                if self.get_energy_we(tw, tile) > 0. {
                    possible_starts.push(PointSafe2(canvas.move_sa_w(p).0))
                };
                if self.get_energy_ns(tile, ts) > 0. {
                    possible_starts.push(PointSafe2(canvas.move_sa_s(p).0))
                };
                if self.get_energy_we(tile, te) > 0. {
                    possible_starts.push(PointSafe2(canvas.move_sa_e(p).0))
                };

                now_empty.push(p);

                match self.determine_fission(canvas, &possible_starts, &now_empty) {
                    super::oldktam_fission::FissionResult::NoFission => Event::MonomerDetachment(p),
                    super::oldktam_fission::FissionResult::FissionGroups(g) => {
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
                        panic!("Ran out of event possibilities at {p:#?}, acc={acc:#?}")
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
                    super::oldktam_fission::FissionResult::NoFission => {
                        Event::PolymerDetachment(now_empty)
                    }
                    super::oldktam_fission::FissionResult::FissionGroups(g) => {
                        //println!("Fission handling {:?} {:?}", p, tile);
                        match self.fission_handling {
                            FissionHandling::NoFission => Event::None,
                            FissionHandling::JustDetach => Event::PolymerDetachment(now_empty),
                            FissionHandling::KeepSeeded => {
                                let sl = System::seed_locs(self);
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
            let mut friends = HashSetType::<Tile>::default();

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
                for (p, t) in f.iter() {
                    v.push((PointSafe2(*p), *t));
                }
            }
        };

        v
    }

    fn update_after_event<S: State>(&self, state: &mut S, event: &Event) {
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
                    self.update_points(state, &points);
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

                    self.update_points(state, &points);
                }
            },
            Event::PolymerDetachment(v) => {
                let mut points = Vec::new();
                for p in v {
                    points.extend(self.points_to_update_around(state, p));
                }
                points.sort_unstable();
                points.dedup();
                self.update_points(state, &points);
            }
            Event::PolymerAttachment(v) | Event::PolymerChange(v) => {
                let mut points = Vec::new();
                for (p, _) in v {
                    points.extend(self.points_to_update_around(state, p));
                }
                points.sort_unstable();
                points.dedup();
                self.update_points(state, &points);
            }
        }
    }

    fn calc_mismatch_locations<S: State>(&self, state: &S) -> Array2<usize> {
        let threshold = 0.1;
        let mut arr = Array2::zeros(state.raw_array().raw_dim());

        for y in 1..(arr.nrows() - 1) {
            for x in 1..(arr.ncols() - 1) {
                let p = PointSafe2((y, x));
                let t = state.tile_at_point(p);

                if t == 0 {
                    arr[(y, x)] = 0;
                    continue;
                }

                let tn = state.tile_to_n(p);
                let te = state.tile_to_e(p);
                let ts = state.tile_to_s(p);
                let tw = state.tile_to_w(p);

                let mm_n = ((tn != 0) & (self.get_energy_ns(tn, t) < threshold)) as usize;
                let mm_e = ((te != 0) & (self.get_energy_we(t, te) < threshold)) as usize;
                let mm_s = ((ts != 0) & (self.get_energy_ns(t, ts) < threshold)) as usize;
                let mm_w = ((tw != 0) & (self.get_energy_we(tw, t) < threshold)) as usize;

                arr[(y, x)] = 8 * mm_n + 4 * mm_e + 2 * mm_s + mm_w;
            }
        }

        arr
    }
}

impl SystemWithDimers for OldKTAM {
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
}

impl FromTileSet for OldKTAM {
    fn from_tileset(tileset: &TileSet) -> Result<Self, RgrowError> {
        let proc = ProcessedTileSet::from_tileset(tileset)?;

        if proc.has_duples {
            return Err(ModelError::DuplesNotSupported.into());
        }

        let seed = if proc.seed.is_empty() {
            Seed::None()
        } else if proc.seed.len() == 1 {
            let (x, y, v) = proc.seed[0];
            Seed::SingleTile {
                point: (x, y),
                tile: v,
            }
        } else {
            let mut hm = HashMap::default();
            hm.extend(proc.seed.iter().map(|(y, x, v)| ((*y, *x), *v)));
            Seed::MultiTile(hm)
        };

        let s = OldKTAM::from_ktam(
            proc.tile_stoics,
            proc.tile_edges,
            proc.glue_strengths,
            proc.gluelinks,
            tileset.options.gse,
            tileset.options.gmc,
            Some(tileset.options.alpha),
            tileset.options.kf,
            Some(seed),
            Some(tileset.options.fission),
            tileset.options.chunk_handling,
            tileset.options.chunk_size,
            proc.tile_names,
            proc.tile_colors,
        );

        Ok(s)
    }
}

impl SystemInfo for OldKTAM {
    fn tile_concs(&self) -> Vec<f64> {
        todo!()
    }

    fn tile_stoics(&self) -> Vec<f64> {
        todo!()
    }
}
