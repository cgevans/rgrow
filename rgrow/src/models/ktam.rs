/// A revised kTAM implementation with duples and fission, and the intention to eventually
/// add dimer detachment and attachment.
///
/// Implementation notes:
///
/// - Concentration is stored *in nM*, as this is most likely to result in reasonable values.
/// - Concentration, rather than stoichiometry and $G_{mc}$ values, are stored internally.
/// - As with Xgrow, duples are treated as two tiles, where the right/bottom tile is "fake".
/// - Unlike Xgrow, there is no *requirement* that there be a seed.
use super::ktam_fission::*;
use crate::{
    base::{GrowError, RgrowError},
    canvas::{PointSafe2, PointSafeHere},
    state::State,
    system::{
        ChunkHandling, ChunkSize, DimerInfo, Event, FissionHandling, NeededUpdate, Orientation,
        System, SystemInfo, SystemWithDimers, TileBondInfo,
    },
    tileset::{FromTileSet, ProcessedTileSet, TileSet},
};

use crate::base::{HashMapType, HashSetType};
use ndarray::prelude::*;
use rand::prelude::Distribution;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::base::{Glue, Tile};

type Conc = f64;
type Strength = f64;
type RatePerConc = f64;
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

const FAKE_EVENT_RATE: f64 = 1e-20;

fn energy_exp_times_u0(x: f64) -> Conc {
    1.0e9 * x.exp()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Seed {
    None(),
    SingleTile { point: PointSafe2, tile: Tile },
    MultiTile(HashMapType<PointSafe2, Tile>),
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
enum TileShape {
    Single,
    DupleToRight(Tile),
    DupleToBottom(Tile),
    DupleToLeft(Tile),
    DupleToTop(Tile),
}

impl Default for TileShape {
    fn default() -> Self {
        Self::Single
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KTAM {
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
    pub tile_colors: Vec<[u8; 4]>,
    pub fission_handling: FissionHandling,
    pub glue_names: Vec<String>,

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
    duple_info: Array1<TileShape>,
    should_be_counted: Array1<bool>,
}

impl System for KTAM {
    fn update_after_event<S: State>(&self, state: &mut S, event: &Event) {
        match event {
            Event::None => todo!(),
            Event::MonomerAttachment(p, _)
            | Event::MonomerDetachment(p)
            | Event::MonomerChange(p, _) => {
                self._update_monomer_points(state, p);
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
            Event::PolymerAttachment(t) | Event::PolymerChange(t) => {
                let mut points = Vec::new();
                for p in t {
                    points.extend(self.points_to_update_around(state, &p.0));
                }
                points.sort_unstable();
                points.dedup();
                self.update_points(state, &points);
            }
        }
    }

    fn calc_ntiles<S: State>(&self, state: &S) -> crate::base::NumTiles {
        state.calc_ntiles_with_tilearray(&self.should_be_counted)
    }

    fn event_rate_at_point<S: State>(
        &self,
        state: &S,
        p: crate::canvas::PointSafeHere,
    ) -> crate::base::Rate {
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

    fn choose_event_at_point<S: State>(
        &self,
        state: &S,
        p: PointSafe2,
        acc: crate::base::Rate,
    ) -> Event {
        // println!("{:?}", acc);
        match self.choose_detachment_at_point(state, p, acc) {
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

    fn perform_event<S: State>(&self, state: &mut S, event: &Event) -> &Self {
        match event {
            Event::None => panic!("Being asked to perform null event."),
            Event::MonomerAttachment(point, tile) => {
                state.set_sa(point, tile);
                match self.tile_shape(*tile) {
                    TileShape::Single => (),
                    TileShape::DupleToRight(dt) => {
                        debug_assert_eq!(state.tile_to_e(*point), 0);
                        state.set_sa_countabletilearray(
                            &PointSafe2(state.move_sa_e(*point).0),
                            &dt,
                            &self.should_be_counted,
                        );
                    }
                    TileShape::DupleToBottom(dt) => {
                        debug_assert_eq!(state.tile_to_s(*point), 0);
                        state.set_sa_countabletilearray(
                            &PointSafe2(state.move_sa_s(*point).0),
                            &dt,
                            &self.should_be_counted,
                        );
                    }
                    TileShape::DupleToLeft(dt) => {
                        debug_assert_eq!(state.tile_to_w(*point), 0);
                        state.set_sa_countabletilearray(
                            &PointSafe2(state.move_sa_w(*point).0),
                            &dt,
                            &self.should_be_counted,
                        );
                    }
                    TileShape::DupleToTop(dt) => {
                        debug_assert_eq!(state.tile_to_n(*point), 0);
                        state.set_sa_countabletilearray(
                            &PointSafe2(state.move_sa_n(*point).0),
                            &dt,
                            &self.should_be_counted,
                        );
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
                        state.set_sa_countabletilearray(
                            &PointSafe2(state.move_sa_e(*point).0),
                            &0,
                            &self.should_be_counted,
                        )
                    }
                    TileShape::DupleToBottom(dt) => {
                        debug_assert_eq!(dt, state.tile_to_s(*point));
                        state.set_sa_countabletilearray(
                            &PointSafe2(state.move_sa_s(*point).0),
                            &0,
                            &self.should_be_counted,
                        )
                    }
                    TileShape::DupleToLeft(dt) => {
                        debug_assert_eq!(dt, state.tile_to_w(*point));
                        state.set_sa_countabletilearray(
                            &PointSafe2(state.move_sa_w(*point).0),
                            &0,
                            &self.should_be_counted,
                        )
                    }
                    TileShape::DupleToTop(dt) => {
                        debug_assert_eq!(dt, state.tile_to_n(*point));
                        state.set_sa_countabletilearray(
                            &PointSafe2(state.move_sa_n(*point).0),
                            &0,
                            &self.should_be_counted,
                        )
                    }
                }

                state.set_sa_countabletilearray(point, tile, &self.should_be_counted);

                match self.tile_shape(*tile) {
                    TileShape::Single => (),
                    TileShape::DupleToRight(dt) => {
                        debug_assert_eq!(state.tile_to_e(*point), 0);
                        state.set_sa_countabletilearray(
                            &PointSafe2(state.move_sa_e(*point).0),
                            &dt,
                            &self.should_be_counted,
                        );
                    }
                    TileShape::DupleToBottom(dt) => {
                        debug_assert_eq!(state.tile_to_s(*point), 0);
                        state.set_sa_countabletilearray(
                            &PointSafe2(state.move_sa_s(*point).0),
                            &dt,
                            &self.should_be_counted,
                        );
                    }
                    TileShape::DupleToLeft(dt) => {
                        debug_assert_eq!(state.tile_to_w(*point), 0);
                        state.set_sa_countabletilearray(
                            &PointSafe2(state.move_sa_w(*point).0),
                            &dt,
                            &self.should_be_counted,
                        );
                    }
                    TileShape::DupleToTop(dt) => {
                        debug_assert_eq!(state.tile_to_n(*point), 0);
                        state.set_sa_countabletilearray(
                            &PointSafe2(state.move_sa_n(*point).0),
                            &dt,
                            &self.should_be_counted,
                        );
                    }
                }
            }
            Event::MonomerDetachment(point) => {
                match self.tile_shape(state.tile_at_point(*point)) {
                    TileShape::Single => (),
                    TileShape::DupleToRight(dt) => {
                        debug_assert_eq!(state.tile_to_e(*point), dt);
                        state.set_sa_countabletilearray(
                            &PointSafe2(state.move_sa_e(*point).0),
                            &0,
                            &self.should_be_counted,
                        );
                    }
                    TileShape::DupleToBottom(dt) => {
                        debug_assert_eq!(state.tile_to_s(*point), dt);
                        state.set_sa_countabletilearray(
                            &PointSafe2(state.move_sa_s(*point).0),
                            &0,
                            &self.should_be_counted,
                        );
                    }
                    TileShape::DupleToLeft(dt) => {
                        debug_assert_eq!(state.tile_to_w(*point), dt);
                        state.set_sa_countabletilearray(
                            &PointSafe2(state.move_sa_w(*point).0),
                            &0,
                            &self.should_be_counted,
                        );
                    }
                    TileShape::DupleToTop(dt) => {
                        debug_assert_eq!(state.tile_to_n(*point), dt);
                        state.set_sa_countabletilearray(
                            &PointSafe2(state.move_sa_n(*point).0),
                            &0,
                            &self.should_be_counted,
                        );
                    }
                }
                state.set_sa_countabletilearray(point, &0, &self.should_be_counted);
            }
            Event::PolymerAttachment(changelist) => {
                for (point, tile) in changelist {
                    state.set_sa_countabletilearray(point, tile, &self.should_be_counted);
                    match self.tile_shape(*tile) {
                        TileShape::Single => (),
                        TileShape::DupleToRight(dt) => {
                            debug_assert_eq!(state.tile_to_e(*point), 0);
                            state.set_sa_countabletilearray(
                                &PointSafe2(state.move_sa_e(*point).0),
                                &dt,
                                &self.should_be_counted,
                            );
                        }
                        TileShape::DupleToBottom(dt) => {
                            debug_assert_eq!(state.tile_to_s(*point), 0);
                            state.set_sa_countabletilearray(
                                &PointSafe2(state.move_sa_s(*point).0),
                                &dt,
                                &self.should_be_counted,
                            );
                        }
                        TileShape::DupleToLeft(dt) => {
                            debug_assert_eq!(state.tile_to_w(*point), 0);
                            state.set_sa_countabletilearray(
                                &PointSafe2(state.move_sa_w(*point).0),
                                &dt,
                                &self.should_be_counted,
                            );
                        }
                        TileShape::DupleToTop(dt) => {
                            debug_assert_eq!(state.tile_to_n(*point), 0);
                            state.set_sa_countabletilearray(
                                &PointSafe2(state.move_sa_n(*point).0),
                                &dt,
                                &self.should_be_counted,
                            );
                        }
                    }
                }
            }
            Event::PolymerChange(changelist) => {
                for (point, tile) in changelist {
                    let oldt = state.tile_at_point(*point);

                    match self.tile_shape(oldt) {
                        // Fixme: somewhat unsafe
                        TileShape::Single => (),
                        TileShape::DupleToRight(dt) => {
                            debug_assert_eq!(dt, state.tile_to_e(*point));
                            state.set_sa_countabletilearray(
                                &PointSafe2(state.move_sa_e(*point).0),
                                &0,
                                &self.should_be_counted,
                            )
                        }
                        TileShape::DupleToBottom(dt) => {
                            debug_assert_eq!(dt, state.tile_to_s(*point));
                            state.set_sa_countabletilearray(
                                &PointSafe2(state.move_sa_s(*point).0),
                                &0,
                                &self.should_be_counted,
                            )
                        }
                        TileShape::DupleToLeft(dt) => {
                            debug_assert_eq!(dt, state.tile_to_w(*point));
                            state.set_sa_countabletilearray(
                                &PointSafe2(state.move_sa_w(*point).0),
                                &0,
                                &self.should_be_counted,
                            )
                        }
                        TileShape::DupleToTop(dt) => {
                            debug_assert_eq!(dt, state.tile_to_n(*point));
                            state.set_sa_countabletilearray(
                                &PointSafe2(state.move_sa_n(*point).0),
                                &0,
                                &self.should_be_counted,
                            )
                        }
                    }

                    state.set_sa_countabletilearray(point, tile, &self.should_be_counted);

                    match self.tile_shape(*tile) {
                        TileShape::Single => (),
                        TileShape::DupleToRight(dt) => {
                            debug_assert_eq!(state.tile_to_e(*point), 0);
                            state.set_sa_countabletilearray(
                                &PointSafe2(state.move_sa_e(*point).0),
                                &dt,
                                &self.should_be_counted,
                            );
                        }
                        TileShape::DupleToBottom(dt) => {
                            debug_assert_eq!(state.tile_to_s(*point), 0);
                            state.set_sa_countabletilearray(
                                &PointSafe2(state.move_sa_s(*point).0),
                                &dt,
                                &self.should_be_counted,
                            );
                        }
                        TileShape::DupleToLeft(dt) => {
                            debug_assert_eq!(state.tile_to_w(*point), 0);
                            state.set_sa_countabletilearray(
                                &PointSafe2(state.move_sa_w(*point).0),
                                &dt,
                                &self.should_be_counted,
                            );
                        }
                        TileShape::DupleToTop(dt) => {
                            debug_assert_eq!(state.tile_to_n(*point), 0);
                            state.set_sa_countabletilearray(
                                &PointSafe2(state.move_sa_n(*point).0),
                                &dt,
                                &self.should_be_counted,
                            );
                        }
                    }
                }
            }
            Event::PolymerDetachment(changelist) => {
                for point in changelist {
                    match self.tile_shape(state.tile_at_point(*point)) {
                        TileShape::Single => (),
                        TileShape::DupleToRight(dt) => {
                            debug_assert_eq!(state.tile_to_e(*point), dt);
                            state.set_sa_countabletilearray(
                                &PointSafe2(state.move_sa_e(*point).0),
                                &0,
                                &self.should_be_counted,
                            );
                        }
                        TileShape::DupleToBottom(dt) => {
                            debug_assert_eq!(state.tile_to_s(*point), dt);
                            state.set_sa_countabletilearray(
                                &PointSafe2(state.move_sa_s(*point).0),
                                &0,
                                &self.should_be_counted,
                            );
                        }
                        TileShape::DupleToLeft(dt) => {
                            debug_assert_eq!(state.tile_to_w(*point), dt);
                            state.set_sa_countabletilearray(
                                &PointSafe2(state.move_sa_w(*point).0),
                                &0,
                                &self.should_be_counted,
                            );
                        }
                        TileShape::DupleToTop(dt) => {
                            debug_assert_eq!(state.tile_to_n(*point), dt);
                            state.set_sa_countabletilearray(
                                &PointSafe2(state.move_sa_n(*point).0),
                                &0,
                                &self.should_be_counted,
                            );
                        }
                    }

                    state.set_sa_countabletilearray(point, &0, &self.should_be_counted);
                }
            }
        }
        state.add_events(1);
        self
    }

    fn seed_locs(&self) -> Vec<(PointSafe2, Tile)> {
        self._seed_locs()
    }

    fn calc_mismatch_locations<S: State>(&self, state: &S) -> Array2<usize> {
        let threshold = 0.5; // Todo: fix this
        let mut mismatch_locations = Array2::<usize>::zeros((state.nrows(), state.ncols()));

        // TODO: this should use an iterator from the canvas, which we should implement.
        for i in 0..state.nrows() {
            for j in 0..state.ncols() {
                if !state.inbounds((i, j)) {
                    continue;
                }
                let p = PointSafe2((i, j));

                let t = state.tile_at_point(p);

                if t == 0 {
                    continue;
                }

                let tn;
                let te;
                let ts;
                let tw;

                // We set duple directions to 0, because these will be
                // excluded from the mismatch calculation.
                match self.tile_shape(t) {
                    TileShape::Single => {
                        tn = state.tile_to_n(p);
                        te = state.tile_to_e(p);
                        ts = state.tile_to_s(p);
                        tw = state.tile_to_w(p);
                    }
                    TileShape::DupleToRight(_) => {
                        tn = state.tile_to_n(p);
                        te = 0;
                        ts = state.tile_to_s(p);
                        tw = state.tile_to_w(p);
                    }
                    TileShape::DupleToBottom(_) => {
                        tn = state.tile_to_n(p);
                        te = state.tile_to_e(p);
                        ts = 0;
                        tw = state.tile_to_w(p);
                    }
                    TileShape::DupleToLeft(_) => {
                        tn = state.tile_to_n(p);
                        te = state.tile_to_e(p);
                        ts = state.tile_to_s(p);
                        tw = 0;
                    }
                    TileShape::DupleToTop(_) => {
                        tn = 0;
                        te = state.tile_to_e(p);
                        ts = state.tile_to_s(p);
                        tw = state.tile_to_w(p);
                    }
                }

                let mm_n = ((tn != 0) & (self.get_energy_ns(tn, t) < threshold)) as usize;
                let mm_e = ((te != 0) & (self.get_energy_we(t, te) < threshold)) as usize;
                let mm_s = ((ts != 0) & (self.get_energy_ns(t, ts) < threshold)) as usize;
                let mm_w = ((tw != 0) & (self.get_energy_we(tw, t) < threshold)) as usize;

                mismatch_locations[(i, j)] = 8 * mm_n + 4 * mm_e + 2 * mm_s + mm_w;
            }
        }

        mismatch_locations
    }

    fn set_param(
        &mut self,
        name: &str,
        value: Box<dyn std::any::Any>,
    ) -> Result<NeededUpdate, GrowError> {
        match name {
            "g_se" => {
                let g_se = value
                    .downcast_ref::<f64>()
                    .ok_or(GrowError::WrongParameterType(name.to_string()))?;
                self.g_se = *g_se;
                self.update_system();
                Ok(NeededUpdate::NonZero)
            }
            "alpha" => {
                let alpha = value
                    .downcast_ref::<f64>()
                    .ok_or(GrowError::WrongParameterType(name.to_string()))?;
                self.alpha = *alpha;
                self.update_system();
                Ok(NeededUpdate::NonZero)
            }
            "kf" => {
                let kf = value
                    .downcast_ref::<f64>()
                    .ok_or(GrowError::WrongParameterType(name.to_string()))?;
                self.kf = *kf;
                self.update_system();
                Ok(NeededUpdate::NonZero)
            }
            _ => Err(GrowError::NoParameter(name.to_string())),
        }
    }

    fn get_param(&self, name: &str) -> Result<Box<dyn std::any::Any>, GrowError> {
        match name {
            "g_se" => Ok(Box::new(self.g_se)),
            "alpha" => Ok(Box::new(self.alpha)),
            "kf" => Ok(Box::new(self.kf)),
            _ => Err(GrowError::NoParameter(name.to_string())),
        }
    }
}

impl SystemWithDimers for KTAM {
    fn calc_dimers(&self) -> Vec<DimerInfo> {
        // It is (reasonably) safe for us to use the same code that we used in the old StaticKTAM, despite duples being
        // here, because our EW/NS energies include the right/bottom tiles.  However, (FIXME), we need to think about
        // how this might actually double-count / double some rates: if, eg, a single tile can attach in two places to
        // a double tile, are we double-counting the rates?  Note also that this relies on
        let mut dvec = Vec::new();

        for ((t1, t2), e) in self.energy_ns.indexed_iter() {
            if *e > 0. {
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
            if *e > 0. {
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
}

impl TileBondInfo for KTAM {
    fn tile_color(&self, tile_number: Tile) -> [u8; 4] {
        self.tile_colors[tile_number as usize]
    }

    fn tile_name(&self, tile_number: Tile) -> &str {
        self.tile_names[tile_number as usize].as_str()
    }

    fn bond_name(&self, bond_number: usize) -> &str {
        &self.glue_names[bond_number]
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

impl SystemInfo for KTAM {
    fn tile_concs(&self) -> Vec<f64> {
        self.tile_concs.clone().into_raw_vec()
    }

    fn tile_stoics(&self) -> Vec<f64> {
        todo!()
    }
}

impl KTAM {
    pub fn new_sized(ntiles: Tile, nglues: Glue) -> Self {
        let ntiles: usize = ntiles as usize;
        Self {
            tile_names: Vec::new(),
            tile_concs: Array1::zeros(ntiles + 1),
            tile_edges: Array2::zeros((ntiles + 1, 4)),
            glue_names: Vec::new(),
            glue_strengths: Array1::zeros(nglues + 1),
            glue_links: Array2::zeros((nglues + 1, nglues + 1)),
            g_se: (9.),
            alpha: (0.),
            kf: (1e-3),
            double_to_right: Array1::zeros(ntiles + 1),
            double_to_bottom: Array1::zeros(ntiles + 1),
            seed: Seed::None(),
            tile_colors: Vec::new(),
            fission_handling: FissionHandling::NoFission,
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
            duple_info: Array1::default(ntiles + 1),
            should_be_counted: Array1::default(ntiles + 1),
        }
    }

    pub fn set_duples(&mut self, hduples: Vec<(Tile, Tile)>, vduples: Vec<(Tile, Tile)>) {
        // Reset double_to_right and double_to_bottom to zeros
        self.double_to_right.fill(0);
        self.double_to_bottom.fill(0);

        // For each hduple, set the first index to the second value
        for (i, j) in hduples {
            self.double_to_right[i as usize] = j;
        }

        // For each vduples, set the first index to the second value
        for (i, j) in vduples {
            self.double_to_bottom[i as usize] = j;
        }

        self.update_system();
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_ktam(
        mut tile_stoics: Array1<f64>,
        tile_edges: Array2<Glue>,
        glue_strengths: Array1<f64>,
        g_se: f64,
        g_mc: f64,
        alpha: Option<f64>,
        _k_f: Option<f64>,
        seed: Option<Seed>,
        fission_handling: Option<FissionHandling>,
        _chunk_handling: Option<ChunkHandling>,
        _chunk_size: Option<ChunkSize>,
        tile_names: Option<Vec<String>>,
        tile_colors: Option<Vec<[u8; 4]>>,
    ) -> Self {
        let ntiles = tile_stoics.len() as Tile;

        tile_stoics.map_inplace(|x| *x *= 1.0e9 * (-g_mc + alpha.unwrap_or(0.)).exp());

        let mut ktam = Self::new_sized(tile_stoics.len() as Tile - 1, glue_strengths.len() - 1);

        ktam.tile_concs = tile_stoics;
        ktam.tile_edges = tile_edges;
        ktam.glue_strengths = glue_strengths;
        ktam.g_se = g_se;
        ktam.alpha = alpha.unwrap_or(ktam.alpha);
        ktam.seed = seed.unwrap_or(ktam.seed);
        //ktam.tile_colors = tile_colors.unwrap_or(ktam.tile_colors);
        ktam.tile_names = tile_names.unwrap_or(ktam.tile_names);

        ktam.tile_colors = match tile_colors {
            Some(tc) => tc,
            None => {
                let mut rng = rand::thread_rng();
                let ug = rand::distributions::Uniform::new(100u8, 254);
                (0..ntiles)
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

        ktam.fission_handling = fission_handling.unwrap_or(ktam.fission_handling);

        ktam.update_system();

        ktam
    }

    pub fn update_system(&mut self) {
        let ntiles = self.tile_concs.len() as Tile;

        for t1 in 0..(ntiles as usize) {
            for t2 in 0..(ntiles as usize) {
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
            self.should_be_counted[t1] =  (t1 > 0) && (self.tile_concs[t1] > 0.);
        }

        if (self.double_to_right.sum() > 0) || (self.double_to_bottom.sum() > 0) {
            self.has_duples = true;
            for (t1, t2) in self.double_to_right.indexed_iter() {
                if (t1 > 0) & (t2 > &0) {
                    let t2 = *t2 as usize;
                    self.duple_info[t2] = TileShape::DupleToLeft(t1 as Tile);
                    self.duple_info[t1] = TileShape::DupleToRight(t2 as Tile);
                    self.should_be_counted[t2] = false;
                    self.energy_we[(t1, t2)] = 0.0;
                }
            }
            for (t1, t2) in self.double_to_bottom.indexed_iter() {
                if (t1 > 0) & (t2 > &0) {
                    let t2 = *t2 as usize;
                    self.duple_info[t2] = TileShape::DupleToTop(t1 as Tile);
                    self.duple_info[t1] = TileShape::DupleToBottom(t2 as Tile);
                    self.should_be_counted[t2] = false;
                    self.energy_ns[(t1, t2)] = 0.0;
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
        for t1 in 0..(ntiles) {
            for t2 in 0..(ntiles) {
                let t1t = t1 as Tile;
                match self.tile_shape(t1t) {
                    TileShape::Single => {
                        if self.get_energy_ns(t2, t1) != 0. {
                            self.friends_n[t2 as usize].insert(t1);
                        }
                        if self.get_energy_we(t2, t1) != 0. {
                            self.friends_w[t2 as usize].insert(t1);
                        }
                        if self.get_energy_ns(t1, t2) != 0. {
                            self.friends_s[t2 as usize].insert(t1);
                        }
                        if self.get_energy_we(t1, t2) != 0. {
                            self.friends_e[t2 as usize].insert(t1);
                        }
                    }
                    TileShape::DupleToRight(td) => {
                        if self.get_energy_ns(t2, td) != 0. {
                            self.friends_ne[t2 as usize].insert(t1);
                        }
                        if self.get_energy_ns(td, t2) != 0. {
                            self.friends_se[t2 as usize].insert(t1);
                        }
                        if self.get_energy_we(td, t2) != 0. {
                            self.friends_ee[t2 as usize].insert(t1);
                        }
                        if self.get_energy_ns(t2, t1) != 0. {
                            self.friends_n[t2 as usize].insert(t1);
                        }
                        if self.get_energy_we(t2, t1) != 0. {
                            self.friends_w[t2 as usize].insert(t1);
                        }
                        if self.get_energy_ns(t1, t2) != 0. {
                            self.friends_s[t2 as usize].insert(t1);
                        }
                        if self.get_energy_we(t1, t2) != 0. {
                            self.friends_e[t2 as usize].insert(t1);
                        }
                    }
                    TileShape::DupleToBottom(td) => {
                        if self.get_energy_we(t2, td) != 0. {
                            self.friends_sw[t2 as usize].insert(t1);
                        }
                        if self.get_energy_we(td, t2) != 0. {
                            self.friends_se[t2 as usize].insert(t1);
                        }
                        if self.get_energy_ns(td, t2) != 0. {
                            self.friends_ss[t2 as usize].insert(t1);
                        }
                        if self.get_energy_ns(t2, t1) != 0. {
                            self.friends_n[t2 as usize].insert(t1);
                        }
                        if self.get_energy_we(t2, t1) != 0. {
                            self.friends_w[t2 as usize].insert(t1);
                        }
                        if self.get_energy_ns(t1, t2) != 0. {
                            self.friends_s[t2 as usize].insert(t1);
                        }
                        if self.get_energy_we(t1, t2) != 0. {
                            self.friends_e[t2 as usize].insert(t1);
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
            } => p == *seed_point,
            Seed::MultiTile(seed_map) => seed_map.contains_key(&p),
        }
    }

    fn is_fake_duple(&self, t: Tile) -> bool {
        match self.duple_info[t as usize] {
            TileShape::Single | TileShape::DupleToRight(_) | TileShape::DupleToBottom(_) => false,
            TileShape::DupleToLeft(_) | TileShape::DupleToTop(_) => true,
        }
    }

    pub fn monomer_detachment_rate_at_point<S: State>(&self, state: &S, p: PointSafe2) -> Rate {
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
        if (self.has_duples) && (self.is_fake_duple(t)) {
            return FAKE_EVENT_RATE;
        }
        self.kf
            * energy_exp_times_u0(-self.bond_energy_of_tile_type_at_point(state, p, t) + self.alpha)
    }

    fn _seed_locs(&self) -> Vec<(PointSafe2, Tile)> {
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

    pub fn choose_detachment_at_point<S: State>(
        &self,
        state: &S,
        p: PointSafe2,
        mut acc: Rate,
    ) -> (bool, Rate, Event) {
        acc -= self.monomer_detachment_rate_at_point(state, p);
        if acc <= 0. {
            // FIXME: may slow things down
            if self.is_seed(p) || ((self.has_duples) && self.is_fake_duple(state.tile_at_point(p)))
            {
                (true, acc, Event::None)
            } else {
                let mut possible_starts = Vec::new();
                let mut now_empty = Vec::new();
                let tile = { state.tile_at_point(p) };

                let tn = { state.tile_to_n(p) };
                let tw = { state.tile_to_w(p) };
                let te = { state.tile_to_e(p) };
                let ts = { state.tile_to_s(p) };
                // FIXME
                if self.get_energy_ns(tn, tile) > 0. {
                    possible_starts.push(PointSafe2(state.move_sa_n(p).0))
                };
                if self.get_energy_we(tw, tile) > 0. {
                    possible_starts.push(PointSafe2(state.move_sa_w(p).0))
                };
                if self.get_energy_ns(tile, ts) > 0. {
                    possible_starts.push(PointSafe2(state.move_sa_s(p).0))
                };
                if self.get_energy_we(tile, te) > 0. {
                    possible_starts.push(PointSafe2(state.move_sa_e(p).0))
                };

                now_empty.push(p);

                match self.determine_fission(state, &possible_starts, &now_empty) {
                    FissionResult::NoFission => (true, acc, Event::MonomerDetachment(p)),
                    FissionResult::FissionGroups(g) => {
                        //println!("Fission handling {:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?} {:?}", p, tile, possible_starts, now_empty, tn, te, ts, tw, canvas.calc_ntiles(), g.map.len());
                        match self.fission_handling {
                            FissionHandling::NoFission => (true, acc, Event::None),
                            FissionHandling::JustDetach => (true, acc, Event::MonomerDetachment(p)),
                            FissionHandling::KeepSeeded => {
                                let sl = self._seed_locs();
                                (
                                    true,
                                    acc,
                                    Event::PolymerDetachment(
                                        g.choose_deletions_seed_unattached(sl),
                                    ),
                                )
                            }
                            FissionHandling::KeepLargest => (
                                true,
                                acc,
                                Event::PolymerDetachment(g.choose_deletions_keep_largest_group()),
                            ),
                            FissionHandling::KeepWeighted => (
                                true,
                                acc,
                                Event::PolymerDetachment(g.choose_deletions_size_weighted()),
                            ),
                        }
                    }
                }
            }
        } else {
            (false, acc, Event::None)
        }
    }

    pub fn total_monomer_attachment_rate_at_point<S: State>(
        &self,
        state: &S,
        p: PointSafe2,
    ) -> Rate {
        match self._find_monomer_attachment_possibilities_at_point(state, p, 0., true) {
            (false, acc, _) => -acc,
            _ => panic!(),
        }
    }

    pub fn choose_attachment_at_point<S: State>(
        &self,
        state: &S,
        p: PointSafe2,
        acc: Rate,
    ) -> (bool, Rate, Event) {
        self.choose_monomer_attachment_at_point(state, p, acc)
    }

    pub fn choose_monomer_attachment_at_point<S: State>(
        &self,
        state: &S,
        p: PointSafe2,
        acc: Rate,
    ) -> (bool, Rate, Event) {
        self._find_monomer_attachment_possibilities_at_point(state, p, acc, false)
    }

    pub fn setup_state<S: State>(&self, state: &mut S) -> Result<(), GrowError> {
        for (p, t) in self.seed_locs() {
            self.set_point(state, p.0, t)?;
        }
        Ok(())
    }

    fn _find_monomer_attachment_possibilities_at_point<S: State>(
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

        let tss: u32;
        let tne: u32;
        let tee: u32;
        let tse: u32;

        if self.has_duples {
            tss = state.tile_to_ss(p);
            tne = state.tile_to_ne(p);
            tee = state.tile_to_ee(p);
            tse = state.tile_to_se(p);
        } else {
            tss = 0;
            tne = 0;
            tee = 0;
            tse = 0;
        }

        // Optimization: if all neighbors are empty, then there is no attachment rate
        if tn == 0 && tw == 0 && te == 0 && ts == 0 {
            if self.has_duples {
                if tss == 0 && tne == 0 && tee == 0 && tse == 0 {
                    return (false, acc, Event::None);
                }
            } else {
                return (false, acc, Event::None);
            }
        }

        let mut friends = HashSetType::<Tile>::default();

        if tn.nonzero() {
            friends.extend(&self.friends_n[tn as usize]);
        }
        if te.nonzero() {
            friends.extend(&self.friends_e[te as usize]);
        }
        if ts.nonzero() {
            friends.extend(&self.friends_s[ts as usize]);
        }
        if tw.nonzero() {
            friends.extend(&self.friends_w[tw as usize]);
        }

        if self.has_duples {
            if tss.nonzero() {
                friends.extend(&self.friends_ss[tss as usize])
            }
            if tne.nonzero() {
                friends.extend(&self.friends_ne[tne as usize])
            }
            if tee.nonzero() {
                friends.extend(&self.friends_ee[tee as usize])
            }
            if tse.nonzero() {
                friends.extend(&self.friends_se[tse as usize])
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
            acc -= self.kf * self.tile_concs[t as usize];
            if !just_calc & (acc <= (0.)) {
                return (true, acc, Event::MonomerAttachment(p, t));
            }
        }
        (false, acc, Event::None)
    }

    pub fn bond_energy_of_tile_type_at_point<S: State>(
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

        energy
    }

    #[inline(always)]
    pub(crate) fn get_energy_ns(&self, tn: Tile, ts: Tile) -> Energy {
        {
            *self.energy_ns.get((tn as usize, ts as usize)).unwrap()
        }
    }

    #[inline(always)]
    pub(crate) fn get_energy_we(&self, tw: Tile, te: Tile) -> Energy {
        {
            *self.energy_we.get((tw as usize, te as usize)).unwrap()
        }
    }

    #[inline(always)]
    fn tile_shape(&self, t: Tile) -> TileShape {
        if self.has_duples {
            unsafe { *self.duple_info.uget(t as usize) }
        } else {
            TileShape::Single
        }
    }

    fn _update_monomer_points<S: State>(&self, state: &mut S, p: &PointSafe2) {
        let points = [
            (
                state.move_sa_n(*p),
                self.event_rate_at_point(state, state.move_sa_n(*p)),
            ),
            (
                state.move_sa_w(*p),
                self.event_rate_at_point(state, state.move_sa_w(*p)),
            ),
            (
                PointSafeHere(p.0),
                self.event_rate_at_point(state, PointSafeHere(p.0)),
            ),
            (
                state.move_sa_e(*p),
                self.event_rate_at_point(state, state.move_sa_e(*p)),
            ),
            (
                state.move_sa_s(*p),
                self.event_rate_at_point(state, state.move_sa_s(*p)),
            ),
            (
                state.move_sa_nn(*p),
                self.event_rate_at_point(state, state.move_sa_nn(*p)),
            ),
            (
                state.move_sa_ne(*p),
                self.event_rate_at_point(state, state.move_sa_ne(*p)),
            ),
            (
                state.move_sa_ee(*p),
                self.event_rate_at_point(state, state.move_sa_ee(*p)),
            ),
            (
                state.move_sa_se(*p),
                self.event_rate_at_point(state, state.move_sa_se(*p)),
            ),
            (
                state.move_sa_ss(*p),
                self.event_rate_at_point(state, state.move_sa_ss(*p)),
            ),
            (
                state.move_sa_sw(*p),
                self.event_rate_at_point(state, state.move_sa_sw(*p)),
            ),
            (
                state.move_sa_ww(*p),
                self.event_rate_at_point(state, state.move_sa_ww(*p)),
            ),
            (
                state.move_sa_nw(*p),
                self.event_rate_at_point(state, state.move_sa_nw(*p)),
            ),
        ];
        state.update_multiple(&points);
    }

    fn points_to_update_around<S: State>(&self, state: &S, p: &PointSafe2) -> Vec<PointSafeHere> {
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
}

impl FromTileSet for KTAM {
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

        let mut newkt = Self::from_ktam(
            proc.tile_stoics,
            proc.tile_edges,
            proc.glue_strengths,
            tileset.options.gse,
            tileset.options.gmc,
            Some(tileset.options.alpha),
            tileset.options.kf,
            Some(seed),
            Some(tileset.options.fission),
            tileset.options.chunk_handling,
            tileset.options.chunk_size,
            Some(proc.tile_names),
            Some(proc.tile_colors),
        );

        newkt.glue_names = proc.glue_names;

        newkt.set_duples(proc.hdoubletiles, proc.vdoubletiles);

        for (g1, g2, s) in proc.gluelinks {
            newkt.glue_links[(g2, g1)] = s;
            newkt.glue_links[(g1, g2)] = s;
        }

        newkt.update_system();

        Ok(newkt)
    }
}

// #[cfg(feature = "python")]
// use pyo3::prelude::*;

// #[cfg(feature = "python")]
// #[pymethods]
// impl KTAM {
//     #[pyfunction]
//     fn py_take_step(&mut self, state: &mut Box<dyn State>) {
//         (self as System).state_step(state, rng, max_time_step);
//     }
// }

#[cfg(test)]
mod tests {
    use anyhow::Context;

    use crate::{
        canvas::{CanvasPeriodic, CanvasSquare, CanvasTube},
        state::{NullStateTracker, QuadTreeState, State, StateCreate},
    };

    use super::*;

    fn test_set_point_newktam<St: StateCreate + State>() -> Result<(), anyhow::Error> {
        let mut system = KTAM::new_sized(5, 5);

        system.update_system();

        let mut state: St = system.new_state((8, 32))?;
        let point = state.center();
        system.set_safe_point(&mut state, point, 1);
        assert_eq!(state.tile_at_point(point), 1);

        let p2 = state.move_sa_e(point);
        system.set_point(&mut state, p2.0, 2)?;
        assert_eq!(state.v_sh(p2), 2);

        Ok(())
    }

    #[test]
    fn test_set_point_for_canvases() -> Result<(), anyhow::Error> {
        test_set_point_newktam::<QuadTreeState<CanvasSquare, NullStateTracker>>()
            .context("CanvasSquare")?;
        test_set_point_newktam::<QuadTreeState<CanvasPeriodic, NullStateTracker>>()
            .context("CanvasPeriodic")?;
        test_set_point_newktam::<QuadTreeState<CanvasTube, NullStateTracker>>()
            .context("CanvasTube")?;
        Ok(())
    }
}
