use std::collections::{HashMap, HashSet};

use ndarray::{Array1, Array2};
use num_traits::Zero;
use serde::{Deserialize, Serialize};

use crate::{
    base::{Glue, GrowError, HashSetType},
    canvas::{PointSafe2, PointSafeHere},
    state::State,
    system::{DimerInfo, Event, FissionHandling, Orientation, System, TileBondInfo},
    type_alias,
    units::*,
};

// Imports for python bindings

#[cfg(feature = "python")]
use crate::python::PyState;
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
use crate::canvas::Canvas;

type_alias!( u32 => Sides );

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
#[cfg_attr(feature = "python", derive(FromPyObject))]
pub struct TileState(pub(crate) u32);

impl TileState {
    pub fn is_null(&self) -> bool {
        self.0 == 0
    }

    pub fn attach_blockers(&self, side: Sides) -> TileState {
        TileState(self.0 | side)
    }

    pub fn detach_blockers(&self, side: Sides) -> TileState {
        TileState(self.0 & (!side))
    }

    pub fn is_blocked(&self, side: Sides) -> bool {
        (self.0 & side) != 0
    }

    pub fn unblock_all(&self) -> TileState {
        TileState(self.0 & NO_BLOCKERS)
    }

    pub fn tile_index(&self) -> TileType {
        TileType((self.0 >> 4) as usize)
    }
}

impl From<u32> for TileState {
    fn from(value: u32) -> Self {
        TileState(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
#[cfg_attr(feature = "python", derive(FromPyObject))]
pub struct TileType(pub(crate) usize);

impl TileType {
    pub fn unblocked(&self) -> TileState {
        TileState((self.0 << 4) as u32)
    }
}

impl From<TileState> for TileType {
    fn from(value: TileState) -> Self {
        TileType((value.0 >> 4) as usize)
    }
}

impl From<TileType> for usize {
    fn from(value: TileType) -> Self {
        value.0
    }
}

impl From<TileState> for u32 {
    fn from(value: TileState) -> Self {
        value.0
    }
}

const NORTH: Sides = 0b0001;
const EAST: Sides = 0b0010;
const SOUTH: Sides = 0b0100;
const WEST: Sides = 0b1000;

const ALL_BLOCKERS: Sides = 0b1111;
const NO_BLOCKERS: Sides = !0b1111;

const ALL_SIDES: [Sides; 4] = [NORTH, EAST, SOUTH, WEST];

pub fn attachments(id: TileState) -> TileState {
    TileState(id.0 & ALL_BLOCKERS)
}

#[inline(always)]
pub const fn inverse(side: Sides) -> Sides {
    match side {
        NORTH => SOUTH,
        SOUTH => NORTH,
        EAST => WEST,
        WEST => EAST,
        _ => panic!("Can only find the inverse of NESW"),
    }
}

pub fn tile_index(tile: TileState) -> TileType {
    TileType((tile.0 >> 4) as usize)
}

/// Index array by side, north = 0, east = 1, south = 2, west = 3
///
/// # Panic
///
/// This will panic when called with a mix of sides (ie, north-east)
pub const fn side_index(side: Sides) -> Option<usize> {
    match side {
        NORTH => Some(0),
        EAST => Some(1),
        SOUTH => Some(2),
        WEST => Some(3),
        _ => None,
    }
}

/// Helper function to help print sides
pub fn side_as_str(side: Sides) -> &'static str {
    match side {
        NORTH => "north",
        EAST => "east",
        SOUTH => "south",
        WEST => "west",
        _ => panic!("Input was not a side"),
    }
}
#[cfg_attr(feature = "python", pyclass(module = "rgrow"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KBlock {
    pub tile_names: Vec<String>,
    pub tile_concentration: Vec<Molar>,
    pub tile_colors: Vec<[u8; 4]>,

    pub glue_names: Vec<String>,
    pub blocker_concentrations: Vec<Molar>,
    pub temperature: Celsius,
    pub seed: HashMap<PointSafe2, TileState>,
    /// Glues of a tile with a given ID
    ///
    /// This is private purposely, use getter function. There are (up to / exactly)
    /// 16 different ids with the same glues, thus we will share ids. This is
    /// enforced by the getter
    ///
    /// [
    ///     (0) -- [North, East, South, West]
    ///     ...
    ///     (n) -- [North, East, South, West]
    /// ]
    tile_glues: Array1<[Glue; 4]>,
    /// Binding strength between two glues
    pub(crate) glue_links: Array2<KcalPerMol>,

    /// What can attach to the north of some *glue*
    ///
    /// For example, if some tile has the glue 6 to the north side, north_friends[6] will
    /// return a hashset with every tile containing a 5 (1<->2, 3<->4, 5<->6)
    ///
    /// To get possible attachemnts to some side of a tile (which is the usual expected use), call
    /// `get_friends_one_side` or `get_friends`
    north_friends: Vec<HashSetType<TileState>>,
    /// Identical to north_friends
    south_friends: Vec<HashSetType<TileState>>,
    /// Identical to north_friends
    east_friends: Vec<HashSetType<TileState>>,
    /// Identical to north_friends
    west_friends: Vec<HashSetType<TileState>>,

    /// Energy of tile and blocker, blocker i contains [N, E, S, W]
    energy_blocker: Array2<KcalPerMol>,

    /// Energy between two tiles, if tile a is to the north of tile b, then
    /// this shoudl be indexed as [(a,b)]
    energy_ns: Array2<KcalPerMol>,
    energy_we: Array2<KcalPerMol>,

    free_blocker_concentrations: Array1<Molar>,

    pub ds_lat: KcalPerMolKelvin,
    pub kf: PerMolarSecond,
    fission_handling: FissionHandling,

    pub no_partially_blocked_attachments: bool,

    pub blocker_energy_adj: KcalPerMol,
}

#[inline(always)]
/// Glue cannot be 0
fn glue_inverse(glue: Glue) -> Glue {
    match glue {
        0 => 0,
        g if g % 2 == 1 => g + 1,
        g => g - 1,
    }
}

impl KBlock {
    pub fn update(&mut self) {
        self.fill_energy_pairs();
        self.fill_energy_blockers();
        self.fill_free_blocker_concentrations();
    }

    /// Get the unblocked friends to one side of some given tile
    pub fn get_unblocked_friends_to_side(
        &self,
        side: Sides,
        tile: TileState,
    ) -> Option<&HashSetType<TileState>> {
        // The tile is blocked, so we dont have any friends
        if tile.is_blocked(side) {
            return None;
        }

        let tile_glue = self.glue_on_side(side, tile);
        Some(match side {
            NORTH => &self.north_friends[tile_glue],
            SOUTH => &self.south_friends[tile_glue],
            EAST => &self.east_friends[tile_glue],
            WEST => &self.west_friends[tile_glue],
            _ => panic!(
                "get_friends_one_side should be called with either NORTH, EAST, SOUTH, or WEST, not a combination"
            ),
        })
    }

    /// Get the friends to some side
    pub fn get_friends(&self, side: Sides, tile: TileState) -> HashSetType<TileState> {
        let mut tile_friends = HashSetType::default();
        for s in ALL_SIDES {
            if side & s != 0 {
                if let Some(ext) = self.get_unblocked_friends_to_side(s, tile) {
                    tile_friends.extend(ext);
                }
            }
        }
        tile_friends
    }

    /// Get the glues. If there are blockers, this wil look past them, and return the
    /// glue that is under it
    pub fn get_tile_raw_glues(&self, tile_id: TileState) -> Vec<Glue> {
        let index = tile_index(tile_id).0;
        self.tile_glues[index].to_vec()
    }

    pub fn glue_on_side(&self, side: Sides, tile_id: TileState) -> Glue {
        let glues = self.get_tile_unblocked_glues(tile_id);
        glues[side_index(side).expect("Side must be NESW")]
    }

    /// Get the glues, with a glue being replaced with 0 if there is a blocker
    pub fn get_tile_unblocked_glues(&self, tile_id: TileState) -> Vec<Glue> {
        // This MUST be exactly 4 length
        let row = self.get_tile_raw_glues(tile_id);
        let mut glues = vec![0; 4];
        for s in ALL_SIDES {
            if !tile_id.is_blocked(s) {
                let i = side_index(s).unwrap() as usize;
                glues[i] = row[i];
            }
        }
        glues
    }

    pub fn fill_friends(&mut self) {
        let len = self.glue_names.len();
        let empty_friends = vec![HashSetType::<TileState>::default(); len];

        let (mut nf, mut sf, mut ef, mut wf) = (
            empty_friends.clone(),
            empty_friends.clone(),
            empty_friends.clone(),
            empty_friends,
        );

        /*
         * For this i will use the same glue standard as in SDC1D, that is:
         *  0 <-> Nothing
         *  1 <-> 2
         *  ...
         * */

        let err_message = "Vector shouldnt have empty index, as it was pre-initialized";
        for (id, [ng, eg, sg, wg]) in self.tile_glues.iter().enumerate() {
            // Add 4 zeros to the binary representation of the number
            let base_id = TileType(id).unblocked();

            if ng != &0 {
                sf.get_mut(glue_inverse(*ng))
                    .expect(err_message)
                    .insert(base_id);
            }
            if sg != &0 {
                nf.get_mut(glue_inverse(*sg))
                    .expect(err_message)
                    .insert(base_id);
            }
            if wg != &0 {
                ef.get_mut(glue_inverse(*wg))
                    .expect(err_message)
                    .insert(base_id);
            }
            if eg != &0 {
                wf.get_mut(glue_inverse(*eg))
                    .expect(err_message)
                    .insert(base_id);
            }
        }
        self.north_friends = nf;
        self.east_friends = ef;
        self.south_friends = sf;
        self.west_friends = wf;
    }

    fn energy_blocker_mut(&mut self, tile: TileType, side: usize) -> &mut KcalPerMol {
        &mut self.energy_blocker[(tile.0, side)]
    }

    fn get_glue_link(&self, glue1: Glue, glue2: Glue) -> KcalPerMol {
        self.glue_links[(glue1, glue2)]
    }

    pub fn fill_energy_blockers(&mut self) {
        let tile_ids = self.tile_names().len();
        for t in (0..tile_ids).map(TileType) {
            let uc = t.unblocked();
            let (tn, te, ts, tw) = (
                self.glue_on_side(NORTH, uc),
                self.glue_on_side(EAST, uc),
                self.glue_on_side(SOUTH, uc),
                self.glue_on_side(WEST, uc),
            );
            *self.energy_blocker_mut(t, 0) = self.get_glue_link(tn, glue_inverse(tn));
            *self.energy_blocker_mut(t, 1) = self.get_glue_link(te, glue_inverse(te));
            *self.energy_blocker_mut(t, 2) = self.get_glue_link(ts, glue_inverse(ts));
            *self.energy_blocker_mut(t, 3) = self.get_glue_link(tw, glue_inverse(tw));
        }
    }

    /// Fill energy_ns, energy_we: Array2<Energy>
    ///
    /// This will mutate the structure
    pub fn fill_energy_pairs(&mut self) {
        // The ids of the tiles with no blockers on any of their sides
        //
        // We will assume that tiles are unblocked when getting their energy
        // this check is done in the energy_to function
        let tile_ids = self.tile_names().len();

        for t1 in (0..tile_ids).map(TileType) {
            let t1_tile_id = t1.unblocked();

            // Glues on the sides of tile 1
            let (t1n, t1e, t1s, t1w) = (
                self.glue_on_side(NORTH, t1_tile_id),
                self.glue_on_side(EAST, t1_tile_id),
                self.glue_on_side(SOUTH, t1_tile_id),
                self.glue_on_side(WEST, t1_tile_id),
            );

            for t2 in (0..tile_ids).map(TileType) {
                let t2_tile_id = t2.unblocked();
                let (t2n, t2e, t2s, t2w) = (
                    self.glue_on_side(NORTH, t2_tile_id),
                    self.glue_on_side(EAST, t2_tile_id),
                    self.glue_on_side(SOUTH, t2_tile_id),
                    self.glue_on_side(WEST, t2_tile_id),
                );

                // t1 -- t2
                self.energy_we[(t1.0, t2.0)] = self.glue_links[(t1e, t2w)];

                // t2 -- t1
                self.energy_we[(t2.0, t1.0)] = self.glue_links[(t2e, t1w)];

                // t1
                // ||
                // t2
                self.energy_ns[(t1.0, t2.0)] = self.glue_links[(t1s, t2n)];

                // t2
                // ||
                // t1
                self.energy_ns[(t2.0, t1.0)] = self.glue_links[(t2s, t1n)];
            }
        }
    }

    pub fn fill_free_blocker_concentrations(&mut self) {
        self.free_blocker_concentrations
            .indexed_iter_mut()
            .for_each(|(gi, free_blocker_conc)| {
                let total_conc_of_tile_glue_usage = self
                    .tile_concentration
                    .iter()
                    .enumerate()
                    .map(|(ti, &c)| {
                        self.tile_glues[ti]
                            .iter()
                            .map(|&g| if g == gi { c } else { Molar::zero() })
                            .sum::<Molar>()
                    })
                    .sum::<Molar>();
                let total_blocker_conc = self.blocker_concentrations[gi];

                let cov_dg = self.glue_links[(gi, glue_inverse(gi))] + self.blocker_energy_adj; // TODO: better adj implementation
                let cov_bdg = cov_dg.times_beta(self.temperature);
                let ebdg = Molar::new(cov_bdg.exp());

                *free_blocker_conc = 0.5
                    * (total_blocker_conc - total_conc_of_tile_glue_usage - ebdg
                        + ((total_conc_of_tile_glue_usage - total_blocker_conc + ebdg).squared()
                            + 4.0 * total_blocker_conc * ebdg)
                            .sqrt());
            });
    }

    /// Add seed to system
    pub fn add_seed<S: State>(&mut self, state: &mut S, seed: HashMap<PointSafe2, TileState>) {
        self.seed = seed;
        for (point, tile) in &self.seed {
            state.set_sa(point, &tile.0)
        }
    }

    pub fn is_seed(&self, point: &PointSafe2) -> bool {
        self.seed.contains_key(point)
    }

    /// SIDE here must be NESW
    fn energy_to(&self, side: Sides, tile1: TileState, tile2: TileState) -> KcalPerMol {
        // If we are blocked on the sticking side, or the other tile has a blocker, then we
        // have no binding energy
        if tile1.is_blocked(side) || tile2.is_blocked(inverse(side)) {
            return KcalPerMol::zero();
        }

        // Ignore blockers
        let (tile1, tile2) = (tile_index(tile1), tile_index(tile2));

        // Now we know that neither the tile, nor the one were attaching to is blocked
        match side {
            NORTH => self.energy_ns[(tile2.0, tile1.0)],
            EAST => self.energy_we[(tile1.0, tile2.0)],
            SOUTH => self.energy_ns[(tile1.0, tile2.0)],
            WEST => self.energy_we[(tile2.0, tile1.0)],
            _ => panic!("Must enter NESW"),
        }
    }

    /// Energy of neighbour bonds
    fn energy_at_point<S: State>(&self, state: &S, point: PointSafe2) -> KcalPerMol {
        let tile_id: TileState = state.tile_at_point(point).into();
        let mut energy = KcalPerMol::zero();
        let mut n_bonds = 0;
        for side in ALL_SIDES {
            let neighbour_tile = Self::tile_to_side(state, side, point);
            let se = self.energy_to(side, tile_id, neighbour_tile);
            if !se.is_zero() {
                energy += se;
                n_bonds += 1;
            }
        }
        energy - (self.ds_lat * self.temperature) * (n_bonds - 1).max(0)
    }

    fn tile_detachment_rate<S: State>(&self, state: &S, p: PointSafe2) -> PerSecond {
        if self.is_seed(&p) {
            return PerSecond::zero();
        }

        let tile = state.tile_at_point(p);
        // If there is no tile, then nothing to attach
        if tile == 0 {
            return PerSecond::zero();
        }
        let energy_with_neighbours = self.energy_at_point(state, p);
        self.kf * Molar::u0_times((energy_with_neighbours.times_beta(self.temperature)).exp())
    }

    /// The rate at which a tile will attach somewhere
    pub fn tile_attachment_rate(&self, tile: TileState) -> PerSecond {
        self.kf * self.tile_concentration(tile)
    }

    #[cfg(feature = "python")]
    fn blocker_attachment_rate_at_side(&self, side: Sides, tile: TileState) -> PerSecond {
        self.kf * self.free_blocker_concentrations[self.glue_on_side(side, tile)]
    }

    /// Get the energy between a tile and a blocker to some side
    fn blocker_detachment_rate_at_side(&self, side: Sides, tile: TileState) -> PerSecond {
        // If there is no blocker in that side, then the detachment rate will be 0
        if !tile.is_blocked(side) {
            return PerSecond::zero();
        };

        let tile = tile.unblock_all();
        self.kf
            * Molar::u0_times(
                (self.energy_blocker[(
                    tile_index(tile).0,
                    side_index(side).expect("Side must be NESW"),
                )] + self.blocker_energy_adj)
                    .times_beta(self.temperature)
                    .exp(),
            )
    }

    fn blocker_detachment_total_rate(&self, tile: TileState) -> PerSecond {
        self.blocker_detachment_rate_at_side(NORTH, tile)
            + self.blocker_detachment_rate_at_side(EAST, tile)
            + self.blocker_detachment_rate_at_side(SOUTH, tile)
            + self.blocker_detachment_rate_at_side(WEST, tile)
    }

    fn maybe_detach_blocker_on_side_event(
        &self,
        tile_state: TileState,
        point: PointSafe2,
        side: Sides,
        acc: &mut PerSecond,
    ) -> Option<(bool, PerSecond, Event)> {
        // Something cannot detach if there is no blocker
        if !tile_state.is_blocked(side) {
            return None;
        }
        *acc -= self.blocker_detachment_rate_at_side(side, tile_state);
        if *acc <= PerSecond::zero() {
            // ^ SIDE will change the bit from 1 to 0, so no longer have a blocker here
            Some((
                true,
                *acc,
                Event::MonomerChange(point, tile_state.detach_blockers(side).into()),
            ))
        } else {
            None
        }
    }

    /// Detach a blocker from tile
    fn event_blocker_detachment<S: State>(
        &self,
        state: &S,
        point: PointSafe2,
        acc: &mut PerSecond,
    ) -> (bool, PerSecond, Event) {
        // Check what blockers the tile has
        let tile = TileState(state.tile_at_point(point));
        if tile.is_null() {
            return (false, *acc, Event::None);
        }

        // Update the acc for each side, if there is no blocker, then None will be returned, if no
        // evene takes place, then acc is updated, and none is returned.
        self.maybe_detach_blocker_on_side_event(tile, point, NORTH, acc)
            .or(self.maybe_detach_blocker_on_side_event(tile, point, EAST, acc))
            .or(self.maybe_detach_blocker_on_side_event(tile, point, SOUTH, acc))
            .or(self.maybe_detach_blocker_on_side_event(tile, point, WEST, acc))
            .unwrap_or((false, *acc, Event::None))
    }

    fn tile_to_side<S: State>(state: &S, side: Sides, p: PointSafe2) -> TileState {
        match side {
            NORTH => state.tile_to_n(p).into(),
            EAST => state.tile_to_e(p).into(),
            SOUTH => state.tile_to_s(p).into(),
            WEST => state.tile_to_w(p).into(),
            _ => panic!("Side must be North, South, East, or West"),
        }
    }

    fn maybe_attach_blocker_on_side_event<S: State>(
        &self,
        tileid: TileState,
        side: Sides,
        point: PointSafe2,
        state: &S,
        acc: &mut PerSecond,
    ) -> Option<(bool, PerSecond, Event)> {
        // A blocker cannot attach to a side with a blocker already attached
        if tileid.is_blocked(side)
        // If a tile is already attached to that side, then nothing can attach
            || !Self::tile_to_side(state,side, point).is_null()
        {
            return None;
        }

        *acc -= self.kf * self.free_blocker_concentrations[self.glue_on_side(side, tileid)];
        if *acc <= PerSecond::zero() {
            // | SIDE will change the bit from 0 to 1
            Some((
                true,
                *acc,
                Event::MonomerChange(point, tileid.attach_blockers(side).into()),
            ))
        } else {
            None
        }
    }

    // Attach a blocker to a tile
    fn event_blocker_attachment<S: State>(
        &self,
        state: &S,
        point: PointSafe2,
        acc: &mut PerSecond,
    ) -> (bool, PerSecond, Event) {
        let tile = TileState(state.tile_at_point(point));
        if tile.is_null() {
            return (false, PerSecond::zero(), Event::None);
        }
        self.maybe_attach_blocker_on_side_event(tile, NORTH, point, state, acc)
            .or(self.maybe_attach_blocker_on_side_event(tile, EAST, point, state, acc))
            .or(self.maybe_attach_blocker_on_side_event(tile, SOUTH, point, state, acc))
            .or(self.maybe_attach_blocker_on_side_event(tile, WEST, point, state, acc))
            .unwrap_or((false, *acc, Event::None))
    }

    // TODO: Handle Fission Here
    fn event_monomer_detachment<S: State>(
        &self,
        state: &S,
        point: PointSafe2,
        acc: &mut PerSecond,
    ) -> (bool, PerSecond, Event) {
        *acc -= self.tile_detachment_rate(state, point);
        if *acc > PerSecond::zero() {
            return (false, *acc, Event::None);
        }

        match self.fission_handling {
            FissionHandling::NoFission => (true, *acc, Event::None),
            FissionHandling::JustDetach => {
                if self.is_seed(&point) {
                    (true, *acc, Event::None)
                } else {
                    (true, *acc, Event::MonomerDetachment(point))
                }
            }
            FissionHandling::KeepSeeded => {
                let mut remove = self.unseeded(state, point);
                remove.push(point);
                (true, *acc, Event::PolymerDetachment(remove))
            }
            _ => panic!("Only NoFission, and JustDetach are supported"),
        }
    }

    /// Get all possible blocker combinations of some tile, such that some side is unblocked
    pub fn blocker_combinations(unblocked_side: Sides, tile: TileState) -> Vec<TileState> {
        (0..16)
            .filter_map(|blocker| {
                if blocker & unblocked_side != 0 {
                    None
                } else {
                    Some(tile.attach_blockers(blocker))
                }
            })
            .collect()
    }

    /// Get all possible tiles that may attach at some given point
    fn possible_tiles_at_point<S: State>(
        &self,
        state: &S,
        point: PointSafe2,
    ) -> HashSetType<TileState> {
        let tile: TileState = state.tile_at_point(point).into();
        let mut friends: HashSetType<TileState> = HashSet::default();

        // tile aready attached here
        if !tile.is_null() {
            return friends;
        }

        for side in ALL_SIDES {
            let neighbour = Self::tile_to_side(state, side, point);
            if neighbour.is_null() {
                continue;
            }

            if self.no_partially_blocked_attachments && neighbour.is_blocked(inverse(side)) {
                return HashSet::default();
            }

            if let Some(possible_attachments) =
                self.get_unblocked_friends_to_side(inverse(side), neighbour)
            {
                let attachments: HashSetType<TileState> = HashSet::from_iter(
                    possible_attachments
                        .iter()
                        .flat_map(|&tile| Self::blocker_combinations(side, tile)),
                );
                friends.extend(attachments);
            }
        }

        if self.no_partially_blocked_attachments {
            friends.retain(|tile| {
                let mut blocked = false;
                for side in ALL_SIDES {
                    if tile.is_blocked(side) && !Self::tile_to_side(state, side, point).is_null() {
                        blocked = true;
                        break;
                    }
                }
                !blocked
            });
        }

        friends
    }

    fn total_attachment_rate_at_point<S: State>(&self, point: PointSafe2, state: &S) -> PerSecond {
        self.possible_tiles_at_point(state, point)
            .iter()
            .fold(PerSecond::zero(), |acc, &tile| {
                acc + self.kf * self.tile_concentration(tile)
            })
    }

    /// Probability of any tile attaching at some point
    fn event_monomer_attachment<S: State>(
        &self,
        state: &S,
        point: PointSafe2,
        acc: &mut PerSecond,
    ) -> (bool, PerSecond, Event) {
        let tile = state.tile_at_point(point);
        // tile aready attached here
        if tile != 0 {
            return (false, *acc, Event::None);
        }

        let friends: HashSetType<TileState> = self.possible_tiles_at_point(state, point);
        // attachment_side is not used, but is relevant in computation, as it accounts for
        // duplicates (some tiles could bind to the north or east, so it should be taken into
        // account twice)
        for tile in friends {
            *acc -= self.kf * self.tile_concentration(tile);
            if *acc <= PerSecond::zero() {
                return (true, *acc, Event::MonomerAttachment(point, tile.into()));
            }
        }
        (false, *acc, Event::None)
    }

    /// Percentage of total concentration of some tile that has a blocker on a given side
    pub fn blocker_percentage(&self, side: Sides, tile: TileState) -> f64 {
        // let detachment_rate = self.blocker_detachment_rate_at_side(side, tile | side);
        // let attachment_rate = self.blocker_attachment_rate_at_side(side, tile);
        // attachment_rate / (attachment_rate + detachment_rate)
        // println!("tile: {}, side: {}", tile_index(tile), side);
        let tile = tile.unblock_all();
        let blocker_glue = self.glue_on_side(side, tile);
        if self.blocker_concentrations[blocker_glue].is_zero() {
            return 0.0;
        }
        let cov_dg =
            self.glue_links[(blocker_glue, glue_inverse(blocker_glue))] + self.blocker_energy_adj; // TODO: better adj implementation
        let cov_bdg = cov_dg.times_beta(self.temperature);
        let embdg = (-cov_bdg).exp();
        let b = self.free_blocker_concentrations[blocker_glue];
        1.0 - (1.0 + b.over_u0() * embdg).recip()
    }

    /// Get the concentration of a specific tile, with blocker as given in the TileId
    pub fn tile_concentration(&self, tile: TileState) -> Molar {
        let mut acc = 1.0;
        for side in ALL_SIDES {
            let blocker_perc = self.blocker_percentage(side, tile);
            if tile.is_blocked(side) {
                acc *= blocker_perc;
            } else {
                acc *= 1.0 - blocker_perc;
            }
        }
        self.tile_concentration[tile_index(tile).0] * acc
    }

    fn total_blocker_attachment_rate<S: State>(&self, state: &S, point: PointSafe2) -> PerSecond {
        // Check that there is a tile at this point
        let tile: TileState = state.tile_at_point(point).into();
        if tile.is_null() {
            return PerSecond::zero();
        }

        let mut rate = PerSecond::zero();
        for s in ALL_SIDES {
            if !tile.is_blocked(s) && Self::tile_to_side(state, s, point).is_null() {
                rate += self.kf * self.free_blocker_concentrations[self.glue_on_side(s, tile)];
            }
        }
        rate
    }

    fn can_bond(&self, tile1: TileState, side: Sides, tile2: TileState) -> bool {
        let g1 = self.glue_on_side(side, tile1);
        let g2 = self.glue_on_side(inverse(side), tile2);
        g1 == glue_inverse(g2)
    }

    /// BFS to see where we can get without passing through the `avoid` point
    fn bfs<S: State>(
        &self,
        state: &S,
        point: PointSafe2,
        avoid: PointSafe2,
    ) -> HashSet<PointSafe2> {
        let mut visited = HashSet::new();
        let mut stack = vec![point];
        while let Some(head) = stack.pop() {
            let head_tile = state.tile_at_point(head);

            // We have already processed this node, or we dont have to at all
            if head_tile == 0 || visited.contains(&head) {
                continue;
            }
            visited.insert(head);

            [
                (NORTH, state.move_sa_n(head)),
                (EAST, state.move_sa_e(head)),
                (SOUTH, state.move_sa_s(head)),
                (WEST, state.move_sa_w(head)),
            ]
            .iter()
            .for_each(|(side, PointSafeHere(x))| {
                let p = PointSafe2(*x);
                if p == avoid {
                    return;
                }

                let neighbour_tile = state.tile_at_point(PointSafe2(*x));
                if self.can_bond(head_tile.into(), *side, neighbour_tile.into()) {
                    stack.push(p);
                }
            });
        }
        visited
    }

    fn unseeded<S: State>(&self, state: &S, point: PointSafe2) -> Vec<PointSafe2> {
        let seed = self
            .seed_locs()
            .first()
            .expect("Must have a seed to use KeepSeed")
            .0;

        [
            state.move_sa_n(point),
            state.move_sa_e(point),
            state.move_sa_s(point),
            state.move_sa_w(point),
        ]
        .iter()
        .fold(
            (HashSet::<PointSafe2>::new(), Vec::new()),
            |(mut acc, mut unseeded), neighbour| {
                let neighbour = PointSafe2(neighbour.0);
                if acc.contains(&neighbour) {
                    return (acc, unseeded);
                }
                let b = self.bfs(state, neighbour, point);
                acc.extend(&b);
                if !b.contains(&seed) {
                    unseeded.extend(&b);
                }
                (acc, unseeded)
            },
        )
        .1
    }
}

/*
* The idea right now is that:
* 1. All tiles have a different id
* 2. If a tile has gets a blocker attachment / detachment,
*    then it becomes a new tile
*    That is to say if two tile A could become tile B by
*    attaching / detaching blockers, then they are different
*    tiles (with different ids), but they have the same base
*    id.
* */

impl TileBondInfo for KBlock {
    fn tile_color(&self, tileid: u32) -> [u8; 4] {
        self.tile_colors[usize::from(tile_index(tileid.into()))]
    }

    fn tile_name(&self, tileid: u32) -> &str {
        self.tile_names[usize::from(tile_index(tileid.into()))].as_str()
    }

    fn bond_name(&self, bond_number: usize) -> &str {
        &self.glue_names[bond_number]
    }

    fn tile_colors(&self) -> &Vec<[u8; 4]> {
        &self.tile_colors
    }

    fn tile_names(&self) -> Vec<&str> {
        self.tile_names.iter().map(|s| s.as_str()).collect()
    }

    fn bond_names(&self) -> Vec<&str> {
        self.glue_names.iter().map(|s| s.as_str()).collect()
    }
}

impl System for KBlock {
    fn system_info(&self) -> String {
        format!("{self:?}")
    }

    fn perform_event<St: State>(&self, state: &mut St, event: &Event) -> f64 {
        match event {
            Event::None => panic!("Canot perform None event"),
            Event::MonomerDetachment(point) => state.set_sa(point, &0),
            // Monomer Change -- blockers
            Event::MonomerChange(point, tile) | Event::MonomerAttachment(point, tile) => {
                state.set_sa(point, tile)
            }
            Event::PolymerAttachment(points) | Event::PolymerChange(points) => {
                for (point, tile) in points.iter() {
                    state.set_sa(point, tile);
                }
            }
            Event::PolymerDetachment(points) => {
                for point in points.iter() {
                    state.set_sa(point, &0);
                }
            }
        };
        f64::NAN // FIXME: should return the energy change
    }

    fn update_after_event<St: crate::state::State>(
        &self,
        state: &mut St,
        event: &crate::system::Event,
    ) {
        match event {
            Event::None => panic!("Canot perform None event"),
            Event::MonomerAttachment(point, _)
            | Event::MonomerDetachment(point)
            | Event::MonomerChange(point, _) => {
                let points = [
                    state.move_sa_n(*point),
                    state.move_sa_w(*point),
                    PointSafeHere(point.0),
                    state.move_sa_e(*point),
                    state.move_sa_s(*point),
                ];
                self.update_points(state, &points);
            }
            Event::PolymerDetachment(v) => {
                let mut points = Vec::new();
                for p in v {
                    points.extend_from_slice(&[
                        // Single moves (no dimer chunks, no duples)
                        state.move_sa_n(*p),
                        state.move_sa_w(*p),
                        PointSafeHere(p.0),
                        state.move_sa_e(*p),
                        state.move_sa_s(*p),
                    ]);
                }
                points.sort_unstable();
                points.dedup();
                self.update_points(state, &points);
            }
            Event::PolymerAttachment(t) | Event::PolymerChange(t) => {
                let mut points = Vec::new();
                for (p, _) in t {
                    points.extend_from_slice(&[
                        state.move_sa_n(*p),
                        state.move_sa_w(*p),
                        PointSafeHere(p.0),
                        state.move_sa_e(*p),
                        state.move_sa_s(*p),
                    ]);
                }
                points.sort_unstable();
                points.dedup();
                self.update_points(state, &points);
            }
        }
    }

    fn event_rate_at_point<S: crate::state::State>(
        &self,
        state: &S,
        p: crate::canvas::PointSafeHere,
    ) -> PerSecond {
        let p = if state.inbounds(p.0) {
            PointSafe2(p.0)
        } else {
            return PerSecond::zero();
        };
        let tile = { state.tile_at_point(p) };
        if tile != 0 {
            self.tile_detachment_rate(state, p)
                + self.blocker_detachment_total_rate(tile.into())
                + self.total_blocker_attachment_rate(state, p)
        } else {
            self.total_attachment_rate_at_point(p, state)
        }
    }

    /// Fixme: currently just ignores rate
    fn choose_event_at_point<St: crate::state::State>(
        &self,
        state: &St,
        point: crate::canvas::PointSafe2,
        acc: PerSecond,
    ) -> (crate::system::Event, f64) {
        let mut acc = acc;

        if let (true, _, event) = self.event_monomer_detachment(state, point, &mut acc) {
            return (event, f64::NAN);
        };
        if let (true, _, event) = self.event_monomer_attachment(state, point, &mut acc) {
            return (event, f64::NAN);
        }
        if let (true, _, event) = self.event_blocker_attachment(state, point, &mut acc) {
            return (event, f64::NAN);
        }
        if let (true, _, event) = self.event_blocker_detachment(state, point, &mut acc) {
            return (event, f64::NAN);
        }

        panic!(
            "Rate: {:?}, {:?}, {:?}, {:?}",
            acc,
            point,
            state,
            state.raw_array()
        )
    }

    fn seed_locs(&self) -> Vec<(crate::canvas::PointSafe2, crate::base::Tile)> {
        self.seed
            .clone()
            .into_iter()
            .map(|(p, t)| (p, t.into()))
            .collect::<Vec<_>>()
    }

    fn calc_mismatch_locations<St: crate::state::State>(&self, state: &St) -> Array2<usize> {
        // cge: Roughly copied from kTAM, because I need this for playing with
        // some algorithmic self-assembly stuff.

        let threshold = KcalPerMol(-0.05); // Todo: fix this (note for kCov, energies are negative)
        let mut mismatch_locations = Array2::<usize>::zeros((state.nrows(), state.ncols()));

        // TODO: this should use an iterator from the canvas, which we should implement.
        for i in 0..state.nrows() {
            for j in 0..state.ncols() {
                if !state.inbounds((i, j)) {
                    continue;
                }
                let p = PointSafe2((i, j));

                let t: TileState = state.tile_at_point(p).into();

                if t.is_null() {
                    continue;
                }

                let tn: TileState = state.tile_to_n(p).into();
                let te: TileState = state.tile_to_e(p).into();
                let ts: TileState = state.tile_to_s(p).into();
                let tw: TileState = state.tile_to_w(p).into();

                let mm_n = ((!tn.is_null()) & (self.energy_to(NORTH, t, tn) > threshold)) as usize;
                let mm_e = ((!te.is_null()) & (self.energy_to(EAST, t, te) > threshold)) as usize;
                let mm_s = ((!ts.is_null()) & (self.energy_to(SOUTH, t, ts) > threshold)) as usize;
                let mm_w = ((!tw.is_null()) & (self.energy_to(WEST, t, tw) > threshold)) as usize;

                mismatch_locations[(i, j)] = 8 * mm_n + 4 * mm_e + 2 * mm_s + mm_w;
            }
        }

        mismatch_locations
    }

    fn calc_dimers(&self) -> Result<Vec<DimerInfo>, GrowError> {
        let mut dvec = Vec::new();

        for (t1, _) in self.tile_concentration.iter().enumerate() {
            let t1: TileState = TileType(t1).unblocked();
            if let Some(friends) = self.get_unblocked_friends_to_side(EAST, t1) {
                for t2 in friends.iter() {
                    let biconc = self.tile_concentration(t1) * self.tile_concentration(*t2);
                    dvec.push(DimerInfo {
                        t1: t1.into(),
                        t2: (*t2).into(),
                        orientation: Orientation::WE,
                        formation_rate: self.kf * biconc,
                        equilibrium_conc: biconc.over_u0()
                            * (self.energy_we[(tile_index(t1).into(), tile_index(*t2).into())]
                                .times_beta(self.temperature))
                            .exp(),
                    });
                }
            }
            if let Some(friends) = self.get_unblocked_friends_to_side(SOUTH, t1) {
                for t2 in friends.iter() {
                    let biconc = self.tile_concentration(t1) * self.tile_concentration(*t2);
                    dvec.push(DimerInfo {
                        t1: t1.into(),
                        t2: (*t2).into(),
                        orientation: Orientation::NS,
                        formation_rate: self.kf * biconc,
                        equilibrium_conc: biconc.over_u0()
                            * (self.energy_ns[(tile_index(t1).into(), tile_index(*t2).into())]
                                .times_beta(self.temperature))
                            .exp(),
                    });
                }
            }
        }

        Ok(dvec)
    }

    fn set_param(
        &mut self,
        name: &str,
        value: Box<dyn std::any::Any>,
    ) -> Result<crate::system::NeededUpdate, crate::base::GrowError> {
        match name {
            "temperature" => {
                let temp = value
                    .downcast_ref::<f64>()
                    .ok_or(crate::base::GrowError::WrongParameterType(name.to_string()))?;
                self.temperature = Celsius(*temp);
                self.update();
                Ok(crate::system::NeededUpdate::NonZero)
            }
            "kf" => {
                let kf = value
                    .downcast_ref::<f64>()
                    .ok_or(crate::base::GrowError::WrongParameterType(name.to_string()))?;
                self.kf = PerMolarSecond::from(*kf);
                self.update();
                Ok(crate::system::NeededUpdate::NonZero)
            }
            "ds_lat" => {
                let ds_lat = value
                    .downcast_ref::<f64>()
                    .ok_or(crate::base::GrowError::WrongParameterType(name.to_string()))?;
                self.ds_lat = KcalPerMolKelvin::from(*ds_lat);
                self.update();
                Ok(crate::system::NeededUpdate::NonZero)
            }
            _ => Err(crate::base::GrowError::NoParameter(name.to_string())),
        }
    }

    fn get_param(&self, name: &str) -> Result<Box<dyn std::any::Any>, crate::base::GrowError> {
        match name {
            "temperature" => Ok(Box::new(f64::from(self.temperature))),
            "kf" => Ok(Box::new(f64::from(self.kf))),
            "ds_lat" => Ok(Box::new(f64::from(self.ds_lat))),
            _ => Err(crate::base::GrowError::NoParameter(name.to_string())),
        }
    }

    fn list_parameters(&self) -> Vec<crate::system::ParameterInfo> {
        use crate::system::ParameterInfo;
        vec![
            ParameterInfo {
                name: "temperature".to_string(),
                units: "°C".to_string(),
                default_increment: 1.0,
                min_value: Some(0.0),
                max_value: Some(100.0),
                description: Some("Simulation temperature".to_string()),
                current_value: f64::from(self.temperature),
            },
            ParameterInfo {
                name: "kf".to_string(),
                units: "M/s".to_string(),
                default_increment: 1e5,
                min_value: Some(0.0),
                max_value: None,
                description: Some("Rate constant for monomer attachment events".to_string()),
                current_value: f64::from(self.kf),
            },
            ParameterInfo {
                name: "ds_lat".to_string(),
                units: "kcal/(mol·K)".to_string(),
                default_increment: 1.0,
                min_value: None,
                max_value: None,
                description: Some("Lateral entropy change".to_string()),
                current_value: f64::from(self.ds_lat),
            },
        ]
    }
}

#[cfg(test)]
mod test_covtile {
    use crate::models::kblock::{tile_index, TileState, EAST, NORTH, WEST};

    #[test]
    fn get_ids() {
        let mut t = TileState(0b10110000);
        t = t.attach_blockers(EAST);
        assert_eq!(t.unblock_all(), TileState(0b10110000));
        assert_eq!(t, TileState(0b10110000 | EAST));

        let mut k = TileState(0b10000);
        k = k.attach_blockers(EAST);
        k = k.attach_blockers(WEST);
        assert_eq!(k.unblock_all().0, 16);
    }

    #[test]
    fn test_tile_index() {
        for i in 0..16 {
            let x = TileState(0b10000);
            assert_eq!(1, usize::from(tile_index(x.attach_blockers(i))))
        }
    }

    #[test]
    fn is_blocked_side() {
        assert!(TileState(NORTH).is_blocked(NORTH));
        assert!(TileState((123 << 4) | NORTH).is_blocked(NORTH));
        assert!(!TileState((123 << 4) | NORTH).is_blocked(EAST));
    }

    #[test]
    fn detach_side() {
        assert_eq!(TileState(0), TileState(NORTH).detach_blockers(NORTH));
        assert_eq!(
            TileState(123 << 4),
            TileState((123 << 4) | NORTH).detach_blockers(NORTH)
        );
    }
}

#[cfg(test)]
mod test_kblock {
    use crate::{
        state::StateEnum,
        tileset::{CanvasType, TrackingType},
    };

    use super::*;
    use ndarray::array;

    fn sample_kblock() -> KBlock {
        const DEFAULT_COLOR: [u8; 4] = [0, 0, 0, 0];
        let tile_glues = Array1::from_vec(vec![
            [0, 0, 0, 0], // zero tile -- Empty
            // N E S W
            [1, 0, 0, 0], // f
            [0, 3, 2, 0], // s
            [0, 0, 0, 4], // t
        ]);

        // The example tiles may attach like this:
        //  s t
        //  f

        let glue_linkns = array![
            //0   1   2   3  4
            [0., 0., 0., 0., 0.], // 0
            [0., 0., 1., 0., 0.], // 1
            [0., 1., 0., 0., 0.], // 2
            [0., 0., 0., 0., 1.], // 3
            [0., 0., 0., 1., 0.], // 4
        ];

        {
            let tile_names = vec![
                "null".to_string(),
                "f".to_string(),
                "s".to_string(),
                "t".to_string(),
            ];
            let tile_concentration = &[1.0, 1.0, 1.0, 1.0];
            let tile_colors = vec![DEFAULT_COLOR; 4];
            let glue_names = vec![
                "null".to_string(),
                "1".to_string(),
                "2".to_string(),
                "3".to_string(),
                "4".to_string(),
            ];
            let blocker_concentrations = vec![0., 1., 1., 1., 1.];
            let seed = HashMap::default();
            let kf = PerMolarSecond::new(1e6);
            let fission_handling = FissionHandling::JustDetach;
            let tilecount = tile_names.len();
            let mut s = KBlock {
                tile_names,
                tile_concentration: tile_concentration.iter().map(|c| (*c).into()).collect(),
                tile_colors,
                glue_names,
                blocker_concentrations: blocker_concentrations
                    .iter()
                    .map(|c| (*c).into())
                    .collect(),
                tile_glues,
                glue_links: glue_linkns.mapv(|x| x.into()),
                temperature: Celsius(60.0),
                seed,
                north_friends: Vec::default(),
                south_friends: Vec::default(),
                east_friends: Vec::default(),
                west_friends: Vec::default(),
                energy_ns: Array2::zeros((tilecount, tilecount)),
                energy_we: Array2::zeros((tilecount, tilecount)),
                energy_blocker: Array2::default((tilecount, 4)),
                ds_lat: 0.0.into(),
                kf,
                fission_handling,
                no_partially_blocked_attachments: false,
                free_blocker_concentrations: Array1::from_vec(
                    blocker_concentrations
                        .into_iter()
                        .map(|c| c.into())
                        .collect(),
                ),
                blocker_energy_adj: 0.0.into(),
            };
            s.fill_friends();
            s.update();
            s
        }
    }

    #[test]
    fn glue_side() {
        let kdcov = sample_kblock();
        assert_eq!(kdcov.glue_on_side(NORTH, TileState(1 << 4)), 1);
        assert_eq!(kdcov.glue_on_side(SOUTH, TileState(1 << 4)), 0);
        assert_eq!(kdcov.glue_on_side(WEST, TileState(3 << 4)), 4);
    }

    #[test]
    fn friends_build() {
        let mut kdcov = sample_kblock();
        //println!("Tile Names: {:?}", kdcov.tile_names());
        kdcov.fill_friends();

        //println!("Tile Names: {:?}", kdcov.tile_names);
        println!("N: {:?}", kdcov.north_friends);
        //println!("S: {:?}", kdcov.south_friends);
        //println!("E: {:?}", kdcov.east_friends);
        //println!("W: {:?}", kdcov.west_friends);

        let mut expected_nf = HashSetType::default();

        expected_nf.insert(TileState(2 << 4));
        // This is a little strange to use, as you need to know the glue on the north side of the
        // tile.
        assert_eq!(kdcov.north_friends[1], expected_nf);
        // These helper methods make it so that you can find every tile that can bond to the north
        // of some tile id
        assert_eq!(
            kdcov.get_unblocked_friends_to_side(NORTH, TileState(1 << 4)),
            Some(&expected_nf)
        );
        assert_eq!(kdcov.get_friends(NORTH, TileState(1 << 4)), expected_nf);
        // You can also get frineds to multiple sides at once
        assert_eq!(
            kdcov.get_friends(NORTH | EAST, TileState(1 << 4)),
            expected_nf
        );

        let mut expected_wf = HashSetType::default();
        expected_wf.insert(TileState(2));
        assert_eq!(kdcov.west_friends[4], expected_nf);
        assert_eq!(kdcov.get_friends(WEST, TileState(3 << 4)), expected_nf);
    }

    #[test]
    fn test_bfs() {
        use crate::canvas::Canvas;
        let tile_a = KBlockTile {
            name: "TileA".to_string(),
            concentration: 1e-2,
            glues: ["A", "A", "A*", "A*"].map(String::from),
            color: [0, 0, 0, 0],
        };
        let mut kblock: KBlock = KBlockParams {
            tiles: vec![tile_a],
            blocker_conc: HashMap::from([
                (GlueIdentifier::Index(0), 1e6.into()),
                (GlueIdentifier::Index(1), 1e6.into()),
            ]),
            ds_lat: KcalPerMolKelvin::new(1.0),
            kf: PerMolarSecond::new(1.0),
            temp: Celsius::new(40.0),
            ..Default::default()
        }
        .into();

        let mut se =
            StateEnum::empty((20, 20), CanvasType::Square, TrackingType::None, 40).unwrap();
        // Add a seed tile (just one)
        kblock.add_seed(
            &mut se,
            HashMap::from([(PointSafe2((2, 2)), TileState(1 << 4))]),
        );

        // Make an L
        se.set_sa(&PointSafe2((3, 2)), &(1 << 4));
        se.set_sa(&PointSafe2((4, 2)), &(1 << 4));
        se.set_sa(&PointSafe2((5, 2)), &(1 << 4));
        se.set_sa(&PointSafe2((6, 2)), &(1 << 4));
        se.set_sa(&PointSafe2((5, 3)), &(1 << 4));
        se.set_sa(&PointSafe2((5, 4)), &(1 << 4));

        let removals = kblock.unseeded(&se, PointSafe2((3, 2)));
        println!("{removals:?}");
        assert_eq!(removals.len(), 5);
        assert!(removals.contains(&PointSafe2((4, 2))));
        assert!(removals.contains(&PointSafe2((5, 2))));
        assert!(removals.contains(&PointSafe2((5, 3))));
        assert!(removals.contains(&PointSafe2((5, 4))));
        assert!(removals.contains(&PointSafe2((6, 2))));

        let removals = kblock.unseeded(&se, PointSafe2((5, 2)));
        assert_eq!(removals.len(), 3);
        assert!(removals.contains(&PointSafe2((5, 3))));
        assert!(removals.contains(&PointSafe2((5, 4))));
        assert!(removals.contains(&PointSafe2((6, 2))));
    }
}

// Python Bindings

#[derive(Debug, Clone, Serialize, Deserialize)]
struct KBlockTile {
    pub name: String,
    pub concentration: f64,
    /// Glues for the tiles North, East, South, West in that order
    pub glues: [String; 4],
    /// Color of the tile, this is used only when displaying
    pub color: [u8; 4],
}

#[cfg(feature = "python")]
impl pyo3::FromPyObject<'_> for KBlockTile {
    fn extract_bound(ob: &pyo3::Bound<'_, pyo3::PyAny>) -> pyo3::PyResult<Self> {
        use pyo3::prelude::*;

        let name: String = ob.getattr("name")?.extract()?;
        let concentration: f64 = ob.getattr("concentration")?.extract()?;
        let glues: [String; 4] = ob.getattr("glues")?.extract()?;

        // Try to extract color as an array first
        let color_result: Result<[u8; 4], _> = ob.getattr("color")?.extract();

        let color = match color_result {
            Ok(color_array) => color_array,
            Err(_) => {
                // If that fails, try to extract as a string and use get_color
                let color_str: String = ob.getattr("color")?.extract()?;
                crate::colors::get_color(&color_str)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?
            }
        };

        Ok(Self {
            name,
            concentration,
            glues,
            color,
        })
    }
}

impl KBlockTile {
    fn empty() -> Self {
        Self {
            name: "empty".to_string(),
            concentration: 0.0,
            glues: ["null"; 4].map(String::from),
            color: [0, 0, 0, 0],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", derive(pyo3::FromPyObject))]
enum StrenOrSeq {
    DG(KcalPerMol),
    Sequence(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "python", derive(pyo3::FromPyObject))]
pub enum TileIdentifier {
    Id(TileState),
    Name(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "python", derive(pyo3::FromPyObject))]
pub enum GlueIdentifier {
    Index(usize),
    Name(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "python", derive(pyo3::FromPyObject))]
struct KBlockParams {
    pub tiles: Vec<KBlockTile>,
    pub blocker_conc: HashMap<GlueIdentifier, Molar>,
    pub seed: HashMap<(usize, usize), TileIdentifier>,
    pub binding_strength: HashMap<String, StrenOrSeq>,
    pub ds_lat: KcalPerMolKelvin,
    pub kf: PerMolarSecond,
    pub temp: Celsius,
    pub no_partially_blocked_attachments: bool,
    pub blocker_energy_adj: KcalPerMol,
}

impl Default for KBlockParams {
    fn default() -> Self {
        Self {
            tiles: vec![],
            blocker_conc: HashMap::default(),
            seed: HashMap::default(),
            binding_strength: HashMap::default(),
            ds_lat: KcalPerMolKelvin::new(-14.12),
            kf: PerMolarSecond::new(1.0e6),
            temp: Celsius::new(40.0),
            no_partially_blocked_attachments: false,
            blocker_energy_adj: 0.0.into(),
        }
    }
}

/// Given some glue, in the form (a|z)+* or (a|z), return itself, as well as its inverse
fn base_inv(mut s: String) -> (String, String) {
    if s.ends_with("*") {
        s.pop();
    }
    (s.clone(), format!("{s}*"))
}

impl From<KBlockParams> for KBlock {
    fn from(mut value: KBlockParams) -> Self {
        let mut tiles = Vec::with_capacity(value.tiles.len() + 1);
        tiles.push(KBlockTile::empty());
        tiles.append(&mut value.tiles);
        let tile_names: Vec<String> = tiles.iter().map(|tile| tile.name.clone()).collect();
        let tile_concentration: Vec<f64> = tiles.iter().map(|tile| tile.concentration).collect();

        // Fixme: this should not be hard-coded
        let tile_colors = tiles
            .iter()
            .flat_map(|tile| {
                let mut colors = Vec::with_capacity(16);
                colors.push(tile.color);
                // Gray color: [128, 128, 128, 255]
                colors.extend(std::iter::repeat_n([128, 128, 128, 255], 15));
                colors
            })
            .collect::<Vec<_>>();

        let mut glues = tiles
            .iter()
            .flat_map(|tile| tile.glues.clone())
            .collect::<Vec<_>>();
        glues.dedup();

        let mut glue_id = 1;
        let mut glue_hashmap: HashMap<String, Glue> = HashMap::from([("null".to_string(), 0)]);
        for glue in glues.iter() {
            // We have already generated a key for this glue
            if glue_hashmap.contains_key(glue) {
                continue;
            }
            let (base, inverse) = base_inv(glue.clone());
            glue_hashmap.insert(base, glue_id);
            glue_hashmap.insert(inverse, glue_id + 1);
            glue_id += 2;
        }

        let tile_glues = tiles
            .iter()
            .map(|tile| tile.glues.clone().map(|x| *glue_hashmap.get(&x).unwrap()))
            .collect();

        // Generate blocker concentrations vector from HashMap
        let mut blocker_concentrations = vec![0.0; glue_id];
        for (glue_id, conc) in value.blocker_conc {
            match glue_id {
                GlueIdentifier::Index(index) => {
                    if index < blocker_concentrations.len() {
                        blocker_concentrations[index] = conc.into();
                    }
                }
                GlueIdentifier::Name(name) => {
                    if let Some(&index) = glue_hashmap.get(&name) {
                        blocker_concentrations[index] = conc.into();
                    }
                }
            }
        }

        // Process seed with either TileId or tile name
        let seed = value
            .seed
            .iter()
            .map(|(pos, tile_id_or_name)| {
                let tile_id = match tile_id_or_name {
                    TileIdentifier::Id(id) => *id,
                    TileIdentifier::Name(name) => {
                        // Find position in tile_names and convert to TileId
                        let pos = tiles
                            .iter()
                            .position(|t| t.name == *name)
                            .unwrap_or_else(|| panic!("Tile name '{name}' not found"));
                        TileType(pos).unblocked()
                    }
                };
                (PointSafe2(*pos), tile_id)
            })
            .collect();

        // Make sure that every glue has its inverse in the array
        let mut glue_links = Array2::zeros((glue_id + 1, glue_id + 1));
        for (glue, strength_new) in value.binding_strength {
            let (glue_str, inverse_str) = base_inv(glue);

            let msg = "Glue was assigned strength, but was not used";
            let glue = *glue_hashmap.get(&glue_str).expect(msg);
            let inverse = *glue_hashmap.get(&inverse_str).expect(msg);

            // Lets check that we are in bounds
            if glue_links.get((glue, inverse)).is_none() {
                panic!("({glue:?} {inverse:?}) not in index ({glue_id:?} {glue_id:?})");
            }

            let stren_dg = match strength_new {
                StrenOrSeq::DG(dg) => dg,
                StrenOrSeq::Sequence(seq) => {
                    // If we want annealing, we need to save the sequences, or store dh & ds
                    // instead of dg
                    // cge: it turns out I made some mistakes with alpha.  We need to use RT*alpha here, which messes
                    // up temperature dependence.  But that doesn't matter right now I suppose.
                    crate::utils::string_dna_delta_g(&seq, value.temp)
                }
            };

            glue_links[(glue, inverse)] = stren_dg;
            glue_links[(inverse, glue)] = stren_dg;
        }

        let mut glue_names = vec!["".to_string(); glue_id];
        for (name, id) in glue_hashmap {
            glue_names[id] = name;
        }

        {
            let temperature = value.temp;
            let kf = value.kf;
            let ds_lat = value.ds_lat;
            let fission_handling = FissionHandling::JustDetach;
            let no_partially_blocked_attachments = value.no_partially_blocked_attachments;
            let blocker_energy_adj = value.blocker_energy_adj;
            let tilecount = tile_names.len();
            let mut s = KBlock {
                tile_names,
                tile_concentration: tile_concentration.iter().map(|c| (*c).into()).collect(),
                tile_colors,
                glue_names,
                blocker_concentrations: blocker_concentrations
                    .iter()
                    .map(|c| (*c).into())
                    .collect(),
                tile_glues,
                glue_links,
                temperature,
                seed,
                north_friends: Vec::default(),
                south_friends: Vec::default(),
                east_friends: Vec::default(),
                west_friends: Vec::default(),
                energy_ns: Array2::zeros((tilecount, tilecount)),
                energy_we: Array2::zeros((tilecount, tilecount)),
                energy_blocker: Array2::default((tilecount, 4)),
                ds_lat,
                kf,
                fission_handling,
                no_partially_blocked_attachments,
                free_blocker_concentrations: Array1::from_vec(
                    blocker_concentrations
                        .into_iter()
                        .map(|c| c.into())
                        .collect(),
                ),
                blocker_energy_adj,
            };
            s.fill_friends();
            s.update();
            s
        }
    }
}

use paste::paste;

/// Generate the getter and the setter for a specific type
macro_rules! getset_single {
    ($model:ty, $name:ident, $t:ty) => {
        paste! {
            #[cfg(feature = "python")]
            #[pymethods]
            impl $model {
                #[getter($name)]
                fn [<py_get_ $name>](&self) -> $t {
                    self.$name.into()
                }

                #[setter($name)]
                fn [<py_set_ $name>](&mut self, to: $t) {
                    self.$name = to.into();
                    self.update();
                }
            }
        }
    };
}

/// Generate many getters and setters of different types
macro_rules! getset {
    ($model:ty, $(($t:ty => $($name:ident),+)),+) => {
        $($( getset_single!($model, $name, $t); )+)+
    };
}

getset!(KBlock,
    // f64 getters and setters
    (f64 => kf, temperature, ds_lat),
    // bool getters and setters
    (bool => no_partially_blocked_attachments)
);

#[cfg(feature = "python")]
#[pymethods]
impl KBlock {
    #[new]
    fn kblock_from_params(kblock_params: KBlockParams) -> Self {
        Self::from(kblock_params)
    }

    /// Get the concentration of a tile with given blockers
    fn tile_conc(&self, tile: TileState) -> f64 {
        self.tile_concentration(tile).into()
    }

    // #[pyo3(name = "blocker_percentage")]
    // fn py_blocker_percentage(&self, side: Side, tile: TileId) -> f64 {
    //     let blocker_conc = self.blocker_concentrations[self.glue_on_side(side, tile)];
    //     if blocker_conc == 0.0 {
    //         return 0.0;
    //     }
    //     blocker_conc / self.tile_concentration(tile)
    // }

    /// Print a string breaking down the total rate at some point
    fn detailed_rate_at_point(&self, state: &PyState, point: (usize, usize)) {
        let point = PointSafe2(point);
        let tile = TileState(state.0.tile_at_point(point));

        if tile.is_null() {
            let possible_tiles = self.possible_tiles_at_point(&state.0, point);

            if possible_tiles.is_empty() {
                println!("No possible tile attachments at this point.");
                return;
            }

            println!("Possible tile attachments:");
            let mut total_rate = PerSecond::zero();

            for &tile in possible_tiles.iter() {
                let rate = self.kf * self.tile_concentration(tile);
                let tile_name = self.tile_name(tile.into());
                let tile_idx = tile_index(tile);
                total_rate += rate;

                // Show tile info with its blockers
                let blockers = &[
                    if tile.is_blocked(NORTH) { "N" } else { "" },
                    if tile.is_blocked(EAST) { "E" } else { "" },
                    if tile.is_blocked(SOUTH) { "S" } else { "" },
                    if tile.is_blocked(WEST) { "W" } else { "" },
                ]
                .join("");

                let blocker_info = if blockers.is_empty() {
                    "no blockers".to_string()
                } else {
                    format!("blockers: {blockers}")
                };
                println!(
                    "  {} (id: {}, {}) - rate: {:.e}",
                    tile_name,
                    usize::from(tile_idx),
                    blocker_info,
                    rate
                );
            }

            println!("Total attachment rate: {total_rate:.e}");
            return;
        }

        let mut acc = String::new();
        for side in ALL_SIDES {
            let (kind, rate) = if tile.is_blocked(side) {
                let rate = self.blocker_detachment_rate_at_side(side, tile);
                ("detachment", rate)
            } else {
                // This assumes that there is nothing to the side, so a blocker can in fact attach
                let mut rate = self.blocker_attachment_rate_at_side(side, tile);
                // If there is a tile already attached on that side, then the attachment rate is 0
                if !Self::tile_to_side(&state.0, side, point).is_null() {
                    rate = PerSecond::zero();
                };
                ("attachment", rate)
            };

            let message = format!(
                "Blocker {} rate on side {}: {:.e}",
                kind,
                side_as_str(side),
                rate
            );
            acc.push_str(message.as_str());
            acc.push('\n');
        }
        // Tile detachment
        let detachment_rate = self.tile_detachment_rate(&state.0, point);
        acc.push_str(format!("Tile detachment rate {detachment_rate:.e}").as_str());
        println!("{acc}")
    }
}
