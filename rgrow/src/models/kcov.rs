use std::collections::HashSet;

use ndarray::{Array1, Array2};
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{
    base::{Energy, Glue, HashSetType, Rate},
    canvas::{PointSafe2, PointSafeHere},
    state::State,
    system::{Event, FissionHandling, System, TileBondInfo},
    type_alias,
};

type_alias!( f64 => Concentration, Strength );
type_alias!( u32 => TileId );

/*
*   Note:
*   - Change to north east south west
* */

const NORTH: u32 = 0b1000 << 28;
const EAST: u32 = 0b0100 << 28;
const SOUTH: u32 = 0b0010 << 28;
const WEST: u32 = 0b0001 << 28;

const ALL_COVERS: u32 = NORTH | EAST | SOUTH | WEST;
const NO_COVERS: u32 = !ALL_COVERS;

const ALL_SIDES: [u32; 4] = [NORTH, EAST, SOUTH, WEST];

const R: f64 = 1.98720425864083 / 1000.0; // in kcal/mol/K

/// Helper methods for tile id
///
/// This can help change the id to attach a cover on the north position, etc ...
mod tileid_helper {
    use super::{TileId, ALL_COVERS, EAST, NORTH, NO_COVERS, SOUTH, WEST};

    /// Unordered list containing all 16 possible tile combinations
    pub fn combinations(id: TileId) -> [TileId; 16] {
        let id = base_id(id);
        // This is not ordered nicely, 1 is west, 2 is south, 3 is both west and south, ...
        //
        // Maybe this *could be an* ordering ?
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15].map(|x| id & (x << 28))
    }

    pub fn attachments(id: TileId) -> TileId {
        id & ALL_COVERS
    }

    pub fn attach<const SIDE: TileId>(id: TileId) -> TileId {
        id | SIDE
    }

    pub fn detach<const SIDE: TileId>(id: TileId) -> TileId {
        id & (!SIDE)
    }

    /// Get the "base id", this is the id of the tile if it had no covers
    pub fn base_id(id: TileId) -> TileId {
        id & NO_COVERS
    }

    pub fn is_covered<const SIDE: TileId>(id: TileId) -> bool {
        (id & SIDE) != 0
    }

    // TODO: This should be compile time, I think
    #[inline(always)]
    pub const fn inverse<const SIDE: TileId>() -> TileId {
        match SIDE {
            NORTH => SOUTH,
            SOUTH => NORTH,
            EAST => WEST,
            WEST => EAST,
            _ => panic!("Can only find the inverse of NESW"),
        }
    }

    pub const fn index<const SIDE: TileId>() -> Option<usize> {
        match SIDE {
            NORTH => Some(0),
            EAST => Some(1),
            SOUTH => Some(2),
            WEST => Some(3),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KCov {
    pub tile_names: Vec<String>,
    pub tile_concentration: Vec<Concentration>,
    pub tile_colors: Vec<[u8; 4]>,

    pub glue_names: Vec<String>,
    pub cover_concentrations: Vec<Concentration>,
    pub temperature: f64,

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
    glue_links: Array2<Strength>,

    /// What can attach to the north of some *glue*
    ///
    /// For example, if some tile has the glue 6 to the north side, north_friends[6] will
    /// return a hashset with every tile containing a 5 (1<->2, 3<->4, 5<->6)
    ///
    /// To get possible attachemnts to some side of a tile (which is the usual expected use), call
    /// `get_friends_one_side` or `get_friends`
    north_friends: Vec<HashSetType<TileId>>,
    /// Identical to north_friends
    south_friends: Vec<HashSetType<TileId>>,
    /// Identical to north_friends
    east_friends: Vec<HashSetType<TileId>>,
    /// Identical to north_friends
    west_friends: Vec<HashSetType<TileId>>,

    /// Energy of tile and cover, cover i contains [N, E, S, W]
    energy_cover: Array1<[Energy; 4]>,

    /// Energy between two tiles, if tile a is to the north of tile b, then
    /// this shoudl be indexed as [(a,b)]
    energy_ns: Array2<Energy>,
    energy_we: Array2<Energy>,

    pub alpha: Energy,
    pub kf: f64,
    fission_handling: FissionHandling,
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

impl KCov {
    const ZERO_RATE: Rate = 0.0;

    pub fn new(
        tile_names: Vec<String>,
        tile_concentration: Vec<Concentration>,
        tile_colors: Vec<[u8; 4]>,
        glue_names: Vec<String>,
        cover_concentrations: Vec<Concentration>,
        tile_glues: Array1<[Glue; 4]>,
        glue_links: Array2<Strength>,
        temperature: f64,
        kf: f64,
        alpha: f64,
        fission_handling: FissionHandling,
    ) -> Self {
        let tilecount = tile_names.len();
        let mut s = Self {
            tile_names,
            tile_concentration,
            tile_colors,
            glue_names,
            cover_concentrations,
            tile_glues,
            glue_links,
            temperature,
            north_friends: Vec::default(),
            south_friends: Vec::default(),
            east_friends: Vec::default(),
            west_friends: Vec::default(),
            energy_ns: Array2::zeros((tilecount, tilecount)),
            energy_we: Array2::zeros((tilecount, tilecount)),
            energy_cover: Array1::default(tilecount),
            alpha,
            kf,
            fission_handling,
        };
        s.fill_friends();
        s.fill_energy_pairs();
        s.fill_energy_covers();
        s
    }

    // Side must be north, south, east, or west
    pub fn get_friends_one_side<const SIDE: TileId>(
        &self,
        tile: TileId,
    ) -> Option<&HashSetType<TileId>> {
        // The tile is covered, so we dont have any friends
        if tileid_helper::is_covered::<SIDE>(tile) {
            return None;
        }

        let tile_glue = self.glue_on_side::<SIDE>(tile);
        Some(match SIDE {
            NORTH => &self.north_friends[tile_glue],
            SOUTH => &self.south_friends[tile_glue],
            EAST => &self.east_friends[tile_glue],
            WEST => &self.west_friends[tile_glue],
            _ => panic!(
                "get_friends_one_side should be called with either NORTH, SOUTH, EAST, or WEST, not a combination"
            ),
        })
    }

    /// Get the friends to some side
    pub fn get_friends<const SIDE: TileId>(&self, tile: TileId) -> HashSetType<TileId> {
        let mut tile_friends = HashSetType::default();

        // If side contains north, then we will add the north friends...
        if SIDE & NORTH != 0 {
            if let Some(ext) = self.get_friends_one_side::<NORTH>(tile) {
                tile_friends.extend(ext);
            }
        }
        if SIDE & SOUTH != 0 {
            if let Some(ext) = self.get_friends_one_side::<SOUTH>(tile) {
                tile_friends.extend(ext);
            }
        }
        if SIDE & EAST != 0 {
            if let Some(ext) = self.get_friends_one_side::<EAST>(tile) {
                tile_friends.extend(ext);
            }
        }
        if SIDE & WEST != 0 {
            if let Some(ext) = self.get_friends_one_side::<WEST>(tile) {
                tile_friends.extend(ext);
            }
        }

        tile_friends
    }

    /// Get the glues. If there are covers, this wil look past them, and return the
    /// glue that is under it
    pub fn get_tile_raw_glues(&self, tile_id: TileId) -> Vec<Glue> {
        self.tile_glues[tileid_helper::base_id(tile_id) as usize].to_vec()
    }

    pub fn glue_on_side<const SIDE: TileId>(&self, tile_id: TileId) -> Glue {
        let glues = self.get_tile_uncovered_glues(tile_id);
        glues[tileid_helper::index::<SIDE>().expect("Side must be NESW")]
    }

    /// Get the glues, with a glue being replaced with 0 if there is a cover
    pub fn get_tile_uncovered_glues(&self, tile_id: TileId) -> Vec<Glue> {
        // This MUST be exactly 4 length
        let row = self.get_tile_raw_glues(tile_id);
        let mut glues = vec![0; 4];

        // If any of the sides have a cover, then the glue is 0
        if tile_id & NORTH == 0 {
            glues[0] = row[0];
        }
        if tile_id & EAST == 0 {
            glues[1] = row[1]
        }
        if tile_id & SOUTH == 0 {
            glues[2] = row[2];
        }
        if tile_id & WEST == 0 {
            glues[3] = row[3]
        }
        glues
    }

    pub fn fill_friends(&mut self) {
        let len = self.glue_names.len();
        let empty_friends = vec![HashSetType::<TileId>::default(); len];

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
            let base_id = tileid_helper::base_id(id as u32);

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

    pub fn fill_energy_covers(&mut self) {
        let tile_ids = self.tile_names().len();
        for t in 0..tile_ids {
            let (tn, te, ts, tw) = (
                self.glue_on_side::<NORTH>(t as TileId),
                self.glue_on_side::<EAST>(t as TileId),
                self.glue_on_side::<SOUTH>(t as TileId),
                self.glue_on_side::<WEST>(t as TileId),
            );
            self.energy_cover[t][0] = self.glue_links[(tn, glue_inverse(tn))];
            self.energy_cover[t][1] = self.glue_links[(te, glue_inverse(te))];
            self.energy_cover[t][2] = self.glue_links[(ts, glue_inverse(ts))];
            self.energy_cover[t][3] = self.glue_links[(tw, glue_inverse(tw))];
        }
    }

    /// Fill energy_ns, energy_we: Array2<Energy>
    ///
    /// This will mutate the structure
    pub fn fill_energy_pairs(&mut self) {
        // The ids of the tiles with no covers on any of their sides
        //
        // We will assume that tiles are uncovered when getting their energy
        // this check is done in the energy_to function
        let tile_ids = self.tile_names().len();

        for t1 in 0..tile_ids {
            // Glues on the sides of tile 1
            let (t1n, t1e, t1s, t1w) = (
                self.glue_on_side::<NORTH>(t1 as TileId),
                self.glue_on_side::<EAST>(t1 as TileId),
                self.glue_on_side::<SOUTH>(t1 as TileId),
                self.glue_on_side::<WEST>(t1 as TileId),
            );

            for t2 in 0..tile_ids {
                let (t2n, t2e, t2s, t2w) = (
                    self.glue_on_side::<NORTH>(t2 as TileId),
                    self.glue_on_side::<EAST>(t2 as TileId),
                    self.glue_on_side::<SOUTH>(t2 as TileId),
                    self.glue_on_side::<WEST>(t2 as TileId),
                );

                // t1 -- t2
                self.energy_we[(t1, t2)] = self.glue_links[(t1e, t2w)];

                // t2 -- t1
                self.energy_we[(t2, t1)] = self.glue_links[(t2e, t1w)];

                // t1
                // ||
                // t2
                self.energy_ns[(t1, t2)] = self.glue_links[(t1s, t2n)];

                // t2
                // ||
                // t1
                self.energy_ns[(t2, t1)] = self.glue_links[(t2s, t1n)];
            }
        }
    }

    /// SIDE here must be NSEW
    pub fn energy_to<const SIDE: TileId>(&self, tile1: TileId, tile2: TileId) -> Energy {
        // If we are covered on the sticking side, or the other tile has a cover, then we
        // have no binding energy
        if tileid_helper::is_covered::<SIDE>(tile1)
            || match SIDE {
                NORTH => tileid_helper::is_covered::<SOUTH>(tile2),
                EAST => tileid_helper::is_covered::<WEST>(tile2),
                SOUTH => tileid_helper::is_covered::<NORTH>(tile2),
                WEST => tileid_helper::is_covered::<EAST>(tile2),
                _ => false,
            }
        {
            return Self::ZERO_RATE;
        }

        // Ignore covers
        let (tile1, tile2) = (tileid_helper::base_id(tile1), tileid_helper::base_id(tile2));
        // Now we know that neither the tile, nor the one were attaching to is covered
        match SIDE {
            NORTH => self.energy_ns[(tile2 as usize, tile1 as usize)],
            EAST => self.energy_we[(tile1 as usize, tile2 as usize)],
            SOUTH => self.energy_ns[(tile1 as usize, tile2 as usize)],
            WEST => self.energy_we[(tile2 as usize, tile1 as usize)],
            _ => panic!("Must enter NSEW"),
        }
    }

    /// Energy of neighbour bonds
    pub fn energy_at_point<S: State>(&self, state: &S, point: PointSafe2) -> Energy {
        let tile_id: TileId = state.tile_at_point(point);
        let mut energy = Self::ZERO_RATE;
        let neighbour_tile = state.tile_to_n(point);
        energy += self.energy_to::<NORTH>(tile_id, neighbour_tile);
        let neighbour_tile = state.tile_to_s(point);
        energy += self.energy_to::<SOUTH>(tile_id, neighbour_tile);
        let neighbour_tile = state.tile_to_e(point);
        energy += self.energy_to::<EAST>(tile_id, neighbour_tile);
        let neighbour_tile = state.tile_to_w(point);
        energy += self.energy_to::<WEST>(tile_id, neighbour_tile);
        energy
    }

    #[inline(always)]
    fn rtval(&self) -> Energy {
        R * (self.temperature + 273.15)
    }

    pub fn tile_detachment_rate<S: State>(&self, state: &S, p: PointSafe2) -> Rate {
        let tile = state.tile_at_point(p);
        // If there is no tile, then nothing to attach
        if tile == 0 {
            return Self::ZERO_RATE;
        }
        let energy_with_neighbours = self.energy_at_point(state, p);
        self.kf * (energy_with_neighbours * (1.0 / self.rtval())).exp()
    }

    /// The rate at which a tile will attach somewhere
    pub fn tile_attachment_rate(&self, tile: TileId) -> f64 {
        self.tile_concentration[tile as usize] * self.kf
    }

    fn get_friend_side_if_empty<const SIDE: TileId, S: State>(
        &self,
        state: &S,
        point: PointSafe2,
    ) -> Option<&HashSetType<TileId>> {
        let neighbour = match SIDE {
            NORTH => state.tile_to_n(point),
            EAST => state.tile_to_e(point),
            SOUTH => state.tile_to_s(point),
            WEST => state.tile_to_w(point),
            _ => panic!("Side must me NESW"),
        };

        // If there has already been an attachemnt, then nothing can attach there
        if neighbour != 0 {
            return None;
        }
        let tile = state.tile_at_point(point);
        self.get_friends_one_side::<SIDE>(tile)
    }

    pub fn cover_attachment_rate_at_side<const SIDE: TileId>(&self, tile: TileId) -> Rate {
        self.kf * self.cover_concentrations[self.glue_on_side::<SIDE>(tile)]
    }

    /// Get the energy between a tile and a cover to some side
    pub fn cover_detachment_rate_at_side<const SIDE: TileId>(&self, tile: TileId) -> Rate {
        // If there is no cover in that side, then the detachment rate will be 0
        if !tileid_helper::is_covered::<SIDE>(tile) {
            return Self::ZERO_RATE;
        };

        let tile = tileid_helper::base_id(tile);
        self.kf
            * (match SIDE {
                NORTH => self.energy_cover[tile as usize][0],
                EAST => self.energy_cover[tile as usize][1],
                SOUTH => self.energy_cover[tile as usize][2],
                WEST => self.energy_cover[tile as usize][3],
                _ => panic!("Side must be NESW"),
            } * (1.0 / self.rtval()))
            .exp()
    }

    pub fn cover_detachment_rate_total(&self, tile: TileId) -> Rate {
        self.cover_detachment_rate_at_side::<NORTH>(tile)
            + self.cover_detachment_rate_at_side::<EAST>(tile)
            + self.cover_detachment_rate_at_side::<SOUTH>(tile)
            + self.cover_detachment_rate_at_side::<WEST>(tile)
    }

    fn maybe_detach_cover_on_side_event<const SIDE: TileId>(
        &self,
        tileid: TileId,
        point: PointSafe2,
        acc: &mut Rate,
    ) -> Option<(bool, Rate, Event)> {
        // Something cannot detach if there is no cover
        if !tileid_helper::is_covered::<SIDE>(tileid) {
            return None;
        }
        *acc -= self.cover_detachment_rate_at_side::<SIDE>(tileid);
        if *acc <= 0.0 {
            // ^ SIDE will change the bit from 1 to 0, so no longer have a cover here
            Some((true, *acc, Event::MonomerChange(point, tileid ^ SIDE)))
        } else {
            None
        }
    }

    /// Detach a cover from tile
    pub fn event_monomer_cover_detachment<S: State>(
        &self,
        state: &S,
        point: PointSafe2,
        acc: &mut Rate,
    ) -> (bool, Rate, Event) {
        // Check what covers the tile has
        let tile = state.tile_at_point(point);
        if tile == 0 {
            return (false, *acc, Event::None);
        }

        // Update the acc for each side, if there is no cover, then None will be returned, if no
        // evene takes place, then acc is updated, and none is returned.
        self.maybe_detach_cover_on_side_event::<NORTH>(tile, point, acc)
            .or(self.maybe_detach_cover_on_side_event::<EAST>(tile, point, acc))
            .or(self.maybe_detach_cover_on_side_event::<SOUTH>(tile, point, acc))
            .or(self.maybe_detach_cover_on_side_event::<WEST>(tile, point, acc))
            .unwrap_or((false, *acc, Event::None))
    }

    pub fn tile_to_side<const SIDE: TileId, S: State>(state: &S, p: PointSafe2) -> TileId {
        match SIDE {
            NORTH => state.tile_to_n(p),
            EAST => state.tile_to_e(p),
            SOUTH => state.tile_to_s(p),
            WEST => state.tile_to_w(p),
            _ => panic!("Side must be North, South, East, or West"),
        }
    }

    fn maybe_attach_side_event<const SIDE: TileId, S: State>(
        &self,
        tileid: TileId,
        point: PointSafe2,
        state: &S,
        acc: &mut Rate,
    ) -> Option<(bool, Rate, Event)> {
        // A cover cannot attach to a side with a cover already attached
        if tileid_helper::is_covered::<SIDE>(tileid)
        // If a tile is already attached to that side, then nothing can attach
            || Self::tile_to_side::<SIDE, S>(state, point) != 0
        {
            return None;
        }

        *acc -= self.kf * self.cover_concentrations[self.glue_on_side::<SIDE>(tileid)];
        if *acc <= 0.0 {
            // | SIDE will change the bit from 0 to 1
            Some((true, *acc, Event::MonomerChange(point, tileid | SIDE)))
        } else {
            None
        }
    }

    // Attach a cover to a tile
    pub fn event_monomer_cover_attachment<S: State>(
        &self,
        state: &S,
        point: PointSafe2,
        acc: &mut Rate,
    ) -> (bool, Rate, Event) {
        let tile = state.tile_at_point(point);
        if tile == 0 {
            return (false, 0.0, Event::None);
        }
        self.maybe_attach_side_event::<NORTH, S>(tile, point, state, acc)
            .or(self.maybe_attach_side_event::<EAST, S>(tile, point, state, acc))
            .or(self.maybe_attach_side_event::<SOUTH, S>(tile, point, state, acc))
            .or(self.maybe_attach_side_event::<WEST, S>(tile, point, state, acc))
            .unwrap_or((false, *acc, Event::None))
    }

    // TODO: Handle Fission Here
    pub fn event_monomer_detachment<S: State>(
        &self,
        state: &S,
        point: PointSafe2,
        acc: &mut Rate,
    ) -> (bool, Rate, Event) {
        *acc -= self.tile_detachment_rate(state, point);
        if *acc > 0.0 {
            return (false, *acc, Event::None);
        }

        match self.fission_handling {
            FissionHandling::NoFission => (true, *acc, Event::None),
            FissionHandling::JustDetach => (true, *acc, Event::MonomerDetachment(point)),
            _ => panic!("Only NoFission, and JustDetach are supported"),
        }
    }

    pub fn possible_tiles_at_point<S: State>(
        &self,
        state: &S,
        point: PointSafe2,
    ) -> HashSetType<TileId> {
        let tile = state.tile_at_point(point);
        let mut friends: HashSetType<TileId> = HashSet::default();

        // tile aready attached here
        if tile != 0 {
            return friends;
        }

        // If there is tile to the north of this point
        let neighbour_tile = state.tile_to_n(point);
        if neighbour_tile != 0 {
            if let Some(northf) = self.get_friends_one_side::<SOUTH>(neighbour_tile) {
                friends.extend(northf);
            }
        }
        let neighbour_tile = state.tile_to_s(point);
        if neighbour_tile != 0 {
            if let Some(southf) = self.get_friends_one_side::<NORTH>(neighbour_tile) {
                friends.extend(southf);
            }
        }
        let neighbour_tile = state.tile_to_e(point);
        if neighbour_tile != 0 {
            if let Some(eastf) = self.get_friends_one_side::<WEST>(neighbour_tile) {
                friends.extend(eastf);
            }
        }
        let neighbour_tile = state.tile_to_w(point);
        if neighbour_tile != 0 {
            if let Some(westf) = self.get_friends_one_side::<EAST>(neighbour_tile) {
                friends.extend(westf);
            }
        }
        friends
    }

    pub fn total_attachment_rate_at_point<S: State>(&self, point: PointSafe2, state: &S) -> Rate {
        self.possible_tiles_at_point(state, point)
            .iter()
            .fold(0.0, |acc, &x| {
                acc + (self.kf * self.tile_concentration[x as usize])
            })
    }

    /// Check if two tiles can bond or not by checking that:
    /// + There are no covers
    /// + The glues are inverses
    fn form_bond<const SIDE: TileId>(&self, tile1: TileId, tile2: TileId) -> bool {
        // Check if either of the tiles have a cover
        if (tile1 & SIDE) != 0 || (tile2 & tileid_helper::inverse::<SIDE>()) != 0 {
            return false;
        }

        let g1 = self.glue_on_side::<SIDE>(tile1);
        let g2 = match SIDE {
            NORTH => self.glue_on_side::<SOUTH>(tile2),
            EAST => self.glue_on_side::<WEST>(tile2),
            SOUTH => self.glue_on_side::<NORTH>(tile2),
            WEST => self.glue_on_side::<EAST>(tile2),
            _ => panic!("Must enter NSEW"),
        };

        if g1 == 0 || g1 != glue_inverse(g2) {
            return false;
        }
        true
    }

    /// Helper functino for choose_attachment_side
    ///
    /// If a tiles neighbour to some side and the tile can bond, then
    /// this will add to some accumulator what the cance is that the
    /// tile is uncoverd
    fn record_bonded_tile_uncovered_chance<const SIDE: TileId, S: State>(
        &self,
        state: &S,
        point: PointSafe2,
        tile: TileId,
        acc: &mut Vec<(TileId, f64)>,
    ) {
        let neighbour = Self::tile_to_side::<SIDE, S>(state, point);
        if neighbour != 0 && self.form_bond::<SIDE>(tile, neighbour) {
            let uncovered_chance = self.uncover_percentage::<SIDE>(tile);
            acc.push((SIDE, uncovered_chance));
        }
    }

    /// Given that some tile is being attached at some point, what other tile is it bonding to ?
    fn choose_attachment_side<S: State>(
        &self,
        state: &S,
        point: PointSafe2,
        tile: TileId,
    ) -> TileId {
        let mut rates = Vec::with_capacity(4);
        self.record_bonded_tile_uncovered_chance::<NORTH, S>(state, point, tile, &mut rates);
        self.record_bonded_tile_uncovered_chance::<EAST, S>(state, point, tile, &mut rates);
        self.record_bonded_tile_uncovered_chance::<SOUTH, S>(state, point, tile, &mut rates);
        self.record_bonded_tile_uncovered_chance::<WEST, S>(state, point, tile, &mut rates);

        let mut sum = 0.0;
        for (_, uncovered_chance) in &rates {
            sum += uncovered_chance
        }
        if sum == 0.0 {
            panic!("All neighbours are covered, or not there!");
        }

        let mut r = rand::thread_rng().gen_range(0.0..sum);
        for (side, uncovered_chance) in rates {
            if uncovered_chance > r {
                return side;
            }
            r -= uncovered_chance;
        }
        panic!("How did we get here!?")
    }

    pub fn maybe_cover_on_side<const SIDE: TileId>(&self, tile: TileId) -> TileId {
        let chance = self.cover_percentage::<SIDE>(tile);
        if chance == 0.0 {
            return 0;
        }
        let r = rand::thread_rng().gen_range(0.0..1.0);
        if r <= chance {
            SIDE
        } else {
            0
        }
    }

    pub fn choose_covers(&self, tile: TileId, except: TileId) -> TileId {
        (self.maybe_cover_on_side::<NORTH>(tile)
            | self.maybe_cover_on_side::<EAST>(tile)
            | self.maybe_cover_on_side::<SOUTH>(tile)
            | self.maybe_cover_on_side::<WEST>(tile))
            & (!except) // Make sure that there is no cover on the except side
    }

    /// Probability of any tile attaching at some point
    pub fn event_monomer_attachment<S: State>(
        &self,
        state: &S,
        point: PointSafe2,
        acc: &mut Rate,
    ) -> (bool, Rate, Event) {
        let tile = state.tile_at_point(point);
        // tile aready attached here
        if tile != 0 {
            return (false, *acc, Event::None);
        }

        // FIXME: This shuold be a HashMap, not hash set. Repetition is important ??
        let friends: HashSetType<TileId> = self.possible_tiles_at_point(state, point);
        for tile in friends {
            // FIXME: This concentration is wrong! It includes, for example the tile with covers
            // everywhere, which is no good.
            *acc -= self.kf * self.tile_concentration[tile as usize];
            if *acc <= 0.0 {
                let attaches_to = self.choose_attachment_side(state, point, tile);
                let covers = self.choose_covers(tile, attaches_to);
                return (true, *acc, Event::MonomerAttachment(point, tile | covers));
            }
        }
        (false, *acc, Event::None)
    }

    /// Percentage of total concentration of some tile that has a cover on a given side
    pub fn cover_percentage<const SIDE: TileId>(&self, tile: TileId) -> f64 {
        let detachment_rate = self.cover_detachment_rate_at_side::<SIDE>(tile | SIDE);
        let attachment_rate = self.cover_attachment_rate_at_side::<SIDE>(tile);
        attachment_rate / (attachment_rate + detachment_rate)
    }

    /// Percentage of total concentration of some tile that has no cover on a given side
    pub fn uncover_percentage<const SIDE: TileId>(&self, tile: TileId) -> f64 {
        1.0 - self.cover_percentage::<SIDE>(tile)
    }

    pub fn total_cover_attachment_rate<S: State>(&self, state: &S, point: PointSafe2) -> Rate {
        // Check that there is a tile at this point
        let tile = state.tile_at_point(point);
        if tile == 0 {
            return Self::ZERO_RATE;
        }

        let mut rate = Self::ZERO_RATE;
        if !tileid_helper::is_covered::<NORTH>(tile) && state.tile_to_n(point) == 0 {
            rate += self.kf * self.cover_concentrations[self.glue_on_side::<NORTH>(tile)];
        }
        if !tileid_helper::is_covered::<SOUTH>(tile) && state.tile_to_s(point) == 0 {
            rate += self.kf * self.cover_concentrations[self.glue_on_side::<SOUTH>(tile)];
        }
        if !tileid_helper::is_covered::<EAST>(tile) && state.tile_to_e(point) == 0 {
            rate += self.kf * self.cover_concentrations[self.glue_on_side::<EAST>(tile)];
        }
        if !tileid_helper::is_covered::<WEST>(tile) && state.tile_to_w(point) == 0 {
            rate += self.kf * self.cover_concentrations[self.glue_on_side::<WEST>(tile)];
        }
        return rate;
    }
}

/*
* The idea right now is that:
* 1. All tiles have a different id
* 2. If a tile has gets a cover attachment / detachment,
*    then it becomes a new tile
*    That is to say if two tile A could become tile B by
*    attaching / detaching covers, then they are different
*    tiles (with different ids), but they have the same base
*    id.
* */

impl TileBondInfo for KCov {
    fn tile_color(&self, tileid: TileId) -> [u8; 4] {
        self.tile_colors[tileid_helper::base_id(tileid) as usize]
    }

    fn tile_name(&self, tileid: TileId) -> &str {
        self.tile_names[tileid_helper::base_id(tileid) as usize].as_str()
    }

    fn bond_name(&self, bond_number: usize) -> &str {
        todo!()
    }

    fn tile_colors(&self) -> &Vec<[u8; 4]> {
        &self.tile_colors
    }

    fn tile_names(&self) -> Vec<&str> {
        self.tile_names.iter().map(|s| s.as_str()).collect()
    }

    fn bond_names(&self) -> Vec<&str> {
        todo!()
    }
}

impl System for KCov {
    fn system_info(&self) -> String {
        todo!()
    }

    fn perform_event<St: State>(&self, state: &mut St, event: &Event) -> &Self {
        match event {
            Event::None => panic!("Canot perform None event"),
            Event::MonomerDetachment(point) => state.set_sa(point, &0),
            // Monomer Change -- Covers
            Event::MonomerChange(point, tile) | Event::MonomerAttachment(point, tile) => {
                state.set_sa(point, tile)
            }
            _ => panic!("Polymer not yet implemented"),
        };
        self
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
            _ => panic!("Polymer not yet implemented"),
        }
    }

    fn event_rate_at_point<S: crate::state::State>(
        &self,
        state: &S,
        p: crate::canvas::PointSafeHere,
    ) -> crate::base::Rate {
        let p = if state.inbounds(p.0) {
            PointSafe2(p.0)
        } else {
            return Self::ZERO_RATE;
        };
        let tile = { state.tile_at_point(p) };
        if tile != 0 {
            self.tile_detachment_rate(state, p)
                + self.cover_detachment_rate_total(tile)
                + self.total_cover_attachment_rate(state, p)
        } else {
            self.total_attachment_rate_at_point(p, state)
        }
    }

    fn choose_event_at_point<St: crate::state::State>(
        &self,
        state: &St,
        point: crate::canvas::PointSafe2,
        acc: crate::base::Rate,
    ) -> crate::system::Event {
        let mut acc = acc;

        if let (true, _, event) = self.event_monomer_detachment(state, point, &mut acc) {
            return event;
        };
        if let (true, _, event) = self.event_monomer_attachment(state, point, &mut acc) {
            return event;
        }
        if let (true, _, event) = self.event_monomer_cover_attachment(state, point, &mut acc) {
            return event;
        }
        if let (true, _, event) = self.event_monomer_cover_detachment(state, point, &mut acc) {
            return event;
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
        todo!()
    }

    fn calc_mismatch_locations<St: crate::state::State>(&self, state: &St) -> Array2<usize> {
        todo!()
    }
}

#[cfg(test)]
mod test_covtile {
    use crate::models::kcov::{tileid_helper, EAST, NORTH, WEST};

    #[test]
    fn get_ids() {
        let mut t = 0b10110000;
        t = tileid_helper::attach::<EAST>(t);
        assert_eq!(tileid_helper::base_id(t), 0b10110000);
        assert_eq!(t, 0b10110000 | EAST);

        let mut k = 1;
        k = tileid_helper::attach::<EAST>(k);
        k = tileid_helper::attach::<WEST>(k);
        assert_eq!(tileid_helper::base_id(k), 1);
    }

    #[test]
    fn is_covered_side() {
        assert!(tileid_helper::is_covered::<NORTH>(NORTH));
        assert!(tileid_helper::is_covered::<NORTH>(123 | NORTH));
        assert!(!tileid_helper::is_covered::<EAST>(123 | NORTH));
    }

    #[test]
    fn detach_side() {
        assert_eq!(0, tileid_helper::detach::<NORTH>(NORTH));
        assert_eq!(123, tileid_helper::detach::<NORTH>(123 | NORTH));
    }
}

#[cfg(test)]
mod test_kcov {
    use super::*;
    use ndarray::array;

    fn sample_kcov() -> KCov {
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

        KCov::new(
            vec![
                "null".to_string(),
                "f".to_string(),
                "s".to_string(),
                "t".to_string(),
            ],
            vec![1.0, 1.0, 1.0, 1.0],
            vec![DEFAULT_COLOR; 4],
            // Glues
            vec![
                "null".to_string(),
                "1".to_string(),
                "2".to_string(),
                "3".to_string(),
                "4".to_string(),
            ],
            vec![0., 1., 1., 1., 1.],
            tile_glues,
            glue_linkns,
            60.0,
            1e6,
            0.0,
            FissionHandling::JustDetach,
        )
    }

    #[test]
    fn glue_side() {
        let kdcov = sample_kcov();
        assert_eq!(kdcov.glue_on_side::<NORTH>(1), 1);
        assert_eq!(kdcov.glue_on_side::<SOUTH>(1), 0);
        assert_eq!(kdcov.glue_on_side::<WEST>(3), 4);
    }

    #[test]
    fn friends_build() {
        let mut kdcov = sample_kcov();
        //println!("Tile Names: {:?}", kdcov.tile_names());
        kdcov.fill_friends();

        //println!("Tile Names: {:?}", kdcov.tile_names);
        //println!("N: {:?}", kdcov.north_friends);
        //println!("S: {:?}", kdcov.south_friends);
        //println!("E: {:?}", kdcov.east_friends);
        //println!("W: {:?}", kdcov.west_friends);

        let mut expected_nf = HashSetType::default();

        expected_nf.insert(2);
        // This is a little strange to use, as you need to know the glue on the north side of the
        // tile.
        assert_eq!(kdcov.north_friends[1], expected_nf);
        // These helper methods make it so that you can find every tile that can bond to the north
        // of some tile id
        assert_eq!(kdcov.get_friends_one_side::<NORTH>(1), Some(&expected_nf));
        assert_eq!(kdcov.get_friends::<NORTH>(1), expected_nf);
        // You can also get frineds to multiple sides at once
        assert_eq!(kdcov.get_friends::<{ NORTH | EAST }>(1), expected_nf);

        let mut expected_wf = HashSetType::default();
        expected_wf.insert(2);
        assert_eq!(kdcov.west_friends[4], expected_nf);
        assert_eq!(kdcov.get_friends::<WEST>(3), expected_nf);
    }

    fn check_energy_at_point() {}
}
