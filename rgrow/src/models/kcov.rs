use std::{
    collections::{HashMap, HashSet},
    usize,
};

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use crate::{
    base::{Energy, Glue, HashSetType, Rate},
    canvas::{Canvas, PointSafe2, PointSafeHere},
    state::State,
    system::{
        DimerInfo, Event, FissionHandling, Orientation, System, SystemWithDimers, TileBondInfo,
    },
    type_alias,
};

// Imports for python bindings

#[cfg(feature = "python")]
use crate::python::PyState;
#[cfg(feature = "python")]
use numpy::ToPyArray;
#[cfg(feature = "python")]
use pyo3::prelude::*;

type_alias!( f64 => Concentration, Strength );
type_alias!( u32 => TileId, Side );

const NORTH: Side = 0b0001;
const EAST: Side = 0b0010;
const SOUTH: Side = 0b0100;
const WEST: Side = 0b1000;

const ALL_COVERS: Side = NORTH | EAST | SOUTH | WEST;
const NO_COVERS: Side = !ALL_COVERS;

const ALL_SIDES: [Side; 4] = [NORTH, EAST, SOUTH, WEST];

const R: f64 = 1.98720425864083 / 1000.0; // in kcal/mol/K

pub fn attachments(id: TileId) -> TileId {
    id & ALL_COVERS
}

pub fn attach(side: Side, id: TileId) -> TileId {
    id | side
}

pub fn detach(side: Side, id: TileId) -> TileId {
    id & (!side)
}

pub fn uncover_all(id: TileId) -> TileId {
    id & NO_COVERS
}

pub fn is_covered(side: Side, id: TileId) -> bool {
    (id & side) != 0
}

#[inline(always)]
pub const fn inverse(side: Side) -> Side {
    match side {
        NORTH => SOUTH,
        SOUTH => NORTH,
        EAST => WEST,
        WEST => EAST,
        _ => panic!("Can only find the inverse of NESW"),
    }
}

pub fn tile_index(tile: TileId) -> usize {
    (tile >> 4) as usize
}

/// Index array by side, north = 0, east = 1, south = 2, west = 3
///
/// # Panic
///
/// This will panic when called with a mix of sides (ie, north-east)
pub const fn side_index(side: Side) -> Option<usize> {
    match side {
        NORTH => Some(0),
        EAST => Some(1),
        SOUTH => Some(2),
        WEST => Some(3),
        _ => None,
    }
}

/// Helper function to help print sides
pub fn side_as_str(side: Side) -> &'static str {
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
pub struct KCov {
    pub tile_names: Vec<String>,
    pub tile_concentration: Vec<Concentration>,
    pub tile_colors: Vec<[u8; 4]>,

    pub glue_names: Vec<String>,
    pub cover_concentrations: Vec<Concentration>,
    pub temperature: f64,
    pub seed: HashMap<PointSafe2, TileId>,
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
    pub(crate) glue_links: Array2<Strength>,

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

    pub fn update(&mut self) {
        self.fill_energy_pairs();
        self.fill_energy_covers();
    }

    pub fn new(
        tile_names: Vec<String>,
        tile_concentration: Vec<Concentration>,
        tile_colors: Vec<[u8; 4]>,
        glue_names: Vec<String>,
        cover_concentrations: Vec<Concentration>,
        tile_glues: Array1<[Glue; 4]>,
        glue_links: Array2<Strength>,
        seed: HashMap<PointSafe2, TileId>,
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
            seed,
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
        s.update();
        s
    }

    /// Get the uncovered friends to one side of some given tile
    pub fn get_uncovered_friends_to_side(
        &self,
        side: Side,
        tile: TileId,
    ) -> Option<&HashSetType<TileId>> {
        // The tile is covered, so we dont have any friends
        if is_covered(side, tile) {
            return None;
        }

        let tile_glue = self.glue_on_side(side, tile);
        Some(match side {
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
    pub fn get_friends(&self, side: Side, tile: TileId) -> HashSetType<TileId> {
        let mut tile_friends = HashSetType::default();
        for s in ALL_SIDES {
            if side & s != 0 {
                if let Some(ext) = self.get_uncovered_friends_to_side(s, tile) {
                    tile_friends.extend(ext);
                }
            }
        }
        tile_friends
    }

    /// Get the glues. If there are covers, this wil look past them, and return the
    /// glue that is under it
    pub fn get_tile_raw_glues(&self, tile_id: TileId) -> Vec<Glue> {
        let index = tile_index(tile_id) as usize;
        self.tile_glues[index].to_vec()
    }

    pub fn glue_on_side(&self, side: Side, tile_id: TileId) -> Glue {
        let glues = self.get_tile_uncovered_glues(tile_id);
        glues[side_index(side).expect("Side must be NESW")]
    }

    /// Get the glues, with a glue being replaced with 0 if there is a cover
    pub fn get_tile_uncovered_glues(&self, tile_id: TileId) -> Vec<Glue> {
        // This MUST be exactly 4 length
        let row = self.get_tile_raw_glues(tile_id);
        let mut glues = vec![0; 4];
        for s in ALL_SIDES {
            if !is_covered(s, tile_id) {
                let i = side_index(s).unwrap() as usize;
                glues[i] = row[i];
            }
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
            // Add 4 zeros to the binary representation of the number
            let base_id = (id << 4) as u32;

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
                self.glue_on_side(NORTH, (t << 4) as TileId),
                self.glue_on_side(EAST, (t << 4) as TileId),
                self.glue_on_side(SOUTH, (t << 4) as TileId),
                self.glue_on_side(WEST, (t << 4) as TileId),
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
            let t1_tile_id = t1 << 4;

            // Glues on the sides of tile 1
            let (t1n, t1e, t1s, t1w) = (
                self.glue_on_side(NORTH, t1_tile_id as TileId),
                self.glue_on_side(EAST, t1_tile_id as TileId),
                self.glue_on_side(SOUTH, t1_tile_id as TileId),
                self.glue_on_side(WEST, t1_tile_id as TileId),
            );

            for t2 in 0..tile_ids {
                let t2_tile_id = t2 << 4;
                let (t2n, t2e, t2s, t2w) = (
                    self.glue_on_side(NORTH, t2_tile_id as TileId),
                    self.glue_on_side(EAST, t2_tile_id as TileId),
                    self.glue_on_side(SOUTH, t2_tile_id as TileId),
                    self.glue_on_side(WEST, t2_tile_id as TileId),
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

    /// Add seed to system
    pub fn add_seed<S: State>(&mut self, state: &mut S, seed: HashMap<PointSafe2, TileId>) {
        self.seed = seed;
        for (point, tile) in &self.seed {
            state.set_sa(point, tile)
        }
    }

    pub fn is_seed(&self, point: &PointSafe2) -> bool {
        self.seed.contains_key(point)
    }

    /// SIDE here must be NSEW
    pub fn energy_to(&self, side: Side, tile1: TileId, tile2: TileId) -> Energy {
        // If we are covered on the sticking side, or the other tile has a cover, then we
        // have no binding energy
        if is_covered(side, tile1) || is_covered(inverse(side), tile2) {
            return Self::ZERO_RATE;
        }

        // Ignore covers
        let (tile1, tile2) = (tile_index(tile1), tile_index(tile2));

        // Now we know that neither the tile, nor the one were attaching to is covered
        match side {
            NORTH => self.energy_ns[(tile2 as usize, tile1 as usize)],
            EAST => self.energy_we[(tile1 as usize, tile2 as usize)],
            SOUTH => self.energy_ns[(tile1 as usize, tile2 as usize)],
            WEST => self.energy_we[(tile2 as usize, tile1 as usize)],
            _ => panic!("Must enter NESW"),
        }
    }

    /// Energy of neighbour bonds
    pub fn energy_at_point<S: State>(&self, state: &S, point: PointSafe2) -> Energy {
        let tile_id: TileId = state.tile_at_point(point);
        let mut energy = Self::ZERO_RATE;
        for side in ALL_SIDES {
            let neighbour_tile = Self::tile_to_side(state, side, point);
            energy += self.energy_to(side, tile_id, neighbour_tile)
        }
        energy + self.alpha * self.rtval()
    }

    #[inline(always)]
    fn rtval(&self) -> Energy {
        R * (self.temperature + 273.15)
    }

    pub fn tile_detachment_rate<S: State>(&self, state: &S, p: PointSafe2) -> Rate {
        if self.is_seed(&p) {
            return Self::ZERO_RATE;
        }

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
        self.tile_concentration(tile) * self.kf
    }

    pub fn cover_attachment_rate_at_side(&self, side: Side, tile: TileId) -> Rate {
        self.kf * self.cover_concentrations[self.glue_on_side(side, tile)]
    }

    /// Get the energy between a tile and a cover to some side
    pub fn cover_detachment_rate_at_side(&self, side: Side, tile: TileId) -> Rate {
        // If there is no cover in that side, then the detachment rate will be 0
        if !is_covered(side, tile) {
            return Self::ZERO_RATE;
        };

        let tile = uncover_all(tile);
        self.kf
            * (self.energy_cover[tile_index(tile)][side_index(side).expect("Side must be NESW")]
                * (1.0 / self.rtval())
                + self.alpha)
                .exp()
    }

    pub fn cover_detachment_total_rate(&self, tile: TileId) -> Rate {
        self.cover_detachment_rate_at_side(NORTH, tile)
            + self.cover_detachment_rate_at_side(EAST, tile)
            + self.cover_detachment_rate_at_side(SOUTH, tile)
            + self.cover_detachment_rate_at_side(WEST, tile)
    }

    fn maybe_detach_cover_on_side_event(
        &self,
        tileid: TileId,
        point: PointSafe2,
        side: Side,
        acc: &mut Rate,
    ) -> Option<(bool, Rate, Event)> {
        // Something cannot detach if there is no cover
        if !is_covered(side, tileid) {
            return None;
        }
        *acc -= self.cover_detachment_rate_at_side(side, tileid);
        if *acc <= 0.0 {
            // ^ SIDE will change the bit from 1 to 0, so no longer have a cover here
            Some((true, *acc, Event::MonomerChange(point, tileid ^ side)))
        } else {
            None
        }
    }

    /// Detach a cover from tile
    pub fn event_cover_detachment<S: State>(
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
        self.maybe_detach_cover_on_side_event(tile, point, NORTH, acc)
            .or(self.maybe_detach_cover_on_side_event(tile, point, EAST, acc))
            .or(self.maybe_detach_cover_on_side_event(tile, point, SOUTH, acc))
            .or(self.maybe_detach_cover_on_side_event(tile, point, WEST, acc))
            .unwrap_or((false, *acc, Event::None))
    }

    pub fn tile_to_side<S: State>(state: &S, side: Side, p: PointSafe2) -> TileId {
        match side {
            NORTH => state.tile_to_n(p),
            EAST => state.tile_to_e(p),
            SOUTH => state.tile_to_s(p),
            WEST => state.tile_to_w(p),
            _ => panic!("Side must be North, South, East, or West"),
        }
    }

    fn maybe_attach_cover_on_side_event<S: State>(
        &self,
        tileid: TileId,
        side: Side,
        point: PointSafe2,
        state: &S,
        acc: &mut Rate,
    ) -> Option<(bool, Rate, Event)> {
        // A cover cannot attach to a side with a cover already attached
        if is_covered(side, tileid)
        // If a tile is already attached to that side, then nothing can attach
            || Self::tile_to_side(state,side, point) != 0
        {
            return None;
        }

        *acc -= self.kf * self.cover_concentrations[self.glue_on_side(side, tileid)];
        if *acc <= 0.0 {
            // | SIDE will change the bit from 0 to 1
            Some((true, *acc, Event::MonomerChange(point, tileid | side)))
        } else {
            None
        }
    }

    // Attach a cover to a tile
    pub fn event_cover_attachment<S: State>(
        &self,
        state: &S,
        point: PointSafe2,
        acc: &mut Rate,
    ) -> (bool, Rate, Event) {
        let tile = state.tile_at_point(point);
        if tile == 0 {
            return (false, 0.0, Event::None);
        }
        self.maybe_attach_cover_on_side_event(tile, NORTH, point, state, acc)
            .or(self.maybe_attach_cover_on_side_event(tile, EAST, point, state, acc))
            .or(self.maybe_attach_cover_on_side_event(tile, SOUTH, point, state, acc))
            .or(self.maybe_attach_cover_on_side_event(tile, WEST, point, state, acc))
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

    /// Get all possible cover combinations of some tile, such that some side is uncovered
    pub fn cover_combinations(uncovered_side: Side, tile: TileId) -> Vec<TileId> {
        (0..16)
            .filter_map(|cover| {
                if cover & uncovered_side != 0 {
                    None
                } else {
                    Some(tile | cover)
                }
            })
            .collect()
    }

    /// Get all possible tiles that may attach at some given point
    ///
    /// This will return both, the tile that is attaching, as well as which side it is attaching
    /// to. This will prevent us from overwritting duplicates. Ie. a tile with no covers may attach
    /// to the north east west or south, so it is more likely to attach than a tile with covers in
    /// the north, east, and south (assuming equal concentrations)
    pub fn possible_tiles_at_point<S: State>(
        &self,
        state: &S,
        point: PointSafe2,
    ) -> HashSetType<(Side, TileId)> {
        let tile = state.tile_at_point(point);
        let mut friends: HashSetType<(Side, TileId)> = HashSet::default();

        // tile aready attached here
        if tile != 0 {
            return friends;
        }

        for side in ALL_SIDES {
            let neighbour = Self::tile_to_side(state, side, point);
            if neighbour == 0 {
                continue;
            }

            if let Some(possible_attachments) =
                self.get_uncovered_friends_to_side(inverse(side), neighbour)
            {
                let attachments: HashSetType<(Side, TileId)> = HashSet::from_iter(
                    possible_attachments
                        .iter()
                        .flat_map(|&tile| Self::cover_combinations(side, tile))
                        .map(|tile| (side, tile)),
                );
                friends.extend(attachments);
            }
        }
        friends
    }

    pub fn total_attachment_rate_at_point<S: State>(&self, point: PointSafe2, state: &S) -> Rate {
        self.possible_tiles_at_point(state, point)
            .iter()
            .fold(0.0, |acc, &(_side, tile)| {
                acc + (self.kf * self.tile_concentration(tile))
            })
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

        let friends: HashSetType<(Side, TileId)> = self.possible_tiles_at_point(state, point);
        // attachment_side is not used, but is relevant in computation, as it accounts for
        // duplicates (some tiles could bind to the north or east, so it should be taken into
        // account twice)
        for (_attachment_side, tile) in friends {
            *acc -= self.kf * self.tile_concentration(tile);
            if *acc <= 0.0 {
                return (true, *acc, Event::MonomerAttachment(point, tile));
            }
        }
        (false, *acc, Event::None)
    }

    /// Percentage of total concentration of some tile that has a cover on a given side
    pub fn cover_percentage(&self, side: Side, tile: TileId) -> f64 {
        let detachment_rate = self.cover_detachment_rate_at_side(side, tile | side);
        let attachment_rate = self.cover_attachment_rate_at_side(side, tile);
        attachment_rate / (attachment_rate + detachment_rate)
    }

    /// Get the concentration of a specific tile, with cover as given in the TileId
    pub fn tile_concentration(&self, tile: TileId) -> f64 {
        let mut acc = 1.0;
        for side in ALL_SIDES {
            let cover_perc = self.cover_percentage(side, tile);
            if is_covered(side, tile) {
                acc *= cover_perc;
            } else {
                acc *= 1.0 - cover_perc;
            }
        }
        self.tile_concentration[tile_index(tile)] * acc
    }

    pub fn total_cover_attachment_rate<S: State>(&self, state: &S, point: PointSafe2) -> Rate {
        // Check that there is a tile at this point
        let tile = state.tile_at_point(point);
        if tile == 0 {
            return Self::ZERO_RATE;
        }

        let mut rate = Self::ZERO_RATE;
        for s in ALL_SIDES {
            if !is_covered(s, tile) && Self::tile_to_side(state, s, point) == 0 {
                rate += self.kf * self.cover_concentrations[self.glue_on_side(s, tile)];
            }
        }
        return rate;
    }

    fn can_bond(&self, tile1: TileId, side: Side, tile2: TileId) -> bool {
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
        while !stack.is_empty() {
            let head = stack.pop().unwrap();
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
                if self.can_bond(head_tile, *side, neighbour_tile) {
                    stack.push(p);
                }
            });
        }
        visited
    }

    fn unseeded<S: State>(&self, state: &S, point: PointSafe2) -> Vec<PointSafe2> {
        let seed = self
            .seed_locs()
            .get(0)
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

impl SystemWithDimers for KCov {
    fn calc_dimers(&self) -> Vec<DimerInfo> {
        let mut dvec = Vec::new();

        for (t1, _) in self.tile_concentration.iter().enumerate() {
            let t1 = t1 << 4;
            if let Some(friends) = self.get_uncovered_friends_to_side(EAST, t1 as u32) {
                for t2 in friends.iter() {
                    let biconc =
                        self.tile_concentration(t1 as u32) * self.tile_concentration(*t2 as u32);
                    dvec.push(DimerInfo {
                        t1: t1 as u32,
                        t2: *t2,
                        orientation: Orientation::WE,
                        formation_rate: self.kf * biconc,
                        equilibrium_conc: biconc
                            * (self.energy_we[(tile_index(t1 as u32), tile_index(*t2 as u32))]
                                / self.rtval()
                                - self.alpha)
                                .exp(),
                    });
                }
            }

            if let Some(friends) = self.get_uncovered_friends_to_side(SOUTH, t1 as u32) {
                for t2 in friends.iter() {
                    let biconc =
                        self.tile_concentration(t1 as u32) * self.tile_concentration(*t2 as u32);
                    dvec.push(DimerInfo {
                        t1: t1 as u32,
                        t2: *t2,
                        orientation: Orientation::NS,
                        formation_rate: self.kf * biconc,
                        equilibrium_conc: biconc
                            * (self.energy_ns[(tile_index(t1 as u32), tile_index(*t2 as u32))]
                                / self.rtval()
                                - self.alpha)
                                .exp(),
                    });
                }
            }
        }

        dvec
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
        self.tile_colors[uncover_all(tileid) as usize]
    }

    fn tile_name(&self, tileid: TileId) -> &str {
        self.tile_names[uncover_all(tileid) as usize].as_str()
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

impl System for KCov {
    fn system_info(&self) -> String {
        format!("{:?}", self)
    }

    fn perform_event<St: State>(&self, state: &mut St, event: &Event) -> &Self {
        match event {
            Event::None => panic!("Canot perform None event"),
            Event::MonomerDetachment(point) => state.set_sa(point, &0),
            // Monomer Change -- Covers
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
    ) -> crate::base::Rate {
        let p = if state.inbounds(p.0) {
            PointSafe2(p.0)
        } else {
            return Self::ZERO_RATE;
        };
        let tile = { state.tile_at_point(p) };
        if tile != 0 {
            self.tile_detachment_rate(state, p)
                + self.cover_detachment_total_rate(tile)
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
        if let (true, _, event) = self.event_cover_attachment(state, point, &mut acc) {
            return event;
        }
        if let (true, _, event) = self.event_cover_detachment(state, point, &mut acc) {
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
        self.seed.clone().into_iter().collect::<Vec<_>>()
    }

    fn calc_mismatch_locations<St: crate::state::State>(&self, state: &St) -> Array2<usize> {
        // cge: Roughly copied from kTAM, because I need this for playing with
        // some algorithmic self-assembly stuff.

        let threshold = -0.05; // Todo: fix this (note for kCov, energies are negative)
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

                let tn = state.tile_to_n(p);
                let te = state.tile_to_e(p);
                let ts = state.tile_to_s(p);
                let tw = state.tile_to_w(p);                
                
                let mm_n = ((tn != 0) & (self.energy_to(NORTH, t, tn) > threshold)) as usize;
                let mm_e = ((te != 0) & (self.energy_to(EAST, t, te) > threshold)) as usize;
                let mm_s = ((ts != 0) & (self.energy_to(SOUTH, t, ts) > threshold)) as usize;
                let mm_w = ((tw != 0) & (self.energy_to(WEST, t, tw) > threshold)) as usize;

                mismatch_locations[(i, j)] = 8 * mm_n + 4 * mm_e + 2 * mm_s + mm_w;
            }
        }

        mismatch_locations
    }
}

#[cfg(test)]
mod test_covtile {
    use crate::models::kcov::{
        attach, detach, is_covered, tile_index, uncover_all, EAST, NORTH, WEST,
    };

    #[test]
    fn get_ids() {
        let mut t = 0b10110000;
        t = attach(EAST, t);
        assert_eq!(uncover_all(t), 0b10110000);
        assert_eq!(t, 0b10110000 | EAST);

        let mut k = 0b10000;
        k = attach(EAST, k);
        k = attach(WEST, k);
        assert_eq!(uncover_all(k), 16);
    }

    #[test]
    fn test_tile_index() {
        for i in 0..16 {
            let x = 0b10000;
            assert_eq!(1, tile_index(x | i))
        }
    }

    #[test]
    fn is_covered_side() {
        assert!(is_covered(NORTH, NORTH));
        assert!(is_covered(NORTH, 123 << 4 | NORTH));
        assert!(!is_covered(EAST, 123 << 4 | NORTH));
    }

    #[test]
    fn detach_side() {
        assert_eq!(0, detach(NORTH, NORTH));
        assert_eq!(123 << 4, detach(NORTH, (123 << 4) | NORTH));
    }
}

#[cfg(test)]
mod test_kcov {
    use crate::{
        state::StateEnum,
        tileset::{CanvasType, TrackingType},
    };

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
            HashMap::default(),
            60.0,
            1e6,
            0.0,
            FissionHandling::JustDetach,
        )
    }

    #[test]
    fn glue_side() {
        let kdcov = sample_kcov();
        assert_eq!(kdcov.glue_on_side(NORTH, 1 << 4), 1);
        assert_eq!(kdcov.glue_on_side(SOUTH, 1 << 4), 0);
        assert_eq!(kdcov.glue_on_side(WEST, 3 << 4), 4);
    }

    #[test]
    fn friends_build() {
        let mut kdcov = sample_kcov();
        //println!("Tile Names: {:?}", kdcov.tile_names());
        kdcov.fill_friends();

        //println!("Tile Names: {:?}", kdcov.tile_names);
        println!("N: {:?}", kdcov.north_friends);
        //println!("S: {:?}", kdcov.south_friends);
        //println!("E: {:?}", kdcov.east_friends);
        //println!("W: {:?}", kdcov.west_friends);

        let mut expected_nf = HashSetType::default();

        expected_nf.insert(2 << 4);
        // This is a little strange to use, as you need to know the glue on the north side of the
        // tile.
        assert_eq!(kdcov.north_friends[1], expected_nf);
        // These helper methods make it so that you can find every tile that can bond to the north
        // of some tile id
        assert_eq!(
            kdcov.get_uncovered_friends_to_side(NORTH, 1 << 4),
            Some(&expected_nf)
        );
        assert_eq!(kdcov.get_friends(NORTH, 1 << 4), expected_nf);
        // You can also get frineds to multiple sides at once
        assert_eq!(kdcov.get_friends(NORTH | EAST, 1 << 4), expected_nf);

        let mut expected_wf = HashSetType::default();
        expected_wf.insert(2);
        assert_eq!(kdcov.west_friends[4], expected_nf);
        assert_eq!(kdcov.get_friends(WEST, 3 << 4), expected_nf);
    }

    #[test]
    fn test_bfs() {
        let tile_a = KCovTile {
            name: "TileA".to_string(),
            concentration: 1e-2,
            glues: ["A", "A", "A*", "A*"].map(String::from),
            color: [0, 0, 0, 0],
        };
        let mut kcov: KCov = KCovParams {
            tiles: vec![tile_a],
            cover_conc: HashMap::from([
                (GlueIdentifier::Index(0), 1e6),
                (GlueIdentifier::Index(1), 1e6),
            ]),
            alpha: 1.0,
            kf: 1.0,
            temp: 40.0,
            seed: HashMap::default(),
            binding_strength: HashMap::default(),
        }
        .into();

        let mut se =
            StateEnum::empty((20, 20), CanvasType::Square, TrackingType::None, 40).unwrap();
        // Add a seed tile (just one)
        kcov.add_seed(&mut se, HashMap::from([(PointSafe2((2, 2)), 1 << 4)]));

        // Make an L
        se.set_sa(&PointSafe2((3, 2)), &(1 << 4));
        se.set_sa(&PointSafe2((4, 2)), &(1 << 4));
        se.set_sa(&PointSafe2((5, 2)), &(1 << 4));
        se.set_sa(&PointSafe2((6, 2)), &(1 << 4));
        se.set_sa(&PointSafe2((5, 3)), &(1 << 4));
        se.set_sa(&PointSafe2((5, 4)), &(1 << 4));

        let removals = kcov.unseeded(&se, PointSafe2((3, 2)));
        println!("{:?}", removals);
        assert_eq!(removals.len(), 5);
        assert!(removals.contains(&PointSafe2((4, 2))));
        assert!(removals.contains(&PointSafe2((5, 2))));
        assert!(removals.contains(&PointSafe2((5, 3))));
        assert!(removals.contains(&PointSafe2((5, 4))));
        assert!(removals.contains(&PointSafe2((6, 2))));

        let removals = kcov.unseeded(&se, PointSafe2((5, 2)));
        assert_eq!(removals.len(), 3);
        assert!(removals.contains(&PointSafe2((5, 3))));
        assert!(removals.contains(&PointSafe2((5, 4))));
        assert!(removals.contains(&PointSafe2((6, 2))));
    }
}

// Python Bindings

#[derive(Debug, Clone, Serialize, Deserialize)]
struct KCovTile {
    pub name: String,
    pub concentration: f64,
    /// Glues for the tiles North, East, South, West in that order
    pub glues: [String; 4],
    /// Color of the tile, this is used only when displaying
    pub color: [u8; 4],
}

#[cfg(feature = "python")]
impl pyo3::FromPyObject<'_> for KCovTile {
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
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?
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

impl KCovTile {
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
    DG(Strength),
    Sequence(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "python", derive(pyo3::FromPyObject))]
pub enum TileIdentifier {
    Id(TileId),
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
struct KCovParams {
    pub tiles: Vec<KCovTile>,
    pub cover_conc: HashMap<GlueIdentifier, Concentration>,
    pub seed: HashMap<(usize, usize), TileIdentifier>,
    pub binding_strength: HashMap<String, StrenOrSeq>,
    pub alpha: f64,
    pub kf: f64,
    pub temp: f64,
}

/// Given some glue, in the form (a|z)+* or (a|z), return itself, as well as its inverse
fn base_inv(mut s: String) -> (String, String) {
    if s.ends_with("*") {
        s.pop();
    }
    (s.clone(), format!("{s}*"))
}

impl From<KCovParams> for KCov {
    fn from(mut value: KCovParams) -> Self {
        let mut tiles = Vec::with_capacity(value.tiles.len() + 1);
        tiles.push(KCovTile::empty());
        tiles.append(&mut value.tiles);
        let tile_names = tiles.iter().map(|tile| tile.name.clone()).collect();
        let tile_concentration = tiles.iter().map(|tile| tile.concentration).collect();
        let tile_colors = tiles.iter().map(|tile| tile.color).collect();
        
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

        // Generate cover concentrations vector from HashMap
        let mut cover_concentrations = vec![0.0; glue_id];
        for (glue_id, conc) in value.cover_conc {
            match glue_id {
                GlueIdentifier::Index(index) => {
                    if index < cover_concentrations.len() {
                        cover_concentrations[index] = conc;
                    }
                },
                GlueIdentifier::Name(name) => {
                    if let Some(&index) = glue_hashmap.get(&name) {
                        cover_concentrations[index] = conc;
                    }
                }
            }
        }

        // Process seed with either TileId or tile name
        let seed = value.seed.iter().map(|(pos, tile_id_or_name)| {
            let tile_id = match tile_id_or_name {
                TileIdentifier::Id(id) => *id,
                TileIdentifier::Name(name) => {
                    // Find position in tile_names and convert to TileId
                    let pos = tiles.iter().position(|t| t.name == *name)
                        .expect(&format!("Tile name '{}' not found", name));
                    (pos as TileId) << 4
                }
            };
            (PointSafe2(*pos), tile_id)
        }).collect();

        // Make sure that every glue has its inverse in the array
        let mut glue_links = Array2::zeros((glue_id + 1, glue_id + 1));
        for (glue, strength_new) in value.binding_strength {
            let (glue_str, inverse_str) = base_inv(glue);

            let msg = "Glue was assigned strength, but was not used";
            let glue = *glue_hashmap.get(&glue_str).expect(msg);
            let inverse = *glue_hashmap.get(&inverse_str).expect(msg);

            // Lets check that we are in bounds
            if glue_links.get((glue, inverse)).is_none() {
                panic!(
                    "({:?} {:?}) not in index ({:?} {:?})",
                    glue, inverse, glue_id, glue_id
                );
            }

            let stren_dg = match strength_new {
                StrenOrSeq::DG(dg) => dg,
                StrenOrSeq::Sequence(seq) => {
                    // If we want annealing, we need to save the sequences, or store dh & ds
                    // instead of dg
                    // cge: it turns out I made some mistakes with alpha.  We need to use RT*alpha here, which messes
                    // up temperature dependence.  But that doesn't matter right now I suppose.
                    crate::utils::string_dna_delta_g(&seq, value.temp) - R * (273.15 + value.temp) * value.alpha
                }
            };

            glue_links[(glue, inverse)] = stren_dg;
            glue_links[(inverse, glue)] = stren_dg;
        }

        let mut glue_names = vec!["".to_string(); glue_id];
        for (name, id) in glue_hashmap {
            glue_names[id] = name;
        }

        Self::new(
            tile_names,
            tile_concentration,
            tile_colors,
            glue_names,
            cover_concentrations,
            tile_glues,
            glue_links,
            seed,
            value.temp,
            value.kf,
            value.alpha,
            FissionHandling::JustDetach,
        )
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
                    self.$name
                }

                #[setter($name)]
                fn [<py_set_ $name>](&mut self, to: $t) {
                    self.$name = to;
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

getset!(KCov,
    // f64 getters and setters
    (f64 => kf, alpha, temperature)
);

#[cfg(feature = "python")]
#[pymethods]
impl KCov {
    #[new]
    fn kcov_from_params(kcov_params: KCovParams) -> Self {
        Self::from(kcov_params)
    }

    /// Get the concentration of a tile with given covers
    fn tile_conc(&self, tile: TileId) -> Concentration {
        self.tile_concentration(tile)
    }

    /// Print a string breaking down the total rate at some point
    fn detailed_rate_at_point(&self, state: &PyState, point: (usize, usize)) {
        let point = PointSafe2(point);
        let tile = state.0.tile_at_point(point);

        if tile == 0 {
            let a_rate = self.total_attachment_rate_at_point(point, &state.0);
            println!("Attachment rate: {a_rate}");
            return;
        }

        let mut acc = String::new();
        for side in ALL_SIDES {
            let (kind, rate) = if is_covered(side, tile) {
                let rate = self.cover_detachment_rate_at_side(side, tile);
                ("detachment", rate)
            } else {
                // This assumes that there is nothing to the side, so a cover can in fact attach
                let mut rate = self.cover_attachment_rate_at_side(side, tile);
                // If there is a tile already attached on that side, then the attachment rate is 0
                if Self::tile_to_side(&state.0, side, point) != 0 {
                    rate = 0.0;
                };
                ("attachment", rate)
            };

            let message = format!(
                "Cover {} rate on side {}: {}",
                kind,
                side_as_str(side),
                rate
            );
            acc.push_str(message.as_str());
            acc.push('\n');
        }
        // Tile detachment
        let detachment_rate = self.tile_detachment_rate(&state.0, point);
        acc.push_str(format!("Tile detachment rate {}", detachment_rate).as_str());
        println!("{}", acc)
    }
}
