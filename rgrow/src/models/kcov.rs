use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

use crate::{
    base::{Glue, HashSetType},
    system::{System, TileBondInfo},
    type_alias,
};

type_alias!( f64 => Concentration, Strength );
type_alias!( u32 => TileId );

const NORTH: u32 = 0b1000 << 28;
const SOUTH: u32 = 0b0100 << 28;

const EAST: u32 = 0b0010 << 28;
const WEST: u32 = 0b0001 << 28;

const ALL_COVERS: u32 = NORTH | SOUTH | EAST | WEST;
const NO_COVERS: u32 = !ALL_COVERS;

const ALL_SIDES: [u32; 4] = [NORTH, SOUTH, EAST, WEST];

/// Helper methods for tile id
///
/// This can help change the id to attach a cover on the north position, etc ...
mod tileid_helper {
    use super::{TileId, ALL_COVERS, EAST, NORTH, NO_COVERS, SOUTH, WEST};

    pub fn attachments(id: TileId) -> TileId {
        id & ALL_COVERS
    }

    pub fn attach_south(id: TileId) -> TileId {
        id | SOUTH
    }

    pub fn attach_north(id: TileId) -> TileId {
        id | NORTH
    }

    pub fn attach_east(id: TileId) -> TileId {
        id | EAST
    }

    pub fn attach_west(id: TileId) -> TileId {
        id | WEST
    }

    /// Get the "base id", this is the id of the tile if it had no covers
    pub fn base_id(id: TileId) -> TileId {
        id & NO_COVERS
    }

    pub fn is_covered<const SIDE: TileId>(id: TileId) -> bool {
        (id & SIDE) == 0
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct KCov {
    pub tile_names: Vec<String>,
    pub tile_concentration: Vec<Concentration>,
    pub tile_colors: Vec<[u8; 4]>,

    pub glue_names: Vec<String>,

    /// Glues of a tile with a given ID
    ///
    /// This is private purposely, use getter function. There are (up to / exactly)
    /// 16 different ids with the same glues, thus we will share ids. This is
    /// enforced by the getter
    ///
    /// [
    ///     (0) -- [North, South, East, West]
    ///     ...
    ///     (n) -- [North, South, East, West]
    /// ]
    tile_glues: Array1<[Glue; 4]>,
    glue_links: Array1<[Strength; 4]>,

    // This hashing shuold be without last four bits -- This would save a lot of space,
    // and may also help with logic
    //
    // For example, if we know that some id has the right shouth glue, we instantly
    // know the 8 possible tiles that can attach to us
    pub north_friends: Vec<HashSetType<TileId>>,
    pub south_friends: Vec<HashSetType<TileId>>,
    pub east_friends: Vec<HashSetType<TileId>>,
    pub west_friends: Vec<HashSetType<TileId>>,

    pub kf: f64,
}

#[rustfmt::skip]
#[inline(always)]
/// Glue cannot be 0
fn glue_inverse(glue: Glue) -> Glue {
    if glue % 2 == 1 { glue + 1 } else { glue - 1 }
}

impl KCov {
    pub fn new(
        tile_names: Vec<String>,
        tile_concentration: Vec<Concentration>,
        tile_colors: Vec<[u8; 4]>,
        glue_names: Vec<String>,
        tile_glues: Array1<[Glue; 4]>,
        glue_links: Array1<[Strength; 4]>,
        kf: f64,
    ) -> Self {
        Self {
            tile_names,
            tile_concentration,
            tile_colors,
            glue_names,
            tile_glues,
            glue_links,
            north_friends: Vec::default(),
            south_friends: Vec::default(),
            east_friends: Vec::default(),
            west_friends: Vec::default(),
            kf,
        }
    }

    // Side must be north, south, east, or west
    pub fn get_friends_one_side<const SIDE: TileId>(
        &self,
        tile: TileId,
    ) -> Option<&HashSetType<TileId>> {
        // The tile is covered, so we dont have any friends
        if tile & SIDE != 0 {
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
        match SIDE {
            NORTH => glues[0],
            SOUTH => glues[1],
            EAST => glues[2],
            WEST => glues[3],
            _ => panic!("Side must be NESW"),
        }
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
        if tile_id & SOUTH == 0 {
            glues[1] = row[1]
        }
        if tile_id & EAST == 0 {
            glues[2] = row[2];
        }
        if tile_id & WEST == 0 {
            glues[3] = row[3]
        }
        glues
    }

    fn fill_friends(&mut self) {
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
        for (id, [ng, sg, eg, wg]) in self.tile_glues.iter().enumerate() {
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
        self.south_friends = sf;
        self.east_friends = ef;
        self.west_friends = wf;
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

    fn update_after_event<St: crate::state::State>(
        &self,
        state: &mut St,
        event: &crate::system::Event,
    ) {
        todo!()
    }

    fn event_rate_at_point<St: crate::state::State>(
        &self,
        state: &St,
        p: crate::canvas::PointSafeHere,
    ) -> crate::base::Rate {
        todo!()
    }

    fn choose_event_at_point<St: crate::state::State>(
        &self,
        state: &St,
        p: crate::canvas::PointSafe2,
        acc: crate::base::Rate,
    ) -> crate::system::Event {
        todo!()
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
    use crate::models::kcov::{tileid_helper, EAST};

    #[test]
    fn get_ids() {
        let mut t = 0b10110000;
        t = tileid_helper::attach_east(t);
        assert_eq!(tileid_helper::base_id(t), 0b10110000);
        assert_eq!(t, 0b10110000 | EAST);

        let mut k = 1;
        k = tileid_helper::attach_east(k);
        k = tileid_helper::attach_west(k);
        assert_eq!(tileid_helper::base_id(k), 1);
    }
}

#[cfg(test)]
mod test_kcov {
    use super::*;

    fn sample_kcov() -> KCov {
        const DEFAULT_COLOR: [u8; 4] = [0, 0, 0, 0];
        let tile_glues = Array1::from_vec(vec![
            [0, 0, 0, 0], // zero tile -- Empty
            // N S E W
            [1, 0, 0, 0], // f
            [0, 2, 3, 0], // s
            [0, 0, 0, 4], // t
        ]);

        // The example tiles may attach like this:
        //  s t
        //  f

        let glue_linkns = Array1::from_vec(vec![[1., 1., 1., 1.]; 4]);
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
            tile_glues,
            glue_linkns,
            1e6,
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
        expected_wf.insert(3);
        assert_eq!(kdcov.west_friends[4], expected_nf);
        assert_eq!(kdcov.get_friends::<WEST>(3), expected_nf);
    }
}
