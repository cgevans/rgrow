use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::{
    base::{Glue, Tile},
    system::{System, TileBondInfo},
    type_alias,
};

type_alias!( f64 => Concentration );
type_alias!( u32 => TileId );

const NORTH: u32 = 0b1000 << 28;
const SOUTH: u32 = 0b0100 << 28;

const EAST: u32 = 0b0010 << 28;
const WEST: u32 = 0b0001 << 28;

const ALL_COVERS: u32 = NORTH | SOUTH | EAST | WEST;
const NO_COVERS: u32 = !ALL_COVERS;

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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct KCov {
    pub tile_names: Vec<String>,
    pub tile_concentration: Vec<Concentration>,
    pub tile_colors: Vec<[u8; 4]>,

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
    tile_glues: Array2<Glue>,

    pub kf: f64,
}

impl KCov {
    /// Get the glues. If there are covers, this wil look past them, and return the
    /// glue that is under it
    pub fn get_tile_raw_glues(&self, tile_id: TileId) -> Vec<Glue> {
        let id = tileid_helper::base_id(tile_id) as usize;
        self.tile_glues.row(id).to_vec()
    }

    pub fn get_tile_uncovered_glues(&self, tile_id: TileId) -> Vec<Glue> {
        // This MUST be exactly 4 length
        let row = self.get_tile_raw_glues(tile_id);
        let mut glues = vec![0; 4];

        // If any of the sides have a cover, then the glue is 0
        if tile_id & NORTH != 0 {
            glues[0] = row[0];
        }
        if tile_id & SOUTH != 0 {
            glues[1] = row[1]
        }
        if tile_id & EAST != 0 {
            glues[2] = row[2];
        }
        if tile_id & WEST != 0 {
            glues[3] = row[3]
        }
        glues
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
*    id. Then to see if two tiles have the same base id,
*    we can just do id & 11...1110000
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

#[cfg(test)]
mod test_covtile {
    use crate::models::kcov::{tileid_helper, EAST};

    #[test]
    fn get_ids() {
        let mut t = 0b10110000;
        t = tileid_helper::attach_east(t);
        assert_eq!(tileid_helper::base_id(t), 0b10110000);
        assert_eq!(t, 0b10110000 | EAST);
    }
}
