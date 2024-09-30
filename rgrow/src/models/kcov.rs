use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::{
    base::{Glue, Tile},
    system::{System, TileBondInfo},
    type_alias,
};

type_alias!( f64 => Concentration );
type_alias!( u32 => TileId );

const NORTH: u32 = 0b1000;
const SOUTH: u32 = 0b0100;

const EAST: u32 = 0b0010;
const WEST: u32 = 0b0001;

const ALL_COVERS: u32 = 0b1111;
const NO_COVERS: u32 = !ALL_COVERS;

/// Helper methods for tile id
///
/// This can help change the id to attach a cover on the north position, etc ...
mod tileid_helper {
    use super::{TileId, ALL_COVERS, EAST, NORTH, SOUTH, WEST};

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
        id & (!ALL_COVERS)
    }
}

struct KCov {
    pub tile_names: Vec<String>,
    pub tile_concentration: Vec<Concentration>,
    pub tile_colors: Vec<[u8; 4]>,

    pub kf: f64,
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
