use crate::type_alias;

type_alias!(
    f64 => Concentration
);

const NORTH: u32 = 0b1000;
const SOUTH: u32 = 0b0100;

const EAST: u32 = 0b0010;
const WEST: u32 = 0b0001;

const ALL_COVERS: u32 = 0b1111;

/// Structure for tile with covers
pub struct CovTile {
    /// The last four binary digits are reserved for cover status -- This will allow us to quickly
    /// find the id of a tile that is identical, but with/without a cover
    id: u32,
}

impl CovTile {
    pub fn new(id: u32) -> Self {
        Self { id }
    }

    pub fn attachments(&self) -> u32 {
        self.id & ALL_COVERS
    }

    pub fn attach_south(&mut self) {
        self.id |= SOUTH;
    }

    pub fn attach_north(&mut self) {
        self.id |= NORTH;
    }

    pub fn attach_east(&mut self) {
        self.id |= EAST;
    }

    pub fn attach_west(&mut self) {
        self.id |= WEST;
    }

    pub fn id(&self) -> u32 {
        self.id
    }

    /// Get the "base id", this is the id of the tile if it had no covers
    pub fn base_id(&self) -> u32 {
        self.id & (!ALL_COVERS)
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
    use crate::models::kcov::EAST;

    use super::CovTile;

    fn test_tile() -> CovTile {
        CovTile::new(0b10110000)
    }

    #[test]
    fn get_ids() {
        let mut t = test_tile();
        t.attach_east();

        assert_eq!(t.base_id(), 0b10110000);
        assert_eq!(t.id(), 0b10110000 | EAST);
    }
}
