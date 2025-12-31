use crate::colors::Color;

/// How to render a single tile.
///
/// To give some flexibility, you can give a tile 4 colors, when drawing the tile,
/// it will be treated as 4 triangles rather than one square. The colors provided are
/// for the NSEW triangles, in that order
#[derive(Debug, Clone, Copy)]
pub struct TileStyle {
    /// RGBA values for each of the sides of a tile
    tri_colors: [Color; 4],
}

pub struct SpriteSquare {
    pub size: usize,
    pub pixels: Box<[u8]>,
}

impl TileStyle {
    #[inline(always)]
    pub fn north_color(&self) -> Color {
        self.tri_colors[0]
    }

    #[inline(always)]
    pub fn east_color(&self) -> Color {
        self.tri_colors[1]
    }

    #[inline(always)]
    pub fn south_color(&self) -> Color {
        self.tri_colors[2]
    }

    #[inline(always)]
    pub fn west_color(&self) -> Color {
        self.tri_colors[3]
    }

    /// Generate the colors for the tile
    pub fn as_sprite(&self, size: usize) -> SpriteSquare {
        let mut pixels = vec![0; size * size * 4];
        for row in 0..size {
            for col in 0..size {
                // North or east
                let ne = col > row;
                let se = row > size - col - 1;
                let color = match (ne, se) {
                    (true, true) => self.east_color(),
                    (true, false) => self.north_color(),
                    (false, true) => self.south_color(),
                    (false, false) => self.west_color(),
                };
                let idx = 4 * (row * size + col);
                pixels[idx..idx + 4].copy_from_slice(color.as_slice());
            }
        }

        SpriteSquare {
            size,
            pixels: pixels.into_boxed_slice(),
        }
    }
}

/// Let a `System` decide how the DNA should be painted in the GUI. This trait needs to be
/// implemented in order for the system to be usable with the GUI.
pub trait TilePainter {
    type TileId;

    /// Given some tile id, choose how it should be rendered in the GUI
    fn tile_style(&self, id: Self::TileId) -> TileStyle;

    /// Turn the tile into a sprite
    fn tile_pixels(&self, id: Self::TileId, size: usize) -> SpriteSquare {
        self.tile_style(id).as_sprite(size)
    }
}
