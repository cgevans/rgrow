use crate::colors::Color;

/// How to render a single tile.
///
/// To give some flexibility, you can give a tile 4 colors, when drawing the tile,
/// it will be treated as 4 triangles rather than one square. The colors provided are
/// for the NSEW triangles, in that order
#[derive(Debug, Clone, Copy)]
struct TileStyle {
    /// RGBA values for each of the sides of a tile
    tri_colors: [Color; 4],
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
}

/// Let a `System` decide how the DNA should be painted in the GUI. This trait needs to be
/// implemented in order for the system to be usable with the GUI.
trait TilePainter<TileId> {
    /// Given some tile id, choose how it should be rendered in the GUI
    fn tile_style(&self, id: TileId) -> TileStyle;
}
