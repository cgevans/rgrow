use ndarray::ArrayView2;

use crate::base::Tile;
use crate::colors::Color;

/// Draw a filled rectangle into an RGBA frame buffer.
/// Pixel region is `x_min..x_max, y_min..y_max` (exclusive upper bounds).
/// `frame_width`: width of the frame in pixels (for row stride calculation).
pub fn draw_rect(
    frame: &mut [u8],
    x_min: usize,
    x_max: usize,
    y_min: usize,
    y_max: usize,
    color: [u8; 4],
    frame_width: usize,
) {
    for py in y_min..y_max {
        for px in x_min..x_max {
            let idx = (py * frame_width + px) * 4;
            if idx + 4 <= frame.len() {
                frame[idx..idx + 4].copy_from_slice(&color);
            }
        }
    }
}

/// How to render a single tile.
///
/// To give some flexibility, you can give a tile 4 colors, when drawing the tile,
/// it will be treated as 4 triangles rather than one square. The colors provided are
/// for the NSEW triangles, in that order
#[derive(Debug, Clone, Copy)]
pub struct TileStyle {
    /// RGBA values for each of the sides of a tile
    pub tri_colors: [Color; 4],
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

/// Blit a pre-rendered tile sprite into the frame buffer at tile grid position
/// (`grid_x`, `grid_y`). `frame_width_px` is the frame width in pixels.
pub fn blit_sprite(
    frame: &mut [u8],
    sprite: &SpriteSquare,
    grid_x: usize,
    grid_y: usize,
    frame_width_px: usize,
) {
    let tile_size = sprite.size;
    let tile_width_bytes = tile_size * 4;
    let frame_stride = frame_width_px * 4;
    let start = grid_y * tile_size * frame_stride + grid_x * tile_size * 4;
    for (e, pixel_row) in sprite.pixels.chunks(tile_width_bytes).enumerate() {
        let from = start + e * frame_stride;
        frame[from..from + tile_width_bytes].copy_from_slice(pixel_row);
    }
}

/// Render all tiles by blitting pre-computed sprites.
/// `sprites` is indexed by tile ID; entries beyond the slice length are skipped.
pub fn render_tiles(
    frame: &mut [u8],
    tiles: ArrayView2<Tile>,
    sprites: &[SpriteSquare],
    frame_width_px: usize,
) {
    for ((y, x), &tileid) in tiles.indexed_iter() {
        if let Some(sprite) = sprites.get(tileid as usize) {
            blit_sprite(frame, sprite, x, y, frame_width_px);
        }
    }
}

/// Draw 1px black outlines around non-empty tiles.
/// Only call when `scale >= 12` (caller checks threshold).
pub fn render_outlines(
    frame: &mut [u8],
    tiles: ArrayView2<Tile>,
    scale: usize,
    frame_width_px: usize,
) {
    let outline_color = [0u8, 0, 0, 255];
    for ((y, x), &tileid) in tiles.indexed_iter() {
        if tileid == 0 {
            continue;
        }
        let tx = x * scale;
        let ty = y * scale;
        // Top edge
        draw_rect(frame, tx, tx + scale, ty, ty + 1, outline_color, frame_width_px);
        // Bottom edge
        draw_rect(frame, tx, tx + scale, ty + scale - 1, ty + scale, outline_color, frame_width_px);
        // Left edge
        draw_rect(frame, tx, tx + 1, ty, ty + scale, outline_color, frame_width_px);
        // Right edge
        draw_rect(frame, tx + scale - 1, tx + scale, ty, ty + scale, outline_color, frame_width_px);
    }
}

/// Draw blocker rectangles protruding outside tile edges.
/// `blocker_masks` is indexed by tile ID; each value is a bitmask:
/// bit 0 = North, bit 1 = East, bit 2 = South, bit 3 = West.
pub fn render_blockers(
    frame: &mut [u8],
    tiles: ArrayView2<Tile>,
    blocker_masks: &[u8],
    scale: usize,
    frame_width_px: usize,
    frame_height_px: usize,
) {
    let depth = (scale / 3).max(2);
    let half_len = (scale / 3).max(2);
    let blocker_color = [140, 140, 140, 255];
    for ((y, x), &tileid) in tiles.indexed_iter() {
        let mask = blocker_masks.get(tileid as usize).copied().unwrap_or(0);
        if mask == 0 {
            continue;
        }
        let tile_x = x * scale;
        let tile_y = y * scale;
        let mid_x = tile_x + scale / 2;
        let mid_y = tile_y + scale / 2;
        // North blocker: rectangle above tile
        if mask & 0b0001 != 0 {
            draw_rect(
                frame,
                mid_x.saturating_sub(half_len),
                mid_x + half_len,
                tile_y.saturating_sub(depth),
                tile_y,
                blocker_color,
                frame_width_px,
            );
        }
        // East blocker: rectangle to the right
        if mask & 0b0010 != 0 {
            let right = tile_x + scale;
            draw_rect(
                frame,
                right,
                (right + depth).min(frame_width_px),
                mid_y.saturating_sub(half_len),
                mid_y + half_len,
                blocker_color,
                frame_width_px,
            );
        }
        // South blocker: rectangle below tile
        if mask & 0b0100 != 0 {
            let bottom = tile_y + scale;
            draw_rect(
                frame,
                mid_x.saturating_sub(half_len),
                mid_x + half_len,
                bottom,
                (bottom + depth).min(frame_height_px),
                blocker_color,
                frame_width_px,
            );
        }
        // West blocker: rectangle to the left
        if mask & 0b1000 != 0 {
            draw_rect(
                frame,
                tile_x.saturating_sub(depth),
                tile_x,
                mid_y.saturating_sub(half_len),
                mid_y + half_len,
                blocker_color,
                frame_width_px,
            );
        }
    }
}

/// Draw mismatch markers (red bars at tile boundaries).
/// `mismatch_locs` has the same shape as the tile grid.
/// Bit 0 = west-edge mismatch, bit 1 = south-edge mismatch.
pub fn render_mismatches(
    frame: &mut [u8],
    mismatch_locs: &ArrayView2<usize>,
    scale: usize,
    frame_width_px: usize,
) {
    let thick = (scale / 4).max(1);
    let long = (scale / 3).max(1);
    let color = [255, 0, 0, 255];
    for ((y, x), &mm) in mismatch_locs.indexed_iter() {
        if mm == 0 {
            continue;
        }
        // S mismatch: horizontal bar straddling bottom edge
        if mm & 0b0010 != 0 {
            let edge_y = y * scale + scale;
            let mid_x = x * scale + scale / 2;
            draw_rect(
                frame,
                mid_x.saturating_sub(long),
                mid_x + long,
                edge_y.saturating_sub(thick),
                edge_y + thick,
                color,
                frame_width_px,
            );
        }
        // W mismatch: vertical bar straddling left edge
        if mm & 0b0001 != 0 {
            let edge_x = x * scale;
            let mid_y = y * scale + scale / 2;
            draw_rect(
                frame,
                edge_x.saturating_sub(thick),
                edge_x + thick,
                mid_y.saturating_sub(long),
                mid_y + long,
                color,
                frame_width_px,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    /// Read a single pixel's RGBA value from the frame buffer.
    fn pixel_at(frame: &[u8], x: usize, y: usize, width: usize) -> [u8; 4] {
        let idx = (y * width + x) * 4;
        [frame[idx], frame[idx + 1], frame[idx + 2], frame[idx + 3]]
    }

    // ── draw_rect ──────────────────────────────────────────────────────

    #[test]
    fn draw_rect_writes_correct_pixels() {
        let (w, h) = (4, 4);
        let mut frame = vec![0u8; w * h * 4];
        let color = [10, 20, 30, 255];
        draw_rect(&mut frame, 1, 3, 1, 3, color, w);

        // Inside
        assert_eq!(pixel_at(&frame, 1, 1, w), color);
        assert_eq!(pixel_at(&frame, 2, 2, w), color);
        // Outside
        assert_eq!(pixel_at(&frame, 0, 0, w), [0, 0, 0, 0]);
        assert_eq!(pixel_at(&frame, 3, 3, w), [0, 0, 0, 0]);
    }

    #[test]
    fn draw_rect_zero_size_is_noop() {
        let mut frame = vec![0u8; 16];
        draw_rect(&mut frame, 2, 2, 0, 1, [255; 4], 2);
        assert!(frame.iter().all(|&b| b == 0));
    }

    #[test]
    fn draw_rect_out_of_bounds_safe() {
        let mut frame = vec![0u8; 4 * 4 * 4];
        // Rectangle extends beyond frame — should not panic
        draw_rect(&mut frame, 2, 10, 2, 10, [255; 4], 4);
        // Pixel (3,3) is in bounds and should be written
        assert_eq!(pixel_at(&frame, 3, 3, 4), [255; 4]);
    }

    // ── TileStyle::as_sprite ───────────────────────────────────────────

    #[test]
    fn as_sprite_triangle_regions() {
        let n = [10, 0, 0, 255];
        let e = [0, 10, 0, 255];
        let s = [0, 0, 10, 255];
        let w = [0, 0, 0, 255];
        let style = TileStyle {
            tri_colors: [n, e, s, w],
        };
        let size = 8;
        let sprite = style.as_sprite(size);
        assert_eq!(sprite.size, size);

        let px = |col: usize, row: usize| -> [u8; 4] {
            let idx = 4 * (row * size + col);
            [
                sprite.pixels[idx],
                sprite.pixels[idx + 1],
                sprite.pixels[idx + 2],
                sprite.pixels[idx + 3],
            ]
        };
        // NE corner (col > row && row <= size-col-1) → north
        assert_eq!(px(size - 1, 0), n);
        // SE corner (col > row is false for col=size-1, row=size-1?
        //   actually col=size-1, row=size-1: ne = (7>7)=false, se = (7>0)=true → south
        assert_eq!(px(size - 1, size - 1), s);
        // SW corner (col=0, row=size-1): ne=false, se=(7>7)=false → west
        assert_eq!(px(0, size - 1), w);
        // A clearly-east pixel: col=7, row=4 → ne=(7>4)=true, se=(4>0)=true → east
        assert_eq!(px(size - 1, size / 2), e);
    }

    // ── blit_sprite ────────────────────────────────────────────────────

    #[test]
    fn blit_sprite_places_pixels_correctly() {
        // 2x2 grid of 2px tiles → 4x4 pixel frame
        let tile_size = 2;
        let grid_w = 2;
        let frame_w = grid_w * tile_size; // 4
        let frame_h = 2 * tile_size; // 4
        let mut frame = vec![0u8; frame_w * frame_h * 4];

        let red = [255, 0, 0, 255];
        let sprite = SpriteSquare {
            size: tile_size,
            pixels: vec![
                red[0], red[1], red[2], red[3], red[0], red[1], red[2], red[3], // row 0
                red[0], red[1], red[2], red[3], red[0], red[1], red[2], red[3], // row 1
            ]
            .into_boxed_slice(),
        };

        // Place at grid (1, 0) → pixel origin (2, 0)
        blit_sprite(&mut frame, &sprite, 1, 0, frame_w);
        assert_eq!(pixel_at(&frame, 2, 0, frame_w), red);
        assert_eq!(pixel_at(&frame, 3, 1, frame_w), red);
        // Pixel (0,0) should be untouched
        assert_eq!(pixel_at(&frame, 0, 0, frame_w), [0, 0, 0, 0]);
    }

    // ── render_tiles ───────────────────────────────────────────────────

    #[test]
    fn render_tiles_blits_all() {
        let scale = 2;
        // 1x2 grid: tile 0 (empty), tile 1
        let tiles = Array2::from_shape_vec((1, 2), vec![0u32, 1u32]).unwrap();
        let frame_w = 2 * scale;
        let mut frame = vec![0u8; frame_w * scale * 4];

        let blue = [0, 0, 200, 255];
        let empty_sprite = SpriteSquare {
            size: scale,
            pixels: vec![0u8; scale * scale * 4].into_boxed_slice(),
        };
        let blue_sprite = SpriteSquare {
            size: scale,
            pixels: vec![blue[0], blue[1], blue[2], blue[3]]
                .repeat(scale * scale)
                .into_boxed_slice(),
        };
        let sprites = vec![empty_sprite, blue_sprite];

        render_tiles(&mut frame, tiles.view(), &sprites, frame_w);
        // Tile 1 occupies pixel columns [2..4]
        assert_eq!(pixel_at(&frame, 2, 0, frame_w), blue);
        // Tile 0 area should be zero (empty sprite)
        assert_eq!(pixel_at(&frame, 0, 0, frame_w), [0, 0, 0, 0]);
    }

    // ── render_outlines ────────────────────────────────────────────────

    #[test]
    fn render_outlines_draws_borders_for_nonempty() {
        let scale = 12;
        let tiles = Array2::from_shape_vec((1, 2), vec![0u32, 1u32]).unwrap();
        let frame_w = 2 * scale;
        let frame_h = scale;
        let mut frame = vec![0u8; frame_w * frame_h * 4];

        render_outlines(&mut frame, tiles.view(), scale, frame_w);

        let black = [0u8, 0, 0, 255];
        // Top-left of tile 1 (grid x=1) → pixel (12, 0) should be outlined
        assert_eq!(pixel_at(&frame, scale, 0, frame_w), black);
        // Bottom-right of tile 1 → pixel (23, 11) should be outlined
        assert_eq!(pixel_at(&frame, 2 * scale - 1, scale - 1, frame_w), black);
        // Tile 0 area should be untouched
        assert_eq!(pixel_at(&frame, 0, 0, frame_w), [0, 0, 0, 0]);
        assert_eq!(pixel_at(&frame, scale - 1, scale - 1, frame_w), [0, 0, 0, 0]);
    }

    // ── render_blockers ────────────────────────────────────────────────

    #[test]
    fn render_blockers_north_bit() {
        let scale = 12;
        // 2-row grid so north blocker has room above
        let tiles = Array2::from_shape_vec((2, 1), vec![0u32, 1u32]).unwrap();
        let frame_w = scale;
        let frame_h = 2 * scale;
        let mut frame = vec![0u8; frame_w * frame_h * 4];

        // tile 1 has north blocker
        let masks: Vec<u8> = vec![0, 0b0001];
        render_blockers(&mut frame, tiles.view(), &masks, scale, frame_w, frame_h);

        let blocker = [140, 140, 140, 255];
        let depth = (scale / 3).max(2);
        // North blocker for tile at grid (0,1) → pixel center above tile_y = 1*scale = 12
        // Check a pixel just above the tile's top edge, at the horizontal center
        let check_y = scale - 1; // just above tile at row 1
        let check_x = scale / 2;
        // This is within the blocker region: y in [12-depth..12), x around mid
        assert!(check_y >= scale.saturating_sub(depth));
        assert_eq!(pixel_at(&frame, check_x, check_y, frame_w), blocker);
    }

    #[test]
    fn render_blockers_each_direction() {
        let scale = 12;
        // 3x3 grid, tile 1 at center with all 4 blockers
        let mut tiles = Array2::zeros((3, 3));
        tiles[[1, 1]] = 1u32;
        let frame_w = 3 * scale;
        let frame_h = 3 * scale;
        let mut frame = vec![0u8; frame_w * frame_h * 4];

        let masks: Vec<u8> = vec![0, 0b1111]; // tile 1 has all 4 blockers
        render_blockers(&mut frame, tiles.view(), &masks, scale, frame_w, frame_h);

        let blocker = [140, 140, 140, 255];
        let mid_x = scale + scale / 2; // pixel center of tile (1,1)
        let mid_y = scale + scale / 2;

        // North: just above tile top edge
        assert_eq!(pixel_at(&frame, mid_x, scale - 1, frame_w), blocker);
        // South: just below tile bottom edge
        assert_eq!(pixel_at(&frame, mid_x, 2 * scale, frame_w), blocker);
        // East: just right of tile right edge
        assert_eq!(pixel_at(&frame, 2 * scale, mid_y, frame_w), blocker);
        // West: just left of tile left edge
        assert_eq!(pixel_at(&frame, scale - 1, mid_y, frame_w), blocker);
    }

    // ── render_mismatches ──────────────────────────────────────────────

    #[test]
    fn render_mismatches_south_bit() {
        let scale = 12;
        let mut locs = Array2::<usize>::zeros((2, 1));
        locs[[0, 0]] = 0b0010; // south mismatch on tile (0,0)
        let frame_w = scale;
        let frame_h = 2 * scale;
        let mut frame = vec![0u8; frame_w * frame_h * 4];

        render_mismatches(&mut frame, &locs.view(), scale, frame_w);

        let red = [255, 0, 0, 255];
        // Horizontal bar straddles y = scale, at horizontal center
        let mid_x = scale / 2;
        assert_eq!(pixel_at(&frame, mid_x, scale, frame_w), red);
    }

    #[test]
    fn render_mismatches_west_bit() {
        let scale = 12;
        let mut locs = Array2::<usize>::zeros((1, 2));
        locs[[0, 1]] = 0b0001; // west mismatch on tile (0,1)
        let frame_w = 2 * scale;
        let frame_h = scale;
        let mut frame = vec![0u8; frame_w * frame_h * 4];

        render_mismatches(&mut frame, &locs.view(), scale, frame_w);

        let red = [255, 0, 0, 255];
        // Vertical bar straddles x = scale, at vertical center
        let mid_y = scale / 2;
        assert_eq!(pixel_at(&frame, scale, mid_y, frame_w), red);
    }

    #[test]
    fn render_mismatches_zero_is_noop() {
        let scale = 12;
        let locs = Array2::<usize>::zeros((2, 2));
        let frame_w = 2 * scale;
        let frame_h = 2 * scale;
        let mut frame = vec![0u8; frame_w * frame_h * 4];

        render_mismatches(&mut frame, &locs.view(), scale, frame_w);

        assert!(frame.iter().all(|&b| b == 0));
    }
}
