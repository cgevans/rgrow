use ndarray::ArrayView2;

use crate::base::Tile;
use crate::canvas::{Canvas, PointSafe2, TileShape};
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

/// Blit a sprite at pixel offset `(dst_x, dst_y)`, only writing pixels with
/// non-zero alpha. Used for diamond sprites whose bounding-box corners are
/// transparent so adjacent diamonds interlock without erasing each other.
pub fn blit_sprite_at_px_alpha(
    frame: &mut [u8],
    sprite: &SpriteSquare,
    dst_x: u32,
    dst_y: u32,
    frame_width_px: u32,
    frame_height_px: u32,
) {
    let tile_size = sprite.size;
    let frame_w = frame_width_px as usize;
    let frame_h = frame_height_px as usize;
    let dst_y = dst_y as usize;
    let dst_x = dst_x as usize;
    for sy in 0..tile_size {
        let dy = dst_y + sy;
        if dy >= frame_h {
            break;
        }
        for sx in 0..tile_size {
            let dx = dst_x + sx;
            if dx >= frame_w {
                break;
            }
            let src_idx = 4 * (sy * tile_size + sx);
            let alpha = sprite.pixels[src_idx + 3];
            if alpha == 0 {
                continue;
            }
            let dst_idx = 4 * (dy * frame_w + dx);
            frame[dst_idx..dst_idx + 4].copy_from_slice(&sprite.pixels[src_idx..src_idx + 4]);
        }
    }
}

/// Blit a sprite at arbitrary pixel offset (no alpha keying). Like `blit_sprite`
/// but takes pixel coordinates directly instead of grid coordinates.
pub fn blit_sprite_at_px(
    frame: &mut [u8],
    sprite: &SpriteSquare,
    dst_x: u32,
    dst_y: u32,
    frame_width_px: u32,
    frame_height_px: u32,
) {
    let tile_size = sprite.size;
    let frame_w = frame_width_px as usize;
    let frame_h = frame_height_px as usize;
    let dst_y = dst_y as usize;
    let dst_x = dst_x as usize;
    let row_bytes = tile_size * 4;
    for sy in 0..tile_size {
        let dy = dst_y + sy;
        if dy >= frame_h {
            break;
        }
        if dst_x >= frame_w {
            break;
        }
        let copy_w = (frame_w - dst_x).min(tile_size);
        let src_start = 4 * sy * tile_size;
        let dst_start = 4 * (dy * frame_w + dst_x);
        let copy_bytes = copy_w * 4;
        debug_assert!(copy_bytes <= row_bytes);
        frame[dst_start..dst_start + copy_bytes]
            .copy_from_slice(&sprite.pixels[src_start..src_start + copy_bytes]);
    }
}

/// Convert a square NSEW-quadrant sprite into a diamond inscribed in the same
/// bounding box. Storage-NSEW colors map to the diamond's NW/NE/SE/SW edges:
/// the same NSEW edge information, just rotated 45° clockwise to match the
/// zigzag-tube neighbor lattice. Pixels outside the inscribed diamond are
/// transparent (alpha 0) so adjacent diamonds tile without overlap.
pub fn square_to_diamond(square: &SpriteSquare) -> SpriteSquare {
    let n = square.size;
    if n == 0 {
        return SpriteSquare {
            size: 0,
            pixels: Vec::new().into_boxed_slice(),
        };
    }
    let mut out = vec![0u8; n * n * 4];
    let half = (n / 2) as isize;
    let n_isize = n as isize;
    // Edge representatives in the source square sprite.
    let n_idx = (n / 2, 0);
    let e_idx = (n - 1, n / 2);
    let s_idx = (n / 2, n - 1);
    let w_idx = (0, n / 2);
    let lookup = |sx: usize, sy: usize| -> [u8; 4] {
        let i = 4 * (sy * n + sx);
        [
            square.pixels[i],
            square.pixels[i + 1],
            square.pixels[i + 2],
            square.pixels[i + 3],
        ]
    };
    let nw_color = lookup(n_idx.0, n_idx.1);
    let ne_color = lookup(e_idx.0, e_idx.1);
    let se_color = lookup(s_idx.0, s_idx.1);
    let sw_color = lookup(w_idx.0, w_idx.1);
    for py in 0..n_isize {
        for px in 0..n_isize {
            let dx = px - half;
            let dy = py - half;
            // Inside the inscribed diamond? |dx| + |dy| < half (strict so
            // adjacent diamonds don't both claim the boundary pixel).
            if dx.abs() + dy.abs() >= half {
                continue;
            }
            let color = match (dx >= 0, dy >= 0) {
                (false, false) => nw_color,
                (true, false) => ne_color,
                (true, true) => se_color,
                (false, true) => sw_color,
            };
            let dst_idx = 4 * ((py as usize) * n + (px as usize));
            out[dst_idx..dst_idx + 4].copy_from_slice(&color);
        }
    }
    SpriteSquare {
        size: n,
        pixels: out.into_boxed_slice(),
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
        draw_rect(
            frame,
            tx,
            tx + scale,
            ty,
            ty + 1,
            outline_color,
            frame_width_px,
        );
        // Bottom edge
        draw_rect(
            frame,
            tx,
            tx + scale,
            ty + scale - 1,
            ty + scale,
            outline_color,
            frame_width_px,
        );
        // Left edge
        draw_rect(
            frame,
            tx,
            tx + 1,
            ty,
            ty + scale,
            outline_color,
            frame_width_px,
        );
        // Right edge
        draw_rect(
            frame,
            tx + scale - 1,
            tx + scale,
            ty,
            ty + scale,
            outline_color,
            frame_width_px,
        );
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

/// Result of `render_frame`. The frame buffer has been filled in place;
/// these are the metadata the caller typically wants alongside.
#[derive(Debug, Clone, Copy)]
pub struct RenderStats {
    pub frame_width: u32,
    pub frame_height: u32,
    pub data_len: usize,
    pub mismatch_count: u32,
    pub n_tiles: crate::base::NumTiles,
    pub time: f64,
    pub total_events: crate::base::NumEvents,
    pub energy: crate::base::Energy,
}

/// Render `state`'s tiles into `frame`, dispatching through the canvas's
/// geometry so that tube canvases place tiles at the correct physical
/// (display-screen) coordinates rather than at raw storage positions.
///
/// This is the shared core of `render_frame` and `render_frame_dyn` — the
/// only difference between those two is which trait family (`System` vs
/// `SystemEnum::DynSystem`) builds the sprites and reads mismatches.
#[allow(clippy::too_many_arguments)]
fn render_into_with_canvas(
    canvas: &dyn Canvas,
    tiles: ArrayView2<Tile>,
    sprites: &[SpriteSquare],
    blocker_masks: &[u8],
    mismatch_locs: Option<&ArrayView2<usize>>,
    scale: usize,
    pixel_frame: &mut [u8],
    frame_width: u32,
    frame_height: u32,
) {
    let scale_u = scale as u32;
    let tile_shape = canvas.tile_shape();

    // Tile placement: ask the canvas where each storage cell lives in pixel
    // space. For square canvases this is identity (col*scale, row*scale);
    // tube canvases shear or stagger.
    for ((y, x), &tileid) in tiles.indexed_iter() {
        let Some(sprite) = sprites.get(tileid as usize) else {
            continue;
        };
        let p = PointSafe2((y, x));
        let (px, py) = canvas.tile_origin_px(p, scale_u);
        match tile_shape {
            TileShape::Square => {
                // Skip tile 0 (empty) when its sprite is fully zero — keeps
                // the frame's zero-init showing through, no behavior change
                // for square canvases vs the previous painter.
                blit_sprite_at_px(pixel_frame, sprite, px, py, frame_width, frame_height);
            }
            TileShape::Diamond => {
                if tileid == 0 {
                    continue;
                }
                blit_sprite_at_px_alpha(pixel_frame, sprite, px, py, frame_width, frame_height);
            }
        }
    }

    // Outlines, blockers, and mismatches: tube-aware versions live below
    // and dispatch through `tile_origin_px` like the tile-blit loop above.
    if scale >= 12 {
        render_outlines_via_canvas(pixel_frame, canvas, tiles, scale_u, frame_width);
    }
    render_blockers_via_canvas(
        pixel_frame,
        canvas,
        tiles,
        blocker_masks,
        scale_u,
        frame_width,
        frame_height,
    );
    if let Some(locs) = mismatch_locs {
        render_mismatches_via_canvas(pixel_frame, canvas, locs, scale_u, frame_width);
    }
}

/// Outline non-empty tiles by drawing a 1px frame around each tile's
/// bounding box. For square canvases this is identical to the original
/// `render_outlines`; for diamond canvases it outlines the bounding box
/// (which is the right thing here — the inscribed diamond's edges are
/// already implicit in the alpha-keyed sprite).
fn render_outlines_via_canvas(
    frame: &mut [u8],
    canvas: &dyn Canvas,
    tiles: ArrayView2<Tile>,
    scale: u32,
    frame_width_px: u32,
) {
    let outline_color = [0u8, 0, 0, 255];
    let tile_size = canvas.tile_size_px(scale) as usize;
    let frame_w = frame_width_px as usize;
    let shape = canvas.tile_shape();
    for ((y, x), &tileid) in tiles.indexed_iter() {
        if tileid == 0 {
            continue;
        }
        let p = PointSafe2((y, x));
        let (ox, oy) = canvas.tile_origin_px(p, scale);
        let tx = ox as usize;
        let ty = oy as usize;
        match shape {
            TileShape::Square => {
                draw_rect(
                    frame,
                    tx,
                    tx + tile_size,
                    ty,
                    ty + 1,
                    outline_color,
                    frame_w,
                );
                draw_rect(
                    frame,
                    tx,
                    tx + tile_size,
                    ty + tile_size - 1,
                    ty + tile_size,
                    outline_color,
                    frame_w,
                );
                draw_rect(
                    frame,
                    tx,
                    tx + 1,
                    ty,
                    ty + tile_size,
                    outline_color,
                    frame_w,
                );
                draw_rect(
                    frame,
                    tx + tile_size - 1,
                    tx + tile_size,
                    ty,
                    ty + tile_size,
                    outline_color,
                    frame_w,
                );
            }
            TileShape::Diamond => {
                // Outline along the inscribed diamond's four edges.
                let half = tile_size / 2;
                let cx = tx + half;
                let cy = ty + half;
                for k in 0..half {
                    // NW edge
                    draw_pixel(frame, cx - half + k, cy - k, outline_color, frame_w);
                    // NE edge
                    draw_pixel(frame, cx + k, cy - half + k, outline_color, frame_w);
                    // SE edge
                    draw_pixel(frame, cx + half - k, cy + k, outline_color, frame_w);
                    // SW edge
                    draw_pixel(frame, cx - k, cy + half - k, outline_color, frame_w);
                }
            }
        }
    }
}

#[inline]
fn draw_pixel(frame: &mut [u8], x: usize, y: usize, color: [u8; 4], frame_width_px: usize) {
    let idx = (y * frame_width_px + x) * 4;
    if idx + 4 <= frame.len() {
        frame[idx..idx + 4].copy_from_slice(&color);
    }
}

fn render_blockers_via_canvas(
    frame: &mut [u8],
    canvas: &dyn Canvas,
    tiles: ArrayView2<Tile>,
    blocker_masks: &[u8],
    scale: u32,
    frame_width_px: u32,
    frame_height_px: u32,
) {
    let tile_size = canvas.tile_size_px(scale) as usize;
    let depth = (tile_size / 3).max(2);
    let half_len = (tile_size / 3).max(2);
    let blocker_color = [140, 140, 140, 255];
    let frame_w = frame_width_px as usize;
    let frame_h = frame_height_px as usize;
    // Diamond blockers don't have a clean bar-shape representation; draw a
    // small dot at the relevant edge midpoint instead.
    let shape = canvas.tile_shape();
    for ((y, x), &tileid) in tiles.indexed_iter() {
        let mask = blocker_masks.get(tileid as usize).copied().unwrap_or(0);
        if mask == 0 {
            continue;
        }
        let p = PointSafe2((y, x));
        let (ox, oy) = canvas.tile_origin_px(p, scale);
        let tile_x = ox as usize;
        let tile_y = oy as usize;
        let mid_x = tile_x + tile_size / 2;
        let mid_y = tile_y + tile_size / 2;
        match shape {
            TileShape::Square => {
                if mask & 0b0001 != 0 {
                    draw_rect(
                        frame,
                        mid_x.saturating_sub(half_len),
                        mid_x + half_len,
                        tile_y.saturating_sub(depth),
                        tile_y,
                        blocker_color,
                        frame_w,
                    );
                }
                if mask & 0b0010 != 0 {
                    let right = tile_x + tile_size;
                    draw_rect(
                        frame,
                        right,
                        (right + depth).min(frame_w),
                        mid_y.saturating_sub(half_len),
                        mid_y + half_len,
                        blocker_color,
                        frame_w,
                    );
                }
                if mask & 0b0100 != 0 {
                    let bottom = tile_y + tile_size;
                    draw_rect(
                        frame,
                        mid_x.saturating_sub(half_len),
                        mid_x + half_len,
                        bottom,
                        (bottom + depth).min(frame_h),
                        blocker_color,
                        frame_w,
                    );
                }
                if mask & 0b1000 != 0 {
                    draw_rect(
                        frame,
                        tile_x.saturating_sub(depth),
                        tile_x,
                        mid_y.saturating_sub(half_len),
                        mid_y + half_len,
                        blocker_color,
                        frame_w,
                    );
                }
            }
            TileShape::Diamond => {
                let dot = (tile_size / 8).max(2);
                // Small filled square on each side of the diamond's
                // outer corner, indicating a blocker on that storage edge.
                if mask & 0b0001 != 0 {
                    // N edge → diamond NW corner area
                    let cx = tile_x + tile_size / 4;
                    let cy = tile_y + tile_size / 4;
                    draw_rect(
                        frame,
                        cx.saturating_sub(dot),
                        (cx + dot).min(frame_w),
                        cy.saturating_sub(dot),
                        (cy + dot).min(frame_h),
                        blocker_color,
                        frame_w,
                    );
                }
                if mask & 0b0010 != 0 {
                    let cx = tile_x + 3 * tile_size / 4;
                    let cy = tile_y + tile_size / 4;
                    draw_rect(
                        frame,
                        cx.saturating_sub(dot),
                        (cx + dot).min(frame_w),
                        cy.saturating_sub(dot),
                        (cy + dot).min(frame_h),
                        blocker_color,
                        frame_w,
                    );
                }
                if mask & 0b0100 != 0 {
                    let cx = tile_x + 3 * tile_size / 4;
                    let cy = tile_y + 3 * tile_size / 4;
                    draw_rect(
                        frame,
                        cx.saturating_sub(dot),
                        (cx + dot).min(frame_w),
                        cy.saturating_sub(dot),
                        (cy + dot).min(frame_h),
                        blocker_color,
                        frame_w,
                    );
                }
                if mask & 0b1000 != 0 {
                    let cx = tile_x + tile_size / 4;
                    let cy = tile_y + 3 * tile_size / 4;
                    draw_rect(
                        frame,
                        cx.saturating_sub(dot),
                        (cx + dot).min(frame_w),
                        cy.saturating_sub(dot),
                        (cy + dot).min(frame_h),
                        blocker_color,
                        frame_w,
                    );
                }
            }
        }
    }
}

fn render_mismatches_via_canvas(
    frame: &mut [u8],
    canvas: &dyn Canvas,
    mismatch_locs: &ArrayView2<usize>,
    scale: u32,
    frame_width_px: u32,
) {
    let tile_size = canvas.tile_size_px(scale) as usize;
    let thick = (tile_size / 4).max(1);
    let long = (tile_size / 3).max(1);
    let color = [255, 0, 0, 255];
    let frame_w = frame_width_px as usize;
    let shape = canvas.tile_shape();
    for ((y, x), &mm) in mismatch_locs.indexed_iter() {
        if mm == 0 {
            continue;
        }
        let p = PointSafe2((y, x));
        let (ox, oy) = canvas.tile_origin_px(p, scale);
        let tx = ox as usize;
        let ty = oy as usize;
        match shape {
            TileShape::Square => {
                if mm & 0b0010 != 0 {
                    let edge_y = ty + tile_size;
                    let mid_x = tx + tile_size / 2;
                    draw_rect(
                        frame,
                        mid_x.saturating_sub(long),
                        mid_x + long,
                        edge_y.saturating_sub(thick),
                        edge_y + thick,
                        color,
                        frame_w,
                    );
                }
                if mm & 0b0001 != 0 {
                    let edge_x = tx;
                    let mid_y = ty + tile_size / 2;
                    draw_rect(
                        frame,
                        edge_x.saturating_sub(thick),
                        edge_x + thick,
                        mid_y.saturating_sub(long),
                        mid_y + long,
                        color,
                        frame_w,
                    );
                }
            }
            TileShape::Diamond => {
                // S mismatch → diamond's SE edge midpoint
                if mm & 0b0010 != 0 {
                    let cx = tx + 3 * tile_size / 4;
                    let cy = ty + 3 * tile_size / 4;
                    draw_rect(
                        frame,
                        cx.saturating_sub(thick),
                        cx + thick,
                        cy.saturating_sub(thick),
                        cy + thick,
                        color,
                        frame_w,
                    );
                }
                // W mismatch → diamond's SW edge midpoint
                if mm & 0b0001 != 0 {
                    let cx = tx + tile_size / 4;
                    let cy = ty + 3 * tile_size / 4;
                    draw_rect(
                        frame,
                        cx.saturating_sub(thick),
                        cx + thick,
                        cy.saturating_sub(thick),
                        cy + thick,
                        color,
                        frame_w,
                    );
                }
            }
        }
    }
}

/// Build per-tile-id sprites in the right shape for `canvas`.
fn build_sprites_for_canvas<F>(
    canvas: &dyn Canvas,
    max_tile: usize,
    scale: usize,
    mut tile_pixels: F,
) -> Vec<SpriteSquare>
where
    F: FnMut(Tile, usize) -> SpriteSquare,
{
    let tile_size = canvas.tile_size_px(scale as u32) as usize;
    match canvas.tile_shape() {
        TileShape::Square => (0..=max_tile)
            .map(|t| tile_pixels(t as Tile, tile_size))
            .collect(),
        TileShape::Diamond => (0..=max_tile)
            .map(|t| {
                let square = tile_pixels(t as Tile, tile_size);
                square_to_diamond(&square)
            })
            .collect(),
    }
}

/// Render the current state of `state` (under `sys`) into `frame` as RGBA8.
///
/// `frame` must be at least `frame_width * frame_height * 4` bytes, where
/// `(frame_width, frame_height) = state.frame_size_px(scale)`. The frame is
/// drawn in-place; existing contents may be overwritten in any order.
///
/// This is the canonical rendering entry point. Both the desktop GUI
/// (`system::gui::evolve_in_window_impl`) and the WebAssembly bindings
/// call this function so the two views cannot drift.
pub fn render_frame<S, St>(
    sys: &S,
    state: &St,
    scale: usize,
    show_mismatches: bool,
    frame: &mut [u8],
) -> RenderStats
where
    S: crate::system::System,
    St: crate::state::State,
{
    let scale_u = scale as u32;
    let (frame_w, frame_h) = state.frame_size_px(scale_u);
    let frame_width = frame_w as usize;
    let frame_height = frame_h as usize;
    let needed = frame_width * frame_height * 4;
    let pixel_frame = &mut frame[..needed];

    let tiles = state.raw_array();
    let max_tile = tiles.iter().copied().max().unwrap_or(0) as usize;
    let sprites = build_sprites_for_canvas(state, max_tile, scale, |t, sz| sys.tile_pixels(t, sz));
    let blocker_masks: Vec<u8> = (0..=max_tile)
        .map(|t| sys.tile_blocker_mask(t as Tile))
        .collect();

    let (mismatch_count, mismatch_locs) = if show_mismatches {
        let locs = sys.calc_mismatch_locations(state);
        let count: u32 = locs
            .iter()
            .map(|x| ((x & 0b01) + ((x & 0b10) >> 1)) as u32)
            .sum();
        (count, Some(locs))
    } else {
        (sys.calc_mismatches(state) as u32, None)
    };

    render_into_with_canvas(
        state,
        tiles,
        &sprites,
        &blocker_masks,
        mismatch_locs.as_ref().map(|a| a.view()).as_ref(),
        scale,
        pixel_frame,
        frame_w,
        frame_h,
    );

    RenderStats {
        frame_width: frame_w,
        frame_height: frame_h,
        data_len: needed,
        mismatch_count,
        n_tiles: state.n_tiles(),
        time: state.time().into(),
        total_events: state.total_events(),
        energy: state.energy(),
    }
}

/// Variant of `render_frame` for the dynamic dispatch types (`SystemEnum`
/// + `StateEnum`). Used by the WebAssembly bindings.
pub fn render_frame_dyn(
    sys: &crate::system::SystemEnum,
    state: &crate::state::StateEnum,
    scale: usize,
    show_mismatches: bool,
    frame: &mut [u8],
) -> RenderStats {
    use crate::state::StateStatus;
    use crate::system::{DynSystem, TileBondInfo};

    let scale_u = scale as u32;
    let (frame_w, frame_h) = state.frame_size_px(scale_u);
    let frame_width = frame_w as usize;
    let frame_height = frame_h as usize;
    let needed = frame_width * frame_height * 4;
    let pixel_frame = &mut frame[..needed];

    let tiles = state.raw_array();
    let max_tile = tiles.iter().copied().max().unwrap_or(0) as usize;
    let sprites = build_sprites_for_canvas(state, max_tile, scale, |t, sz| sys.tile_pixels(t, sz));
    let blocker_masks: Vec<u8> = (0..=max_tile)
        .map(|t| sys.tile_blocker_mask(t as Tile))
        .collect();

    let (mismatch_count, mismatch_locs) = if show_mismatches {
        let locs = sys.calc_mismatch_locations(state);
        let count: u32 = locs
            .iter()
            .map(|x| ((x & 0b01) + ((x & 0b10) >> 1)) as u32)
            .sum();
        (count, Some(locs))
    } else {
        (sys.calc_mismatches(state) as u32, None)
    };

    render_into_with_canvas(
        state,
        tiles,
        &sprites,
        &blocker_masks,
        mismatch_locs.as_ref().map(|a| a.view()).as_ref(),
        scale,
        pixel_frame,
        frame_w,
        frame_h,
    );

    RenderStats {
        frame_width: frame_w,
        frame_height: frame_h,
        data_len: needed,
        mismatch_count,
        n_tiles: state.n_tiles(),
        time: state.time().into(),
        total_events: state.total_events(),
        energy: state.energy(),
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
        assert_eq!(
            pixel_at(&frame, scale - 1, scale - 1, frame_w),
            [0, 0, 0, 0]
        );
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

    // ── square_to_diamond ─────────────────────────────────────────────────

    #[test]
    fn square_to_diamond_alpha_keys_corners() {
        // 12x12 sprite, all opaque green. Diamond should leave the four
        // corners transparent.
        let n = 12;
        let green = [0u8, 200, 0, 255];
        let pixels = vec![green[0], green[1], green[2], green[3]]
            .repeat(n * n)
            .into_boxed_slice();
        let square = SpriteSquare { size: n, pixels };
        let diamond = square_to_diamond(&square);
        assert_eq!(diamond.size, n);
        // Top-left corner (0, 0) is outside the inscribed diamond.
        let alpha_at = |x: usize, y: usize| diamond.pixels[(y * n + x) * 4 + 3];
        assert_eq!(alpha_at(0, 0), 0, "NW corner should be transparent");
        assert_eq!(alpha_at(n - 1, 0), 0, "NE corner should be transparent");
        assert_eq!(alpha_at(0, n - 1), 0, "SW corner should be transparent");
        assert_eq!(alpha_at(n - 1, n - 1), 0, "SE corner should be transparent");
        // Center should be opaque.
        assert_eq!(alpha_at(n / 2, n / 2), 255, "center should be opaque");
    }

    #[test]
    fn square_to_diamond_quadrant_colors() {
        // Square sprite with distinct colors on each NSEW edge.
        let n = 16;
        let north = [10, 0, 0, 255];
        let east = [0, 20, 0, 255];
        let south = [0, 0, 30, 255];
        let west = [40, 40, 40, 255];
        let style = TileStyle {
            tri_colors: [north, east, south, west],
        };
        let square = style.as_sprite(n);
        let diamond = square_to_diamond(&square);
        let pixel = |x: usize, y: usize| -> [u8; 4] {
            let i = (y * n + x) * 4;
            [
                diamond.pixels[i],
                diamond.pixels[i + 1],
                diamond.pixels[i + 2],
                diamond.pixels[i + 3],
            ]
        };
        // NW quadrant (small dx, small dy from center, both negative) maps
        // to storage-N color.
        let half = n / 2;
        assert_eq!(pixel(half - 2, half - 2), north);
        // NE quadrant → E color.
        assert_eq!(pixel(half + 2, half - 2), east);
        // SE quadrant → S color.
        assert_eq!(pixel(half + 2, half + 2), south);
        // SW quadrant → W color.
        assert_eq!(pixel(half - 2, half + 2), west);
    }
}
