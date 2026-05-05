use crate::base::{GrowError, GrowResult, NumTiles, Point, Tile};
use crate::canvas::{Canvas, CanvasCreate, PointSafe2, TileShape};
use ndarray::{Array1, Array2, ArrayView2, ArrayViewMut2};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanvasTube(Array2<Tile>);

impl CanvasTube {
    pub fn half_width(&self) -> usize {
        self.0.nrows() / 2
    }
}

impl CanvasCreate for CanvasTube {
    type Params = (usize, usize);

    fn new_sized(shape: Self::Params) -> GrowResult<Self> {
        let width = shape.0;
        if !width.is_multiple_of(2) {
            return Err(GrowError::WrongCanvasSize(width, shape.1));
        }
        if shape.0 == 0 || shape.1 < 3 {
            return Err(GrowError::WrongCanvasSize(shape.0, shape.1));
        }
        Ok(Self(Array2::zeros(shape)))
    }

    fn from_array(arr: Array2<Tile>) -> GrowResult<Self> {
        if !arr.shape()[0].is_multiple_of(2) {
            Err(GrowError::WrongCanvasSize(arr.shape()[0], arr.shape()[1]))
        } else {
            Ok(Self(arr))
        }
    }
}

impl Canvas for CanvasTube {
    unsafe fn uv_pr(&self, p: Point) -> &Tile {
        self.0.uget(p)
    }

    unsafe fn uvm_p(&mut self, p: Point) -> &mut Tile {
        self.0.uget_mut(p)
    }

    fn u_move_point_n(&self, p: Point) -> Point {
        if p.0.is_multiple_of(2) {
            (
                (p.0 as i64 - 1).rem_euclid(self.nrows() as i64) as usize,
                p.1 - 1,
            )
        } else {
            (
                (p.0 as i64 - 1).rem_euclid(self.nrows() as i64) as usize,
                p.1,
            )
        }
    }

    fn u_move_point_e(&self, p: Point) -> Point {
        if p.0.is_multiple_of(2) {
            (
                (p.0 as i64 - 1).rem_euclid(self.nrows() as i64) as usize,
                p.1,
            )
        } else {
            (
                (p.0 as i64 - 1).rem_euclid(self.nrows() as i64) as usize,
                p.1 + 1,
            )
        }
    }

    fn u_move_point_s(&self, p: Point) -> Point {
        if p.0.is_multiple_of(2) {
            ((p.0 + 1).rem_euclid(self.nrows()), p.1)
        } else {
            ((p.0 + 1).rem_euclid(self.nrows()), p.1 + 1)
        }
    }

    fn u_move_point_w(&self, p: Point) -> Point {
        if p.0.is_multiple_of(2) {
            ((p.0 + 1).rem_euclid(self.nrows()), p.1 - 1)
        } else {
            ((p.0 + 1).rem_euclid(self.nrows()), p.1)
        }
    }

    fn inbounds(&self, p: Point) -> bool {
        let (is, js) = self.0.dim();
        (p.0 < is) & (p.1 < js - 1) & (p.1 >= 1)
    }

    fn calc_n_tiles(&self) -> NumTiles {
        self.0.fold(0, |x, y| x + u32::from(*y != 0))
    }

    fn calc_n_tiles_with_tilearray(&self, should_be_counted: &Array1<bool>) -> NumTiles {
        self.0
            .fold(0, |x, y| x + u32::from(should_be_counted[*y as usize]))
    }

    fn raw_array(&self) -> ArrayView2<'_, Tile> {
        self.0.view()
    }

    fn raw_array_mut(&mut self) -> ArrayViewMut2<'_, Tile> {
        self.0.view_mut()
    }

    fn nrows(&self) -> usize {
        self.0.nrows()
    }

    fn ncols(&self) -> usize {
        self.0.ncols()
    }

    fn center(&self) -> PointSafe2 {
        PointSafe2((self.nrows() / 2, self.ncols() / 2))
    }

    fn nrows_usable(&self) -> usize {
        self.0.nrows()
    }

    fn ncols_usable(&self) -> usize {
        self.0.ncols() - 2
    }

    // ── Display geometry (diamond / zigzag layout) ────────────────────────
    //
    // The zigzag tube's storage neighbors form a tilted-square lattice: from
    // an even-`i` cell, NSEW neighbors live at `(i±1, j-1)` and `(i±1, j)`;
    // from an odd-`i` cell, at `(i±1, j)` and `(i±1, j+1)`. All four
    // neighbors share an *edge* with the source cell on a 45°-rotated grid,
    // so each storage cell is rendered as a diamond. The mapping from
    // storage compass to diamond edges is:
    //   storage-N → diamond-NW edge
    //   storage-E → diamond-NE edge
    //   storage-S → diamond-SE edge
    //   storage-W → diamond-SW edge
    //
    // Diamonds are placed on a half-cell grid (each diamond occupies a 2×2
    // half-cell bounding box). Storage `(i, j)`'s bounding-box top-left in
    // half-cells is `(2*j + (i mod 2), i)`. Even-`i` rows are flush left;
    // odd-`i` rows are offset right by one half-cell — the brick pattern.

    fn tile_shape(&self) -> TileShape {
        TileShape::Diamond
    }

    fn frame_size_subcells(&self) -> (u32, u32) {
        // Half-cell extent: x ∈ [0, 2*ncols], y ∈ [0, nrows + 1].
        let w = (2 * self.ncols() + 1) as u32;
        let h = (self.nrows() + 1) as u32;
        (w, h)
    }

    fn tile_origin_px(&self, p: PointSafe2, scale: u32) -> (u32, u32) {
        let (i, j) = p.0;
        let s = self.subcell_size_px(scale);
        let half_x = (2 * j + (i & 1)) as u32;
        let half_y = i as u32;
        (half_x * s, half_y * s)
    }

    fn pixel_to_storage(&self, px: u32, py: u32, scale: u32) -> Option<PointSafe2> {
        let s = self.subcell_size_px(scale);
        if s == 0 {
            return None;
        }
        // Map pixel to diamond-center half-cell coordinates by adding half a
        // tile (one half-cell) so we hit the diamond's center, not its
        // bounding-box top-left.
        let hx = (px / s) as isize;
        let hy = (py / s) as isize;
        // The diamond covering pixel (px, py) is the one whose center is the
        // closest half-cell point of the form (2j + parity + 1, i + 1) such
        // that |hx - cx| + |hy - cy| < 1 isn't quite right — instead we
        // check membership by trying the two candidate cells whose bboxes
        // overlap (px, py).
        // Candidate 1: top-left bbox = (hx_floor & ~1, hy & ~0) — even row
        //   carries no x-shift; odd row shifts +1.
        // Practical approach: scan a small neighborhood and pick the cell
        // that contains (px, py) in its diamond.
        let nrows = self.nrows() as isize;
        let ncols = self.ncols() as isize;
        for di in -1isize..=1 {
            for dj in -1isize..=1 {
                // Estimate (i, j) from (hx, hy) by inverting the layout
                // formulas, then refine in a small window.
                let i_est = hy + di;
                if i_est < 0 || i_est >= nrows {
                    continue;
                }
                let parity = i_est & 1;
                let j_est = (hx - parity) / 2 + dj;
                if j_est < 0 || j_est >= ncols {
                    continue;
                }
                // Diamond center for this candidate (in pixel coords):
                let cx = ((2 * j_est + parity) as u32 + 1) * s;
                let cy = (i_est as u32 + 1) * s;
                // Manhattan distance in pixels:
                let mdx = (px as i64 - cx as i64).unsigned_abs();
                let mdy = (py as i64 - cy as i64).unsigned_abs();
                if (mdx + mdy) < s as u64 {
                    return Some(PointSafe2((i_est as usize, j_est as usize)));
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canvas::{Canvas, PointSafe2, TileShape};

    #[test]
    fn tile_shape_is_diamond() {
        let c = CanvasTube::new_sized((4, 8)).unwrap();
        assert_eq!(c.tile_shape(), TileShape::Diamond);
    }

    #[test]
    fn frame_size_uses_half_cells() {
        // 4×8 storage. Half-cell extent: x ∈ [0, 2*8] = [0, 16] (17 wide),
        // y ∈ [0, 4+1] = [0, 5] (6 tall).
        let c = CanvasTube::new_sized((4, 8)).unwrap();
        assert_eq!(c.frame_size_subcells(), (17, 5));
    }

    #[test]
    fn subcell_size_halves_scale_for_diamond() {
        let c = CanvasTube::new_sized((4, 8)).unwrap();
        assert_eq!(c.subcell_size_px(8), 4);
        assert_eq!(c.subcell_size_px(12), 6);
    }

    #[test]
    fn tile_origin_staggers_even_odd_rows() {
        let c = CanvasTube::new_sized((4, 8)).unwrap();
        let scale = 8;
        // Even row i=0: top-left x = 2*j + 0 half-cells.
        assert_eq!(c.tile_origin_px(PointSafe2((0, 0)), scale), (0, 0));
        assert_eq!(c.tile_origin_px(PointSafe2((0, 1)), scale), (8, 0));
        assert_eq!(c.tile_origin_px(PointSafe2((0, 3)), scale), (24, 0));
        // Odd row i=1: top-left x = 2*j + 1 half-cells (offset +1 half-cell).
        assert_eq!(c.tile_origin_px(PointSafe2((1, 0)), scale), (4, 4));
        assert_eq!(c.tile_origin_px(PointSafe2((1, 1)), scale), (12, 4));
    }

    #[test]
    fn pixel_to_storage_round_trips_at_diamond_center() {
        let c = CanvasTube::new_sized((4, 8)).unwrap();
        let scale = 8;
        for i in 0..4 {
            for j in 0..8 {
                let p = PointSafe2((i, j));
                let (ox, oy) = c.tile_origin_px(p, scale);
                let tile_size = c.tile_size_px(scale);
                let cx = ox + tile_size / 2;
                let cy = oy + tile_size / 2;
                let recovered = c.pixel_to_storage(cx, cy, scale);
                assert_eq!(recovered, Some(p), "round-trip failed for ({i}, {j})");
            }
        }
    }
}
