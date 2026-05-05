use crate::base::{GrowError, GrowResult, NumTiles, Point, Tile};
use crate::canvas::PointSafe2;
use crate::canvas::{Canvas, CanvasCreate};
use ndarray::prelude::*;

use serde::{Deserialize, Serialize};
use std::fmt::Debug;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanvasTubeDiagonals(Array2<Tile>);

impl CanvasTubeDiagonals {
    pub fn half_width(&self) -> usize {
        self.0.nrows() / 2
    }
}

impl CanvasCreate for CanvasTubeDiagonals {
    type Params = (usize, usize);

    fn new_sized(shape: Self::Params) -> GrowResult<Self> {
        let width = shape.0;
        if !width.is_multiple_of(2) {
            return Err(GrowError::WrongCanvasSize(width, shape.1));
        }
        // inbounds requires p.1 >= 2 + hw && p.1 < ncols - 2 - hw where hw = nrows/2
        let hw = shape.0 / 2;
        if shape.0 == 0 || shape.1 <= 4 + 2 * hw {
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

impl Canvas for CanvasTubeDiagonals {
    unsafe fn uv_pr(&self, p: Point) -> &Tile {
        self.0.uget(p)
    }

    unsafe fn uvm_p(&mut self, p: Point) -> &mut Tile {
        self.0.uget_mut(p)
    }

    fn u_move_point_n(&self, p: Point) -> Point {
        if p.0 == 0 {
            (self.nrows() - 1, p.1 - self.half_width())
        } else {
            (p.0 - 1, p.1)
        }
    }

    fn u_move_point_e(&self, p: Point) -> Point {
        if p.0 == 0 {
            (self.nrows() - 1, p.1 - self.half_width() + 1)
        } else {
            (p.0 - 1, p.1 + 1)
        }
    }

    fn u_move_point_s(&self, p: Point) -> Point {
        if p.0 == self.nrows() - 1 {
            (0, p.1 + self.half_width())
        } else {
            (p.0 + 1, p.1)
        }
    }

    fn u_move_point_w(&self, p: Point) -> Point {
        if p.0 == self.nrows() - 1 {
            (0, p.1 + self.half_width() - 1)
        } else {
            (p.0 + 1, p.1 - 1)
        }
    }

    fn inbounds(&self, p: Point) -> bool {
        let (xs, ys) = self.0.dim();
        (p.0 < xs) & (p.1 < ys - 2 - self.half_width()) & (p.1 >= 2 + self.half_width())
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
        self.0.nrows() // FIXME: is this correct?
    }

    fn ncols_usable(&self) -> usize {
        self.0.ncols() // FIXME: is this correct?
    }

    // ── Display geometry ──────────────────────────────────────────────────
    //
    // Storage `(i, j)` represents the tile in storage row `i`, column `j`,
    // but on the actual tube the south direction also moves the tile one
    // column right (helical winding). Mapping `(i, j) → (j, i+j)` in
    // physical coordinates makes NSEW point straight up/right/down/left in
    // screen space.
    //
    // Storage range: `i ∈ [0, nrows-1]`, `j ∈ [0, ncols-1]`. So the physical
    // x range is `[0, ncols-1]` and the physical y range is `[0, nrows+ncols-2]`.

    fn frame_size_subcells(&self) -> (u32, u32) {
        let w = self.ncols() as u32;
        let h = (self.nrows() + self.ncols()).saturating_sub(1) as u32;
        (w, h)
    }

    fn tile_origin_px(&self, p: crate::canvas::PointSafe2, scale: u32) -> (u32, u32) {
        let (i, j) = p.0;
        let s = self.subcell_size_px(scale);
        ((j as u32) * s, ((i + j) as u32) * s)
    }

    fn pixel_to_storage(&self, px: u32, py: u32, scale: u32) -> Option<crate::canvas::PointSafe2> {
        let s = self.subcell_size_px(scale);
        if s == 0 {
            return None;
        }
        let phys_x = (px / s) as usize;
        let phys_y = (py / s) as usize;
        // Inverse: phys_x = j, phys_y = i + j → j = phys_x, i = phys_y - j
        let j = phys_x;
        if phys_y < j {
            return None; // upper-left empty triangle
        }
        let i = phys_y - j;
        if i >= self.nrows() || j >= self.ncols() {
            return None; // lower-right empty triangle / out-of-bounds
        }
        Some(crate::canvas::PointSafe2((i, j)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canvas::{Canvas, PointSafe2, TileShape};

    #[test]
    fn tile_shape_is_square() {
        let c = CanvasTubeDiagonals::new_sized((4, 12)).unwrap();
        assert_eq!(c.tile_shape(), TileShape::Square);
    }

    #[test]
    fn frame_size_is_sheared_parallelogram() {
        // 4×12 storage, sheared: physical x ∈ [0, 11], y ∈ [0, 4+12-2] = [0, 14].
        let c = CanvasTubeDiagonals::new_sized((4, 12)).unwrap();
        assert_eq!(c.frame_size_subcells(), (12, 15));
    }

    #[test]
    fn tile_origin_shears_storage() {
        let c = CanvasTubeDiagonals::new_sized((6, 16)).unwrap();
        let scale = 8;
        // Storage (0, 0) → physical (0, 0).
        assert_eq!(c.tile_origin_px(PointSafe2((0, 0)), scale), (0, 0));
        // Storage (0, 5) → physical (5, 5) → pixels (40, 40).
        assert_eq!(c.tile_origin_px(PointSafe2((0, 5)), scale), (40, 40));
        // Storage (3, 2) → physical (2, 5) → pixels (16, 40).
        assert_eq!(c.tile_origin_px(PointSafe2((3, 2)), scale), (16, 40));
    }

    #[test]
    fn pixel_to_storage_round_trips() {
        let c = CanvasTubeDiagonals::new_sized((4, 12)).unwrap();
        let scale = 10;
        for i in 0..4 {
            for j in 0..12 {
                let p = PointSafe2((i, j));
                let (px, py) = c.tile_origin_px(p, scale);
                // Probe the center of each cell so rounding doesn't fall on
                // a boundary.
                let center_px = px + scale / 2;
                let center_py = py + scale / 2;
                let recovered = c.pixel_to_storage(center_px, center_py, scale);
                assert_eq!(recovered, Some(p), "round-trip failed for ({i}, {j})");
            }
        }
    }

    #[test]
    fn pixel_to_storage_returns_none_for_empty_triangles() {
        let c = CanvasTubeDiagonals::new_sized((4, 12)).unwrap();
        let scale = 10;
        // Pixel near (0, 30): physical_y=3, physical_x=0 → j=0, i=3 (in
        // bounds, fine). But pixel (0, 100): physical_y=10, physical_x=0
        // → j=0, i=10 (i >= nrows=4 → None).
        let result = c.pixel_to_storage(0, 100, scale);
        assert_eq!(result, None);
        // Upper-left empty triangle: pixel (50, 0) → physical_x=5,
        // physical_y=0, i = 0 - 5 < 0 → None.
        let result = c.pixel_to_storage(50, 0, scale);
        assert_eq!(result, None);
    }
}
