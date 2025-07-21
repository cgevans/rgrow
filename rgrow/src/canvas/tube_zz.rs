use crate::base::{GrowError, GrowResult, NumTiles, Point, Tile};
use crate::canvas::{Canvas, CanvasCreate, PointSafe2};
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
            Err(GrowError::WrongCanvasSize(width, shape.1))
        } else {
            Ok(Self(Array2::zeros(shape)))
        }
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

    fn draw_size(&self) -> (u32, u32) {
        let s = (self.nrows() + self.ncols()) as u32;
        (s, s)
    }

    fn draw(&self, _frame: &mut [u8], _colors: &[[u8; 4]]) {
        // let s = self.nrows() + self.ncols();
        // let mut pi: usize;
        // let mut pj: usize;
        // let mut pos: usize;

        // for ((i, j), t) in self.0.indexed_iter() {
        //     pj = j;
        //     pi = i + j;
        //     pos = 4 * (pi * s + pj);
        //     frame[pos..pos + 4].copy_from_slice(&colors[*t as usize])
        // }
        todo!()
    }

    fn draw_scaled(
        &self,
        _frame: &mut [u8],
        _colors: &[[u8; 4]],
        _tile_size: usize,
        _edge_size: usize,
    ) {
        // let scale = tile_size + 2 * edge_size;
        // let csc = self.ncols() * scale;

        // for (i, pixel) in frame.chunks_exact_mut(4).enumerate() {
        //     let framex = i % csc;
        //     let framey = i / csc;

        //     let y = framey / scale;
        //     let xr = framex / scale;
        //     if xr > y {
        //         continue;
        //     }
        //     if xr + y > self.nrows() - 1 {
        //         continue;
        //     }
        //     let x = y - xr;

        //     let blockx = framex % scale;
        //     let blocky = framey % scale;

        //     let tv = unsafe { self.uv_p((y, x)) };

        //     pixel.copy_from_slice(
        //         &(if (tv > 0)
        //             & (blockx + 1 > edge_size)
        //             & (blocky + 1 > edge_size)
        //             & (blockx < edge_size + tile_size)
        //             & (blocky < edge_size + tile_size)
        //         {
        //             colors[tv as usize]
        //         } else {
        //             [0, 0, 0, 0x00]
        //         }),
        //     );
        // }
        todo!()
    }

    fn draw_scaled_with_mm(
        &self,
        _frame: &mut [u8],
        _colors: &[[u8; 4]],
        _mismatches: Array2<usize>,
        _tile_size: usize,
        _edge_size: usize,
    ) {
        // let scale = tile_size + 2 * edge_size;
        // let csc = self.ncols() * scale;

        // for (i, pixel) in frame.chunks_exact_mut(4).enumerate() {
        //     let framex = i % csc;
        //     let framey = i / csc;

        //     let y = framey / scale;
        //     let xr = framex / scale;
        //     if xr > y {
        //         continue;
        //     }
        //     if xr + y > self.nrows() - 1 {
        //         continue;
        //     }
        //     let x = y - xr;

        //     let blockx = framex % scale;
        //     let blocky = framey % scale;

        //     let tv = unsafe { self.uv_p((y, x)) };

        //     pixel.copy_from_slice(
        //         &(if (tv > 0)
        //             & (blockx + 1 > edge_size)
        //             & (blocky + 1 > edge_size)
        //             & (blockx < edge_size + tile_size)
        //             & (blocky < edge_size + tile_size)
        //         {
        //             colors[tv as usize]
        //         } else if ((blockx < edge_size)
        //             & (blocky + 1 > edge_size)
        //             & (blocky < edge_size + tile_size)
        //             & (mismatches[(y, x)] & 0b0001 == 0b0001))
        //             | ((blockx >= edge_size + tile_size)
        //                 & (blocky + 1 > edge_size)
        //                 & (blocky < edge_size + tile_size)
        //                 & (mismatches[(y, x)] & 0b0100 == 0b0100))
        //             | ((blocky < edge_size)
        //                 & (blockx + 1 > edge_size)
        //                 & (blockx < edge_size + tile_size)
        //                 & (mismatches[(y, x)] & 0b1000 == 0b1000))
        //             | ((blocky >= edge_size + tile_size)
        //                 & (blockx + 1 > edge_size)
        //                 & (blockx < edge_size + tile_size)
        //                 & (mismatches[(y, x)] & 0b010 == 0b010))
        //         {
        //             [0xff, 0x00, 0x00, 0xff]
        //         } else {
        //             [0, 0, 0, 0x00]
        //         }),
        //     );
        // }
        todo!()
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
}
