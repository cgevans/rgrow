use super::base::{GrowError, GrowResult, NumTiles, Point, Tile};
use enum_dispatch::enum_dispatch;
use ndarray::prelude::*;
use serde::{Deserialize, Serialize};

pub trait CanvasCreate: Sized + Canvas {
    type Params;
    fn new_sized(shape: Self::Params) -> GrowResult<Self>;
    fn from_array(arr: Array2<Tile>) -> GrowResult<Self>;
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd, Debug, Serialize, Deserialize)]
pub struct PointSafe2(pub Point);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd, Debug)]
pub struct PointSafeHere(pub Point);

#[enum_dispatch]
pub trait Canvas: std::fmt::Debug + Sync + Send {
    /// # Safety
    /// Assumes that the point is inbounds.  Should not normally be used unwrapped.
    unsafe fn uv_p(&self, p: Point) -> Tile {
        *self.uv_pr(p)
    }
    /// # Safety
    /// Assumes that the point is inbounds.  Should not normally be used unwrapped.
    unsafe fn uv_pr(&self, p: Point) -> &Tile;
    /// # Safety
    /// Assumes that the point is inbounds.  Should not normally be used unwrapped.
    unsafe fn uvm_p(&mut self, p: Point) -> &mut Tile;
    fn u_move_point_n(&self, p: Point) -> Point;
    fn u_move_point_e(&self, p: Point) -> Point;
    fn u_move_point_s(&self, p: Point) -> Point;
    fn u_move_point_w(&self, p: Point) -> Point;
    fn inbounds(&self, p: Point) -> bool;
    fn inbounds_bounds(&self, p: Point, bounds: ((i64, i64), (i64, i64))) -> bool {
        (p.0 as i64 + bounds.0.0 >= 0 ) &
        (p.1 as i64 + bounds.1.0 >= 0 ) &
        (p.0 as i64 + bounds.0.1 < self.nrows() as i64 ) &
        (p.1 as i64 + bounds.1.1 < self.ncols() as i64 )
    }
    fn calc_n_tiles(&self) -> NumTiles;
    fn calc_n_tiles_with_tilearray(&self, should_be_counted: &Array1<bool>) -> NumTiles;
    fn raw_array(&self) -> ArrayView2<Tile>;
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;

    fn set_sa_countabletilearray(
        &mut self,
        p: &PointSafe2,
        t: &Tile,
        _should_be_counted: &Array1<bool>,
    ) {
        self.set_sa(p, t);
    }

    fn tile_at_offset(&self, p: Point, offset: (i64, i64)) -> Tile;

    fn move_sa_n(&self, p: PointSafe2) -> PointSafeHere {
        PointSafeHere(self.u_move_point_n(p.0))
    }
    fn move_sa_e(&self, p: PointSafe2) -> PointSafeHere {
        PointSafeHere(self.u_move_point_e(p.0))
    }
    fn u_move_sa_e(&self, p: PointSafe2) -> PointSafe2 {
        PointSafe2(self.u_move_point_e(p.0))
    }
    fn move_sa_s(&self, p: PointSafe2) -> PointSafeHere {
        PointSafeHere(self.u_move_point_s(p.0))
    }
    fn move_sa_w(&self, p: PointSafe2) -> PointSafeHere {
        PointSafeHere(self.u_move_point_w(p.0))
    }

    fn move_sa_nn(&self, p: PointSafe2) -> PointSafeHere {
        PointSafeHere(self.u_move_point_nn(p.0))
    }
    fn move_sa_ne(&self, p: PointSafe2) -> PointSafeHere {
        PointSafeHere(self.u_move_point_ne(p.0))
    }
    fn move_sa_ee(&self, p: PointSafe2) -> PointSafeHere {
        PointSafeHere(self.u_move_point_ee(p.0))
    }
    fn move_sa_se(&self, p: PointSafe2) -> PointSafeHere {
        PointSafeHere(self.u_move_point_se(p.0))
    }
    fn move_sa_ss(&self, p: PointSafe2) -> PointSafeHere {
        PointSafeHere(self.u_move_point_ss(p.0))
    }
    fn move_sa_sw(&self, p: PointSafe2) -> PointSafeHere {
        PointSafeHere(self.u_move_point_sw(p.0))
    }
    fn move_sa_ww(&self, p: PointSafe2) -> PointSafeHere {
        PointSafeHere(self.u_move_point_ww(p.0))
    }
    fn move_sa_nw(&self, p: PointSafe2) -> PointSafeHere {
        PointSafeHere(self.u_move_point_nw(p.0))
    }

    fn move_sh_n(&self, p: PointSafeHere) -> Point {
        self.u_move_point_n(p.0)
    }
    fn move_sh_e(&self, p: PointSafeHere) -> Point {
        self.u_move_point_e(p.0)
    }
    fn move_sh_s(&self, p: PointSafeHere) -> Point {
        self.u_move_point_s(p.0)
    }
    fn move_sh_w(&self, p: PointSafeHere) -> Point {
        self.u_move_point_w(p.0)
    }

    fn set_sa(&mut self, p: &PointSafe2, t: &Tile) {
        unsafe { *self.uvm_p(p.0) = *t };
    }

    #[inline(always)]
    fn u_move_point_ne(&self, p: Point) -> Point {
        self.u_move_point_e(self.u_move_point_n(p))
    }

    #[inline(always)]
    fn u_move_point_se(&self, p: Point) -> Point {
        self.u_move_point_e(self.u_move_point_s(p))
    }

    #[inline(always)]
    fn u_move_point_sw(&self, p: Point) -> Point {
        self.u_move_point_s(self.u_move_point_w(p))
    }

    #[inline(always)]
    fn u_move_point_nw(&self, p: Point) -> Point {
        self.u_move_point_n(self.u_move_point_w(p))
    }

    #[inline(always)]
    fn u_move_point_nn(&self, p: Point) -> Point {
        self.u_move_point_n(self.u_move_point_n(p))
    }

    #[inline(always)]
    fn u_move_point_ee(&self, p: Point) -> Point {
        self.u_move_point_e(self.u_move_point_e(p))
    }

    #[inline(always)]
    fn u_move_point_ss(&self, p: Point) -> Point {
        self.u_move_point_s(self.u_move_point_s(p))
    }

    #[inline(always)]
    fn u_move_point_ww(&self, p: Point) -> Point {
        self.u_move_point_w(self.u_move_point_w(p))
    }

    fn tile_at_point(&self, p: PointSafe2) -> Tile {
        self.tile_at_offset(p.0, (0, 0))
    }

    fn v_sh(&self, p: PointSafeHere) -> Tile {
        self.tile_at_offset(p.0, (0, 0))
    }

    fn tile_to_n(&self, p: PointSafe2) -> Tile {
        self.tile_at_offset(p.0, (-1, 0))
    }

    fn tile_to_e(&self, p: PointSafe2) -> Tile {
        self.tile_at_offset(p.0, (0, 1))
    }

    fn tile_to_s(&self, p: PointSafe2) -> Tile {
        self.tile_at_offset(p.0, (1, 0))
    }

    fn tile_to_w(&self, p: PointSafe2) -> Tile {
        self.tile_at_offset(p.0, (0, -1))
    }

    fn tile_to_nn(&self, p: PointSafe2) -> Tile {
        self.tile_at_offset(p.0, (-2, 0))
    }

    fn tile_to_ne(&self, p: PointSafe2) -> Tile {
        self.tile_at_offset(p.0, (-1, 1))
    }

    fn tile_to_ee(&self, p: PointSafe2) -> Tile {
        self.tile_at_offset(p.0, (0, 2))
    }

    fn tile_to_se(&self, p: PointSafe2) -> Tile {
        self.tile_at_offset(p.0, (1, 1))
    }

    fn tile_to_ss(&self, p: PointSafe2) -> Tile {
        self.tile_at_offset(p.0, (2, 0))
    }

    fn tile_to_sw(&self, p: PointSafe2) -> Tile {
        self.tile_at_offset(p.0, (1, -1))
    }

    fn tile_to_ww(&self, p: PointSafe2) -> Tile {
        self.tile_at_offset(p.0, (0, -2))
    }

    fn tile_to_nw(&self, p: PointSafe2) -> Tile {
        self.tile_at_offset(p.0, (-1, -1))
    }

    /// # Safety
    /// Assumes that the point is inbounds.  Should not normally be used unwrapped.
    #[inline(always)]
    unsafe fn uv_n(&self, p: Point) -> Tile {
        self.uv_p(self.u_move_point_n(p))
    }
    /// # Safety
    /// Assumes that the point is inbounds.  Should not normally be used unwrapped.
    #[inline(always)]
    unsafe fn uv_e(&self, p: Point) -> Tile {
        self.uv_p(self.u_move_point_e(p))
    }
    /// # Safety
    /// Assumes that the point is inbounds.  Should not normally be used unwrapped.
    #[inline(always)]
    unsafe fn uv_s(&self, p: Point) -> Tile {
        self.uv_p(self.u_move_point_s(p))
    }
    /// # Safety
    /// Assumes that the point is inbounds.  Should not normally be used unwrapped.
    #[inline(always)]
    unsafe fn uv_w(&self, p: Point) -> Tile {
        self.uv_p(self.u_move_point_w(p))
    }

    /// # Safety
    /// Assumes that the point is inbounds.  Should not normally be used unwrapped.
    #[inline(always)]
    unsafe fn uv_nw(&self, p: Point) -> Tile {
        self.uv_p(self.u_move_point_nw(p))
    }

    /// # Safety
    /// Assumes that the point is inbounds.  Should not normally be used unwrapped.
    #[inline(always)]
    unsafe fn uv_ne(&self, p: Point) -> Tile {
        self.uv_p(self.u_move_point_ne(p))
    }

    /// # Safety
    /// Assumes that the point is inbounds.  Should not normally be used unwrapped.
    #[inline(always)]
    unsafe fn uv_sw(&self, p: Point) -> Tile {
        self.uv_p(self.u_move_point_sw(p))
    }

    /// # Safety
    /// Assumes that the point is inbounds.  Should not normally be used unwrapped.
    #[inline(always)]
    unsafe fn uv_se(&self, p: Point) -> Tile {
        self.uv_p(self.u_move_point_se(p))
    }

    /// # Safety
    /// Assumes that the point is inbounds.  Should not normally be used unwrapped.
    #[inline(always)]
    unsafe fn uvm_n(&mut self, p: Point) -> &mut Tile {
        self.uvm_p(self.u_move_point_n(p))
    }
    /// # Safety
    /// Assumes that the point is inbounds.  Should not normally be used unwrapped.
    #[inline(always)]
    unsafe fn uvm_e(&mut self, p: Point) -> &mut Tile {
        self.uvm_p(self.u_move_point_e(p))
    }
    /// # Safety
    /// Assumes that the point is inbounds.  Should not normally be used unwrapped.
    #[inline(always)]
    unsafe fn uvm_s(&mut self, p: Point) -> &mut Tile {
        self.uvm_p(self.u_move_point_s(p))
    }
    /// # Safety
    /// Assumes that the point is inbounds.  Should not normally be used unwrapped.
    #[inline(always)]
    unsafe fn uvm_w(&mut self, p: Point) -> &mut Tile {
        self.uvm_p(self.u_move_point_w(p))
    }

    fn draw(&self, frame: &mut [u8], colors: &[[u8; 4]]) {
        for (p, v) in Iterator::zip(frame.chunks_exact_mut(4), self.raw_array().iter()) {
            let color = colors[*v as usize];
            p.copy_from_slice(&color);
        }
    }

    fn draw_scaled(
        &self,
        frame: &mut [u8],
        colors: &[[u8; 4]],
        tile_size: usize,
        edge_size: usize,
    ) {
        let scale = tile_size + 2 * edge_size;
        let csc = self.ncols() * scale;

        for (i, pixel) in frame.chunks_exact_mut(4).enumerate() {
            let framex = i % csc;
            let framey = i / csc;

            let x = framex / scale;
            let y = framey / scale;

            let blockx = framex % scale;
            let blocky = framey % scale;

            let tv = unsafe { self.uv_p((y, x)) };

            pixel.copy_from_slice(
                &(if (tv > 0)
                    & (blockx + 1 > edge_size)
                    & (blocky + 1 > edge_size)
                    & (blockx < edge_size + tile_size)
                    & (blocky < edge_size + tile_size)
                {
                    colors[tv as usize]
                } else {
                    [0, 0, 0, 0x00]
                }),
            );
        }
    }

    fn draw_scaled_with_mm(
        &self,
        frame: &mut [u8],
        colors: &[[u8; 4]],
        mismatches: Array2<usize>,
        tile_size: usize,
        edge_size: usize,
    ) {
        let scale = tile_size + 2 * edge_size;
        let csc = self.ncols() * scale;

        for (i, pixel) in frame.chunks_exact_mut(4).enumerate() {
            let framex = i % csc;
            let framey = i / csc;

            let x = framex / scale;
            let y = framey / scale;

            let blockx = framex % scale;
            let blocky = framey % scale;

            let tv = unsafe { self.uv_p((y, x)) };

            pixel.copy_from_slice(
                &(if (tv > 0)
                    & (blockx + 1 > edge_size)
                    & (blocky + 1 > edge_size)
                    & (blockx < edge_size + tile_size)
                    & (blocky < edge_size + tile_size)
                {
                    colors[tv as usize]
                } else if ((blockx < edge_size)
                    & (blocky + 1 > edge_size)
                    & (blocky < edge_size + tile_size)
                    & (mismatches[(y, x)] & 0b0001 == 0b0001))
                    | ((blockx >= edge_size + tile_size)
                        & (blocky + 1 > edge_size)
                        & (blocky < edge_size + tile_size)
                        & (mismatches[(y, x)] & 0b0100 == 0b0100))
                    | ((blocky < edge_size)
                        & (blockx + 1 > edge_size)
                        & (blockx < edge_size + tile_size)
                        & (mismatches[(y, x)] & 0b1000 == 0b1000))
                    | ((blocky >= edge_size + tile_size)
                        & (blockx + 1 > edge_size)
                        & (blockx < edge_size + tile_size)
                        & (mismatches[(y, x)] & 0b010 == 0b010))
                {
                    [0xff, 0x00, 0x00, 0xff]
                } else {
                    [0, 0, 0, 0x00]
                }),
            );
        }
    }

    fn draw_size(&self) -> (u32, u32) {
        (self.ncols() as u32, self.nrows() as u32)
    }

    fn center(&self) -> PointSafe2 {
        PointSafe2((self.nrows() / 2, self.ncols() / 2))
    }
}

#[derive(Clone, Debug)]
pub struct CanvasSquare(Array2<Tile>);

impl CanvasCreate for CanvasSquare {
    type Params = (usize, usize);

    fn new_sized(shape: Self::Params) -> GrowResult<Self> {
        Ok(Self(Array2::zeros(shape)))
    }

    fn from_array(arr: Array2<Tile>) -> GrowResult<Self> {
        Ok(Self(arr))
    }
}

impl Canvas for CanvasSquare {
    fn tile_at_offset(&self, p: Point, offset: (i64, i64)) -> Tile {
        let x = (p.0 as i64) + (offset.0);
        if x < 0 {
            return 0;
        }
        let y = (p.1 as i64) + (offset.1);
        if y < 0 {
            return 0;
        }
        if !self.inbounds((x as usize, y as usize)) { return 0; }
        unsafe { self.uv_p((x as usize, y as usize)) }
    }

    #[inline(always)]
    unsafe fn uv_pr(&self, p: Point) -> &Tile {
        self.0.uget(p)
    }

    #[inline(always)]
    unsafe fn uvm_p(&mut self, p: Point) -> &mut Tile {
        self.0.uget_mut(p)
    }

    #[inline(always)]
    fn inbounds(&self, p: Point) -> bool {
        (p.0 < self.nrows() - 0) & (p.1 < self.ncols() - 0)
    }

    #[inline(always)]
    fn u_move_point_n(&self, p: Point) -> Point {
        (p.0 - 1, p.1)
    }

    #[inline(always)]
    fn u_move_point_e(&self, p: Point) -> Point {
        (p.0, p.1 + 1)
    }

    #[inline(always)]
    fn u_move_point_s(&self, p: Point) -> Point {
        (p.0 + 1, p.1)
    }

    #[inline(always)]
    fn u_move_point_w(&self, p: Point) -> Point {
        (p.0, p.1 - 1)
    }

    #[inline(always)]
    fn u_move_point_ne(&self, p: Point) -> Point {
        (p.0 - 1, p.1 + 1)
    }

    #[inline(always)]
    fn u_move_point_se(&self, p: Point) -> Point {
        (p.0 + 1, p.1 + 1)
    }

    #[inline(always)]
    fn u_move_point_sw(&self, p: Point) -> Point {
        (p.0 + 1, p.1 - 1)
    }

    #[inline(always)]
    fn u_move_point_nw(&self, p: Point) -> Point {
        (p.0 - 1, p.1 - 1)
    }

    #[inline(always)]
    fn calc_n_tiles(&self) -> NumTiles {
        self.0.fold(0, |x, y| x + u32::from(*y != 0))
    }

    fn raw_array(&self) -> ArrayView2<Tile> {
        self.0.view()
    }

    fn nrows(&self) -> usize {
        self.0.nrows()
    }

    fn ncols(&self) -> usize {
        self.0.ncols()
    }

    fn calc_n_tiles_with_tilearray(&self, should_be_counted: &Array1<bool>) -> NumTiles {
        self.0
            .fold(0, |x, y| x + u32::from(should_be_counted[*y as usize]))
    }
}

#[derive(Debug, Clone)]
pub struct CanvasPeriodic(Array2<Tile>);

impl CanvasCreate for CanvasPeriodic {
    type Params = (usize, usize);

    fn new_sized(shape: Self::Params) -> GrowResult<Self> {
        Ok(Self(Array2::zeros(shape)))
    }

    fn from_array(arr: Array2<Tile>) -> GrowResult<Self> {
        Ok(Self(arr))
    }
}

impl Canvas for CanvasPeriodic {
    unsafe fn uv_pr(&self, p: Point) -> &Tile {
        self.0.uget(p)
    }

    unsafe fn uvm_p(&mut self, p: Point) -> &mut Tile {
        self.0.uget_mut(p)
    }

    fn u_move_point_n(&self, p: Point) -> Point {
        if p.0 == 0 {
            (self.0.nrows() - 1, p.1)
        } else {
            (p.0 - 1, p.1)
        }
    }

    fn tile_at_offset(&self, p: Point, offset: (i64, i64)) -> Tile {
        let dims = self.0.dim();

        let x = (p.0 as i64 + offset.0 as i64).rem_euclid(dims.0 as i64) as usize;
        let y = (p.1 as i64 + offset.1 as i64).rem_euclid(dims.1 as i64) as usize;

        unsafe { self.uv_p((x, y)) }
    }

    fn inbounds_bounds(&self, p: Point, _bounds: ((i64, i64), (i64, i64))) -> bool {
        self.inbounds(p)
    }

    fn u_move_point_e(&self, p: Point) -> Point {
        (p.0, (p.1 + 1) % self.0.ncols())
    }

    fn u_move_point_s(&self, p: Point) -> Point {
        (((p.0 + 1) % self.0.nrows()), p.1)
    }

    fn u_move_point_w(&self, p: Point) -> Point {
        if p.1 == 0 {
            (p.0, self.0.ncols() - 1)
        } else {
            (p.0, p.1 - 1)
        }
    }

    fn inbounds(&self, p: Point) -> bool {
        (p.0 < self.0.nrows()) & (p.1 < self.0.ncols())
    }

    fn calc_n_tiles(&self) -> NumTiles {
        self.0.fold(0, |x, y| x + u32::from(*y != 0))
    }

    fn calc_n_tiles_with_tilearray(&self, should_be_counted: &Array1<bool>) -> NumTiles {
        self.0
            .fold(0, |x, y| x + u32::from(should_be_counted[*y as usize]))
    }

    fn raw_array(&self) -> ArrayView2<Tile> {
        self.0.view()
    }

    fn nrows(&self) -> usize {
        self.0.nrows()
    }

    fn ncols(&self) -> usize {
        self.0.ncols()
    }
}

#[derive(Debug, Clone)]
pub struct CanvasTube(Array2<Tile>);

impl CanvasCreate for CanvasTube {
    type Params = (usize, usize);

    fn new_sized(shape: Self::Params) -> GrowResult<Self> {
        let width = shape.0;
        if width % 2 != 0 {
            Err(GrowError::WrongCanvasSize(width, shape.1))
        } else {
            Ok(Self(Array2::zeros(shape)))
        }
    }

    fn from_array(arr: Array2<Tile>) -> GrowResult<Self> {
        if arr.shape()[0] % 2 != 0 {
            Err(GrowError::WrongCanvasSize(arr.shape()[0], arr.shape()[1]))
        } else {
            Ok(Self(arr))
        }
    }
}

impl Canvas for CanvasTube {
    fn tile_at_offset(&self, p: Point, offset: (i64, i64)) -> Tile {
        let x = (p.0 as i64) + (offset.0);
        if x < 0 {
            return 0;
        }
        let y = (p.1 as i64) + (offset.1);
        if y < 0 {
            return 0;
        }
        if !self.inbounds((x as usize, y as usize)) { return 0; }
        unsafe { self.uv_p((x as usize, y as usize)) }
    }

    unsafe fn uv_pr(&self, p: Point) -> &Tile {
        self.0.uget(p)
    }

    unsafe fn uvm_p(&mut self, p: Point) -> &mut Tile {
        self.0.uget_mut(p)
    }

    fn u_move_point_n(&self, p: Point) -> Point {
        if p.0 == 0 {
            (self.nrows() - 1, p.1 - 2)
        } else {
            (p.0 - 1, p.1)
        }
    }

    fn u_move_point_e(&self, p: Point) -> Point {
        if p.0 == 0 {
            (self.nrows() - 1, p.1 - 1)
        } else {
            (p.0 - 1, p.1 + 1)
        }
    }

    fn u_move_point_s(&self, p: Point) -> Point {
        if p.0 == self.nrows() - 1 {
            (0, p.1 + 2)
        } else {
            (p.0 + 1, p.1)
        }
    }

    fn u_move_point_w(&self, p: Point) -> Point {
        if p.0 == self.nrows() - 1 {
            (0, p.1 + 1)
        } else {
            (p.0 + 1, p.1 - 1)
        }
    }

    fn inbounds(&self, p: Point) -> bool {
        let (xs, ys) = self.0.dim();
        (p.0 < xs) & (p.1 < ys - 2 - (self.nrows() / 2)) & (p.1 >= 2 + (self.nrows() / 2))
    }

    fn calc_n_tiles(&self) -> NumTiles {
        self.0.fold(0, |x, y| x + u32::from(*y != 0))
    }

    fn calc_n_tiles_with_tilearray(&self, should_be_counted: &Array1<bool>) -> NumTiles {
        self.0
            .fold(0, |x, y| x + u32::from(should_be_counted[*y as usize]))
    }

    fn raw_array(&self) -> ArrayView2<Tile> {
        self.0.view()
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

    fn draw(&self, frame: &mut [u8], colors: &[[u8; 4]]) {
        let s = self.nrows() + self.ncols();
        let mut px: usize;
        let mut py: usize;
        let mut pos: usize;

        for ((x, y), t) in self.0.indexed_iter() {
            py = y;
            px = x + y;
            pos = 4 * (px * s + py);
            frame[pos..pos + 4].copy_from_slice(&colors[*t as usize])
        }
    }

    fn draw_scaled(
        &self,
        frame: &mut [u8],
        colors: &[[u8; 4]],
        tile_size: usize,
        edge_size: usize,
    ) {
        let scale = tile_size + 2 * edge_size;
        let csc = self.ncols() * scale;

        for (i, pixel) in frame.chunks_exact_mut(4).enumerate() {
            let framex = i % csc;
            let framey = i / csc;

            let y = framey / scale;
            let xr = framex / scale;
            if xr > y {
                continue;
            }
            if xr + y > self.nrows() - 1 {
                continue;
            }
            let x = y - xr;

            let blockx = framex % scale;
            let blocky = framey % scale;

            let tv = unsafe { self.uv_p((y, x)) };

            pixel.copy_from_slice(
                &(if (tv > 0)
                    & (blockx + 1 > edge_size)
                    & (blocky + 1 > edge_size)
                    & (blockx < edge_size + tile_size)
                    & (blocky < edge_size + tile_size)
                {
                    colors[tv as usize]
                } else {
                    [0, 0, 0, 0x00]
                }),
            );
        }
    }

    fn draw_scaled_with_mm(
        &self,
        frame: &mut [u8],
        colors: &[[u8; 4]],
        mismatches: Array2<usize>,
        tile_size: usize,
        edge_size: usize,
    ) {
        let scale = tile_size + 2 * edge_size;
        let csc = self.ncols() * scale;

        for (i, pixel) in frame.chunks_exact_mut(4).enumerate() {
            let framex = i % csc;
            let framey = i / csc;

            let y = framey / scale;
            let xr = framex / scale;
            if xr > y {
                continue;
            }
            if xr + y > self.nrows() - 1 {
                continue;
            }
            let x = y - xr;

            let blockx = framex % scale;
            let blocky = framey % scale;

            let tv = unsafe { self.uv_p((y, x)) };

            pixel.copy_from_slice(
                &(if (tv > 0)
                    & (blockx + 1 > edge_size)
                    & (blocky + 1 > edge_size)
                    & (blockx < edge_size + tile_size)
                    & (blocky < edge_size + tile_size)
                {
                    colors[tv as usize]
                } else if ((blockx < edge_size)
                    & (blocky + 1 > edge_size)
                    & (blocky < edge_size + tile_size)
                    & (mismatches[(y, x)] & 0b0001 == 0b0001))
                    | ((blockx >= edge_size + tile_size)
                        & (blocky + 1 > edge_size)
                        & (blocky < edge_size + tile_size)
                        & (mismatches[(y, x)] & 0b0100 == 0b0100))
                    | ((blocky < edge_size)
                        & (blockx + 1 > edge_size)
                        & (blockx < edge_size + tile_size)
                        & (mismatches[(y, x)] & 0b1000 == 0b1000))
                    | ((blocky >= edge_size + tile_size)
                        & (blockx + 1 > edge_size)
                        & (blockx < edge_size + tile_size)
                        & (mismatches[(y, x)] & 0b010 == 0b010))
                {
                    [0xff, 0x00, 0x00, 0xff]
                } else {
                    [0, 0, 0, 0x00]
                }),
            );
        }
    }

    fn center(&self) -> PointSafe2 {
        PointSafe2((self.nrows() / 2, self.ncols() / 2))
    }
}
