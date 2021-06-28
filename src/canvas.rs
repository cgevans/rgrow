use super::base::{GrowError, GrowResult, NumTiles, Point, Tile};
use ndarray::prelude::*;

pub trait CanvasCreate: Sized + Canvas {
    fn from_array(canvas: Array2<Tile>) -> GrowResult<Self>;
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd, Debug)]
pub struct PointSafeAdjs(pub Point);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd, Debug)]
pub struct PointSafeHere(pub Point);

pub trait Canvas: std::fmt::Debug {
    unsafe fn uv_p(&self, p: Point) -> Tile {
        *self.uv_pr(p)
    }
    unsafe fn uv_pr(&self, p: Point) -> &Tile;
    unsafe fn uvm_p(&mut self, p: Point) -> &mut Tile;
    fn u_move_point_n(&self, p: Point) -> Point;
    fn u_move_point_e(&self, p: Point) -> Point;
    fn u_move_point_s(&self, p: Point) -> Point;
    fn u_move_point_w(&self, p: Point) -> Point;
    fn inbounds(&self, p: Point) -> bool;
    fn calc_ntiles(&self) -> NumTiles;
    fn raw_array(&self) -> ArrayView2<Tile>;
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;

    fn move_sa_n(&self, p: PointSafeAdjs) -> PointSafeHere {
        PointSafeHere(self.u_move_point_n(p.0))
    }
    fn move_sa_e(&self, p: PointSafeAdjs) -> PointSafeHere {
        PointSafeHere(self.u_move_point_e(p.0))
    }
    fn move_sa_s(&self, p: PointSafeAdjs) -> PointSafeHere {
        PointSafeHere(self.u_move_point_s(p.0))
    }
    fn move_sa_w(&self, p: PointSafeAdjs) -> PointSafeHere {
        PointSafeHere(self.u_move_point_w(p.0))
    }

    fn move_sa_nw(&self, p: PointSafeAdjs) -> PointSafeHere {
        PointSafeHere(self.u_move_point_nw(p.0))
    }
    fn move_sa_ne(&self, p: PointSafeAdjs) -> PointSafeHere {
        PointSafeHere(self.u_move_point_ne(p.0))
    }
    fn move_sa_se(&self, p: PointSafeAdjs) -> PointSafeHere {
        PointSafeHere(self.u_move_point_se(p.0))
    }
    fn move_sa_sw(&self, p: PointSafeAdjs) -> PointSafeHere {
        PointSafeHere(self.u_move_point_sw(p.0))
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

    fn set_sa(&mut self, p: &PointSafeAdjs, t: &Tile) {
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

    fn v_sa(&self, p: PointSafeAdjs) -> Tile {
        unsafe { self.uv_p(p.0) }
    }

    fn v_sh(&self, p: PointSafeHere) -> Tile {
        unsafe { self.uv_p(p.0) }
    }

    fn v_sa_n(&self, p: PointSafeAdjs) -> Tile {
        self.v_sh(self.move_sa_n(p))
    }

    fn v_sa_e(&self, p: PointSafeAdjs) -> Tile {
        self.v_sh(self.move_sa_e(p))
    }

    fn v_sa_s(&self, p: PointSafeAdjs) -> Tile {
        self.v_sh(self.move_sa_s(p))
    }

    fn v_sa_w(&self, p: PointSafeAdjs) -> Tile {
        self.v_sh(self.move_sa_w(p))
    }

    fn v_sa_nw(&self, p: PointSafeAdjs) -> Tile {
        self.v_sh(self.move_sa_nw(p))
    }

    fn v_sa_ne(&self, p: PointSafeAdjs) -> Tile {
        self.v_sh(self.move_sa_ne(p))
    }

    fn v_sa_se(&self, p: PointSafeAdjs) -> Tile {
        self.v_sh(self.move_sa_se(p))
    }

    fn v_sa_sw(&self, p: PointSafeAdjs) -> Tile {
        self.v_sh(self.move_sa_sw(p))
    }

    #[inline(always)]
    unsafe fn uv_n(&self, p: Point) -> Tile {
        self.uv_p(self.u_move_point_n(p))
    }
    #[inline(always)]
    unsafe fn uv_e(&self, p: Point) -> Tile {
        self.uv_p(self.u_move_point_e(p))
    }
    #[inline(always)]
    unsafe fn uv_s(&self, p: Point) -> Tile {
        self.uv_p(self.u_move_point_s(p))
    }
    #[inline(always)]
    unsafe fn uv_w(&self, p: Point) -> Tile {
        self.uv_p(self.u_move_point_w(p))
    }

    #[inline(always)]
    unsafe fn uv_nw(&self, p: Point) -> Tile {
        self.uv_p(self.u_move_point_nw(p))
    }

    #[inline(always)]
    unsafe fn uv_ne(&self, p: Point) -> Tile {
        self.uv_p(self.u_move_point_ne(p))
    }

    #[inline(always)]
    unsafe fn uv_sw(&self, p: Point) -> Tile {
        self.uv_p(self.u_move_point_sw(p))
    }

    #[inline(always)]
    unsafe fn uv_se(&self, p: Point) -> Tile {
        self.uv_p(self.u_move_point_se(p))
    }

    #[inline(always)]
    unsafe fn uvm_n(&mut self, p: Point) -> &mut Tile {
        self.uvm_p(self.u_move_point_n(p))
    }
    #[inline(always)]
    unsafe fn uvm_e(&mut self, p: Point) -> &mut Tile {
        self.uvm_p(self.u_move_point_e(p))
    }
    #[inline(always)]
    unsafe fn uvm_s(&mut self, p: Point) -> &mut Tile {
        self.uvm_p(self.u_move_point_s(p))
    }
    #[inline(always)]
    unsafe fn uvm_w(&mut self, p: Point) -> &mut Tile {
        self.uvm_p(self.u_move_point_w(p))
    }
}

pub trait CanvasSquarable: Canvas {
    fn square_size(&self) -> usize;
}

#[derive(Clone, Debug)]
pub struct CanvasSquare {
    canvas: Array2<Tile>,
    size: usize,
}

impl CanvasSquarable for CanvasSquare {
    #[inline(always)]
    fn square_size(&self) -> usize {
        return self.size;
    }
}

impl CanvasCreate for CanvasSquare {
    fn from_array(canvas: Array2<Tile>) -> GrowResult<Self> {
        let size = canvas.nrows();
        if canvas.nrows() != canvas.ncols() {
            Err(GrowError::WrongCanvasSize(size, canvas.ncols()))
        } else if (size & (size - 1)) != 0 {
            Err(GrowError::WrongCanvasSize(size, canvas.ncols()))
        } else {
            Ok(Self { canvas, size })
        }
    }
}

impl Canvas for CanvasSquare {
    #[inline(always)]
    unsafe fn uv_pr(&self, p: Point) -> &Tile {
        self.canvas.uget(p)
    }

    #[inline(always)]
    unsafe fn uvm_p(&mut self, p: Point) -> &mut Tile {
        self.canvas.uget_mut(p)
    }

    #[inline(always)]
    fn inbounds(&self, p: Point) -> bool {
        return (p.0 >= 1) & (p.1 >= 1) & (p.0 < self.size - 1) & (p.1 < self.size - 1);
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
    fn calc_ntiles(&self) -> NumTiles {
        self.canvas
            .fold(0, |x, y| x + (if *y == 0 { 0 } else { 1 }))
    }

    fn raw_array(&self) -> ArrayView2<Tile> {
        self.canvas.view()
    }

    fn nrows(&self) -> usize {
        self.canvas.nrows()
    }

    fn ncols(&self) -> usize {
        self.canvas.ncols()
    }
}

#[derive(Debug, Clone)]
pub struct CanvasPeriodic {
    values: Array2<Tile>,
}

impl CanvasCreate for CanvasPeriodic {
    fn from_array(values: Array2<Tile>) -> GrowResult<Self> {
        Ok(Self { values })
    }
}

impl CanvasSquarable for CanvasPeriodic {
    fn square_size(&self) -> usize {
        let largest = self.values.nrows().max(self.values.ncols());
        2usize.pow(f64::log2(largest as f64).ceil() as u32)
    }
}

impl Canvas for CanvasPeriodic {
    unsafe fn uv_pr(&self, p: Point) -> &Tile {
        &self.values.uget(p)
    }

    unsafe fn uvm_p(&mut self, p: Point) -> &mut Tile {
        self.values.uget_mut(p)
    }

    fn u_move_point_n(&self, p: Point) -> Point {
        if p.0 == 0 {
            (self.values.nrows() - 1, p.1)
        } else {
            (p.0 - 1, p.1)
        }
    }

    fn u_move_point_e(&self, p: Point) -> Point {
        (p.0, (p.1 + 1) % self.values.ncols())
    }

    fn u_move_point_s(&self, p: Point) -> Point {
        (((p.0 + 1) % self.values.nrows()), p.1)
    }

    fn u_move_point_w(&self, p: Point) -> Point {
        if p.1 == 0 {
            (p.0, self.values.ncols() - 1)
        } else {
            (p.0, p.1 - 1)
        }
    }

    fn inbounds(&self, p: Point) -> bool {
        (p.0 < self.values.nrows()) & (p.1 < self.values.ncols())
    }

    fn calc_ntiles(&self) -> NumTiles {
        self.values
            .fold(0, |x, y| x + (if *y == 0 { 0 } else { 1 }))
    }

    fn raw_array(&self) -> ArrayView2<Tile> {
        self.values.view()
    }

    fn nrows(&self) -> usize {
        self.values.nrows()
    }

    fn ncols(&self) -> usize {
        self.values.ncols()
    }
}

#[derive(Debug)]
pub struct CanvasTube {
    values: Array2<Tile>,
}

impl Canvas for CanvasTube {
    unsafe fn uv_pr(&self, p: Point) -> &Tile {
        self.values.uget((p.1 - p.0, p.0))
    }

    unsafe fn uvm_p(&mut self, p: Point) -> &mut Tile {
        self.values.uget_mut((p.1 - p.0, p.0))
    }

    fn u_move_point_n(&self, p: Point) -> Point {
        (p.0 - 1, p.1)
    }

    fn u_move_point_e(&self, p: Point) -> Point {
        (p.0 - 1, p.1 + 1)
    }

    fn u_move_point_s(&self, p: Point) -> Point {
        (p.0 + 1, p.1)
    }

    fn u_move_point_w(&self, p: Point) -> Point {
        (p.0 + 1, p.1 - 1)
    }

    fn inbounds(&self, p: Point) -> bool {
        let (ys, xs) = self.values.dim();
        (p.1 < xs) & (p.0 - p.1 < ys) & (p.0 - p.1 > xs)
    }

    fn calc_ntiles(&self) -> NumTiles {
        self.values
            .fold(0, |x, y| x + (if *y == 0 { 0 } else { 1 }))
    }

    fn raw_array(&self) -> ArrayView2<Tile> {
        self.values.view()
    }

    fn nrows(&self) -> usize {
        self.values.nrows()
    }

    fn ncols(&self) -> usize {
        self.values.ncols()
    }
}
