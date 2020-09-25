use super::base::{Point, Tile, NumTiles};
use ndarray::prelude::*;

pub trait CanvasCreate {
    fn from_array(canvas: Array2<Tile>) -> Self;
}

pub trait Canvas: std::fmt::Debug {
    unsafe fn uv_p(&self, p: Point) -> Tile;
    unsafe fn uvm_p(&mut self, p: Point) -> &mut Tile;
    fn u_move_point_n(&self, p: Point) -> Point;
    fn u_move_point_e(&self, p: Point) -> Point;
    fn u_move_point_s(&self, p: Point) -> Point;
    fn u_move_point_w(&self, p: Point) -> Point;
    fn inbounds(&self, p: Point) -> bool;
    fn calc_ntiles(&self) -> NumTiles;

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
    fn size(&self) -> usize;
}

#[derive(Clone, Debug)]
pub struct CanvasSquare {
    pub canvas: Array2<Tile>,
    size: usize,
}

impl CanvasSquarable for CanvasSquare {
    #[inline(always)]
    fn size(&self) -> usize {
        return self.size;
    }
}

impl CanvasCreate for CanvasSquare {
    fn from_array(canvas: Array2<Tile>) -> Self {
        let size = canvas.nrows();
        Self { canvas, size }
    }
}

impl Canvas for CanvasSquare {
    #[inline(always)]
    unsafe fn uv_p(&self, p: Point) -> Tile {
        *self.canvas.uget(p)
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
        (p.0-1, p.1)
    }

    #[inline(always)]
    fn u_move_point_e(&self, p: Point) -> Point {
        (p.0, p.1+1)
    }

    #[inline(always)]
    fn u_move_point_s(&self, p: Point) -> Point {
        (p.0+1, p.1)
    }

    #[inline(always)]
    fn u_move_point_w(&self, p: Point) -> Point {
        (p.0, p.1-1)
    }


    #[inline(always)]
    fn u_move_point_ne(&self, p: Point) -> Point {
        (p.0-1, p.1+1)
    }

    #[inline(always)]
    fn u_move_point_se(&self, p: Point) -> Point {
        (p.0+1, p.1+1)
    }

    #[inline(always)]
    fn u_move_point_sw(&self, p: Point) -> Point {
        (p.0+1, p.1-1)
    }

    #[inline(always)]
    fn u_move_point_nw(&self, p: Point) -> Point {
        (p.0-1, p.1-1)
    }

    #[inline(always)]
    fn calc_ntiles(&self) -> NumTiles {
        self
        .canvas
        .fold(0, |x, y| x + (if *y == 0 { 0 } else { 1 }))
    }
}

#[derive(Debug)]
pub struct CanvasTube {
    values: Array2<Tile>
}

impl Canvas for CanvasTube {
    unsafe fn uv_p(&self, p: Point) -> Tile {
        *self.values.uget((p.1 - p.0, p.0))
    }

    unsafe fn uvm_p(&mut self, p: Point) -> &mut Tile {
        self.values.uget_mut((p.1 - p.0, p.0))
    }

    fn u_move_point_n(&self, p: Point) -> Point {
        (p.0-1, p.1)
    }

    fn u_move_point_e(&self, p: Point) -> Point {
        (p.0-1, p.1+1)
    }

    fn u_move_point_s(&self, p: Point) -> Point {
        (p.0+1, p.1)
    }

    fn u_move_point_w(&self, p: Point) -> Point {
        (p.0+1, p.1-1)
    }

    fn inbounds(&self, p: Point) -> bool {
        let (ys, xs) = self.values.dim();
        (p.1 < xs) & (p.0 - p.1 < ys) & (p.0 -p.1 > xs)
    }

    fn calc_ntiles(&self) -> NumTiles {
        todo!()
    }
}