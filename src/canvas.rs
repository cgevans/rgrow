use super::base::{Point, Tile, NumTiles};
use ndarray::prelude::*;
pub trait Canvas {
    unsafe fn uv_p(&self, p: Point) -> Tile;
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
    unsafe fn uv_nw(&self, p: Point) -> Tile;
    unsafe fn uv_ne(&self, p: Point) -> Tile;
    unsafe fn uv_sw(&self, p: Point) -> Tile;
    unsafe fn uv_se(&self, p: Point) -> Tile;
    unsafe fn uvm_n(&mut self, p: Point) -> &mut Tile;
    unsafe fn uvm_e(&mut self, p: Point) -> &mut Tile;
    unsafe fn uvm_s(&mut self, p: Point) -> &mut Tile;
    unsafe fn uvm_w(&mut self, p: Point) -> &mut Tile;
    unsafe fn uvm_p(&mut self, p: Point) -> &mut Tile;
    fn u_move_point_n(&self, p: Point) -> Point;
    fn u_move_point_e(&self, p: Point) -> Point;
    fn u_move_point_s(&self, p: Point) -> Point;
    fn u_move_point_w(&self, p: Point) -> Point;
    fn inbounds(&self, p: Point) -> bool;
    fn calc_ntiles(&self) -> NumTiles;
    fn from_canvas(canvas: Array2<Tile>) -> Self;
}

pub trait CanvasSize {
    fn size(&self) -> usize;
    fn raw_dim(&self) -> ndarray::Dim<[usize; 2]>;
}

#[derive(Clone, Debug)]
pub struct CanvasSquare {
    pub canvas: Array2<Tile>,
    size: usize,
}

impl CanvasSize for CanvasSquare {
    #[inline(always)]
    fn size(&self) -> usize {
        return self.size;
    }

    fn raw_dim(&self) -> ndarray::Dim<[usize; 2]> {
        self.canvas.raw_dim()
    }
}

impl Canvas for CanvasSquare {
    fn from_canvas(canvas: Array2<Tile>) -> Self {
        let size = canvas.nrows();
        Self { canvas, size }
    }


    #[inline(always)]
    unsafe fn uv_n(&self, p: Point) -> Tile {
        *self.canvas.uget((p.0 - 1, p.1))
    }

    #[inline(always)]
    unsafe fn uv_nw(&self, p: Point) -> Tile {
        *self.canvas.uget((p.0 - 1, p.1 -1))
    }

    #[inline(always)]
    unsafe fn uv_ne(&self, p: Point) -> Tile {
        *self.canvas.uget((p.0 - 1, p.1 + 1))
    }

    #[inline(always)]
    unsafe fn uv_sw(&self, p: Point) -> Tile {
        *self.canvas.uget((p.0 + 1, p.1 - 1))
    }

    #[inline(always)]
    unsafe fn uv_se(&self, p: Point) -> Tile {
        *self.canvas.uget((p.0 + 1, p.1 + 1))
    }


    #[inline(always)]
    unsafe fn uv_e(&self, p: Point) -> Tile {
        *self.canvas.uget((p.0, p.1 + 1))
    }

    #[inline(always)]
    unsafe fn uv_s(&self, p: Point) -> Tile {
        *self.canvas.uget((p.0 + 1, p.1))
    }

    #[inline(always)]
    unsafe fn uv_w(&self, p: Point) -> Tile {
        *self.canvas.uget((p.0, p.1 - 1))
    }

    #[inline(always)]
    unsafe fn uv_p(&self, p: Point) -> Tile {
        *self.canvas.uget(p)
    }

    #[inline(always)]
    unsafe fn uvm_n(&mut self, p: Point) -> &mut Tile {
        self.canvas.uget_mut((p.0 - 1, p.1))
    }

    #[inline(always)]
    unsafe fn uvm_e(&mut self, p: Point) -> &mut Tile {
        self.canvas.uget_mut((p.0, p.1 + 1))
    }

    #[inline(always)]
    unsafe fn uvm_s(&mut self, p: Point) -> &mut Tile {
        self.canvas.uget_mut((p.0 + 1, p.1))
    }

    #[inline(always)]
    unsafe fn uvm_w(&mut self, p: Point) -> &mut Tile {
        self.canvas.uget_mut((p.0, p.1 - 1))
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
    fn calc_ntiles(&self) -> NumTiles {
        self
        .canvas
        .fold(0, |x, y| x + (if *y == 0 { 0 } else { 1 }))
    }
}