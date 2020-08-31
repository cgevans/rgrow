extern crate ndarray;

use ndarray::prelude::*;

use std::error::Error;

type NumTiles = u32;
type NumEvents = u64;

type Point = (usize, usize);

type Tile = usize;

trait CanvasGet {
    // Return tile at a point
    fn v(self, point: Point) -> Tile;

    // Return tile to the North of a point
    fn v_n(self, point: Point) -> Tile;

    // Return tile to the East of a point
    fn v_e(self, point: Point) -> Tile;

    // Return tile to the South of a point
    fn v_s(self, point: Point) -> Tile;

    // Return tile to the West of a point
    fn v_w(self, point: Point) -> Tile;
}

trait CanvasCreate {
    // Creates a canvas
    fn create_canvas(size: Point) -> Self;
}

impl CanvasCreate for Array2<usize> {
    fn create_canvas(size: Point) -> Self {
        Array2::<usize>::zeros((size.0 + 2, size.1 + 2))
    }
}

fn main() {
}
