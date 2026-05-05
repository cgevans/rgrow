use super::base::{GrowError, GrowResult, NumTiles, Point, Tile};
use enum_dispatch::enum_dispatch;
use ndarray::prelude::*;
use serde::{Deserialize, Serialize};

// pub mod tube_diagonal;
pub mod tube_diagonals;
pub mod tube_zz;
// pub use tube_diagonal::CanvasTube;
pub use tube_diagonals::CanvasTubeDiagonals;
pub use tube_zz::CanvasTube;

pub trait CanvasCreate: Sized + Canvas + Clone {
    type Params;
    fn new_sized(shape: Self::Params) -> GrowResult<Self>;
    fn from_array(arr: Array2<Tile>) -> GrowResult<Self>;
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd, Debug, Serialize, Deserialize)]
pub struct PointSafe2(pub Point);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd, Debug)]
pub struct PointSafeHere(pub Point);

/// Shape that storage cells render as.
///
/// `Square`: axis-aligned square. NSEW colors map to the top, right, bottom,
/// left edges. Adjacent cells abut edge-to-edge and tile the plane without gaps.
///
/// `Diamond`: 45°-rotated square. NSEW colors map to the diamond's NW, NE, SE,
/// SW edges (storage-N is the upper-left edge, storage-E the upper-right, etc.).
/// Used by `CanvasTube` (zigzag) where storage neighbors form a tilted-square
/// lattice rather than an axis-aligned one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileShape {
    Square,
    Diamond,
}

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
    fn calc_n_tiles(&self) -> NumTiles;
    fn calc_n_tiles_with_tilearray(&self, should_be_counted: &Array1<bool>) -> NumTiles;
    fn raw_array(&self) -> ArrayView2<'_, Tile>;
    fn raw_array_mut(&mut self) -> ArrayViewMut2<'_, Tile>;
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
    fn nrows_usable(&self) -> usize;
    fn ncols_usable(&self) -> usize;

    fn set_sa_countabletilearray(
        &mut self,
        p: &PointSafe2,
        t: &Tile,
        _should_be_counted: &Array1<bool>,
    ) {
        self.set_sa(p, t);
    }

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
        unsafe { self.uv_p(p.0) }
    }

    fn v_sh(&self, p: PointSafeHere) -> Tile {
        unsafe { self.uv_p(p.0) }
    }

    fn tile_to_n(&self, p: PointSafe2) -> Tile {
        self.v_sh(self.move_sa_n(p))
    }

    fn tile_to_e(&self, p: PointSafe2) -> Tile {
        self.v_sh(self.move_sa_e(p))
    }

    fn tile_to_s(&self, p: PointSafe2) -> Tile {
        self.v_sh(self.move_sa_s(p))
    }

    fn tile_to_w(&self, p: PointSafe2) -> Tile {
        self.v_sh(self.move_sa_w(p))
    }

    fn tile_to_nn(&self, p: PointSafe2) -> Tile {
        self.v_sh(self.move_sa_nn(p))
    }

    fn tile_to_ne(&self, p: PointSafe2) -> Tile {
        self.v_sh(self.move_sa_ne(p))
    }

    fn tile_to_ee(&self, p: PointSafe2) -> Tile {
        self.v_sh(self.move_sa_ee(p))
    }

    fn tile_to_se(&self, p: PointSafe2) -> Tile {
        self.v_sh(self.move_sa_se(p))
    }

    fn tile_to_ss(&self, p: PointSafe2) -> Tile {
        self.v_sh(self.move_sa_ss(p))
    }

    fn tile_to_sw(&self, p: PointSafe2) -> Tile {
        self.v_sh(self.move_sa_sw(p))
    }

    fn tile_to_ww(&self, p: PointSafe2) -> Tile {
        self.v_sh(self.move_sa_ww(p))
    }

    fn tile_to_nw(&self, p: PointSafe2) -> Tile {
        self.v_sh(self.move_sa_nw(p))
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

    /// Frame size in pre-scale "subcell" units.
    ///
    /// `frame_size_px(scale) = frame_size_subcells() * subcell_size_px(scale)`.
    /// For square canvases, one subcell equals one storage cell. For diamond
    /// canvases (zigzag tube), one subcell equals half a tile so that even/odd
    /// row stagger is representable as integer offsets.
    fn frame_size_subcells(&self) -> (u32, u32) {
        (self.ncols() as u32, self.nrows() as u32)
    }

    /// Pixel size of one subcell at the given painter scale.
    ///
    /// Square canvases use the full scale; diamond canvases use half because
    /// each tile occupies a 2×2 block of subcells.
    fn subcell_size_px(&self, scale: u32) -> u32 {
        match self.tile_shape() {
            TileShape::Square => scale,
            // Round up so even an odd `scale` still yields a tile that's
            // `subcell_size_px(scale) * 2 == scale + (scale & 1)` pixels —
            // close enough that the diamond doesn't lose a pixel of width.
            TileShape::Diamond => scale.div_ceil(2),
        }
    }

    /// Frame size in pixels at the given painter scale.
    fn frame_size_px(&self, scale: u32) -> (u32, u32) {
        let (w, h) = self.frame_size_subcells();
        let s = self.subcell_size_px(scale);
        (w * s, h * s)
    }

    /// How tile sprites should be rendered.
    fn tile_shape(&self) -> TileShape {
        TileShape::Square
    }

    /// Pixel side length of a tile sprite (its bounding box).
    ///
    /// For diamond canvases this is the diamond's bounding box (the diamond
    /// itself is inscribed in a square of this size with transparent corners).
    fn tile_size_px(&self, scale: u32) -> u32 {
        match self.tile_shape() {
            TileShape::Square => scale,
            TileShape::Diamond => self.subcell_size_px(scale) * 2,
        }
    }

    /// Top-left pixel of the tile-sprite bounding box for storage cell `p`.
    ///
    /// Default places `(row, col)` at `(col*scale, row*scale)`. Tube canvases
    /// override to shear (diagonal) or stagger (zigzag).
    fn tile_origin_px(&self, p: PointSafe2, scale: u32) -> (u32, u32) {
        let (row, col) = p.0;
        let s = self.subcell_size_px(scale);
        (col as u32 * s, row as u32 * s)
    }

    /// Inverse of `tile_origin_px`: pixel `(px, py)` → storage cell, or `None`
    /// if the pixel falls outside any tile (e.g. the empty triangle corners
    /// of a sheared tube canvas, or beyond the frame).
    fn pixel_to_storage(&self, px: u32, py: u32, scale: u32) -> Option<PointSafe2> {
        let s = self.subcell_size_px(scale);
        if s == 0 {
            return None;
        }
        let col = (px / s) as usize;
        let row = (py / s) as usize;
        if row >= self.nrows() || col >= self.ncols() {
            return None;
        }
        Some(PointSafe2((row, col)))
    }

    /// Storage extent in unit cells. Retained for callers that pre-compute
    /// shared-memory or buffer sizes from `state.draw_size() * scale` — but
    /// new code should use `frame_size_px(scale)` directly.
    fn draw_size(&self) -> (u32, u32) {
        self.frame_size_subcells()
    }

    fn center(&self) -> PointSafe2 {
        PointSafe2((self.nrows() / 2, self.ncols() / 2))
    }

    fn array_size_needed(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CanvasSquare(Array2<Tile>);

impl CanvasCreate for CanvasSquare {
    type Params = (usize, usize);

    fn new_sized(shape: Self::Params) -> GrowResult<Self> {
        if shape.0 < 5 || shape.1 < 5 {
            return Err(GrowError::WrongCanvasSize(shape.0, shape.1));
        }
        Ok(Self(Array2::zeros(shape)))
    }

    fn from_array(arr: Array2<Tile>) -> GrowResult<Self> {
        Ok(Self(arr))
    }
}

impl Canvas for CanvasSquare {
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
        (p.0 >= 2) & (p.1 >= 2) & (p.0 < self.nrows() - 2) & (p.1 < self.ncols() - 2)
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

    fn calc_n_tiles_with_tilearray(&self, should_be_counted: &Array1<bool>) -> NumTiles {
        self.0
            .fold(0, |x, y| x + u32::from(should_be_counted[*y as usize]))
    }

    fn nrows_usable(&self) -> usize {
        self.0.nrows() - 4
    }

    fn ncols_usable(&self) -> usize {
        self.0.ncols() - 4
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanvasPeriodic(Array2<Tile>);

impl CanvasCreate for CanvasPeriodic {
    type Params = (usize, usize);

    fn new_sized(shape: Self::Params) -> GrowResult<Self> {
        if shape.0 == 0 || shape.1 == 0 {
            return Err(GrowError::WrongCanvasSize(shape.0, shape.1));
        }
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

    fn nrows_usable(&self) -> usize {
        self.0.nrows()
    }

    fn ncols_usable(&self) -> usize {
        self.0.ncols()
    }
}

/// CanvasSquareCompact tries to have the advantages of the two-tile-zero border of CanvasSquare, without the associated
/// nuisances, like always having to worry about padding and avoiding the border.   Rather than using a two-tile border around
/// the entire array, visible to the user and part of the coordinate system, it uses a two-tile border on the high-index sides of
/// the array, and a periodic boundary on the low-index sides, wrapping into the high-index two-tile border.  This allows code
/// to treat the canvas the same way as CanvasSquare (eg, tiles are always at PointSafe2 coordinates, and PointSafe2 coordinates always
/// allow two moves away without moving out of *memory* bounds).)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CanvasSquareCompact(Array2<Tile>);

impl CanvasCreate for CanvasSquareCompact {
    type Params = (usize, usize);

    fn new_sized(shape: Self::Params) -> GrowResult<Self> {
        if shape.0 == 0 || shape.1 == 0 {
            return Err(GrowError::WrongCanvasSize(shape.0, shape.1));
        }
        Ok(Self(Array2::zeros((shape.0 + 2, shape.1 + 2))))
    }

    fn from_array(arr: Array2<Tile>) -> GrowResult<Self> {
        // TODO: maybe we want to pad this
        Ok(Self(arr))
    }
}

impl Canvas for CanvasSquareCompact {
    #[inline(always)]
    unsafe fn uv_pr(&self, p: Point) -> &Tile {
        self.0.uget((p.0, p.1))
    }

    #[inline(always)]
    unsafe fn uvm_p(&mut self, p: Point) -> &mut Tile {
        self.0.uget_mut((p.0, p.1))
    }

    #[inline(always)]
    fn inbounds(&self, p: Point) -> bool {
        (p.0 < self.nrows_usable()) & (p.1 < self.ncols_usable())
    }

    fn u_move_point_n(&self, p: Point) -> Point {
        if p.0 == 0 {
            (self.0.nrows() - 1, p.1)
        } else {
            (p.0 - 1, p.1)
        }
    }

    fn u_move_point_e(&self, p: Point) -> Point {
        (p.0, p.1 + 1)
    }

    fn u_move_point_s(&self, p: Point) -> Point {
        (p.0 + 1, p.1)
    }

    fn u_move_point_w(&self, p: Point) -> Point {
        if p.1 == 0 {
            (p.0, self.0.ncols() - 1)
        } else {
            (p.0, p.1 - 1)
        }
    }

    #[inline(always)]
    fn calc_n_tiles(&self) -> NumTiles {
        self.0.fold(0, |x, y| x + u32::from(*y != 0))
    }

    fn raw_array(&self) -> ArrayView2<'_, Tile> {
        self.0
            .slice(s![0..self.0.nrows() - 2, 0..self.0.ncols() - 2])
    }

    fn raw_array_mut(&mut self) -> ArrayViewMut2<'_, Tile> {
        let nrows = self.0.nrows();
        let ncols = self.0.ncols();
        self.0.slice_mut(s![0..nrows - 2, 0..ncols - 2])
    }

    fn nrows(&self) -> usize {
        self.0.nrows() - 2
    }

    fn ncols(&self) -> usize {
        self.0.ncols() - 2
    }

    fn calc_n_tiles_with_tilearray(&self, should_be_counted: &Array1<bool>) -> NumTiles {
        self.0
            .fold(0, |x, y| x + u32::from(should_be_counted[*y as usize]))
    }

    fn nrows_usable(&self) -> usize {
        self.0.nrows() - 2
    }

    fn ncols_usable(&self) -> usize {
        self.0.ncols() - 2
    }

    fn array_size_needed(&self) -> (usize, usize) {
        (self.0.nrows(), self.0.ncols())
    }
}
